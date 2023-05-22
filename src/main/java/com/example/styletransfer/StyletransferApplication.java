package com.example.styletransfer;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.translator.StyleTransferTranslatorFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

@SpringBootApplication
public class StyletransferApplication {

	private static final Logger logger = LoggerFactory.getLogger(StyletransferApplication.class);

    public StyletransferApplication() {}

    public enum Artist {
        CEZANNE,
        MONET,
        UKIYOE,
        VANGOGH
    }

	public static void main(String[] args) throws IOException, ModelException, TranslateException {
		SpringApplication.run(StyletransferApplication.class, args);

		Artist artist = Artist.MONET;
		System.out.println("Working Directory = " + System.getProperty("user.dir"));
        String imagePath = "mountains.png";
        Image input = ImageFactory.getInstance().fromFile(Paths.get(imagePath));
        Image output = transfer(input, artist);

        logger.info(
                "Using PyTorch Engine. {} painting generated. Image saved in build/output/cyclegan",
                artist);
        save(output, artist.toString(), "build/output/cyclegan/");
	}

	public static Image transfer(Image image, Artist artist)
            throws IOException, ModelNotFoundException, MalformedModelException,
                    TranslateException {
        String modelName = "style_" + artist.toString().toLowerCase() + ".zip";
        String modelUrl =
                "https://mlrepo.djl.ai/model/cv/image_generation/ai/djl/pytorch/cyclegan/0.0.1/"
                        + modelName;

        Criteria<Image, Image> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.IMAGE_GENERATION)
                        .setTypes(Image.class, Image.class)
                        .optModelUrls(modelUrl)
                        .optProgress(new ProgressBar())
                        .optTranslatorFactory(new StyleTransferTranslatorFactory())
                        .optEngine("PyTorch")
                        .build();

        try (ZooModel<Image, Image> model = criteria.loadModel();
                Predictor<Image, Image> styler = model.newPredictor()) {
            return styler.predict(image);
        }
    }

    public static void save(Image image, String name, String path) throws IOException {
        Path outputPath = Paths.get(path);
        Files.createDirectories(outputPath);
        Path imagePath = outputPath.resolve(name + ".png");
        image.save(Files.newOutputStream(imagePath), "png");
    }

}
