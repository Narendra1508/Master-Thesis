# Master Thesis Project

This repository is a Tensorflow / Keras implementation of Modeling Long- and Short-Term Temporal Patterns in Frequency Restoration Reserve with Deep Neural Networks

# Dataset

As described in the paper the data is composed of 2 publicly available datasets "Actual Generation" and "Automatic Frequency Restoration Reserve" downloadable from https://www.smard.de/home/downloadcenter/download-marktdaten and https://www.regelleistung.net/apps/datacenter/tenders/:

1 dataset provided by Prof.Tobia Vieth, which was obtained from the "Phasor Measurement Units" (PMUs) of the wide area monitoring system.[https://gridradar.net/de/wide-area-monitoring-system] 

- Actual Generation: A collection of 12 months (2020/01-2020/12) quarter hourly data of actual electricty generation in Ggermany from all type of sources.

- Automatic Frequency Restoration Reserve: The negative and positive afrr data rom 2020/01 to 2020/12, sampled every 15 minutes.

- Phase angle: Phasor data measured at various locations recorded every 15 minutes from 2020/01 to 2020/12.

# Environment

Primary environment: Google Colab

The results were obtained on a system with the following versions:

Python 3.7.10

TensorFlow 2.4.1

Kera 2.4.0

