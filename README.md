## Glazy Data Analysis

(Work in progress)
This project describes my attempts to use machine learning to understand the effect of chemical composition on the firing temperature of ceramic glazes. The dataset of glazes I'm using is taken from Glazy[\https:www.glazy.org], an open-source database of glaze recipes. 

If you're impatient, here's a simplified version of the main result: I tried to predict the firing temperature of glazes based on their chemical compositions, and the predictions aren't particularly accurate. The chart below shows the given firing temperatures vs the predicted ones. 

[INSERT IMAGE]

This is a bit misleading, since some points overlap, so here's a histogram showing the distribution of errors (predicted - actual).

![Histogram of errors](Images/Prediction_error_histogram.png)

In an ideal world, the firing temperature would be a function of the chemical composition of a glaze, and with enough data we'd be able to approximate this function reasonably well. Unfortunately, things aren't so simple, for several reasons. The first is that the maturity of a glaze also depends not only on the maximum temperature, but also on the rate at which the temperature increases. For this reason, [pyrometric cones](https://en.wikipedia.org/wiki/Pyrometric_cone) are used instead of temperatures. However, for a fixed rate of temperature rise, a pyrometric cone will bend at a pre-determined temperature. The glazes in Glazy are described in terms of Orton cones, and I've used the [chart](https://www.ortonceramic.com/files/2676/File/Orton-Cone-Chart-C-022-14-2016) published by Orton, with a 60C/hr rate of temperature rise to map from Orton cones (regular, self-supporting) to temperatures. We'll continue to refer to the firing temperature of a glaze, with the understanding that this implicitly refers to its Orton cone under the inverse of this mapping.

A complication that we can't overcome is that the oxide composition doesn't tell the whole story; the materials that make up a recipe matter. Whether they are crystalline or glassy, have large or small particle sizes, can have an effect on how well melted a glaze is at a given temperature. So we'll have to accept some variability based on the materials used. 

Another source of variability is that the atmosphere of the kiln can have an effect on the firing temperature, particularly for glazes high in iron. While there is the option to indicate the firing atmosphere of glazes in Glazy, these are not consistently filled in, so I've decided not to try control this source of variability, at least for now. 

A further problem is that the temperature at which a glaze is considered mature can be quite subjective. A given glaze might be deemed satisfactory over a fairly large range of temperatures, depending on what effects the potter is looking for. To reduce the amount of variability, I've excluded glazes that are obviously not completely melted. Even so, some glazes are well-melted and stable over a relatively wide temperature range. In Glazy, there are fields to indicate the lower and upper cones to which a glaze can be fired to, so in principle one could try predict these bounds. However, since many entries simply list a single temperature (whatever the potter using the glaze fires to), I've decided to simply predict the midpoint of this range. Of course, the temperature listed might not be the midpoint. 

If we could determine, for each oxide composition, the range of temperatures at which a glaze is fully melted, smooth, and hasn't run off the pot, we could take the midpoint of this range, and use this to define an average firing temperature as a function of the oxide composition (ignoring the effects of material properties and atmosphere for the moment). We'd expect this function to vary relatively smoothly, for the most part, although there will be points where eutectic troughs give rise to sharp local minima. We'd expect the average firing temperatures we obtain from the Glazy data, by contrast, to deviate from this function in a relatively unpredictable manner, since this depends on the personal aesthetics of the potters who've contributed to the database.

...

Problems with the dataset:

Some errors (eg cone 08 vs cone 8), some glazes that are underfired, materials whose analyses differ significantly from their theoretical analyses (Colemanite), glazes with cone info missing.

Non-uniform distribution of firing temperatures.

![Histogram of firing temperatures](Images/Firing_temperature_histogram.png)

A big issue with this dataset is that there are many duplicates and slight variants. If these are not dealt with, the test set will overlap with the training set, and this will artificially decrease the test error. While the duplicates are easy to identify, the slight variants pose a substantial challenge.
 
I've only posted the result of a clustering algorithm (K-means), followed by manually verifying that the glazes in each cluster do have a common origin, and that separate clusters aren't related, and splitting or combining them as necessary. The manual verification process is incomplete, however. I've posted the verified clusters here: 

https://pietermostert.github.io/glazy-data-analysis/html/verified-clusters.html,

and the unverified ones here:

https://pietermostert.github.io/glazy-data-analysis/html/unverified-clusters.html
