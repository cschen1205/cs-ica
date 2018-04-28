# cs-ica

Independent Component Analysis algorithm for solving linear noise less "cocktail party problem"

# Install 

```bash
Install-Package cs-ica
```

# Usage

The library only requires one line of code to split a single source into two separate sources:

```cs
List<double[]> single_source = GetData();
List<double[]> source1;
List<double[]> source2;
ICA.LinearNoiseLessICA.solve(single_source, out source1, out source2);
```
