Statistical Distribution Explorer

Paul A. Bristow
John Maddock

Copyright (C) 2008, 2009, 2010, 2012 Paul A. Bristow, John Maddock

Distributed under the Boost Software License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

A Windows utility to show the properties of statistical distributions using parameters provided interactively by the user.

The 31 distributions provided (by version 1.0.1.52) are:

    * bernoulli
    * beta
    * binomial
    * cauchy
    * chi_squared
    * exponential
    * extreme_value
    * fisher_f
    * gamma
    * geometric
    * hypergeometric
    * inverse_chi_squared
    * inverse_gamma
    * inverse_gaussian
    * laplace
    * logistic
    * lognormal
    * negative_binomial
    * non-central beta
    * non-central_chi_squared
    * non-central_F
    * non-central_t
    * normal (Gaussian)
    * pareto
    * poisson
    * rayleigh
    * students_t
    * skew_normal
    * triangular
    * uniform
    * weibull

Properties of distributions computed (where possible) are:

    * mean
    * mode
    * median
    * variance
    * standard deviation
    * coefficient of variation,
    * skewness
    * kurtosis
    * excess
    * range supported

Calculated, from values provided, are:

    * probability density (or mass) function (PDF)
    * cumulative distribution function (CDF), and complement
    * Quantiles (percentiles) are calculated for typical risk (alpha) probabilities (0.001, 0.01, 0.5, 0.1, 0.333) and for additional probabilities that can be requested by the user.

Results can be saved to text files using Save or SaveAs. All the values on the four tabs are output to the file chosen, and are tab separated to assist input to other programs, for example, spreadsheets or text editors.

Note: Excel (for example), by default, only shows 10 decimal digits: to display the maximum possible precision (about 15 decimal digits), it is necessary to format all cells to display this precision. Although unusually accurate, not all values computed by Statistical Distribution Explorer will be as accurate as 15 decimal digits. Values shown as NaN cannot be calculated from the value(s) given, most commonly because the value input is outside the range for the distribution.

For more information, including downloads, and this index.html file, see Distexplorer at Sourceforge.

This Microsoft Windows 32 package was generated from a C# program and uses a boost_math.dll generated using the Boost.Math C++ source code containing the underlying statistical distribution classes and functions (C++ was compiled in CLI mode).

All source code is freely available for view and for use under the Boost Open Source License.

Math Toolkit C++ source code to produce boost_math.dll is in the most recent Boost release, currently 1.52.0 (and the Boost version used to build the explorer is embedded in its version number). You can download the current Boost library and find the source to build the Distribution Explorer at /libs/math/dot_net_example/.

Installation
============
Statistical Distribution Explorer is distributed as a single Windows Installer package setupdistex.msi. Unzip the distexplorer.zip file to a temporary location of your choice and run setupdistex.msi.

(If the program installs OK, but does not run, it will be necessary to run setup.exe to install Microsoft redistributables. This is because .NET Framework 4.0 Client Profile and VC Redistributable X86 are requirements for this program. Most recent and updated Windows environments will already have these installed, but they are quickly, easily and safely installed from the Microsoft site if required. This program has been tested on Windows 200 up to Windows 8).

(The package cannot be run on other platforms at present but it should be possible to build an equivalent utility on any C/C++ platform if anyone would like to undertake this task.)

Last revised: 15 Oct 2012
	
