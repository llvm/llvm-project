Boost Math Library
============================

>ANNOUNCEMENT: This library requires a compliant C++14 compiler.

|                  |  Master  |   Develop   |
|------------------|----------|-------------|
| Drone            | [![Build Status](https://drone.cpp.al/api/badges/boostorg/math/status.svg?ref=refs/heads/master)](https://drone.cpp.al/boostorg/math)   | [![Build Status](https://drone.cpp.al/api/badges/boostorg/math/status.svg)](https://drone.cpp.al/boostorg/math) |
| Github Actions   | [![Build Status](https://github.com/boostorg/math/workflows/CI/badge.svg?branch=master)](https://github.com/boostorg/math/actions)      | [![Build Status](https://github.com/boostorg/math/workflows/CI/badge.svg?branch=develop)](https://github.com/boostorg/math/actions) |
| Codecov          | [![codecov](https://codecov.io/gh/boostorg/math/branch/master/graph/badge.svg)](https://codecov.io/gh/boostorg/math/branch/master)      | [![codecov](https://codecov.io/gh/boostorg/math/branch/develop/graph/badge.svg)](https://codecov.io/gh/boostorg/math/branch/develop) |


The Math library provides numerous advanced mathematical functions
implemented in modern C++. The library strives to deliver the utmost
in numerical and syntactical correctness while still
maintaining high-performance.

All code is header-only, facilitating easy client setup
and use throughout the entire diverse collection of functions.

The library is divided into several interconnected parts:

### Floating Point Utilities

Utility functions for dealing with floating point arithmetic, includes functions for floating point classification (fpclassify, isnan, isinf etc), sign manipulation, rounding, comparison, and computing the distance between floating point numbers.

### Specific Width Floating Point Types

A set of typedefs similar to those provided by `<cstdint>` but for floating point types.

### Mathematical Constants

A wide range of constants ranging from various multiples of π, fractions, Euler's constant, etc.

These are of course usable from template code, or as non-templates with a simplified interface if that is more appropriate.

### Statistical Distributions

Provides a reasonably comprehensive set of statistical distributions, upon which higher level statistical tests can be built.

The initial focus is on the central univariate distributions. Both continuous (like normal & Fisher) and discrete (like binomial & Poisson) distributions are provided.

A comprehensive tutorial is provided, along with a series of worked examples illustrating how the library is used to conduct statistical tests.

### Special Functions

Provides a wide range of high quality special functions; initially these were concentrated
on functions used in statistical applications along with those in the Technical Report
on C++ Library Extensions.

The function families currently implemented are the gamma, beta and error functions
along with the incomplete gamma and beta functions (four variants of each)
and all the possible inverses of these, plus the digamma, various factorial
functions, Bessel functions, elliptic integrals, hypergeometrics, sinus cardinals
(along with their hyperbolic variants), inverse hyperbolic functions,
Legrendre/Laguerre/Hermite/Chebyshev polynomials
and various special power and logarithmic functions.

All the implementations are fully generic and support the use of arbitrary "real-number" types,
including those in [Boost.Multiprecision](https://github.com/boostorg/multiprecision).
Most functions are, however, optimized for use with types with known significand (or mantissa) sizes:
typically built-in `float`, `double` or `long double`.

These functions also provide the basis of support for the TR1 special functions,
many of which became standardized in [C++17](https://en.cppreference.com/w/cpp/numeric/special_functions).

### Root Finding

A comprehensive set of root-finding algorithms over the real line, both with derivatives and derivative free.

### Optimization

Minimization of cost functions via Brent's method and differential evolution.

### Polynomials and Rational Functions

Tools for manipulating polynomials and for efficient evaluation of rationals or polynomials.

### Interpolation

Function interpolation via barycentric rational interpolation,
compactly supported quadratic, cubic, and quintic B-splines,
the Chebyshev transform, trigonometric polynomials, Makima,
pchip, cubic Hermite splines, and bilinear interpolation.

### Numerical Integration and Differentiation

A reasonably comprehensive set of routines for integration
(trapezoidal, Gauss-Legendre, Gauss-Kronrod, Gauss-Chebyshev, double-exponential, and Monte-Carlo)
and differentiation (Chebyshev transform, finite difference, the complex step derivative,
and forward-mode automatic differentiation).

The integration routines are usable for functions returning complex results - and hence can be used for computation of  contour integrals.

### Quaternions and Octonions

Quaternion and Octonion are class templates similar to std::complex.

The full documentation is available on [boost.org](http://www.boost.org/doc/libs/release/libs/math).

### Standalone Mode

Defining BOOST_MATH_STANDALONE allows Boost.Math to be used without any Boost dependencies.
Some functionality is reduced in this mode. A static_assert message will alert you
if a particular feature has been disabled by standalone mode. Standalone mode is not designed to 
be used with the rest of boost, and may result in compiler errors.

## Supported Compilers ##

The following compilers are tested with the CI system, and are known to work.
Currently a compiler that is fully compliant with C++14 is required to use Boost.Math.

* g++ 5 or later
* clang++ 5 or later
* Visual Studio 2015 (14.0) or later

## Support, bugs and feature requests ##

Bugs and feature requests can be reported through the [GitHub issue tracker](https://github.com/boostorg/math/issues)
(see [open issues](https://github.com/boostorg/math/issues) and
[closed issues](https://github.com/boostorg/math/issues?utf8=%E2%9C%93&q=is%3Aissue+is%3Aclosed)).

You can submit your changes through a [pull request](https://github.com/boostorg/math/pulls).

There is no mailing-list specific to Boost Math, although you can use the general-purpose Boost [mailing-list](http://lists.boost.org/mailman/listinfo.cgi/boost-users) using the tag [math].


## Development ##

Clone the whole boost project, which includes the individual Boost projects as submodules ([see boost+git doc](https://github.com/boostorg/boost/wiki/Getting-Started)):

    $ git clone https://github.com/boostorg/boost
    $ cd boost
    $ git submodule update --init

The Boost Math Library is located in `libs/math/`.

### Running tests ###
First, make sure you are in `libs/math/test`.
You can either run all the tests listed in `Jamfile.v2` or run a single test:

    test$ ../../../b2                        <- run all tests
    test$ ../../../b2 static_assert_test     <- single test
    test$ # A more advanced syntax, demoing various options for building the tests:
    test$ ../../../b2 -a -j2 -q --reconfigure toolset=clang cxxflags="--std=c++14 -fsanitize=address -fsanitize=undefined" linkflags="-fsanitize=undefined -fsanitize=address"

### Continuous Integration ###
The default action for a PR or commit to a PR is for CI to run the full complement of tests. The following can be appended to the end of a commit message to modify behavior:

    * [ci skip] to skip all tests
    * [linux] to test using GCC Versions 5-12 and Clang Versions 5-14 on Ubuntu LTS versions 18.04-22.04.
    * [apple] to test Apple Clang on the latest version of MacOS.
    * [windows] to test MSVC-14.0, MSVC-14.2, MSVC-14.3, CYGWIN, and mingw on the latest version of Windows.
    * [standalone] to run standalone mode compile tests
     
### Building documentation ###

Full instructions can be found [here](https://svn.boost.org/trac10/wiki/BoostDocs/GettingStarted), but to reiterate slightly:

```bash
libs/math/doc$ brew install docbook-xsl # on mac
libs/math/doc$ touch ~/user-config.jam
libs/math/doc$ # now edit so that:
libs/math/doc$ cat ~/user-config.jam
using darwin ;

using xsltproc ;

using boostbook
    : /usr/local/opt/docbook-xsl/docbook-xsl
    ;

using doxygen ;
using quickbook ;
libs/math/doc$ ../../../b2
```
