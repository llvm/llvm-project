// normal_misc_examples.cpp

// Copyright Paul A. Bristow 2007, 2010.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Example of using normal distribution.

// Note that this file contains Quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

//[normal_basic1
/*`
First we need some includes to access the normal distribution
(and some std output of course).
*/

#include <boost/math/distributions/normal.hpp> // for normal_distribution
  using boost::math::normal; // typedef provides default type is double.

#include <iostream>
  using std::cout; using std::endl; using std::left; using std::showpoint; using std::noshowpoint;
#include <iomanip>
  using std::setw; using std::setprecision;
#include <limits>
  using std::numeric_limits;

int main()
{
  cout << "Example: Normal distribution, Miscellaneous Applications.";

  try
  {
    { // Traditional tables and values.
/*`Let's start by printing some traditional tables.
*/
      double step = 1.; // in z
      double range = 4; // min and max z = -range to +range.
      int precision = 17; // traditional tables are only computed to much lower precision.
      // but std::numeric_limits<double>::max_digits10; on new Standard Libraries gives
      // 17, the maximum number of digits that can possibly be significant.
      // std::numeric_limits<double>::digits10; == 15 is number of guaranteed digits,
      // the other two digits being 'noisy'.

      // Construct a standard normal distribution s
        normal s; // (default mean = zero, and standard deviation = unity)
        cout << "Standard normal distribution, mean = "<< s.mean()
          << ", standard deviation = " << s.standard_deviation() << endl;

/*` First the probability distribution function (pdf).
*/
      cout << "Probability distribution function values" << endl;
      cout << "  z " "      pdf " << endl;
      cout.precision(5);
      for (double z = -range; z < range + step; z += step)
      {
        cout << left << setprecision(3) << setw(6) << z << " "
          << setprecision(precision) << setw(12) << pdf(s, z) << endl;
      }
      cout.precision(6); // default
      /*`And the area under the normal curve from -[infin] up to z,
      the cumulative distribution function (cdf).
*/
      // For a standard normal distribution
      cout << "Standard normal mean = "<< s.mean()
        << ", standard deviation = " << s.standard_deviation() << endl;
      cout << "Integral (area under the curve) from - infinity up to z " << endl;
      cout << "  z " "      cdf " << endl;
      for (double z = -range; z < range + step; z += step)
      {
        cout << left << setprecision(3) << setw(6) << z << " "
          << setprecision(precision) << setw(12) << cdf(s, z) << endl;
      }
      cout.precision(6); // default

/*`And all this you can do with a nanoscopic amount of work compared to
the team of *human computers* toiling with Milton Abramovitz and Irene Stegen
at the US National Bureau of Standards (now [@http://www.nist.gov NIST]).
Starting in 1938, their "Handbook of Mathematical Functions with Formulas, Graphs and Mathematical Tables",
was eventually published in 1964, and has been reprinted numerous times since.
(A major replacement is planned at [@http://dlmf.nist.gov Digital Library of Mathematical Functions]).

Pretty-printing a traditional 2-dimensional table is left as an exercise for the student,
but why bother now that the Math Toolkit lets you write
*/
    double z = 2.;
    cout << "Area for z = " << z << " is " << cdf(s, z) << endl; // to get the area for z.
/*`
Correspondingly, we can obtain the traditional 'critical' values for significance levels.
For the 95% confidence level, the significance level usually called alpha,
is 0.05 = 1 - 0.95 (for a one-sided test), so we can write
*/
     cout << "95% of area has a z below " << quantile(s, 0.95) << endl;
   // 95% of area has a z below 1.64485
/*`and a two-sided test (a comparison between two levels, rather than a one-sided test)

*/
     cout << "95% of area has a z between " << quantile(s, 0.975)
       << " and " << -quantile(s, 0.975) << endl;
   // 95% of area has a z between 1.95996 and -1.95996
/*`

First, define a table of significance levels: these are the probabilities
that the true occurrence frequency lies outside the calculated interval.

It is convenient to have an alpha level for the probability that z lies outside just one standard deviation.
This will not be some nice neat number like 0.05, but we can easily calculate it,
*/
    double alpha1 = cdf(s, -1) * 2; // 0.3173105078629142
    cout << setprecision(17) << "Significance level for z == 1 is " << alpha1 << endl;
/*`
    and place in our array of favorite alpha values.
*/
    double alpha[] = {0.3173105078629142, // z for 1 standard deviation.
      0.20, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001 };
/*`

Confidence value as % is (1 - alpha) * 100 (so alpha 0.05 == 95% confidence)
that the true occurrence frequency lies *inside* the calculated interval.

*/
    cout << "level of significance (alpha)" << setprecision(4) << endl;
    cout << "2-sided       1 -sided          z(alpha) " << endl;
    for (unsigned i = 0; i < sizeof(alpha)/sizeof(alpha[0]); ++i)
    {
      cout << setw(15) << alpha[i] << setw(15) << alpha[i] /2 << setw(10) << quantile(complement(s,  alpha[i]/2)) << endl;
      // Use quantile(complement(s, alpha[i]/2)) to avoid potential loss of accuracy from quantile(s,  1 - alpha[i]/2)
    }
    cout << endl;

/*`Notice the distinction between one-sided (also called one-tailed)
where we are using a > *or* < test (and not both)
and considering the area of the tail (integral) from z up to +[infin],
and a two-sided test where we are using two > *and* < tests, and thus considering two tails,
from -[infin] up to z low and z high up to +[infin].

So the 2-sided values alpha[i] are calculated using alpha[i]/2.

If we consider a simple example of alpha = 0.05, then for a two-sided test,
the lower tail area from -[infin] up to -1.96 is 0.025 (alpha/2)
and the upper tail area from +z up to +1.96 is also  0.025 (alpha/2),
and the area between -1.96 up to 12.96 is alpha = 0.95.
and the sum of the two tails is 0.025 + 0.025 = 0.05,

*/
//] [/[normal_basic1]

//[normal_basic2

/*`Armed with the cumulative distribution function, we can easily calculate the
easy to remember proportion of values that lie within 1, 2 and 3 standard deviations from the mean.

*/
    cout.precision(3);
    cout << showpoint << "cdf(s, s.standard_deviation()) = "
      << cdf(s, s.standard_deviation()) << endl;  // from -infinity to 1 sd
    cout << "cdf(complement(s, s.standard_deviation())) = "
      << cdf(complement(s, s.standard_deviation())) << endl;
    cout << "Fraction 1 standard deviation within either side of mean is "
      << 1 -  cdf(complement(s, s.standard_deviation())) * 2 << endl;
    cout << "Fraction 2 standard deviations within either side of mean is "
      << 1 -  cdf(complement(s, 2 * s.standard_deviation())) * 2 << endl;
    cout << "Fraction 3 standard deviations within either side of mean is "
      << 1 -  cdf(complement(s, 3 * s.standard_deviation())) * 2 << endl;

/*`
To a useful precision, the 1, 2 & 3 percentages are 68, 95 and 99.7,
and these are worth memorising as useful 'rules of thumb', as, for example, in
[@http://en.wikipedia.org/wiki/Standard_deviation standard deviation]:

[pre
Fraction 1 standard deviation within either side of mean is 0.683
Fraction 2 standard deviations within either side of mean is 0.954
Fraction 3 standard deviations within either side of mean is 0.997
]

We could of course get some really accurate values for these
[@http://en.wikipedia.org/wiki/Confidence_interval confidence intervals]
by using cout.precision(15);

[pre
Fraction 1 standard deviation within either side of mean is 0.682689492137086
Fraction 2 standard deviations within either side of mean is 0.954499736103642
Fraction 3 standard deviations within either side of mean is 0.997300203936740
]

But before you get too excited about this impressive precision,
don't forget that the *confidence intervals of the standard deviation* are surprisingly wide,
especially if you have estimated the standard deviation from only a few measurements.
*/
//] [/[normal_basic2]


//[normal_bulbs_example1
/*`
Examples from K. Krishnamoorthy, Handbook of Statistical Distributions with Applications,
ISBN 1 58488 635 8, page 125... implemented using the Math Toolkit library.

A few very simple examples are shown here:
*/
// K. Krishnamoorthy, Handbook of Statistical Distributions with Applications,
 // ISBN 1 58488 635 8, page 125, example 10.3.5
/*`Mean lifespan of 100 W bulbs is 1100 h with standard deviation of 100 h.
Assuming, perhaps with little evidence and much faith, that the distribution is normal,
we construct a normal distribution called /bulbs/ with these values:
*/
    double mean_life = 1100.;
    double life_standard_deviation = 100.;
    normal bulbs(mean_life, life_standard_deviation);
    double expected_life = 1000.;

/*`The we can use the Cumulative distribution function to predict fractions
(or percentages, if * 100) that will last various lifetimes.
*/
    cout << "Fraction of bulbs that will last at best (<=) " // P(X <= 1000)
      << expected_life << " is "<< cdf(bulbs, expected_life) << endl;
    cout << "Fraction of bulbs that will last at least (>) " // P(X > 1000)
      << expected_life << " is "<< cdf(complement(bulbs, expected_life)) << endl;
    double min_life = 900;
    double max_life = 1200;
    cout << "Fraction of bulbs that will last between "
      << min_life << " and " << max_life << " is "
      << cdf(bulbs, max_life)  // P(X <= 1200)
       - cdf(bulbs, min_life) << endl; // P(X <= 900)
/*`
[note Real-life failures are often very ab-normal,
with a significant number that 'dead-on-arrival' or suffer failure very early in their life:
the lifetime of the survivors of 'early mortality' may be well described by the normal distribution.]
*/
//] [/normal_bulbs_example1 Quickbook end]
  }
  {
    // K. Krishnamoorthy, Handbook of Statistical Distributions with Applications,
    // ISBN 1 58488 635 8, page 125, Example 10.3.6

//[normal_bulbs_example3
/*`Weekly demand for 5 lb sacks of onions at a store is normally distributed with mean 140 sacks and standard deviation 10.
*/
  double mean = 140.; // sacks per week.
  double standard_deviation = 10;
  normal sacks(mean, standard_deviation);

  double stock = 160.; // per week.
  cout << "Percentage of weeks overstocked "
    << cdf(sacks, stock) * 100. << endl; // P(X <=160)
  // Percentage of weeks overstocked 97.7

/*`So there will be lots of mouldy onions!
So we should be able to say what stock level will meet demand 95% of the weeks.
*/
  double stock_95 = quantile(sacks, 0.95);
  cout << "Store should stock " << int(stock_95) << " sacks to meet 95% of demands." << endl;
/*`And it is easy to estimate how to meet 80% of demand, and waste even less.
*/
  double stock_80 = quantile(sacks, 0.80);
  cout << "Store should stock " << int(stock_80) << " sacks to meet 8 out of 10 demands." << endl;
//] [/normal_bulbs_example3 Quickbook end]
  }
  { // K. Krishnamoorthy, Handbook of Statistical Distributions with Applications,
    // ISBN 1 58488 635 8, page 125, Example 10.3.7

//[normal_bulbs_example4

/*`A machine is set to pack 3 kg of ground beef per pack.
Over a long period of time it is found that the average packed was 3 kg
with a standard deviation of 0.1 kg.
Assuming the packing is normally distributed,
we can find the fraction (or %) of packages that weigh more than 3.1 kg.
*/

double mean = 3.; // kg
double standard_deviation = 0.1; // kg
normal packs(mean, standard_deviation);

double max_weight = 3.1; // kg
cout << "Percentage of packs > " << max_weight << " is "
<< cdf(complement(packs, max_weight)) << endl; // P(X > 3.1)

double under_weight = 2.9;
cout <<"fraction of packs <= " << under_weight << " with a mean of " << mean
  << " is " << cdf(complement(packs, under_weight)) << endl;
// fraction of packs <= 2.9 with a mean of 3 is 0.841345
// This is 0.84 - more than the target 0.95
// Want 95% to be over this weight, so what should we set the mean weight to be?
// KK StatCalc says:
double over_mean = 3.0664;
normal xpacks(over_mean, standard_deviation);
cout << "fraction of packs >= " << under_weight
<< " with a mean of " << xpacks.mean()
  << " is " << cdf(complement(xpacks, under_weight)) << endl;
// fraction of packs >= 2.9 with a mean of 3.06449 is 0.950005
double under_fraction = 0.05;  // so 95% are above the minimum weight mean - sd = 2.9
double low_limit = standard_deviation;
double offset = mean - low_limit - quantile(packs, under_fraction);
double nominal_mean = mean + offset;

normal nominal_packs(nominal_mean, standard_deviation);
cout << "Setting the packer to " << nominal_mean << " will mean that "
  << "fraction of packs >= " << under_weight
  << " is " << cdf(complement(nominal_packs, under_weight)) << endl;

/*`
Setting the packer to 3.06449 will mean that fraction of packs >= 2.9 is 0.95.

Setting the packer to 3.13263 will mean that fraction of packs >= 2.9 is 0.99,
but will more than double the mean loss from 0.0644 to 0.133.

Alternatively, we could invest in a better (more precise) packer with a lower standard deviation.

To estimate how much better (how much smaller standard deviation) it would have to be,
we need to get the 5% quantile to be located at the under_weight limit, 2.9
*/
double p = 0.05; // wanted p th quantile.
cout << "Quantile of " << p << " = " << quantile(packs, p)
  << ", mean = " << packs.mean() << ", sd = " << packs.standard_deviation() << endl; //
/*`
Quantile of 0.05 = 2.83551, mean = 3, sd = 0.1

With the current packer (mean = 3, sd = 0.1), the 5% quantile is at 2.8551 kg,
a little below our target of 2.9 kg.
So we know that the standard deviation is going to have to be smaller.

Let's start by guessing that it (now 0.1) needs to be halved, to a standard deviation of 0.05
*/
normal pack05(mean, 0.05);
cout << "Quantile of " << p << " = " << quantile(pack05, p)
  << ", mean = " << pack05.mean() << ", sd = " << pack05.standard_deviation() << endl;

cout <<"Fraction of packs >= " << under_weight << " with a mean of " << mean
  << " and standard deviation of " << pack05.standard_deviation()
  << " is " << cdf(complement(pack05, under_weight)) << endl;
//
/*`
Fraction of packs >= 2.9 with a mean of 3 and standard deviation of 0.05 is 0.9772

So 0.05 was quite a good guess, but we are a little over the 2.9 target,
so the standard deviation could be a tiny bit more. So we could do some
more guessing to get closer, say by increasing to 0.06
*/

normal pack06(mean, 0.06);
cout << "Quantile of " << p << " = " << quantile(pack06, p)
  << ", mean = " << pack06.mean() << ", sd = " << pack06.standard_deviation() << endl;

cout <<"Fraction of packs >= " << under_weight << " with a mean of " << mean
  << " and standard deviation of " << pack06.standard_deviation()
  << " is " << cdf(complement(pack06, under_weight)) << endl;
/*`
Fraction of packs >= 2.9 with a mean of 3 and standard deviation of 0.06 is 0.9522

Now we are getting really close, but to do the job properly,
we could use root finding method, for example the tools provided, and used elsewhere,
in the Math Toolkit, see __root_finding_without_derivatives.

But in this normal distribution case, we could be even smarter and make a direct calculation.
*/

normal s; // For standard normal distribution,
double sd = 0.1;
double x = 2.9; // Our required limit.
// then probability p = N((x - mean) / sd)
// So if we want to find the standard deviation that would be required to meet this limit,
// so that the p th quantile is located at x,
// in this case the 0.95 (95%) quantile at 2.9 kg pack weight, when the mean is 3 kg.

double prob =  pdf(s, (x - mean) / sd);
double qp = quantile(s, 0.95);
cout << "prob = " << prob << ", quantile(p) " << qp << endl; // p = 0.241971, quantile(p) 1.64485
// Rearranging, we can directly calculate the required standard deviation:
double sd95 = std::abs((x - mean)) / qp;

cout << "If we want the "<< p << " th quantile to be located at "
  << x << ", would need a standard deviation of " << sd95 << endl;

normal pack95(mean, sd95);  // Distribution of the 'ideal better' packer.
cout <<"Fraction of packs >= " << under_weight << " with a mean of " << mean
  << " and standard deviation of " << pack95.standard_deviation()
  << " is " << cdf(complement(pack95, under_weight)) << endl;

// Fraction of packs >= 2.9 with a mean of 3 and standard deviation of 0.0608 is 0.95

/*`Notice that these two deceptively simple questions
(do we over-fill or measure better) are actually very common.
The weight of beef might be replaced by a measurement of more or less anything.
But the calculations rely on the accuracy of the standard deviation - something
that is almost always less good than we might wish,
especially if based on a few measurements.
*/

//] [/normal_bulbs_example4 Quickbook end]
  }

  { // K. Krishnamoorthy, Handbook of Statistical Distributions with Applications,
    // ISBN 1 58488 635 8, page 125, example 10.3.8
//[normal_bulbs_example5
/*`A bolt is usable if between 3.9 and 4.1 long.
From a large batch of bolts, a sample of 50 show a
mean length of 3.95 with standard deviation 0.1.
Assuming a normal distribution, what proportion is usable?
The true sample mean is unknown,
but we can use the sample mean and standard deviation to find approximate solutions.
*/

    normal bolts(3.95, 0.1);
    double top = 4.1;
    double bottom = 3.9;

cout << "Fraction long enough [ P(X <= " << top << ") ] is " << cdf(bolts, top) << endl;
cout << "Fraction too short [ P(X <= " << bottom << ") ] is " << cdf(bolts, bottom) << endl;
cout << "Fraction OK  -between " << bottom << " and " << top
  << "[ P(X <= " << top  << ") - P(X<= " << bottom << " ) ] is "
  << cdf(bolts, top) - cdf(bolts, bottom) << endl;

cout << "Fraction too long [ P(X > " << top << ") ] is "
  << cdf(complement(bolts, top)) << endl;

cout << "95% of bolts are shorter than " << quantile(bolts, 0.95) << endl;

//] [/normal_bulbs_example5 Quickbook end]
  }
  }
  catch(const std::exception& e)
  { // Always useful to include try & catch blocks because default policies
    // are to throw exceptions on arguments that cause errors like underflow, overflow.
    // Lacking try & catch blocks, the program will abort without a message below,
    // which may give some helpful clues as to the cause of the exception.
    std::cout <<
      "\n""Message from thrown exception was:\n   " << e.what() << std::endl;
  }
  return 0;
}  // int main()


/*

Output is:

Autorun "i:\boost-06-05-03-1300\libs\math\test\Math_test\debug\normal_misc_examples.exe"
Example: Normal distribution, Miscellaneous Applications.Standard normal distribution, mean = 0, standard deviation = 1
Probability distribution function values
  z       pdf
-4     0.00013383022576488537
-3     0.0044318484119380075
-2     0.053990966513188063
-1     0.24197072451914337
0      0.3989422804014327
1      0.24197072451914337
2      0.053990966513188063
3      0.0044318484119380075
4      0.00013383022576488537
Standard normal mean = 0, standard deviation = 1
Integral (area under the curve) from - infinity up to z
  z       cdf
-4     3.1671241833119979e-005
-3     0.0013498980316300959
-2     0.022750131948179219
-1     0.1586552539314571
0      0.5
1      0.84134474606854293
2      0.97724986805182079
3      0.9986501019683699
4      0.99996832875816688
Area for z = 2 is 0.97725
95% of area has a z below 1.64485
95% of area has a z between 1.95996 and -1.95996
Significance level for z == 1 is 0.3173105078629142
level of significance (alpha)
2-sided       1 -sided          z(alpha)
0.3173         0.1587         1
0.2            0.1            1.282
0.1            0.05           1.645
0.05           0.025          1.96
0.01           0.005          2.576
0.001          0.0005         3.291
0.0001         5e-005         3.891
1e-005         5e-006         4.417
cdf(s, s.standard_deviation()) = 0.841
cdf(complement(s, s.standard_deviation())) = 0.159
Fraction 1 standard deviation within either side of mean is 0.683
Fraction 2 standard deviations within either side of mean is 0.954
Fraction 3 standard deviations within either side of mean is 0.997
Fraction of bulbs that will last at best (<=) 1.00e+003 is 0.159
Fraction of bulbs that will last at least (>) 1.00e+003 is 0.841
Fraction of bulbs that will last between 900. and 1.20e+003 is 0.819
Percentage of weeks overstocked 97.7
Store should stock 156 sacks to meet 95% of demands.
Store should stock 148 sacks to meet 8 out of 10 demands.
Percentage of packs > 3.10 is 0.159
fraction of packs <= 2.90 with a mean of 3.00 is 0.841
fraction of packs >= 2.90 with a mean of 3.07 is 0.952
Setting the packer to 3.06 will mean that fraction of packs >= 2.90 is 0.950
Quantile of 0.0500 = 2.84, mean = 3.00, sd = 0.100
Quantile of 0.0500 = 2.92, mean = 3.00, sd = 0.0500
Fraction of packs >= 2.90 with a mean of 3.00 and standard deviation of 0.0500 is 0.977
Quantile of 0.0500 = 2.90, mean = 3.00, sd = 0.0600
Fraction of packs >= 2.90 with a mean of 3.00 and standard deviation of 0.0600 is 0.952
prob = 0.242, quantile(p) 1.64
If we want the 0.0500 th quantile to be located at 2.90, would need a standard deviation of 0.0608
Fraction of packs >= 2.90 with a mean of 3.00 and standard deviation of 0.0608 is 0.950
Fraction long enough [ P(X <= 4.10) ] is 0.933
Fraction too short [ P(X <= 3.90) ] is 0.309
Fraction OK  -between 3.90 and 4.10[ P(X <= 4.10) - P(X<= 3.90 ) ] is 0.625
Fraction too long [ P(X > 4.10) ] is 0.0668
95% of bolts are shorter than 4.11

*/
