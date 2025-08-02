// Copyright Paul A. Bristow 2007, 2009, 2010
// Copyright John Maddock 2006

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// binomial_examples_quiz.cpp

// Simple example of computing probabilities and quantiles for a binomial random variable
// representing the correct guesses on a multiple-choice test.

// source http://www.stat.wvu.edu/SRS/Modules/Binomial/test.html

//[binomial_quiz_example1
/*`
A multiple choice test has four possible answers to each of 16 questions.
A student guesses the answer to each question,
so the probability of getting a correct answer on any given question is
one in four, a quarter, 1/4, 25% or fraction 0.25.
The conditions of the binomial experiment are assumed to be met:
n = 16 questions constitute the trials;
each question results in one of two possible outcomes (correct or incorrect);
the probability of being correct is 0.25 and is constant if no knowledge about the subject is assumed;
the questions are answered independently if the student's answer to a question
in no way influences his/her answer to another question.

First, we need to be able to use the binomial distribution constructor
(and some std input/output, of course).
*/

#include <boost/math/distributions/binomial.hpp>
  using boost::math::binomial;

#include <iostream>
  using std::cout; using std::endl;
  using std::ios; using std::flush; using std::left; using std::right; using std::fixed;
#include <iomanip>
  using std::setw; using std::setprecision;
#include <exception>
  


//][/binomial_quiz_example1]

int main()
{
  try
  {
  cout << "Binomial distribution example - guessing in a quiz." << endl;
//[binomial_quiz_example2
/*`
The number of correct answers, X, is distributed as a binomial random variable
with binomial distribution parameters: questions n and success fraction probability p.
So we construct a binomial distribution:
*/
  int questions = 16; // All the questions in the quiz.
  int answers = 4; // Possible answers to each question.
  double success_fraction = 1. / answers; // If a random guess, p = 1/4 = 0.25.
  binomial quiz(questions, success_fraction);
/*`
and display the distribution parameters we used thus:
*/
  cout << "In a quiz with " << quiz.trials()
    << " questions and with a probability of guessing right of "
    << quiz.success_fraction() * 100 << " %"
    << " or 1 in " << static_cast<int>(1. / quiz.success_fraction()) << endl;
/*`
Show a few probabilities of just guessing:
*/
  cout << "Probability of getting none right is " << pdf(quiz, 0) << endl; // 0.010023
  cout << "Probability of getting exactly one right is " << pdf(quiz, 1) << endl;
  cout << "Probability of getting exactly two right is " << pdf(quiz, 2) << endl;
  int pass_score = 11;
  cout << "Probability of getting exactly " << pass_score << " answers right by chance is "
    << pdf(quiz, pass_score) << endl;
  cout << "Probability of getting all " << questions << " answers right by chance is "
    << pdf(quiz, questions) << endl;
/*`
[pre
Probability of getting none right is 0.0100226
Probability of getting exactly one right is 0.0534538
Probability of getting exactly two right is 0.133635
Probability of getting exactly 11 right is 0.000247132
Probability of getting exactly all 16 answers right by chance is 2.32831e-010
]
These don't give any encouragement to guessers!

We can tabulate the 'getting exactly right' ( == ) probabilities thus:
*/
  cout << "\n" "Guessed Probability" << right << endl;
  for (int successes = 0; successes <= questions; successes++)
  {
    double probability = pdf(quiz, successes);
    cout << setw(2) << successes << "      " << probability << endl;
  }
  cout << endl;
/*`
[pre
Guessed Probability
 0      0.0100226
 1      0.0534538
 2      0.133635
 3      0.207876
 4      0.225199
 5      0.180159
 6      0.110097
 7      0.0524273
 8      0.0196602
 9      0.00582526
10      0.00135923
11      0.000247132
12      3.43239e-005
13      3.5204e-006
14      2.51457e-007
15      1.11759e-008
16      2.32831e-010
]
Then we can add the probabilities of some 'exactly right' like this:
*/
  cout << "Probability of getting none or one right is " << pdf(quiz, 0) + pdf(quiz, 1) << endl;

/*`
[pre
Probability of getting none or one right is 0.0634764
]
But if more than a couple of scores are involved, it is more convenient (and may be more accurate)
to use the Cumulative Distribution Function (cdf) instead:
*/
  cout << "Probability of getting none or one right is " << cdf(quiz, 1) << endl;
/*`
[pre
Probability of getting none or one right is 0.0634764
]
Since the cdf is inclusive, we can get the probability of getting up to 10 right ( <= )
*/
  cout << "Probability of getting <= 10 right (to fail) is " << cdf(quiz, 10) << endl;
/*`
[pre
Probability of getting <= 10 right (to fail) is 0.999715
]
To get the probability of getting 11 or more right (to pass),
it is tempting to use ``1 - cdf(quiz, 10)`` to get the probability of > 10
*/
  cout << "Probability of getting > 10 right (to pass) is " << 1 - cdf(quiz, 10) << endl;
/*`
[pre
Probability of getting > 10 right (to pass) is 0.000285239
]
But this should be resisted in favor of using the __complements function (see __why_complements).
*/
  cout << "Probability of getting > 10 right (to pass) is " << cdf(complement(quiz, 10)) << endl;
/*`
[pre
Probability of getting > 10 right (to pass) is 0.000285239
]
And we can check that these two, <= 10 and > 10,  add up to unity.
*/
BOOST_MATH_ASSERT((cdf(quiz, 10) + cdf(complement(quiz, 10))) == 1.);
/*`
If we want a < rather than a <= test, because the CDF is inclusive, we must subtract one from the score.
*/
  cout << "Probability of getting less than " << pass_score
    << " (< " << pass_score << ") answers right by guessing is "
    << cdf(quiz, pass_score -1) << endl;
/*`
[pre
Probability of getting less than 11 (< 11) answers right by guessing is 0.999715
]
and similarly to get a >= rather than a > test
we also need to subtract one from the score (and can again check the sum is unity).
This is because if the cdf is /inclusive/,
then its complement must be /exclusive/ otherwise there would be one possible
outcome counted twice!
*/
  cout << "Probability of getting at least " << pass_score
    << "(>= " << pass_score << ") answers right by guessing is "
    << cdf(complement(quiz, pass_score-1))
    << ", only 1 in " << 1/cdf(complement(quiz, pass_score-1)) << endl;

  BOOST_MATH_ASSERT((cdf(quiz, pass_score -1) + cdf(complement(quiz, pass_score-1))) == 1);

/*`
[pre
Probability of getting at least 11 (>= 11) answers right by guessing is 0.000285239, only 1 in 3505.83
]
Finally we can tabulate some probabilities:
*/
  cout << "\n" "At most (<=)""\n""Guessed OK   Probability" << right << endl;
  for (int score = 0; score <= questions; score++)
  {
    cout << setw(2) << score << "           " << setprecision(10)
      << cdf(quiz, score) << endl;
  }
  cout << endl;
/*`
[pre
At most (<=)
Guessed OK   Probability
 0           0.01002259576
 1           0.0634764398
 2           0.1971110499
 3           0.4049871101
 4           0.6301861752
 5           0.8103454274
 6           0.9204427481
 7           0.9728700437
 8           0.9925302796
 9           0.9983555346
10           0.9997147608
11           0.9999618928
12           0.9999962167
13           0.9999997371
14           0.9999999886
15           0.9999999998
16           1
]
*/
  cout << "\n" "At least (>)""\n""Guessed OK   Probability" << right << endl;
  for (int score = 0; score <= questions; score++)
  {
    cout << setw(2) << score << "           "  << setprecision(10)
      << cdf(complement(quiz, score)) << endl;
  }
/*`
[pre
At least (>)
Guessed OK   Probability
 0           0.9899774042
 1           0.9365235602
 2           0.8028889501
 3           0.5950128899
 4           0.3698138248
 5           0.1896545726
 6           0.07955725188
 7           0.02712995629
 8           0.00746972044
 9           0.001644465374
10           0.0002852391917
11           3.810715862e-005
12           3.783265129e-006
13           2.628657967e-007
14           1.140870154e-008
15           2.328306437e-010
16           0
]
We now consider the probabilities of *ranges* of correct guesses.

First, calculate the probability of getting a range of guesses right,
by adding the exact probabilities of each from low ... high.
*/
  int low = 3; // Getting at least 3 right.
  int high = 5; // Getting as most 5 right.
  double sum = 0.;
  for (int i = low; i <= high; i++)
  {
    sum += pdf(quiz, i);
  }
  cout.precision(4);
  cout << "Probability of getting between "
    << low << " and " << high << " answers right by guessing is "
    << sum  << endl; // 0.61323
/*`
[pre
Probability of getting between 3 and 5 answers right by guessing is 0.6132
]
Or, usually better, we can use the difference of cdfs instead:
*/
  cout << "Probability of getting between " << low << " and " << high << " answers right by guessing is "
    <<  cdf(quiz, high) - cdf(quiz, low - 1) << endl; // 0.61323
/*`
[pre
Probability of getting between 3 and 5 answers right by guessing is 0.6132
]
And we can also try a few more combinations of high and low choices:
*/
  low = 1; high = 6;
  cout << "Probability of getting between " << low << " and " << high << " answers right by guessing is "
    <<  cdf(quiz, high) - cdf(quiz, low - 1) << endl; // 1 and 6 P= 0.91042
  low = 1; high = 8;
  cout << "Probability of getting between " << low << " and " << high << " answers right by guessing is "
    <<  cdf(quiz, high) - cdf(quiz, low - 1) << endl; // 1 <= x 8 P = 0.9825
  low = 4; high = 4;
  cout << "Probability of getting between " << low << " and " << high << " answers right by guessing is "
    <<  cdf(quiz, high) - cdf(quiz, low - 1) << endl; // 4 <= x 4 P = 0.22520

/*`
[pre
Probability of getting between 1 and 6 answers right by guessing is 0.9104
Probability of getting between 1 and 8 answers right by guessing is 0.9825
Probability of getting between 4 and 4 answers right by guessing is 0.2252
]
[h4 Using Binomial distribution moments]
Using moments of the distribution, we can say more about the spread of results from guessing.
*/
  cout << "By guessing, on average, one can expect to get " << mean(quiz) << " correct answers." << endl;
  cout << "Standard deviation is " << standard_deviation(quiz) << endl;
  cout << "So about 2/3 will lie within 1 standard deviation and get between "
    <<  ceil(mean(quiz) - standard_deviation(quiz))  << " and "
    << floor(mean(quiz) + standard_deviation(quiz)) << " correct." << endl;
  cout << "Mode (the most frequent) is " << mode(quiz) << endl;
  cout << "Skewness is " << skewness(quiz) << endl;

/*`
[pre
By guessing, on average, one can expect to get 4 correct answers.
Standard deviation is 1.732
So about 2/3 will lie within 1 standard deviation and get between 3 and 5 correct.
Mode (the most frequent) is 4
Skewness is 0.2887
]
[h4 Quantiles]
The quantiles (percentiles or percentage points) for a few probability levels:
*/
  cout << "Quartiles " << quantile(quiz, 0.25) << " to "
    << quantile(complement(quiz, 0.25)) << endl; // Quartiles
  cout << "1 standard deviation " << quantile(quiz, 0.33) << " to "
    << quantile(quiz, 0.67) << endl; // 1 sd
  cout << "Deciles " << quantile(quiz, 0.1)  << " to "
    << quantile(complement(quiz, 0.1))<< endl; // Deciles
  cout << "5 to 95% " << quantile(quiz, 0.05)  << " to "
    << quantile(complement(quiz, 0.05))<< endl; // 5 to 95%
  cout << "2.5 to 97.5% " << quantile(quiz, 0.025) << " to "
    <<  quantile(complement(quiz, 0.025)) << endl; // 2.5 to 97.5%
  cout << "2 to 98% " << quantile(quiz, 0.02)  << " to "
    << quantile(complement(quiz, 0.02)) << endl; //  2 to 98%

  cout << "If guessing then percentiles 1 to 99% will get " << quantile(quiz, 0.01)
    << " to " << quantile(complement(quiz, 0.01)) << " right." << endl;
/*`
Notice that these output integral values because the default policy is `integer_round_outwards`.
[pre
Quartiles 2 to 5
1 standard deviation 2 to 5
Deciles 1 to 6
5 to 95% 0 to 7
2.5 to 97.5% 0 to 8
2 to 98% 0 to 8
]
*/

//] [/binomial_quiz_example2]

//[discrete_quantile_real
/*`
Quantiles values are controlled by the __understand_dis_quant  quantile policy chosen.
The default is `integer_round_outwards`,
so the lower quantile is rounded down, and the upper quantile is rounded up.

But we might believe that the real values tell us a little more - see __math_discrete.

We could control the policy for *all* distributions by

  #define BOOST_MATH_DISCRETE_QUANTILE_POLICY real

  at the head of the program would make this policy apply
to this *one, and only*, translation unit.

Or we can now create a (typedef for) policy that has discrete quantiles real
(here avoiding any 'using namespaces ...' statements):
*/
  using boost::math::policies::policy;
  using boost::math::policies::discrete_quantile;
  using boost::math::policies::real;
  using boost::math::policies::integer_round_outwards; // Default.
  typedef boost::math::policies::policy<discrete_quantile<real> > real_quantile_policy;
/*`
Add a custom binomial distribution called ``real_quantile_binomial`` that uses ``real_quantile_policy``
*/
  using boost::math::binomial_distribution;
  typedef binomial_distribution<double, real_quantile_policy> real_quantile_binomial;
/*`
Construct an object of this custom distribution:
*/
  real_quantile_binomial quiz_real(questions, success_fraction);
/*`
And use this to show some quantiles - that now have real rather than integer values.
*/
  cout << "Quartiles " << quantile(quiz, 0.25) << " to "
    << quantile(complement(quiz_real, 0.25)) << endl; // Quartiles 2 to 4.6212
  cout << "1 standard deviation " << quantile(quiz_real, 0.33) << " to "
    << quantile(quiz_real, 0.67) << endl; // 1 sd 2.6654 4.194
  cout << "Deciles " << quantile(quiz_real, 0.1)  << " to "
    << quantile(complement(quiz_real, 0.1))<< endl; // Deciles 1.3487 5.7583
  cout << "5 to 95% " << quantile(quiz_real, 0.05)  << " to "
    << quantile(complement(quiz_real, 0.05))<< endl; // 5 to 95% 0.83739 6.4559
  cout << "2.5 to 97.5% " << quantile(quiz_real, 0.025) << " to "
    <<  quantile(complement(quiz_real, 0.025)) << endl; // 2.5 to 97.5% 0.42806 7.0688
  cout << "2 to 98% " << quantile(quiz_real, 0.02)  << " to "
    << quantile(complement(quiz_real, 0.02)) << endl; //  2 to 98% 0.31311 7.7880

  cout << "If guessing, then percentiles 1 to 99% will get " << quantile(quiz_real, 0.01)
    << " to " << quantile(complement(quiz_real, 0.01)) << " right." << endl;
/*`
[pre
Real Quantiles
Quartiles 2 to 4.621
1 standard deviation 2.665 to 4.194
Deciles 1.349 to 5.758
5 to 95% 0.8374 to 6.456
2.5 to 97.5% 0.4281 to 7.069
2 to 98% 0.3131 to 7.252
If guessing then percentiles 1 to 99% will get 0 to 7.788 right.
]
*/

//] [/discrete_quantile_real]
  }
  catch(const std::exception& e)
  { // Always useful to include try & catch blocks because
    // default policies are to throw exceptions on arguments that cause
    // errors like underflow, overflow.
    // Lacking try & catch blocks, the program will abort without a message below,
    // which may give some helpful clues as to the cause of the exception.
    std::cout <<
      "\n""Message from thrown exception was:\n   " << e.what() << std::endl;
  }
  return 0;
} // int main()



/*

Output is:

BAutorun "i:\boost-06-05-03-1300\libs\math\test\Math_test\debug\binomial_quiz_example.exe"
Binomial distribution example - guessing in a quiz.
In a quiz with 16 questions and with a probability of guessing right of 25 % or 1 in 4
Probability of getting none right is 0.0100226
Probability of getting exactly one right is 0.0534538
Probability of getting exactly two right is 0.133635
Probability of getting exactly 11 answers right by chance is 0.000247132
Probability of getting all 16 answers right by chance is 2.32831e-010
Guessed Probability
 0      0.0100226
 1      0.0534538
 2      0.133635
 3      0.207876
 4      0.225199
 5      0.180159
 6      0.110097
 7      0.0524273
 8      0.0196602
 9      0.00582526
10      0.00135923
11      0.000247132
12      3.43239e-005
13      3.5204e-006
14      2.51457e-007
15      1.11759e-008
16      2.32831e-010
Probability of getting none or one right is 0.0634764
Probability of getting none or one right is 0.0634764
Probability of getting <= 10 right (to fail) is 0.999715
Probability of getting > 10 right (to pass) is 0.000285239
Probability of getting > 10 right (to pass) is 0.000285239
Probability of getting less than 11 (< 11) answers right by guessing is 0.999715
Probability of getting at least 11(>= 11) answers right by guessing is 0.000285239, only 1 in 3505.83
At most (<=)
Guessed OK   Probability
 0           0.01002259576
 1           0.0634764398
 2           0.1971110499
 3           0.4049871101
 4           0.6301861752
 5           0.8103454274
 6           0.9204427481
 7           0.9728700437
 8           0.9925302796
 9           0.9983555346
10           0.9997147608
11           0.9999618928
12           0.9999962167
13           0.9999997371
14           0.9999999886
15           0.9999999998
16           1
At least (>)
Guessed OK   Probability
 0           0.9899774042
 1           0.9365235602
 2           0.8028889501
 3           0.5950128899
 4           0.3698138248
 5           0.1896545726
 6           0.07955725188
 7           0.02712995629
 8           0.00746972044
 9           0.001644465374
10           0.0002852391917
11           3.810715862e-005
12           3.783265129e-006
13           2.628657967e-007
14           1.140870154e-008
15           2.328306437e-010
16           0
Probability of getting between 3 and 5 answers right by guessing is 0.6132
Probability of getting between 3 and 5 answers right by guessing is 0.6132
Probability of getting between 1 and 6 answers right by guessing is 0.9104
Probability of getting between 1 and 8 answers right by guessing is 0.9825
Probability of getting between 4 and 4 answers right by guessing is 0.2252
By guessing, on average, one can expect to get 4 correct answers.
Standard deviation is 1.732
So about 2/3 will lie within 1 standard deviation and get between 3 and 5 correct.
Mode (the most frequent) is 4
Skewness is 0.2887
Quartiles 2 to 5
1 standard deviation 2 to 5
Deciles 1 to 6
5 to 95% 0 to 7
2.5 to 97.5% 0 to 8
2 to 98% 0 to 8
If guessing then percentiles 1 to 99% will get 0 to 8 right.
Quartiles 2 to 4.621
1 standard deviation 2.665 to 4.194
Deciles 1.349 to 5.758
5 to 95% 0.8374 to 6.456
2.5 to 97.5% 0.4281 to 7.069
2 to 98% 0.3131 to 7.252
If guessing, then percentiles 1 to 99% will get 0 to 7.788 right.

*/

