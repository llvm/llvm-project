
// normal_misc_examples.cpp

// Copyright Paul A. Bristow 2007, 2010, 2014, 2016.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Example of using normal distribution.

// Note that this file contains Quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

/*`
First we need some includes to access the normal distribution
(and some std output of course).
*/

#include <boost/cstdfloat.hpp> // MUST be first include!!!
// See Implementation of Float128 type, Overloading template functions with float128_t.

#include <boost/math/distributions/normal.hpp> // for normal_distribution.
  using boost::math::normal; // typedef provides default type of double.

#include <iostream>
  //using std::cout; using std::endl;
  //using std::left; using std::showpoint; using std::noshowpoint;
#include <iomanip>
  //using std::setw; using std::setprecision;
#include <limits>
  //using std::numeric_limits;

  /*!
Function max_digits10
Returns maximum number of possibly significant decimal digits for a floating-point type FPT,
even for older compilers/standard libraries that
lack support for std::std::numeric_limits<FPT>::max_digits10,
when the Kahan formula 2 + binary_digits * 0.3010 is used instead.
Also provides the correct result for Visual Studio 2010 where the max_digits10 provided for float is wrong.
*/
namespace boost
{
namespace math
{
template <typename FPT>
int max_digits10()
{
// Since max_digits10 is not defined (or wrong) on older systems, define a local max_digits10.

  // Usage:   int m = max_digits10<boost::float64_t>();
  const int m =
#if (defined BOOST_NO_CXX11_NUMERIC_LIMITS) || (_MSC_VER == 1600) // is wrongly 8 not 9 for VS2010.
  2 + std::numeric_limits<FPT>::digits * 3010/10000;
#else
  std::numeric_limits<FPT>::max_digits10;
#endif
  return m;
}
} // namespace math
} // namespace boost

template <typename FPT>
void normal_table()
{
  using namespace boost::math;

  FPT step = static_cast<FPT>(1.); // step in z.
  FPT range = static_cast<FPT>(10.); // min and max z = -range to +range.

  // Traditional tables are only computed to much lower precision.
  // but @c std::std::numeric_limits<double>::max_digits10;
  // on new Standard Libraries gives 17,
  // the maximum number of digits from 64-bit double that can possibly be significant.
  // @c std::std::numeric_limits<double>::digits10; == 15
  // is number of @b guaranteed digits, the other two digits being 'noisy'.
  // Here we use a custom version of max_digits10 which deals with those platforms
  // where @c std::numeric_limits is not specialized,
  // or @c std::numeric_limits<>::max_digits10 not implemented, or wrong.
  int precision = boost::math::max_digits10<FPT>();

// std::cout << typeid(FPT).name() << std::endl;
// demo_normal.cpp:85: undefined reference to `typeinfo for __float128'
// [@http://gcc.gnu.org/bugzilla/show_bug.cgi?id=43622   GCC 43622]
//  typeinfo for __float128 was missing GCC 4.9 Mar 2014, but OK for GCC 6.1.1.

   // Construct a standard normal distribution s, with
   // (default mean = zero, and standard deviation = unity)
   normal s;
   std::cout << "\nStandard normal distribution, mean = "<< s.mean()
      << ", standard deviation = " << s.standard_deviation() << std::endl;

  std::cout << "maxdigits_10 is " << precision
    << ", digits10 is " << std::numeric_limits<FPT>::digits10 << std::endl;

  std::cout << "Probability distribution function values" << std::endl;

  std::cout << "  z " "   PDF " << std::endl;
  for (FPT z = -range; z < range + step; z += step)
  {
    std::cout << std::left << std::setprecision(3) << std::setw(6) << z << " "
      << std::setprecision(precision) << std::setw(12) << pdf(s, z) << std::endl;
  }
  std::cout.precision(6); // Restore to default precision.

/*`And the area under the normal curve from -[infin] up to z,
  the cumulative distribution function (CDF).
*/
  // For a standard normal distribution:
  std::cout << "Standard normal mean = "<< s.mean()
    << ", standard deviation = " << s.standard_deviation() << std::endl;
  std::cout << "Integral (area under the curve) from - infinity up to z." << std::endl;
  std::cout << "  z " "   CDF " << std::endl;
  for (FPT z = -range; z < range + step; z += step)
  {
    std::cout << std::left << std::setprecision(3) << std::setw(6) << z << " "
      << std::setprecision(precision) << std::setw(12) << cdf(s, z) << std::endl;
  }
  std::cout.precision(6); // Reset to default precision.
} // template <typename FPT> void normal_table()

int main()
{
  std::cout << "\nExample: Normal distribution tables." << std::endl;

  using namespace boost::math;

  try
  {// Tip - always use try'n'catch blocks to ensure that messages from thrown exceptions are shown.

//[normal_table_1
#ifdef BOOST_FLOAT32_C
    normal_table<boost::float32_t>(); // Usually type float
#endif
    normal_table<boost::float64_t>(); // Usually type double. Assume that float64_t is always available.
#ifdef BOOST_FLOAT80_C
    normal_table<boost::float80_t>(); // Type long double on some X86 platforms.
#endif
#ifdef BOOST_FLOAT128_C
    normal_table<boost::float128_t>(); // Type _Quad on some Intel and __float128 on some GCC platforms.
#endif
    normal_table<boost::floatmax_t>();
//] [/normal_table_1 ]
  }
  catch(std::exception ex)
  {
    std::cout << "exception thrown " << ex.what() << std::endl;
  }

  return 0;
}  // int main()


/*

GCC 4.8.1 with quadmath

Example: Normal distribution tables.

Standard normal distribution, mean = 0, standard deviation = 1
maxdigits_10 is 9, digits10 is 6
Probability distribution function values
  z    PDF
-10    7.69459863e-023
-9     1.02797736e-018
-8     5.05227108e-015
-7     9.13472041e-012
-6     6.07588285e-009
-5     1.48671951e-006
-4     0.000133830226
-3     0.00443184841
-2     0.0539909665
-1     0.241970725
0      0.39894228
1      0.241970725
2      0.0539909665
3      0.00443184841
4      0.000133830226
5      1.48671951e-006
6      6.07588285e-009
7      9.13472041e-012
8      5.05227108e-015
9      1.02797736e-018
10     7.69459863e-023
Standard normal mean = 0, standard deviation = 1
Integral (area under the curve) from - infinity up to z.
  z    CDF
-10    7.61985302e-024
-9     1.12858841e-019
-8     6.22096057e-016
-7     1.27981254e-012
-6     9.86587645e-010
-5     2.86651572e-007
-4     3.16712418e-005
-3     0.00134989803
-2     0.0227501319
-1     0.158655254
0      0.5
1      0.841344746
2      0.977249868
3      0.998650102
4      0.999968329
5      0.999999713
6      0.999999999
7      1
8      1
9      1
10     1

Standard normal distribution, mean = 0, standard deviation = 1
maxdigits_10 is 17, digits10 is 15
Probability distribution function values
  z    PDF
-10    7.6945986267064199e-023
-9     1.0279773571668917e-018
-8     5.0522710835368927e-015
-7     9.1347204083645953e-012
-6     6.0758828498232861e-009
-5     1.4867195147342979e-006
-4     0.00013383022576488537
-3     0.0044318484119380075
-2     0.053990966513188063
-1     0.24197072451914337
0      0.3989422804014327
1      0.24197072451914337
2      0.053990966513188063
3      0.0044318484119380075
4      0.00013383022576488537
5      1.4867195147342979e-006
6      6.0758828498232861e-009
7      9.1347204083645953e-012
8      5.0522710835368927e-015
9      1.0279773571668917e-018
10     7.6945986267064199e-023
Standard normal mean = 0, standard deviation = 1
Integral (area under the curve) from - infinity up to z.
  z    CDF
-10    7.6198530241605945e-024
-9     1.1285884059538422e-019
-8     6.2209605742718204e-016
-7     1.279812543885835e-012
-6     9.865876450377014e-010
-5     2.8665157187919455e-007
-4     3.1671241833119972e-005
-3     0.0013498980316300957
-2     0.022750131948179216
-1     0.15865525393145705
0      0.5
1      0.84134474606854293
2      0.97724986805182079
3      0.9986501019683699
4      0.99996832875816688
5      0.99999971334842808
6      0.9999999990134123
7      0.99999999999872013
8      0.99999999999999933
9      1
10     1

Standard normal distribution, mean = 0, standard deviation = 1
maxdigits_10 is 21, digits10 is 18
Probability distribution function values
  z    PDF
-10    7.69459862670641993759e-023
-9     1.0279773571668916523e-018
-8     5.05227108353689273243e-015
-7     9.13472040836459525705e-012
-6     6.07588284982328608733e-009
-5     1.48671951473429788965e-006
-4     0.00013383022576488536764
-3     0.00443184841193800752729
-2     0.0539909665131880628364
-1     0.241970724519143365328
0      0.398942280401432702863
1      0.241970724519143365328
2      0.0539909665131880628364
3      0.00443184841193800752729
4      0.00013383022576488536764
5      1.48671951473429788965e-006
6      6.07588284982328608733e-009
7      9.13472040836459525705e-012
8      5.05227108353689273243e-015
9      1.0279773571668916523e-018
10     7.69459862670641993759e-023
Standard normal mean = 0, standard deviation = 1
Integral (area under the curve) from - infinity up to z.
  z    CDF
-10    7.61985302416059451083e-024
-9     1.12858840595384222719e-019
-8     6.22096057427182035917e-016
-7     1.279812543885834962e-012
-6     9.86587645037701399241e-010
-5     2.86651571879194547129e-007
-4     3.16712418331199717608e-005
-3     0.00134989803163009566139
-2     0.0227501319481792155242
-1     0.158655253931457046468
0      0.5
1      0.841344746068542925777
2      0.977249868051820791415
3      0.998650101968369896532
4      0.999968328758166880021
5      0.999999713348428076465
6      0.999999999013412299576
7      0.999999999998720134897
8      0.999999999999999333866
9      1
10     1

Standard normal distribution, mean = 0, standard deviation = 1
maxdigits_10 is 36, digits10 is 34
Probability distribution function values
  z    PDF
-10    7.69459862670641993759264402330435296e-023
-9     1.02797735716689165230378750485667109e-018
-8     5.0522710835368927324337437844893081e-015
-7     9.13472040836459525705208369548147081e-012
-6     6.07588284982328608733411870229841611e-009
-5     1.48671951473429788965346931561839483e-006
-4     0.00013383022576488536764006964663309418
-3     0.00443184841193800752728870762098267733
-2     0.0539909665131880628363703067407186609
-1     0.241970724519143365327522587904240936
0      0.398942280401432702863218082711682655
1      0.241970724519143365327522587904240936
2      0.0539909665131880628363703067407186609
3      0.00443184841193800752728870762098267733
4      0.00013383022576488536764006964663309418
5      1.48671951473429788965346931561839483e-006
6      6.07588284982328608733411870229841611e-009
7      9.13472040836459525705208369548147081e-012
8      5.0522710835368927324337437844893081e-015
9      1.02797735716689165230378750485667109e-018
10     7.69459862670641993759264402330435296e-023
Standard normal mean = 0, standard deviation = 1
Integral (area under the curve) from - infinity up to z.
  z    CDF
-10    7.61985302416059451083278826816793623e-024
-9     1.1285884059538422271881384555435713e-019
-8     6.22096057427182035917417257601387863e-016
-7     1.27981254388583496200054074948511201e-012
-6     9.86587645037701399241244820583623953e-010
-5     2.86651571879194547128505464808623238e-007
-4     3.16712418331199717608064048146587766e-005
-3     0.001349898031630095661392854111682027
-2     0.0227501319481792155241528519127314212
-1     0.158655253931457046467912164189328905
0      0.5
1      0.841344746068542925776512220181757584
2      0.977249868051820791414741051994496956
3      0.998650101968369896532351503992686048
4      0.999968328758166880021462930017150939
5      0.999999713348428076464813329948810861
6      0.999999999013412299575520592043176293
7      0.999999999998720134897212119540199637
8      0.999999999999999333866185224906075746
9      1
10     1

Standard normal distribution, mean = 0, standard deviation = 1
maxdigits_10 is 36, digits10 is 34
Probability distribution function values
  z    PDF
-10    7.69459862670641993759264402330435296e-023
-9     1.02797735716689165230378750485667109e-018
-8     5.0522710835368927324337437844893081e-015
-7     9.13472040836459525705208369548147081e-012
-6     6.07588284982328608733411870229841611e-009
-5     1.48671951473429788965346931561839483e-006
-4     0.00013383022576488536764006964663309418
-3     0.00443184841193800752728870762098267733
-2     0.0539909665131880628363703067407186609
-1     0.241970724519143365327522587904240936
0      0.398942280401432702863218082711682655
1      0.241970724519143365327522587904240936
2      0.0539909665131880628363703067407186609
3      0.00443184841193800752728870762098267733
4      0.00013383022576488536764006964663309418
5      1.48671951473429788965346931561839483e-006
6      6.07588284982328608733411870229841611e-009
7      9.13472040836459525705208369548147081e-012
8      5.0522710835368927324337437844893081e-015
9      1.02797735716689165230378750485667109e-018
10     7.69459862670641993759264402330435296e-023
Standard normal mean = 0, standard deviation = 1
Integral (area under the curve) from - infinity up to z.
  z    CDF
-10    7.61985302416059451083278826816793623e-024
-9     1.1285884059538422271881384555435713e-019
-8     6.22096057427182035917417257601387863e-016
-7     1.27981254388583496200054074948511201e-012
-6     9.86587645037701399241244820583623953e-010
-5     2.86651571879194547128505464808623238e-007
-4     3.16712418331199717608064048146587766e-005
-3     0.001349898031630095661392854111682027
-2     0.0227501319481792155241528519127314212
-1     0.158655253931457046467912164189328905
0      0.5
1      0.841344746068542925776512220181757584
2      0.977249868051820791414741051994496956
3      0.998650101968369896532351503992686048
4      0.999968328758166880021462930017150939
5      0.999999713348428076464813329948810861
6      0.999999999013412299575520592043176293
7      0.999999999998720134897212119540199637
8      0.999999999999999333866185224906075746
9      1
10     1

MSVC 2013 64-bit
1>
1>  Example: Normal distribution tables.
1>
1>  Standard normal distribution, mean = 0, standard deviation = 1
1>  maxdigits_10 is 9, digits10 is 6
1>  Probability distribution function values
1>    z    PDF
1>  -10    7.69459863e-023
1>  -9     1.02797736e-018
1>  -8     5.05227108e-015
1>  -7     9.13472041e-012
1>  -6     6.07588285e-009
1>  -5     1.48671951e-006
1>  -4     0.000133830226
1>  -3     0.00443184841
1>  -2     0.0539909665
1>  -1     0.241970725
1>  0      0.39894228
1>  1      0.241970725
1>  2      0.0539909665
1>  3      0.00443184841
1>  4      0.000133830226
1>  5      1.48671951e-006
1>  6      6.07588285e-009
1>  7      9.13472041e-012
1>  8      5.05227108e-015
1>  9      1.02797736e-018
1>  10     7.69459863e-023
1>  Standard normal mean = 0, standard deviation = 1
1>  Integral (area under the curve) from - infinity up to z.
1>    z    CDF
1>  -10    7.61985302e-024
1>  -9     1.12858841e-019
1>  -8     6.22096057e-016
1>  -7     1.27981254e-012
1>  -6     9.86587645e-010
1>  -5     2.86651572e-007
1>  -4     3.16712418e-005
1>  -3     0.00134989803
1>  -2     0.0227501319
1>  -1     0.158655254
1>  0      0.5
1>  1      0.841344746
1>  2      0.977249868
1>  3      0.998650102
1>  4      0.999968329
1>  5      0.999999713
1>  6      0.999999999
1>  7      1
1>  8      1
1>  9      1
1>  10     1
1>
1>  Standard normal distribution, mean = 0, standard deviation = 1
1>  maxdigits_10 is 17, digits10 is 15
1>  Probability distribution function values
1>    z    PDF
1>  -10    7.6945986267064199e-023
1>  -9     1.0279773571668917e-018
1>  -8     5.0522710835368927e-015
1>  -7     9.1347204083645953e-012
1>  -6     6.0758828498232861e-009
1>  -5     1.4867195147342979e-006
1>  -4     0.00013383022576488537
1>  -3     0.0044318484119380075
1>  -2     0.053990966513188063
1>  -1     0.24197072451914337
1>  0      0.3989422804014327
1>  1      0.24197072451914337
1>  2      0.053990966513188063
1>  3      0.0044318484119380075
1>  4      0.00013383022576488537
1>  5      1.4867195147342979e-006
1>  6      6.0758828498232861e-009
1>  7      9.1347204083645953e-012
1>  8      5.0522710835368927e-015
1>  9      1.0279773571668917e-018
1>  10     7.6945986267064199e-023
1>  Standard normal mean = 0, standard deviation = 1
1>  Integral (area under the curve) from - infinity up to z.
1>    z    CDF
1>  -10    7.6198530241605813e-024
1>  -9     1.1285884059538408e-019
1>  -8     6.2209605742718292e-016
1>  -7     1.2798125438858352e-012
1>  -6     9.8658764503770161e-010
1>  -5     2.8665157187919439e-007
1>  -4     3.1671241833119979e-005
1>  -3     0.0013498980316300957
1>  -2     0.022750131948179219
1>  -1     0.15865525393145707
1>  0      0.5
1>  1      0.84134474606854293
1>  2      0.97724986805182079
1>  3      0.9986501019683699
1>  4      0.99996832875816688
1>  5      0.99999971334842808
1>  6      0.9999999990134123
1>  7      0.99999999999872013
1>  8      0.99999999999999933
1>  9      1
1>  10     1
1>
1>  Standard normal distribution, mean = 0, standard deviation = 1
1>  maxdigits_10 is 17, digits10 is 15
1>  Probability distribution function values
1>    z    PDF
1>  -10    7.6945986267064199e-023
1>  -9     1.0279773571668917e-018
1>  -8     5.0522710835368927e-015
1>  -7     9.1347204083645953e-012
1>  -6     6.0758828498232861e-009
1>  -5     1.4867195147342979e-006
1>  -4     0.00013383022576488537
1>  -3     0.0044318484119380075
1>  -2     0.053990966513188063
1>  -1     0.24197072451914337
1>  0      0.3989422804014327
1>  1      0.24197072451914337
1>  2      0.053990966513188063
1>  3      0.0044318484119380075
1>  4      0.00013383022576488537
1>  5      1.4867195147342979e-006
1>  6      6.0758828498232861e-009
1>  7      9.1347204083645953e-012
1>  8      5.0522710835368927e-015
1>  9      1.0279773571668917e-018
1>  10     7.6945986267064199e-023
1>  Standard normal mean = 0, standard deviation = 1
1>  Integral (area under the curve) from - infinity up to z.
1>    z    CDF
1>  -10    7.6198530241605813e-024
1>  -9     1.1285884059538408e-019
1>  -8     6.2209605742718292e-016
1>  -7     1.2798125438858352e-012
1>  -6     9.8658764503770161e-010
1>  -5     2.8665157187919439e-007
1>  -4     3.1671241833119979e-005
1>  -3     0.0013498980316300957
1>  -2     0.022750131948179219
1>  -1     0.15865525393145707
1>  0      0.5
1>  1      0.84134474606854293
1>  2      0.97724986805182079
1>  3      0.9986501019683699
1>  4      0.99996832875816688
1>  5      0.99999971334842808
1>  6      0.9999999990134123
1>  7      0.99999999999872013
1>  8      0.99999999999999933
1>  9      1
1>  10     1


*/
