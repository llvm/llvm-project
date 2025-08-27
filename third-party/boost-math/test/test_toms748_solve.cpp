//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pch.hpp>

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <boost/math/tools/toms748_solve.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <iostream>
#include <iomanip>

//
// Test functor implements the same test cases as used by
// "Algorithm 748: Enclosing Zeros of Continuous Functions"
// by G.E. Alefeld, F.A. Potra and Yixun Shi.
//
// Plus two more: one for inverting the incomplete gamma,
// and one for inverting the incomplete beta.
//
template <class T>
struct toms748tester
{
   toms748tester(unsigned i) : k(i), ip(0), a(0), b(0){}
   toms748tester(unsigned i, int ip_) : k(i), ip(ip_), a(0), b(0){}
   toms748tester(unsigned i, T a_, T b_) : k(i), ip(0), a(a_), b(b_){}

   static unsigned total_calls()
   {
      return invocations;
   }
   static void reset()
   {
      invocations = 0;
   }

   T operator()(T x)
   {
      using namespace std;

      ++invocations;
      switch(k)
      {
      case 1:
         return sin(x) - x / 2;
      case 2:
         {
            T r = 0;
            for(int i = 1; i <= 20; ++i)
            {
               T p = (2 * i - 5);
               T q = x - i * i;
               r += p * p / (q * q * q);
            }
            r *= -2;
            return r;
         }
      case 3:
         return a * x * exp(b * x);
      case 4:
         return pow(x, b) - a;
      case 5:
         return sin(x) - 0.5;
      case 6:
         return 2 * x * exp(-T(ip)) - 2 * exp(-ip * x) + 1;
      case 7:
         return (1 + (1 - ip) * (1 - ip)) * x - (1 - ip * x) * (1 - ip * x);
      case 8:
         return x * x - pow(1 - x, ip);
      case 9:
         return (1 + (1 - ip) * (1 - ip) * (1 - ip) * (1 - ip)) * x - (1 - ip * x) * (1 - ip * x) * (1 - ip * x) * (1 - ip * x);
      case 10:
         return exp(-ip * x) * (x - 1) + pow(x, T(ip));
      case 11:
         return (ip * x - 1) / ((ip - 1) * x);
      case 12:
         return pow(x, T(1)/ip) - pow(T(ip), T(1)/ip);
      case 13:
         return x == 0 ? 0 : x * exp(-1 / (x * x));
      case 14:
         return x >= 0 ? (T(ip)/20) * (x / 1.5f + sin(x) - 1) : -T(ip)/20;
      case 15:
         {
            T d = 2e-3f / (1+ip);
            if(x > d)
               return exp(1.0) - 1.859;
            else if(x > 0)
               return exp((ip+1)*x*1000 / 2) - 1.859;
            return -0.859f;
         }
      case 16:
         {
            return boost::math::gamma_q(x, a) - b;
         }
      case 17:
         return boost::math::ibeta(x, a, 0.5) - b;
      }
      return 0;
   }
private:
   int k; // index of problem.
   int ip; // integer parameter
   T a, b; // real parameter

   static unsigned invocations;
};

template <class T>
unsigned toms748tester<T>::invocations = 0;

std::uintmax_t total = 0;
std::uintmax_t invocations = 0;

template <class T>
void run_test(T a, T b, int id)
{
   std::uintmax_t c = 1000;
   std::pair<double, double> r = toms748_solve(toms748tester<double>(id), 
      a, 
      b, 
      boost::math::tools::eps_tolerance<double>(std::numeric_limits<double>::digits), 
      c);
   BOOST_CHECK_EQUAL(c, toms748tester<double>::total_calls());
   total += c;
   invocations += toms748tester<double>::total_calls();
   std::cout << "Function " << id << "\nresult={" << r.first << ", " << r.second << "} total calls=" << toms748tester<double>::total_calls() << "\n\n";
   toms748tester<double>::reset();
}

template <class T>
void run_test(T a, T b, int id, int p)
{
   std::uintmax_t c = 1000;
   std::pair<double, double> r = toms748_solve(toms748tester<double>(id, p), 
      a, 
      b, 
      boost::math::tools::eps_tolerance<double>(std::numeric_limits<double>::digits), 
      c);
   BOOST_CHECK_EQUAL(c, toms748tester<double>::total_calls());
   total += c;
   invocations += toms748tester<double>::total_calls();
   std::cout << "Function " << id << "\nresult={" << r.first << ", " << r.second << "} total calls=" << toms748tester<double>::total_calls() << "\n\n";
   toms748tester<double>::reset();
}

template <class T>
void run_test(T a, T b, int id, T p1, T p2)
{
   std::uintmax_t c = 1000;
   std::pair<double, double> r = toms748_solve(toms748tester<double>(id, p1, p2), 
      a, 
      b, 
      boost::math::tools::eps_tolerance<double>(std::numeric_limits<double>::digits), 
      c);
   BOOST_CHECK_EQUAL(c, toms748tester<double>::total_calls());
   total += c;
   invocations += toms748tester<double>::total_calls();
   std::cout << "Function " << id << "\n   Result={" << r.first << ", " << r.second << "} total calls=" << toms748tester<double>::total_calls() << "\n\n";
   toms748tester<double>::reset();
}

BOOST_AUTO_TEST_CASE( test_main )
{
   std::cout << std::setprecision(18);
   run_test(3.14/2, 3.14, 1);

   for(int i = 1; i <= 10; i += 1)
   {
      run_test(i*i + 1e-9, (i+1)*(i+1) - 1e-9, 2);
   }

   run_test(-9.0, 31.0, 3, -40.0, -1.0);
   run_test(-9.0, 31.0, 3, -100.0, -2.0);
   run_test(-9.0, 31.0, 3, -200.0, -3.0);

   for(int n = 4; n <= 12; n += 2)
   {
      run_test(0.0, 5.0, 4, 0.2, double(n));
   }
   for(int n = 4; n <= 12; n += 2)
   {
      run_test(0.0, 5.0, 4, 1.0, double(n));
   }
   for(int n = 8; n <= 14; n += 2)
   {
      run_test(-0.95, 4.05, 4, 1.0, double(n));
   }
   run_test(0.0, 1.5, 5);
   for(int n = 1; n <= 5; ++n)
   {
      run_test(0.0, 1.0, 6, n);
   }
   for(int n = 20; n <= 100; n += 20)
   {
      run_test(0.0, 1.0, 6, n);
   }
   run_test(0.0, 1.0, 7, 5);
   run_test(0.0, 1.0, 7, 10);
   run_test(0.0, 1.0, 7, 20);
   run_test(0.0, 1.0, 8, 2);
   run_test(0.0, 1.0, 8, 5);
   run_test(0.0, 1.0, 8, 10);
   run_test(0.0, 1.0, 8, 15);
   run_test(0.0, 1.0, 8, 20);
   run_test(0.0, 1.0, 9, 1);
   run_test(0.0, 1.0, 9, 2);
   run_test(0.0, 1.0, 9, 4);
   run_test(0.0, 1.0, 9, 5);
   run_test(0.0, 1.0, 9, 8);
   run_test(0.0, 1.0, 9, 15);
   run_test(0.0, 1.0, 9, 20);
   run_test(0.0, 1.0, 10, 1);
   run_test(0.0, 1.0, 10, 5);
   run_test(0.0, 1.0, 10, 10);
   run_test(0.0, 1.0, 10, 15);
   run_test(0.0, 1.0, 10, 20);

   run_test(0.01, 1.0, 11, 2);
   run_test(0.01, 1.0, 11, 5);
   run_test(0.01, 1.0, 11, 15);
   run_test(0.01, 1.0, 11, 20);

   for(int n = 2; n <= 6; ++n)
      run_test(1.0, 100.0, 12, n);
   for(int n = 7; n <= 33; n+=2)
      run_test(1.0, 100.0, 12, n);

   run_test(-1.0, 4.0, 13);

   for(int n = 1; n <= 40; ++n)
      run_test(-1e4, 3.14/2, 14, n);

   for(int n = 20; n <= 40; ++n)
      run_test(-1e4, 1e-4, 15, n);

   for(int n = 100; n <= 1000; n+=100)
      run_test(-1e4, 1e-4, 15, n);

   std::cout << "Total iterations consumed = " << total << std::endl;
   std::cout << "Total function invocations consumed = " << invocations << std::endl << std::endl;

   BOOST_CHECK(invocations < 3150);

   std::cout << std::setprecision(18);

   for(int n = 5; n <= 100; n += 10)
      run_test(sqrt(double(n)), double(n+1), 16, (double)n, 0.4);

   for(int n = 5; n <= 100; n += 10)
      run_test(double(n / 2), double(2*n), 17, (double)n, 0.4);


   for(int n = 4; n <= 12; n += 2)
   {
      std::uintmax_t c = 1000;
      std::pair<double, double> r = bracket_and_solve_root(toms748tester<double>(4, 0.2, double(n)), 
         2.0, 
         2.0,
         true,
         boost::math::tools::eps_tolerance<double>(std::numeric_limits<double>::digits), 
         c);
      std::cout << std::setprecision(18);
      std::cout << "Function " << 4 << "\n   Result={" << r.first << ", " << r.second << "} total calls=" << toms748tester<double>::total_calls() << "\n\n";
      toms748tester<double>::reset();
      BOOST_CHECK(c < 20);
   }
}

