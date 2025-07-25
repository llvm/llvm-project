///////////////////////////////////////////////////////////////
//  Copyright 2017 John Maddock. Distributed under the Boost
//  Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_

#define BOOST_CHRONO_HEADER_ONLY

#ifdef _MSC_VER
#  define _SCL_SECURE_NO_WARNINGS
#endif


#include "performance.hpp"
#include "table_helper.hpp"
#include <boost/random.hpp>
#include <boost/math/tools/polynomial.hpp>
#include <boost/multiprecision/cpp_int.hpp>

unsigned max_reps = 1000;

template <class T>
struct tester
{
   tester()
   {
      a.assign(500, T());
      for(int i = 0; i < 500; ++i)
      {
         b.push_back(generate_random(false));
         c.push_back(generate_random(false));
         small.push_back(generate_random(true));
      }
   }
   double test_add()
   {
      stopwatch<boost::chrono::high_resolution_clock> w;
      for (unsigned repeats = 0; repeats < max_reps; ++repeats)
      {
         for(unsigned i = 0; i < b.size(); ++i)
            a[i] = b[i] + c[i];
      }
      return boost::chrono::duration_cast<boost::chrono::duration<double> >(w.elapsed()).count();
   }
   double test_subtract()
   {
      stopwatch<boost::chrono::high_resolution_clock> w;
      for (unsigned repeats = 0; repeats < max_reps; ++repeats)
      {
         for(unsigned i = 0; i < b.size(); ++i)
            a[i] = b[i] - c[i];
      }
      return boost::chrono::duration_cast<boost::chrono::duration<double> >(w.elapsed()).count();
   }
   double test_add_int()
   {
      stopwatch<boost::chrono::high_resolution_clock> w;
      for (unsigned repeats = 0; repeats < max_reps; ++repeats)
      {
         for(unsigned i = 0; i < b.size(); ++i)
            a[i] = b[i] + 1;
      }
      return boost::chrono::duration_cast<boost::chrono::duration<double> >(w.elapsed()).count();
   }
   double test_subtract_int()
   {
      stopwatch<boost::chrono::high_resolution_clock> w;
      for (unsigned repeats = 0; repeats < max_reps; ++repeats)
      {
         for(unsigned i = 0; i < b.size(); ++i)
            a[i] = b[i] - 1;
      }
      return boost::chrono::duration_cast<boost::chrono::duration<double> >(w.elapsed()).count();
   }
   double test_multiply()
   {
      stopwatch<boost::chrono::high_resolution_clock> w;
      for (unsigned repeats = 0; repeats < max_reps; ++repeats)
      {
         for(unsigned k = 0; k < b.size(); ++k)
            a[k] = b[k] * c[k];
      }
      return boost::chrono::duration_cast<boost::chrono::duration<double> >(w.elapsed()).count();
   }
   double test_multiply_int()
   {
      stopwatch<boost::chrono::high_resolution_clock> w;
      for (unsigned repeats = 0; repeats < max_reps; ++repeats)
      {
         for(unsigned i = 0; i < b.size(); ++i)
            a[i] = b[i] * 3;
      }
      return boost::chrono::duration_cast<boost::chrono::duration<double> >(w.elapsed()).count();
   }
   double test_divide()
   {
      stopwatch<boost::chrono::high_resolution_clock> w;
      for (unsigned repeats = 0; repeats < max_reps; ++repeats)
      {
         for(unsigned i = 0; i < b.size(); ++i)
            a[i] = b[i] / small[i];
      }
      return boost::chrono::duration_cast<boost::chrono::duration<double> >(w.elapsed()).count();
   }
   double test_divide_int()
   {
      stopwatch<boost::chrono::high_resolution_clock> w;
      for (unsigned repeats = 0; repeats < max_reps; ++repeats)
      {
         for(unsigned i = 0; i < b.size(); ++i)
            a[i] = b[i] / 3;
      }
      return boost::chrono::duration_cast<boost::chrono::duration<double> >(w.elapsed()).count();
   }
   double test_gcd()
   {
      using boost::integer::gcd;
      stopwatch<boost::chrono::high_resolution_clock> w;
      for (unsigned repeats = 0; repeats < max_reps; ++repeats)
      {
         for(unsigned i = 0; i < b.size(); ++i)
            a[i] = gcd(b[i], c[i]);
      }
      return boost::chrono::duration_cast<boost::chrono::duration<double> >(w.elapsed()).count();
   }

   double test_inplace_add()
   {
      stopwatch<boost::chrono::high_resolution_clock> w;
      for (unsigned repeats = 0; repeats < max_reps; ++repeats)
      {
         for (unsigned i = 0; i < b.size(); ++i)
            b[i] += c[i];
      }
      return boost::chrono::duration_cast<boost::chrono::duration<double> >(w.elapsed()).count();
   }
   double test_inplace_subtract()
   {
      stopwatch<boost::chrono::high_resolution_clock> w;
      for (unsigned repeats = 0; repeats < max_reps; ++repeats)
      {
         for (unsigned i = 0; i < b.size(); ++i)
            b[i] -= c[i];
      }
      return boost::chrono::duration_cast<boost::chrono::duration<double> >(w.elapsed()).count();
   }
   double test_inplace_add_int()
   {
      stopwatch<boost::chrono::high_resolution_clock> w;
      for (unsigned repeats = 0; repeats < max_reps; ++repeats)
      {
         for (unsigned i = 0; i < b.size(); ++i)
            b[i] += 1;
      }
      return boost::chrono::duration_cast<boost::chrono::duration<double> >(w.elapsed()).count();
   }
   double test_inplace_subtract_int()
   {
      stopwatch<boost::chrono::high_resolution_clock> w;
      for (unsigned repeats = 0; repeats < max_reps; ++repeats)
      {
         for (unsigned i = 0; i < b.size(); ++i)
            b[i] -= 1;
      }
      return boost::chrono::duration_cast<boost::chrono::duration<double> >(w.elapsed()).count();
   }
   double test_inplace_multiply()
   {
      stopwatch<boost::chrono::high_resolution_clock> w;
      for (unsigned repeats = 0; repeats < max_reps; ++repeats)
      {
         for (unsigned k = 0; k < b.size(); ++k)
            b[k] *= c[k];
      }
      return boost::chrono::duration_cast<boost::chrono::duration<double> >(w.elapsed()).count();
   }
   double test_inplace_multiply_int()
   {
      stopwatch<boost::chrono::high_resolution_clock> w;
      for (unsigned repeats = 0; repeats < max_reps; ++repeats)
      {
         for (unsigned i = 0; i < b.size(); ++i)
            b[i] *= 3;
      }
      return boost::chrono::duration_cast<boost::chrono::duration<double> >(w.elapsed()).count();
   }
   double test_inplace_divide()
   {
      stopwatch<boost::chrono::high_resolution_clock> w;
      for (unsigned repeats = 0; repeats < max_reps; ++repeats)
      {
         for (unsigned i = 0; i < b.size(); ++i)
            a[i] /= small[i];
      }
      return boost::chrono::duration_cast<boost::chrono::duration<double> >(w.elapsed()).count();
   }
   double test_inplace_divide_int()
   {
      stopwatch<boost::chrono::high_resolution_clock> w;
      for (unsigned repeats = 0; repeats < max_reps; ++repeats)
      {
         for (unsigned i = 0; i < b.size(); ++i)
            b[i] /= 3;
      }
      return boost::chrono::duration_cast<boost::chrono::duration<double> >(w.elapsed()).count();
   }
private:
   T generate_random(bool issmall)
   {
      boost::uniform_int<> ui(2, issmall ? 5 : 40), ui2(1, 10000);
      std::size_t  len = ui(gen);
      std::vector<typename T::value_type> values;
      for (std::size_t i = 0; i < len; ++i)
      {
         values.push_back(static_cast<typename T::value_type>(ui2(gen)));
      }
      return T(values.begin(), values.end());
   }
   std::vector<T> a, b, c, small;
   static boost::random::mt19937 gen;
};

template <class N>
boost::random::mt19937 tester<N>::gen;

template <class Number>
void test(const char* type)
{
   std::cout << "Testing type: " << type << std::endl;
   tester<boost::math::tools::polynomial<Number> > t;
   int count = 500 * max_reps;
   std::string table_name = "Polynomial Arithmetic (" + compiler_name() + ", " + platform_name() + ")";
   //
   // Now the actual tests:
   //
   report_execution_time(t.test_add() / count, table_name, "operator +", type);
   report_execution_time(t.test_subtract() / count, table_name, "operator -", type);
   report_execution_time(t.test_multiply() / count, table_name, "operator *", type);
   report_execution_time(t.test_divide() / count, table_name, "operator /", type);
   report_execution_time(t.test_add_int() / count, table_name, "operator + (int)", type);
   report_execution_time(t.test_subtract_int() / count, table_name, "operator - (int)", type);
   report_execution_time(t.test_multiply_int() / count, table_name, "operator * (int)", type);
   report_execution_time(t.test_divide_int() / count, table_name, "operator / (int)", type);
   report_execution_time(t.test_inplace_add() / count, table_name, "operator +=", type);
   report_execution_time(t.test_inplace_subtract() / count, table_name, "operator -=", type);
   report_execution_time(t.test_inplace_multiply() / count, table_name, "operator *=", type);
   report_execution_time(t.test_inplace_divide() / count, table_name, "operator /=", type);
   report_execution_time(t.test_inplace_add_int() / count, table_name, "operator += (int)", type);
   report_execution_time(t.test_inplace_subtract_int() / count, table_name, "operator -= (int)", type);
   report_execution_time(t.test_inplace_multiply_int() / count, table_name, "operator *= (int)", type);
   report_execution_time(t.test_inplace_divide_int() / count, table_name, "operator /= (int)", type);
   //report_execution_time(t.test_gcd() / count, table_name, "gcd", type);
}


int main()
{
   test<std::uint64_t>("std::uint64_t");
   test<double>("double");
   max_reps = 100;
   test<boost::multiprecision::cpp_int>("cpp_int");
   return 0;
}

