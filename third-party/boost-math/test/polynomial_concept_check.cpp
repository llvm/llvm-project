// Copyright Nick Thompson, 2017
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/concepts/std_real_concept.hpp>
#include <boost/math/tools/polynomial.hpp>
#include <boost/multiprecision/cpp_int.hpp>

template <class T>
bool check_concepts()
{
   boost::math::tools::polynomial<T> a(2), b(3), c(4);

   a += b;
   a -= b;
   a *= b;
   a /= b;
   a %= b;
   a = c;
   a += b + c;
   a += b - c;
   a += b * c;
   a += b / c;
   a += b % c;

   int i = 4;

   a += i;
   a -= i;
   a *= i;
   a /= i;
   a %= i;
   a += b + i;
   a += i + b;
   a += b - i;
   a += i - b;
   a += b * i;
   a += i * b;
   a += b / i;
   a += b % i;

   bool bb = false;
   bb |= a == b;
   bb |= a != b;

   return bb;
}

int main()
{
   check_concepts<int>();
   check_concepts<boost::multiprecision::cpp_int>();
   return 0;
}

