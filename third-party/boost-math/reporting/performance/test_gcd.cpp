//  Copyright Jeremy Murphy 2016.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma warning (disable : 4224)
#endif

#include <boost/integer/common_factor_rt.hpp>
#include <boost/math/special_functions/prime.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/integer.hpp>
#include <boost/random.hpp>
#include <boost/array.hpp>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>
#include <functional>
#include "fibonacci.hpp"
#include "../../test/table_type.hpp"
#include "table_helper.hpp"
#include "performance.hpp"


using namespace std;

boost::multiprecision::cpp_int total_sum(0);

template <typename Func, class Table>
double exec_timed_test_foo(Func f, const Table& data, double min_elapsed = 0.5)
{
    double t = 0;
    unsigned repeats = 1;
    typename Table::value_type::first_type sum{0};
    stopwatch<boost::chrono::high_resolution_clock> w;
    do
    {
       for(unsigned count = 0; count < repeats; ++count)
       {
          for(typename Table::size_type n = 0; n < data.size(); ++n)
            sum += f(data[n].first, data[n].second);
       }

        t = boost::chrono::duration_cast<boost::chrono::duration<double>>(w.elapsed()).count();
        if(t < min_elapsed)
            repeats *= 2;
    }
    while(t < min_elapsed);
    total_sum += sum;
    return t / repeats;
}


template <typename T>
struct test_function_template
{
    vector<pair<T, T> > const & data;
    const char* data_name;
    
    test_function_template(vector<pair<T, T> > const &data, const char* name) : data(data), data_name(name) {}
    
    template <typename Function>
    void operator()(pair<Function, string> const &f) const
    {
        auto result = exec_timed_test_foo(f.first, data);
        auto table_name = string("gcd method comparison with ") + compiler_name() + string(" on ") + platform_name();

        report_execution_time(result, 
                            table_name,
                            string(data_name), 
                            string(f.second) + "\n" + boost_name());
    }
};

boost::random::mt19937 rng;
boost::random::uniform_int_distribution<> d_0_6(0, 6);
boost::random::uniform_int_distribution<> d_1_20(1, 20);

template <class T>
T get_prime_products()
{
   int n_primes = d_0_6(rng);
   switch(n_primes)
   {
   case 0:
      // Generate a power of 2:
      return static_cast<T>(1u) << d_1_20(rng);
   case 1:
      // prime number:
      return boost::math::prime(d_1_20(rng) + 3);
   }
   T result = 1;
   for(int i = 0; i < n_primes; ++i)
      result *= boost::math::prime(d_1_20(rng) + 3) * boost::math::prime(d_1_20(rng) + 3) * boost::math::prime(d_1_20(rng) + 3) * boost::math::prime(d_1_20(rng) + 3) * boost::math::prime(d_1_20(rng) + 3);
   return result;
}

template <class T>
T get_uniform_random()
{
   static boost::random::uniform_int_distribution<T> minimax((std::numeric_limits<T>::min)(), (std::numeric_limits<T>::max)());
   return minimax(rng);
}

template <class T>
inline bool even(T const& val)
{
   return !(val & 1u);
}

template <class Backend, boost::multiprecision::expression_template_option ExpressionTemplates>
inline bool even(boost::multiprecision::number<Backend, ExpressionTemplates> const& val)
{
   return !bit_test(val, 0);
}

template <class T>
T euclid_textbook(T a, T b)
{
   using std::swap;
   if(a < b)
      swap(a, b);
   while(b)
   {
      T t = b;
      b = a % b;
      a = t;
   }
   return a;
}

template <class T>
T binary_textbook(T u, T v)
{
   if(u && v)
   {
      unsigned shifts = (std::min)(boost::multiprecision::lsb(u), boost::multiprecision::lsb(v));
      if(shifts)
      {
         u >>= shifts;
         v >>= shifts;
      }
      while(u)
      {
         unsigned bit_index = boost::multiprecision::lsb(u);
         if(bit_index)
         {
            u >>= bit_index;
         }
         else if(bit_index = boost::multiprecision::lsb(v))
         {
            v >>= bit_index;
         }
         else
         {
            if(u < v)
               v = (v - u) >> 1u;
            else
               u = (u - v) >> 1u;
         }
      }
      return v << shifts;
   }
   return u + v;
}

template <typename Integer>
inline BOOST_CXX14_CONSTEXPR Integer gcd_default(Integer a, Integer b) BOOST_GCD_NOEXCEPT(Integer)
{
   using boost::integer::gcd;
   return gcd(a, b);
}


template <class T>
void test_type(const char* name)
{
   using namespace boost::integer::gcd_detail;
   typedef T int_type;
   std::vector<pair<int_type, int_type> > data;

   for(unsigned i = 0; i < 1000; ++i)
   {
      data.push_back(std::make_pair(get_prime_products<T>(), get_prime_products<T>()));
   }
   std::string row_name("gcd<");
   row_name += name;
   row_name += "> (random prime number products)";
   
   typedef pair< function<int_type(int_type, int_type)>, string> f_test;
   array<f_test, 6> test_functions{ { 
      { gcd_default<int_type>, "gcd" },
      { Euclid_gcd<int_type>, "Euclid_gcd" },
      { Stein_gcd<int_type>, "Stein_gcd" } ,
      { mixed_binary_gcd<int_type>, "mixed_binary_gcd" }, 
      { binary_textbook<int_type>, "Stein_gcd_textbook" },
      { euclid_textbook<int_type>, "gcd_euclid_textbook" },
   } };
   for_each(begin(test_functions), end(test_functions), test_function_template<int_type>(data, row_name.c_str()));

   data.clear();
   for(unsigned i = 0; i < 1000; ++i)
   {
      data.push_back(std::make_pair(get_uniform_random<T>(), get_uniform_random<T>()));
   }
   row_name.erase();
   row_name += "gcd<";
   row_name += name;
   row_name += "> (uniform random numbers)";
   for_each(begin(test_functions), end(test_functions), test_function_template<int_type>(data, row_name.c_str()));

   // Fibonacci number tests:
   row_name.erase();
   row_name += "gcd<";
   row_name += name;
   row_name += "> (adjacent Fibonacci numbers)";
   for_each(begin(test_functions), end(test_functions), test_function_template<int_type>(fibonacci_numbers_permution_1<T>(), row_name.c_str()));

   row_name.erase();
   row_name += "gcd<";
   row_name += name;
   row_name += "> (permutations of Fibonacci numbers)";
   for_each(begin(test_functions), end(test_functions), test_function_template<int_type>(fibonacci_numbers_permution_2<T>(), row_name.c_str()));

   row_name.erase();
   row_name += "gcd<";
   row_name += name;
   row_name += "> (Trivial cases)";
   for_each(begin(test_functions), end(test_functions), test_function_template<int_type>(trivial_gcd_test_cases<T>(), row_name.c_str()));
}

/*******************************************************************************************************************/

template <class T>
T generate_random(unsigned bits_wanted)
{
   static boost::random::mt19937 gen;
   typedef boost::random::mt19937::result_type random_type;

   T max_val;
   unsigned digits;
   if(std::numeric_limits<T>::is_bounded && (bits_wanted == (unsigned)std::numeric_limits<T>::digits))
   {
      max_val = (std::numeric_limits<T>::max)();
      digits = std::numeric_limits<T>::digits;
   }
   else
   {
      max_val = T(1) << bits_wanted;
      digits = bits_wanted;
   }

   unsigned bits_per_r_val = std::numeric_limits<random_type>::digits - 1;
   while((random_type(1) << bits_per_r_val) > (gen.max)()) --bits_per_r_val;

   unsigned terms_needed = digits / bits_per_r_val + 1;

   T val = 0;
   for(unsigned i = 0; i < terms_needed; ++i)
   {
      val *= (gen.max)();
      val += gen();
   }
   val %= max_val;
   return val;
}

template <typename N>
N gcd_stein(N m, N n)
{
   BOOST_MATH_ASSERT(m >= static_cast<N>(0));
   BOOST_MATH_ASSERT(n >= static_cast<N>(0));
   if(m == N(0)) return n;
   if(n == N(0)) return m;
           // m > 0 && n > 0
   unsigned d_m = 0;
   while(even(m)) { m >>= 1; d_m++; }
   unsigned d_n = 0;
   while(even(n)) { n >>= 1; d_n++; }
           // odd(m) && odd(n)
      while(m != n) {
      if(n > m) swap(n, m);
      m -= n;
      do m >>= 1; while(even(m));
                  // m == n
   }
   return m << (std::min)(d_m, d_n);
}


boost::multiprecision::cpp_int big_gcd(const boost::multiprecision::cpp_int& a, const boost::multiprecision::cpp_int& b)
{
   return boost::multiprecision::gcd(a, b);
}

namespace boost { namespace multiprecision { namespace backends {

template <unsigned MinBits1, unsigned MaxBits1, cpp_integer_type SignType1, cpp_int_check_type Checked1, class Allocator1>
inline typename enable_if_c<!is_trivial_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value>::type
   eval_gcd_new(
      cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>& result, 
      const cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>& a, 
      const cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>& b)
{
   using default_ops::eval_lsb;
   using default_ops::eval_is_zero;
   using default_ops::eval_get_sign;

   if(a.size() == 1)
   {
      eval_gcd(result, b, *a.limbs());
      return;
   }
   if(b.size() == 1)
   {
      eval_gcd(result, a, *b.limbs());
      return;
   }

   int shift;

   cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> u(a), v(b), mod;

   int s = eval_get_sign(u);

   /* GCD(0,x) := x */
   if(s < 0)
   {
      u.negate();
   }
   else if(s == 0)
   {
      result = v;
      return;
   }
   s = eval_get_sign(v);
   if(s < 0)
   {
      v.negate();
   }
   else if(s == 0)
   {
      result = u;
      return;
   }

   /* Let shift := lg K, where K is the greatest power of 2
   dividing both u and v. */

   unsigned us = eval_lsb(u);
   unsigned vs = eval_lsb(v);
   shift = (std::min)(us, vs);
   eval_right_shift(u, us);
   eval_right_shift(v, vs);

   // From now on access u and v via pointers, that way we have a trivial swap:
   cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>* up(&u), *vp(&v), *mp(&mod);

   do 
   {
      /* Now u and v are both odd, so diff(u, v) is even.
      Let u = min(u, v), v = diff(u, v)/2. */
      s = up->compare(*vp);
      if(s > 0)
         std::swap(up, vp);
      if(s == 0)
         break;
      if(vp->size() <= 2)
      {
         if(vp->size() == 1)
            *up = boost::integer::gcd_detail::mixed_binary_gcd(*vp->limbs(), *up->limbs());
         else
         {
            double_limb_type i, j;
            i = vp->limbs()[0] | (static_cast<double_limb_type>(vp->limbs()[1]) << sizeof(limb_type) * CHAR_BIT);
            j = (up->size() == 1) ? *up->limbs() : up->limbs()[0] | (static_cast<double_limb_type>(up->limbs()[1]) << sizeof(limb_type) * CHAR_BIT);
            u = boost::integer::gcd_detail::mixed_binary_gcd(i, j);
         }
         break;
      }
      if(vp->size() > up->size() /*eval_msb(*vp) > eval_msb(*up) + 32*/)
      {
         eval_modulus(*mp, *vp, *up);
         std::swap(vp, mp);
         eval_subtract(*up, *vp);
         if(eval_is_zero(*vp) == 0)
         {
            vs = eval_lsb(*vp);
            eval_right_shift(*vp, vs);
         }
         else
            break;
         if(eval_is_zero(*up) == 0)
         {
            vs = eval_lsb(*up);
            eval_right_shift(*up, vs);
         }
         else
         {
            std::swap(up, vp);
            break;
         }
      }
      else
      {
         eval_subtract(*vp, *up);
         vs = eval_lsb(*vp);
         eval_right_shift(*vp, vs);
      }
   } 
   while(true);

   result = *up;
   eval_left_shift(result, shift);
}

}}}


boost::multiprecision::cpp_int big_gcd_new(const boost::multiprecision::cpp_int& a, const boost::multiprecision::cpp_int& b)
{
   boost::multiprecision::cpp_int result;
   boost::multiprecision::backends::eval_gcd_new(result.backend(), a.backend(), b.backend());
   return result;
}

#if 0
void test_n_bits(unsigned n, std::string data_name, const std::vector<pair<boost::multiprecision::cpp_int, boost::multiprecision::cpp_int> >* p_data = 0)
{
   using namespace boost::math::detail;
   typedef boost::multiprecision::cpp_int int_type;
   std::vector<pair<int_type, int_type> > data, data2;

   for(unsigned i = 0; i < 1000; ++i)
   {
      data.push_back(std::make_pair(generate_random<int_type>(n), generate_random<int_type>(n)));
   }

   typedef pair< function<int_type(int_type, int_type)>, string> f_test;
   array<f_test, 2> test_functions{ { /*{ Stein_gcd<int_type>, "Stein_gcd" } ,{ Euclid_gcd<int_type>, "Euclid_gcd" },{ binary_textbook<int_type>, "Stein_gcd_textbook" },{ euclid_textbook<int_type>, "gcd_euclid_textbook" },{ mixed_binary_gcd<int_type>, "mixed_binary_gcd" },{ gcd_stein<int_type>, "gcd_stein" },*/{ big_gcd, "boost::multiprecision::gcd" },{ big_gcd_new, "big_gcd_new" } } };
   for_each(begin(test_functions), end(test_functions), test_function_template<int_type>(p_data ? *p_data : data, data_name.c_str(), true));
}
#endif

int main()
{
    test_type<unsigned short>("unsigned short");
    test_type<unsigned>("unsigned");
    test_type<unsigned long>("unsigned long");
    test_type<unsigned long long>("unsigned long long");

    test_type<boost::multiprecision::uint256_t>("boost::multiprecision::uint256_t");
    test_type<boost::multiprecision::uint512_t>("boost::multiprecision::uint512_t");
    test_type<boost::multiprecision::uint1024_t>("boost::multiprecision::uint1024_t");

    /*
    test_n_bits(16, "   16 bit random values");
    test_n_bits(32, "   32 bit random values");
    test_n_bits(64, "   64 bit random values");
    test_n_bits(125, "  125 bit random values");
    test_n_bits(250, "  250 bit random values");
    test_n_bits(500, "  500 bit random values");
    test_n_bits(1000, " 1000 bit random values");
    test_n_bits(5000, " 5000 bit random values");
    test_n_bits(10000, "10000 bit random values");
    //test_n_bits(100000);
    //test_n_bits(1000000);

    test_n_bits(0, "consecutive first 1000 fibonacci numbers", &fibonacci_numbers_cpp_int_permution_1());
    test_n_bits(0, "permutations of first 1000 fibonacci numbers", &fibonacci_numbers_cpp_int_permution_2());
    */
    return 0;
}
