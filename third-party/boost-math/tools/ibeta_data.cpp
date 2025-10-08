//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "mp_t.hpp"
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <map>
#include <boost/math/tools/test_data.hpp>
#include <boost/random.hpp>

using namespace boost::math::tools;
using namespace boost::math;
using namespace std;

template <class T>
struct ibeta_fraction1_t
{
   typedef std::pair<T, T> result_type;

   ibeta_fraction1_t(T a_, T b_, T x_) : a(a_), b(b_), x(x_), k(1) {}

   result_type operator()()
   {
      T aN;
      if(k & 1)
      {
         int m = (k - 1) / 2;
         aN = -(a + m) * (a + b + m) * x;
         aN /= a + 2*m;
         aN /= a + 2*m + 1;
      }
      else
      {
         int m = k / 2;
         aN = m * (b - m) *x;
         aN /= a + 2*m - 1;
         aN /= a + 2*m;
      }
      ++k;
      return std::make_pair(aN, T(1));
   }

private:
   T a, b, x;
   int k;
};

//
// This function caches previous calls to beta
// just so we can speed things up a bit:
//
template <class T>
T get_beta(T a, T b)
{
   static std::map<std::pair<T, T>, T> m;

   if(a < b)
      std::swap(a, b);

   std::pair<T, T> p(a, b);
   typename std::map<std::pair<T, T>, T>::const_iterator i = m.find(p);
   if(i != m.end())
      return i->second;

   T r = beta(a, b);
   p.first = a;
   p.second = b;
   m[p] = r;

   return r;
}

//
// compute the continued fraction:
//
template <class T>
T get_ibeta_fraction1(T a, T b, T x, T log_scaling = 0)
{
   using std::isnormal;
   ibeta_fraction1_t<T> f(a, b, x);
   T fract = boost::math::tools::continued_fraction_a(f, boost::math::policies::digits<T, boost::math::policies::policy<> >());
   T denom = (a * (fract + 1));
   T num = pow(x, a) * pow(1 - x, b);
   if(denom == 0)
      return -1;
   else if (!isnormal(num) || log_scaling)
   {
      return exp(a * log(x) + b * log(1 - x) - log(denom) - log_scaling);
   }
   else if (num == 0)
      return 0;
   
   return num / denom;
}
//
// calculate the incomplete beta from the fraction:
//
template <class T>
std::pair<T,T> ibeta_fraction1(T a, T b, T x)
{
   T bet = get_beta(a, b);
   if(x > ((a+1)/(a+b+2)))
   {
      T fract = get_ibeta_fraction1(b, a, T(1-x));
      if(fract/bet > 0.75)
      {
         fract = get_ibeta_fraction1(a, b, x);
         return std::make_pair(fract, bet - fract);
      }
      return std::make_pair(bet - fract, fract);
   }
   T fract = get_ibeta_fraction1(a, b, x);
   if(fract/bet > 0.75)
   {
      fract = get_ibeta_fraction1(b, a, T(1-x));
      return std::make_pair(bet - fract, fract);
   }
   return std::make_pair(fract, bet - fract);

}
//
// calculate the regularised incomplete beta from the fraction:
//
template <class T>
std::pair<T,T> ibeta_fraction1_regular(T a, T b, T x)
{
   T bet = get_beta(a, b);
   T log_scaling = 0;
   if (bet == 0)
   {
      log_scaling = boost::math::lgamma(a) + boost::math::lgamma(b) - boost::math::lgamma(a + b);
      bet = 1;
   }
   if(x > ((a+1)/(a+b+2)))
   {
      T fract = get_ibeta_fraction1(b, a, T(1-x), log_scaling);
      if(fract == 0)
         bet = 1;  // normalise so we don't get 0/0
      else if(bet == 0)
         return std::make_pair(T(-1), T(-1)); // Yikes!!
      if(fract / bet > 0.75)
      {
         fract = get_ibeta_fraction1(a, b, x, log_scaling);
         return std::make_pair(fract / bet, 1 - (fract / bet));
      }
      return std::make_pair(1 - (fract / bet), fract / bet);
   }
   T fract = get_ibeta_fraction1(a, b, x, log_scaling);
   if(fract / bet > 0.75)
   {
      fract = get_ibeta_fraction1(b, a, T(1-x), log_scaling);
      return std::make_pair(1 - (fract / bet), fract / bet);
   }
   return std::make_pair(fract / bet, 1 - (fract / bet));
}

//
// we absolutely must truncate the input values to float
// precision: we have to be certain that the input values
// can be represented exactly in whatever width floating
// point type we are testing, otherwise the output will 
// necessarily be off.
//
float external_f;
float force_truncate(const float* f)
{
   external_f = *f;
   return external_f;
}

float truncate_to_float(mp_t r)
{
   float f = boost::math::tools::real_cast<float>(r);
   return force_truncate(&f);
}

boost::mt19937 rnd;
boost::uniform_real<float> ur_a(1.0F, 5.0F);
boost::variate_generator<boost::mt19937, boost::uniform_real<float> > gen(rnd, ur_a);
boost::uniform_real<float> ur_a2(0.0F, 100.0F);
boost::variate_generator<boost::mt19937, boost::uniform_real<float> > gen2(rnd, ur_a2);

struct beta_data_generator
{
   boost::math::tuple<mp_t, mp_t, mp_t, mp_t, mp_t, mp_t, mp_t> operator()(mp_t ap, mp_t bp, mp_t x_)
   {
      float a = truncate_to_float(real_cast<float>(gen() * pow(mp_t(10), ap)));      
      float b = truncate_to_float(real_cast<float>(gen() * pow(mp_t(10), bp))); 
      float x = truncate_to_float(real_cast<float>(x_));
      std::cout << a << " " << b << " " << x << std::endl;
      std::pair<mp_t, mp_t> ib_full = ibeta_fraction1(mp_t(a), mp_t(b), mp_t(x));
      std::pair<mp_t, mp_t> ib_reg = ibeta_fraction1_regular(mp_t(a), mp_t(b), mp_t(x));
      return boost::math::make_tuple(a, b, x, ib_full.first, ib_full.second, ib_reg.first, ib_reg.second);
   }
};

// medium sized values:
struct beta_data_generator_medium
{
   boost::math::tuple<mp_t, mp_t, mp_t, mp_t, mp_t, mp_t, mp_t> operator()(mp_t x_)
   {
      mp_t a = gen2();      
      mp_t b = gen2(); 
      mp_t x = x_;
      a = ConvPrec(a, 22);
      b = ConvPrec(b, 22);
      x = ConvPrec(x, 22);
      std::cout << a << " " << b << " " << x << std::endl;
      //mp_t exp_beta = boost::math::beta(a, b, x);
      std::pair<mp_t, mp_t> ib_full = ibeta_fraction1(mp_t(a), mp_t(b), mp_t(x));
      /*exp_beta = boost::math::tools::relative_error(ib_full.first, exp_beta);
      if(exp_beta > 1e-40)
      {
         std::cout << exp_beta << std::endl;
      }*/
      std::pair<mp_t, mp_t> ib_reg = ibeta_fraction1_regular(mp_t(a), mp_t(b), mp_t(x));
      return boost::math::make_tuple(a, b, x, ib_full.first, ib_full.second, ib_reg.first, ib_reg.second);
   }
};

struct beta_data_generator_small
{
   boost::math::tuple<mp_t, mp_t, mp_t, mp_t, mp_t, mp_t, mp_t> operator()(mp_t x_)
   {
      float a = truncate_to_float(gen2()/10);      
      float b = truncate_to_float(gen2()/10); 
      float x = truncate_to_float(real_cast<float>(x_));
      std::cout << a << " " << b << " " << x << std::endl;
      std::pair<mp_t, mp_t> ib_full = ibeta_fraction1(mp_t(a), mp_t(b), mp_t(x));
      std::pair<mp_t, mp_t> ib_reg = ibeta_fraction1_regular(mp_t(a), mp_t(b), mp_t(x));
      return boost::math::make_tuple(a, b, x, ib_full.first, ib_full.second, ib_reg.first, ib_reg.second);
   }
};

struct beta_data_generator_int
{
   boost::math::tuple<mp_t, mp_t, mp_t, mp_t, mp_t, mp_t, mp_t> operator()(mp_t a, mp_t b, mp_t x_)
   {
      float x = truncate_to_float(real_cast<float>(x_));
      std::cout << a << " " << b << " " << x << std::endl;
      std::pair<mp_t, mp_t> ib_full = ibeta_fraction1(a, b, mp_t(x));
      std::pair<mp_t, mp_t> ib_reg = ibeta_fraction1_regular(a, b, mp_t(x));
      return boost::math::make_tuple(a, b, x, ib_full.first, ib_full.second, ib_reg.first, ib_reg.second);
   }
};

template <class T>
bool is_01(const T& x)
{
   return (x >= 0) && (x <= 1);
}

template <class T>
bool is_interesting(const T& x)
{
   return (x > 0) && (x < 1);
}

struct beta_data_generator_asym
{
   boost::math::tuple<mp_t, mp_t, mp_t, mp_t, mp_t, mp_t, mp_t> operator()(mp_t a_, mp_t b_)
   {
      float a = truncate_to_float(a_);
      float b = truncate_to_float(b_);
      float x0 = a / (a + b);
      static const float mul[] = { 0.9, 0.99, 0.9999, 0.99999, 0.999999, 0.99999999, 1.001, 1.1, 1.00001, 1.0000001, 1.000000001 };
      static int index = 0;
      float x = truncate_to_float(real_cast<float>(x0 * mul[index]));
      if (x >= 1)
         throw std::domain_error("");
      index = index == 10 ? 0 : ++index;
      std::pair<mp_t, mp_t> ib_full = ibeta_fraction1(mp_t(a), mp_t(b), mp_t(x));
      std::pair<mp_t, mp_t> ib_reg = ibeta_fraction1_regular(mp_t(a), mp_t(b), mp_t(x));
      if (!isnormal(ib_full.first) || !isnormal(ib_reg.first) || !isnormal(ib_reg.second) || !isnormal(ib_full.second) || !is_01(ib_reg.first) || !is_01(ib_reg.second))
         throw std::domain_error("");
      std::cout << a << " " << b << " " << x << " " << ib_full.first << " " << ib_reg.first << " " << ib_reg.second << std::endl;
      return boost::math::make_tuple(a, b, x, ib_full.first, ib_full.second, ib_reg.first, ib_reg.second);
   }
};

struct beta_data_generator_asym2
{
   boost::math::tuple<mp_t, mp_t, mp_t, mp_t, mp_t, mp_t, mp_t> operator()(mp_t a_power, mp_t _b_mul, mp_t _x_mul)
   {
      float b_mul = truncate_to_float(_b_mul);
      float x_mul = truncate_to_float(_x_mul);
      float a = truncate_to_float(pow(mp_t(10), a_power));
      float b = truncate_to_float(a * b_mul);
      float x0 = a / (a + b);

      std::cout << a << " " << b_mul << " " << x_mul << std::endl;

      float x = truncate_to_float(x0 * x_mul);
      if (x >= 1)
         throw std::domain_error("");
      if(std::isinf(a))
         throw std::domain_error("");

      std::pair<mp_t, mp_t> ib_full = ibeta_fraction1(mp_t(a), mp_t(b), mp_t(x));
      std::pair<mp_t, mp_t> ib_reg = ibeta_fraction1_regular(mp_t(a), mp_t(b), mp_t(x));
      if (isnan(ib_full.first) || !isnormal(ib_reg.first) || !isnormal(ib_reg.second) || isnan(ib_full.second) || !is_01(ib_reg.first) || !is_01(ib_reg.second) || !(is_interesting(ib_reg.first) || is_interesting(ib_reg.second)))
         throw std::domain_error("");
      std::cout << a << " " << b << " " << x << " " << ib_full.first << " " << ib_reg.first << " " << ib_reg.second << std::endl;
      return boost::math::make_tuple(a, b, x, ib_full.first, ib_full.second, ib_reg.first, ib_reg.second);
   }
};

int main(int, char* [])
{
   parameter_info<mp_t> arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12;
   test_data<mp_t> data;

   std::cout << "Welcome.\n"
      "This program will generate spot tests for the incomplete beta functions:\n"
      "  beta(a, b, x) and ibeta(a, b, x)\n\n"
      "This is not an interactive program be prepared for a long wait!!!\n\n";

   arg1 = make_periodic_param(mp_t(-5), mp_t(6), 11);
   arg2 = make_periodic_param(mp_t(-5), mp_t(6), 11);
   arg3 = make_random_param(mp_t(0.0001), mp_t(1), 10);
   arg4 = make_random_param(mp_t(0.0001), mp_t(1), 100 /*500*/);
   arg5 = make_periodic_param(mp_t(1), mp_t(41), 10);
   arg6 = make_power_param(mp_t(0), 20, 40);
   arg7 = make_power_param(mp_t(0), 20, 40);
   arg8 = make_power_param(mp_t(0), 50, 70);
   arg9 = make_power_param(mp_t(0), 50, 70);
   arg10 = make_periodic_param(mp_t(10), mp_t(18), 4);
   arg11 = make_periodic_param(mp_t(1) - 10 * mp_t(1)/2048, mp_t(1) + 10 * mp_t(1) / 2048, 20);
   arg12 = make_periodic_param(mp_t(1) - 10 * mp_t(1)/2048, mp_t(1) + 10 * mp_t(1) / 2048, 20);

   arg1.type |= dummy_param;
   arg2.type |= dummy_param;
   arg3.type |= dummy_param;
   arg4.type |= dummy_param;
   arg5.type |= dummy_param;
   arg6.type |= dummy_param;
   arg7.type |= dummy_param;
   arg8.type |= dummy_param;
   arg9.type |= dummy_param;
   arg10.type |= dummy_param;
   arg11.type |= dummy_param;
   arg12.type |= dummy_param;

   // comment out all but one of the following when running
   // or this program will take forever to complete!
   //data.insert(beta_data_generator(), arg1, arg2, arg3);
   //data.insert(beta_data_generator_medium(), arg4);
   //data.insert(beta_data_generator_small(), arg4);
   //data.insert(beta_data_generator_int(), arg5, arg5, arg3);
   //data.insert(beta_data_generator_asym(), arg6, arg7);
   //data.insert(beta_data_generator_asym(), arg8, arg9);
   data.insert(beta_data_generator_asym2(), arg10, arg11, arg12);

   test_data<mp_t>::const_iterator i, j;
   i = data.begin();
   j = data.end();
   while(i != j)
   {
      mp_t v1 = beta((*i)[0], (*i)[1], (*i)[2]);
      mp_t v2 = relative_error(v1, (*i)[3]);
      std::string s = boost::lexical_cast<std::string>((*i)[3]);
      mp_t v3 = boost::lexical_cast<mp_t>(s);
      mp_t v4 = relative_error(v3, (*i)[3]);
      if(v2 > 1e-40)
      {
         std::cout << v2 << std::endl;
      }
      if(v4 > 1e-60)
      {
         std::cout << v4 << std::endl;
      }
      ++ i;
   }

   std::cout << "Enter name of test data file [default=ibeta_data.ipp]";
   std::string line;
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if(line == "")
      line = "ibeta_data.ipp";
   std::ofstream ofs(line.c_str());
   ofs << std::scientific << std::setprecision(40);
   write_code(ofs, data, "ibeta_data");
   
   return 0;
}


