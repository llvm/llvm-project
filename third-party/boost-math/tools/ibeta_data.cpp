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
T get_ibeta_fraction1(T a, T b, T x)
{
   ibeta_fraction1_t<T> f(a, b, x);
   T fract = boost::math::tools::continued_fraction_a(f, boost::math::policies::digits<T, boost::math::policies::policy<> >());
   T denom = (a * (fract + 1));
   T num = pow(x, a) * pow(1 - x, b);
   if(num == 0)
      return 0;
   else if(denom == 0)
      return -1;
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
      T fract = get_ibeta_fraction1(b, a, 1-x);
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
      fract = get_ibeta_fraction1(b, a, 1-x);
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
   if(x > ((a+1)/(a+b+2)))
   {
      T fract = get_ibeta_fraction1(b, a, 1-x);
      if(fract == 0)
         bet = 1;  // normalise so we don't get 0/0
      else if(bet == 0)
         return std::make_pair(T(-1), T(-1)); // Yikes!!
      if(fract / bet > 0.75)
      {
         fract = get_ibeta_fraction1(a, b, x);
         return std::make_pair(fract / bet, 1 - (fract / bet));
      }
      return std::make_pair(1 - (fract / bet), fract / bet);
   }
   T fract = get_ibeta_fraction1(a, b, x);
   if(fract / bet > 0.75)
   {
      fract = get_ibeta_fraction1(b, a, 1-x);
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

int main(int, char* [])
{
   parameter_info<mp_t> arg1, arg2, arg3, arg4, arg5;
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

   arg1.type |= dummy_param;
   arg2.type |= dummy_param;
   arg3.type |= dummy_param;
   arg4.type |= dummy_param;
   arg5.type |= dummy_param;

   // comment out all but one of the following when running
   // or this program will take forever to complete!
   //data.insert(beta_data_generator(), arg1, arg2, arg3);
   //data.insert(beta_data_generator_medium(), arg4);
   //data.insert(beta_data_generator_small(), arg4);
   data.insert(beta_data_generator_int(), arg5, arg5, arg3);

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


