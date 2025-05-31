//  (C) Copyright John Maddock 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_MAX_SERIES_ITERATION_POLICY 10000000
#define BOOST_MATH_USE_MPFR

#include "mp_t.hpp"
#include <boost/math/special_functions/hypergeometric_1F1.hpp>
#include <boost/math/distributions/non_central_t.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <map>
#include <boost/math/tools/test_data.hpp>
#include <boost/random.hpp>

using namespace boost::math::tools;
using namespace boost::math;
using namespace std;

struct nc_t_pdf_gen
{
   mp_t operator()(mp_t v, mp_t mu, mp_t x)
   {
      //
      // Arbitrary smallest value we accept, below this, we will likely underflow to zero
      // when computing at double precision:
      //
      static const mp_t minimum = 1e-270;

      mp_t Av = boost::math::hypergeometric_1F1((v + 1) / 2, boost::math::constants::half<mp_t>(), mu * mu * x * x / (2 * (x * x + v)));
      mp_t Bv = boost::math::hypergeometric_1F1(v / 2 + mp_t(1), mp_t(3) / 2, mu * mu * x * x / (2 * (x * x + v)));
      Bv *= boost::math::tgamma_ratio(v / 2 + mp_t(1), (v + mp_t(1)) / 2);
      Bv *= boost::math::constants::root_two<mp_t>() * mu * x / sqrt(x * x + v);

      Av += Bv;
      Av *= exp(-mu * mu / 2) * pow(1 + x * x / v, -(v + 1) / 2) * boost::math::tgamma_ratio((v + mp_t(1)) / 2, v / 2);
      Av /= sqrt(v) * boost::math::constants::root_pi<mp_t>();

      if (Av < minimum)
         throw std::domain_error("Result too small!");

      return Av;
   }
};


int main(int, char* [])
{
   parameter_info<mp_t> arg1, arg2, arg3;
   test_data<mp_t> data;

   std::cout << "Welcome.\n"
      "This program will generate spot tests for the non central t PDF:\n";

   std::string line;
   bool cont;

   random_ns::mt19937 rnd;
   random_ns::uniform_real_distribution<float> ur_a(0, 20);


   do
   {
      if (0 == get_user_parameter_info(arg1, "v"))
         return 1;
      if (0 == get_user_parameter_info(arg2, "nc"))
         return 1;

      arg3 = make_periodic_param(mp_t(-20), mp_t(200), 170);

      data.insert(nc_t_pdf_gen(), arg1, arg2, arg3);
      std::cout << "Any more data [y/n]?";
      std::getline(std::cin, line);
      boost::algorithm::trim(line);
      cont = (line == "y");
   } while (cont);

   std::cout << "Enter name of test data file [default=nc_t_pdf_data.ipp]";
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if (line == "")
      line = "nc_t_pdf_data.ipp";
   std::ofstream ofs(line.c_str());
   ofs << std::scientific << std::setprecision(40);
   write_code(ofs, data, line.c_str());

   return 0;
}


