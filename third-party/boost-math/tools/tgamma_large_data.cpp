// Copyright John Maddock 2013.
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//[special_data_example

#include <boost/multiprecision/mpfr.hpp>
#include <boost/math/tools/test_data.hpp>
#include <boost/test/included/prg_exec_monitor.hpp>
#include <boost/math/tools/tuple.hpp>
#include <fstream>

using namespace boost::math::tools;
using namespace boost::math;
using namespace std;
using namespace boost::multiprecision;

typedef number<mpfr_float_backend<1000> > mp_type;


boost::math::tuple<mp_type, mp_type, mp_type> generate(mp_type a)
{
   mp_type tg, lg;
   mpfr_gamma(tg.backend().data(), a.backend().data(), GMP_RNDN);
   mpfr_lngamma(lg.backend().data(), a.backend().data(), GMP_RNDN);
   return boost::math::make_tuple(a, tg, lg);
}

int cpp_main(int argc, char*argv [])
{
   parameter_info<mp_type> arg1, arg2;
   test_data<mp_type> data;

   bool cont;
   std::string line;

   if(argc < 1)
      return 1;

   do{
      //
      // User interface which prompts for 
      // range of input parameters:
      //
      if(0 == get_user_parameter_info(arg1, "a"))
         return 1;
      arg1.type |= dummy_param;

      typedef boost::math::tuple<mp_type, mp_type, mp_type>(*proc_type)(mp_type);

      proc_type p = &generate;

      data.insert(p, arg1);

      std::cout << "Any more data [y/n]?";
      std::getline(std::cin, line);
      boost::algorithm::trim(line);
      cont = (line == "y");
   }while(cont);
   //
   // Just need to write the results to a file:
   //
   std::cout << "Enter name of test data file [default=gamma.ipp]";
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if(line == "")
      line = "gamma.ipp";
   std::ofstream ofs(line.c_str());
   line.erase(line.find('.'));
   ofs << std::scientific << std::setprecision(500);
   write_code(ofs, data, line.c_str());

   return 0;
}

//]

