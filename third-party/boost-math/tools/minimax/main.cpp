//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE foobar
#define BOOST_UBLAS_TYPE_CHECK_EPSILON (type_traits<real_type>::type_sqrt (boost::math::tools::epsilon <real_type>()))
#define BOOST_UBLAS_TYPE_CHECK_MIN (type_traits<real_type>::type_sqrt ( boost::math::tools::min_value<real_type>()))
#define BOOST_UBLAS_NDEBUG

#include "multiprecision.hpp"

#include <boost/math/tools/remez.hpp>
#include <boost/math/tools/test.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/spirit/include/classic_core.hpp>
#include <boost/spirit/include/classic_actor.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <iomanip>
#include <string>
#include <boost/test/included/unit_test.hpp> // for test_main
#include <boost/multiprecision/cpp_bin_float.hpp>


extern mp_type f(const mp_type& x, int variant);
extern void show_extra(
   const boost::math::tools::polynomial<mp_type>& n, 
   const boost::math::tools::polynomial<mp_type>& d, 
   const mp_type& x_offset, 
   const mp_type& y_offset, 
   int variant);

using namespace boost::spirit::classic;

mp_type a(0), b(1);   // range to optimise over
bool rel_error(true);
bool pin(false);
int orderN(3);
int orderD(1);
int target_precision = boost::math::tools::digits<long double>();
int working_precision = target_precision * 2;
bool started(false);
int variant(0);
int skew(0);
int brake(50);
mp_type x_offset(0), y_offset(0), x_scale(1);
bool auto_offset_y;

boost::shared_ptr<boost::math::tools::remez_minimax<mp_type> > p_remez;

mp_type the_function(const mp_type& val)
{
   return f(x_scale * (val + x_offset), variant) + y_offset;
}

void step_some(unsigned count)
{
   try{
      set_working_precision(working_precision);
      if(!started)
      {
         //
         // If we have an automatic y-offset calculate it now:
         //
         if(auto_offset_y)
         {
            mp_type fa, fb, fm;
            fa = f(x_scale * (a + x_offset), variant);
            fb = f(x_scale * (b + x_offset), variant);
            fm = f(x_scale * ((a+b)/2 + x_offset), variant);
            y_offset = -(fa + fb + fm) / 3;
            set_output_precision(5);
            std::cout << "Setting auto-y-offset to " << y_offset << std::endl;
         }
         //
         // Truncate offsets to float precision:
         //
         x_offset = round_to_precision(x_offset, 20);
         y_offset = round_to_precision(y_offset, 20);
         //
         // Construct new Remez state machine:
         //
         p_remez.reset(new boost::math::tools::remez_minimax<mp_type>(
            &the_function, 
            orderN, orderD, 
            a, b, 
            pin, 
            rel_error, 
            skew, 
            working_precision));
         std::cout << "Max error in interpolated form: " << std::setprecision(3) << std::scientific << boost::math::tools::real_cast<double>(p_remez->max_error()) << std::endl;
         //
         // Signal that we've started:
         //
         started = true;
      }
      unsigned i;
      for(i = 0; i < count; ++i)
      {
         std::cout << "Stepping..." << std::endl;
         p_remez->set_brake(brake);
         mp_type r = p_remez->iterate();
         set_output_precision(3);
         std::cout 
            << "Maximum Deviation Found:                     " << std::setprecision(3) << std::scientific << boost::math::tools::real_cast<double>(p_remez->max_error()) << std::endl
            << "Expected Error Term:                         " << std::setprecision(3) << std::scientific << boost::math::tools::real_cast<double>(p_remez->error_term()) << std::endl
            << "Maximum Relative Change in Control Points:   " << std::setprecision(3) << std::scientific << boost::math::tools::real_cast<double>(r) << std::endl;
      }
   }
   catch(const std::exception& e)
   {
      std::cout << "Step failed with exception: " << e.what() << std::endl;
   }
}

void step(const char*, const char*)
{
   step_some(1);
}

void show(const char*, const char*)
{
   set_working_precision(working_precision);
   if(started)
   {
      boost::math::tools::polynomial<mp_type> n = p_remez->numerator();
      boost::math::tools::polynomial<mp_type> d = p_remez->denominator();
      std::vector<mp_type> cn = n.chebyshev();
      std::vector<mp_type> cd = d.chebyshev();
      int prec = 2 + (target_precision * 3010LL)/10000;
      std::cout << std::scientific << std::setprecision(prec);
      set_output_precision(prec);
      boost::numeric::ublas::vector<mp_type> v = p_remez->zero_points();
      
      std::cout << "  Zeros = {\n";
      unsigned i;
      for(i = 0; i < v.size(); ++i)
      {
         std::cout << "    " << v[i] << std::endl;
      }
      std::cout << "  }\n";

      v = p_remez->chebyshev_points();
      std::cout << "  Chebeshev Control Points = {\n";
      for(i = 0; i < v.size(); ++i)
      {
         std::cout << "    " << v[i] << std::endl;
      }
      std::cout << "  }\n";

      std::cout << "X offset: " << x_offset << std::endl;
      std::cout << "X scale:  " << x_scale << std::endl;
      std::cout << "Y offset: " << y_offset << std::endl;

      std::cout << "P = {";
      for(i = 0; i < n.size(); ++i)
      {
         std::cout << "    " << n[i] << "L," << std::endl;
      }
      std::cout << "  }\n";

      std::cout << "Q = {";
      for(i = 0; i < d.size(); ++i)
      {
         std::cout << "    " << d[i] << "L," << std::endl;
      }
      std::cout << "  }\n";

      std::cout << "CP = {";
      for(i = 0; i < cn.size(); ++i)
      {
         std::cout << "    " << cn[i] << "L," << std::endl;
      }
      std::cout << "  }\n";

      std::cout << "CQ = {";
      for(i = 0; i < cd.size(); ++i)
      {
         std::cout << "    " << cd[i] << "L," << std::endl;
      }
      std::cout << "  }\n";

      show_extra(n, d, x_offset, y_offset, variant);
   }
   else
   {
      std::cerr << "Nothing to display" << std::endl;
   }
}

void do_graph(unsigned points)
{
   set_working_precision(working_precision);
   mp_type step = (b - a) / (points - 1);
   mp_type x = a;
   while(points > 1)
   {
      set_output_precision(10);
      std::cout << std::setprecision(10) << std::setw(30) << std::left 
         << boost::lexical_cast<std::string>(x) << the_function(x) << std::endl;
      --points;
      x += step;
   }
   std::cout << std::setprecision(10) << std::setw(30) << std::left 
      << boost::lexical_cast<std::string>(b) << the_function(b) << std::endl;
}

void graph(const char*, const char*)
{
   do_graph(3);
}

template <class T>
mp_type convert_to_rr(const T& val)
{
   return val;
}
template <class Backend, boost::multiprecision::expression_template_option ET>
mp_type convert_to_rr(const boost::multiprecision::number<Backend, ET>& val)
{
   return boost::lexical_cast<mp_type>(val.str());
}

template <class T>
void do_test(T, const char* name)
{
   set_working_precision(working_precision);
   if(started)
   {
      //
      // We want to test the approximation at fixed precision:
      // either float, double or long double.  Begin by getting the
      // polynomials:
      //
      boost::math::tools::polynomial<T> n, d;
      boost::math::tools::polynomial<mp_type> nr, dr;
      nr = p_remez->numerator();
      dr = p_remez->denominator();
      n = nr;
      d = dr;

      std::vector<mp_type> cn1, cd1;
      cn1 = nr.chebyshev();
      cd1 = dr.chebyshev();
      std::vector<T> cn, cd;
      for(unsigned i = 0; i < cn1.size(); ++i)
      {
         cn.push_back(boost::math::tools::real_cast<T>(cn1[i]));
      }
      for(unsigned i = 0; i < cd1.size(); ++i)
      {
         cd.push_back(boost::math::tools::real_cast<T>(cd1[i]));
      }
      //
      // We'll test at the Chebeshev control points which is where
      // (in theory) the largest deviation should occur.  For good
      // measure we'll test at the zeros as well:
      //
      boost::numeric::ublas::vector<mp_type> 
         zeros(p_remez->zero_points()),
         cheb(p_remez->chebyshev_points());

      mp_type max_error(0), cheb_max_error(0);

      //
      // Do the tests at the zeros:
      //
      std::cout << "Starting tests at " << name << " precision...\n";
      std::cout << "Abscissa        Error (Poly)   Error (Cheb)\n";
      for(unsigned i = 0; i < zeros.size(); ++i)
      {
         mp_type true_result = the_function(zeros[i]);
         T abscissa = boost::math::tools::real_cast<T>(zeros[i]);
         mp_type test_result = convert_to_rr(n.evaluate(abscissa) / d.evaluate(abscissa));
         mp_type cheb_result = convert_to_rr(boost::math::tools::evaluate_chebyshev(cn, abscissa) / boost::math::tools::evaluate_chebyshev(cd, abscissa));
         mp_type err, cheb_err;
         if(rel_error)
         {
            err = boost::math::tools::relative_error(test_result, true_result);
            cheb_err = boost::math::tools::relative_error(cheb_result, true_result);
         }
         else
         {
            err = fabs(test_result - true_result);
            cheb_err = fabs(cheb_result - true_result);
         }
         if(err > max_error)
            max_error = err;
         if(cheb_err > cheb_max_error)
            cheb_max_error = cheb_err;
         std::cout << std::setprecision(6) << std::setw(15) << std::left << abscissa
            << std::setw(15) << std::left << boost::math::tools::real_cast<T>(err) << boost::math::tools::real_cast<T>(cheb_err) << std::endl;
      }
      //
      // Do the tests at the Chebeshev control points:
      //
      for(unsigned i = 0; i < cheb.size(); ++i)
      {
         mp_type true_result = the_function(cheb[i]);
         T abscissa = boost::math::tools::real_cast<T>(cheb[i]);
         mp_type test_result = convert_to_rr(n.evaluate(abscissa) / d.evaluate(abscissa));
         mp_type cheb_result = convert_to_rr(boost::math::tools::evaluate_chebyshev(cn, abscissa) / boost::math::tools::evaluate_chebyshev(cd, abscissa));
         mp_type err, cheb_err;
         if(rel_error)
         {
            err = boost::math::tools::relative_error(test_result, true_result);
            cheb_err = boost::math::tools::relative_error(cheb_result, true_result);
         }
         else
         {
            err = fabs(test_result - true_result);
            cheb_err = fabs(cheb_result - true_result);
         }
         if(err > max_error)
            max_error = err;
         std::cout << std::setprecision(6) << std::setw(15) << std::left << abscissa
            << std::setw(15) << std::left << boost::math::tools::real_cast<T>(err) << 
            boost::math::tools::real_cast<T>(cheb_err) << std::endl;
      }
      std::string msg = "Max Error found at ";
      msg += name;
      msg += " precision = ";
      msg.append(62 - 17 - msg.size(), ' ');
      std::cout << msg << std::setprecision(6) << "Poly: " << std::setw(20) << std::left
         << boost::math::tools::real_cast<T>(max_error) << "Cheb: " << boost::math::tools::real_cast<T>(cheb_max_error) << std::endl;
   }
   else
   {
      std::cout << "Nothing to test: try converging an approximation first!!!" << std::endl;
   }
}

void test_float(const char*, const char*)
{
   do_test(float(0), "float");
}

void test_double(const char*, const char*)
{
   do_test(double(0), "double");
}

void test_long(const char*, const char*)
{
   do_test((long double)(0), "long double");
}

void test_float80(const char*, const char*)
{
   do_test((boost::multiprecision::cpp_bin_float_double_extended)(0), "float80");
}

void test_float128(const char*, const char*)
{
   do_test((boost::multiprecision::cpp_bin_float_quad)(0), "float128");
}

void test_all(const char*, const char*)
{
   do_test(float(0), "float");
   do_test(double(0), "double");
   do_test((long double)(0), "long double");
}

template <class T>
void do_test_n(T, const char* name, unsigned count)
{
   set_working_precision(working_precision);
   if(started)
   {
      //
      // We want to test the approximation at fixed precision:
      // either float, double or long double.  Begin by getting the
      // polynomials:
      //
      boost::math::tools::polynomial<T> n, d;
      boost::math::tools::polynomial<mp_type> nr, dr;
      nr = p_remez->numerator();
      dr = p_remez->denominator();
      n = nr;
      d = dr;

      std::vector<mp_type> cn1, cd1;
      cn1 = nr.chebyshev();
      cd1 = dr.chebyshev();
      std::vector<T> cn, cd;
      for(unsigned i = 0; i < cn1.size(); ++i)
      {
         cn.push_back(boost::math::tools::real_cast<T>(cn1[i]));
      }
      for(unsigned i = 0; i < cd1.size(); ++i)
      {
         cd.push_back(boost::math::tools::real_cast<T>(cd1[i]));
      }

      mp_type max_error(0), max_cheb_error(0);
      mp_type step = (b - a) / count;

      //
      // Do the tests at the zeros:
      //
      std::cout << "Starting tests at " << name << " precision...\n";
      std::cout << "Abscissa        Error (poly)   Error (Cheb)\n";
      for(mp_type x = a; x <= b; x += step)
      {
         mp_type true_result = the_function(x);
         //std::cout << true_result << std::endl;
         T abscissa = boost::math::tools::real_cast<T>(x);
         mp_type test_result = convert_to_rr(n.evaluate(abscissa) / d.evaluate(abscissa));
         //std::cout << test_result << std::endl;
         mp_type cheb_result = convert_to_rr(boost::math::tools::evaluate_chebyshev(cn, abscissa) / boost::math::tools::evaluate_chebyshev(cd, abscissa));
         //std::cout << cheb_result << std::endl;
         mp_type err, cheb_err;
         if(rel_error)
         {
            err = boost::math::tools::relative_error(test_result, true_result);
            cheb_err = boost::math::tools::relative_error(cheb_result, true_result);
         }
         else
         {
            err = fabs(test_result - true_result);
            cheb_err = fabs(cheb_result - true_result);
         }
         if(err > max_error)
            max_error = err;
         if(cheb_err > max_cheb_error)
            max_cheb_error = cheb_err;
         std::cout << std::setprecision(6) << std::setw(15) << std::left << boost::math::tools::real_cast<double>(abscissa)
            << (test_result < true_result ? "-" : "") << std::setw(20) << std::left 
            << boost::math::tools::real_cast<double>(err) 
            << boost::math::tools::real_cast<double>(cheb_err) << std::endl;
      }
      std::string msg = "Max Error found at ";
      msg += name;
      msg += " precision = ";
      //msg.append(62 - 17 - msg.size(), ' ');
      std::cout << msg << "Poly: " << std::setprecision(6) 
         //<< std::setw(15) << std::left 
         << boost::math::tools::real_cast<T>(max_error) 
         << " Cheb: " << boost::math::tools::real_cast<T>(max_cheb_error) << std::endl;
   }
   else
   {
      std::cout << "Nothing to test: try converging an approximation first!!!" << std::endl;
   }
}

void test_n(unsigned n)
{
   do_test_n(mp_type(), "mp_type", n);
}

void test_float_n(unsigned n)
{
   do_test_n(float(0), "float", n);
}

void test_double_n(unsigned n)
{
   do_test_n(double(0), "double", n);
}

void test_long_n(unsigned n)
{
   do_test_n((long double)(0), "long double", n);
}

void test_float80_n(unsigned n)
{
   do_test_n((boost::multiprecision::cpp_bin_float_double_extended)(0), "float80", n);
}

void test_float128_n(unsigned n)
{
   do_test_n((boost::multiprecision::cpp_bin_float_quad)(0), "float128", n);
}

void rotate(const char*, const char*)
{
   if(p_remez)
   {
      p_remez->rotate();
   }
   else
   {
      std::cerr << "Nothing to rotate" << std::endl;
   }
}

void rescale(const char*, const char*)
{
   if(p_remez)
   {
      p_remez->rescale(a, b);
   }
   else
   {
      std::cerr << "Nothing to rescale" << std::endl;
   }
}

void graph_poly(const char*, const char*)
{
   int i = 50;
   set_working_precision(working_precision);
   if(started)
   {
      //
      // We want to test the approximation at fixed precision:
      // either float, double or long double.  Begin by getting the
      // polynomials:
      //
      boost::math::tools::polynomial<mp_type> n, d;
      n = p_remez->numerator();
      d = p_remez->denominator();

      mp_type max_error(0);
      mp_type step = (b - a) / i;

      std::cout << "Evaluating Numerator...\n";
      mp_type val;
      for(val = a; val <= b; val += step)
         std::cout << n.evaluate(val) << std::endl;
      std::cout << "Evaluating Denominator...\n";
      for(val = a; val <= b; val += step)
         std::cout << d.evaluate(val) << std::endl;
   }
   else
   {
      std::cout << "Nothing to test: try converging an approximation first!!!" << std::endl;
   }
}

BOOST_AUTO_TEST_CASE( test_main )
{
   std::string line;
   real_parser<long double/*mp_type*/ > const rr_p;
   while(std::getline(std::cin, line))
   {
      if(parse(line.c_str(), str_p("quit"), space_p).full)
         return;
      if(false == parse(line.c_str(), 
         (

            str_p("range")[assign_a(started, false)] && real_p[assign_a(a)] && real_p[assign_a(b)]
      ||
            str_p("relative")[assign_a(started, false)][assign_a(rel_error, true)]
      ||
            str_p("absolute")[assign_a(started, false)][assign_a(rel_error, false)]
      ||
            str_p("pin")[assign_a(started, false)] && str_p("true")[assign_a(pin, true)]
      ||
            str_p("pin")[assign_a(started, false)] && str_p("false")[assign_a(pin, false)]
      ||
            str_p("pin")[assign_a(started, false)] && str_p("1")[assign_a(pin, true)]
      ||
            str_p("pin")[assign_a(started, false)] && str_p("0")[assign_a(pin, false)]
      ||
            str_p("pin")[assign_a(started, false)][assign_a(pin, true)]
      ||
            str_p("order")[assign_a(started, false)] && uint_p[assign_a(orderN)] && uint_p[assign_a(orderD)]
      ||
            str_p("order")[assign_a(started, false)] && uint_p[assign_a(orderN)]
      ||
            str_p("target-precision") && uint_p[assign_a(target_precision)]
      ||
            str_p("working-precision")[assign_a(started, false)] && uint_p[assign_a(working_precision)]
      ||
            str_p("variant")[assign_a(started, false)] && int_p[assign_a(variant)]
      ||
            str_p("skew")[assign_a(started, false)] && int_p[assign_a(skew)]
      ||
            str_p("brake") && int_p[assign_a(brake)]
      ||
            str_p("step") && int_p[&step_some]
      ||
            str_p("step")[&step]
      ||
            str_p("poly")[&graph_poly]
      ||
            str_p("info")[&show]
      ||
            str_p("graph") && uint_p[&do_graph]
      ||
            str_p("graph")[&graph]
      ||
            str_p("x-offset") && real_p[assign_a(x_offset)]
      ||
            str_p("x-scale") && real_p[assign_a(x_scale)]
      ||
            str_p("y-offset") && str_p("auto")[assign_a(auto_offset_y, true)]
      ||
            str_p("y-offset") && real_p[assign_a(y_offset)][assign_a(auto_offset_y, false)]
      ||
            str_p("test") && str_p("float80") && uint_p[&test_float80_n]
      ||
            str_p("test") && str_p("float80")[&test_float80]
      ||
            str_p("test") && str_p("float128") && uint_p[&test_float128_n]
      ||
            str_p("test") && str_p("float128")[&test_float128]
      ||
            str_p("test") && str_p("float") && uint_p[&test_float_n]
      ||
            str_p("test") && str_p("float")[&test_float]
      ||
            str_p("test") && str_p("double") && uint_p[&test_double_n]
      ||
            str_p("test") && str_p("double")[&test_double]
      ||
            str_p("test") && str_p("long") && uint_p[&test_long_n]
      ||
            str_p("test") && str_p("long")[&test_long]
      ||
            str_p("test") && str_p("all")[&test_all]
      ||
            str_p("test") && uint_p[&test_n]
      ||
            str_p("rotate")[&rotate]
      ||
            str_p("rescale") && real_p[assign_a(a)] && real_p[assign_a(b)] && epsilon_p[&rescale]

         ), space_p).full)
      {
         std::cout << "Unable to parse directive: \"" << line << "\"" << std::endl;
      }
      else
      {
         std::cout << "Variant              = " << variant << std::endl;
         std::cout << "range                = [" << a << "," << b << "]" << std::endl;
         std::cout << "Relative Error       = " << rel_error << std::endl;
         std::cout << "Pin to Origin        = " << pin << std::endl;
         std::cout << "Order (Num/Denom)    = " << orderN << "/" << orderD << std::endl;
         std::cout << "Target Precision     = " << target_precision << std::endl;
         std::cout << "Working Precision    = " << working_precision << std::endl;
         std::cout << "Skew                 = " << skew << std::endl;
         std::cout << "Brake                = " << brake << std::endl;
         std::cout << "X Offset             = " << x_offset << std::endl;
         std::cout << "X scale              = " << x_scale << std::endl;
         std::cout << "Y Offset             = ";
         if(auto_offset_y)
            std::cout << "Auto (";
         std::cout << y_offset;
         if(auto_offset_y)
            std::cout << ")";
         std::cout << std::endl;
     }
   }
}
