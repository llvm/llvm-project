// Copyright 2017 John Maddock

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/relative_difference.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/random.hpp>
#include <boost/svg_plot/svg_2d_plot.hpp>
#include <sstream>

#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
#endif

typedef boost::multiprecision::number<boost::multiprecision::cpp_bin_float<256, boost::multiprecision::backends::digit_base_2>, boost::multiprecision::et_on> mp_type;

template <class T>
void test_type(const char* name)
{
   typedef boost::math::policies::policy<
      boost::math::policies::promote_double<false>, 
      boost::math::policies::promote_float<false>, 
      boost::math::policies::overflow_error<boost::math::policies::ignore_error> 
   > policy_type;
   boost::random::mt19937 dist;
   boost::random::uniform_real_distribution<T> ur(0, 7.75);
   boost::random::uniform_real_distribution<T> ur2(0, 1 / 7.75);

   float max_err = 0;

   std::map<double, double> small, medium, large;

   for(unsigned i = 0; i < 1000; ++i)
   {
      T input = ur(dist);
      mp_type input2(input);
      T result = boost::math::cyl_bessel_i(1, input, policy_type());
      mp_type result2 = boost::math::cyl_bessel_i(1, input2);
      mp_type err = boost::math::relative_difference(result2, mp_type(result)) / mp_type(std::numeric_limits<T>::epsilon());
      if(result2 < mp_type(result))
         err = -err;
      if(fabs(err) > max_err)
      {
         /*
         std::cout << std::setprecision(34) << input << std::endl;
         std::cout << std::setprecision(34) << input2 << std::endl;
         std::cout << std::setprecision(34) << result << std::endl;
         std::cout << std::setprecision(34) << result2 << std::endl;
         std::cout << "New max error at x = " << input << " expected " << result2 << " got " << result << " error " << err << std::endl;
         */
         max_err = static_cast<float>(fabs(err));
      }
      if(fabs(err) <= 1)
         small[static_cast<double>(input)] = static_cast<double>(err);
      else if(fabs(err) <= 2)
         medium[static_cast<double>(input)] = static_cast<double>(err);
      else
         large[static_cast<double>(input)] = static_cast<double>(err);
   }

   int y_interval = static_cast<int>(ceil(max_err / 5));

   std::stringstream ss;
   ss << "cyl_bessel_i&lt;" << name << "&gt;(1, x) over [0, 7.75]\n(max error = " << std::setprecision(2) << max_err << ")" << std::endl;

   boost::svg::svg_2d_plot my_plot;
   // Size of SVG image and X and Y range settings.
   my_plot.x_range(0, 7.75).image_x_size(700).legend_border_color(boost::svg::lightgray).plot_border_color(boost::svg::lightgray).background_border_color(boost::svg::lightgray)
      .y_range(-(int)ceil(max_err), (int)ceil(max_err)).x_label("x").title(ss.str()).y_major_interval(y_interval).x_major_interval(1.0).legend_on(true).plot_window_on(true);
   my_plot.plot(small, "&lt; 1eps").stroke_color(boost::svg::green).fill_color(boost::svg::green).size(2);
   if(max_err > 1)
   {
      my_plot.plot(medium, "&lt; 2eps").stroke_color(boost::svg::orange).fill_color(boost::svg::orange).size(2);
      if(max_err > 2)
         my_plot.plot(large, "&gt; 2eps").stroke_color(boost::svg::red).fill_color(boost::svg::red).size(2);
   }
   std::string filename("bessel_i1_0_7_");
   filename += name;
   filename += ".svg";
   my_plot.write(filename);
   std::cout << "Maximum error for type " << name << " was: " << max_err << std::endl;

   max_err = 0;
   for(unsigned i = 0; i < 1000; ++i)
   {
      T input = 1 / ur2(dist);
      mp_type input2(input);
      T result = boost::math::cyl_bessel_i(1, input, policy_type());
      mp_type result2 = boost::math::cyl_bessel_i(1, input2);
      mp_type err = boost::math::relative_difference(result2, mp_type(result)) / mp_type(std::numeric_limits<T>::epsilon());
      if(boost::math::isinf(result))
      {
         if(result2 > mp_type((std::numeric_limits<T>::max)()))
            err = 0;
         else
            std::cout << "Bad result at x = " << input << " result = " << result << " true result = " << result2 << std::endl;
      }
      if(result2 < mp_type(result))
         err = -err;
      if(fabs(err) > max_err)
      {
         max_err = static_cast<float>(fabs(err));
      }
      if(fabs(err) <= 1)
         small[1 / static_cast<double>(input)] = static_cast<double>(err);
      else if(fabs(err) <= 2)
         medium[1 / static_cast<double>(input)] = static_cast<double>(err);
      else
         large[1 / static_cast<double>(input)] = static_cast<double>(err);
   }

   y_interval = static_cast<int>(ceil(max_err / 5));
   ss.str("");
   ss << "cyl_bessel_i&lt;" << name << "&gt;(1, x) over [0, 7.75]\n(max error = " << std::setprecision(2) << max_err << ")" << std::endl;
   boost::svg::svg_2d_plot my_plot2;
   // Size of SVG image and X and Y range settings.
   my_plot2.x_range(0, 1 / 7.75).image_x_size(700).legend_border_color(boost::svg::lightgray).plot_border_color(boost::svg::lightgray).background_border_color(boost::svg::lightgray)
      .y_range(-(int)ceil(max_err), (int)ceil(max_err)).x_label("1 / x").title(ss.str()).y_major_interval(y_interval).x_major_interval(0.01).legend_on(true).plot_window_on(true);
   my_plot2.plot(small, "&lt; 1eps").stroke_color(boost::svg::green).fill_color(boost::svg::green).size(2);
   if(max_err > 1)
   {
      my_plot2.plot(medium, "&lt; 2eps").stroke_color(boost::svg::orange).fill_color(boost::svg::orange).size(2);
      if(max_err > 2)
         my_plot2.plot(large, "&gt; 2eps").stroke_color(boost::svg::red).fill_color(boost::svg::red).size(2);
   }
   filename = "bessel_i1_7_inf_";
   filename += name;
   filename += ".svg";
   my_plot2.write(filename);

   std::cout << "Maximum error for type " << name << " was: " << max_err << std::endl;
}


int main()
{
   test_type<float>("float");
   test_type<double>("double");
#if LDBL_MANT_DIG == 64
   test_type<long double>("long double");
#else
   test_type<boost::multiprecision::cpp_bin_float_double_extended>("double-extended");
#endif
#ifdef BOOST_HAS_FLOAT128
   test_type<boost::multiprecision::float128>("float128");
#else
   test_type<boost::multiprecision::cpp_bin_float_quad>("quad");
#endif

    return 0;
}

