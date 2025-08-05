//  Copyright John Maddock 2018.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <map>
#include <boost/config.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
#endif
#include <boost/svg_plot/svg_2d_plot.hpp>

template <class Real>
Real interval_from_range(Real x)
{
   BOOST_MATH_STD_USING
   Real l = floor(log10(x));
   l = pow(10, l);
   if (x / l < 2)
      l /= 10;
   return l;
}


std::string normalise_filename(std::string name)
{
   for(std::string::size_type i = 0; i < name.size(); ++i)
   {
      if (!std::isalnum(name[i]))
         name[i] = '_';
      else
         name[i] = std::tolower(name[i]);
   }
   return name;
}

template <class F, class Real>
void plot_errors_1d(F f, Real start, Real end, unsigned points, const char* function_name, Real max_y_scale = (std::numeric_limits<Real>::max)(), unsigned num_bins = 200)
{
   BOOST_MATH_STD_USING
   std::cout << "Generating points for " << function_name << std::endl;
   Real pos = start;
   Real interval = (end - start) / points;

   std::map<double, double> points_upper, points_lower;

   Real max_distance(0), min_distance(0), max_error(0), max_error_location(0);

   constexpr unsigned limb_bits = (sizeof(boost::multiprecision::limb_type) * CHAR_BIT);
   constexpr unsigned mp_digits = (((std::numeric_limits<Real>::digits * 2) / limb_bits + ((std::numeric_limits<Real>::digits * 2) % limb_bits ? 1 : 0))) * limb_bits;

   typedef boost::multiprecision::number<boost::multiprecision::cpp_bin_float<mp_digits, boost::multiprecision::backends::digit_base_2> > mp_type;

   while (pos <= end)
   {
      try
      {
         Real found_value = f(pos);
         Real exact_value = static_cast<Real>(f(mp_type(pos)));
         Real distance = boost::math::sign(found_value - exact_value) * boost::math::epsilon_difference(found_value, exact_value);
         Real bin = start + ((end - start) / num_bins) * boost::math::itrunc(num_bins * (pos - start) / (end - start));
         if (points_lower.find((double)bin) == points_lower.end())
            points_lower[(double)bin] = 0;
         if (points_upper.find((double)bin) == points_upper.end())
            points_upper[(double)bin] = 0;
         if (distance > 0)
         {
            if (points_upper[(double)bin] < distance)
               points_upper[(double)bin] = (double)(std::min)(distance, max_y_scale);
         }
         else
         {
            if (points_lower[(double)bin] > distance)
               points_lower[(double)bin] = (double)(std::max)(distance, -max_y_scale);
         }
         if (max_distance < distance)
            max_distance = (std::min)(distance, max_y_scale);
         if (min_distance > distance)
            min_distance = (std::max)(distance, -max_y_scale);
         if (fabs(distance) > max_error)
         {
            max_error = fabs(distance);
            max_error_location = pos;
         }
         pos += interval;
      }
      catch (const std::exception& e)
      {
         std::cout << "Found exception at point " << pos << " : " << e.what() << std::endl;
         pos += interval;
      }
   }

   std::cout << "Max error was " << std::setprecision(3) << max_error << " at location " << std::setprecision(std::numeric_limits<Real>::max_digits10) << max_error_location << std::endl;

   boost::svg::svg_2d_plot plot;
   Real x_start(start), x_end(end);
   if (end - start > 3)
   {
      x_start = floor(start);
      x_end = ceil(end);
   }
   if (min_distance == 0)
      min_distance = -1;
   if (max_distance == 0)
      max_distance = 1;


   plot.title(std::string("Errors in ") + function_name).x_range((double)x_start, (double)x_end).image_x_size(700).legend_border_color(boost::svg::lightgray).plot_border_color(boost::svg::lightgray).background_border_color(boost::svg::lightgray)
      .y_range((int)floor(min_distance), (int)ceil(max_distance)).x_label("x").y_major_interval((double)interval_from_range(max_distance) * 2).x_major_interval((double)interval_from_range(end - start)).legend_on(true).plot_window_on(true).legend_on(false);
   plot.plot(points_upper).stroke_color(boost::svg::green).fill_color(boost::svg::green).size(1).line_on(true).area_fill(boost::svg::green);
   plot.plot(points_lower).stroke_color(boost::svg::green).fill_color(boost::svg::green).size(1).line_on(true).area_fill(boost::svg::green);

   plot.write(normalise_filename(function_name) + ".svg");

}

#include <boost/math/special_functions.hpp>

struct digamma_func
{
   template <class T>
   T operator()(T x)
   {
      return boost::math::digamma(x);
   }
};

struct tgamma_func
{
   template <class T>
   T operator()(T x)
   {
      return boost::math::tgamma(x);
   }
};

struct lgamma_func
{
   template <class T>
   T operator()(T x)
   {
      return boost::math::lgamma(x);
   }
};

struct trigamma_func
{
   template <class T>
   T operator()(T x)
   {
      return boost::math::tgamma(x);
   }
};

struct erf_func
{
   template <class T>
   T operator()(T x)
   {
      return boost::math::erf(x);
   }
};

struct erfc_func
{
   template <class T>
   T operator()(T x)
   {
      return boost::math::erfc(x);
   }
};

struct j0_func
{
   template <class T>
   T operator()(T x)
   {
      return boost::math::cyl_bessel_j(0, x);
   }
};

struct j1_func
{
   template <class T>
   T operator()(T x)
   {
      return boost::math::cyl_bessel_j(1, x);
   }
};

struct y0_func
{
   template <class T>
   T operator()(T x)
   {
      return boost::math::cyl_neumann(0, x);
   }
};

struct y1_func
{
   template <class T>
   T operator()(T x)
   {
      return boost::math::cyl_neumann(1, x);
   }
};

struct i0_func
{
   template <class T>
   T operator()(T x)
   {
      return boost::math::cyl_bessel_i(0, x);
   }
};

struct i1_func
{
   template <class T>
   T operator()(T x)
   {
      return boost::math::cyl_bessel_i(1, x);
   }
};

struct k0_func
{
   template <class T>
   T operator()(T x)
   {
      return boost::math::cyl_bessel_k(0, x);
   }
};

struct k1_func
{
   template <class T>
   T operator()(T x)
   {
      return boost::math::cyl_bessel_k(1, x);
   }
};

struct ai_func
{
   template <class T>
   T operator()(T x)
   {
      return boost::math::airy_ai(x);
   }
};

struct aip_func
{
   template <class T>
   T operator()(T x)
   {
      return boost::math::airy_ai_prime(x);
   }
};

struct bi_func
{
   template <class T>
   T operator()(T x)
   {
      return boost::math::airy_bi(x);
   }
};

struct bip_func
{
   template <class T>
   T operator()(T x)
   {
      return boost::math::airy_bi_prime(x);
   }
};

struct ellint_1_func
{
   template <class T>
   T operator()(T x)
   {
      return boost::math::ellint_1(x);
   }
};

struct ellint_2_func
{
   template <class T>
   T operator()(T x)
   {
      return boost::math::ellint_2(x);
   }
};

struct ellint_d_func
{
   template <class T>
   T operator()(T x)
   {
      return boost::math::ellint_d(x);
   }
};

struct zeta_func
{
   template <class T>
   T operator()(T x)
   {
      return boost::math::zeta(x);
   }
};

struct ei_func
{
   template <class T>
   T operator()(T x)
   {
      return boost::math::expint(x);
   }
};

int main()
{
   plot_errors_1d(digamma_func(), 1e-200, 10.0, 10000, "digamma, double");
   plot_errors_1d(tgamma_func(), 1e-200, 150.0, 10000, "tgamma, double");
   plot_errors_1d(lgamma_func(), 1e-200, 1000.0, 10000, "lgamma, double");
   plot_errors_1d(trigamma_func(), 1e-200, 10.0, 10000, "trigamma, double");
   plot_errors_1d(erf_func(), -5.0, 5.0, 10000, "erf, double");
   plot_errors_1d(erfc_func(), -5.0, 30.0, 10000, "erfc, double");
   plot_errors_1d(j0_func(), 0.0, 50.0, 10000, "j0, double", 50.0);
   plot_errors_1d(j1_func(), 0.0, 50.0, 10000, "j1, double", 50.0);
   plot_errors_1d(y0_func(), 1e-100, 50.0, 10000, "y0, double", 50.0);
   plot_errors_1d(y1_func(), 1e-100, 50.0, 10000, "y1, double", 50.0);
   plot_errors_1d(i0_func(), 0.0, 50.0, 10000, "i0, double");
   plot_errors_1d(i1_func(), 0.0, 50.0, 10000, "i1, double");
   plot_errors_1d(k0_func(), 1e-100, 50.0, 10000, "k0, double");
   plot_errors_1d(k1_func(), 1e-100, 50.0, 10000, "k1, double");
   plot_errors_1d(ai_func(), -20.0, 20.0, 10000, "Ai, double", 100.0);
   plot_errors_1d(bi_func(), -20.0, 20.0, 10000, "Bi, double", 100.0);
   plot_errors_1d(aip_func(), -20.0, 20.0, 10000, "Ai Prime, double", 100.0);
   plot_errors_1d(bip_func(), -20.0, 20.0, 10000, "Bi Prime, double", 100.0);

   plot_errors_1d(ellint_1_func(), -1.0, 1.0, 10000, "Elliptic Integral K, double");
   plot_errors_1d(ellint_2_func(), -1.0, 1.0, 10000, "Elliptic Integral E, double");
   plot_errors_1d(ellint_d_func(), -1.0, 1.0, 10000, "Elliptic Integral D, double");

   plot_errors_1d(zeta_func(), -20.0, 20.0, 10000, "Zeta, double");
   plot_errors_1d(ei_func(), -20.0, 20.0, 10000, "Exponential Integral Ei, double");

#if LDBL_MANT_DIG == 64
   plot_errors_1d(digamma_func(), 1e-200L, 10.0L, 10000, "digamma, 80-bit long double");
   plot_errors_1d(tgamma_func(), 1e-200L, 150.0L, 10000, "tgamma, 80-bit long double");
   plot_errors_1d(lgamma_func(), 1e-200L, 1000.0L, 10000, "lgamma, 80-bit long double");
   plot_errors_1d(trigamma_func(), 1e-200L, 10.0L, 10000, "trigamma, 80-bit long double");
   plot_errors_1d(erf_func(), -5.0L, 5.0L, 10000, "erf, 80-bit long double");
   plot_errors_1d(erfc_func(), -5.0L, 120.0L, 10000, "erfc, 80-bit long double");
   plot_errors_1d(j0_func(), 0.0L, 50.0L, 10000, "j0, 80 bit long double", 50.0L);
   plot_errors_1d(j1_func(), 0.0L, 50.0L, 10000, "j1, 80 bit long double", 50.0L);
   plot_errors_1d(y0_func(), 1e-100L, 50.0L, 10000, "y0, 80 bit long double", 50.0L);
   plot_errors_1d(y1_func(), 1e-100L, 50.0L, 10000, "y1, 80 bit long double", 50.0L);
   plot_errors_1d(i0_func(), 0.0L, 50.0L, 10000, "i0, 80 bit long double");
   plot_errors_1d(i1_func(), 0.0L, 50.0L, 10000, "i1, 80 bit long double");
   plot_errors_1d(k0_func(), 1e-100L, 50.0L, 10000, "k0, 80 bit long double");
   plot_errors_1d(k1_func(), 1e-100L, 50.0L, 10000, "k1, 80 bit long double");
   plot_errors_1d(ai_func(), -20.0L, 20.0L, 10000, "Ai, 80 bit long double", 100.0L);
   plot_errors_1d(bi_func(), -20.0L, 20.0L, 10000, "Bi, 80 bit long double", 100.0L);
   plot_errors_1d(aip_func(), -20.0L, 20.0L, 10000, "Ai Prime, 80 bit long double", 100.0L);
   plot_errors_1d(bip_func(), -20.0L, 20.0L, 10000, "Bi Prime, 80 bit long double", 100.0L);

   plot_errors_1d(ellint_1_func(), -1.0L, 1.0L, 10000, "Elliptic Integral K, 80 bit long double");
   plot_errors_1d(ellint_2_func(), -1.0L, 1.0L, 10000, "Elliptic Integral E, 80 bit long double");
   plot_errors_1d(ellint_d_func(), -1.0L, 1.0L, 10000, "Elliptic Integral D, 80 bit long double");

   plot_errors_1d(zeta_func(), -20.0L, 20.0L, 10000, "Zeta, 80 bit long double");
   plot_errors_1d(ei_func(), -20.0L, 20.0L, 10000, "Exponential Integral Ei, 80 bit long double");
#endif
#ifdef BOOST_HAS_FLOAT128
   plot_errors_1d(digamma_func(), boost::multiprecision::float128(1e-200), boost::multiprecision::float128(10.0), 10000, "digamma, __float128");
   plot_errors_1d(tgamma_func(), boost::multiprecision::float128(1e-200), boost::multiprecision::float128(150.0), 10000, "tgamma, __float128");
   plot_errors_1d(lgamma_func(), boost::multiprecision::float128(1e-200), boost::multiprecision::float128(1000.0), 10000, "lgamma, __float128");
   plot_errors_1d(trigamma_func(), boost::multiprecision::float128(1e-200), boost::multiprecision::float128(10.0), 10000, "trigamma, __float128");
   plot_errors_1d(erf_func(), -boost::multiprecision::float128(5.0), boost::multiprecision::float128(5.0), 10000, "erf, __float128");
   plot_errors_1d(erfc_func(), -boost::multiprecision::float128(5.0), boost::multiprecision::float128(120.0), 10000, "erfc, __float128");
   plot_errors_1d(j0_func(), boost::multiprecision::float128(0.0), boost::multiprecision::float128(50.0), 10000, "j0, __float128", boost::multiprecision::float128(50.0));
   plot_errors_1d(j1_func(), boost::multiprecision::float128(0.0), boost::multiprecision::float128(50.0), 10000, "j1, __float128", boost::multiprecision::float128(50.0));
   plot_errors_1d(y0_func(), boost::multiprecision::float128(1e-100), boost::multiprecision::float128(50.0), 10000, "y0, __float128", boost::multiprecision::float128(50.0));
   plot_errors_1d(y1_func(), boost::multiprecision::float128(1e-100), boost::multiprecision::float128(50.0), 10000, "y1, __float128", boost::multiprecision::float128(50.0));
   plot_errors_1d(i0_func(), boost::multiprecision::float128(0.0), boost::multiprecision::float128(50.0), 10000, "i0, __float128");
   plot_errors_1d(i1_func(), boost::multiprecision::float128(0.0), boost::multiprecision::float128(50.0), 10000, "i1, __float128");
   plot_errors_1d(k0_func(), boost::multiprecision::float128(1e-100), boost::multiprecision::float128(50.0), 10000, "k0, __float128");
   plot_errors_1d(k1_func(), boost::multiprecision::float128(1e-100), boost::multiprecision::float128(50.0), 10000, "k1, __float128");
   plot_errors_1d(ai_func(), -boost::multiprecision::float128(20.0), boost::multiprecision::float128(20.0), 10000, "Ai, __float128", boost::multiprecision::float128(100.0));
   plot_errors_1d(bi_func(), -boost::multiprecision::float128(20.0), boost::multiprecision::float128(20.0), 10000, "Bi, __float128", boost::multiprecision::float128(100.0));
   plot_errors_1d(aip_func(), -boost::multiprecision::float128(20.0), boost::multiprecision::float128(20.0), 10000, "Ai Prime, __float128", boost::multiprecision::float128(100.0));
   plot_errors_1d(bip_func(), -boost::multiprecision::float128(20.0), boost::multiprecision::float128(20.0), 10000, "Bi Prime, __float128", boost::multiprecision::float128(100.0));

   plot_errors_1d(ellint_1_func(), -boost::multiprecision::float128(1.0), boost::multiprecision::float128(1.0), 10000, "Elliptic Integral K, __float128");
   plot_errors_1d(ellint_2_func(), -boost::multiprecision::float128(1.0), boost::multiprecision::float128(1.0), 10000, "Elliptic Integral E, __float128");
   plot_errors_1d(ellint_d_func(), -boost::multiprecision::float128(1.0), boost::multiprecision::float128(1.0), 10000, "Elliptic Integral D, __float128");

   plot_errors_1d(zeta_func(), -boost::multiprecision::float128(20.0), boost::multiprecision::float128(20.0), 10000, "Zeta, __float128");
   plot_errors_1d(ei_func(), -boost::multiprecision::float128(20.0), boost::multiprecision::float128(20.0), 10000, "Exponential Integral Ei, __float128");
#endif
   return 0;
}
