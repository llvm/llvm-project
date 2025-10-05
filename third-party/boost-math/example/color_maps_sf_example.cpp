//  (C) Copyright John Maddock 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//
// This program plots various special functions as color maps,
// run with the "help" option to see the various command line options.
//

#include <cmath>
#include <cstdint>
#include <array>
#include <complex>
#include <tuple>
#include <iostream>
#include <vector>
#include <limits>
#include <boost/math/tools/color_maps.hpp>
#include <boost/math/special_functions.hpp>

#if !__has_include("lodepng.h")
 #error "lodepng.h is required to run this example."
#endif
#include "lodepng.h"
#include <iostream>
#include <string>
#include <vector>
#include <functional>

// In lodepng, the vector is expected to be row major, with the top row
// specified first. Note that this is a bit confusing sometimes as it's more
// natural to let y increase moving *up*.
unsigned write_png(const std::string& filename,
   const std::vector<std::uint8_t>& img, std::size_t width,
   std::size_t height) {
   unsigned error = lodepng::encode(filename, img, width, height,
      LodePNGColorType::LCT_RGBA, 8);
   if (error) {
      std::cerr << "Error encoding png: " << lodepng_error_text(error) << "\n";
   }
   return error;
}

double hypergeometric_1F1_at_half(double x, double y)
{
   try 
   {
      return boost::math::hypergeometric_1F1(x, y, -3.5);
   }
   catch (const std::domain_error&)
   {
      return 0;
   }
}

void show_help()
{
   std::cout <<
      "The following command line options are supported:\n"
      "  gamma_p|gamma_q|gamma_p_inv|gamma_q_inv|cyl_bessel_j|cyl_neumann|cyl_bessel_i|cyl_bessel_k\n"
      "  |cyl_bessel_d|ellint_1|ellint_2|ellint_3|jacobi_zeta|heuman_lambda|jacobi_theta1|1F1\n"
      "     Sets the function to be plotted.\n"
      "     Note that the defaults for the options below change depending on the function selected here,\n"
      "     so set this option first, and then fine tune with the following options:\n"
      "  smooth_cool_warm|plasma|black_body|inferno|kindlmann|extended_kindlmann\n"
      "     Sets the color map used.\n"
      "  width=XX\n"
      "  height=XX\n"
      "     Sets the width and height of the bitmap.\n"
      "  x_min=XX\n"
      "  x_max=XX\n"
      "  y_min=XX\n"
      "  y_max=XX\n"
      "     Sets the extent of the x and y variables passed to the function.\n"
      "  log=false|true|0|1\n"
      "     Turns logarithmic scale on or off (default off)\n";
}

int main(int argc, char** argv)
{
    using Real = double;
    using boost::math::tools::viridis;
    using std::sqrt;

    std::function<std::array<Real, 3>(Real)> color_map = viridis<Real>;
    std::string requested_color_map = "viridis";
    std::string function_name = "gamma_p";
    int64_t image_width = 1024;
    int64_t image_height = 1024;

    double x_min{ 0.001 }, x_max{ 20 };
    double y_min{ 0.001 }, y_max{ 20 };

    Real(*the_function)(Real, Real) = boost::math::gamma_p;
    bool log_scale = false;
    bool debug = false;

    for(unsigned i = 1; i < argc; ++i)
    {
       std::string arg = std::string(argv[i]);
       if (arg == "smooth_cool_warm") {
          requested_color_map = arg;
          color_map = boost::math::tools::smooth_cool_warm<Real>;
       }
       else if (arg == "plasma") {
          requested_color_map = arg;
          color_map = boost::math::tools::plasma<Real>;
       }
       else if (arg == "black_body") {
          requested_color_map = arg;
          color_map = boost::math::tools::black_body<Real>;
       }
       else if (arg == "inferno") {
          requested_color_map = arg;
          color_map = boost::math::tools::inferno<Real>;
       }
       else if (arg == "kindlmann") {
          requested_color_map = arg;
          color_map = boost::math::tools::kindlmann<Real>;
       }
       else if (arg == "extended_kindlmann") {
          requested_color_map = arg;
          color_map = boost::math::tools::extended_kindlmann<Real>;
       }
       else if (arg.compare(0, 6, "width=") == 0)
       {
          image_width = std::strtol(arg.c_str() + 6, nullptr, 10);
       }
       else if (arg.compare(0, 7, "height=") == 0)
       {
          image_height = std::strtol(arg.c_str() + 7, nullptr, 10);
       }
       else if (arg.compare(0, 6, "x_min=") == 0)
       {
          x_min = std::strtod(arg.c_str() + 6, nullptr);
       }
       else if (arg.compare(0, 6, "x_max=") == 0)
       {
          x_max = std::strtod(arg.c_str() + 6, nullptr);
       }
       else if (arg.compare(0, 6, "y_min=") == 0)
       {
          y_min = std::strtod(arg.c_str() + 6, nullptr);
       }
       else if (arg.compare(0, 6, "y_max=") == 0)
       {
          y_max = std::strtod(arg.c_str() + 6, nullptr);
       }
       else if (arg == "log=1")
       {
          log_scale = true;
       }
       else if (arg == "log=0")
       {
          log_scale = false;
       }
       else if (arg == "log=true")
       {
          log_scale = true;
       }
       else if (arg == "log=false")
       {
          log_scale = false;
       }
       else if (arg == "debug")
       {
          debug = true;
       }
       else if (arg == "gamma_p")
       {
          the_function = boost::math::gamma_p;
          function_name = arg;
          if (requested_color_map == "viridis")
          {
             std::cout << "Setting default gamma_p color map to extended_kindlmann" << std::endl;
             requested_color_map = "extended_kindlmann";
             color_map = boost::math::tools::extended_kindlmann<Real>;
          }
       }
       else if (arg == "gamma_q")
       {
          the_function = boost::math::gamma_q;
          function_name = arg;
          if (requested_color_map == "viridis")
          {
             std::cout << "Setting default gamma_p color map to extended_kindlmann" << std::endl;
             requested_color_map = "extended_kindlmann";
             color_map = boost::math::tools::extended_kindlmann<Real>;
          }
       }
       else if (arg == "gamma_p_inv")
       {
          the_function = boost::math::gamma_p_inv;
          function_name = arg;
          if (y_max > 1)
          {
             std::cout << "Setting y range to [0.01, 0.99] for gamma_p_inv" << std::endl;
             y_min = 0.01;
             y_max = 0.99;
          }
          if (requested_color_map == "viridis")
          {
             std::cout << "Setting default gamma_p_inv color map to inferno" << std::endl;
             requested_color_map = "inferno";
             color_map = boost::math::tools::inferno<Real>;
          }
       }
       else if (arg == "gamma_q_inv")
       {
          the_function = boost::math::gamma_q_inv;
          function_name = arg;
          if (y_max > 1)
          {
             std::cout << "Setting y range to [0.01, 0.99] for gamma_p_inv" << std::endl;
             y_min = 0.01;
             y_max = 0.99;
          }
          if (requested_color_map == "viridis")
          {
             std::cout << "Setting default gamma_p_inv color map to inferno" << std::endl;
             requested_color_map = "inferno";
             color_map = boost::math::tools::inferno<Real>;
          }
       }
       else if (arg == "beta")
       {
          the_function = boost::math::beta;
          function_name = arg;
          if (log_scale == false)
          {
             std::cout << "Setting log scale to true for beta" << std::endl;
             log_scale = true;
          }
          if (requested_color_map == "viridis")
          {
             std::cout << "Setting default beta color map to smooth_cool_warm" << std::endl;
             requested_color_map = "smooth_cool_warm";
             color_map = boost::math::tools::smooth_cool_warm<Real>;
          }
       }
       else if (arg == "cyl_bessel_j")
       {
          the_function = boost::math::cyl_bessel_j;
          function_name = arg;
          if (requested_color_map == "viridis")
          {
             std::cout << "Setting default beta color map to smooth_cool_warm" << std::endl;
             requested_color_map = "smooth_cool_warm";
             color_map = boost::math::tools::smooth_cool_warm<Real>;
          }
       }
       else if (arg == "cyl_neumann")
       {
          the_function = boost::math::cyl_neumann;
          function_name = arg;
          if (requested_color_map == "viridis")
          {
             std::cout << "Setting default cyl_neumann color map to black_body" << std::endl;
             requested_color_map = "black_body";
             color_map = boost::math::tools::black_body<Real>;
          }
          if (x_max > 1.5)
          {
             std::cout << "Setting cyl_neumann default x range to [0.5,1.5]" << std::endl;
             x_min = 0.5;
             x_max = 1.5;
          }
          if (log_scale == false)
          {
             std::cout << "Turning on log scale for cyl_neumann" << std::endl;
             log_scale = true;
          }
       }
       else if (arg == "cyl_bessel_i")
       {
          the_function = boost::math::cyl_bessel_i;
          function_name = arg;
          if (requested_color_map == "viridis")
          {
             std::cout << "Setting default cyl_bessel_i color map to black_body" << std::endl;
             requested_color_map = "black_body";
             color_map = boost::math::tools::black_body<Real>;
          }
          if (x_max > 1.5)
          {
             std::cout << "Setting cyl_bessel_i default x range to [0.5,1.5]" << std::endl;
             x_min = 0.5;
             x_max = 1.5;
          }
          if (log_scale == false)
          {
             std::cout << "Turning on log scale for cyl_bessel_i" << std::endl;
             log_scale = true;
          }
       }
       else if (arg == "cyl_bessel_k")
       {
          the_function = boost::math::cyl_bessel_k;
          function_name = arg;
          if (requested_color_map == "viridis")
          {
             std::cout << "Setting default cyl_bessel_k color map to plasma" << std::endl;
             requested_color_map = "plasma";
             color_map = boost::math::tools::plasma<Real>;
          }
          if (x_max > 1.5)
          {
             std::cout << "Setting cyl_bessel_k default x range to [0,5]" << std::endl;
             x_min = 0.01;
             x_max = 5;
          }
          if (log_scale == false)
          {
             std::cout << "Turning on log scale for cyl_bessel_k" << std::endl;
             log_scale = true;
          }
       }
       else if (arg == "cyl_bessel_d")
       {
          the_function = boost::math::cyl_bessel_k;
          function_name = arg;
          if (log_scale == false)
          {
             std::cout << "Turning on log scale for cyl_bessel_d" << std::endl;
             log_scale = true;
          }
       }
       else if (arg == "ellint_1")
       {
          the_function = boost::math::ellint_1;
          function_name = arg;
          // x_max=1 y_max=1.5 kindlmann log=true
          if (x_max >= 20)
          {
             std::cout << "Setting ellint_1 x range to [0, 1]" << std::endl;
             x_max = 1;
          }
          if (y_max >= 20)
          {
             std::cout << "Setting ellint_1 y range to [0, 1.5]" << std::endl;
             x_max = 1.5;
          }
          if (log_scale == false)
          {
             std::cout << "Turning on log scale for ellint_1" << std::endl;
             log_scale = true;
          }
          if (requested_color_map == "viridis")
          {
             std::cout << "Setting default ellint_1 color map to kindlmann" << std::endl;
             requested_color_map = "kindlmann";
             color_map = boost::math::tools::kindlmann<Real>;
          }
       }
       else if (arg == "ellint_2")
       {
          the_function = boost::math::ellint_2;
          function_name = arg;
          // x_max=1 y_max=1.5 kindlmann log=true
          if (x_max >= 20)
          {
             std::cout << "Setting ellint_2 x range to [-1, 1]" << std::endl;
             x_max = 1;
             x_min = -1;
          }
          if (y_max >= 20)
          {
             std::cout << "Setting ellint_2 y range to [0, 1.5]" << std::endl;
             y_max = 1.5;
          }
          if (log_scale == false)
          {
             std::cout << "Turning on log scale for ellint_2" << std::endl;
             log_scale = true;
          }
          if (requested_color_map == "viridis")
          {
             std::cout << "Setting default ellint_1 color map to kindlmann" << std::endl;
             requested_color_map = "kindlmann";
             color_map = boost::math::tools::kindlmann<Real>;
          }
       }
       else if (arg == "ellint_3")
       {
          the_function = boost::math::ellint_3;
          function_name = arg;
          // x_max=1 y_max=1.5 kindlmann log=true
          if (x_max >= 20)
          {
             std::cout << "Setting ellint_3 x range to [-0.99, 0.99]" << std::endl;
             x_max = 0.99;
             x_min = -0.99;
          }
          if (y_max >= 20)
          {
             std::cout << "Setting ellint_3 y range to [0, 1]" << std::endl;
             y_max = 1;
          }
          if (log_scale == false)
          {
             std::cout << "Turning on log scale for ellint_3" << std::endl;
             log_scale = true;
          }
          if (requested_color_map == "viridis")
          {
             std::cout << "Setting default ellint_3 color map to kindlmann" << std::endl;
             requested_color_map = "kindlmann";
             color_map = boost::math::tools::kindlmann<Real>;
          }
       }
       else if (arg == "jacobi_zeta")
       {
          the_function = boost::math::jacobi_zeta;
          function_name = arg;
          // x_max=1 y_max=1.5 kindlmann log=true
          if (x_max >= 20)
          {
             std::cout << "Setting jacobi_zeta x range to [-0.99, 0.99]" << std::endl;
             x_max = 0.99;
             x_min = -0.99;
          }
          if (y_max >= 20)
          {
             std::cout << "Setting jacobi_zeta y range to [0, 1]" << std::endl;
             y_max = 0.99;
          }
          if (requested_color_map == "viridis")
          {
             std::cout << "Setting default jacobi_zeta color map to kindlmann" << std::endl;
             requested_color_map = "kindlmann";
             color_map = boost::math::tools::kindlmann<Real>;
          }
       }
       else if (arg == "heuman_lambda")
       {
          the_function = boost::math::heuman_lambda;
          function_name = arg;
          // x_max=1 y_max=1.5 kindlmann log=true
          if (x_max >= 20)
          {
             std::cout << "Setting heuman_lambda x range to [-0.99, 0.99]" << std::endl;
             x_max = 0.99;
             x_min = -0.99;
          }
          if (y_max >= 20)
          {
             std::cout << "Setting heuman_lambda y range to [0, 1]" << std::endl;
             y_max = 0.99;
          }
          if (requested_color_map == "viridis")
          {
             std::cout << "Setting default heuman_lambda color map to kindlmann" << std::endl;
             requested_color_map = "kindlmann";
             color_map = boost::math::tools::kindlmann<Real>;
          }
       }
       else if (arg == "jacobi_theta1")
       {
          the_function = boost::math::jacobi_theta1;
          function_name = arg;
          if (y_max >= 20)
          {
             std::cout << "Setting jacobi_theta1 y range to [0, 1]" << std::endl;
             y_max = 0.99;
          }
          if (requested_color_map == "viridis")
          {
             std::cout << "Setting default jacobi_theta1 color map to kindlmann" << std::endl;
             requested_color_map = "kindlmann";
             color_map = boost::math::tools::kindlmann<Real>;
          }
       }
       else if (arg == "1F1")
       {
          the_function = hypergeometric_1F1_at_half;
          function_name = arg;
          if (x_min >= 0)
          {
             std::cout << "Setting 1F1 x range to [-20,20]" << std::endl;
             x_min = -20;
             x_max = 20;
          }
          if (y_min >= 0)
          {
             std::cout << "Setting 1F1 y range to [-20,20]" << std::endl;
             y_min = -20;
             y_max = 20;
          }
          if (requested_color_map == "viridis")
          {
             std::cout << "Setting default 1F1 color map to extended_kindlmann" << std::endl;
             requested_color_map = "extended_kindlmann";
             color_map = boost::math::tools::extended_kindlmann<Real>;
          }
          if (!log_scale)
          {
             std::cout << "Turning on logarithmic scale for 1F1" << std::endl;
             log_scale = true;
          }
       }
       else if (arg == "help")
       {
          show_help();
          return 0;
       }
       else 
       {
          std::cerr << "Could not recognize argument " << argv[i] << ".\n\n";
          show_help();
          return 1;
       }
    }

    std::vector<std::uint8_t> img(4*image_width*image_height, 0);
    std::vector<Real>         points(image_width*image_height, 0);

    Real min_value{ std::numeric_limits<Real>::infinity() }, max_value{ -std::numeric_limits<Real>::infinity() };
    //
    // Get a matrix of points:
    //
    for (int64_t i = 0; i < image_width; ++i)
    {
       for (int64_t j = 0; j < image_height; ++j)
      {
            double x = x_max - (image_width - i) * (x_max - x_min) / image_width;
            double y = y_max - (image_height - j) * (y_max - y_min) / image_height;

            Real p = the_function(x, y);
            if (std::isnan(p))
               std::cerr << "Ooops, got a NaN" << std::endl;
            if (p < min_value)
               min_value = p;
            if (p > max_value)
               max_value = p;
            points[i + image_width * (image_height - j - 1)] = p;
        }
    }
    std::cout << "Function range is: [" << std::setprecision(3) << min_value << "," << max_value << "]\n";
    //
    // Handle log scale, the formula differs depending on whether we have found negative values or not:
    //
    if (log_scale)
    {
       Real new_max = -std::numeric_limits<Real>::infinity();
       Real new_min = std::numeric_limits<Real>::infinity();
       for (int64_t i = 0; i < points.size(); ++i)
       {
          Real p = points[i];
          if (min_value <= 0)
             p = boost::math::sign(p) * log10(1 + std::fabs(p * boost::math::constants::ln_ten<Real>()));
          else
             p = log(p);
          if (std::isnan(p))
             std::cerr << "Ooops, got a NaN" << std::endl;
          if (p < new_min)
             new_min = p;
          if (p > new_max)
             new_max = p;
          points[i] = p;
       }
       max_value = new_max;
       min_value = new_min;
       std::cout << "Function range is: [" << std::setprecision(3) << min_value << "," << max_value << "]\n";
    }

    //
    // Normalize the points so they are all in [0,1]
    //
    for (int64_t i = 0; i < points.size(); ++i)
    {
      double p = points[i];
      p -= min_value;
      p /= (max_value - min_value);
      points[i] = p;
    }
    //
    // debugging, adds an alternating 0 and 1 row on the second half of the zeroth row:
    //
    if (debug)
    {
       for (int64_t i = image_width / 2; i < image_width; ++i)
          points[image_width * (image_height - 1) + i] = i & 1;
    }

    //
    // Now calculate the RGB bitmap from the points:
    //
    for (int64_t i = 0; i < image_width; ++i)
    {
       for (int64_t j = 0; j < image_height; ++j)
       {
          double p = points[i + image_width * j];
          auto c = boost::math::tools::to_8bit_rgba(color_map(p));
          int64_t idx = 4 * (image_width * j + i);
          img[idx + 0] = c[0];
          img[idx + 1] = c[1];
          img[idx + 2] = c[2];
          img[idx + 3] = c[3];
       }
    }

    // Requires lodepng.h
    // See: https://github.com/lvandeve/lodepng for download and compilation instructions
    write_png(requested_color_map + "_" + function_name + ".png", img, image_width, image_height);
}
