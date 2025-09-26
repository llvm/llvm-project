// Copyright Paul A. Bristow 2017, 2018
// Copyright John Z. Maddock 2017

// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or
//  copy at http ://www.boost.org/LICENSE_1_0.txt).

/*! \brief Graph showing differences of Lambert W function double from nearest representable values.

\details

*/

#include <boost/math/special_functions/lambert_w.hpp>
using boost::math::lambert_w0;
using boost::math::lambert_wm1;
#include <boost/math/special_functions.hpp>
using boost::math::isfinite;
#include <boost/svg_plot/svg_2d_plot.hpp>
using namespace boost::svg;

// For higher precision computation of Lambert W.
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/math/special_functions/next.hpp> // For float_distance.
using boost::math::float_distance;

#include <iostream>
// using std::cout;
// using std::endl;
#include <exception>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>
#include <utility>
using std::pair;
#include <map>
using std::map;
#include <set>
using std::multiset;
#include <limits>
using std::numeric_limits;
#include <cmath> // exp

/*!
*/


int main()
{
  try
  {
    std::cout << "Lambert W errors graph." << std::endl;
    using boost::multiprecision::cpp_bin_float_50; 
    using boost::multiprecision::cpp_bin_float_quad; 

    typedef cpp_bin_float_quad HPT; // High precision type.

    using boost::math::float_distance;
    using boost::math::policies::precision;
    using boost::math::policies::digits10;
    using boost::math::policies::digits2;
    using boost::math::policies::policy;

    std::cout.precision(std::numeric_limits<double>::max_digits10);

    //[lambert_w_graph_1

    //] [/lambert_w_graph_1]
    {
      std::map<const double, double> w0s;   // Lambert W0 branch values, default double precision, digits2 = 53.
      std::map<const double, double> w0s_50;   // Lambert W0 branch values digits2 = 50.

      int max_distance = 0;
      int total_distance = 0;
      int count = 0;
      const int bits = 7;
      double min_z = -0.367879; // Close to singularity at -0.3678794411714423215955237701614608727 -exp(-1)
      //double min_z = 0.06; // Above 0.05 switch point.
      double max_z = 99.99;
      double step_z = 0.05;

      for (HPT z = min_z; z < max_z; z += step_z)
      {
        double zd = static_cast<double>(z);
        double w0d = lambert_w0(zd); // double result from same default.
        HPT w0_best = lambert_w0<HPT>(z);
        double w0_best_d = static_cast<double>(w0_best); // reference result.
       // w0s[zd] = (w0d - w0_best_d); // absolute difference.
        // w0s[z] = 100 * (w0 - w0_best) / w0_best; // difference relative % .
        w0s[zd] = float_distance<double>(w0d, w0_best_d); // difference in bits.
        double fd = float_distance<double>(w0d, w0_best_d);
        int distance = static_cast<int>(fd);
        int abs_distance = abs(distance);

         // std::cout << count << " " << zd << " " << w0d << " " << w0_best_d
         //   << ", Difference = " << w0d - w0_best_d << ", % = " << (w0d - w0_best_d) / w0d << ", Distance = " << distance << std::endl;

        total_distance += abs_distance;
        if (abs_distance > max_distance)
        {
          max_distance = abs_distance;
        }
        count++;
      } // for z
      std::cout << "points " << count << std::endl;
      std::cout.precision(3);
      std::cout << "max distance " << max_distance << ", total distances = " << total_distance
        << ", mean distance " << (float)total_distance / count << std::endl;

      typedef std::map<const double, double>::const_iterator Map_Iterator;

   /* for (std::map<const double, double>::const_iterator it = w0s.begin(); it != w0s.end(); ++it)
      {
        std::cout  << " " << *(it) << "\n";
      }
  */
      svg_2d_plot data_plot_0; // <-0.368, -46> <-0.358, -4> <-0.348, 1>...

      data_plot_0.title("Lambert W0 function differences from 'best' for double.")
        .title_font_size(11)
        .x_size(400)
        .y_size(200)
        .legend_on(false)
        //.legend_font_weight(1)
        .x_label("z")
        .y_label("W0 difference (bits)")
        //.x_label_on(true)
        //.y_label_on(true)
        //.xy_values_on(false)
        .x_range(-1, 100.)
        .y_range(-4., +4.)
        .x_major_interval(10.)
        .y_major_interval(2.)
        .x_major_grid_on(true)
        .y_major_grid_on(true)
        .x_label_font_size(9)
        .y_label_font_size(9)
        //.x_values_on(true)
        //.y_values_on(true)
        .y_values_rotation(horizontal)
        //.plot_window_on(true)
        .x_values_precision(3)
        .y_values_precision(3)
        .coord_precision(3) // Needed to avoid stepping on curves.
        //.coord_precision(4) // Needed to avoid stepping on curves.
        .copyright_holder("Paul A. Bristow")
        .copyright_date("2018")
        //.background_border_color(black);
        ;


      data_plot_0.plot(w0s, "W0 branch").line_color(red).shape(none).line_on(true).bezier_on(false).line_width(0.2);
      //data_plot.plot(wm1s, "W-1 branch").line_color(blue).shape(none).line_on(true).bezier_on(false).line_width(1);
      data_plot_0.write("./lambert_w0_errors_graph");

    } // end W0 branch plot.
    { // Repeat for Lambert W-1 branch.

      std::map<const double, double> wm1s;   // Lambert W-1 branch values.
      std::map<const double, double> wm1s_50;   // Lambert Wm1 branch values digits2 = 50.

      int max_distance = 0;
      int total_distance = 0;
      int count = 0;
      const int bits = 7;
      double min_z = -0.367879; // Close to singularity at -0.3678794411714423215955237701614608727 -exp(-1)
                                //double min_z = 0.06; // Above 0.05 switch point.
      double max_z = -0.0001;
      double step_z = 0.001;

      for (HPT z = min_z; z < max_z; z += step_z)
      {
        if (z > max_z)
        {
          break;
        }
        double zd = static_cast<double>(z);
        double wm1d = lambert_wm1(zd); // double result from same default.
        HPT wm1_best = lambert_wm1<HPT>(z);
        double wm1_best_d = static_cast<double>(wm1_best); // reference result.
                                                         // wm1s[zd] = (wm1d - wm1_best_d); // absolute difference.
                                                         // wm1s[z] = 100 * (wm1 - wm1_best) / wm1_best; // difference relative % .
        wm1s[zd] = float_distance<double>(wm1d, wm1_best_d); // difference in bits.
        double fd = float_distance<double>(wm1d, wm1_best_d);
        int distance = static_cast<int>(fd);
        int abs_distance = abs(distance);

         //std::cout << count << " " << zd << " " << wm1d << " " << wm1_best_d
         //  << ", Difference = " << wm1d - wm1_best_d << ", % = " << (wm1d - wm1_best_d) / wm1d << ", Distance = " << distance << std::endl;

        total_distance += abs_distance;
        if (abs_distance > max_distance)
        {
          max_distance = abs_distance;
        }
        count++;

      } // for z
      std::cout << "points " << count << std::endl;
      std::cout.precision(3);
      std::cout << "max distance " << max_distance << ", total distances = " << total_distance
        << ", mean distance " << (float)total_distance / count << std::endl;

      typedef std::map<const double, double>::const_iterator Map_Iterator;

      /* for (std::map<const double, double>::const_iterator it = wm1s.begin(); it != wm1s.end(); ++it)
      {
      std::cout  << " " << *(it) << "\n";
      }
      */
      svg_2d_plot data_plot_m1; // <-0.368, -46> <-0.358, -4> <-0.348, 1>...

      data_plot_m1.title("Lambert W-1 function differences from 'best' for double.")
        .title_font_size(11)
        .x_size(400)
        .y_size(200)
        .legend_on(false)
        //.legend_font_weight(1)
        .x_label("z")
        .y_label("W-1 difference (bits)")
        .x_range(-0.39, +0.0001)
        .y_range(-4., +4.)
        .x_major_interval(0.1)
        .y_major_interval(2.)
        .x_major_grid_on(true)
        .y_major_grid_on(true)
        .x_label_font_size(9)
        .y_label_font_size(9)
        //.x_values_on(true)
        //.y_values_on(true)
        .y_values_rotation(horizontal)
        //.plot_window_on(true)
        .x_values_precision(3)
        .y_values_precision(3)
        .coord_precision(3) // Needed to avoid stepping on curves.
                            //.coord_precision(4) // Needed to avoid stepping on curves.
        .copyright_holder("Paul A. Bristow")
        .copyright_date("2018")
        //.background_border_color(black);
        ;
        data_plot_m1.plot(wm1s, "W-1 branch").line_color(darkblue).shape(none).line_on(true).bezier_on(false).line_width(0.2);
        data_plot_m1.write("./lambert_wm1_errors_graph");
    }
  }
  catch (std::exception& ex)
  {
    std::cout << ex.what() << std::endl;
  }
}  // int main()

   /*
   //[lambert_w_errors_graph_1_output
   Lambert W errors graph.
   points 2008
   max distance 46, total distances = 717, mean distance 0.357

   points 368
   max distance 23, total distances = 329, mean distance 0.894

   //] [/lambert_w_errors_graph_1_output]
   */
