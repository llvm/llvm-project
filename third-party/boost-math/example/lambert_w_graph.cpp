// Copyright Paul A. Bristow 2017
// Copyright John Z. Maddock 2017

// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or
//  copy at http ://www.boost.org/LICENSE_1_0.txt).

/*! \brief Graph showing use of Lambert W function.

\details

Both Lambert W0 and W-1 branches can be shown on one graph.
But useful to have another graph for larger values of argument z.
Need two separate graphs for Lambert W0 and -1 prime because
the sensible ranges and axes are too different.  

One would get too small LambertW0 in top right and W-1 in bottom left.

*/
#ifndef BOOST_MATH_STANDALONE

#include <boost/math/special_functions/lambert_w.hpp>
using boost::math::lambert_w0;
using boost::math::lambert_wm1;
using boost::math::lambert_w0_prime;
using boost::math::lambert_wm1_prime;

#include <boost/math/special_functions.hpp>
using boost::math::isfinite;
#include <boost/svg_plot/svg_2d_plot.hpp>
using namespace boost::svg;
#include <boost/svg_plot/show_2d_settings.hpp>
using boost::svg::show_2d_plot_settings;

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
#include <cmath> //

  /*!
  */
int main()
{
  try
  {
    std::cout << "Lambert W graph example." << std::endl;

//[lambert_w_graph_1
//] [/lambert_w_graph_1]
    {
      std::map<const double, double> wm1s;   // Lambert W-1 branch values.
      std::map<const double, double> w0s;   // Lambert W0 branch values.

      std::cout.precision(std::numeric_limits<double>::max_digits10);

      int count = 0;
      for (double z = -0.36787944117144232159552377016146086744581113103176804; z < 2.8; z += 0.001)
      {
        double w0 = lambert_w0(z);
        w0s[z] = w0;
   //     std::cout << "z " << z << ", w = " << w0 << std::endl;
        count++;
      }
      std::cout << "points " << count << std::endl;

      count = 0;
      for (double z = -0.3678794411714423215955237701614608727; z < -0.001; z += 0.001)
      {
        double wm1 = lambert_wm1(z);
        wm1s[z] = wm1;
        count++;
      }
      std::cout << "points " << count << std::endl;

      svg_2d_plot data_plot;
      data_plot.title("Lambert W function.")
        .x_size(400)
        .y_size(300)
        .legend_on(true)
        .legend_lines(true)
        .x_label("z")
        .y_label("W")
        .x_range(-1, 3.)
        .y_range(-4., +1.)
        .x_major_interval(1.)
        .y_major_interval(1.)
        .x_major_grid_on(true)
        .y_major_grid_on(true)
        //.x_values_on(true)
        //.y_values_on(true)
        .y_values_rotation(horizontal)
        //.plot_window_on(true)
        .x_values_precision(3)
        .y_values_precision(3)
        .coord_precision(4) // Needed to avoid stepping on curves.
        .copyright_holder("Paul A. Bristow")
        .copyright_date("2018")
        //.background_border_color(black);
        ;
      data_plot.plot(w0s, "W0 branch").line_color(red).shape(none).line_on(true).bezier_on(false).line_width(1);
      data_plot.plot(wm1s, "W-1 branch").line_color(blue).shape(none).line_on(true).bezier_on(false).line_width(1);
      data_plot.write("./lambert_w_graph");

      show_2d_plot_settings(data_plot); // For plot diagnosis only.

    } // small z Lambert W

    {  // bigger argument z Lambert W

      std::map<const double, double> w0s_big;   // Lambert W0 branch values for large z and W.
      std::map<const double, double> wm1s_big;   // Lambert W-1 branch values for small z and large -W.
      int count = 0;
      for (double z = -0.3678794411714423215955237701614608727; z < 10000.; z += 50.)
      {
        double w0 = lambert_w0(z);
        w0s_big[z] = w0;
        count++;
      }
      std::cout << "points " << count << std::endl;

      count = 0;
      for (double z = -0.3678794411714423215955237701614608727; z < -0.001; z += 0.001)
      {
        double wm1 = lambert_wm1(z);
        wm1s_big[z] = wm1;
        count++;
      }
     std::cout << "Lambert W0 large z argument points = " << count << std::endl;

     svg_2d_plot data_plot2;
     data_plot2.title("Lambert W0 function for larger z.")
      .x_size(400)
      .y_size(300)
      .legend_on(false)
      .x_label("z")
      .y_label("W")
      //.x_label_on(true)
      //.y_label_on(true)
      //.xy_values_on(false)
      .x_range(-1, 10000.)
      .y_range(-1., +8.)
      .x_major_interval(2000.)
      .y_major_interval(1.)
      .x_major_grid_on(true)
      .y_major_grid_on(true)
      //.x_values_on(true)
      //.y_values_on(true)
      .y_values_rotation(horizontal)
      //.plot_window_on(true)
      .x_values_precision(3)
      .y_values_precision(3)
      .coord_precision(4) // Needed to avoid stepping on curves.
      .copyright_holder("Paul A. Bristow")
      .copyright_date("2018")
      //.background_border_color(black);
    ;

    data_plot2.plot(w0s_big, "W0 branch").line_color(red).shape(none).line_on(true).bezier_on(false).line_width(1);
    // data_plot2.plot(wm1s_big, "W-1 branch").line_color(blue).shape(none).line_on(true).bezier_on(false).line_width(1);
    // This wouldn't show anything useful.
    data_plot2.write("./lambert_w_graph_big_w");
   } // Big argument z Lambert W

    { //  Lambert W0 Derivative plots

    //  std::map<const double, double> wm1ps;   // Lambert W-1 prime branch values.
      std::map<const double, double> w0ps;   // Lambert W0 prime branch values.

      std::cout.precision(std::numeric_limits<double>::max_digits10);

      int count = 0;
      for (double z = -0.36; z < 3.; z += 0.001)
      {
        double w0p = lambert_w0_prime(z);
        w0ps[z] = w0p;
        // std::cout << "z " << z << ", w0 = " << w0 << std::endl;
        count++;
      }
      std::cout << "points " << count << std::endl;

      //count = 0;
      //for (double z = -0.36; z < -0.1; z += 0.001)
      //{
      //  double wm1p = lambert_wm1_prime(z);
      //  std::cout << "z " << z << ", w-1 = " << wm1p << std::endl;
      //  wm1ps[z] = wm1p;
      //  count++;
      //}
      //std::cout << "points " << count << std::endl;

      svg_2d_plot data_plotp;
      data_plotp.title("Lambert W0 prime function.")
        .x_size(400)
        .y_size(300)
        .legend_on(false)
        .x_label("z")
        .y_label("W0'")
        .x_range(-0.3, +1.)
        .y_range(0., +5.)
        .x_major_interval(0.2)
        .y_major_interval(2.)
        .x_major_grid_on(true)
        .y_major_grid_on(true)
        .y_values_rotation(horizontal)
        .x_values_precision(3)
        .y_values_precision(3)
        .coord_precision(4) // Needed to avoid stepping on curves.
        .copyright_holder("Paul A. Bristow")
        .copyright_date("2018")
        ;

      // derivative of N[productlog(0, x), 55]  at x=0 to 10
      // Plot[D[N[ProductLog[0, x], 55], x], {x, 0, 10}]
      // Plot[ProductLog[x]/(x + x ProductLog[x]), {x, 0, 10}]
      data_plotp.plot(w0ps, "W0 prime branch").line_color(red).shape(none).line_on(true).bezier_on(false).line_width(1);
      data_plotp.write("./lambert_w0_prime_graph");
  } // Lambert W0 Derivative plots

    { //  Lambert Wm1 Derivative plots

    std::map<const double, double> wm1ps;   // Lambert W-1 prime branch values.

    std::cout.precision(std::numeric_limits<double>::max_digits10);

    int count = 0;
    for (double z = -0.3678; z < -0.00001; z += 0.001)
    {
      double wm1p = lambert_wm1_prime(z);
      // std::cout << "z " << z << ", w-1 = " << wm1p << std::endl;
      wm1ps[z] = wm1p;
      count++;
    }
    std::cout << "Lambert W-1 prime points = " << count << std::endl;

    svg_2d_plot data_plotp;
    data_plotp.title("Lambert W-1 prime function.")
      .x_size(400)
      .y_size(300)
      .legend_on(false)
      .x_label("z")
      .y_label("W-1'")
      .x_range(-0.4, +0.01)
      .x_major_interval(0.1)
      .y_range(-20., -5.)
      .y_major_interval(5.)
      .x_major_grid_on(true)
      .y_major_grid_on(true)
      .y_values_rotation(horizontal)
      .x_values_precision(3)
      .y_values_precision(3)
      .coord_precision(4) // Needed to avoid stepping on curves.
      .copyright_holder("Paul A. Bristow")
      .copyright_date("2018")
      ;

      // derivative of N[productlog(0, x), 55]  at x=0 to 10
      // Plot[D[N[ProductLog[0, x], 55], x], {x, 0, 10}]
      // Plot[ProductLog[x]/(x + x ProductLog[x]), {x, 0, 10}]
      data_plotp.plot(wm1ps, "W-1 prime branch").line_color(blue).shape(none).line_on(true).bezier_on(false).line_width(1);
      data_plotp.write("./lambert_wm1_prime_graph");
    } // Lambert W-1 prime graph
 } // try
  catch (std::exception& ex)
  {
    std::cout << ex.what() << std::endl;
  }
}  // int main()

   /*

   //[lambert_w_graph_1_output

   //] [/lambert_w_graph_1_output]
   */

#endif // BOOST_MATH_STANDALONE
