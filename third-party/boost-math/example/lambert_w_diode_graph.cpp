// Copyright Paul A. Bristow 2016
// Copyright John Z. Maddock 2016

// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or
//  copy at http ://www.boost.org/LICENSE_1_0.txt).

/*! \brief Graph showing use of Lambert W function to compute current
through a diode-connected transistor with preset series resistance.

\details T. C. Banwell and A. Jayakumar,
Exact analytical solution of current flow through diode with series resistance,
Electron Letters, 36(4):291-2 (2000).
DOI:  doi.org/10.1049/el:20000301

The current through a diode connected NPN bipolar junction transistor (BJT)
type 2N2222 (See https://en.wikipedia.org/wiki/2N2222 and
https://www.fairchildsemi.com/datasheets/PN/PN2222.pdf Datasheet)
was measured, for a voltage between 0.3 to 1 volt, see Fig 2 for a log plot, showing a knee visible at about 0.6 V.

The transistor parameter I sat was estimated to be 25 fA and the ideality factor = 1.0.
The intrinsic emitter resistance re was estimated from the rsat = 0 data to be 0.3 ohm.

The solid curves in Figure 2 are calculated using equation 5 with rsat included with re.

http://www3.imperial.ac.uk/pls/portallive/docs/1/7292572.PDF

*/

#ifndef BOOST_MATH_STANDALONE

#include <boost/math/special_functions/lambert_w.hpp>
using boost::math::lambert_w0;
#include <boost/math/special_functions.hpp>
using boost::math::isfinite;
#include <boost/svg_plot/svg_2d_plot.hpp>
using namespace boost::svg;

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
Compute thermal voltage as a function of temperature,
about 25 mV at room temperature.
https://en.wikipedia.org/wiki/Boltzmann_constant#Role_in_semiconductor_physics:_the_thermal_voltage

\param temperature Temperature (degrees Celsius).
*/
const double v_thermal(double temperature)
{
  constexpr const double boltzmann_k = 1.38e-23; // joules/kelvin.
  constexpr double charge_q = 1.6021766208e-19; // Charge of an electron (columb).
  double temp = +273; // Degrees C to K.
  return boltzmann_k * temp / charge_q;
} // v_thermal

  /*!
  Banwell & Jayakumar, equation 2, page 291.
  */
double i(double isat, double vd, double vt, double nu)
{
  double i = isat * (exp(vd / (nu * vt)) - 1);
  return i;
} //

  /*!
  Banwell & Jayakumar, Equation 4, page 291.
  i current flow = isat
  v voltage source.
  isat reverse saturation current in equation 4.
  (might implement equation 4 instead of simpler equation 5?).
  vd voltage drop = v - i* rs  (equation 1).
  vt  thermal voltage, 0.0257025 = 25 mV.
  nu junction ideality factor (default = unity), also known as the emission coefficient.
  re intrinsic emitter resistance, estimated to be 0.3 ohm from low current.
  rsat reverse saturation current

  \param v Voltage V to compute current I(V).
  \param vt Thermal voltage, for example 0.0257025 = 25 mV, computed from boltzmann_k * temp / charge_q;
  \param rsat Resistance in series with the diode.
  \param re Intrinsic emitter resistance (estimated to be 0.3 ohm from the Rs = 0 data)
  \param isat Reverse saturation current (See equation 2).
  \param nu Ideality factor (default = unity).

  \returns I amp as function of V volt.
  */

//[lambert_w_diode_graph_2
double iv(double v, double vt, double rsat, double re, double isat, double nu = 1.)
{
  // V thermal 0.0257025 = 25 mV
  // was double i = (nu * vt/r) * lambert_w((i0 * r) / (nu * vt)); equ 5.

  rsat = rsat + re;
  double i = nu * vt / rsat;
 // std::cout << "nu * vt / rsat = " << i << std::endl; // 0.000103223

  double x = isat * rsat / (nu * vt);
//  std::cout << "isat * rsat / (nu * vt) = " << x << std::endl;

  double eterm = (v + isat * rsat) / (nu * vt);
 // std::cout << "(v + isat * rsat) / (nu * vt) = " << eterm << std::endl;

  double e = exp(eterm);
//  std::cout << "exp(eterm) = " << e << std::endl;

  double w0 = lambert_w0(x * e);
//  std::cout << "w0 = " << w0 << std::endl;
  return i * w0 - isat;
} // double iv

//] [\lambert_w_diode_graph_2]


std::array<double, 5> rss = { 0., 2.18, 10., 51., 249 };  // series resistance (ohm).
std::array<double, 7> vds = { 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };  // Diode voltage.
std::array<double, 7> lni = { -19.65, -15.75, -11.86, -7.97, -4.08, -0.0195, 3.6 }; // ln(current).

int main()
{
  try
  {
    std::cout << "Lambert W diode current example." << std::endl;

//[lambert_w_diode_graph_1
    double nu = 1.0; // Assumed ideal.
    double vt = v_thermal(25); // v thermal, Shockley equation, expect about 25 mV at room temperature.
    double boltzmann_k = 1.38e-23; // joules/kelvin
    double temp = 273 + 25;
    double charge_q = 1.6e-19; // column
    vt = boltzmann_k * temp / charge_q;
    std::cout << "V thermal " << vt << std::endl; // V thermal 0.0257025 = 25 mV
    double rsat = 0.;
    double isat = 25.e-15; //  25 fA;
    std::cout << "Isat = " << isat << std::endl;
    double re = 0.3;  // Estimated from slope of straight section of graph (equation 6).
    double v = 0.9;
    double icalc = iv(v, vt, 249., re, isat);
    std::cout << "voltage = " << v << ", current = " << icalc << ", " << log(icalc) << std::endl; // voltage = 0.9, current = 0.00108485, -6.82631
//] [/lambert_w_diode_graph_1]

    // Plot a few measured data points.
    std::map<const double, double> zero_data;  // Extrapolated from slope of measurements with no external resistor.
    zero_data[0.3] = -19.65;
    zero_data[0.4] = -15.75;
    zero_data[0.5] = -11.86;
    zero_data[0.6] = -7.97;
    zero_data[0.7] = -4.08;
    zero_data[0.8] = -0.0195;
    zero_data[0.9] =  3.9;

    std::map<const double, double>  measured_zero_data; // No external series resistor.
    measured_zero_data[0.3] = -19.65;
    measured_zero_data[0.4] = -15.75;
    measured_zero_data[0.5] = -11.86;
    measured_zero_data[0.6] = -7.97;
    measured_zero_data[0.7] = -4.2;
    measured_zero_data[0.72] = -3.5;
    measured_zero_data[0.74] = -2.8;
    measured_zero_data[0.76] = -2.3;
    measured_zero_data[0.78] = -2.0;
    // Measured from Fig 2 as raw data not available.

    double step = 0.1;
    for (int i = 0; i < vds.size(); i++)
    {
      zero_data[vds[i]] = lni[i];
      std::cout << lni[i] << "  " << vds[i] << std::endl;
    }
    step = 0.01;

    std::map<const double, double> data_2;
    for (double v = 0.3; v < 1.; v += step)
    {
      double current = iv(v, vt, 2., re, isat);
      data_2[v] = log(current);
      // std::cout << "v " << v << ", current = " << current << " log current = " << log(current) << std::endl;
    }
    std::map<const double, double> data_10;
    for (double v = 0.3; v < 1.; v += step)
    {
      double current = iv(v, vt, 10., re, isat);
      data_10[v] = log(current);
    //  std::cout << "v " << v << ", current = " << current << " log current = " << log(current) << std::endl;
    }
    std::map<const double, double> data_51;
    for (double v = 0.3; v < 1.; v += step)
    {
      double current = iv(v, vt, 51., re, isat);
      data_51[v] = log(current);
     // std::cout << "v " << v << ", current = " << current << " log current = " << log(current) << std::endl;
    }
    std::map<const double, double> data_249;
    for (double v = 0.3; v < 1.; v += step)
    {
      double current = iv(v, vt, 249., re, isat);
      data_249[v] = log(current);
      // std::cout << "v " << v << ", current = " << current << " log current = " << log(current) << std::endl;
    }

    svg_2d_plot data_plot;

    data_plot.title("Diode current versus voltage")
      .x_size(400)
      .y_size(300)
      .legend_on(true)
      .legend_lines(true)
      .x_label("voltage (V)")
      .y_label("log(current) (A)")
      //.x_label_on(true)
      //.y_label_on(true)
      //.xy_values_on(false)
      .x_range(0.25, 1.)
      .y_range(-20., +4.)
      .x_major_interval(0.1)
      .y_major_interval(4)
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
      .copyright_date("2016")
      //.background_border_color(black);
      ;

    // &#x2080; = subscript zero.
    data_plot.plot(zero_data, "I&#x2080;(V)").fill_color(lightgray).shape(none).size(3).line_on(true).line_width(0.5);
    data_plot.plot(measured_zero_data, "Rs=0 &#x3A9;").fill_color(lightgray).shape(square).size(3).line_on(true).line_width(0.5);
    data_plot.plot(data_2, "Rs=2 &#x3A9;").line_color(blue).shape(none).line_on(true).bezier_on(false).line_width(1);
    data_plot.plot(data_10, "Rs=10 &#x3A9;").line_color(purple).shape(none).line_on(true).bezier_on(false).line_width(1);
    data_plot.plot(data_51, "Rs=51 &#x3A9;").line_color(green).shape(none).line_on(true).line_width(1);
    data_plot.plot(data_249, "Rs=249 &#x3A9;").line_color(red).shape(none).line_on(true).line_width(1);
    data_plot.write("./diode_iv_plot");

    // bezier_on(true);
  }
  catch (std::exception& ex)
  {
    std::cout << ex.what() << std::endl;
  }


}  // int main()

   /*

   //[lambert_w_output_1
   Output:
   Lambert W diode current example.
   V thermal 0.0257025
   Isat = 2.5e-14
   voltage = 0.9, current = 0.00108485, -6.82631
   -19.65  0.3
   -15.75  0.4
   -11.86  0.5
   -7.97  0.6
   -4.08  0.7
   -0.0195  0.8
   3.6  0.9

   //] [/lambert_w_output_1]
   */
#endif // BOOST_MATH_STANDALONE
