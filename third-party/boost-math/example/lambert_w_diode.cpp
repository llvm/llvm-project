// Copyright Paul A. Bristow 2016
// Copyright John Z. Maddock 2016

// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or
//  copy at http ://www.boost.org/LICENSE_1_0.txt).

/*! brief Example of using Lambert W function to compute current through a diode connected transistor with preset series resistance.
  \details T. C. Banwell and A. Jayakumar,
  Exact analytical solution of current flow through diode with series resistance,
  Electron Letters, 36(4):291-2 (2000)
  DOI:  doi.org/10.1049/el:20000301 

  The current through a diode connected NPN bipolar junction transistor (BJT) 
  type 2N2222 (See https://en.wikipedia.org/wiki/2N2222 and 
  https://www.fairchildsemi.com/datasheets/PN/PN2222.pdf Datasheet)
  was measured, for a voltage between 0.3 to 1 volt, see Fig 2 for a log plot,
  showing a knee visible at about 0.6 V.

  The transistor parameter isat was estimated to be 25 fA and the ideality factor = 1.0.
  The intrinsic emitter resistance re was estimated from the rsat = 0 data to be 0.3 ohm.

  The solid curves in Figure 2 are calculated using equation 5 with rsat included with re.

  http://www3.imperial.ac.uk/pls/portallive/docs/1/7292572.PDF
*/

#include <boost/math/special_functions/lambert_w.hpp>
using boost::math::lambert_w0;

#include <iostream>
// using std::cout;
// using std::endl;
#include <exception>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>

/*!
Compute thermal voltage as a function of temperature,
about 25 mV at room temperature.
https://en.wikipedia.org/wiki/Boltzmann_constant#Role_in_semiconductor_physics:_the_thermal_voltage

\param temperature Temperature (degrees centigrade).
*/
const double v_thermal(double temperature)
{
  constexpr const double boltzmann_k = 1.38e-23; // joules/kelvin.
  const double charge_q = 1.6021766208e-19; // Charge of an electron (columb).
  double temp =+ 273; // Degrees C to K.
  return boltzmann_k * temp / charge_q;
} // v_thermal

/*!
Banwell & Jayakumar, equation 2
*/
double i(double isat, double vd, double vt, double nu)
{
  double i = isat * (exp(vd / (nu * vt)) - 1);
  return i;
} // 

/*!

Banwell & Jayakumar, Equation 4.
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
double iv(double v, double vt, double rsat, double re, double isat, double nu = 1.)
{ 
  // V thermal 0.0257025 = 25 mV
  // was double i = (nu * vt/r) * lambert_w((i0 * r) / (nu * vt)); equ 5.

  rsat = rsat + re; 
  double i = nu * vt / rsat;
  std::cout << "nu * vt / rsat = " << i << std::endl; // 0.000103223

  double x = isat * rsat / (nu * vt);
  std::cout << "isat * rsat / (nu * vt) = " << x << std::endl;

  double eterm = (v + isat * rsat) / (nu * vt);
  std::cout << "(v + isat * rsat) / (nu * vt) = " << eterm << std::endl;

  double e = exp(eterm);
  std::cout << "exp(eterm) = " << e << std::endl;

  double w0 = lambert_w0(x * e);
  std::cout << "w0 = " << w0 << std::endl;
  return i * w0 - isat;

} // double iv

std::array<double, 5> rss = {0., 2.18, 10., 51., 249};  // series resistance (ohm).
std::array<double, 8> vds = { 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };  // Diode voltage.

int main()
{
  try
  {
    std::cout << "Lambert W diode current example." << std::endl;

    //[lambert_w_diode_example_1
    double x = 0.01;
    //std::cout << "Lambert W (" << x << ") = " << lambert_w(x) << std::endl; // 0.00990147

    double nu = 1.0;                  // Assumed ideal.
    double vt = v_thermal(25);        // v thermal, Shockley equation, expect about 25 mV at room temperature.
    double boltzmann_k = 1.38e-23;    // joules/kelvin
    double temp = 273 + 25;
    double charge_q = 1.6e-19;        // column
    vt = boltzmann_k * temp / charge_q;
    std::cout << "V thermal " 
       << vt << std::endl;            // V thermal 0.0257025 = 25 mV
    double rsat = 0.;
    double isat = 25.e-15;            //  25 fA;
    std::cout << "Isat = " << isat << std::endl;

    double re = 0.3;  // Estimated from slope of straight section of graph (equation 6).

    double v = 0.9;
    double icalc = iv(v, vt, 249., re, isat);

    std::cout << "voltage = " << v << ", current = " << icalc << ", " << log(icalc) << std::endl; // voltage = 0.9, current = 0.00108485, -6.82631
 //] [/lambert_w_diode_example_1]
  }
  catch (std::exception& ex)
  {
    std::cout << ex.what() << std::endl;
  }
}  // int main()

/*
   Output:
//[lambert_w_output_1
   Lambert W diode current example.
   V thermal 0.0257025
   Isat = 2.5e-14
   nu * vt / rsat = 0.000103099
   isat * rsat / (nu * vt) = 2.42486e-10
   (v + isat * rsat) / (nu * vt) = 35.016
   exp(eterm) = 1.61167e+15
   w0 = 10.5225
   voltage = 0.9, current = 0.00108485, -6.82631

//] [/lambert_w_output_1]
*/

