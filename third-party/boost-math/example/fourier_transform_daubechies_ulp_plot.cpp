// boost-no-inspect
//  (C) Copyright Nick Thompson 2023.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/special_functions/fourier_transform_daubechies.hpp>
#include <boost/math/tools/ulps_plot.hpp>

using boost::math::fourier_transform_daubechies_scaling;
using boost::math::tools::ulps_plot;

template<int p>
void real_part() {
   auto phi_real_hi_acc = [](double omega) {
       auto z = fourier_transform_daubechies_scaling<double, p>(omega);
       return z.real();
   };

   auto phi_real_lo_acc = [](float omega) {
      auto z = fourier_transform_daubechies_scaling<float, p>(omega);
      return z.real();
   };
   auto plot = ulps_plot<decltype(phi_real_hi_acc), double, float>(phi_real_hi_acc, float(0.0), float(100.0), 20000);
   plot.ulp_envelope(false);
   plot.add_fn(phi_real_lo_acc);
   plot.clip(100);
   plot.title("Accuracy of 𝔑(𝓕[𝜙](ω)) with " + std::to_string(p) + " vanishing moments.");
   plot.write("real_ft_daub_scaling_"  + std::to_string(p) + ".svg");
 
}

template<int p>
void imaginary_part() {
   auto phi_imag_hi_acc = [](double omega) {
       auto z = fourier_transform_daubechies_scaling<double, p>(omega);
       return z.imag();
   };

   auto phi_imag_lo_acc = [](float omega) {
      auto z = fourier_transform_daubechies_scaling<float, p>(omega);
      return z.imag();
   };
   auto plot = ulps_plot<decltype(phi_imag_hi_acc), double, float>(phi_imag_hi_acc, float(0.0), float(100.0), 20000);
   plot.ulp_envelope(false);
   plot.add_fn(phi_imag_lo_acc);
   plot.clip(100);
   plot.title("Accuracy of 𝕴(𝓕[𝜙](ω)) with " + std::to_string(p) + " vanishing moments.");
   plot.write("imag_ft_daub_scaling_"  + std::to_string(p) + ".svg");
 
}


int main() {
   real_part<3>();
   imaginary_part<3>();
   real_part<6>();
   imaginary_part<6>();
   return 0;
}
