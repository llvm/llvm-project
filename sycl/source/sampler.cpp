//==------------------- sampler.cpp - SYCL standard header file ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/sampler.hpp>

namespace cl {
namespace sycl {
sampler::sampler(coordinate_normalization_mode normalizationMode,
                 addressing_mode addressingMode, filtering_mode filteringMode)
    : impl(std::make_shared<detail::sampler_impl>(
          normalizationMode, addressingMode, filteringMode)) {}

sampler::sampler(cl_sampler clSampler, const context &syclContext)
    : impl(std::make_shared<detail::sampler_impl>(clSampler, syclContext)) {}

addressing_mode sampler::get_addressing_mode() const {
  return impl->get_addressing_mode();
}

filtering_mode sampler::get_filtering_mode() const {
  return impl->get_filtering_mode();
}

coordinate_normalization_mode
sampler::get_coordinate_normalization_mode() const {
  return impl->get_coordinate_normalization_mode();
}

bool sampler::operator==(const sampler &rhs) const {
  return (impl == rhs.impl);
}

bool sampler::operator!=(const sampler &rhs) const {
  return !(impl == rhs.impl);
}

} // namespace sycl
} // namespace cl
