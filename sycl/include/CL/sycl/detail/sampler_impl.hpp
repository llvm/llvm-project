//==----------------- sampler_impl.hpp - SYCL standard header file ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_types.hpp>
#include <CL/sycl/context.hpp>

#include <unordered_map>

namespace cl {
namespace sycl {

enum class addressing_mode : unsigned int;
enum class filtering_mode : unsigned int;
enum class coordinate_normalization_mode : unsigned int;

namespace detail {
class sampler_impl {
public:
#ifdef __SYCL_DEVICE_ONLY__
  __ocl_sampler_t m_Sampler;
  sampler_impl(__ocl_sampler_t Sampler) : m_Sampler(Sampler) {}
#else
  std::unordered_map<context, cl_sampler> m_contextToSampler;

private:
  coordinate_normalization_mode m_CoordNormMode;
  addressing_mode m_AddrMode;
  filtering_mode m_FiltMode;

public:
  sampler_impl(coordinate_normalization_mode normalizationMode,
               addressing_mode addressingMode, filtering_mode filteringMode);

  sampler_impl(cl_sampler clSampler, const context &syclContext);

  addressing_mode get_addressing_mode() const;

  filtering_mode get_filtering_mode() const;

  coordinate_normalization_mode get_coordinate_normalization_mode() const;

  cl_sampler getOrCreateSampler(const context &Context);
#endif

#ifdef __SYCL_DEVICE_ONLY__
  ~sampler_impl() = default;
#else
  ~sampler_impl();
#endif
};

} // namespace detail
} // namespace sycl
} // namespace cl
