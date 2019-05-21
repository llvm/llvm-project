//==----------------- sampler_impl.cpp - SYCL sampler ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/sampler_impl.hpp>

namespace cl {
namespace sycl {
namespace detail {

sampler_impl::sampler_impl(coordinate_normalization_mode normalizationMode,
                           addressing_mode addressingMode,
                           filtering_mode filteringMode)
    : m_CoordNormMode(normalizationMode), m_AddrMode(addressingMode),
      m_FiltMode(filteringMode) {}

sampler_impl::sampler_impl(cl_sampler clSampler, const context &syclContext) {

  m_contextToSampler[syclContext] = clSampler;
  CHECK_OCL_CODE(clRetainSampler(clSampler));
  CHECK_OCL_CODE(clGetSamplerInfo(clSampler, CL_SAMPLER_NORMALIZED_COORDS,
                                  sizeof(cl_bool), &m_CoordNormMode, nullptr));
  CHECK_OCL_CODE(clGetSamplerInfo(clSampler, CL_SAMPLER_ADDRESSING_MODE,
                                  sizeof(cl_addressing_mode), &m_AddrMode,
                                  nullptr));
  CHECK_OCL_CODE(clGetSamplerInfo(clSampler, CL_SAMPLER_FILTER_MODE,
                                  sizeof(cl_filter_mode), &m_FiltMode,
                                  nullptr));
}

sampler_impl::~sampler_impl() {
  for (auto &Iter : m_contextToSampler) {
    // TODO replace CHECK_OCL_CODE_NO_EXC to CHECK_OCL_CODE and
    // TODO catch an exception and add it to the list of asynchronous exceptions
    CHECK_OCL_CODE_NO_EXC(clReleaseSampler(Iter.second));
  }
}

cl_sampler sampler_impl::getOrCreateSampler(const context &Context) {
  cl_int errcode_ret = CL_SUCCESS;
  if (m_contextToSampler[Context])
    return m_contextToSampler[Context];

#if CL_TARGET_OPENCL_VERSION > 120
  const cl_sampler_properties sprops[] = {
      CL_SAMPLER_NORMALIZED_COORDS,
      static_cast<cl_sampler_properties>(m_CoordNormMode),
      CL_SAMPLER_ADDRESSING_MODE,
      static_cast<cl_sampler_properties>(m_AddrMode),
      CL_SAMPLER_FILTER_MODE,
      static_cast<cl_sampler_properties>(m_FiltMode),
      0};
  m_contextToSampler[Context] =
      clCreateSamplerWithProperties(Context.get(), sprops, &errcode_ret);
#else
  m_contextToSampler[Context] =
      clCreateSampler(Context.get(), static_cast<cl_bool>(m_CoordNormMode),
                      static_cast<cl_addressing_mode>(m_AddrMode),
                      static_cast<cl_filter_mode>(m_FiltMode), &errcode_ret);
#endif
  CHECK_OCL_CODE(errcode_ret);
  return m_contextToSampler[Context];
}

addressing_mode sampler_impl::get_addressing_mode() const { return m_AddrMode; }

filtering_mode sampler_impl::get_filtering_mode() const { return m_FiltMode; }

coordinate_normalization_mode
sampler_impl::get_coordinate_normalization_mode() const {
  return m_CoordNormMode;
}

} // namespace detail
} // namespace sycl
} // namespace cl
