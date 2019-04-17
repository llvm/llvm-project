//==----------------- sampler_impl.cpp - SYCL standard header file ---------==//
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
      m_FiltMode(filteringMode), m_ReleaseSampler(false) {}

sampler_impl::sampler_impl(cl_sampler clSampler, const context &syclContext)
    : m_Sampler(clSampler), m_SyclContext(syclContext), m_ReleaseSampler(true) {

  m_contextToSampler[syclContext] = m_Sampler;
  CHECK_OCL_CODE(clRetainSampler(m_Sampler));
  CHECK_OCL_CODE(clGetSamplerInfo(m_Sampler, CL_SAMPLER_NORMALIZED_COORDS,
                                  sizeof(cl_bool), &m_CoordNormMode, nullptr));
  CHECK_OCL_CODE(clGetSamplerInfo(m_Sampler, CL_SAMPLER_ADDRESSING_MODE,
                                  sizeof(cl_addressing_mode), &m_AddrMode,
                                  nullptr));
  CHECK_OCL_CODE(clGetSamplerInfo(m_Sampler, CL_SAMPLER_FILTER_MODE,
                                  sizeof(cl_filter_mode), &m_FiltMode,
                                  nullptr));
}

sampler_impl::~sampler_impl() {
  if (m_ReleaseSampler)
    CHECK_OCL_CODE(clReleaseSampler(m_Sampler));
}

cl_sampler sampler_impl::getOrCreateSampler(const context &Context) {
  cl_int errcode_ret = CL_SUCCESS;
  if (m_contextToSampler[Context])
    return m_contextToSampler[Context];

  m_contextToSampler[Context] =
      clCreateSampler(Context.get(), static_cast<cl_bool>(m_CoordNormMode),
                      static_cast<cl_addressing_mode>(m_AddrMode),
                      static_cast<cl_filter_mode>(m_FiltMode), &errcode_ret);
  CHECK_OCL_CODE(errcode_ret);
  m_ReleaseSampler = true;
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
