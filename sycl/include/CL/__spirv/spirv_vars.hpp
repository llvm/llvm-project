//==---------- spirv_vars.hpp --- SPIRV variables -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===-------------------------------------------------------------------=== //

#pragma once

#ifdef __SYCL_DEVICE_ONLY__

namespace cl {
namespace __spirv {
typedef size_t size_t_vec __attribute__((ext_vector_type(3)));

extern const __constant size_t_vec VarGlobalSize;
extern const __constant size_t_vec VarGlobalInvocationId;
extern const __constant size_t_vec VarWorkgroupSize;
extern const __constant size_t_vec VarLocalInvocationId;
extern const __constant size_t_vec VarWorkgroupId;
extern const __constant size_t_vec VarGlobalOffset;

#define DEFINE_INT_ID_TO_XYZ_CONVERTER(POSTFIX)                                \
  template <int ID> static size_t get##POSTFIX();                              \
  template <> static size_t get##POSTFIX<0>() { return Var##POSTFIX.x; }       \
  template <> static size_t get##POSTFIX<1>() { return Var##POSTFIX.y; }       \
  template <> static size_t get##POSTFIX<2>() { return Var##POSTFIX.z; }

DEFINE_INT_ID_TO_XYZ_CONVERTER(GlobalSize);
DEFINE_INT_ID_TO_XYZ_CONVERTER(GlobalInvocationId)
DEFINE_INT_ID_TO_XYZ_CONVERTER(WorkgroupSize)
DEFINE_INT_ID_TO_XYZ_CONVERTER(LocalInvocationId)
DEFINE_INT_ID_TO_XYZ_CONVERTER(WorkgroupId)
DEFINE_INT_ID_TO_XYZ_CONVERTER(GlobalOffset)

#undef DEFINE_INT_ID_TO_XYZ_CONVERTER

extern const __constant uint32_t VarSubgroupSize;
extern const __constant uint32_t VarSubgroupMaxSize;
extern const __constant uint32_t VarNumSubgroups;
extern const __constant uint32_t VarNumEnqueuedSubgroups;
extern const __constant uint32_t VarSubgroupId;
extern const __constant uint32_t VarSubgroupLocalInvocationId;

} // namespace __spirv
} // namespace cl
#endif // __SYCL_DEVICE_ONLY__
