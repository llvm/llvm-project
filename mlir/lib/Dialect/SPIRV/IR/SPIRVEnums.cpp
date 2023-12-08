//===- SPIRVEnums.cpp - MLIR SPIR-V Enums ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the C/C++ enums from SPIR-V spec.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"

#include <iterator>

using namespace mlir;

// Pull in all enum utility function definitions
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.cpp.inc"

// Pull in all enum type availability query function definitions
#include "mlir/Dialect/SPIRV/IR/SPIRVEnumAvailability.cpp.inc"

//===----------------------------------------------------------------------===//
// Availability relationship
//===----------------------------------------------------------------------===//

ArrayRef<spirv::Extension> spirv::getImpliedExtensions(spirv::Version version) {
  // Note: the following lists are from "Appendix A: Changes" of the spec.

#define V_1_3_IMPLIED_EXTS                                                     \
  Extension::SPV_KHR_shader_draw_parameters, Extension::SPV_KHR_16bit_storage, \
      Extension::SPV_KHR_device_group, Extension::SPV_KHR_multiview,           \
      Extension::SPV_KHR_storage_buffer_storage_class,                         \
      Extension::SPV_KHR_variable_pointers

#define V_1_4_IMPLIED_EXTS                                                     \
  Extension::SPV_KHR_no_integer_wrap_decoration,                               \
      Extension::SPV_GOOGLE_decorate_string,                                   \
      Extension::SPV_GOOGLE_hlsl_functionality1,                               \
      Extension::SPV_KHR_float_controls

#define V_1_5_IMPLIED_EXTS                                                     \
  Extension::SPV_KHR_8bit_storage, Extension::SPV_EXT_descriptor_indexing,     \
      Extension::SPV_EXT_shader_viewport_index_layer,                          \
      Extension::SPV_EXT_physical_storage_buffer,                              \
      Extension::SPV_KHR_physical_storage_buffer,                              \
      Extension::SPV_KHR_vulkan_memory_model

#define V_1_6_IMPLIED_EXTS                                                     \
  Extension::SPV_KHR_non_semantic_info,                                        \
      Extension::SPV_KHR_integer_dot_product,                                  \
      Extension::SPV_KHR_terminate_invocation,                                 \
      Extension::SPV_EXT_demote_to_helper_invocation

  switch (version) {
  default:
    return {};
  case Version::V_1_3: {
    // The following manual ArrayRef constructor call is to satisfy GCC 5.
    static const Extension exts[] = {V_1_3_IMPLIED_EXTS};
    return exts;
  }
  case Version::V_1_4: {
    static const Extension exts[] = {V_1_3_IMPLIED_EXTS, V_1_4_IMPLIED_EXTS};
    return exts;
  }
  case Version::V_1_5: {
    static const Extension exts[] = {V_1_3_IMPLIED_EXTS, V_1_4_IMPLIED_EXTS,
                                     V_1_5_IMPLIED_EXTS};
    return exts;
  }
  case Version::V_1_6: {
    static const Extension exts[] = {V_1_3_IMPLIED_EXTS, V_1_4_IMPLIED_EXTS,
                                     V_1_5_IMPLIED_EXTS, V_1_6_IMPLIED_EXTS};
    return exts;
  }
  }

#undef V_1_6_IMPLIED_EXTS
#undef V_1_5_IMPLIED_EXTS
#undef V_1_4_IMPLIED_EXTS
#undef V_1_3_IMPLIED_EXTS
}

// Pull in utility function definition for implied capabilities
#include "mlir/Dialect/SPIRV/IR/SPIRVCapabilityImplication.inc"

SmallVector<spirv::Capability, 0>
spirv::getRecursiveImpliedCapabilities(spirv::Capability cap) {
  ArrayRef<spirv::Capability> directCaps = getDirectImpliedCapabilities(cap);
  SetVector<spirv::Capability, SmallVector<spirv::Capability, 0>> allCaps(
      directCaps.begin(), directCaps.end());

  // TODO: This is insufficient; find a better way to handle this
  // (e.g., using static lists) if this turns out to be a bottleneck.
  for (unsigned i = 0; i < allCaps.size(); ++i)
    for (Capability c : getDirectImpliedCapabilities(allCaps[i]))
      allCaps.insert(c);

  return allCaps.takeVector();
}
