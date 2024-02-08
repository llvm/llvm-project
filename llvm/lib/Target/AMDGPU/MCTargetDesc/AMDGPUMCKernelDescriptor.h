//===--- AMDGPUMCKernelDescriptor.h ---------------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// AMDHSA kernel descriptor MCExpr struct for use in MC layer. Uses
/// AMDHSAKernelDescriptor.h for sizes and constants.
///
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUMCKERNELDESCRIPTOR_H
#define LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUMCKERNELDESCRIPTOR_H

#include "llvm/Support/AMDHSAKernelDescriptor.h"

namespace llvm {
class MCExpr;
class MCContext;
namespace AMDGPU {

struct MCKernelDescriptor {
  const MCExpr *group_segment_fixed_size;
  const MCExpr *private_segment_fixed_size;
  const MCExpr *kernarg_size;
  const MCExpr *compute_pgm_rsrc3;
  const MCExpr *compute_pgm_rsrc1;
  const MCExpr *compute_pgm_rsrc2;
  const MCExpr *kernel_code_properties;
  const MCExpr *kernarg_preload;

  static void bits_set(const MCExpr *&Dst, const MCExpr *Value, uint32_t Shift,
                       uint32_t Mask, MCContext &Ctx);
  static const MCExpr *bits_get(const MCExpr *Src, uint32_t Shift,
                                uint32_t Mask, MCContext &Ctx);
};

enum : uint32_t {
  SIZEOF_GROUP_SEGMENT_FIXED_SIZE =
      sizeof(amdhsa::kernel_descriptor_t::group_segment_fixed_size),
  SIZEOF_PRIVATE_SEGMENT_FIXED_SIZE =
      sizeof(amdhsa::kernel_descriptor_t::private_segment_fixed_size),
  SIZEOF_KERNARG_SIZE = sizeof(amdhsa::kernel_descriptor_t::kernarg_size),
  SIZEOF_RESERVED0 = sizeof(amdhsa::kernel_descriptor_t::reserved0),
  SIZEOF_KERNEL_CODE_ENTRY_BYTE_OFFSET =
      sizeof(amdhsa::kernel_descriptor_t::kernel_code_entry_byte_offset),
  SIZEOF_RESERVED1 = sizeof(amdhsa::kernel_descriptor_t::reserved1),
  SIZEOF_COMPUTE_PGM_RSRC3 =
      sizeof(amdhsa::kernel_descriptor_t::compute_pgm_rsrc3),
  SIZEOF_COMPUTE_PGM_RSRC1 =
      sizeof(amdhsa::kernel_descriptor_t::compute_pgm_rsrc1),
  SIZEOF_COMPUTE_PGM_RSRC2 =
      sizeof(amdhsa::kernel_descriptor_t::compute_pgm_rsrc2),
  SIZEOF_KERNEL_CODE_PROPERTIES =
      sizeof(amdhsa::kernel_descriptor_t::kernel_code_properties),
  SIZEOF_KERNARG_PRELOAD = sizeof(amdhsa::kernel_descriptor_t::kernarg_preload),
  SIZEOF_RESERVED3 = sizeof(amdhsa::kernel_descriptor_t::reserved3),
  SIZEOF_KERNEL_DESCRIPTOR = sizeof(amdhsa::kernel_descriptor_t)
};

} // end namespace AMDGPU
} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUMCKERNELDESCRIPTOR_H
