//===- AMDGPUKernelCodeTUtils.h - helpers for amd_kernel_code_t -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file AMDKernelCodeTUtils.h
/// MC layer struct for AMDGPUMCKernelCodeT, provides MCExpr functionality where
/// required.
///
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUMCKERNELCODET_H
#define LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUMCKERNELCODET_H

#include "AMDKernelCodeT.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class MCAsmParser;
class MCContext;
class MCExpr;
class MCStreamer;
class MCSubtargetInfo;
class raw_ostream;
namespace AMDGPU {

struct AMDGPUMCKernelCodeT {
  AMDGPUMCKernelCodeT();

  uint32_t amd_kernel_code_version_major;
  uint32_t amd_kernel_code_version_minor;
  uint16_t amd_machine_kind;
  uint16_t amd_machine_version_major;
  uint16_t amd_machine_version_minor;
  uint16_t amd_machine_version_stepping;
  int64_t kernel_code_entry_byte_offset;
  int64_t kernel_code_prefetch_byte_offset;
  uint64_t kernel_code_prefetch_byte_size;
  uint64_t reserved0;
  uint64_t compute_pgm_resource_registers;
  uint32_t code_properties;
  uint32_t workgroup_group_segment_byte_size;
  uint32_t gds_segment_byte_size;
  uint64_t kernarg_segment_byte_size;
  uint32_t workgroup_fbarrier_count;
  uint16_t reserved_vgpr_first;
  uint16_t reserved_vgpr_count;
  uint16_t reserved_sgpr_first;
  uint16_t reserved_sgpr_count;
  uint16_t debug_wavefront_private_segment_offset_sgpr;
  uint16_t debug_private_segment_buffer_sgpr;
  uint8_t kernarg_segment_alignment;
  uint8_t group_segment_alignment;
  uint8_t private_segment_alignment;
  uint8_t wavefront_size;
  int32_t call_convention;
  uint8_t reserved3[12];
  uint64_t runtime_loader_kernel_symbol;
  uint64_t control_directives[16];

  const MCExpr *compute_pgm_resource1_registers = nullptr;
  const MCExpr *compute_pgm_resource2_registers = nullptr;

  // Duplicated fields, but uses MCExpr instead.
  // Name has to be the same as the ones used in AMDKernelCodeTInfo.h.
  const MCExpr *is_dynamic_callstack = nullptr;
  const MCExpr *wavefront_sgpr_count = nullptr;
  const MCExpr *workitem_vgpr_count = nullptr;
  const MCExpr *workitem_private_segment_byte_size = nullptr;

  void initDefault(const MCSubtargetInfo *STI, MCContext &Ctx,
                   bool InitMCExpr = true);
  void validate(const MCSubtargetInfo *STI, MCContext &Ctx);

  const MCExpr *&getMCExprForIndex(int Index);

  bool ParseKernelCodeT(StringRef ID, MCAsmParser &MCParser, raw_ostream &Err);
  void EmitKernelCodeT(raw_ostream &OS, MCContext &Ctx);
  void EmitKernelCodeT(MCStreamer &OS, MCContext &Ctx);
};

} // end namespace AMDGPU
} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUMCKERNELCODET_H
