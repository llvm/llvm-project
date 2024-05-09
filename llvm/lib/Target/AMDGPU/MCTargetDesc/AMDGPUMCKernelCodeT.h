//===--- AMDGPUMCKernelCodeT.h --------------------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// MC layer struct for amd_kernel_code_t, provides MCExpr functionality where
/// required.
///
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUMCKERNELCODET_H
#define LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUMCKERNELCODET_H

#include "AMDKernelCodeT.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ArrayRef.h"

namespace llvm {
class MCAsmParser;
class MCContext;
class MCExpr;
class MCStreamer;
class MCSubtargetInfo;
class raw_ostream;
namespace AMDGPU {

struct MCKernelCodeT {
  MCKernelCodeT() = default;

  amd_kernel_code_t KernelCode;
  const MCExpr *compute_pgm_resource1_registers = nullptr;
  const MCExpr *compute_pgm_resource2_registers = nullptr;

  // Duplicated fields, but uses MCExpr instead.
  // Name has to be the same as the ones used in AMDKernelCodeTInfo.h.
  const MCExpr *is_dynamic_callstack = nullptr;
  const MCExpr *wavefront_sgpr_count = nullptr;
  const MCExpr *workitem_vgpr_count = nullptr;
  const MCExpr *workitem_private_segment_byte_size = nullptr;

  void initDefault(const MCSubtargetInfo *STI, MCContext &Ctx);
  void validate(const MCSubtargetInfo *STI, MCContext &Ctx);

  const MCExpr *&getMCExprForIndex(int Index);

  bool ParseKernelCodeT(StringRef ID, MCAsmParser &MCParser, raw_ostream &Err);
  void EmitKernelCodeT(raw_ostream &OS, const char *tab, MCContext &Ctx);
  void EmitKernelCodeT(MCStreamer &OS, MCContext &Ctx);
};

} // end namespace AMDGPU
} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUMCKERNELCODET_H
