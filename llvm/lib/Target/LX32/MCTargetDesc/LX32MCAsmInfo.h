//===-- LX32MCAsmInfo.h - LX32 MC Assembler Information ------------------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// This file defines the LX32-specific MCAsmInfo class declaration.
// It is organized into the following sections:
//
//   Section 0 — Forward declarations
//   Section 1 — LX32MCAsmInfo declaration
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_LX32_MCTARGETDESC_LX32MCASMINFO_H
#define LLVM_LIB_TARGET_LX32_MCTARGETDESC_LX32MCASMINFO_H

#include "llvm/MC/MCAsmInfoELF.h"

namespace llvm {
class Triple;
} // namespace llvm

//===----------------------------------------------------------------------===//
// Section 1 — LX32MCAsmInfo declaration
//===----------------------------------------------------------------------===//

class LX32MCAsmInfo : public llvm::MCAsmInfoELF {
  void anchor() override;
public:
  // Build MC assembler defaults for the selected target triple.
  explicit LX32MCAsmInfo(const llvm::Triple &TargetTriple);
};

#endif // LLVM_LIB_TARGET_LX32_MCTARGETDESC_LX32MCASMINFO_H
