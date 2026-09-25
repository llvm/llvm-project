//===--- IntelGpuXe3.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// Xe3 uArch definition. Xe3 is the third generation of Intel Xe GPUs and
// includes Crescent Island (CRI). The base instruction set comes from the
// shared Khronos OpenCL extensions defined in uArchBase.h; Xe3 only adds the
// scaled DPAS extension. Subclass and override here only when a CRI-specific
// instruction diverges from the SPIRV defaults.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_XEGPU_UARCH_INTELGPUXE3_H
#define MLIR_DIALECT_XEGPU_UARCH_INTELGPUXE3_H

#include "mlir/Dialect/XeGPU/uArch/uArchBase.h"

namespace mlir {
namespace xegpu {
namespace uArch {

struct Xe3 : public uArch {
  Xe3(Kind kind, llvm::ArrayRef<const Instruction *> instructionRegistry)
      : uArch(kind, instructionRegistry) {}
  int getSubgroupSize() const override { return 16; }
  unsigned getGeneralPackedFormatBitSize() const override { return 32; }

  static bool classof(const uArch *u) {
    return u->getKind() >= Kind::Xe3_First && u->getKind() <= Kind::Xe3_Last;
  }
};

//===----------------------------------------------------------------------===//
// uArch instances
//===----------------------------------------------------------------------===//

struct CRIuArch final : public Xe3 {
  static llvm::ArrayRef<const Instruction *> getCriInstructionRegistry() {
    static const SubgroupMatrixMultiplyAcc dpasInst{16, 32};
    static const SubgroupScaledMatrixMultiplyAcc dpasMxInst{16, 32};
    static const Subgroup2DBlockLoadInstruction loadNdInst;
    static const Subgroup2DBlockStoreInstruction storeNdInst;
    static const Subgroup2DBlockPrefetchInstruction prefetchNdInst;
    static const StoreScatterInstruction storeScatterInst;
    static const LoadGatherInstruction loadGatherInst;
    static const Instruction *arr[] = {
        &dpasInst,       &dpasMxInst,       &loadNdInst,    &storeNdInst,
        &prefetchNdInst, &storeScatterInst, &loadGatherInst};
    return arr;
  }

  CRIuArch() : Xe3(Kind::CRI, getCriInstructionRegistry()) {}
  static bool classof(const uArch *u) { return u->getKind() == Kind::CRI; }
  static const uArch *getInstance() {
    static const CRIuArch instance;
    return &instance;
  }
};

} // namespace uArch
} // namespace xegpu
} // namespace mlir

#endif // MLIR_DIALECT_XEGPU_UARCH_INTELGPUXE3_H
