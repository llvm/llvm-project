//===--- IntelGpuXe2.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// Xe2 uArch definition. Xe2 is the second generation of Intel Xe GPUs.
// This file defines the uArch details for Xe2 and its derived architectures.
// This includes Ponte Vecchio (PVC) and Battlemage (BMG) architectures.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_XEGPU_UARCH_INTELGPUXE2_H
#define MLIR_DIALECT_XEGPU_UARCH_INTELGPUXE2_H

#include "mlir/Dialect/XeGPU/uArch/uArchBase.h"

namespace mlir {
namespace xegpu {
namespace uArch {

struct Xe2 : public uArch {
  Xe2(Kind kind, llvm::ArrayRef<const Instruction *> instructionRegistry)
      : uArch(kind, instructionRegistry) {}
  int getSubgroupSize() const override { return 16; }
  unsigned getGeneralPackedFormatBitSize() const override { return 32; }

  static bool classof(const uArch *u) {
    return u->getKind() >= Kind::Xe2_First && u->getKind() <= Kind::Xe2_Last;
  }
};

//===----------------------------------------------------------------------===//
// uArch instances
//
// PVC and BMG share the same Khronos-extension instruction set.
//===----------------------------------------------------------------------===//

namespace detail {
inline llvm::ArrayRef<const Instruction *> getXe2InstructionRegistry() {
  static const SubgroupMatrixMultiplyAcc dpasInst{16, 32};
  static const Subgroup2DBlockLoadInstruction loadNdInst;
  static const Subgroup2DBlockStoreInstruction storeNdInst;
  static const Subgroup2DBlockPrefetchInstruction prefetchNdInst;
  static const StoreScatterInstruction storeScatterInst;
  static const LoadGatherInstruction loadGatherInst;
  static const Instruction *arr[] = {&dpasInst,         &loadNdInst,
                                     &storeNdInst,      &prefetchNdInst,
                                     &storeScatterInst, &loadGatherInst};
  return arr;
}
} // namespace detail

struct PVCuArch final : public Xe2 {
  PVCuArch() : Xe2(Kind::PVC, detail::getXe2InstructionRegistry()) {}
  static bool classof(const uArch *u) { return u->getKind() == Kind::PVC; }
  static const uArch *getInstance() {
    static const PVCuArch instance;
    return &instance;
  }
};

struct BMGuArch final : public Xe2 {
  BMGuArch() : Xe2(Kind::BMG, detail::getXe2InstructionRegistry()) {}
  static bool classof(const uArch *u) { return u->getKind() == Kind::BMG; }
  static const uArch *getInstance() {
    static const BMGuArch instance;
    return &instance;
  }
};

} // namespace uArch
} // namespace xegpu
} // namespace mlir

#endif // MLIR_DIALECT_XEGPU_UARCH_INTELGPUXE2_H
