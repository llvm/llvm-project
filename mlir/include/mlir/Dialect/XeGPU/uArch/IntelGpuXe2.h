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
// Restrictions Xe2 places on the 2D memory region accessed by the subgroup 2D
// block load / store / prefetch instructions.
//
// These are the hardware requirements as implemented by the Intel Graphics
// Compiler, and may differ from the restrictions documented for the Khronos
// extension that the XeVM lowering emits calls to. That extension only states
// that behavior is undefined when its restrictions are not met; it does not
// require an implementation to reject such cases, so the compiler is free to
// support them.
//
// The base address must be cache-line (64 byte) aligned. The base width must be
// at least 32 bytes and a multiple of 4 bytes. The base pitch must be at least
// 32 bytes and a multiple of 16 bytes.
inline constexpr BlockIOMemoryRestrictions kXe2BlockIORestrictions = {
    /*baseAddressAlignmentBytes=*/64,
    /*minBaseWidthBytes=*/32,
    /*baseWidthAlignmentBytes=*/4,
    /*minBasePitchBytes=*/32,
    /*basePitchAlignmentBytes=*/16,
};

inline llvm::ArrayRef<const Instruction *> getXe2InstructionRegistry() {
  static const SubgroupMatrixMultiplyAcc dpasInst{16, 32};
  static const Subgroup2DBlockLoadInstruction loadNdInst{
      kXe2BlockIORestrictions};
  static const Subgroup2DBlockStoreInstruction storeNdInst{
      kXe2BlockIORestrictions};
  static const Subgroup2DBlockPrefetchInstruction prefetchNdInst{
      kXe2BlockIORestrictions};
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
