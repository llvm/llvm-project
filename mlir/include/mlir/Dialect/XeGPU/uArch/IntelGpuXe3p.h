//===--- IntelGpuXe3p.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// Xe3p uArch definition. Xe3p is a derivative of the third generation of Intel
// Xe GPUs and includes Crescent Island (CRI). Note that plain Xe3 exposes no
// XeGPU-relevant differences over Xe2 and is therefore covered by
// IntelGpuXe2.h; only the Xe3p derivative is modelled here.
//
// The base instruction set comes from the shared Khronos OpenCL extensions
// defined in uArchBase.h; on top of those Xe3p adds the scaled DPAS (MXFP)
// extension and relaxes the 2D block memory restrictions. Subclass and override
// here only when an Xe3p-specific instruction diverges from the SPIRV defaults.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_XEGPU_UARCH_INTELGPUXE3P_H
#define MLIR_DIALECT_XEGPU_UARCH_INTELGPUXE3P_H

#include "mlir/Dialect/XeGPU/uArch/uArchBase.h"

namespace mlir {
namespace xegpu {
namespace uArch {

struct Xe3p : public uArch {
  Xe3p(Kind kind, llvm::ArrayRef<const Instruction *> instructionRegistry)
      : uArch(kind, instructionRegistry) {}
  int getSubgroupSize() const override { return 16; }
  unsigned getGeneralPackedFormatBitSize() const override { return 32; }

  static bool classof(const uArch *u) {
    return u->getKind() >= Kind::Xe3p_First && u->getKind() <= Kind::Xe3p_Last;
  }
};

//===----------------------------------------------------------------------===//
// uArch instances
//===----------------------------------------------------------------------===//

namespace detail {
// Restrictions Xe3p places on the 2D memory region accessed by the subgroup 2D
// block load / store / prefetch instructions.
//
// Xe3p relaxes every one of these relative to Xe2, which requires a 64 byte
// aligned base address, a base width and pitch of at least 32 bytes, and a
// pitch that is a multiple of 16 bytes. On Xe3p the base address, width and
// pitch all only need to be a multiple of 4 bytes. There is no minimum size
// restriction on the width or the pitch, so their minimum is just that 4 byte
// granularity -- a zero-sized surface would be meaningless.
inline constexpr BlockIOMemoryRestrictions kXe3pBlockIORestrictions = {
    /*baseAddressAlignmentBytes=*/4,
    /*minBaseWidthBytes=*/4,
    /*baseWidthAlignmentBytes=*/4,
    /*minBasePitchBytes=*/4,
    /*basePitchAlignmentBytes=*/4,
};

inline llvm::ArrayRef<const Instruction *> getXe3pInstructionRegistry() {
  static const SubgroupMatrixMultiplyAcc dpasInst{16, 32};
  static const SubgroupScaledMatrixMultiplyAcc dpasMxInst{16, 32};
  static const Subgroup2DBlockLoadInstruction loadNdInst{
      kXe3pBlockIORestrictions};
  static const Subgroup2DBlockStoreInstruction storeNdInst{
      kXe3pBlockIORestrictions};
  static const Subgroup2DBlockPrefetchInstruction prefetchNdInst{
      kXe3pBlockIORestrictions};
  static const StoreScatterInstruction storeScatterInst;
  static const LoadGatherInstruction loadGatherInst;
  static const Instruction *arr[] = {
      &dpasInst,       &dpasMxInst,       &loadNdInst,    &storeNdInst,
      &prefetchNdInst, &storeScatterInst, &loadGatherInst};
  return arr;
}
} // namespace detail

struct CRIuArch final : public Xe3p {
  CRIuArch() : Xe3p(Kind::CRI, detail::getXe3pInstructionRegistry()) {}
  static bool classof(const uArch *u) { return u->getKind() == Kind::CRI; }
  static const uArch *getInstance() {
    static const CRIuArch instance;
    return &instance;
  }
};

} // namespace uArch
} // namespace xegpu
} // namespace mlir

#endif // MLIR_DIALECT_XEGPU_UARCH_INTELGPUXE3P_H
