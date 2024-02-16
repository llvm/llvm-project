//===- AMDGPUMCExpr.h - AMDGPU specific MC expression classes ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUMCEXPR_H
#define LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUMCEXPR_H

#include "llvm/MC/MCExpr.h"

namespace llvm {

class AMDGPUMCExpr : public MCTargetExpr {
  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Target;
  }
  void fixELFSymbolsInTLSFixups(MCAssembler &) const override{};
};

class AMDGPUVariadicMCExpr : public AMDGPUMCExpr {
public:
  enum AMDGPUVariadicKind { AGVK_None, AGVK_Or, AGVK_Max };

private:
  AMDGPUVariadicKind Kind;
  std::vector<const MCExpr *> Args;

  AMDGPUVariadicMCExpr(AMDGPUVariadicKind Kind,
                       const std::vector<const MCExpr *> &Args)
      : Kind(Kind), Args(Args) {
    assert(Args.size() >= 1 && "Can't take the maximum of 0 expressions.");
  }

public:
  static const AMDGPUVariadicMCExpr *
  create(AMDGPUVariadicKind Kind, const std::vector<const MCExpr *> &Args,
         MCContext &Ctx);

  static const AMDGPUVariadicMCExpr *
  createOr(const std::vector<const MCExpr *> &Args, MCContext &Ctx) {
    return create(AMDGPUVariadicKind::AGVK_Or, Args, Ctx);
  }

  static const AMDGPUVariadicMCExpr *
  createMax(const std::vector<const MCExpr *> &Args, MCContext &Ctx) {
    return create(AMDGPUVariadicKind::AGVK_Max, Args, Ctx);
  }

  AMDGPUVariadicKind getKind() const { return Kind; }
  const MCExpr *getSubExpr(size_t index) const;

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
  bool evaluateAsRelocatableImpl(MCValue &Res, const MCAsmLayout *Layout,
                                 const MCFixup *Fixup) const override;
  void visitUsedExpr(MCStreamer &Streamer) const override;
  MCFragment *findAssociatedFragment() const override;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUMCEXPR_H