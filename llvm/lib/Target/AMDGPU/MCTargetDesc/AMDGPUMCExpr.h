//===- AMDGPUMCExpr.h - AMDGPU specific MC expression classes ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUMCEXPR_H
#define LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUMCEXPR_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCExpr.h"

namespace llvm {

/// AMDGPU target specific variadic MCExpr operations.
///
/// Takes in a minimum of 1 argument to be used with an operation. The supported
/// operations are:
///   - (bitwise) or
///   - max
///
/// \note If the 'or'/'max' operations are provided only a single argument, the
/// operation will act as a no-op and simply resolve as the provided argument.
///
class AMDGPUVariadicMCExpr : public MCTargetExpr {
public:
  enum VariadicKind { AGVK_None, AGVK_Or, AGVK_Max };

private:
  VariadicKind Kind;
  MCContext &Ctx;
  const MCExpr **RawArgs;
  ArrayRef<const MCExpr *> Args;

  AMDGPUVariadicMCExpr(VariadicKind Kind, ArrayRef<const MCExpr *> Args,
                       MCContext &Ctx);
  ~AMDGPUVariadicMCExpr();

public:
  static const AMDGPUVariadicMCExpr *
  create(VariadicKind Kind, ArrayRef<const MCExpr *> Args, MCContext &Ctx);

  static const AMDGPUVariadicMCExpr *createOr(ArrayRef<const MCExpr *> Args,
                                              MCContext &Ctx) {
    return create(VariadicKind::AGVK_Or, Args, Ctx);
  }

  static const AMDGPUVariadicMCExpr *createMax(ArrayRef<const MCExpr *> Args,
                                               MCContext &Ctx) {
    return create(VariadicKind::AGVK_Max, Args, Ctx);
  }

  VariadicKind getKind() const { return Kind; }
  const MCExpr *getSubExpr(size_t Index) const;

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
  bool evaluateAsRelocatableImpl(MCValue &Res, const MCAsmLayout *Layout,
                                 const MCFixup *Fixup) const override;
  void visitUsedExpr(MCStreamer &Streamer) const override;
  MCFragment *findAssociatedFragment() const override;
  void fixELFSymbolsInTLSFixups(MCAssembler &) const override{};

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Target;
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUMCEXPR_H
