//===- AMDGPUMCExpr.cpp - AMDGPU specific MC expression classes -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUMCExpr.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

const AMDGPUVariadicMCExpr *
AMDGPUVariadicMCExpr::create(AMDGPUVariadicKind Kind,
                             ArrayRef<const MCExpr *> Args, MCContext &Ctx) {
  return new (Ctx) AMDGPUVariadicMCExpr(Kind, Args);
}

const MCExpr *AMDGPUVariadicMCExpr::getSubExpr(size_t index) const {
  assert(index < Args.size() &&
         "Indexing out of bounds AMDGPUVariadicMCExpr sub-expr");
  return Args[index];
}

void AMDGPUVariadicMCExpr::printImpl(raw_ostream &OS,
                                     const MCAsmInfo *MAI) const {
  switch (Kind) {
  default:
    llvm_unreachable("Unknown AMDGPUVariadicMCExpr kind.");
  case AGVK_Or:
    OS << "or(";
    break;
  case AGVK_Max:
    OS << "max(";
    break;
  }
  for (auto It = Args.begin(); It != Args.end(); ++It) {
    (*It)->print(OS, MAI, /*InParens=*/false);
    if ((It + 1) != Args.end())
      OS << ", ";
  }
  OS << ")";
}

bool AMDGPUVariadicMCExpr::evaluateAsRelocatableImpl(
    MCValue &Res, const MCAsmLayout *Layout, const MCFixup *Fixup) const {
  int64_t Total = INT64_MIN;

  auto Op = [this](int64_t Arg1, int64_t Arg2) -> int64_t {
    switch (Kind) {
    default:
      llvm_unreachable("Unknown AMDGPUVariadicMCExpr kind.");
    case AGVK_Max:
      return std::max(Arg1, Arg2);
    case AGVK_Or:
      return Arg1 || Arg2;
    }
  };

  for (const MCExpr *Arg : Args) {
    MCValue ArgRes;
    if (!Arg->evaluateAsRelocatable(ArgRes, Layout, Fixup) ||
        !ArgRes.isAbsolute())
      return false;

    if (Total == INT64_MIN)
      Total = ArgRes.getConstant();
    Total = Op(Total, ArgRes.getConstant());
  }

  Res = MCValue::get(Total);
  return true;
}

void AMDGPUVariadicMCExpr::visitUsedExpr(MCStreamer &Streamer) const {
  for (const MCExpr *Arg : Args)
    Streamer.visitUsedExpr(*Arg);
}

MCFragment *AMDGPUVariadicMCExpr::findAssociatedFragment() const {
  for (const MCExpr *Arg : Args) {
    if (Arg->findAssociatedFragment())
      return Arg->findAssociatedFragment();
  }
  return nullptr;
}
