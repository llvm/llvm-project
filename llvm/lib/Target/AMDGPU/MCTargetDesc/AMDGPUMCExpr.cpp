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
#include <optional>

using namespace llvm;

AMDGPUVariadicMCExpr::AMDGPUVariadicMCExpr(VariadicKind Kind,
                                           ArrayRef<const MCExpr *> Args,
                                           MCContext &Ctx)
    : Kind(Kind), Ctx(Ctx) {
  assert(Args.size() >= 1 && "Needs a minimum of one expression.");
  assert(Kind != AGVK_None &&
         "Cannot construct AMDGPUVariadicMCExpr of kind none.");

  // Allocating the variadic arguments through the same allocation mechanism
  // that the object itself is allocated with so they end up in the same memory.
  //
  // Will result in an asan failure if allocated on the heap through standard
  // allocation (e.g., through SmallVector's grow).
  RawArgs = static_cast<const MCExpr **>(
      Ctx.allocate(sizeof(const MCExpr *) * Args.size()));
  std::uninitialized_copy(Args.begin(), Args.end(), RawArgs);
  this->Args = ArrayRef<const MCExpr *>(RawArgs, Args.size());
}

AMDGPUVariadicMCExpr::~AMDGPUVariadicMCExpr() { Ctx.deallocate(RawArgs); }

const AMDGPUVariadicMCExpr *
AMDGPUVariadicMCExpr::create(VariadicKind Kind, ArrayRef<const MCExpr *> Args,
                             MCContext &Ctx) {
  return new (Ctx) AMDGPUVariadicMCExpr(Kind, Args, Ctx);
}

const MCExpr *AMDGPUVariadicMCExpr::getSubExpr(size_t Index) const {
  assert(Index < Args.size() &&
         "Indexing out of bounds AMDGPUVariadicMCExpr sub-expr");
  return Args[Index];
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
  OS << ')';
}

static int64_t op(AMDGPUVariadicMCExpr::VariadicKind Kind, int64_t Arg1,
                  int64_t Arg2) {
  switch (Kind) {
  default:
    llvm_unreachable("Unknown AMDGPUVariadicMCExpr kind.");
  case AMDGPUVariadicMCExpr::AGVK_Max:
    return std::max(Arg1, Arg2);
  case AMDGPUVariadicMCExpr::AGVK_Or:
    return Arg1 | Arg2;
  }
}

bool AMDGPUVariadicMCExpr::evaluateAsRelocatableImpl(
    MCValue &Res, const MCAsmLayout *Layout, const MCFixup *Fixup) const {
  std::optional<int64_t> Total;

  for (const MCExpr *Arg : Args) {
    MCValue ArgRes;
    if (!Arg->evaluateAsRelocatable(ArgRes, Layout, Fixup) ||
        !ArgRes.isAbsolute())
      return false;

    if (!Total.has_value())
      Total = ArgRes.getConstant();
    Total = op(Kind, *Total, ArgRes.getConstant());
  }

  Res = MCValue::get(*Total);
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
