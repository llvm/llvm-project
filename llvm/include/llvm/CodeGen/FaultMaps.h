//===- FaultMaps.h - The "FaultMaps" section --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_FAULTMAPS_H
#define LLVM_CODEGEN_FAULTMAPS_H

#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Compiler.h"
#include <map>
#include <vector>

namespace llvm {

class AsmPrinter;
class MCExpr;

class FaultMaps {
public:
  enum FaultKind {
    FaultingLoad = 1,
    FaultingLoadStore,
    FaultingStore,
    FaultKindMax
  };

  LLVM_ABI explicit FaultMaps(AsmPrinter &AP);

  LLVM_ABI static const char *faultTypeToString(FaultKind);

  LLVM_ABI void recordFaultingOp(FaultKind FaultTy,
                                 const MCSymbol *FaultingLabel,
                                 const MCSymbol *HandlerLabel);
  LLVM_ABI void serializeToFaultMapSection();
  void reset() {
    FunctionInfos.clear();
  }

private:
  static const char *WFMP;

  struct FaultInfo {
    FaultKind Kind = FaultKindMax;
    const MCExpr *FaultingOffsetExpr = nullptr;
    const MCExpr *HandlerOffsetExpr = nullptr;

    FaultInfo() = default;

    explicit FaultInfo(FaultMaps::FaultKind Kind, const MCExpr *FaultingOffset,
                       const MCExpr *HandlerOffset)
        : Kind(Kind), FaultingOffsetExpr(FaultingOffset),
          HandlerOffsetExpr(HandlerOffset) {}
  };

  using FunctionFaultInfos = std::vector<FaultInfo>;

  // We'd like to keep a stable iteration order for FunctionInfos to help
  // FileCheck based testing.
  struct MCSymbolComparator {
    bool operator()(const MCSymbol *LHS, const MCSymbol *RHS) const {
      return LHS->getName() < RHS->getName();
    }
  };

  std::map<const MCSymbol *, FunctionFaultInfos, MCSymbolComparator>
      FunctionInfos;
  AsmPrinter &AP;

  void emitFunctionInfo(const MCSymbol *FnLabel, const FunctionFaultInfos &FFI);
};

} // end namespace llvm

#endif // LLVM_CODEGEN_FAULTMAPS_H
