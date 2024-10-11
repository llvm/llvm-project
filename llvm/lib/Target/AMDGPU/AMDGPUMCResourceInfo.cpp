//===- AMDGPUMCResourceInfo.cpp --- MC Resource Info ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief MC infrastructure to propagate the function level resource usage
/// info.
///
//===----------------------------------------------------------------------===//

#include "AMDGPUMCResourceInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"

using namespace llvm;

MCSymbol *MCResourceInfo::getSymbol(StringRef FuncName, ResourceInfoKind RIK,
                                    MCContext &OutContext) {
  auto GOCS = [FuncName, &OutContext](StringRef Suffix) {
    return OutContext.getOrCreateSymbol(FuncName + Twine(Suffix));
  };
  switch (RIK) {
  case RIK_NumVGPR:
    return GOCS(".num_vgpr");
  case RIK_NumAGPR:
    return GOCS(".num_agpr");
  case RIK_NumSGPR:
    return GOCS(".numbered_sgpr");
  case RIK_PrivateSegSize:
    return GOCS(".private_seg_size");
  case RIK_UsesVCC:
    return GOCS(".uses_vcc");
  case RIK_UsesFlatScratch:
    return GOCS(".uses_flat_scratch");
  case RIK_HasDynSizedStack:
    return GOCS(".has_dyn_sized_stack");
  case RIK_HasRecursion:
    return GOCS(".has_recursion");
  case RIK_HasIndirectCall:
    return GOCS(".has_indirect_call");
  }
  llvm_unreachable("Unexpected ResourceInfoKind.");
}

const MCExpr *MCResourceInfo::getSymRefExpr(StringRef FuncName,
                                            ResourceInfoKind RIK,
                                            MCContext &Ctx) {
  return MCSymbolRefExpr::create(getSymbol(FuncName, RIK, Ctx), Ctx);
}

void MCResourceInfo::assignMaxRegs(MCContext &OutContext) {
  // Assign expression to get the max register use to the max_num_Xgpr symbol.
  MCSymbol *MaxVGPRSym = getMaxVGPRSymbol(OutContext);
  MCSymbol *MaxAGPRSym = getMaxAGPRSymbol(OutContext);
  MCSymbol *MaxSGPRSym = getMaxSGPRSymbol(OutContext);

  auto assignMaxRegSym = [&OutContext](MCSymbol *Sym, int32_t RegCount) {
    const MCExpr *MaxExpr = MCConstantExpr::create(RegCount, OutContext);
    Sym->setVariableValue(MaxExpr);
  };

  assignMaxRegSym(MaxVGPRSym, MaxVGPR);
  assignMaxRegSym(MaxAGPRSym, MaxAGPR);
  assignMaxRegSym(MaxSGPRSym, MaxSGPR);
}

void MCResourceInfo::reset() { *this = MCResourceInfo(); }

void MCResourceInfo::finalize(MCContext &OutContext) {
  assert(!Finalized && "Cannot finalize ResourceInfo again.");
  Finalized = true;
  assignMaxRegs(OutContext);
}

MCSymbol *MCResourceInfo::getMaxVGPRSymbol(MCContext &OutContext) {
  return OutContext.getOrCreateSymbol("amdgpu.max_num_vgpr");
}

MCSymbol *MCResourceInfo::getMaxAGPRSymbol(MCContext &OutContext) {
  return OutContext.getOrCreateSymbol("amdgpu.max_num_agpr");
}

MCSymbol *MCResourceInfo::getMaxSGPRSymbol(MCContext &OutContext) {
  return OutContext.getOrCreateSymbol("amdgpu.max_num_sgpr");
}

// The (partially complete) expression should have no recursion in it. After
// all, we're trying to avoid recursion using this codepath. Returns true if
// Sym is found within Expr without recursing on Expr, false otherwise.
static bool findSymbolInExpr(MCSymbol *Sym, const MCExpr *Expr,
                             SmallVectorImpl<const MCExpr *> &Exprs,
                             SmallPtrSetImpl<const MCExpr *> &Visited) {
  // Assert if any of the expressions is already visited (i.e., there is
  // existing recursion).
  if (!Visited.insert(Expr).second)
    llvm_unreachable("already visited expression");

  switch (Expr->getKind()) {
  default:
    return false;
  case MCExpr::ExprKind::SymbolRef: {
    const MCSymbolRefExpr *SymRefExpr = cast<MCSymbolRefExpr>(Expr);
    const MCSymbol &SymRef = SymRefExpr->getSymbol();
    if (Sym == &SymRef)
      return true;
    if (SymRef.isVariable())
      Exprs.push_back(SymRef.getVariableValue(/*isUsed=*/false));
    return false;
  }
  case MCExpr::ExprKind::Binary: {
    const MCBinaryExpr *BExpr = cast<MCBinaryExpr>(Expr);
    Exprs.push_back(BExpr->getLHS());
    Exprs.push_back(BExpr->getRHS());
    return false;
  }
  case MCExpr::ExprKind::Unary: {
    const MCUnaryExpr *UExpr = cast<MCUnaryExpr>(Expr);
    Exprs.push_back(UExpr->getSubExpr());
    return false;
  }
  case MCExpr::ExprKind::Target: {
    const AMDGPUMCExpr *AGVK = cast<AMDGPUMCExpr>(Expr);
    for (const MCExpr *E : AGVK->getArgs())
      Exprs.push_back(E);
    return false;
  }
  }
}

// Symbols whose values eventually are used through their defines (i.e.,
// recursive) must be avoided. Do a walk over Expr to see if Sym will occur in
// it. The Expr is an MCExpr given through a callee's equivalent MCSymbol so if
// no recursion is found Sym can be safely assigned to a (sub-)expr which
// contains the symbol Expr is associated with. Returns true if Sym exists
// in Expr or its sub-expressions, false otherwise.
static bool foundRecursiveSymbolDef(MCSymbol *Sym, const MCExpr *Expr) {
  SmallVector<const MCExpr *, 8> WorkList;
  SmallPtrSet<const MCExpr *, 8> Visited;
  WorkList.push_back(Expr);

  while (!WorkList.empty()) {
    const MCExpr *CurExpr = WorkList.pop_back_val();
    if (findSymbolInExpr(Sym, CurExpr, WorkList, Visited))
      return true;
  }

  return false;
}

void MCResourceInfo::assignResourceInfoExpr(
    int64_t LocalValue, ResourceInfoKind RIK, AMDGPUMCExpr::VariantKind Kind,
    const MachineFunction &MF, const SmallVectorImpl<const Function *> &Callees,
    MCContext &OutContext) {
  const MCConstantExpr *LocalConstExpr =
      MCConstantExpr::create(LocalValue, OutContext);
  const MCExpr *SymVal = LocalConstExpr;
  MCSymbol *Sym = getSymbol(MF.getName(), RIK, OutContext);
  if (!Callees.empty()) {
    SmallVector<const MCExpr *, 8> ArgExprs;
    // Avoid recursive symbol assignment.
    SmallPtrSet<const Function *, 8> Seen;
    ArgExprs.push_back(LocalConstExpr);
    const Function &F = MF.getFunction();
    Seen.insert(&F);

    for (const Function *Callee : Callees) {
      if (!Seen.insert(Callee).second)
        continue;
      MCSymbol *CalleeValSym = getSymbol(Callee->getName(), RIK, OutContext);
      bool CalleeIsVar = CalleeValSym->isVariable();
      if (!CalleeIsVar ||
          (CalleeIsVar &&
           !foundRecursiveSymbolDef(
               Sym, CalleeValSym->getVariableValue(/*IsUsed=*/false)))) {
        ArgExprs.push_back(MCSymbolRefExpr::create(CalleeValSym, OutContext));
      }
    }
    if (ArgExprs.size() > 1)
      SymVal = AMDGPUMCExpr::create(Kind, ArgExprs, OutContext);
  }
  Sym->setVariableValue(SymVal);
}

void MCResourceInfo::gatherResourceInfo(
    const MachineFunction &MF,
    const AMDGPUResourceUsageAnalysis::SIFunctionResourceInfo &FRI,
    MCContext &OutContext) {
  // Worst case VGPR use for non-hardware-entrypoints.
  MCSymbol *MaxVGPRSym = getMaxVGPRSymbol(OutContext);
  MCSymbol *MaxAGPRSym = getMaxAGPRSymbol(OutContext);
  MCSymbol *MaxSGPRSym = getMaxSGPRSymbol(OutContext);

  if (!AMDGPU::isEntryFunctionCC(MF.getFunction().getCallingConv())) {
    addMaxVGPRCandidate(FRI.NumVGPR);
    addMaxAGPRCandidate(FRI.NumAGPR);
    addMaxSGPRCandidate(FRI.NumExplicitSGPR);
  }

  auto SetMaxReg = [&](MCSymbol *MaxSym, int32_t numRegs,
                       ResourceInfoKind RIK) {
    if (!FRI.HasIndirectCall) {
      assignResourceInfoExpr(numRegs, RIK, AMDGPUMCExpr::AGVK_Max, MF,
                             FRI.Callees, OutContext);
    } else {
      const MCExpr *SymRef = MCSymbolRefExpr::create(MaxSym, OutContext);
      MCSymbol *LocalNumSym = getSymbol(MF.getName(), RIK, OutContext);
      const MCExpr *MaxWithLocal = AMDGPUMCExpr::createMax(
          {MCConstantExpr::create(numRegs, OutContext), SymRef}, OutContext);
      LocalNumSym->setVariableValue(MaxWithLocal);
    }
  };

  SetMaxReg(MaxVGPRSym, FRI.NumVGPR, RIK_NumVGPR);
  SetMaxReg(MaxAGPRSym, FRI.NumAGPR, RIK_NumAGPR);
  SetMaxReg(MaxSGPRSym, FRI.NumExplicitSGPR, RIK_NumSGPR);

  {
    // The expression for private segment size should be: FRI.PrivateSegmentSize
    // + max(FRI.Callees, FRI.CalleeSegmentSize)
    SmallVector<const MCExpr *, 8> ArgExprs;
    MCSymbol *Sym = getSymbol(MF.getName(), RIK_PrivateSegSize, OutContext);
    if (FRI.CalleeSegmentSize)
      ArgExprs.push_back(
          MCConstantExpr::create(FRI.CalleeSegmentSize, OutContext));

    SmallPtrSet<const Function *, 8> Seen;
    Seen.insert(&MF.getFunction());
    for (const Function *Callee : FRI.Callees) {
      if (!Seen.insert(Callee).second)
        continue;
      if (!Callee->isDeclaration()) {
        MCSymbol *CalleeValSym =
            getSymbol(Callee->getName(), RIK_PrivateSegSize, OutContext);
        bool CalleeIsVar = CalleeValSym->isVariable();
        if (!CalleeIsVar ||
            (CalleeIsVar &&
             !foundRecursiveSymbolDef(
                 Sym, CalleeValSym->getVariableValue(/*IsUsed=*/false)))) {
          ArgExprs.push_back(MCSymbolRefExpr::create(CalleeValSym, OutContext));
        }
      }
    }
    const MCExpr *localConstExpr =
        MCConstantExpr::create(FRI.PrivateSegmentSize, OutContext);
    if (!ArgExprs.empty()) {
      const AMDGPUMCExpr *transitiveExpr =
          AMDGPUMCExpr::createMax(ArgExprs, OutContext);
      localConstExpr =
          MCBinaryExpr::createAdd(localConstExpr, transitiveExpr, OutContext);
    }
    Sym->setVariableValue(localConstExpr);
  }

  auto SetToLocal = [&](int64_t LocalValue, ResourceInfoKind RIK) {
    MCSymbol *Sym = getSymbol(MF.getName(), RIK, OutContext);
    Sym->setVariableValue(MCConstantExpr::create(LocalValue, OutContext));
  };

  if (!FRI.HasIndirectCall) {
    assignResourceInfoExpr(FRI.UsesVCC, ResourceInfoKind::RIK_UsesVCC,
                           AMDGPUMCExpr::AGVK_Or, MF, FRI.Callees, OutContext);
    assignResourceInfoExpr(FRI.UsesFlatScratch,
                           ResourceInfoKind::RIK_UsesFlatScratch,
                           AMDGPUMCExpr::AGVK_Or, MF, FRI.Callees, OutContext);
    assignResourceInfoExpr(FRI.HasDynamicallySizedStack,
                           ResourceInfoKind::RIK_HasDynSizedStack,
                           AMDGPUMCExpr::AGVK_Or, MF, FRI.Callees, OutContext);
    assignResourceInfoExpr(FRI.HasRecursion, ResourceInfoKind::RIK_HasRecursion,
                           AMDGPUMCExpr::AGVK_Or, MF, FRI.Callees, OutContext);
    assignResourceInfoExpr(FRI.HasIndirectCall,
                           ResourceInfoKind::RIK_HasIndirectCall,
                           AMDGPUMCExpr::AGVK_Or, MF, FRI.Callees, OutContext);
  } else {
    SetToLocal(FRI.UsesVCC, ResourceInfoKind::RIK_UsesVCC);
    SetToLocal(FRI.UsesFlatScratch, ResourceInfoKind::RIK_UsesFlatScratch);
    SetToLocal(FRI.HasDynamicallySizedStack,
               ResourceInfoKind::RIK_HasDynSizedStack);
    SetToLocal(FRI.HasRecursion, ResourceInfoKind::RIK_HasRecursion);
    SetToLocal(FRI.HasIndirectCall, ResourceInfoKind::RIK_HasIndirectCall);
  }
}

const MCExpr *MCResourceInfo::createTotalNumVGPRs(const MachineFunction &MF,
                                                  MCContext &Ctx) {
  return AMDGPUMCExpr::createTotalNumVGPR(
      getSymRefExpr(MF.getName(), RIK_NumAGPR, Ctx),
      getSymRefExpr(MF.getName(), RIK_NumVGPR, Ctx), Ctx);
}

const MCExpr *MCResourceInfo::createTotalNumSGPRs(const MachineFunction &MF,
                                                  bool hasXnack,
                                                  MCContext &Ctx) {
  return MCBinaryExpr::createAdd(
      getSymRefExpr(MF.getName(), RIK_NumSGPR, Ctx),
      AMDGPUMCExpr::createExtraSGPRs(
          getSymRefExpr(MF.getName(), RIK_UsesVCC, Ctx),
          getSymRefExpr(MF.getName(), RIK_UsesFlatScratch, Ctx), hasXnack, Ctx),
      Ctx);
}
