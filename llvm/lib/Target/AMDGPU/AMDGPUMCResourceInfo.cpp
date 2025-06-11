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
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "amdgpu-mc-resource-usage"

using namespace llvm;

MCSymbol *MCResourceInfo::getSymbol(StringRef FuncName, ResourceInfoKind RIK,
                                    MCContext &OutContext, bool IsLocal) {
  auto GOCS = [FuncName, &OutContext, IsLocal](StringRef Suffix) {
    StringRef Prefix =
        IsLocal ? OutContext.getAsmInfo()->getPrivateGlobalPrefix() : "";
    return OutContext.getOrCreateSymbol(Twine(Prefix) + FuncName +
                                        Twine(Suffix));
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
                                            MCContext &Ctx, bool IsLocal) {
  return MCSymbolRefExpr::create(getSymbol(FuncName, RIK, Ctx, IsLocal), Ctx);
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

// Tries to flatten recursive call register resource gathering. Simple cycle
// avoiding dfs to find the constants in the propagated symbols.
// Assumes:
// - RecSym has been confirmed to recurse (this means the callee symbols should
//   all be populated, started at RecSym).
// - Shape of the resource symbol's MCExpr (`max` args are order agnostic):
//   RecSym.MCExpr := max(<constant>+, <callee_symbol>*)
const MCExpr *MCResourceInfo::flattenedCycleMax(MCSymbol *RecSym,
                                                ResourceInfoKind RIK,
                                                MCContext &OutContext) {
  SmallPtrSet<const MCExpr *, 8> Seen;
  SmallVector<const MCExpr *, 8> WorkList;
  int64_t Maximum = 0;

  const MCExpr *RecExpr = RecSym->getVariableValue();
  WorkList.push_back(RecExpr);

  while (!WorkList.empty()) {
    const MCExpr *CurExpr = WorkList.pop_back_val();
    switch (CurExpr->getKind()) {
    default: {
      // Assuming the recursion is of shape `max(<constant>, <callee_symbol>)`
      // where <callee_symbol> will eventually recurse. If this condition holds,
      // the recursion occurs within some other (possibly unresolvable) MCExpr,
      // thus using the worst case value then.
      if (!AMDGPUMCExpr::isSymbolUsedInExpression(RecSym, CurExpr)) {
        LLVM_DEBUG(dbgs() << "MCResUse:   " << RecSym->getName()
                          << ": Recursion in unexpected sub-expression, using "
                             "module maximum\n");
        switch (RIK) {
        default:
          break;
        case RIK_NumVGPR:
          return MCSymbolRefExpr::create(getMaxVGPRSymbol(OutContext),
                                         OutContext);
          break;
        case RIK_NumSGPR:
          return MCSymbolRefExpr::create(getMaxSGPRSymbol(OutContext),
                                         OutContext);
          break;
        case RIK_NumAGPR:
          return MCSymbolRefExpr::create(getMaxAGPRSymbol(OutContext),
                                         OutContext);
          break;
        }
      }
      break;
    }
    case MCExpr::ExprKind::Constant: {
      int64_t Val = cast<MCConstantExpr>(CurExpr)->getValue();
      Maximum = std::max(Maximum, Val);
      break;
    }
    case MCExpr::ExprKind::SymbolRef: {
      const MCSymbolRefExpr *SymExpr = cast<MCSymbolRefExpr>(CurExpr);
      const MCSymbol &SymRef = SymExpr->getSymbol();
      if (SymRef.isVariable()) {
        const MCExpr *SymVal = SymRef.getVariableValue();
        if (Seen.insert(SymVal).second)
          WorkList.push_back(SymVal);
      }
      break;
    }
    case MCExpr::ExprKind::Target: {
      const AMDGPUMCExpr *TargetExpr = cast<AMDGPUMCExpr>(CurExpr);
      if (TargetExpr->getKind() == AMDGPUMCExpr::VariantKind::AGVK_Max) {
        for (auto &Arg : TargetExpr->getArgs())
          WorkList.push_back(Arg);
      }
      break;
    }
    }
  }

  LLVM_DEBUG(dbgs() << "MCResUse:   " << RecSym->getName()
                    << ": Using flattened max: << " << Maximum << '\n');

  return MCConstantExpr::create(Maximum, OutContext);
}

void MCResourceInfo::assignResourceInfoExpr(
    int64_t LocalValue, ResourceInfoKind RIK, AMDGPUMCExpr::VariantKind Kind,
    const MachineFunction &MF, const SmallVectorImpl<const Function *> &Callees,
    MCContext &OutContext) {
  const TargetMachine &TM = MF.getTarget();
  bool IsLocal = MF.getFunction().hasLocalLinkage();
  MCSymbol *FnSym = TM.getSymbol(&MF.getFunction());
  const MCConstantExpr *LocalConstExpr =
      MCConstantExpr::create(LocalValue, OutContext);
  const MCExpr *SymVal = LocalConstExpr;
  MCSymbol *Sym = getSymbol(FnSym->getName(), RIK, OutContext, IsLocal);
  LLVM_DEBUG(dbgs() << "MCResUse:   " << Sym->getName() << ": Adding "
                    << LocalValue << " as function local usage\n");
  if (!Callees.empty()) {
    SmallVector<const MCExpr *, 8> ArgExprs;
    SmallPtrSet<const Function *, 8> Seen;
    ArgExprs.push_back(LocalConstExpr);

    for (const Function *Callee : Callees) {
      if (!Seen.insert(Callee).second)
        continue;

      bool IsCalleeLocal = Callee->hasLocalLinkage();
      MCSymbol *CalleeFnSym = TM.getSymbol(&Callee->getFunction());
      MCSymbol *CalleeValSym =
          getSymbol(CalleeFnSym->getName(), RIK, OutContext, IsCalleeLocal);

      // Avoid constructing recursive definitions by detecting whether `Sym` is
      // found transitively within any of its `CalleeValSym`.
      if (!CalleeValSym->isVariable() ||
          !AMDGPUMCExpr::isSymbolUsedInExpression(
              Sym, CalleeValSym->getVariableValue())) {
        LLVM_DEBUG(dbgs() << "MCResUse:   " << Sym->getName() << ": Adding "
                          << CalleeValSym->getName() << " as callee\n");
        ArgExprs.push_back(MCSymbolRefExpr::create(CalleeValSym, OutContext));
      } else {
        LLVM_DEBUG(dbgs() << "MCResUse:   " << Sym->getName()
                          << ": Recursion found, attempt flattening of cycle "
                             "for resource usage\n");
        // In case of recursion for vgpr/sgpr/agpr resource usage: try to
        // flatten and use the max of the call cycle. May still end up emitting
        // module max if not fully resolvable.
        switch (RIK) {
        default:
          break;
        case RIK_NumVGPR:
        case RIK_NumSGPR:
        case RIK_NumAGPR:
          ArgExprs.push_back(flattenedCycleMax(CalleeValSym, RIK, OutContext));
          break;
        }
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
  bool IsLocal = MF.getFunction().hasLocalLinkage();

  if (!AMDGPU::isEntryFunctionCC(MF.getFunction().getCallingConv())) {
    addMaxVGPRCandidate(FRI.NumVGPR);
    addMaxAGPRCandidate(FRI.NumAGPR);
    addMaxSGPRCandidate(FRI.NumExplicitSGPR);
  }

  const TargetMachine &TM = MF.getTarget();
  MCSymbol *FnSym = TM.getSymbol(&MF.getFunction());

  LLVM_DEBUG(dbgs() << "MCResUse: Gathering resource information for "
                    << FnSym->getName() << '\n');
  LLVM_DEBUG({
    if (!FRI.Callees.empty()) {
      dbgs() << "MCResUse: Callees:\n";
      for (const Function *Callee : FRI.Callees) {
        MCSymbol *CalleeFnSym = TM.getSymbol(&Callee->getFunction());
        dbgs() << "MCResUse:   " << CalleeFnSym->getName() << '\n';
      }
    }
  });

  auto SetMaxReg = [&](MCSymbol *MaxSym, int32_t numRegs,
                       ResourceInfoKind RIK) {
    if (!FRI.HasIndirectCall) {
      assignResourceInfoExpr(numRegs, RIK, AMDGPUMCExpr::AGVK_Max, MF,
                             FRI.Callees, OutContext);
    } else {
      const MCExpr *SymRef = MCSymbolRefExpr::create(MaxSym, OutContext);
      MCSymbol *LocalNumSym =
          getSymbol(FnSym->getName(), RIK, OutContext, IsLocal);
      const MCExpr *MaxWithLocal = AMDGPUMCExpr::createMax(
          {MCConstantExpr::create(numRegs, OutContext), SymRef}, OutContext);
      LocalNumSym->setVariableValue(MaxWithLocal);
      LLVM_DEBUG(dbgs() << "MCResUse:   " << LocalNumSym->getName()
                        << ": Indirect callee within, using module maximum\n");
    }
  };

  LLVM_DEBUG(dbgs() << "MCResUse: " << FnSym->getName() << '\n');
  SetMaxReg(MaxVGPRSym, FRI.NumVGPR, RIK_NumVGPR);
  SetMaxReg(MaxAGPRSym, FRI.NumAGPR, RIK_NumAGPR);
  SetMaxReg(MaxSGPRSym, FRI.NumExplicitSGPR, RIK_NumSGPR);

  {
    // The expression for private segment size should be: FRI.PrivateSegmentSize
    // + max(FRI.Callees, FRI.CalleeSegmentSize)
    SmallVector<const MCExpr *, 8> ArgExprs;
    MCSymbol *Sym =
        getSymbol(FnSym->getName(), RIK_PrivateSegSize, OutContext, IsLocal);
    if (FRI.CalleeSegmentSize) {
      LLVM_DEBUG(dbgs() << "MCResUse:   " << Sym->getName() << ": Adding "
                        << FRI.CalleeSegmentSize
                        << " for indirect/recursive callees within\n");
      ArgExprs.push_back(
          MCConstantExpr::create(FRI.CalleeSegmentSize, OutContext));
    }

    SmallPtrSet<const Function *, 8> Seen;
    Seen.insert(&MF.getFunction());
    for (const Function *Callee : FRI.Callees) {
      if (!Seen.insert(Callee).second)
        continue;
      if (!Callee->isDeclaration()) {
        bool IsCalleeLocal = Callee->hasLocalLinkage();
        MCSymbol *CalleeFnSym = TM.getSymbol(&Callee->getFunction());
        MCSymbol *CalleeValSym =
            getSymbol(CalleeFnSym->getName(), RIK_PrivateSegSize, OutContext,
                      IsCalleeLocal);

        // Avoid constructing recursive definitions by detecting whether `Sym`
        // is found transitively within any of its `CalleeValSym`.
        if (!CalleeValSym->isVariable() ||
            !AMDGPUMCExpr::isSymbolUsedInExpression(
                Sym, CalleeValSym->getVariableValue())) {
          LLVM_DEBUG(dbgs() << "MCResUse:   " << Sym->getName() << ": Adding "
                            << CalleeValSym->getName() << " as callee\n");
          ArgExprs.push_back(MCSymbolRefExpr::create(CalleeValSym, OutContext));
        }
      }
    }
    const MCExpr *localConstExpr =
        MCConstantExpr::create(FRI.PrivateSegmentSize, OutContext);
    LLVM_DEBUG(dbgs() << "MCResUse:   " << Sym->getName() << ": Adding "
                      << FRI.PrivateSegmentSize
                      << " as function local usage\n");
    if (!ArgExprs.empty()) {
      const AMDGPUMCExpr *transitiveExpr =
          AMDGPUMCExpr::createMax(ArgExprs, OutContext);
      localConstExpr =
          MCBinaryExpr::createAdd(localConstExpr, transitiveExpr, OutContext);
    }
    Sym->setVariableValue(localConstExpr);
  }

  auto SetToLocal = [&](int64_t LocalValue, ResourceInfoKind RIK) {
    MCSymbol *Sym = getSymbol(FnSym->getName(), RIK, OutContext, IsLocal);
    LLVM_DEBUG(
        dbgs() << "MCResUse:   " << Sym->getName() << ": Adding " << LocalValue
               << ", no further propagation as indirect callee found within\n");
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
  const TargetMachine &TM = MF.getTarget();
  MCSymbol *FnSym = TM.getSymbol(&MF.getFunction());
  bool IsLocal = MF.getFunction().hasLocalLinkage();
  return AMDGPUMCExpr::createTotalNumVGPR(
      getSymRefExpr(FnSym->getName(), RIK_NumAGPR, Ctx, IsLocal),
      getSymRefExpr(FnSym->getName(), RIK_NumVGPR, Ctx, IsLocal), Ctx);
}

const MCExpr *MCResourceInfo::createTotalNumSGPRs(const MachineFunction &MF,
                                                  bool hasXnack,
                                                  MCContext &Ctx) {
  const TargetMachine &TM = MF.getTarget();
  MCSymbol *FnSym = TM.getSymbol(&MF.getFunction());
  bool IsLocal = MF.getFunction().hasLocalLinkage();
  return MCBinaryExpr::createAdd(
      getSymRefExpr(FnSym->getName(), RIK_NumSGPR, Ctx, IsLocal),
      AMDGPUMCExpr::createExtraSGPRs(
          getSymRefExpr(FnSym->getName(), RIK_UsesVCC, Ctx, IsLocal),
          getSymRefExpr(FnSym->getName(), RIK_UsesFlatScratch, Ctx, IsLocal),
          hasXnack, Ctx),
      Ctx);
}
