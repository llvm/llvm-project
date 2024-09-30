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
  auto GOCS = [this, FuncName, &OutContext](StringRef Suffix) {
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

  auto assignMaxRegSym = [this, &OutContext](MCSymbol *Sym, int32_t RegCount) {
    const MCExpr *MaxExpr = MCConstantExpr::create(RegCount, OutContext);
    Sym->setVariableValue(MaxExpr);
  };

  assignMaxRegSym(MaxVGPRSym, MaxVGPR);
  assignMaxRegSym(MaxAGPRSym, MaxAGPR);
  assignMaxRegSym(MaxSGPRSym, MaxSGPR);
}

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

void MCResourceInfo::assignResourceInfoExpr(
    int64_t LocalValue, ResourceInfoKind RIK, AMDGPUMCExpr::VariantKind Kind,
    const MachineFunction &MF, const SmallVectorImpl<const Function *> &Callees,
    MCContext &OutContext) {
  const MCConstantExpr *LocalConstExpr =
      MCConstantExpr::create(LocalValue, OutContext);
  const MCExpr *SymVal = LocalConstExpr;
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
      ArgExprs.push_back(MCSymbolRefExpr::create(CalleeValSym, OutContext));
    }
    SymVal = AMDGPUMCExpr::create(Kind, ArgExprs, OutContext);
  }
  MCSymbol *Sym = getSymbol(MF.getName(), RIK, OutContext);
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
    if (FRI.CalleeSegmentSize)
      ArgExprs.push_back(
          MCConstantExpr::create(FRI.CalleeSegmentSize, OutContext));

    if (!FRI.HasIndirectCall) {
      for (const Function *Callee : FRI.Callees) {
        MCSymbol *calleeValSym =
            getSymbol(Callee->getName(), RIK_PrivateSegSize, OutContext);
        ArgExprs.push_back(MCSymbolRefExpr::create(calleeValSym, OutContext));
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
    getSymbol(MF.getName(), RIK_PrivateSegSize, OutContext)
        ->setVariableValue(localConstExpr);
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
