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

MCSymbol *MCResourceInfo::getSymbol(StringRef FuncName, ResourceInfoKind RIK) {
  switch (RIK) {
  case RIK_NumVGPR:
    return OutContext.getOrCreateSymbol(FuncName + Twine(".num_vgpr"));
  case RIK_NumAGPR:
    return OutContext.getOrCreateSymbol(FuncName + Twine(".num_agpr"));
  case RIK_NumSGPR:
    return OutContext.getOrCreateSymbol(FuncName + Twine(".num_sgpr"));
  case RIK_PrivateSegSize:
    return OutContext.getOrCreateSymbol(FuncName + Twine(".private_seg_size"));
  case RIK_UsesVCC:
    return OutContext.getOrCreateSymbol(FuncName + Twine(".uses_vcc"));
  case RIK_UsesFlatScratch:
    return OutContext.getOrCreateSymbol(FuncName + Twine(".uses_flat_scratch"));
  case RIK_HasDynSizedStack:
    return OutContext.getOrCreateSymbol(FuncName +
                                        Twine(".has_dyn_sized_stack"));
  case RIK_HasRecursion:
    return OutContext.getOrCreateSymbol(FuncName + Twine(".has_recursion"));
  case RIK_HasIndirectCall:
    return OutContext.getOrCreateSymbol(FuncName + Twine(".has_indirect_call"));
  }
  llvm_unreachable("Unexpected ResourceInfoKind.");
}

const MCExpr *MCResourceInfo::getSymRefExpr(StringRef FuncName,
                                            ResourceInfoKind RIK,
                                            MCContext &Ctx) {
  return MCSymbolRefExpr::create(getSymbol(FuncName, RIK), Ctx);
}

void MCResourceInfo::assignMaxRegs() {
  // Assign expression to get the max register use to the max_num_Xgpr symbol.
  MCSymbol *MaxVGPRSym = getMaxVGPRSymbol();
  MCSymbol *MaxAGPRSym = getMaxAGPRSymbol();
  MCSymbol *MaxSGPRSym = getMaxSGPRSymbol();

  auto assignMaxRegSym = [this](MCSymbol *Sym, int32_t RegCount) {
    const MCExpr *MaxExpr = MCConstantExpr::create(RegCount, OutContext);
    Sym->setVariableValue(MaxExpr);
  };

  assignMaxRegSym(MaxVGPRSym, MaxVGPR);
  assignMaxRegSym(MaxAGPRSym, MaxAGPR);
  assignMaxRegSym(MaxSGPRSym, MaxSGPR);
}

void MCResourceInfo::Finalize() {
  assert(!finalized && "Cannot finalize ResourceInfo again.");
  finalized = true;
  assignMaxRegs();
}

MCSymbol *MCResourceInfo::getMaxVGPRSymbol() {
  return OutContext.getOrCreateSymbol("max_num_vgpr");
}

MCSymbol *MCResourceInfo::getMaxAGPRSymbol() {
  return OutContext.getOrCreateSymbol("max_num_agpr");
}

MCSymbol *MCResourceInfo::getMaxSGPRSymbol() {
  return OutContext.getOrCreateSymbol("max_num_sgpr");
}

void MCResourceInfo::assignResourceInfoExpr(
    int64_t localValue, ResourceInfoKind RIK, AMDGPUMCExpr::VariantKind Kind,
    const MachineFunction &MF,
    const SmallVectorImpl<const Function *> &Callees) {
  const MCConstantExpr *localConstExpr =
      MCConstantExpr::create(localValue, OutContext);
  const MCExpr *SymVal = localConstExpr;
  if (Callees.size() > 0) {
    std::vector<const MCExpr *> ArgExprs;
    // Avoid recursive symbol assignment.
    SmallSet<StringRef, 8> Seen;
    ArgExprs.push_back(localConstExpr);
    Seen.insert(MF.getName());

    for (const Function *Callee : Callees) {
      if (Seen.contains(Callee->getName()))
        continue;
      Seen.insert(Callee->getName());
      MCSymbol *calleeValSym = getSymbol(Callee->getName(), RIK);
      ArgExprs.push_back(MCSymbolRefExpr::create(calleeValSym, OutContext));
    }
    SymVal = AMDGPUMCExpr::create(Kind, ArgExprs, OutContext);
  }
  MCSymbol *Sym = getSymbol(MF.getName(), RIK);
  Sym->setVariableValue(SymVal);
}

void MCResourceInfo::gatherResourceInfo(
    const MachineFunction &MF,
    const AMDGPUResourceUsageAnalysis::SIFunctionResourceInfo &FRI) {
  // Worst case VGPR use for non-hardware-entrypoints.
  MCSymbol *maxVGPRSym = getMaxVGPRSymbol();
  MCSymbol *maxAGPRSym = getMaxAGPRSymbol();
  MCSymbol *maxSGPRSym = getMaxSGPRSymbol();

  if (!AMDGPU::isEntryFunctionCC(MF.getFunction().getCallingConv())) {
    addMaxVGPRCandidate(FRI.NumVGPR);
    addMaxAGPRCandidate(FRI.NumAGPR);
    addMaxSGPRCandidate(FRI.NumExplicitSGPR);
  }

  auto setMaxReg = [&](MCSymbol *MaxSym, int32_t numRegs,
                       ResourceInfoKind RIK) {
    if (!FRI.HasIndirectCall) {
      assignResourceInfoExpr(numRegs, RIK, AMDGPUMCExpr::AGVK_Max, MF,
                             FRI.Callees);
    } else {
      const MCExpr *SymRef = MCSymbolRefExpr::create(MaxSym, OutContext);
      MCSymbol *LocalNumSym = getSymbol(MF.getName(), RIK);
      const MCExpr *MaxWithLocal = AMDGPUMCExpr::createMax(
          {MCConstantExpr::create(numRegs, OutContext), SymRef}, OutContext);
      LocalNumSym->setVariableValue(MaxWithLocal);
    }
  };

  setMaxReg(maxVGPRSym, FRI.NumVGPR, RIK_NumVGPR);
  setMaxReg(maxAGPRSym, FRI.NumAGPR, RIK_NumAGPR);
  setMaxReg(maxSGPRSym, FRI.NumExplicitSGPR, RIK_NumSGPR);

  {
    // The expression for private segment size should be: FRI.PrivateSegmentSize
    // + max(FRI.Callees, FRI.CalleeSegmentSize)
    std::vector<const MCExpr *> ArgExprs;
    if (FRI.CalleeSegmentSize)
      ArgExprs.push_back(
          MCConstantExpr::create(FRI.CalleeSegmentSize, OutContext));

    if (!FRI.HasIndirectCall) {
      for (const Function *Callee : FRI.Callees) {
        MCSymbol *calleeValSym =
            getSymbol(Callee->getName(), RIK_PrivateSegSize);
        ArgExprs.push_back(MCSymbolRefExpr::create(calleeValSym, OutContext));
      }
    }
    const MCExpr *localConstExpr =
        MCConstantExpr::create(FRI.PrivateSegmentSize, OutContext);
    if (ArgExprs.size() > 0) {
      const AMDGPUMCExpr *transitiveExpr =
          AMDGPUMCExpr::createMax(ArgExprs, OutContext);
      localConstExpr =
          MCBinaryExpr::createAdd(localConstExpr, transitiveExpr, OutContext);
    }
    getSymbol(MF.getName(), RIK_PrivateSegSize)
        ->setVariableValue(localConstExpr);
  }

  auto setToLocal = [&](int64_t localValue, ResourceInfoKind RIK) {
    MCSymbol *Sym = getSymbol(MF.getName(), RIK);
    Sym->setVariableValue(MCConstantExpr::create(localValue, OutContext));
  };

  if (!FRI.HasIndirectCall) {
    assignResourceInfoExpr(FRI.UsesVCC, ResourceInfoKind::RIK_UsesVCC,
                           AMDGPUMCExpr::AGVK_Or, MF, FRI.Callees);
    assignResourceInfoExpr(FRI.UsesFlatScratch,
                           ResourceInfoKind::RIK_UsesFlatScratch,
                           AMDGPUMCExpr::AGVK_Or, MF, FRI.Callees);
    assignResourceInfoExpr(FRI.HasDynamicallySizedStack,
                           ResourceInfoKind::RIK_HasDynSizedStack,
                           AMDGPUMCExpr::AGVK_Or, MF, FRI.Callees);
    assignResourceInfoExpr(FRI.HasRecursion, ResourceInfoKind::RIK_HasRecursion,
                           AMDGPUMCExpr::AGVK_Or, MF, FRI.Callees);
    assignResourceInfoExpr(FRI.HasIndirectCall,
                           ResourceInfoKind::RIK_HasIndirectCall,
                           AMDGPUMCExpr::AGVK_Or, MF, FRI.Callees);
  } else {
    setToLocal(FRI.UsesVCC, ResourceInfoKind::RIK_UsesVCC);
    setToLocal(FRI.UsesFlatScratch, ResourceInfoKind::RIK_UsesFlatScratch);
    setToLocal(FRI.HasDynamicallySizedStack,
               ResourceInfoKind::RIK_HasDynSizedStack);
    setToLocal(FRI.HasRecursion, ResourceInfoKind::RIK_HasRecursion);
    setToLocal(FRI.HasIndirectCall, ResourceInfoKind::RIK_HasIndirectCall);
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
