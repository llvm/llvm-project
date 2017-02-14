///===- MachineOptimizationRemarkEmitter.cpp - Opt Diagnostic -*- C++ -*---===//
///
///                     The LLVM Compiler Infrastructure
///
/// This file is distributed under the University of Illinois Open Source
/// License. See LICENSE.TXT for details.
///
///===---------------------------------------------------------------------===//
/// \file
/// Optimization diagnostic interfaces for machine passes.  It's packaged as an
/// analysis pass so that by using this service passes become dependent on MBFI
/// as well.  MBFI is used to compute the "hotness" of the diagnostic message.
///
///===---------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/LazyMachineBlockFrequencyInfo.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/LLVMContext.h"

using namespace llvm;

Optional<uint64_t>
MachineOptimizationRemarkEmitter::computeHotness(const MachineBasicBlock &MBB) {
  if (!MBFI)
    return None;

  return MBFI->getBlockProfileCount(&MBB);
}

void MachineOptimizationRemarkEmitter::computeHotness(
    DiagnosticInfoMIROptimization &Remark) {
  const MachineBasicBlock *MBB = Remark.getBlock();
  if (MBB)
    Remark.setHotness(computeHotness(*MBB));
}

void MachineOptimizationRemarkEmitter::emit(
    DiagnosticInfoOptimizationBase &OptDiagCommon) {
  auto &OptDiag = cast<DiagnosticInfoMIROptimization>(OptDiagCommon);
  computeHotness(OptDiag);

  LLVMContext &Ctx = MF.getFunction()->getContext();
  yaml::Output *Out = Ctx.getDiagnosticsOutputFile();
  if (Out) {
    auto *P = &const_cast<DiagnosticInfoOptimizationBase &>(OptDiagCommon);
    *Out << P;
  }
  // FIXME: now that IsVerbose is part of DI, filtering for this will be moved
  // from here to clang.
  if (!OptDiag.isVerbose() || shouldEmitVerbose())
    Ctx.diagnose(OptDiag);
}

MachineOptimizationRemarkEmitterPass::MachineOptimizationRemarkEmitterPass()
    : MachineFunctionPass(ID) {
  initializeMachineOptimizationRemarkEmitterPassPass(
      *PassRegistry::getPassRegistry());
}

bool MachineOptimizationRemarkEmitterPass::runOnMachineFunction(
    MachineFunction &MF) {
  MachineBlockFrequencyInfo *MBFI;

  if (MF.getFunction()->getContext().getDiagnosticHotnessRequested())
    MBFI = &getAnalysis<LazyMachineBlockFrequencyInfoPass>().getBFI();
  else
    MBFI = nullptr;

  ORE = llvm::make_unique<MachineOptimizationRemarkEmitter>(MF, MBFI);
  return false;
}

void MachineOptimizationRemarkEmitterPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  LazyMachineBlockFrequencyInfoPass::getLazyMachineBFIAnalysisUsage(AU);
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

char MachineOptimizationRemarkEmitterPass::ID = 0;
static const char ore_name[] = "Machine Optimization Remark Emitter";
#define ORE_NAME "machine-opt-remark-emitter"

INITIALIZE_PASS_BEGIN(MachineOptimizationRemarkEmitterPass, ORE_NAME, ore_name,
                      false, true)
INITIALIZE_PASS_DEPENDENCY(LazyMachineBFIPass)
INITIALIZE_PASS_END(MachineOptimizationRemarkEmitterPass, ORE_NAME, ore_name,
                    false, true)
