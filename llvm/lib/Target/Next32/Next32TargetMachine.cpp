//===-- Next32TargetMachine.cpp - Define TargetMachine for Next32 ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the info about Next32 target spec.
//
//===----------------------------------------------------------------------===//

#include "Next32TargetMachine.h"
#include "Next32.h"
#include "Next32MachineFunctionInfo.h"
#include "Next32TargetObjectFile.h"
#include "Next32TargetTransformInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"

using namespace llvm;

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeNext32Target() {
  // Register the target.
  RegisterTargetMachine<Next32TargetMachine> X(getTheNext32Target());
}

static std::string computeDataLayout(const Triple &TT) {
  // S32 - so Next32 can easily store small variables on the stack
  // p0:64:64:46 (implicit) - Next32 pointers are 64 bit,
  // (not to be confused with bb/func that are actually feeder addresses)
  // n32 - Next32 native integer size is 32 bit (our arithmetics use 32 bit)
  return "e-S128-m:e-n8:16:32-i64:64";
}

// TODO: Think about relocation in Next32
static Reloc::Model getEffectiveRelocModel(std::optional<Reloc::Model> RM) {
  if (!RM.has_value())
    return Reloc::PIC_;
  return *RM;
}

static CodeGenOptLevel coerceOptimizationLevel(CodeGenOptLevel OL) {
  if (OL == CodeGenOptLevel::None) {
    errs() << "warning: target optimization level coerced from -O0 to -O1\n";
    return CodeGenOptLevel::Less;
  }
  return OL;
}

Next32TargetMachine::Next32TargetMachine(const Target &T, const Triple &TT,
                                         StringRef CPU, StringRef FS,
                                         const TargetOptions &Options,
                                         std::optional<Reloc::Model> RM,
                                         std::optional<CodeModel::Model> CM,
                                         CodeGenOptLevel OL, bool JIT)
    : LLVMTargetMachine(T, computeDataLayout(TT), TT, CPU, FS, Options,
                        getEffectiveRelocModel(RM),
                        getEffectiveCodeModel(CM, CodeModel::Small),
                        coerceOptimizationLevel(OL)),
      TLOF(std::make_unique<Next32ELFTargetObjectFile>()),
      Subtarget(TT, CPU, FS, *this) {
  initAsmInfo();
}
namespace {
// Next32 Code Generator Pass Configuration Options.
class Next32PassConfig : public TargetPassConfig {
public:
  Next32PassConfig(Next32TargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {
    // Disable tail-merge (Done by Next32ConditionalInline)
    setEnableTailMerge(false);
    disablePass(&MachineBlockPlacementID);
    disablePass(&MachineLateInstrsCleanupID);
  }

  Next32TargetMachine &getNext32TargetMachine() const {
    return getTM<Next32TargetMachine>();
  }

  void addIRPasses() override;
  bool addInstSelector() override;
  void addPreRegAlloc() override;
  void addPostRegAlloc() override;
  void addPreEmitPass() override;
  bool disableLSRPass() override;
};
} // namespace

TargetPassConfig *Next32TargetMachine::createPassConfig(PassManagerBase &PM) {
  return new Next32PassConfig(*this, PM);
}

MachineFunctionInfo *Next32TargetMachine::createMachineFunctionInfo(
    BumpPtrAllocator &Allocator, const Function &F,
    const TargetSubtargetInfo *STI) const {
  return Next32MachineFunctionInfo::create<Next32MachineFunctionInfo>(Allocator,
                                                                      F, STI);
}

TargetTransformInfo
Next32TargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(Next32TTIImpl(this, F));
}

void Next32PassConfig::addIRPasses() {
  addPass(createAtomicExpandLegacyPass());
  addPass(createNext32PromotePass());
  TargetPassConfig::addIRPasses();
}

// Install an instruction selector pass using
// the ISelDag to gen Next32 code.
bool Next32PassConfig::addInstSelector() {
  addPass(createNext32ISelDag(getNext32TargetMachine(), getOptLevel()));
  return false;
}

void Next32PassConfig::addPreRegAlloc() {
  addPass(createNext32CallSplits());
  addPass(createNext32AddRetFid());
  addPass(createNext32CondBranchFixup());
  addPass(createNext32OrderCallChain());
}

void Next32PassConfig::addPostRegAlloc() {
  addPass(createNext32CallTerminators());
}

void Next32PassConfig::addPreEmitPass() {
  addPass(createNext32EliminateCallTerminators());
  addPass(createNext32CalculateFeeders());
  addPass(createNext32WriterChains());
}

bool Next32PassConfig::disableLSRPass() { return true; }
