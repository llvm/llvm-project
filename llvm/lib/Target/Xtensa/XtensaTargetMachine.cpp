//===- XtensaTargetMachine.cpp - Define TargetMachine for Xtensa ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the info about Xtensa target spec.
//
//===----------------------------------------------------------------------===//

#include "XtensaTargetMachine.h"
#include "TargetInfo/XtensaTargetInfo.h"
#include "XtensaMachineFunctionInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/PassRegistry.h"
#include "llvm/Transforms/Scalar.h"
#include <optional>

using namespace llvm;

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeXtensaTarget() {
  // Register the target.
  RegisterTargetMachine<XtensaTargetMachine> A(getTheXtensaTarget());
  PassRegistry &PR = *PassRegistry::getPassRegistry();
  initializeXtensaAsmPrinterPass(PR);
}

static Reloc::Model getEffectiveRelocModel(bool JIT,
                                           std::optional<Reloc::Model> RM) {
  if (!RM || JIT)
     return Reloc::Static;
  return *RM;
}

XtensaTargetMachine::XtensaTargetMachine(const Target &T, const Triple &TT,
                                         StringRef CPU, StringRef FS,
                                         const TargetOptions &Options,
                                         std::optional<Reloc::Model> RM,
                                         std::optional<CodeModel::Model> CM,
                                         CodeGenOptLevel OL, bool JIT,
                                         bool IsLittle)
    : CodeGenTargetMachineImpl(T, TT.computeDataLayout(), TT, CPU, FS, Options,
                               getEffectiveRelocModel(JIT, RM),
                               getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(std::make_unique<TargetLoweringObjectFileELF>()) {
  initAsmInfo();
}

XtensaTargetMachine::XtensaTargetMachine(const Target &T, const Triple &TT,
                                         StringRef CPU, StringRef FS,
                                         const TargetOptions &Options,
                                         std::optional<Reloc::Model> RM,
                                         std::optional<CodeModel::Model> CM,
                                         CodeGenOptLevel OL, bool JIT)
    : XtensaTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, JIT, true) {}

const XtensaSubtarget *
XtensaTargetMachine::getSubtargetImpl(const Function &F) const {
  Attribute CPUAttr = F.getFnAttribute("target-cpu");
  Attribute FSAttr = F.getFnAttribute("target-features");

  auto CPU = CPUAttr.isValid() ? CPUAttr.getValueAsString().str() : TargetCPU;
  auto FS = FSAttr.isValid() ? FSAttr.getValueAsString().str() : TargetFS;

  auto &I = SubtargetMap[CPU + FS];
  if (!I) {
    // This needs to be done before we create a new subtarget since any
    // creation will depend on the TM and the code generation flags on the
    // function that reside in TargetOptions.
    resetTargetOptions(F);
    I = std::make_unique<XtensaSubtarget>(TargetTriple, CPU, FS, *this);
  }
  return I.get();
}

MachineFunctionInfo *XtensaTargetMachine::createMachineFunctionInfo(
    BumpPtrAllocator &Allocator, const Function &F,
    const TargetSubtargetInfo *STI) const {
  return XtensaMachineFunctionInfo::create<XtensaMachineFunctionInfo>(Allocator,
                                                                      F, STI);
}

namespace {
/// Xtensa Code Generator Pass Configuration Options.
class XtensaPassConfig : public TargetPassConfig {
public:
  XtensaPassConfig(XtensaTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  XtensaTargetMachine &getXtensaTargetMachine() const {
    return getTM<XtensaTargetMachine>();
  }

  bool addInstSelector() override;
  void addIRPasses() override;
  void addPreEmitPass() override;
};
} // end anonymous namespace

bool XtensaPassConfig::addInstSelector() {
  addPass(createXtensaISelDag(getXtensaTargetMachine(), getOptLevel()));
  return false;
}

void XtensaPassConfig::addIRPasses() {
  addPass(createAtomicExpandLegacyPass());
  TargetPassConfig::addIRPasses();
}

void XtensaPassConfig::addPreEmitPass() { addPass(&BranchRelaxationPassID); }

TargetPassConfig *XtensaTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new XtensaPassConfig(*this, PM);
}
