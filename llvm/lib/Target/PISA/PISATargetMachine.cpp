//===-- PISATargetMachine.cpp - Define TargetMachine for PISA -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISATargetMachine.h"
#include "PISA.h"
#include "TargetInfo/PISATargetInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

// NOLINTNEXTLINE(readability-identifier-naming)
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializePISATarget() {
  // Register the target.
  RegisterTargetMachine<PISATargetMachine> Y(getThePISATarget());
}

static Reloc::Model getEffectiveRelocModel(std::optional<Reloc::Model> RM) {
  if (!RM)
    return Reloc::PIC_;
  return *RM;
}

PISATargetMachine::PISATargetMachine(const Target &T, const Triple &TT,
                                     StringRef CPU, StringRef FS,
                                     const TargetOptions &Options,
                                     std::optional<Reloc::Model> RM,
                                     std::optional<CodeModel::Model> CM,
                                     CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(T, TT.computeDataLayout(), TT, CPU, FS, Options,
                               getEffectiveRelocModel(RM),
                               getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(std::make_unique<TargetLoweringObjectFileELF>()),
      Subtarget(TT, CPU.str(), FS.str(), *this) {
  initAsmInfo();
  setGlobalISel(true);
  setFastISel(false);
  setO0WantsFastISel(false);
  setRequiresStructuredCFG(false);
}

const PISASubtarget *
PISATargetMachine::getSubtargetImpl(const Function &F) const {
  Attribute CPUAttr = F.getFnAttribute("target-cpu");
  Attribute FSAttr = F.getFnAttribute("target-features");

  StringRef CPU =
      CPUAttr.isValid() ? CPUAttr.getValueAsString() : getTargetCPU();
  StringRef FS =
      FSAttr.isValid() ? FSAttr.getValueAsString() : getTargetFeatureString();

  SmallString<128> Key(CPU);
  Key.append(FS);

  auto &I = SubtargetMap[Key];
  if (!I)
    I = std::make_unique<PISASubtarget>(TargetTriple, CPU.str(), FS.str(),
                                        *this);
  return I.get();
}

namespace {
class PISAPassConfig : public TargetPassConfig {
public:
  PISAPassConfig(PISATargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  PISATargetMachine &getPISATargetMachine() const {
    return getTM<PISATargetMachine>();
  }
};
} // namespace

TargetPassConfig *PISATargetMachine::createPassConfig(PassManagerBase &PM) {
  return new PISAPassConfig(*this, PM);
}
