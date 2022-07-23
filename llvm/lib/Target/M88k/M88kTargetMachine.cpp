//===-- M88kTargetMachine.cpp - Define TargetMachine for M88k ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "M88kTargetMachine.h"
#include "M88k.h"
//#include "M88kTargetObjectFile.h"
#include "TargetInfo/M88kTargetInfo.h"
#include "llvm/CodeGen/GlobalISel/IRTranslator.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelect.h"
#include "llvm/CodeGen/GlobalISel/Legalizer.h"
#include "llvm/CodeGen/GlobalISel/Localizer.h"
#include "llvm/CodeGen/GlobalISel/RegBankSelect.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

static cl::opt<bool>
    NoZeroDivCheck("m88k-no-check-zero-division", cl::Hidden,
                   cl::desc("M88k: Don't trap on integer division by zero."),
                   cl::init(false));

static cl::opt<bool> UseDivInstr(
    "m88k-use-div-instruction", cl::Hidden,
    cl::desc("M88k: Use the div instruction for signed integer division."),
    cl::init(false));

static cl::opt<bool>
    BranchRelaxation("m88k-enable-branch-relax", cl::Hidden, cl::init(true),
                     cl::desc("Relax out of range conditional branches"));

static cl::opt<cl::boolOrDefault>
    EnableDelaySlotFiller("m88k-enable-delay-slot-filler",
                          cl::desc("Fill delay slots."), cl::Hidden);

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeM88kTarget() {
  // Register the target and target specific passes.
  RegisterTargetMachine<M88kTargetMachine> X(getTheM88kTarget());
  PassRegistry &PR = *PassRegistry::getPassRegistry();
  initializeGlobalISel(PR);
  initializeM88kPreLegalizerCombinerPass(PR);
  initializeM88kPostLegalizerCombinerPass(PR);
  initializeM88kPostLegalizerLoweringPass(PR);
  initializeM88kDelaySlotFillerPass(PR);
  initializeM88kDivInstrPass(PR);
}

namespace {
// TODO: Check.
std::string computeDataLayout(const Triple &TT, StringRef CPU, StringRef FS) {
  std::string Ret;

  // Big endian.
  Ret += "E";

  // Data mangling.
  Ret += DataLayout::getManglingComponent(TT);

  // Pointers are 32 bit.
  Ret += "-p:32:32:32";

  // All scalar types are naturally aligned.
  Ret += "-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64";

  // Floats and doubles are also naturally aligned.
  Ret += "-f32:32:32-f64:64:64";

  // TODO: Add f80 for mc88110.

  // We prefer 16 bits of aligned for all globals; see above.
  Ret += "-a:8:16";

  // Integer registers are 32bits.
  Ret += "-n32";

  return Ret;
}

// TODO: Check.
Reloc::Model getEffectiveRelocModel(Optional<Reloc::Model> RM) {
  if (!RM.hasValue() || *RM == Reloc::DynamicNoPIC)
    return Reloc::Static;
  return *RM;
}

} // namespace

/// Create an M88k architecture model
M88kTargetMachine::M88kTargetMachine(const Target &T, const Triple &TT,
                                     StringRef CPU, StringRef FS,
                                     const TargetOptions &Options,
                                     Optional<Reloc::Model> RM,
                                     Optional<CodeModel::Model> CM,
                                     CodeGenOpt::Level OL, bool JIT)
    : LLVMTargetMachine(T, computeDataLayout(TT, CPU, FS), TT, CPU, FS, Options,
                        getEffectiveRelocModel(RM),
                        getEffectiveCodeModel(CM, CodeModel::Medium), OL),
      TLOF(std::make_unique<TargetLoweringObjectFileELF>()) {
  initAsmInfo();

  // Only GlobalISel is implemented. Disable the fallback mode, because there is
  // no fallback.
  setGlobalISel(true);
  setGlobalISelAbort(GlobalISelAbortMode::Enable);
}

M88kTargetMachine::~M88kTargetMachine() {}

const M88kSubtarget *
M88kTargetMachine::getSubtargetImpl(const Function &F) const {
  Attribute CPUAttr = F.getFnAttribute("target-cpu");
  Attribute TuneAttr = F.getFnAttribute("tune-cpu");
  Attribute FSAttr = F.getFnAttribute("target-features");

  std::string CPU =
      CPUAttr.isValid() ? CPUAttr.getValueAsString().str() : TargetCPU;
  std::string TuneCPU =
      TuneAttr.isValid() ? TuneAttr.getValueAsString().str() : CPU;
  std::string FS =
      FSAttr.isValid() ? FSAttr.getValueAsString().str() : TargetFS;

  auto &I = SubtargetMap[CPU + TuneCPU + FS];
  if (!I) {
    // This needs to be done before we create a new subtarget since any
    // creation will depend on the TM and the code generation flags on the
    // function that reside in TargetOptions.
    resetTargetOptions(F);
    I = std::make_unique<M88kSubtarget>(TargetTriple, CPU, TuneCPU, FS, *this);
  }

  return I.get();
}

bool M88kTargetMachine::useDivInstr() const { return UseDivInstr; }
bool M88kTargetMachine::noZeroDivCheck() const { return NoZeroDivCheck; }

namespace {
/// M88k Code Generator Pass Configuration Options.
class M88kPassConfig : public TargetPassConfig {
public:
  M88kPassConfig(M88kTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  M88kTargetMachine &getM88kTargetMachine() const {
    return getTM<M88kTargetMachine>();
  }

  void addMachineSSAOptimization() override;
  void addPreEmitPass() override;

  // GlobalISEL
  bool addIRTranslator() override;
  void addPreLegalizeMachineIR() override;
  bool addLegalizeMachineIR() override;
  void addPreRegBankSelect() override;
  bool addRegBankSelect() override;
  bool addGlobalInstructionSelect() override;
};
} // namespace

TargetPassConfig *M88kTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new M88kPassConfig(*this, PM);
}

void M88kPassConfig::addMachineSSAOptimization() {
  addPass(createM88kDivInstr(getTM<M88kTargetMachine>()));
  TargetPassConfig::addMachineSSAOptimization();
}

void M88kPassConfig::addPreEmitPass() {
  // Relax conditional branch instructions if they're otherwise out of
  // range of their destination.
  if (BranchRelaxation)
    addPass(&BranchRelaxationPassID);

  // Enable the delay slot filler for optimizing builds or if explicitly
  // requested.
  // TODO: When targetting MC88110 it might be better to not enable it.
  if ((getOptLevel() != CodeGenOpt::None &&
       EnableDelaySlotFiller != cl::BOU_FALSE) ||
      EnableDelaySlotFiller == cl::BOU_TRUE)
    addPass(createM88kDelaySlotFiller());
}

// Global ISEL
bool M88kPassConfig::addIRTranslator() {
  addPass(new IRTranslator());
  return false;
}

void M88kPassConfig::addPreLegalizeMachineIR() {
  addPass(createM88kPreLegalizerCombiner());
}

bool M88kPassConfig::addLegalizeMachineIR() {
  addPass(new Legalizer());
  return false;
}

void M88kPassConfig::addPreRegBankSelect() {
  bool IsOptNone = getOptLevel() == CodeGenOpt::None;
  addPass(createM88kPostLegalizerCombiner(IsOptNone));
  addPass(createM88kPostLegalizerLowering());
}

bool M88kPassConfig::addRegBankSelect() {
  addPass(new RegBankSelect());
  return false;
}

bool M88kPassConfig::addGlobalInstructionSelect() {
  addPass(new InstructionSelect(getOptLevel()));
  return false;
}
