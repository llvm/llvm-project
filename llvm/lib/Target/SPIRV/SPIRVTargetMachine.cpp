//===- SPIRVTargetMachine.cpp - Define TargetMachine for SPIR-V -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the info about SPIR-V target spec.
//
//===----------------------------------------------------------------------===//

#include "SPIRVTargetMachine.h"
#include "SPIRV.h"
#include "SPIRVCallLowering.h"
#include "SPIRVGlobalRegistry.h"
#include "SPIRVLegalizerInfo.h"
#include "SPIRVStructurizerWrapper.h"
#include "SPIRVTargetObjectFile.h"
#include "SPIRVTargetTransformInfo.h"
#include "TargetInfo/SPIRVTargetInfo.h"
#include "llvm/CodeGen/GlobalISel/IRTranslator.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelect.h"
#include "llvm/CodeGen/GlobalISel/Legalizer.h"
#include "llvm/CodeGen/GlobalISel/RegBankSelect.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Scalar/Reg2Mem.h"
#include "llvm/Transforms/Utils.h"
#include <optional>

using namespace llvm;

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeSPIRVTarget() {
  // Register the target.
  RegisterTargetMachine<SPIRVTargetMachine> X(getTheSPIRV32Target());
  RegisterTargetMachine<SPIRVTargetMachine> Y(getTheSPIRV64Target());
  RegisterTargetMachine<SPIRVTargetMachine> Z(getTheSPIRVLogicalTarget());

  PassRegistry &PR = *PassRegistry::getPassRegistry();
  initializeGlobalISel(PR);
  initializeSPIRVModuleAnalysisPass(PR);
  initializeSPIRVConvergenceRegionAnalysisWrapperPassPass(PR);
  initializeSPIRVStructurizerPass(PR);
  initializeSPIRVPreLegalizerCombinerPass(PR);
}

static std::string computeDataLayout(const Triple &TT) {
  const auto Arch = TT.getArch();
  // TODO: this probably needs to be revisited:
  // Logical SPIR-V has no pointer size, so any fixed pointer size would be
  // wrong. The choice to default to 32 or 64 is just motivated by another
  // memory model used for graphics: PhysicalStorageBuffer64. But it shouldn't
  // mean anything.
  if (Arch == Triple::spirv32)
    return "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-"
           "v256:256-v512:512-v1024:1024-n8:16:32:64-G1";
  if (TT.getVendor() == Triple::VendorType::AMD &&
      TT.getOS() == Triple::OSType::AMDHSA)
    return "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-"
           "v512:512-v1024:1024-n32:64-S32-G1-P4-A0";
  return "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-"
         "v512:512-v1024:1024-n8:16:32:64-G1";
}

static Reloc::Model getEffectiveRelocModel(std::optional<Reloc::Model> RM) {
  if (!RM)
    return Reloc::PIC_;
  return *RM;
}

// Pin SPIRVTargetObjectFile's vtables to this file.
SPIRVTargetObjectFile::~SPIRVTargetObjectFile() {}

SPIRVTargetMachine::SPIRVTargetMachine(const Target &T, const Triple &TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       std::optional<Reloc::Model> RM,
                                       std::optional<CodeModel::Model> CM,
                                       CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(T, computeDataLayout(TT), TT, CPU, FS, Options,
                               getEffectiveRelocModel(RM),
                               getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(std::make_unique<SPIRVTargetObjectFile>()),
      Subtarget(TT, CPU.str(), FS.str(), *this) {
  initAsmInfo();
  setGlobalISel(true);
  setFastISel(false);
  setO0WantsFastISel(false);
  setRequiresStructuredCFG(false);
}

void SPIRVTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB) {
#define GET_PASS_REGISTRY "SPIRVPassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"
}

namespace {
// SPIR-V Code Generator Pass Configuration Options.
class SPIRVPassConfig : public TargetPassConfig {
public:
  SPIRVPassConfig(SPIRVTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM), TM(TM) {}

  SPIRVTargetMachine &getSPIRVTargetMachine() const {
    return getTM<SPIRVTargetMachine>();
  }
  void addMachineSSAOptimization() override;
  void addIRPasses() override;
  void addISelPrepare() override;

  bool addIRTranslator() override;
  void addPreLegalizeMachineIR() override;
  bool addLegalizeMachineIR() override;
  bool addRegBankSelect() override;
  bool addGlobalInstructionSelect() override;

  FunctionPass *createTargetRegisterAllocator(bool) override;
  void addFastRegAlloc() override {}
  void addOptimizedRegAlloc() override {}

  void addPostRegAlloc() override;
  void addPreEmitPass() override;

private:
  const SPIRVTargetMachine &TM;
};
} // namespace

// We do not use physical registers, and maintain virtual registers throughout
// the entire pipeline, so return nullptr to disable register allocation.
FunctionPass *SPIRVPassConfig::createTargetRegisterAllocator(bool) {
  return nullptr;
}

// A place to disable passes that may break CFG.
void SPIRVPassConfig::addMachineSSAOptimization() {
  TargetPassConfig::addMachineSSAOptimization();
}

// Disable passes that break from assuming no virtual registers exist.
void SPIRVPassConfig::addPostRegAlloc() {
  // Do not work with vregs instead of physical regs.
  disablePass(&MachineCopyPropagationID);
  disablePass(&PostRAMachineSinkingID);
  disablePass(&PostRASchedulerID);
  disablePass(&FuncletLayoutID);
  disablePass(&StackMapLivenessID);
  disablePass(&PatchableFunctionID);
  disablePass(&ShrinkWrapID);
  disablePass(&LiveDebugValuesID);
  disablePass(&MachineLateInstrsCleanupID);
  disablePass(&RemoveLoadsIntoFakeUsesID);

  // Do not work with OpPhi.
  disablePass(&BranchFolderPassID);
  disablePass(&MachineBlockPlacementID);

  TargetPassConfig::addPostRegAlloc();
}

TargetTransformInfo
SPIRVTargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(SPIRVTTIImpl(this, F));
}

TargetPassConfig *SPIRVTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new SPIRVPassConfig(*this, PM);
}

void SPIRVPassConfig::addIRPasses() {
  TargetPassConfig::addIRPasses();

  if (TM.getSubtargetImpl()->isVulkanEnv()) {
    // 1.  Simplify loop for subsequent transformations. After this steps, loops
    // have the following properties:
    //  - loops have a single entry edge (pre-header to loop header).
    //  - all loop exits are dominated by the loop pre-header.
    //  - loops have a single back-edge.
    addPass(createLoopSimplifyPass());

    // 2. Removes registers whose lifetime spans across basic blocks. Also
    // removes phi nodes. This will greatly simplify the next steps.
    addPass(createRegToMemWrapperPass());

    // 3. Merge the convergence region exit nodes into one. After this step,
    // regions are single-entry, single-exit. This will help determine the
    // correct merge block.
    addPass(createSPIRVMergeRegionExitTargetsPass());

    // 4. Structurize.
    addPass(createSPIRVStructurizerPass());

    // 5. Reduce the amount of variables required by pushing some operations
    // back to virtual registers.
    addPass(createPromoteMemoryToRegisterPass());
  }

  addPass(createSPIRVRegularizerPass());
  addPass(createSPIRVPrepareFunctionsPass(TM));
  addPass(createSPIRVStripConvergenceIntrinsicsPass());
}

void SPIRVPassConfig::addISelPrepare() {
  addPass(createSPIRVEmitIntrinsicsPass(&getTM<SPIRVTargetMachine>()));
  TargetPassConfig::addISelPrepare();
}

bool SPIRVPassConfig::addIRTranslator() {
  addPass(new IRTranslator(getOptLevel()));
  return false;
}

void SPIRVPassConfig::addPreLegalizeMachineIR() {
  addPass(createSPIRVPreLegalizerCombiner());
  addPass(createSPIRVPreLegalizerPass());
}

// Use the default legalizer.
bool SPIRVPassConfig::addLegalizeMachineIR() {
  addPass(new Legalizer());
  addPass(createSPIRVPostLegalizerPass());
  return false;
}

// Do not add the RegBankSelect pass, as we only ever need virtual registers.
bool SPIRVPassConfig::addRegBankSelect() {
  disablePass(&RegBankSelect::ID);
  return false;
}

static cl::opt<bool> SPVEnableNonSemanticDI(
    "spv-emit-nonsemantic-debug-info",
    cl::desc("Emit SPIR-V NonSemantic.Shader.DebugInfo.100 instructions"),
    cl::Optional, cl::init(false));

void SPIRVPassConfig::addPreEmitPass() {
  if (SPVEnableNonSemanticDI) {
    addPass(createSPIRVEmitNonSemanticDIPass(&getTM<SPIRVTargetMachine>()));
  }
}

namespace {
// A custom subclass of InstructionSelect, which is mostly the same except from
// not requiring RegBankSelect to occur previously.
class SPIRVInstructionSelect : public InstructionSelect {
  // We don't use register banks, so unset the requirement for them
  MachineFunctionProperties getRequiredProperties() const override {
    return InstructionSelect::getRequiredProperties().reset(
        MachineFunctionProperties::Property::RegBankSelected);
  }
};
} // namespace

// Add the custom SPIRVInstructionSelect from above.
bool SPIRVPassConfig::addGlobalInstructionSelect() {
  addPass(new SPIRVInstructionSelect());
  return false;
}
