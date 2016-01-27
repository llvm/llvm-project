//===-- AMDGPUTargetMachine.cpp - TargetMachine for hw codegen targets-----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief The AMDGPU target machine contains all of the hardware specific
/// information  needed to emit code for R600 and SI GPUs.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUTargetMachine.h"
#include "AMDGPUTargetObjectFile.h"
#include "AMDGPU.h"
#include "AMDGPUTargetTransformInfo.h"
#include "R600ISelLowering.h"
#include "R600InstrInfo.h"
#include "R600MachineScheduler.h"
#include "SIISelLowering.h"
#include "SIInstrInfo.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include <llvm/CodeGen/Passes.h>

using namespace llvm;

extern "C" void LLVMInitializeAMDGPUTarget() {
  // Register the target
  RegisterTargetMachine<R600TargetMachine> X(TheAMDGPUTarget);
  RegisterTargetMachine<GCNTargetMachine> Y(TheGCNTarget);

  PassRegistry *PR = PassRegistry::getPassRegistry();
  initializeSILowerI1CopiesPass(*PR);
  initializeSIFixSGPRCopiesPass(*PR);
  initializeSIFoldOperandsPass(*PR);
  initializeSIFixSGPRLiveRangesPass(*PR);
  initializeSIFixControlFlowLiveIntervalsPass(*PR);
  initializeSILoadStoreOptimizerPass(*PR);
  initializeAMDGPUAnnotateKernelFeaturesPass(*PR);
  initializeAMDGPUAnnotateUniformValuesPass(*PR);
  initializeSIAnnotateControlFlowPass(*PR);
}

static std::unique_ptr<TargetLoweringObjectFile> createTLOF(const Triple &TT) {
  if (TT.getOS() == Triple::AMDHSA)
    return make_unique<AMDGPUHSATargetObjectFile>();

  return make_unique<AMDGPUTargetObjectFile>();
}

static ScheduleDAGInstrs *createR600MachineScheduler(MachineSchedContext *C) {
  return new ScheduleDAGMILive(C, make_unique<R600SchedStrategy>());
}

static MachineSchedRegistry
R600SchedRegistry("r600", "Run R600's custom scheduler",
                   createR600MachineScheduler);

static MachineSchedRegistry
SISchedRegistry("si", "Run SI's custom scheduler",
                createSIMachineScheduler);

static std::string computeDataLayout(const Triple &TT) {
  std::string Ret = "e-p:32:32";

  if (TT.getArch() == Triple::amdgcn) {
    // 32-bit private, local, and region pointers. 64-bit global and constant.
    Ret += "-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-p24:64:64";
  }

  Ret += "-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256"
         "-v512:512-v1024:1024-v2048:2048-n32:64";

  return Ret;
}

LLVM_READNONE
static StringRef getGPUOrDefault(const Triple &TT, StringRef GPU) {
  if (!GPU.empty())
    return GPU;

  // HSA only supports CI+, so change the default GPU to a CI for HSA.
  if (TT.getArch() == Triple::amdgcn)
    return (TT.getOS() == Triple::AMDHSA) ? "kaveri" : "tahiti";

  return "";
}

AMDGPUTargetMachine::AMDGPUTargetMachine(const Target &T, const Triple &TT,
                                         StringRef CPU, StringRef FS,
                                         TargetOptions Options, Reloc::Model RM,
                                         CodeModel::Model CM,
                                         CodeGenOpt::Level OptLevel)
    : LLVMTargetMachine(T, computeDataLayout(TT), TT,
                        getGPUOrDefault(TT, CPU), FS, Options, RM, CM,
                        OptLevel),
      TLOF(createTLOF(getTargetTriple())),
      Subtarget(TT, getTargetCPU(), FS, *this),
      IntrinsicInfo() {
  setRequiresStructuredCFG(true);
  initAsmInfo();
}

AMDGPUTargetMachine::~AMDGPUTargetMachine() { }

//===----------------------------------------------------------------------===//
// R600 Target Machine (R600 -> Cayman)
//===----------------------------------------------------------------------===//

R600TargetMachine::R600TargetMachine(const Target &T, const Triple &TT,
                                     StringRef FS, StringRef CPU,
                                     TargetOptions Options, Reloc::Model RM,
                                     CodeModel::Model CM, CodeGenOpt::Level OL)
    : AMDGPUTargetMachine(T, TT, FS, CPU, Options, RM, CM, OL) {}

//===----------------------------------------------------------------------===//
// GCN Target Machine (SI+)
//===----------------------------------------------------------------------===//

GCNTargetMachine::GCNTargetMachine(const Target &T, const Triple &TT,
                                   StringRef FS, StringRef CPU,
                                   TargetOptions Options, Reloc::Model RM,
                                   CodeModel::Model CM, CodeGenOpt::Level OL)
    : AMDGPUTargetMachine(T, TT, FS, CPU, Options, RM, CM, OL) {}

//===----------------------------------------------------------------------===//
// AMDGPU Pass Setup
//===----------------------------------------------------------------------===//

namespace {
class AMDGPUPassConfig : public TargetPassConfig {
public:
  AMDGPUPassConfig(TargetMachine *TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {

    // Exceptions and StackMaps are not supported, so these passes will never do
    // anything.
    disablePass(&StackMapLivenessID);
    disablePass(&FuncletLayoutID);
  }

  AMDGPUTargetMachine &getAMDGPUTargetMachine() const {
    return getTM<AMDGPUTargetMachine>();
  }

  ScheduleDAGInstrs *
  createMachineScheduler(MachineSchedContext *C) const override {
    const AMDGPUSubtarget &ST = *getAMDGPUTargetMachine().getSubtargetImpl();
    if (ST.getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
      return createR600MachineScheduler(C);
    else if (ST.enableSIScheduler())
      return createSIMachineScheduler(C);
    return nullptr;
  }

  void addIRPasses() override;
  void addCodeGenPrepare() override;
  bool addPreISel() override;
  bool addInstSelector() override;
  bool addGCPasses() override;
};

class R600PassConfig : public AMDGPUPassConfig {
public:
  R600PassConfig(TargetMachine *TM, PassManagerBase &PM)
    : AMDGPUPassConfig(TM, PM) { }

  bool addPreISel() override;
  void addPreRegAlloc() override;
  void addPreSched2() override;
  void addPreEmitPass() override;
};

class GCNPassConfig : public AMDGPUPassConfig {
public:
  GCNPassConfig(TargetMachine *TM, PassManagerBase &PM)
    : AMDGPUPassConfig(TM, PM) { }
  bool addPreISel() override;
  bool addInstSelector() override;
  void addFastRegAlloc(FunctionPass *RegAllocPass) override;
  void addOptimizedRegAlloc(FunctionPass *RegAllocPass) override;
  void addPreRegAlloc() override;
  void addPostRegAlloc() override;
  void addPreSched2() override;
  void addPreEmitPass() override;
};

} // End of anonymous namespace

TargetIRAnalysis AMDGPUTargetMachine::getTargetIRAnalysis() {
  return TargetIRAnalysis([this](const Function &F) {
    return TargetTransformInfo(
        AMDGPUTTIImpl(this, F.getParent()->getDataLayout()));
  });
}

void AMDGPUPassConfig::addIRPasses() {
  // Function calls are not supported, so make sure we inline everything.
  addPass(createAMDGPUAlwaysInlinePass());
  addPass(createAlwaysInlinerPass());
  // We need to add the barrier noop pass, otherwise adding the function
  // inlining pass will cause all of the PassConfigs passes to be run
  // one function at a time, which means if we have a nodule with two
  // functions, then we will generate code for the first function
  // without ever running any passes on the second.
  addPass(createBarrierNoopPass());

  // Handle uses of OpenCL image2d_t, image3d_t and sampler_t arguments.
  addPass(createAMDGPUOpenCLImageTypeLoweringPass());

  TargetPassConfig::addIRPasses();
}

void AMDGPUPassConfig::addCodeGenPrepare() {
  const AMDGPUSubtarget &ST = *getAMDGPUTargetMachine().getSubtargetImpl();
  if (ST.isPromoteAllocaEnabled()) {
    addPass(createAMDGPUPromoteAlloca(ST));
    addPass(createSROAPass());
  }
  TargetPassConfig::addCodeGenPrepare();
}

bool
AMDGPUPassConfig::addPreISel() {
  const AMDGPUSubtarget &ST = *getAMDGPUTargetMachine().getSubtargetImpl();
  addPass(createFlattenCFGPass());
  if (ST.IsIRStructurizerEnabled())
    addPass(createStructurizeCFGPass());
  return false;
}

bool AMDGPUPassConfig::addInstSelector() {
  addPass(createAMDGPUISelDag(getAMDGPUTargetMachine()));
  return false;
}

bool AMDGPUPassConfig::addGCPasses() {
  // Do nothing. GC is not supported.
  return false;
}

//===----------------------------------------------------------------------===//
// R600 Pass Setup
//===----------------------------------------------------------------------===//

bool R600PassConfig::addPreISel() {
  AMDGPUPassConfig::addPreISel();
  addPass(createR600TextureIntrinsicsReplacer());
  return false;
}

void R600PassConfig::addPreRegAlloc() {
  addPass(createR600VectorRegMerger(*TM));
}

void R600PassConfig::addPreSched2() {
  const AMDGPUSubtarget &ST = *getAMDGPUTargetMachine().getSubtargetImpl();
  addPass(createR600EmitClauseMarkers(), false);
  if (ST.isIfCvtEnabled())
    addPass(&IfConverterID, false);
  addPass(createR600ClauseMergePass(*TM), false);
}

void R600PassConfig::addPreEmitPass() {
  addPass(createAMDGPUCFGStructurizerPass(), false);
  addPass(createR600ExpandSpecialInstrsPass(*TM), false);
  addPass(&FinalizeMachineBundlesID, false);
  addPass(createR600Packetizer(*TM), false);
  addPass(createR600ControlFlowFinalizer(*TM), false);
}

TargetPassConfig *R600TargetMachine::createPassConfig(PassManagerBase &PM) {
  return new R600PassConfig(this, PM);
}

//===----------------------------------------------------------------------===//
// GCN Pass Setup
//===----------------------------------------------------------------------===//

bool GCNPassConfig::addPreISel() {
  AMDGPUPassConfig::addPreISel();

  // FIXME: We need to run a pass to propagate the attributes when calls are
  // supported.
  addPass(&AMDGPUAnnotateKernelFeaturesID);

  addPass(createSinkingPass());
  addPass(createSITypeRewriter());
  addPass(createSIAnnotateControlFlowPass());
  addPass(createAMDGPUAnnotateUniformValues());

  return false;
}

bool GCNPassConfig::addInstSelector() {
  AMDGPUPassConfig::addInstSelector();
  addPass(createSILowerI1CopiesPass());
  addPass(&SIFixSGPRCopiesID);
  addPass(createSIFoldOperandsPass());
  return false;
}

void GCNPassConfig::addPreRegAlloc() {
  const AMDGPUSubtarget &ST = *getAMDGPUTargetMachine().getSubtargetImpl();

  // This needs to be run directly before register allocation because
  // earlier passes might recompute live intervals.
  // TODO: handle CodeGenOpt::None; fast RA ignores spill weights set by the pass
  if (getOptLevel() > CodeGenOpt::None) {
    insertPass(&MachineSchedulerID, &SIFixControlFlowLiveIntervalsID);
  }

  if (getOptLevel() > CodeGenOpt::None && ST.loadStoreOptEnabled()) {
    // Don't do this with no optimizations since it throws away debug info by
    // merging nonadjacent loads.

    // This should be run after scheduling, but before register allocation. It
    // also need extra copies to the address operand to be eliminated.
    insertPass(&MachineSchedulerID, &SILoadStoreOptimizerID);
    insertPass(&MachineSchedulerID, &RegisterCoalescerID);
  }
  addPass(createSIShrinkInstructionsPass(), false);
}

void GCNPassConfig::addFastRegAlloc(FunctionPass *RegAllocPass) {
  addPass(&SIFixSGPRLiveRangesID);
  TargetPassConfig::addFastRegAlloc(RegAllocPass);
}

void GCNPassConfig::addOptimizedRegAlloc(FunctionPass *RegAllocPass) {
  // We want to run this after LiveVariables is computed to avoid computing them
  // twice.
  // FIXME: We shouldn't disable the verifier here. r249087 introduced a failure
  // that needs to be fixed.
  insertPass(&LiveVariablesID, &SIFixSGPRLiveRangesID, /*VerifyAfter=*/false);
  TargetPassConfig::addOptimizedRegAlloc(RegAllocPass);
}

void GCNPassConfig::addPostRegAlloc() {
  addPass(createSIShrinkInstructionsPass(), false);
}

void GCNPassConfig::addPreSched2() {
}

void GCNPassConfig::addPreEmitPass() {
  addPass(createSIInsertWaits(*TM), false);
  addPass(createSILowerControlFlowPass(*TM), false);
}

TargetPassConfig *GCNTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new GCNPassConfig(this, PM);
}
