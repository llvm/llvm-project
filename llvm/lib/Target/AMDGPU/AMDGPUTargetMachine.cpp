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
#include "llvm/CodeGen/GlobalISel/IRTranslator.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
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
  initializeSIShrinkInstructionsPass(*PR);
  initializeSIFixControlFlowLiveIntervalsPass(*PR);
  initializeSILoadStoreOptimizerPass(*PR);
  initializeAMDGPUAnnotateKernelFeaturesPass(*PR);
  initializeAMDGPUAnnotateUniformValuesPass(*PR);
  initializeAMDGPUPromoteAllocaPass(*PR);
  initializeSIAnnotateControlFlowPass(*PR);
  initializeSIDebuggerInsertNopsPass(*PR);
  initializeSIInsertWaitsPass(*PR);
  initializeSIWholeQuadModePass(*PR);
  initializeSILowerControlFlowPass(*PR);
  initializeSIDebuggerInsertNopsPass(*PR);
}

static std::unique_ptr<TargetLoweringObjectFile> createTLOF(const Triple &TT) {
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

static StringRef computeDataLayout(const Triple &TT) {
  if (TT.getArch() == Triple::r600) {
    // 32-bit pointers.
    return "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
            "-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64";
  }

  // 32-bit private, local, and region pointers. 64-bit global, constant and
  // flat.
  return "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32"
         "-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
         "-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64";
}

LLVM_READNONE
static StringRef getGPUOrDefault(const Triple &TT, StringRef GPU) {
  if (!GPU.empty())
    return GPU;

  // HSA only supports CI+, so change the default GPU to a CI for HSA.
  if (TT.getArch() == Triple::amdgcn)
    return (TT.getOS() == Triple::AMDHSA) ? "kaveri" : "tahiti";

  return "r600";
}

static Reloc::Model getEffectiveRelocModel(Optional<Reloc::Model> RM) {
  if (!RM.hasValue())
    return Reloc::PIC_;
  return *RM;
}

AMDGPUTargetMachine::AMDGPUTargetMachine(const Target &T, const Triple &TT,
                                         StringRef CPU, StringRef FS,
                                         TargetOptions Options,
                                         Optional<Reloc::Model> RM,
                                         CodeModel::Model CM,
                                         CodeGenOpt::Level OptLevel)
    : LLVMTargetMachine(T, computeDataLayout(TT), TT, getGPUOrDefault(TT, CPU),
                        FS, Options, getEffectiveRelocModel(RM), CM, OptLevel),
      TLOF(createTLOF(getTargetTriple())),
      Subtarget(TT, getTargetCPU(), FS, *this), IntrinsicInfo() {
  setRequiresStructuredCFG(true);
  initAsmInfo();
}

AMDGPUTargetMachine::~AMDGPUTargetMachine() { }

//===----------------------------------------------------------------------===//
// R600 Target Machine (R600 -> Cayman)
//===----------------------------------------------------------------------===//

R600TargetMachine::R600TargetMachine(const Target &T, const Triple &TT,
                                     StringRef CPU, StringRef FS,
                                     TargetOptions Options,
                                     Optional<Reloc::Model> RM,
                                     CodeModel::Model CM, CodeGenOpt::Level OL)
    : AMDGPUTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL) {}

//===----------------------------------------------------------------------===//
// GCN Target Machine (SI+)
//===----------------------------------------------------------------------===//

GCNTargetMachine::GCNTargetMachine(const Target &T, const Triple &TT,
                                   StringRef CPU, StringRef FS,
                                   TargetOptions Options,
                                   Optional<Reloc::Model> RM,
                                   CodeModel::Model CM, CodeGenOpt::Level OL)
    : AMDGPUTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL) {}

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

class R600PassConfig final : public AMDGPUPassConfig {
public:
  R600PassConfig(TargetMachine *TM, PassManagerBase &PM)
    : AMDGPUPassConfig(TM, PM) { }

  bool addPreISel() override;
  void addPreRegAlloc() override;
  void addPreSched2() override;
  void addPreEmitPass() override;
};

class GCNPassConfig final : public AMDGPUPassConfig {
public:
  GCNPassConfig(TargetMachine *TM, PassManagerBase &PM)
    : AMDGPUPassConfig(TM, PM) { }
  bool addPreISel() override;
  void addMachineSSAOptimization() override;
  bool addInstSelector() override;
#ifdef LLVM_BUILD_GLOBAL_ISEL
  bool addIRTranslator() override;
  bool addRegBankSelect() override;
#endif
  void addFastRegAlloc(FunctionPass *RegAllocPass) override;
  void addOptimizedRegAlloc(FunctionPass *RegAllocPass) override;
  void addPreRegAlloc() override;
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
  // There is no reason to run these.
  disablePass(&StackMapLivenessID);
  disablePass(&FuncletLayoutID);
  disablePass(&PatchableFunctionID);

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
  const AMDGPUTargetMachine &TM = getAMDGPUTargetMachine();
  const AMDGPUSubtarget &ST = *TM.getSubtargetImpl();
  if (TM.getOptLevel() > CodeGenOpt::None && ST.isPromoteAllocaEnabled()) {
    addPass(createAMDGPUPromoteAlloca(&TM));
    addPass(createSROAPass());
  }
  TargetPassConfig::addCodeGenPrepare();
}

bool
AMDGPUPassConfig::addPreISel() {
  addPass(createFlattenCFGPass());
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
  const AMDGPUSubtarget &ST = *getAMDGPUTargetMachine().getSubtargetImpl();
  if (ST.IsIRStructurizerEnabled())
    addPass(createStructurizeCFGPass());
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
  addPass(createStructurizeCFGPass(true)); // true -> SkipUniformRegions
  addPass(createSinkingPass());
  addPass(createSITypeRewriter());
  addPass(createAMDGPUAnnotateUniformValues());
  addPass(createSIAnnotateControlFlowPass());

  return false;
}

void GCNPassConfig::addMachineSSAOptimization() {
  TargetPassConfig::addMachineSSAOptimization();

  // We want to fold operands after PeepholeOptimizer has run (or as part of
  // it), because it will eliminate extra copies making it easier to fold the
  // real source operand. We want to eliminate dead instructions after, so that
  // we see fewer uses of the copies. We then need to clean up the dead
  // instructions leftover after the operands are folded as well.
  //
  // XXX - Can we get away without running DeadMachineInstructionElim again?
  addPass(&SIFoldOperandsID);
  addPass(&DeadMachineInstructionElimID);
}

bool GCNPassConfig::addInstSelector() {
  AMDGPUPassConfig::addInstSelector();
  addPass(createSILowerI1CopiesPass());
  addPass(&SIFixSGPRCopiesID);
  return false;
}

#ifdef LLVM_BUILD_GLOBAL_ISEL
bool GCNPassConfig::addIRTranslator() {
  addPass(new IRTranslator());
  return false;
}

bool GCNPassConfig::addRegBankSelect() {
  return false;
}
#endif

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
  addPass(createSIWholeQuadModePass());
}

void GCNPassConfig::addFastRegAlloc(FunctionPass *RegAllocPass) {
  TargetPassConfig::addFastRegAlloc(RegAllocPass);
}

void GCNPassConfig::addOptimizedRegAlloc(FunctionPass *RegAllocPass) {
  TargetPassConfig::addOptimizedRegAlloc(RegAllocPass);
}

void GCNPassConfig::addPreSched2() {
}

void GCNPassConfig::addPreEmitPass() {

  // The hazard recognizer that runs as part of the post-ra scheduler does not
  // gaurantee to be able handle all hazards correctly.  This is because
  // if there are multiple scheduling regions in a basic block, the regions
  // are scheduled bottom up, so when we begin to schedule a region we don't
  // know what instructions were emitted directly before it.
  //
  // Here we add a stand-alone hazard recognizer pass which can handle all cases.
  // hazard recognizer pass.
  addPass(&PostRAHazardRecognizerID);

  addPass(createSIInsertWaitsPass());
  addPass(createSIShrinkInstructionsPass());
  addPass(createSILowerControlFlowPass(), false);
  addPass(createSIDebuggerInsertNopsPass(), false);
}

TargetPassConfig *GCNTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new GCNPassConfig(this, PM);
}
