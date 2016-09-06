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
#include "AMDGPU.h"
#include "AMDGPUCallLowering.h"
#include "AMDGPUTargetObjectFile.h"
#include "AMDGPUTargetTransformInfo.h"
#include "GCNSchedStrategy.h"
#include "R600ISelLowering.h"
#include "R600InstrInfo.h"
#include "R600MachineScheduler.h"
#include "SIISelLowering.h"
#include "SIInstrInfo.h"
#include "SIMachineScheduler.h"
#include "llvm/CodeGen/GlobalISel/IRTranslator.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Vectorize.h"
#include "llvm/IR/LegacyPassManager.h"

using namespace llvm;

static cl::opt<bool> EnableR600StructurizeCFG(
  "r600-ir-structurize",
  cl::desc("Use StructurizeCFG IR pass"),
  cl::init(true));

static cl::opt<bool> EnableSROA(
  "amdgpu-sroa",
  cl::desc("Run SROA after promote alloca pass"),
  cl::ReallyHidden,
  cl::init(true));

static cl::opt<bool> EnableR600IfConvert(
  "r600-if-convert",
  cl::desc("Use if conversion pass"),
  cl::ReallyHidden,
  cl::init(true));

// Option to disable vectorizer for tests.
static cl::opt<bool> EnableLoadStoreVectorizer(
  "amdgpu-load-store-vectorizer",
  cl::desc("Enable load store vectorizer"),
  cl::init(false),
  cl::Hidden);

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
  initializeAMDGPUCodeGenPreparePass(*PR);
  initializeSIAnnotateControlFlowPass(*PR);
  initializeSIInsertWaitsPass(*PR);
  initializeSIWholeQuadModePass(*PR);
  initializeSILowerControlFlowPass(*PR);
  initializeSIInsertSkipsPass(*PR);
  initializeSIDebuggerInsertNopsPass(*PR);
  initializeAMDConvertAtomicLibCallsPass(*PR);
}

static std::unique_ptr<TargetLoweringObjectFile> createTLOF(const Triple &TT) {
  return make_unique<AMDGPUTargetObjectFile>();
}

static ScheduleDAGInstrs *createR600MachineScheduler(MachineSchedContext *C) {
  return new ScheduleDAGMILive(C, make_unique<R600SchedStrategy>());
}

static ScheduleDAGInstrs *createSIMachineScheduler(MachineSchedContext *C) {
  return new SIScheduleDAGMI(C);
}

static ScheduleDAGInstrs *
createGCNMaxOccupancyMachineScheduler(MachineSchedContext *C) {
  ScheduleDAGMILive *DAG =
      new ScheduleDAGMILive(C, make_unique<GCNMaxOccupancySchedStrategy>(C));
  DAG->addMutation(createLoadClusterDAGMutation(DAG->TII, DAG->TRI));
  return DAG;
}

static MachineSchedRegistry
R600SchedRegistry("r600", "Run R600's custom scheduler",
                   createR600MachineScheduler);

static MachineSchedRegistry
SISchedRegistry("si", "Run SI's custom scheduler",
                createSIMachineScheduler);

static MachineSchedRegistry
GCNMaxOccupancySchedRegistry("gcn-max-occupancy",
                             "Run GCN scheduler to maximize occupancy",
                             createGCNMaxOccupancyMachineScheduler);

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
  // The AMDGPU toolchain only supports generating shared objects, so we
  // must always use PIC.
  return Reloc::PIC_;
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
    IntrinsicInfo() {
  setRequiresStructuredCFG(true);
  initAsmInfo();
}

AMDGPUTargetMachine::~AMDGPUTargetMachine() { }

StringRef AMDGPUTargetMachine::getGPUName(const Function &F) const {
  Attribute GPUAttr = F.getFnAttribute("target-cpu");
  return GPUAttr.hasAttribute(Attribute::None) ?
    getTargetCPU() : GPUAttr.getValueAsString();
}

StringRef AMDGPUTargetMachine::getFeatureString(const Function &F) const {
  Attribute FSAttr = F.getFnAttribute("target-features");

  return FSAttr.hasAttribute(Attribute::None) ?
    getTargetFeatureString() :
    FSAttr.getValueAsString();
}

void AMDGPUTargetMachine::addPreLinkPasses(PassManagerBase & PM) {
  PM.add(llvm::createAMDConvertAtomicLibCallsPass());
}

//===----------------------------------------------------------------------===//
// R600 Target Machine (R600 -> Cayman)
//===----------------------------------------------------------------------===//

R600TargetMachine::R600TargetMachine(const Target &T, const Triple &TT,
                                     StringRef CPU, StringRef FS,
                                     TargetOptions Options,
                                     Optional<Reloc::Model> RM,
                                     CodeModel::Model CM, CodeGenOpt::Level OL)
  : AMDGPUTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL) {}

const R600Subtarget *R600TargetMachine::getSubtargetImpl(
  const Function &F) const {
  StringRef GPU = getGPUName(F);
  StringRef FS = getFeatureString(F);

  SmallString<128> SubtargetKey(GPU);
  SubtargetKey.append(FS);

  auto &I = SubtargetMap[SubtargetKey];
  if (!I) {
    // This needs to be done before we create a new subtarget since any
    // creation will depend on the TM and the code generation flags on the
    // function that reside in TargetOptions.
    resetTargetOptions(F);
    I = llvm::make_unique<R600Subtarget>(TargetTriple, GPU, FS, *this);
  }

  return I.get();
}

//===----------------------------------------------------------------------===//
// GCN Target Machine (SI+)
//===----------------------------------------------------------------------===//

#ifdef LLVM_BUILD_GLOBAL_ISEL
namespace {
struct SIGISelActualAccessor : public GISelAccessor {
  std::unique_ptr<AMDGPUCallLowering> CallLoweringInfo;
  const AMDGPUCallLowering *getCallLowering() const override {
    return CallLoweringInfo.get();
  }
};
} // End anonymous namespace.
#endif

GCNTargetMachine::GCNTargetMachine(const Target &T, const Triple &TT,
                                   StringRef CPU, StringRef FS,
                                   TargetOptions Options,
                                   Optional<Reloc::Model> RM,
                                   CodeModel::Model CM, CodeGenOpt::Level OL)
  : AMDGPUTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL) {}

const SISubtarget *GCNTargetMachine::getSubtargetImpl(const Function &F) const {
  StringRef GPU = getGPUName(F);
  StringRef FS = getFeatureString(F);

  SmallString<128> SubtargetKey(GPU);
  SubtargetKey.append(FS);

  auto &I = SubtargetMap[SubtargetKey];
  if (!I) {
    // This needs to be done before we create a new subtarget since any
    // creation will depend on the TM and the code generation flags on the
    // function that reside in TargetOptions.
    resetTargetOptions(F);
    I = llvm::make_unique<SISubtarget>(TargetTriple, GPU, FS, *this);

#ifndef LLVM_BUILD_GLOBAL_ISEL
    GISelAccessor *GISel = new GISelAccessor();
#else
    SIGISelActualAccessor *GISel = new SIGISelActualAccessor();
    GISel->CallLoweringInfo.reset(
      new AMDGPUCallLowering(*I->getTargetLowering()));
#endif

    I->setGISelAccessor(*GISel);
  }

  return I.get();
}

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

  void addEarlyCSEOrGVNPass();
  void addStraightLineScalarOptimizationPasses();
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

  ScheduleDAGInstrs *createMachineScheduler(
    MachineSchedContext *C) const override {
    return createR600MachineScheduler(C);
  }

  bool addPreISel() override;
  void addPreRegAlloc() override;
  void addPreSched2() override;
  void addPreEmitPass() override;
};

class GCNPassConfig final : public AMDGPUPassConfig {
public:
  GCNPassConfig(TargetMachine *TM, PassManagerBase &PM)
    : AMDGPUPassConfig(TM, PM) { }

  GCNTargetMachine &getGCNTargetMachine() const {
    return getTM<GCNTargetMachine>();
  }

  ScheduleDAGInstrs *
  createMachineScheduler(MachineSchedContext *C) const override;

  void addIRPasses() override;
  bool addPreISel() override;
  void addMachineSSAOptimization() override;
  bool addInstSelector() override;
#ifdef LLVM_BUILD_GLOBAL_ISEL
  bool addIRTranslator() override;
  bool addLegalizeMachineIR() override;
  bool addRegBankSelect() override;
  bool addGlobalInstructionSelect() override;
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
    return TargetTransformInfo(AMDGPUTTIImpl(this, F));
  });
}

void AMDGPUPassConfig::addEarlyCSEOrGVNPass() {
  if (getOptLevel() == CodeGenOpt::Aggressive)
    addPass(createGVNPass());
  else
    addPass(createEarlyCSEPass());
}

void AMDGPUPassConfig::addStraightLineScalarOptimizationPasses() {
  addPass(createSeparateConstOffsetFromGEPPass());
  addPass(createSpeculativeExecutionPass());
  // ReassociateGEPs exposes more opportunites for SLSR. See
  // the example in reassociate-geps-and-slsr.ll.
  addPass(createStraightLineStrengthReducePass());
  // SeparateConstOffsetFromGEP and SLSR creates common expressions which GVN or
  // EarlyCSE can reuse.
  addEarlyCSEOrGVNPass();
  // Run NaryReassociate after EarlyCSE/GVN to be more effective.
  addPass(createNaryReassociatePass());
  // NaryReassociate on GEPs creates redundant common expressions, so run
  // EarlyCSE after it.
  addPass(createEarlyCSEPass());
}

void AMDGPUPassConfig::addIRPasses() {
  // There is no reason to run these.
  disablePass(&StackMapLivenessID);
  disablePass(&FuncletLayoutID);
  disablePass(&PatchableFunctionID);

  // Function calls are not supported, so make sure we inline everything.
  addPass(createAMDGPUAlwaysInlinePass());
  addPass(createAlwaysInlinerLegacyPass());
  // We need to add the barrier noop pass, otherwise adding the function
  // inlining pass will cause all of the PassConfigs passes to be run
  // one function at a time, which means if we have a nodule with two
  // functions, then we will generate code for the first function
  // without ever running any passes on the second.
  addPass(createBarrierNoopPass());

  // Handle uses of OpenCL image2d_t, image3d_t and sampler_t arguments.
  addPass(createAMDGPUOpenCLImageTypeLoweringPass());

  const AMDGPUTargetMachine &TM = getAMDGPUTargetMachine();
  if (TM.getOptLevel() > CodeGenOpt::None) {
    addPass(createAMDGPUPromoteAlloca(&TM));

    if (EnableSROA)
      addPass(createSROAPass());
  }

  addStraightLineScalarOptimizationPasses();

  TargetPassConfig::addIRPasses();

  // EarlyCSE is not always strong enough to clean up what LSR produces. For
  // example, GVN can combine
  //
  //   %0 = add %a, %b
  //   %1 = add %b, %a
  //
  // and
  //
  //   %0 = shl nsw %a, 2
  //   %1 = shl %a, 2
  //
  // but EarlyCSE can do neither of them.
  if (getOptLevel() != CodeGenOpt::None)
    addEarlyCSEOrGVNPass();
}

void AMDGPUPassConfig::addCodeGenPrepare() {
  TargetPassConfig::addCodeGenPrepare();

  if (EnableLoadStoreVectorizer)
    addPass(createLoadStoreVectorizerPass());
}

bool AMDGPUPassConfig::addPreISel() {
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

  if (EnableR600StructurizeCFG)
    addPass(createStructurizeCFGPass());
  return false;
}

void R600PassConfig::addPreRegAlloc() {
  addPass(createR600VectorRegMerger(*TM));
}

void R600PassConfig::addPreSched2() {
  addPass(createR600EmitClauseMarkers(), false);
  if (EnableR600IfConvert)
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

ScheduleDAGInstrs *GCNPassConfig::createMachineScheduler(
  MachineSchedContext *C) const {
  const SISubtarget &ST = C->MF->getSubtarget<SISubtarget>();
  if (ST.enableSIScheduler())
    return createSIMachineScheduler(C);
  return createGCNMaxOccupancyMachineScheduler(C);
}

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
  addPass(&SILoadStoreOptimizerID);
}

void GCNPassConfig::addIRPasses() {
  // TODO: May want to move later or split into an early and late one.
  addPass(createAMDGPUCodeGenPreparePass(&getGCNTargetMachine()));

  AMDGPUPassConfig::addIRPasses();
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

bool GCNPassConfig::addLegalizeMachineIR() {
  return false;
}

bool GCNPassConfig::addRegBankSelect() {
  return false;
}

bool GCNPassConfig::addGlobalInstructionSelect() {
  return false;
}
#endif

void GCNPassConfig::addPreRegAlloc() {

  addPass(createSIShrinkInstructionsPass());
  addPass(createSIWholeQuadModePass());
}

void GCNPassConfig::addFastRegAlloc(FunctionPass *RegAllocPass) {
  // FIXME: We have to disable the verifier here because of PHIElimination +
  // TwoAddressInstructions disabling it.
  insertPass(&TwoAddressInstructionPassID, &SILowerControlFlowID, false);

  TargetPassConfig::addFastRegAlloc(RegAllocPass);
}

void GCNPassConfig::addOptimizedRegAlloc(FunctionPass *RegAllocPass) {
  // This needs to be run directly before register allocation because earlier
  // passes might recompute live intervals.
  insertPass(&MachineSchedulerID, &SIFixControlFlowLiveIntervalsID);

  // TODO: It might be better to run this right after phi elimination, but for
  // now that would require not running the verifier.
  insertPass(&RenameIndependentSubregsID, &SILowerControlFlowID);

  TargetPassConfig::addOptimizedRegAlloc(RegAllocPass);
}

void GCNPassConfig::addPreSched2() {
}

void GCNPassConfig::addPreEmitPass() {
  // The hazard recognizer that runs as part of the post-ra scheduler does not
  // guarantee to be able handle all hazards correctly. This is because if there
  // are multiple scheduling regions in a basic block, the regions are scheduled
  // bottom up, so when we begin to schedule a region we don't know what
  // instructions were emitted directly before it.
  //
  // Here we add a stand-alone hazard recognizer pass which can handle all
  // cases.
  addPass(&PostRAHazardRecognizerID);

  addPass(createSIInsertWaitsPass());
  addPass(createSIShrinkInstructionsPass());
  addPass(&SIInsertSkipsPassID);
  addPass(createSIDebuggerInsertNopsPass());
}

TargetPassConfig *GCNTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new GCNPassConfig(this, PM);
}
