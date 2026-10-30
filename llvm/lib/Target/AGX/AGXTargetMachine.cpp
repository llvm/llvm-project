//===-- AGXTargetMachine.cpp - Define TargetMachine for AGX ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The AGX code generation pipeline. Calls are inlined by default; the opt-in
// leaf-call path keeps eligible callees. The control flow graph is
// structurized before instruction selection.
// Unlike AMDGPU, AGX has no scalar unit and no scalar branches: every
// conditional branch, uniform or not, is lowered to execution mask
// operations on the per-thread nesting counter r0l.
//
//===----------------------------------------------------------------------===//

#include "AGXTargetMachine.h"
#include "AGX.h"
#include "AGXAliasAnalysis.h"
#include "AGXMachineFunctionInfo.h"
#include "AGXTargetTransformInfo.h"
#include "TargetInfo/AGXTargetInfo.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/NaryReassociate.h"
#include "llvm/Transforms/Scalar/StraightLineStrengthReduce.h"
#include "llvm/Transforms/Scalar/Scalarizer.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Vectorize/LoadStoreVectorizer.h"

using namespace llvm;

static cl::opt<bool> AGXPruneBeforeInline(
    "agx-prune-before-inline", cl::init(true), cl::Hidden,
    cl::desc("Prune unreachable AGX device functions before forced inlining"));

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeAGXTarget() {
  RegisterTargetMachine<AGXTargetMachine> X(getTheAGXTarget());
  PassRegistry &PR = *PassRegistry::getPassRegistry();
  initializeAGXAlwaysInlinePass(PR);
  initializeAGXLowerAIRLegacyPass(PR);
  initializeAGXAAWrapperPassPass(PR);
  initializeAGXExternalAAWrapperPass(PR);
  initializeAGXPromoteKernelArgsPass(PR);
  initializeAGXCheckedAccessesPass(PR);
  initializeAGXCodeGenPreparePass(PR);
  initializeAGXStrictFPPass(PR);
  initializeAGXLowerFPEnvPass(PR);
  initializeAGXSoftF64Pass(PR);
  initializeAGXExpandWideAtomicsPass(PR);
  initializeAGXAnnotateControlFlowPass(PR);
  initializeAGXSinkAddressesPass(PR);
  initializeAGXThreadgroupAddressesPass(PR);
  initializeAGXHoistGuardedLoadsPass(PR);
  initializeAGXNarrowInductionsPass(PR);
  initializeAGXSpecialRegisterRangesPassPass(PR);
  initializeAGXVectorAccessesPass(PR);
  initializeAGXLowerGenericAddressesPass(PR);
  initializeAGXDAGToDAGISelLegacyPass(PR);
  initializeAGXFoldOperandsPass(PR);
  initializeAGXUniformConstantsPassPass(PR);
  initializeAGXHoistFromElseFlowPass(PR);
  initializeAGXLowerControlFlowPass(PR);
  initializeAGXInsertWaitsPass(PR);
  initializeAGXAsmPrinterPass(PR);
}

static Reloc::Model getEffectiveRelocModel(std::optional<Reloc::Model> RM) {
  return RM.value_or(Reloc::PIC_);
}

AGXTargetMachine::AGXTargetMachine(const Target &T, const Triple &TT,
                                   StringRef CPU, StringRef FS,
                                   const TargetOptions &Options,
                                   std::optional<Reloc::Model> RM,
                                   std::optional<CodeModel::Model> CM,
                                   CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(T, TT, CPU.empty() ? "generic" : CPU, FS,
                               Options,
                               getEffectiveRelocModel(RM),
                               getEffectiveCodeModel(CM, CodeModel::Small),
                               OL),
      TLOF(std::make_unique<TargetLoweringObjectFileELF>()) {
  setRequiresStructuredCFG(true);
  initAsmInfo();
}

AGXTargetMachine::~AGXTargetMachine() = default;

const AGXSubtarget *
AGXTargetMachine::getSubtargetImpl(const Function &F) const {
  Attribute CPUAttr = F.getFnAttribute("target-cpu");
  Attribute FSAttr = F.getFnAttribute("target-features");
  std::string CPU =
      CPUAttr.isValid() ? CPUAttr.getValueAsString().str() : TargetCPU;
  std::string FS =
      FSAttr.isValid() ? FSAttr.getValueAsString().str() : TargetFS;
  // The module flag "agx-strict" selects Mesa's instruction forms for every
  // function of the module.
  if (const auto *Flag = mdconst::extract_or_null<ConstantInt>(
          F.getParent()->getModuleFlag("agx-strict")))
    if (!Flag->isZero())
      FS = FS.empty() ? "+agx-strict" : FS + ",+agx-strict";
  if (const auto *Flag = mdconst::extract_or_null<ConstantInt>(
          F.getParent()->getModuleFlag("agx-checked")))
    if (!Flag->isZero())
      FS = FS.empty() ? "+agx-checked" : FS + ",+agx-checked";

  auto &I = SubtargetMap[CPU + FS];
  if (!I)
    I = std::make_unique<AGXSubtarget>(TargetTriple, CPU, FS, *this);
  return I.get();
}

TargetTransformInfo
AGXTargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(std::make_unique<AGXTTIImpl>(this, F));
}

MachineFunctionInfo *AGXTargetMachine::createMachineFunctionInfo(
    BumpPtrAllocator &Allocator, const Function &F,
    const TargetSubtargetInfo *STI) const {
  return AGXMachineFunctionInfo::create<AGXMachineFunctionInfo>(Allocator, F,
                                                                STI);
}

bool AGXTargetMachine::isNoopAddrSpaceCast(const DataLayout &DL, unsigned SrcAS,
                                           unsigned DestAS) const {
  // Generic, device and constant pointers all address device memory.
  auto IsDevice = [](unsigned AS) {
    return AS == AGX::GENERIC_ADDRESS || AS == AGX::DEVICE_ADDRESS ||
           AS == AGX::CONSTANT_ADDRESS;
  };
  return IsDevice(SrcAS) && IsDevice(DestAS);
}

namespace {

class AGXPassConfig : public TargetPassConfig {
public:
  AGXPassConfig(AGXTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {
    // There is no stack unwinding or shrink wrapping.
    disablePass(&StackMapLivenessID);
    disablePass(&FuncletLayoutID);
    disablePass(&PatchableFunctionID);
    disablePass(&ShrinkWrapID);
    disablePass(&PostRAMachineSinkingID);
    // Keep the block order of the structurizer: execution mask code falls
    // through from each construct into the next, as the vendor and Mesa
    // compilers lay it out.
    disablePass(&MachineBlockPlacementID);
    substitutePass(&PostRASchedulerID, &PostMachineSchedulerID);
  }

  AGXTargetMachine &getAGXTargetMachine() const {
    return getTM<AGXTargetMachine>();
  }

  void addIRPasses() override;
  bool addPreISel() override;
  bool addInstSelector() override;
  void addMachineSSAOptimization() override;
  void addPreEmitPass() override;
};

} // namespace

void AGXTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB) {
  // AIR input becomes AGX before anything optimizes it.
  PB.registerPipelineStartEPCallback(
      [this](ModulePassManager &MPM, OptimizationLevel) {
        MPM.addPass(AGXLowerAIRPass(*this));
      });
  PB.registerPipelineParsingCallback(
      [this](StringRef Name, ModulePassManager &MPM,
             ArrayRef<PassBuilder::PipelineElement>) {
        if (Name == "agx-raise-to-air") {
          MPM.addPass(AGXRaiseToAIRPass(this));
          return true;
        }
        if (Name != "agx-lower-air")
          return false;
        MPM.addPass(AGXLowerAIRPass(*this));
        return true;
      });
  PB.registerAnalysisRegistrationCallback([](FunctionAnalysisManager &FAM) {
    FAM.registerPass([] { return AGXAA(); });
  });
  PB.registerParseAACallback([](StringRef Name, AAManager &AAM) {
    if (Name != "agx-aa")
      return false;
    AAM.registerFunctionAnalysis<AGXAA>();
    return true;
  });
}

void AGXTargetMachine::registerEarlyDefaultAliasAnalyses(AAManager &AAM) {
  AAM.registerFunctionAnalysis<AGXAA>();
}

ScheduleDAGInstrs *
AGXTargetMachine::createMachineScheduler(MachineSchedContext *C) const {
  ScheduleDAGMILive *DAG = createSchedLive(C);
  DAG->addMutation(createAGXMemClusterMutation());
  return DAG;
}

ScheduleDAGInstrs *
AGXTargetMachine::createPostMachineScheduler(MachineSchedContext *C) const {
  return createAGXPostRAScheduler(C);
}

TargetPassConfig *AGXTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new AGXPassConfig(*this, PM);
}

static cl::opt<unsigned> AGXStraightLineSRMaxInstructions(
    "agx-straight-line-sr-max-instructions", cl::init(200000), cl::Hidden,
    cl::desc("Straight-line strength reduction and n-ary reassociation only "
             "for functions of at most this many instructions (0: all)"));

namespace {
/// Straight-line strength reduction, n-ary reassociation and the CSE after
/// them, on functions of at most AGXStraightLineSRMaxInstructions
/// instructions. Their scalar evolution of a kernel with millions of
/// instructions (Blender Cycles' shade_shadow, everything inlined) took
/// gigabytes, for address arithmetic sharing meant for unrolled tiles.
class AGXStraightLineSRPass : public FunctionPass {
  TargetMachine *TM;
  unsigned Level;

public:
  static char ID;
  AGXStraightLineSRPass(TargetMachine *TM, unsigned Level)
      : FunctionPass(ID), TM(TM), Level(Level) {}
  StringRef getPassName() const override {
    return "AGX straight-line strength reduction (size limited)";
  }
  bool runOnFunction(Function &F) override {
    if (skipFunction(F) || (AGXStraightLineSRMaxInstructions &&
                            F.getInstructionCount() >
                                AGXStraightLineSRMaxInstructions))
      return false;
    PassBuilder PB(TM);
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
    FunctionPassManager FPM;
    FPM.addPass(StraightLineStrengthReducePass());
    FPM.addPass(EarlyCSEPass());
    FPM.addPass(NaryReassociatePass());
    if (Level > 1)
      FPM.addPass(GVNPass());
    else
      FPM.addPass(EarlyCSEPass());
    return !FPM.run(F, FAM).areAllPreserved();
  }
};
char AGXStraightLineSRPass::ID = 0;
} // namespace

static cl::opt<unsigned> AGXStraightLineSR(
    "agx-straight-line-sr", cl::init(1), cl::Hidden,
    cl::desc("Straight-line strength reduction and n-ary reassociation of "
             "address arithmetic before vectorization (1: then EarlyCSE, "
             "2: then GVN)"));

void AGXPassConfig::addIRPasses() {
  // AIR input becomes AGX kernels first (a no-op without !air.kernel).
  addPass(createAGXLowerAIRPass(&getAGXTargetMachine()));
  // Disjoint address spaces do not alias (for every IR and machine pass that
  // asks alias analysis, the scheduler's memory dependences among them).
  addPass(createAGXAAWrapperPass());
  addPass(createAGXExternalAAWrapperPass());
  // Inline by default; the experimental leaf-call option leaves eligible
  // functions outlined.
  addPass(createAGXAlwaysInlinePass());
  // Internalization can leave entire unused call chains alive: the inliner
  // otherwise expands their bodies before discovering they have no roots.
  if (AGXPruneBeforeInline)
    addPass(createGlobalDCEPass());
  addPass(createAlwaysInlinerLegacyPass());
  // Inlining non-callable wrappers can leave outlined leaves with no users.
  // Removing them before per-function codegen is material for large PTX
  // modules, where even dead device functions may be thousands of blocks.
  if (agxExperimentalLeafCallsEnabled()) {
    addPass(createGlobalDCEPass());
    // Called functions reach the global data buffer through a parameter.
    addPass(createAGXGlobalDataArgsPass());
  }

  // 64- and 128-bit atomics take locks; the others are hardware atomics.
  addPass(createAGXExpandWideAtomicsPass(&getAGXTargetMachine()));
  addPass(createAtomicExpandLegacyPass());
  if (getOptLevel() != CodeGenOptLevel::None) {
    addPass(createInferAddressSpacesPass());
    addPass(createSROAPass());
  }
  // Checked mode: check every access, adding the hidden arguments the
  // runtime fills (before generic pointers and kernel arguments change form).
  addPass(createAGXCheckedAccessesPass(&getAGXTargetMachine()));
  // Generic pointers that may name threadgroup memory become tagged integers.
  addPass(createAGXLowerGenericAddressesPass());
  // By-reference kernel arguments become uniform register values.
  addPass(createAGXPromoteKernelArgsPass());
  // Neighbouring 32-bit accesses become one vector access (after checked
  // mode, which checks the accesses as the source wrote them); threadgroup
  // and private addresses computed in integers become base + constant first.
  if (getOptLevel() != CodeGenOptLevel::None) {
    // Address arithmetic shared between straight-line computations (the
    // unrolled tiles of agx_libs' GEMMs): each address from a neighbour's
    // plus a difference, and sums reassociated so common parts are computed
    // once, as NVPTX does (straight-line strength reduction and n-ary
    // reassociation).
    if (AGXStraightLineSR)
      addPass(new AGXStraightLineSRPass(TM, AGXStraightLineSR));
    // Loads that all-pairs loops skip for one element issue together.
    addPass(createAGXHoistGuardedLoadsPass());
    addPass(createAGXThreadgroupAddressesPass());
    addPass(createAGXVectorAccessesPass(/*BeforeVectorizer=*/true));
    addPass(createLoadStoreVectorizerPass());
    addPass(createAGXVectorAccessesPass(/*BeforeVectorizer=*/false));
  }
  // Vector arithmetic is scalar code (as in instruction selection), which the
  // software floating point lowerings below then see.
  addPass(createScalarizerPass());
  // The floating point mode register and constrained floating point
  // intrinsics (directed rounding in software).
  addPass(createAGXLowerFPEnvPass());
  addPass(createAGXSoftF64Pass());
  addPass(createAGXCodeGenPreparePass());
  // Software IEEE binary32 arithmetic and canonical NaNs, when enabled.
  addPass(createAGXStrictFPPass());

  TargetPassConfig::addIRPasses();
}

bool AGXPassConfig::addPreISel() {
  // After loop strength reduction, which makes 64-bit counters of PTX's
  // 64-bit address arithmetic.
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(createAGXNarrowInductionsPass());
  // After loop strength reduction, which would form 64-bit offsets from
  // the bounded indices (EXPERIMENTS.md L14-E1).
  addPass(createAGXSpecialRegisterRangesPass());
  // Bring the CFG into the structured form AGXAnnotateControlFlow expects.
  addPass(createLowerSwitchPass());
  addPass(createFixIrreduciblePass());
  addPass(createUnifyLoopExitsPass());
  // Every branch is lowered to execution mask operations, so uniform
  // regions are structurized too.
  addPass(createStructurizeCFGPass(/*SkipUniformRegions=*/false));
  addPass(createAGXAnnotateControlFlowPass());
  // Before LCSSA, which would put phis between addresses and their uses.
  addPass(createAGXSinkAddressesPass());
  addPass(createLCSSAPass());
  return false;
}

bool AGXPassConfig::addInstSelector() {
  addPass(createAGXISelDag(getAGXTargetMachine()));
  return false;
}

void AGXPassConfig::addMachineSSAOptimization() {
  TargetPassConfig::addMachineSSAOptimization();
  addPass(createAGXUniformConstantsPass());
  addPass(createAGXFoldOperandsPass());
  // MachineCSE and MachineSink may have put code where only the threads of
  // a then side run it.
  addPass(createAGXHoistFromElseFlowPass());
}

void AGXPassConfig::addPreEmitPass() {
  addPass(createAGXLowerControlFlowPass());
  addPass(createAGXInsertWaitsPass());
}
