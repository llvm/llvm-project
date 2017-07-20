//===-- AArch64TargetMachine.cpp - Define TargetMachine for AArch64 -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "AArch64TargetMachine.h"
#include "AArch64.h"
#include "AArch64MacroFusion.h"
#include "AArch64Subtarget.h"
#include "AArch64TargetObjectFile.h"
#include "AArch64TargetTransformInfo.h"
#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/GlobalISel/IRTranslator.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelect.h"
#include "llvm/CodeGen/GlobalISel/Legalizer.h"
#include "llvm/CodeGen/GlobalISel/Localizer.h"
#include "llvm/CodeGen/GlobalISel/RegBankSelect.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Scalar.h"
#include <memory>
#include <string>

using namespace llvm;

static cl::opt<bool> EnableCCMP("aarch64-enable-ccmp",
                                cl::desc("Enable the CCMP formation pass"),
                                cl::init(true), cl::Hidden);

static cl::opt<bool>
    EnableCondBrTuning("aarch64-enable-cond-br-tune",
                       cl::desc("Enable the conditional branch tuning pass"),
                       cl::init(true), cl::Hidden);

static cl::opt<bool> EnableMCR("aarch64-enable-mcr",
                               cl::desc("Enable the machine combiner pass"),
                               cl::init(true), cl::Hidden);

static cl::opt<bool> EnableStPairSuppress("aarch64-enable-stp-suppress",
                                          cl::desc("Suppress STP for AArch64"),
                                          cl::init(true), cl::Hidden);

static cl::opt<bool> EnableAdvSIMDScalar(
    "aarch64-enable-simd-scalar",
    cl::desc("Enable use of AdvSIMD scalar integer instructions"),
    cl::init(false), cl::Hidden);

static cl::opt<bool>
    EnablePromoteConstant("aarch64-enable-promote-const",
                          cl::desc("Enable the promote constant pass"),
                          cl::init(true), cl::Hidden);

static cl::opt<bool> EnableCollectLOH(
    "aarch64-enable-collect-loh",
    cl::desc("Enable the pass that emits the linker optimization hints (LOH)"),
    cl::init(true), cl::Hidden);

static cl::opt<bool>
    EnableDeadRegisterElimination("aarch64-enable-dead-defs", cl::Hidden,
                                  cl::desc("Enable the pass that removes dead"
                                           " definitons and replaces stores to"
                                           " them with stores to the zero"
                                           " register"),
                                  cl::init(true));

static cl::opt<bool> EnableRedundantCopyElimination(
    "aarch64-enable-copyelim",
    cl::desc("Enable the redundant copy elimination pass"), cl::init(true),
    cl::Hidden);

static cl::opt<bool> EnableLoadStoreOpt("aarch64-enable-ldst-opt",
                                        cl::desc("Enable the load/store pair"
                                                 " optimization pass"),
                                        cl::init(true), cl::Hidden);

static cl::opt<bool> EnableAtomicTidy(
    "aarch64-enable-atomic-cfg-tidy", cl::Hidden,
    cl::desc("Run SimplifyCFG after expanding atomic operations"
             " to make use of cmpxchg flow-based information"),
    cl::init(true));

static cl::opt<bool>
EnableEarlyIfConversion("aarch64-enable-early-ifcvt", cl::Hidden,
                        cl::desc("Run early if-conversion"),
                        cl::init(true));

static cl::opt<bool>
    EnableCondOpt("aarch64-enable-condopt",
                  cl::desc("Enable the condition optimizer pass"),
                  cl::init(true), cl::Hidden);

static cl::opt<bool>
EnableA53Fix835769("aarch64-fix-cortex-a53-835769", cl::Hidden,
                cl::desc("Work around Cortex-A53 erratum 835769"),
                cl::init(false));

static cl::opt<bool>
    EnableGEPOpt("aarch64-enable-gep-opt", cl::Hidden,
                 cl::desc("Enable optimizations on complex GEPs"),
                 cl::init(false));

static cl::opt<bool>
    BranchRelaxation("aarch64-enable-branch-relax", cl::Hidden, cl::init(true),
                     cl::desc("Relax out of range conditional branches"));

// FIXME: Unify control over GlobalMerge.
static cl::opt<cl::boolOrDefault>
    EnableGlobalMerge("aarch64-enable-global-merge", cl::Hidden,
                      cl::desc("Enable the global merge pass"));

static cl::opt<bool>
    EnableLoopDataPrefetch("aarch64-enable-loop-data-prefetch", cl::Hidden,
                           cl::desc("Enable the loop data prefetch pass"),
                           cl::init(true));

static cl::opt<int> EnableGlobalISelAtO(
    "aarch64-enable-global-isel-at-O", cl::Hidden,
    cl::desc("Enable GlobalISel at or below an opt level (-1 to disable)"),
    cl::init(-1));

extern "C" void LLVMInitializeAArch64Target() {
  // Register the target.
  RegisterTargetMachine<AArch64leTargetMachine> X(getTheAArch64leTarget());
  RegisterTargetMachine<AArch64beTargetMachine> Y(getTheAArch64beTarget());
  RegisterTargetMachine<AArch64leTargetMachine> Z(getTheARM64Target());
  auto PR = PassRegistry::getPassRegistry();
  initializeGlobalISel(*PR);
  initializeAArch64A53Fix835769Pass(*PR);
  initializeAArch64A57FPLoadBalancingPass(*PR);
  initializeAArch64AdvSIMDScalarPass(*PR);
  initializeAArch64CollectLOHPass(*PR);
  initializeAArch64ConditionalComparesPass(*PR);
  initializeAArch64ConditionOptimizerPass(*PR);
  initializeAArch64DeadRegisterDefinitionsPass(*PR);
  initializeAArch64ExpandPseudoPass(*PR);
  initializeAArch64LoadStoreOptPass(*PR);
  initializeAArch64VectorByElementOptPass(*PR);
  initializeAArch64PromoteConstantPass(*PR);
  initializeAArch64RedundantCopyEliminationPass(*PR);
  initializeAArch64StorePairSuppressPass(*PR);
  initializeLDTLSCleanupPass(*PR);
}

//===----------------------------------------------------------------------===//
// AArch64 Lowering public interface.
//===----------------------------------------------------------------------===//
static std::unique_ptr<TargetLoweringObjectFile> createTLOF(const Triple &TT) {
  if (TT.isOSBinFormatMachO())
    return llvm::make_unique<AArch64_MachoTargetObjectFile>();
  if (TT.isOSBinFormatCOFF())
    return llvm::make_unique<AArch64_COFFTargetObjectFile>();

  return llvm::make_unique<AArch64_ELFTargetObjectFile>();
}

// Helper function to build a DataLayout string
static std::string computeDataLayout(const Triple &TT,
                                     const MCTargetOptions &Options,
                                     bool LittleEndian) {
  if (Options.getABIName() == "ilp32")
    return "e-m:e-p:32:32-i8:8-i16:16-i64:64-S128";
  if (TT.isOSBinFormatMachO())
    return "e-m:o-i64:64-i128:128-n32:64-S128";
  if (TT.isOSBinFormatCOFF())
    return "e-m:w-i64:64-i128:128-n32:64-S128";
  if (LittleEndian)
    return "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128";
  return "E-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128";
}

static Reloc::Model getEffectiveRelocModel(const Triple &TT,
                                           Optional<Reloc::Model> RM) {
  // AArch64 Darwin is always PIC.
  if (TT.isOSDarwin())
    return Reloc::PIC_;
  // On ELF platforms the default static relocation model has a smart enough
  // linker to cope with referencing external symbols defined in a shared
  // library. Hence DynamicNoPIC doesn't need to be promoted to PIC.
  if (!RM.hasValue() || *RM == Reloc::DynamicNoPIC)
    return Reloc::Static;
  return *RM;
}

/// Create an AArch64 architecture model.
///
AArch64TargetMachine::AArch64TargetMachine(
    const Target &T, const Triple &TT, StringRef CPU, StringRef FS,
    const TargetOptions &Options, Optional<Reloc::Model> RM,
    CodeModel::Model CM, CodeGenOpt::Level OL, bool LittleEndian)
    // This nested ternary is horrible, but DL needs to be properly
    // initialized before TLInfo is constructed.
    : LLVMTargetMachine(T, computeDataLayout(TT, Options.MCOptions,
                                             LittleEndian),
                        TT, CPU, FS, Options,
			getEffectiveRelocModel(TT, RM), CM, OL),
      TLOF(createTLOF(getTargetTriple())),
      isLittle(LittleEndian) {
  initAsmInfo();
}

AArch64TargetMachine::~AArch64TargetMachine() = default;

const AArch64Subtarget *
AArch64TargetMachine::getSubtargetImpl(const Function &F) const {
  Attribute CPUAttr = F.getFnAttribute("target-cpu");
  Attribute FSAttr = F.getFnAttribute("target-features");

  std::string CPU = !CPUAttr.hasAttribute(Attribute::None)
                        ? CPUAttr.getValueAsString().str()
                        : TargetCPU;
  std::string FS = !FSAttr.hasAttribute(Attribute::None)
                       ? FSAttr.getValueAsString().str()
                       : TargetFS;

  auto &I = SubtargetMap[CPU + FS];
  if (!I) {
    // This needs to be done before we create a new subtarget since any
    // creation will depend on the TM and the code generation flags on the
    // function that reside in TargetOptions.
    resetTargetOptions(F);
    I = llvm::make_unique<AArch64Subtarget>(TargetTriple, CPU, FS, *this,
                                            isLittle);
  }
  return I.get();
}

void AArch64leTargetMachine::anchor() { }

AArch64leTargetMachine::AArch64leTargetMachine(
    const Target &T, const Triple &TT, StringRef CPU, StringRef FS,
    const TargetOptions &Options, Optional<Reloc::Model> RM,
    CodeModel::Model CM, CodeGenOpt::Level OL)
    : AArch64TargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, true) {}

void AArch64beTargetMachine::anchor() { }

AArch64beTargetMachine::AArch64beTargetMachine(
    const Target &T, const Triple &TT, StringRef CPU, StringRef FS,
    const TargetOptions &Options, Optional<Reloc::Model> RM,
    CodeModel::Model CM, CodeGenOpt::Level OL)
    : AArch64TargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, false) {}

namespace {

/// AArch64 Code Generator Pass Configuration Options.
class AArch64PassConfig : public TargetPassConfig {
public:
  AArch64PassConfig(AArch64TargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {
    if (TM.getOptLevel() != CodeGenOpt::None)
      substitutePass(&PostRASchedulerID, &PostMachineSchedulerID);
  }

  AArch64TargetMachine &getAArch64TargetMachine() const {
    return getTM<AArch64TargetMachine>();
  }

  ScheduleDAGInstrs *
  createMachineScheduler(MachineSchedContext *C) const override {
    const AArch64Subtarget &ST = C->MF->getSubtarget<AArch64Subtarget>();
    ScheduleDAGMILive *DAG = createGenericSchedLive(C);
    DAG->addMutation(createLoadClusterDAGMutation(DAG->TII, DAG->TRI));
    DAG->addMutation(createStoreClusterDAGMutation(DAG->TII, DAG->TRI));
    if (ST.hasFusion())
      DAG->addMutation(createAArch64MacroFusionDAGMutation());
    return DAG;
  }

  ScheduleDAGInstrs *
  createPostMachineScheduler(MachineSchedContext *C) const override {
    const AArch64Subtarget &ST = C->MF->getSubtarget<AArch64Subtarget>();
    if (ST.hasFusion()) {
      // Run the Macro Fusion after RA again since literals are expanded from
      // pseudos then (v. addPreSched2()).
      ScheduleDAGMI *DAG = createGenericSchedPostRA(C);
      DAG->addMutation(createAArch64MacroFusionDAGMutation());
      return DAG;
    }

    return nullptr;
  }

  void addIRPasses()  override;
  bool addPreISel() override;
  bool addInstSelector() override;
#ifdef LLVM_BUILD_GLOBAL_ISEL
  bool addIRTranslator() override;
  bool addLegalizeMachineIR() override;
  bool addRegBankSelect() override;
  void addPreGlobalInstructionSelect() override;
  bool addGlobalInstructionSelect() override;
#endif
  bool addILPOpts() override;
  void addPreRegAlloc() override;
  void addPostRegAlloc() override;
  void addPreSched2() override;
  void addPreEmitPass() override;

  bool isGlobalISelEnabled() const override;
};

} // end anonymous namespace

TargetIRAnalysis AArch64TargetMachine::getTargetIRAnalysis() {
  return TargetIRAnalysis([this](const Function &F) {
    return TargetTransformInfo(AArch64TTIImpl(this, F));
  });
}

TargetPassConfig *AArch64TargetMachine::createPassConfig(PassManagerBase &PM) {
  return new AArch64PassConfig(*this, PM);
}

void AArch64PassConfig::addIRPasses() {
  // Always expand atomic operations, we don't deal with atomicrmw or cmpxchg
  // ourselves.
  addPass(createAtomicExpandPass());

  // Cmpxchg instructions are often used with a subsequent comparison to
  // determine whether it succeeded. We can exploit existing control-flow in
  // ldrex/strex loops to simplify this, but it needs tidying up.
  if (TM->getOptLevel() != CodeGenOpt::None && EnableAtomicTidy)
    addPass(createCFGSimplificationPass());

  // Run LoopDataPrefetch
  //
  // Run this before LSR to remove the multiplies involved in computing the
  // pointer values N iterations ahead.
  if (TM->getOptLevel() != CodeGenOpt::None && EnableLoopDataPrefetch)
    addPass(createLoopDataPrefetchPass());

  TargetPassConfig::addIRPasses();

  // Match interleaved memory accesses to ldN/stN intrinsics.
  if (TM->getOptLevel() != CodeGenOpt::None)
    addPass(createInterleavedAccessPass());

  if (TM->getOptLevel() == CodeGenOpt::Aggressive && EnableGEPOpt) {
    // Call SeparateConstOffsetFromGEP pass to extract constants within indices
    // and lower a GEP with multiple indices to either arithmetic operations or
    // multiple GEPs with single index.
    addPass(createSeparateConstOffsetFromGEPPass(TM, true));
    // Call EarlyCSE pass to find and remove subexpressions in the lowered
    // result.
    addPass(createEarlyCSEPass());
    // Do loop invariant code motion in case part of the lowered result is
    // invariant.
    addPass(createLICMPass());
  }
}

// Pass Pipeline Configuration
bool AArch64PassConfig::addPreISel() {
  // Run promote constant before global merge, so that the promoted constants
  // get a chance to be merged
  if (TM->getOptLevel() != CodeGenOpt::None && EnablePromoteConstant)
    addPass(createAArch64PromoteConstantPass());
  // FIXME: On AArch64, this depends on the type.
  // Basically, the addressable offsets are up to 4095 * Ty.getSizeInBytes().
  // and the offset has to be a multiple of the related size in bytes.
  if ((TM->getOptLevel() != CodeGenOpt::None &&
       EnableGlobalMerge == cl::BOU_UNSET) ||
      EnableGlobalMerge == cl::BOU_TRUE) {
    bool OnlyOptimizeForSize = (TM->getOptLevel() < CodeGenOpt::Aggressive) &&
                               (EnableGlobalMerge == cl::BOU_UNSET);
    addPass(createGlobalMergePass(TM, 4095, OnlyOptimizeForSize));
  }

  return false;
}

bool AArch64PassConfig::addInstSelector() {
  addPass(createAArch64ISelDag(getAArch64TargetMachine(), getOptLevel()));

  // For ELF, cleanup any local-dynamic TLS accesses (i.e. combine as many
  // references to _TLS_MODULE_BASE_ as possible.
  if (TM->getTargetTriple().isOSBinFormatELF() &&
      getOptLevel() != CodeGenOpt::None)
    addPass(createAArch64CleanupLocalDynamicTLSPass());

  return false;
}

#ifdef LLVM_BUILD_GLOBAL_ISEL
bool AArch64PassConfig::addIRTranslator() {
  addPass(new IRTranslator());
  return false;
}

bool AArch64PassConfig::addLegalizeMachineIR() {
  addPass(new Legalizer());
  return false;
}

bool AArch64PassConfig::addRegBankSelect() {
  addPass(new RegBankSelect());
  return false;
}

void AArch64PassConfig::addPreGlobalInstructionSelect() {
  // Workaround the deficiency of the fast register allocator.
  if (TM->getOptLevel() == CodeGenOpt::None)
    addPass(new Localizer());
}

bool AArch64PassConfig::addGlobalInstructionSelect() {
  addPass(new InstructionSelect());
  return false;
}
#endif

bool AArch64PassConfig::isGlobalISelEnabled() const {
  return TM->getOptLevel() <= EnableGlobalISelAtO;
}

bool AArch64PassConfig::addILPOpts() {
  if (EnableCondOpt)
    addPass(createAArch64ConditionOptimizerPass());
  if (EnableCCMP)
    addPass(createAArch64ConditionalCompares());
  if (EnableMCR)
    addPass(&MachineCombinerID);
  if (EnableCondBrTuning)
    addPass(createAArch64CondBrTuning());
  if (EnableEarlyIfConversion)
    addPass(&EarlyIfConverterID);
  if (EnableStPairSuppress)
    addPass(createAArch64StorePairSuppressPass());
  addPass(createAArch64VectorByElementOptPass());
  return true;
}

void AArch64PassConfig::addPreRegAlloc() {
  // Change dead register definitions to refer to the zero register.
  if (TM->getOptLevel() != CodeGenOpt::None && EnableDeadRegisterElimination)
    addPass(createAArch64DeadRegisterDefinitions());

  // Use AdvSIMD scalar instructions whenever profitable.
  if (TM->getOptLevel() != CodeGenOpt::None && EnableAdvSIMDScalar) {
    addPass(createAArch64AdvSIMDScalar());
    // The AdvSIMD pass may produce copies that can be rewritten to
    // be register coaleascer friendly.
    addPass(&PeepholeOptimizerID);
  }
}

void AArch64PassConfig::addPostRegAlloc() {
  // Remove redundant copy instructions.
  if (TM->getOptLevel() != CodeGenOpt::None && EnableRedundantCopyElimination)
    addPass(createAArch64RedundantCopyEliminationPass());

  if (TM->getOptLevel() != CodeGenOpt::None && usingDefaultRegAlloc())
    // Improve performance for some FP/SIMD code for A57.
    addPass(createAArch64A57FPLoadBalancing());
}

void AArch64PassConfig::addPreSched2() {
  // Expand some pseudo instructions to allow proper scheduling.
  addPass(createAArch64ExpandPseudoPass());
  // Use load/store pair instructions when possible.
  if (TM->getOptLevel() != CodeGenOpt::None && EnableLoadStoreOpt)
    addPass(createAArch64LoadStoreOptimizationPass());
}

void AArch64PassConfig::addPreEmitPass() {
  if (EnableA53Fix835769)
    addPass(createAArch64A53Fix835769());
  // Relax conditional branch instructions if they're otherwise out of
  // range of their destination.
  if (BranchRelaxation)
    addPass(&BranchRelaxationPassID);

  if (TM->getOptLevel() != CodeGenOpt::None && EnableCollectLOH &&
      TM->getTargetTriple().isOSBinFormatMachO())
    addPass(createAArch64CollectLOHPass());
}
