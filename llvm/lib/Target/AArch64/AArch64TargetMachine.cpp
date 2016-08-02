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

#include "AArch64.h"
#include "AArch64TargetMachine.h"
#include "AArch64TargetObjectFile.h"
#include "AArch64TargetTransformInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Scalar.h"
using namespace llvm;

static cl::opt<bool>
EnableCCMP("aarch64-ccmp", cl::desc("Enable the CCMP formation pass"),
           cl::init(true), cl::Hidden);

static cl::opt<bool> EnableMCR("aarch64-mcr",
                               cl::desc("Enable the machine combiner pass"),
                               cl::init(true), cl::Hidden);

static cl::opt<bool>
EnableStPairSuppress("aarch64-stp-suppress", cl::desc("Suppress STP for AArch64"),
                     cl::init(true), cl::Hidden);

static cl::opt<bool>
EnableAdvSIMDScalar("aarch64-simd-scalar", cl::desc("Enable use of AdvSIMD scalar"
                    " integer instructions"), cl::init(false), cl::Hidden);

static cl::opt<bool>
EnablePromoteConstant("aarch64-promote-const", cl::desc("Enable the promote "
                      "constant pass"), cl::init(true), cl::Hidden);

static cl::opt<bool>
EnableCollectLOH("aarch64-collect-loh", cl::desc("Enable the pass that emits the"
                 " linker optimization hints (LOH)"), cl::init(true),
                 cl::Hidden);

static cl::opt<bool>
EnableDeadRegisterElimination("aarch64-dead-def-elimination", cl::Hidden,
                              cl::desc("Enable the pass that removes dead"
                                       " definitons and replaces stores to"
                                       " them with stores to the zero"
                                       " register"),
                              cl::init(true));

static cl::opt<bool>
EnableLoadStoreOpt("aarch64-load-store-opt", cl::desc("Enable the load/store pair"
                   " optimization pass"), cl::init(true), cl::Hidden);

static cl::opt<bool>
EnableAtomicTidy("aarch64-atomic-cfg-tidy", cl::Hidden,
                 cl::desc("Run SimplifyCFG after expanding atomic operations"
                          " to make use of cmpxchg flow-based information"),
                 cl::init(true));

static cl::opt<bool>
EnableEarlyIfConversion("aarch64-enable-early-ifcvt", cl::Hidden,
                        cl::desc("Run early if-conversion"),
                        cl::init(true));

static cl::opt<bool>
EnableCondOpt("aarch64-condopt",
              cl::desc("Enable the condition optimizer pass"),
              cl::init(true), cl::Hidden);

static cl::opt<bool>
EnableA53Fix835769("aarch64-fix-cortex-a53-835769", cl::Hidden,
                cl::desc("Work around Cortex-A53 erratum 835769"),
                cl::init(false));

static cl::opt<bool>
EnableGEPOpt("aarch64-gep-opt", cl::Hidden,
             cl::desc("Enable optimizations on complex GEPs"),
             cl::init(false));

// FIXME: Unify control over GlobalMerge.
static cl::opt<cl::boolOrDefault>
EnableGlobalMerge("aarch64-global-merge", cl::Hidden,
                  cl::desc("Enable the global merge pass"));

extern "C" void LLVMInitializeAArch64Target() {
  // Register the target.
  RegisterTargetMachine<AArch64leTargetMachine> X(TheAArch64leTarget);
  RegisterTargetMachine<AArch64beTargetMachine> Y(TheAArch64beTarget);
  RegisterTargetMachine<AArch64leTargetMachine> Z(TheARM64Target);
}

//===----------------------------------------------------------------------===//
// AArch64 Lowering public interface.
//===----------------------------------------------------------------------===//
static std::unique_ptr<TargetLoweringObjectFile> createTLOF(const Triple &TT) {
  if (TT.isOSBinFormatMachO())
    return make_unique<AArch64_MachoTargetObjectFile>();

  return make_unique<AArch64_ELFTargetObjectFile>();
}

// Helper function to build a DataLayout string
static std::string computeDataLayout(const Triple &TT, bool LittleEndian) {
  if (TT.isOSBinFormatMachO())
    return "e-m:o-i64:64-i128:128-n32:64-S128";
  if (LittleEndian)
    return "e-m:e-i64:64-i128:128-n32:64-S128";
  return "E-m:e-i64:64-i128:128-n32:64-S128";
}

/// TargetMachine ctor - Create an AArch64 architecture model.
///
AArch64TargetMachine::AArch64TargetMachine(const Target &T, const Triple &TT,
                                           StringRef CPU, StringRef FS,
                                           const TargetOptions &Options,
                                           Reloc::Model RM, CodeModel::Model CM,
                                           CodeGenOpt::Level OL,
                                           bool LittleEndian)
    // This nested ternary is horrible, but DL needs to be properly
    // initialized before TLInfo is constructed.
    : LLVMTargetMachine(T, computeDataLayout(TT, LittleEndian), TT, CPU, FS,
                        Options, RM, CM, OL),
      TLOF(createTLOF(getTargetTriple())),
      isLittle(LittleEndian) {
  initAsmInfo();
}

AArch64TargetMachine::~AArch64TargetMachine() {}

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
    const TargetOptions &Options, Reloc::Model RM, CodeModel::Model CM,
    CodeGenOpt::Level OL)
    : AArch64TargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, true) {}

void AArch64beTargetMachine::anchor() { }

AArch64beTargetMachine::AArch64beTargetMachine(
    const Target &T, const Triple &TT, StringRef CPU, StringRef FS,
    const TargetOptions &Options, Reloc::Model RM, CodeModel::Model CM,
    CodeGenOpt::Level OL)
    : AArch64TargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, false) {}

namespace {
/// AArch64 Code Generator Pass Configuration Options.
class AArch64PassConfig : public TargetPassConfig {
public:
  AArch64PassConfig(AArch64TargetMachine *TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {
    if (TM->getOptLevel() != CodeGenOpt::None)
      substitutePass(&PostRASchedulerID, &PostMachineSchedulerID);
  }

  AArch64TargetMachine &getAArch64TargetMachine() const {
    return getTM<AArch64TargetMachine>();
  }

  void addIRPasses()  override;
  bool addPreISel() override;
  bool addInstSelector() override;
  bool addILPOpts() override;
  void addPreRegAlloc() override;
  void addPostRegAlloc() override;
  void addPreSched2() override;
  void addPreEmitPass() override;
};
} // namespace

TargetIRAnalysis AArch64TargetMachine::getTargetIRAnalysis() {
  return TargetIRAnalysis([this](const Function &F) {
    return TargetTransformInfo(AArch64TTIImpl(this, F));
  });
}

TargetPassConfig *AArch64TargetMachine::createPassConfig(PassManagerBase &PM) {
  return new AArch64PassConfig(this, PM);
}

void AArch64PassConfig::addIRPasses() {
  // Always expand atomic operations, we don't deal with atomicrmw or cmpxchg
  // ourselves.
  addPass(createAtomicExpandPass(TM));

  // Cmpxchg instructions are often used with a subsequent comparison to
  // determine whether it succeeded. We can exploit existing control-flow in
  // ldrex/strex loops to simplify this, but it needs tidying up.
  if (TM->getOptLevel() != CodeGenOpt::None && EnableAtomicTidy)
    addPass(createCFGSimplificationPass());

  TargetPassConfig::addIRPasses();

  // Match interleaved memory accesses to ldN/stN intrinsics.
  if (TM->getOptLevel() != CodeGenOpt::None)
    addPass(createInterleavedAccessPass(TM));

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

  if (TM->getOptLevel() != CodeGenOpt::None)
    addPass(createAArch64AddressTypePromotionPass());

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

bool AArch64PassConfig::addILPOpts() {
  if (EnableCondOpt)
    addPass(createAArch64ConditionOptimizerPass());
  if (EnableCCMP)
    addPass(createAArch64ConditionalCompares());
  if (EnableMCR)
    addPass(&MachineCombinerID);
  if (EnableEarlyIfConversion)
    addPass(&EarlyIfConverterID);
  if (EnableStPairSuppress)
    addPass(createAArch64StorePairSuppressPass());
  return true;
}

void AArch64PassConfig::addPreRegAlloc() {
  // Use AdvSIMD scalar instructions whenever profitable.
  if (TM->getOptLevel() != CodeGenOpt::None && EnableAdvSIMDScalar) {
    addPass(createAArch64AdvSIMDScalar());
    // The AdvSIMD pass may produce copies that can be rewritten to
    // be register coaleascer friendly.
    addPass(&PeepholeOptimizerID);
  }
}

void AArch64PassConfig::addPostRegAlloc() {
  // Change dead register definitions to refer to the zero register.
  if (TM->getOptLevel() != CodeGenOpt::None && EnableDeadRegisterElimination)
    addPass(createAArch64DeadRegisterDefinitions());
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
  addPass(createAArch64BranchRelaxation());
  if (TM->getOptLevel() != CodeGenOpt::None && EnableCollectLOH &&
      TM->getTargetTriple().isOSBinFormatMachO())
    addPass(createAArch64CollectLOHPass());
}
