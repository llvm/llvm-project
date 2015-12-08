//===-- PPCTargetMachine.cpp - Define TargetMachine for PowerPC -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the PowerPC target.
//
//===----------------------------------------------------------------------===//

#include "PPCTargetMachine.h"
#include "PPC.h"
#include "PPCTargetObjectFile.h"
#include "PPCTargetTransformInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Scalar.h"
using namespace llvm;

static cl::
opt<bool> DisableCTRLoops("disable-ppc-ctrloops", cl::Hidden,
                        cl::desc("Disable CTR loops for PPC"));

static cl::
opt<bool> DisablePreIncPrep("disable-ppc-preinc-prep", cl::Hidden,
                            cl::desc("Disable PPC loop preinc prep"));

static cl::opt<bool>
VSXFMAMutateEarly("schedule-ppc-vsx-fma-mutation-early",
  cl::Hidden, cl::desc("Schedule VSX FMA instruction mutation early"));

static cl::
opt<bool> DisableVSXSwapRemoval("disable-ppc-vsx-swap-removal", cl::Hidden,
                                cl::desc("Disable VSX Swap Removal for PPC"));

static cl::
opt<bool> DisableMIPeephole("disable-ppc-peephole", cl::Hidden,
                            cl::desc("Disable machine peepholes for PPC"));

static cl::opt<bool>
EnableGEPOpt("ppc-gep-opt", cl::Hidden,
             cl::desc("Enable optimizations on complex GEPs"),
             cl::init(true));

static cl::opt<bool>
EnablePrefetch("enable-ppc-prefetching",
                  cl::desc("disable software prefetching on PPC"),
                  cl::init(false), cl::Hidden);

static cl::opt<bool>
EnableExtraTOCRegDeps("enable-ppc-extra-toc-reg-deps",
                      cl::desc("Add extra TOC register dependencies"),
                      cl::init(true), cl::Hidden);

static cl::opt<bool>
EnableMachineCombinerPass("ppc-machine-combiner",
                          cl::desc("Enable the machine combiner pass"),
                          cl::init(true), cl::Hidden);

extern "C" void LLVMInitializePowerPCTarget() {
  // Register the targets
  RegisterTargetMachine<PPC32TargetMachine> A(ThePPC32Target);
  RegisterTargetMachine<PPC64TargetMachine> B(ThePPC64Target);
  RegisterTargetMachine<PPC64TargetMachine> C(ThePPC64LETarget);

  PassRegistry &PR = *PassRegistry::getPassRegistry();
  initializePPCBoolRetToIntPass(PR);
}

/// Return the datalayout string of a subtarget.
static std::string getDataLayoutString(const Triple &T) {
  bool is64Bit = T.getArch() == Triple::ppc64 || T.getArch() == Triple::ppc64le;
  std::string Ret;

  // Most PPC* platforms are big endian, PPC64LE is little endian.
  if (T.getArch() == Triple::ppc64le)
    Ret = "e";
  else
    Ret = "E";

  Ret += DataLayout::getManglingComponent(T);

  // PPC32 has 32 bit pointers. The PS3 (OS Lv2) is a PPC64 machine with 32 bit
  // pointers.
  if (!is64Bit || T.getOS() == Triple::Lv2)
    Ret += "-p:32:32";

  // Note, the alignment values for f64 and i64 on ppc64 in Darwin
  // documentation are wrong; these are correct (i.e. "what gcc does").
  if (is64Bit || !T.isOSDarwin())
    Ret += "-i64:64";
  else
    Ret += "-f64:32:64";

  // PPC64 has 32 and 64 bit registers, PPC32 has only 32 bit ones.
  if (is64Bit)
    Ret += "-n32:64";
  else
    Ret += "-n32";

  return Ret;
}

static std::string computeFSAdditions(StringRef FS, CodeGenOpt::Level OL,
                                      const Triple &TT) {
  std::string FullFS = FS;

  // Make sure 64-bit features are available when CPUname is generic
  if (TT.getArch() == Triple::ppc64 || TT.getArch() == Triple::ppc64le) {
    if (!FullFS.empty())
      FullFS = "+64bit," + FullFS;
    else
      FullFS = "+64bit";
  }

  if (OL >= CodeGenOpt::Default) {
    if (!FullFS.empty())
      FullFS = "+crbits," + FullFS;
    else
      FullFS = "+crbits";
  }

  if (OL != CodeGenOpt::None) {
    if (!FullFS.empty())
      FullFS = "+invariant-function-descriptors," + FullFS;
    else
      FullFS = "+invariant-function-descriptors";
  }

  return FullFS;
}

static std::unique_ptr<TargetLoweringObjectFile> createTLOF(const Triple &TT) {
  // If it isn't a Mach-O file then it's going to be a linux ELF
  // object file.
  if (TT.isOSDarwin())
    return make_unique<TargetLoweringObjectFileMachO>();

  return make_unique<PPC64LinuxTargetObjectFile>();
}

static PPCTargetMachine::PPCABI computeTargetABI(const Triple &TT,
                                                 const TargetOptions &Options) {
  if (Options.MCOptions.getABIName().startswith("elfv1"))
    return PPCTargetMachine::PPC_ABI_ELFv1;
  else if (Options.MCOptions.getABIName().startswith("elfv2"))
    return PPCTargetMachine::PPC_ABI_ELFv2;

  assert(Options.MCOptions.getABIName().empty() &&
         "Unknown target-abi option!");

  if (!TT.isMacOSX()) {
    switch (TT.getArch()) {
    case Triple::ppc64le:
      return PPCTargetMachine::PPC_ABI_ELFv2;
    case Triple::ppc64:
      return PPCTargetMachine::PPC_ABI_ELFv1;
    default:
      // Fallthrough.
      ;
    }
  }
  return PPCTargetMachine::PPC_ABI_UNKNOWN;
}

// The FeatureString here is a little subtle. We are modifying the feature
// string with what are (currently) non-function specific overrides as it goes
// into the LLVMTargetMachine constructor and then using the stored value in the
// Subtarget constructor below it.
PPCTargetMachine::PPCTargetMachine(const Target &T, const Triple &TT,
                                   StringRef CPU, StringRef FS,
                                   const TargetOptions &Options,
                                   Reloc::Model RM, CodeModel::Model CM,
                                   CodeGenOpt::Level OL)
    : LLVMTargetMachine(T, getDataLayoutString(TT), TT, CPU,
                        computeFSAdditions(FS, OL, TT), Options, RM, CM, OL),
      TLOF(createTLOF(getTargetTriple())),
      TargetABI(computeTargetABI(TT, Options)),
      Subtarget(TargetTriple, CPU, computeFSAdditions(FS, OL, TT), *this) {

  // For the estimates, convergence is quadratic, so we essentially double the
  // number of digits correct after every iteration. For both FRE and FRSQRTE,
  // the minimum architected relative accuracy is 2^-5. When hasRecipPrec(),
  // this is 2^-14. IEEE float has 23 digits and double has 52 digits.
  unsigned RefinementSteps = Subtarget.hasRecipPrec() ? 1 : 3,
           RefinementSteps64 = RefinementSteps + 1;

  this->Options.Reciprocals.setDefaults("sqrtf", true, RefinementSteps);
  this->Options.Reciprocals.setDefaults("vec-sqrtf", true, RefinementSteps);
  this->Options.Reciprocals.setDefaults("divf", true, RefinementSteps);
  this->Options.Reciprocals.setDefaults("vec-divf", true, RefinementSteps);

  this->Options.Reciprocals.setDefaults("sqrtd", true, RefinementSteps64);
  this->Options.Reciprocals.setDefaults("vec-sqrtd", true, RefinementSteps64);
  this->Options.Reciprocals.setDefaults("divd", true, RefinementSteps64);
  this->Options.Reciprocals.setDefaults("vec-divd", true, RefinementSteps64);

  initAsmInfo();
}

PPCTargetMachine::~PPCTargetMachine() {}

void PPC32TargetMachine::anchor() { }

PPC32TargetMachine::PPC32TargetMachine(const Target &T, const Triple &TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       Reloc::Model RM, CodeModel::Model CM,
                                       CodeGenOpt::Level OL)
    : PPCTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL) {}

void PPC64TargetMachine::anchor() { }

PPC64TargetMachine::PPC64TargetMachine(const Target &T, const Triple &TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       Reloc::Model RM, CodeModel::Model CM,
                                       CodeGenOpt::Level OL)
    : PPCTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL) {}

const PPCSubtarget *
PPCTargetMachine::getSubtargetImpl(const Function &F) const {
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
    I = llvm::make_unique<PPCSubtarget>(
        TargetTriple, CPU,
        // FIXME: It would be good to have the subtarget additions here
        // not necessary. Anything that turns them on/off (overrides) ends
        // up being put at the end of the feature string, but the defaults
        // shouldn't require adding them. Fixing this means pulling Feature64Bit
        // out of most of the target cpus in the .td file and making it set only
        // as part of initialization via the TargetTriple.
        computeFSAdditions(FS, getOptLevel(), getTargetTriple()), *this);
  }
  return I.get();
}

//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

namespace {
/// PPC Code Generator Pass Configuration Options.
class PPCPassConfig : public TargetPassConfig {
public:
  PPCPassConfig(PPCTargetMachine *TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {}

  PPCTargetMachine &getPPCTargetMachine() const {
    return getTM<PPCTargetMachine>();
  }

  void addIRPasses() override;
  bool addPreISel() override;
  bool addILPOpts() override;
  bool addInstSelector() override;
  void addMachineSSAOptimization() override;
  void addPreRegAlloc() override;
  void addPreSched2() override;
  void addPreEmitPass() override;
};
} // namespace

TargetPassConfig *PPCTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new PPCPassConfig(this, PM);
}

void PPCPassConfig::addIRPasses() {
  if (TM->getOptLevel() != CodeGenOpt::None)
    addPass(createPPCBoolRetToIntPass());
  addPass(createAtomicExpandPass(&getPPCTargetMachine()));

  // For the BG/Q (or if explicitly requested), add explicit data prefetch
  // intrinsics.
  bool UsePrefetching = TM->getTargetTriple().getVendor() == Triple::BGQ &&
                        getOptLevel() != CodeGenOpt::None;
  if (EnablePrefetch.getNumOccurrences() > 0)
    UsePrefetching = EnablePrefetch;
  if (UsePrefetching)
    addPass(createPPCLoopDataPrefetchPass());

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

  TargetPassConfig::addIRPasses();
}

bool PPCPassConfig::addPreISel() {
  if (!DisablePreIncPrep && getOptLevel() != CodeGenOpt::None)
    addPass(createPPCLoopPreIncPrepPass(getPPCTargetMachine()));

  if (!DisableCTRLoops && getOptLevel() != CodeGenOpt::None)
    addPass(createPPCCTRLoops(getPPCTargetMachine()));

  return false;
}

bool PPCPassConfig::addILPOpts() {
  addPass(&EarlyIfConverterID);

  if (EnableMachineCombinerPass)
    addPass(&MachineCombinerID);

  return true;
}

bool PPCPassConfig::addInstSelector() {
  // Install an instruction selector.
  addPass(createPPCISelDag(getPPCTargetMachine()));

#ifndef NDEBUG
  if (!DisableCTRLoops && getOptLevel() != CodeGenOpt::None)
    addPass(createPPCCTRLoopsVerify());
#endif

  addPass(createPPCVSXCopyPass());
  return false;
}

void PPCPassConfig::addMachineSSAOptimization() {
  TargetPassConfig::addMachineSSAOptimization();
  // For little endian, remove where possible the vector swap instructions
  // introduced at code generation to normalize vector element order.
  if (TM->getTargetTriple().getArch() == Triple::ppc64le &&
      !DisableVSXSwapRemoval)
    addPass(createPPCVSXSwapRemovalPass());
  // Target-specific peephole cleanups performed after instruction
  // selection.
  if (!DisableMIPeephole) {
    addPass(createPPCMIPeepholePass());
    addPass(&DeadMachineInstructionElimID);
  }
}

void PPCPassConfig::addPreRegAlloc() {
  initializePPCVSXFMAMutatePass(*PassRegistry::getPassRegistry());
  insertPass(VSXFMAMutateEarly ? &RegisterCoalescerID : &MachineSchedulerID,
             &PPCVSXFMAMutateID);
  if (getPPCTargetMachine().getRelocationModel() == Reloc::PIC_)
    addPass(createPPCTLSDynamicCallPass());
  if (EnableExtraTOCRegDeps)
    addPass(createPPCTOCRegDepsPass());
}

void PPCPassConfig::addPreSched2() {
  if (getOptLevel() != CodeGenOpt::None)
    addPass(&IfConverterID);
}

void PPCPassConfig::addPreEmitPass() {
  if (getOptLevel() != CodeGenOpt::None)
    addPass(createPPCEarlyReturnPass(), false);
  // Must run branch selection immediately preceding the asm printer.
  addPass(createPPCBranchSelectionPass(), false);
}

TargetIRAnalysis PPCTargetMachine::getTargetIRAnalysis() {
  return TargetIRAnalysis([this](const Function &F) {
    return TargetTransformInfo(PPCTTIImpl(this, F));
  });
}
