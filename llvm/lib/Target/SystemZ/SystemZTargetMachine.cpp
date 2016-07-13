//===-- SystemZTargetMachine.cpp - Define TargetMachine for SystemZ -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SystemZTargetMachine.h"
#include "SystemZTargetTransformInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"

using namespace llvm;

extern cl::opt<bool> MISchedPostRA;
extern "C" void LLVMInitializeSystemZTarget() {
  // Register the target.
  RegisterTargetMachine<SystemZTargetMachine> X(TheSystemZTarget);
}

// Determine whether we use the vector ABI.
static bool UsesVectorABI(StringRef CPU, StringRef FS) {
  // We use the vector ABI whenever the vector facility is avaiable.
  // This is the case by default if CPU is z13 or later, and can be
  // overridden via "[+-]vector" feature string elements.
  bool VectorABI = true;
  if (CPU.empty() || CPU == "generic" ||
      CPU == "z10" || CPU == "z196" || CPU == "zEC12")
    VectorABI = false;

  SmallVector<StringRef, 3> Features;
  FS.split(Features, ',', -1, false /* KeepEmpty */);
  for (auto &Feature : Features) {
    if (Feature == "vector" || Feature == "+vector")
      VectorABI = true;
    if (Feature == "-vector")
      VectorABI = false;
  }

  return VectorABI;
}

static std::string computeDataLayout(const Triple &TT, StringRef CPU,
                                     StringRef FS) {
  bool VectorABI = UsesVectorABI(CPU, FS);
  std::string Ret = "";

  // Big endian.
  Ret += "E";

  // Data mangling.
  Ret += DataLayout::getManglingComponent(TT);

  // Make sure that global data has at least 16 bits of alignment by
  // default, so that we can refer to it using LARL.  We don't have any
  // special requirements for stack variables though.
  Ret += "-i1:8:16-i8:8:16";

  // 64-bit integers are naturally aligned.
  Ret += "-i64:64";

  // 128-bit floats are aligned only to 64 bits.
  Ret += "-f128:64";

  // When using the vector ABI, 128-bit vectors are also aligned to 64 bits.
  if (VectorABI)
    Ret += "-v128:64";

  // We prefer 16 bits of aligned for all globals; see above.
  Ret += "-a:8:16";

  // Integer registers are 32 or 64 bits.
  Ret += "-n32:64";

  return Ret;
}

static Reloc::Model getEffectiveRelocModel(Optional<Reloc::Model> RM) {
  // Static code is suitable for use in a dynamic executable; there is no
  // separate DynamicNoPIC model.
  if (!RM.hasValue() || *RM == Reloc::DynamicNoPIC)
    return Reloc::Static;
  return *RM;
}

SystemZTargetMachine::SystemZTargetMachine(const Target &T, const Triple &TT,
                                           StringRef CPU, StringRef FS,
                                           const TargetOptions &Options,
                                           Optional<Reloc::Model> RM,
                                           CodeModel::Model CM,
                                           CodeGenOpt::Level OL)
    : LLVMTargetMachine(T, computeDataLayout(TT, CPU, FS), TT, CPU, FS, Options,
                        getEffectiveRelocModel(RM), CM, OL),
      TLOF(make_unique<TargetLoweringObjectFileELF>()),
      Subtarget(TT, CPU, FS, *this) {
  initAsmInfo();
}

SystemZTargetMachine::~SystemZTargetMachine() {}

namespace {
/// SystemZ Code Generator Pass Configuration Options.
class SystemZPassConfig : public TargetPassConfig {
public:
  SystemZPassConfig(SystemZTargetMachine *TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {}

  SystemZTargetMachine &getSystemZTargetMachine() const {
    return getTM<SystemZTargetMachine>();
  }

  void addIRPasses() override;
  bool addInstSelector() override;
  void addPreSched2() override;
  void addPreEmitPass() override;
};
} // end anonymous namespace

void SystemZPassConfig::addIRPasses() {
  if (getOptLevel() != CodeGenOpt::None)
    addPass(createSystemZTDCPass());

  TargetPassConfig::addIRPasses();
}

bool SystemZPassConfig::addInstSelector() {
  addPass(createSystemZISelDag(getSystemZTargetMachine(), getOptLevel()));

 if (getOptLevel() != CodeGenOpt::None)
    addPass(createSystemZLDCleanupPass(getSystemZTargetMachine()));

  return false;
}

void SystemZPassConfig::addPreSched2() {
  if (getOptLevel() != CodeGenOpt::None)
    addPass(&IfConverterID);
}

void SystemZPassConfig::addPreEmitPass() {

  // Do instruction shortening before compare elimination because some
  // vector instructions will be shortened into opcodes that compare
  // elimination recognizes.
  if (getOptLevel() != CodeGenOpt::None)
    addPass(createSystemZShortenInstPass(getSystemZTargetMachine()), false);

  // We eliminate comparisons here rather than earlier because some
  // transformations can change the set of available CC values and we
  // generally want those transformations to have priority.  This is
  // especially true in the commonest case where the result of the comparison
  // is used by a single in-range branch instruction, since we will then
  // be able to fuse the compare and the branch instead.
  //
  // For example, two-address NILF can sometimes be converted into
  // three-address RISBLG.  NILF produces a CC value that indicates whether
  // the low word is zero, but RISBLG does not modify CC at all.  On the
  // other hand, 64-bit ANDs like NILL can sometimes be converted to RISBG.
  // The CC value produced by NILL isn't useful for our purposes, but the
  // value produced by RISBG can be used for any comparison with zero
  // (not just equality).  So there are some transformations that lose
  // CC values (while still being worthwhile) and others that happen to make
  // the CC result more useful than it was originally.
  //
  // Another reason is that we only want to use BRANCH ON COUNT in cases
  // where we know that the count register is not going to be spilled.
  //
  // Doing it so late makes it more likely that a register will be reused
  // between the comparison and the branch, but it isn't clear whether
  // preventing that would be a win or not.
  if (getOptLevel() != CodeGenOpt::None)
    addPass(createSystemZElimComparePass(getSystemZTargetMachine()), false);
  addPass(createSystemZLongBranchPass(getSystemZTargetMachine()));

  // Do final scheduling after all other optimizations, to get an
  // optimal input for the decoder (branch relaxation must happen
  // after block placement).
  if (getOptLevel() != CodeGenOpt::None) {
    if (MISchedPostRA)
      addPass(&PostMachineSchedulerID);
    else
      addPass(&PostRASchedulerID);
  }
}

TargetPassConfig *SystemZTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new SystemZPassConfig(this, PM);
}

TargetIRAnalysis SystemZTargetMachine::getTargetIRAnalysis() {
  return TargetIRAnalysis([this](const Function &F) {
    return TargetTransformInfo(SystemZTTIImpl(this, F));
  });
}
