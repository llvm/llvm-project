//===-- SystemZTargetMachine.cpp - Define TargetMachine for SystemZ -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemZTargetMachine.h"
#include "MCTargetDesc/SystemZMCTargetDesc.h"
#include "SystemZ.h"
#include "SystemZMachineScheduler.h"
#include "SystemZTargetTransformInfo.h"
#include "TargetInfo/SystemZTargetInfo.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Transforms/Scalar.h"
#include <string>

using namespace llvm;

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeSystemZTarget() {
  // Register the target.
  RegisterTargetMachine<SystemZTargetMachine> X(getTheSystemZTarget());
  auto &PR = *PassRegistry::getPassRegistry();
  initializeSystemZElimComparePass(PR);
  initializeSystemZShortenInstPass(PR);
  initializeSystemZLongBranchPass(PR);
  initializeSystemZLDCleanupPass(PR);
  initializeSystemZShortenInstPass(PR);
  initializeSystemZPostRewritePass(PR);
  initializeSystemZTDCPassPass(PR);
}

// Determine whether we use the vector ABI.
static bool UsesVectorABI(StringRef CPU, StringRef FS) {
  // We use the vector ABI whenever the vector facility is avaiable.
  // This is the case by default if CPU is z13 or later, and can be
  // overridden via "[+-]vector" feature string elements.
  bool VectorABI = true;
  bool SoftFloat = false;
  if (CPU.empty() || CPU == "generic" ||
      CPU == "z10" || CPU == "z196" || CPU == "zEC12" ||
      CPU == "arch8" || CPU == "arch9" || CPU == "arch10")
    VectorABI = false;

  SmallVector<StringRef, 3> Features;
  FS.split(Features, ',', -1, false /* KeepEmpty */);
  for (auto &Feature : Features) {
    if (Feature == "vector" || Feature == "+vector")
      VectorABI = true;
    if (Feature == "-vector")
      VectorABI = false;
    if (Feature == "soft-float" || Feature == "+soft-float")
      SoftFloat = true;
    if (Feature == "-soft-float")
      SoftFloat = false;
  }

  return VectorABI && !SoftFloat;
}

static std::string computeDataLayout(const Triple &TT, StringRef CPU,
                                     StringRef FS) {
  bool VectorABI = UsesVectorABI(CPU, FS);
  std::string Ret;

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

  // When using the vector ABI on Linux, 128-bit vectors are also aligned to 64
  // bits. On z/OS, vector types are always aligned to 64 bits.
  if (VectorABI || TT.isOSzOS())
    Ret += "-v128:64";

  // We prefer 16 bits of aligned for all globals; see above.
  Ret += "-a:8:16";

  // Integer registers are 32 or 64 bits.
  Ret += "-n32:64";

  return Ret;
}

static std::unique_ptr<TargetLoweringObjectFile> createTLOF(const Triple &TT) {
  if (TT.isOSzOS())
    return std::make_unique<TargetLoweringObjectFileGOFF>();

  // Note: Some times run with -triple s390x-unknown.
  // In this case, default to ELF unless z/OS specifically provided.
  return std::make_unique<TargetLoweringObjectFileELF>();
}

static Reloc::Model getEffectiveRelocModel(Optional<Reloc::Model> RM) {
  // Static code is suitable for use in a dynamic executable; there is no
  // separate DynamicNoPIC model.
  if (!RM.hasValue() || *RM == Reloc::DynamicNoPIC)
    return Reloc::Static;
  return *RM;
}

// For SystemZ we define the models as follows:
//
// Small:  BRASL can call any function and will use a stub if necessary.
//         Locally-binding symbols will always be in range of LARL.
//
// Medium: BRASL can call any function and will use a stub if necessary.
//         GOT slots and locally-defined text will always be in range
//         of LARL, but other symbols might not be.
//
// Large:  Equivalent to Medium for now.
//
// Kernel: Equivalent to Medium for now.
//
// This means that any PIC module smaller than 4GB meets the
// requirements of Small, so Small seems like the best default there.
//
// All symbols bind locally in a non-PIC module, so the choice is less
// obvious.  There are two cases:
//
// - When creating an executable, PLTs and copy relocations allow
//   us to treat external symbols as part of the executable.
//   Any executable smaller than 4GB meets the requirements of Small,
//   so that seems like the best default.
//
// - When creating JIT code, stubs will be in range of BRASL if the
//   image is less than 4GB in size.  GOT entries will likewise be
//   in range of LARL.  However, the JIT environment has no equivalent
//   of copy relocs, so locally-binding data symbols might not be in
//   the range of LARL.  We need the Medium model in that case.
static CodeModel::Model
getEffectiveSystemZCodeModel(Optional<CodeModel::Model> CM, Reloc::Model RM,
                             bool JIT) {
  if (CM) {
    if (*CM == CodeModel::Tiny)
      report_fatal_error("Target does not support the tiny CodeModel", false);
    if (*CM == CodeModel::Kernel)
      report_fatal_error("Target does not support the kernel CodeModel", false);
    return *CM;
  }
  if (JIT)
    return RM == Reloc::PIC_ ? CodeModel::Small : CodeModel::Medium;
  return CodeModel::Small;
}

SystemZTargetMachine::SystemZTargetMachine(const Target &T, const Triple &TT,
                                           StringRef CPU, StringRef FS,
                                           const TargetOptions &Options,
                                           Optional<Reloc::Model> RM,
                                           Optional<CodeModel::Model> CM,
                                           CodeGenOpt::Level OL, bool JIT)
    : LLVMTargetMachine(
          T, computeDataLayout(TT, CPU, FS), TT, CPU, FS, Options,
          getEffectiveRelocModel(RM),
          getEffectiveSystemZCodeModel(CM, getEffectiveRelocModel(RM), JIT),
          OL),
      TLOF(createTLOF(getTargetTriple())) {
  initAsmInfo();
}

SystemZTargetMachine::~SystemZTargetMachine() = default;

const SystemZSubtarget *
SystemZTargetMachine::getSubtargetImpl(const Function &F) const {
  Attribute CPUAttr = F.getFnAttribute("target-cpu");
  Attribute FSAttr = F.getFnAttribute("target-features");

  std::string CPU =
      CPUAttr.isValid() ? CPUAttr.getValueAsString().str() : TargetCPU;
  std::string FS =
      FSAttr.isValid() ? FSAttr.getValueAsString().str() : TargetFS;

  // FIXME: This is related to the code below to reset the target options,
  // we need to know whether or not the soft float flag is set on the
  // function, so we can enable it as a subtarget feature.
  bool softFloat = F.getFnAttribute("use-soft-float").getValueAsBool();

  if (softFloat)
    FS += FS.empty() ? "+soft-float" : ",+soft-float";

  auto &I = SubtargetMap[CPU + FS];
  if (!I) {
    // This needs to be done before we create a new subtarget since any
    // creation will depend on the TM and the code generation flags on the
    // function that reside in TargetOptions.
    resetTargetOptions(F);
    I = std::make_unique<SystemZSubtarget>(TargetTriple, CPU, FS, *this);
  }

  return I.get();
}

namespace {

/// SystemZ Code Generator Pass Configuration Options.
class SystemZPassConfig : public TargetPassConfig {
public:
  SystemZPassConfig(SystemZTargetMachine &TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {}

  SystemZTargetMachine &getSystemZTargetMachine() const {
    return getTM<SystemZTargetMachine>();
  }

  ScheduleDAGInstrs *
  createPostMachineScheduler(MachineSchedContext *C) const override {
    return new ScheduleDAGMI(C,
                             std::make_unique<SystemZPostRASchedStrategy>(C),
                             /*RemoveKillFlags=*/true);
  }

  void addIRPasses() override;
  bool addInstSelector() override;
  bool addILPOpts() override;
  void addPreRegAlloc() override;
  void addPostRewrite() override;
  void addPostRegAlloc() override;
  void addPreSched2() override;
  void addPreEmitPass() override;
};

} // end anonymous namespace

void SystemZPassConfig::addIRPasses() {
  if (getOptLevel() != CodeGenOpt::None) {
    addPass(createSystemZTDCPass());
    addPass(createLoopDataPrefetchPass());
  }

  TargetPassConfig::addIRPasses();
}

bool SystemZPassConfig::addInstSelector() {
  addPass(createSystemZISelDag(getSystemZTargetMachine(), getOptLevel()));

 if (getOptLevel() != CodeGenOpt::None)
    addPass(createSystemZLDCleanupPass(getSystemZTargetMachine()));

  return false;
}

bool SystemZPassConfig::addILPOpts() {
  addPass(&EarlyIfConverterID);
  return true;
}

void SystemZPassConfig::addPreRegAlloc() {
  addPass(createSystemZCopyPhysRegsPass(getSystemZTargetMachine()));
}

void SystemZPassConfig::addPostRewrite() {
  addPass(createSystemZPostRewritePass(getSystemZTargetMachine()));
}

void SystemZPassConfig::addPostRegAlloc() {
  // PostRewrite needs to be run at -O0 also (in which case addPostRewrite()
  // is not called).
  if (getOptLevel() == CodeGenOpt::None)
    addPass(createSystemZPostRewritePass(getSystemZTargetMachine()));
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
    addPass(createSystemZShortenInstPass(getSystemZTargetMachine()));

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
    addPass(createSystemZElimComparePass(getSystemZTargetMachine()));
  addPass(createSystemZLongBranchPass(getSystemZTargetMachine()));

  // Do final scheduling after all other optimizations, to get an
  // optimal input for the decoder (branch relaxation must happen
  // after block placement).
  if (getOptLevel() != CodeGenOpt::None)
    addPass(&PostMachineSchedulerID);
}

TargetPassConfig *SystemZTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new SystemZPassConfig(*this, PM);
}

TargetTransformInfo
SystemZTargetMachine::getTargetTransformInfo(const Function &F) {
  return TargetTransformInfo(SystemZTTIImpl(this, F));
}
