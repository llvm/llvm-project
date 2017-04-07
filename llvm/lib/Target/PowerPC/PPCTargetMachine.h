//===-- PPCTargetMachine.h - Define TargetMachine for PowerPC ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the PowerPC specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_POWERPC_PPCTARGETMACHINE_H
#define LLVM_LIB_TARGET_POWERPC_PPCTARGETMACHINE_H

#include "PPCInstrInfo.h"
#include "PPCSubtarget.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

/// Common code between 32-bit and 64-bit PowerPC targets.
///
class PPCTargetMachine : public LLVMTargetMachine {
public:
  enum PPCABI { PPC_ABI_UNKNOWN, PPC_ABI_ELFv1, PPC_ABI_ELFv2 };
private:
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  PPCABI TargetABI;

  mutable StringMap<std::unique_ptr<PPCSubtarget>> SubtargetMap;

public:
  PPCTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                   StringRef FS, const TargetOptions &Options,
                   Optional<Reloc::Model> RM, CodeModel::Model CM,
                   CodeGenOpt::Level OL);

  ~PPCTargetMachine() override;

  const PPCSubtarget *getSubtargetImpl(const Function &F) const override;

  // Pass Pipeline Configuration
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  TargetIRAnalysis getTargetIRAnalysis() override;

  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }
  bool isELFv2ABI() const { return TargetABI == PPC_ABI_ELFv2; }
  bool isPPC64() const {
    const Triple &TT = getTargetTriple();
    return (TT.getArch() == Triple::ppc64 || TT.getArch() == Triple::ppc64le);
  };
};

/// PowerPC 32-bit target machine.
///
class PPC32TargetMachine : public PPCTargetMachine {
  virtual void anchor();
public:
  PPC32TargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                     StringRef FS, const TargetOptions &Options,
                     Optional<Reloc::Model> RM, CodeModel::Model CM,
                     CodeGenOpt::Level OL);
};

/// PowerPC 64-bit target machine.
///
class PPC64TargetMachine : public PPCTargetMachine {
  virtual void anchor();
public:
  PPC64TargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                     StringRef FS, const TargetOptions &Options,
                     Optional<Reloc::Model> RM, CodeModel::Model CM,
                     CodeGenOpt::Level OL);
};

} // end namespace llvm

#endif
