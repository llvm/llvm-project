//===-- ARMTargetMachine.h - Define TargetMachine for ARM -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the ARM specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_ARMTARGETMACHINE_H
#define LLVM_LIB_TARGET_ARM_ARMTARGETMACHINE_H

#include "ARMSubtarget.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"
#include <memory>

namespace llvm {

class ARMBaseTargetMachine : public LLVMTargetMachine {
public:
  enum ARMABI {
    ARM_ABI_UNKNOWN,
    ARM_ABI_APCS,
    ARM_ABI_AAPCS, // ARM EABI
    ARM_ABI_AAPCS16
  } TargetABI;

protected:
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  bool isLittle;
  mutable StringMap<std::unique_ptr<ARMSubtarget>> SubtargetMap;

public:
  ARMBaseTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                       StringRef FS, const TargetOptions &Options,
                       Optional<Reloc::Model> RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL, bool isLittle);
  ~ARMBaseTargetMachine() override;

  const ARMSubtarget *getSubtargetImpl(const Function &F) const override;
  // The no argument getSubtargetImpl, while it exists on some targets, is
  // deprecated and should not be used.
  const ARMSubtarget *getSubtargetImpl() const = delete;
  bool isLittleEndian() const { return isLittle; }

  /// \brief Get the TargetIRAnalysis for this target.
  TargetIRAnalysis getTargetIRAnalysis() override;

  // Pass Pipeline Configuration
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }

  bool isMachineVerifierClean() const override {
    return false;
  }
};

/// ARM/Thumb little endian target machine.
///
class ARMLETargetMachine : public ARMBaseTargetMachine {
public:
  ARMLETargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                     StringRef FS, const TargetOptions &Options,
                     Optional<Reloc::Model> RM, CodeModel::Model CM,
                     CodeGenOpt::Level OL);
};

/// ARM/Thumb big endian target machine.
///
class ARMBETargetMachine : public ARMBaseTargetMachine {
public:
  ARMBETargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                     StringRef FS, const TargetOptions &Options,
                     Optional<Reloc::Model> RM, CodeModel::Model CM,
                     CodeGenOpt::Level OL);
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_ARM_ARMTARGETMACHINE_H
