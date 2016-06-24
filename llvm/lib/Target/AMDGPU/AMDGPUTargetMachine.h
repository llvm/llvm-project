//===-- AMDGPUTargetMachine.h - AMDGPU TargetMachine Interface --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief The AMDGPU TargetMachine interface definition for hw codgen targets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUTARGETMACHINE_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUTARGETMACHINE_H

#include "AMDGPUIntrinsicInfo.h"
#include "AMDGPUSubtarget.h"

namespace llvm {

//===----------------------------------------------------------------------===//
// AMDGPU Target Machine (R600+)
//===----------------------------------------------------------------------===//

class AMDGPUTargetMachine : public LLVMTargetMachine {
protected:
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  AMDGPUIntrinsicInfo IntrinsicInfo;

public:
  AMDGPUTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                      StringRef FS, TargetOptions Options,
                      Optional<Reloc::Model> RM, CodeModel::Model CM,
                      CodeGenOpt::Level OL);
  ~AMDGPUTargetMachine();

  const AMDGPUSubtarget *getSubtargetImpl() const;
  const AMDGPUSubtarget *getSubtargetImpl(const Function &) const override;

  const AMDGPUIntrinsicInfo *getIntrinsicInfo() const override {
    return &IntrinsicInfo;
  }
  TargetIRAnalysis getTargetIRAnalysis() override;

  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }
};

//===----------------------------------------------------------------------===//
// R600 Target Machine (R600 -> Cayman)
//===----------------------------------------------------------------------===//

class R600TargetMachine final : public AMDGPUTargetMachine {
private:
  R600Subtarget Subtarget;

public:
  R600TargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                    StringRef FS, TargetOptions Options,
                    Optional<Reloc::Model> RM, CodeModel::Model CM,
                    CodeGenOpt::Level OL);

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  const R600Subtarget *getSubtargetImpl() const {
    return &Subtarget;
  }

  const R600Subtarget *getSubtargetImpl(const Function &) const override {
    return &Subtarget;
  }
};

//===----------------------------------------------------------------------===//
// GCN Target Machine (SI+)
//===----------------------------------------------------------------------===//

class GCNTargetMachine final : public AMDGPUTargetMachine {
private:
    SISubtarget Subtarget;

public:
  GCNTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                   StringRef FS, TargetOptions Options,
                   Optional<Reloc::Model> RM, CodeModel::Model CM,
                   CodeGenOpt::Level OL);

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  const SISubtarget *getSubtargetImpl() const {
    return &Subtarget;
  }

  const SISubtarget *getSubtargetImpl(const Function &) const override {
    return &Subtarget;
  }
};

inline const AMDGPUSubtarget *AMDGPUTargetMachine::getSubtargetImpl() const {
  if (getTargetTriple().getArch() == Triple::amdgcn)
    return static_cast<const GCNTargetMachine *>(this)->getSubtargetImpl();
  return static_cast<const R600TargetMachine *>(this)->getSubtargetImpl();
}

inline const AMDGPUSubtarget *AMDGPUTargetMachine::getSubtargetImpl(
  const Function &F) const {
  if (getTargetTriple().getArch() == Triple::amdgcn)
    return static_cast<const GCNTargetMachine *>(this)->getSubtargetImpl(F);
  return static_cast<const R600TargetMachine *>(this)->getSubtargetImpl(F);
}

} // End namespace llvm

#endif
