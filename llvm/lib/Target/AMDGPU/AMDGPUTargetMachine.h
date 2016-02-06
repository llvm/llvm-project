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

#ifndef LLVM_LIB_TARGET_R600_AMDGPUTARGETMACHINE_H
#define LLVM_LIB_TARGET_R600_AMDGPUTARGETMACHINE_H

#include "AMDGPUFrameLowering.h"
#include "AMDGPUInstrInfo.h"
#include "AMDGPUIntrinsicInfo.h"
#include "AMDGPUSubtarget.h"
#include "R600ISelLowering.h"
#include "llvm/IR/DataLayout.h"

namespace llvm {

//===----------------------------------------------------------------------===//
// AMDGPU Target Machine (R600+)
//===----------------------------------------------------------------------===//

class AMDGPUTargetMachine : public LLVMTargetMachine {
private:

protected:
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  AMDGPUSubtarget Subtarget;
  AMDGPUIntrinsicInfo IntrinsicInfo;

public:
  AMDGPUTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                      StringRef FS, TargetOptions Options, Reloc::Model RM,
                      CodeModel::Model CM, CodeGenOpt::Level OL);
  ~AMDGPUTargetMachine();

  const AMDGPUSubtarget *getSubtargetImpl() const { return &Subtarget; }
  const AMDGPUSubtarget *getSubtargetImpl(const Function &) const override {
    return &Subtarget;
  }
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

class R600TargetMachine : public AMDGPUTargetMachine {

public:
  R600TargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                    StringRef FS, TargetOptions Options, Reloc::Model RM,
                    CodeModel::Model CM, CodeGenOpt::Level OL);

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;
};

//===----------------------------------------------------------------------===//
// GCN Target Machine (SI+)
//===----------------------------------------------------------------------===//

class GCNTargetMachine : public AMDGPUTargetMachine {

public:
  GCNTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                   StringRef FS, TargetOptions Options, Reloc::Model RM,
                   CodeModel::Model CM, CodeGenOpt::Level OL);

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;
};

} // End namespace llvm

#endif
