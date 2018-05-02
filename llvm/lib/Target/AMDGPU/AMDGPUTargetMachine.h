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
/// The AMDGPU TargetMachine interface definition for hw codgen targets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUTARGETMACHINE_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUTARGETMACHINE_H

#include "AMDGPUIntrinsicInfo.h"
#include "AMDGPUSubtarget.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"
#include <memory>

namespace llvm {

//===----------------------------------------------------------------------===//
// AMDGPU Target Machine (R600+)
//===----------------------------------------------------------------------===//

class AMDGPUTargetMachine : public LLVMTargetMachine {
protected:
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  AMDGPUIntrinsicInfo IntrinsicInfo;
  AMDGPUAS AS;

  StringRef getGPUName(const Function &F) const;
  StringRef getFeatureString(const Function &F) const;

public:
  static bool EnableLateStructurizeCFG;

  AMDGPUTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                      StringRef FS, TargetOptions Options,
                      Optional<Reloc::Model> RM, Optional<CodeModel::Model> CM,
                      CodeGenOpt::Level OL);
  ~AMDGPUTargetMachine() override;

  const AMDGPUSubtarget *getSubtargetImpl() const;
  const AMDGPUSubtarget *getSubtargetImpl(const Function &) const override = 0;

  const AMDGPUIntrinsicInfo *getIntrinsicInfo() const override {
    return &IntrinsicInfo;
  }
  TargetTransformInfo getTargetTransformInfo(const Function &F) override;

  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }
  AMDGPUAS getAMDGPUAS() const {
    return AS;
  }

  void adjustPassManager(PassManagerBuilder &) override;
  /// Get the integer value of a null pointer in the given address space.
  uint64_t getNullPointerValue(unsigned AddrSpace) const {
    if (AddrSpace == AS.LOCAL_ADDRESS || AddrSpace == AS.REGION_ADDRESS)
      return -1;
    return 0;
  }
};

//===----------------------------------------------------------------------===//
// R600 Target Machine (R600 -> Cayman)
//===----------------------------------------------------------------------===//

class R600TargetMachine final : public AMDGPUTargetMachine {
private:
  mutable StringMap<std::unique_ptr<R600Subtarget>> SubtargetMap;

public:
  R600TargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                    StringRef FS, TargetOptions Options,
                    Optional<Reloc::Model> RM, Optional<CodeModel::Model> CM,
                    CodeGenOpt::Level OL, bool JIT);

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  const R600Subtarget *getSubtargetImpl(const Function &) const override;

  bool isMachineVerifierClean() const override {
    return false;
  }
};

//===----------------------------------------------------------------------===//
// GCN Target Machine (SI+)
//===----------------------------------------------------------------------===//

class GCNTargetMachine final : public AMDGPUTargetMachine {
private:
  mutable StringMap<std::unique_ptr<SISubtarget>> SubtargetMap;

public:
  GCNTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                   StringRef FS, TargetOptions Options,
                   Optional<Reloc::Model> RM, Optional<CodeModel::Model> CM,
                   CodeGenOpt::Level OL, bool JIT);

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  const SISubtarget *getSubtargetImpl(const Function &) const override;

  bool useIPRA() const override {
    return true;
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUTARGETMACHINE_H
