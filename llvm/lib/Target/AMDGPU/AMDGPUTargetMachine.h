//===-- AMDGPUTargetMachine.h - AMDGPU TargetMachine Interface --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// The AMDGPU TargetMachine interface definition for hw codgen targets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUTARGETMACHINE_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUTARGETMACHINE_H

#include "AMDGPUSubtarget.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

//===----------------------------------------------------------------------===//
// AMDGPU Target Machine (R600+)
//===----------------------------------------------------------------------===//

class AMDGPUTargetMachine : public LLVMTargetMachine {
protected:
  std::unique_ptr<TargetLoweringObjectFile> TLOF;

  StringRef getGPUName(const Function &F) const;
  StringRef getFeatureString(const Function &F) const;

public:
  static bool EnableLateStructurizeCFG;
  static bool EnableFunctionCalls;
  static bool EnableFixedFunctionABI;

  AMDGPUTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                      StringRef FS, TargetOptions Options,
                      Optional<Reloc::Model> RM, Optional<CodeModel::Model> CM,
                      CodeGenOpt::Level OL);
  ~AMDGPUTargetMachine() override;

  const TargetSubtargetInfo *getSubtargetImpl() const;
  const TargetSubtargetInfo *getSubtargetImpl(const Function &) const override = 0;

  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }

  void adjustPassManager(PassManagerBuilder &) override;

  void registerPassBuilderCallbacks(PassBuilder &PB,
                                    bool DebugPassManager) override;
  void registerDefaultAliasAnalyses(AAManager &) override;

  /// Get the integer value of a null pointer in the given address space.
  static int64_t getNullPointerValue(unsigned AddrSpace);

  bool isNoopAddrSpaceCast(unsigned SrcAS, unsigned DestAS) const override;

  unsigned getAssumedAddrSpace(const Value *V) const override;
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

  TargetTransformInfo getTargetTransformInfo(const Function &F) override;

  bool isMachineVerifierClean() const override {
    return false;
  }
};

//===----------------------------------------------------------------------===//
// GCN Target Machine (SI+)
//===----------------------------------------------------------------------===//

class GCNTargetMachine final : public AMDGPUTargetMachine {
private:
  mutable StringMap<std::unique_ptr<GCNSubtarget>> SubtargetMap;

public:
  GCNTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                   StringRef FS, TargetOptions Options,
                   Optional<Reloc::Model> RM, Optional<CodeModel::Model> CM,
                   CodeGenOpt::Level OL, bool JIT);

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  const GCNSubtarget *getSubtargetImpl(const Function &) const override;

  TargetTransformInfo getTargetTransformInfo(const Function &F) override;

  bool useIPRA() const override {
    return true;
  }

  yaml::MachineFunctionInfo *createDefaultFuncInfoYAML() const override;
  yaml::MachineFunctionInfo *
  convertFuncInfoToYAML(const MachineFunction &MF) const override;
  bool parseMachineFunctionInfo(const yaml::MachineFunctionInfo &,
                                PerFunctionMIParsingState &PFS,
                                SMDiagnostic &Error,
                                SMRange &SourceRange) const override;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUTARGETMACHINE_H
