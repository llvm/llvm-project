//===-- NVPTXTargetMachine.h - Define TargetMachine for NVPTX ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the NVPTX specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_NVPTXTARGETMACHINE_H
#define LLVM_LIB_TARGET_NVPTX_NVPTXTARGETMACHINE_H

#include "NVPTXSubtarget.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include <optional>
#include <utility>

namespace llvm {

/// NVPTXTargetMachine
///
class NVPTXTargetMachine : public CodeGenTargetMachineImpl {
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  NVPTXSubtarget Subtarget;

public:
  NVPTXTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                     StringRef FS, const TargetOptions &Options,
                     std::optional<Reloc::Model> RM,
                     std::optional<CodeModel::Model> CM, CodeGenOptLevel OL,
                     bool JIT);
  ~NVPTXTargetMachine() override;
  const NVPTXSubtarget *getSubtargetImpl(const Function &) const override {
    return &Subtarget;
  }
  const NVPTXSubtarget *getSubtargetImpl() const { return &Subtarget; }
  NVPTX::DrvInterface getDrvInterface() const {
    return getTargetTriple().getOS() == Triple::NVCL ? NVPTX::NVCL
                                                     : NVPTX::CUDA;
  }
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  // Emission of machine code through MCJIT is not supported.
  bool addPassesToEmitMC(PassManagerBase &, MCContext *&, raw_pwrite_stream &,
                         bool = true) override {
    return true;
  }
  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }

  MachineFunctionInfo *
  createMachineFunctionInfo(BumpPtrAllocator &Allocator, const Function &F,
                            const TargetSubtargetInfo *STI) const override;

  void registerEarlyDefaultAliasAnalyses(AAManager &AAM) override;

  void registerPassBuilderCallbacks(PassBuilder &PB) override;

  Error buildCodeGenPipeline(ModulePassManager &MPM, ModuleAnalysisManager &MAM,
                             raw_pwrite_stream &Out, raw_pwrite_stream *DwoOut,
                             CodeGenFileType FileType,
                             const CGPassBuilderOption &Opt, MCContext &Ctx,
                             PassInstrumentationCallbacks *PIC) override;

  TargetTransformInfo getTargetTransformInfo(const Function &F) const override;

  bool isMachineVerifierClean() const override { return false; }

  std::pair<const Value *, unsigned>
  getPredicatedAddrSpace(const Value *V) const override;
}; // NVPTXTargetMachine.

} // end namespace llvm

#endif
