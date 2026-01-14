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

#include "NVPTX.h"
#include "NVPTXSubtarget.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include <optional>
#include <utility>

namespace llvm {

/// NVPTXTargetMachine
///
class NVPTXTargetMachine : public CodeGenTargetMachineImpl {
  bool is64bit;
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  NVPTX::DrvInterface drvInterface;
  NVPTXSubtarget Subtarget;

  // Hold Strings that can be free'd all together with NVPTXTargetMachine
  BumpPtrAllocator StrAlloc;
  UniqueStringSaver StrPool;

  // Per-function cache policy storage for !mem.cache_hint metadata.
  // Mutable because it's modified during const lowering operations.
  // Data is keyed by Function* and each function is processed sequentially
  // through the pipeline, so no synchronization is needed.
  // IMPORTANT: Data must be cleared after instruction selection completes
  // via clearCachePolicyData() in NVPTXDAGToDAGISel::runOnMachineFunction().
  // The unique_ptr ensures cleanup even if clearCachePolicyData is not called,
  // but explicit clearing prevents unbounded memory growth.
  mutable DenseMap<const Function *,
                   std::unique_ptr<NVPTX::FunctionCachePolicyData>>
      CachePolicyData;

public:
  NVPTXTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                     StringRef FS, const TargetOptions &Options,
                     std::optional<Reloc::Model> RM,
                     std::optional<CodeModel::Model> CM, CodeGenOptLevel OP,
                     bool is64bit);
  ~NVPTXTargetMachine() override;
  const NVPTXSubtarget *getSubtargetImpl(const Function &) const override {
    return &Subtarget;
  }
  const NVPTXSubtarget *getSubtargetImpl() const { return &Subtarget; }
  bool is64Bit() const { return is64bit; }
  NVPTX::DrvInterface getDrvInterface() const { return drvInterface; }
  UniqueStringSaver &getStrPool() const {
    return const_cast<UniqueStringSaver &>(StrPool);
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

  TargetTransformInfo getTargetTransformInfo(const Function &F) const override;

  bool isMachineVerifierClean() const override {
    return false;
  }

  std::pair<const Value *, unsigned>
  getPredicatedAddrSpace(const Value *V) const override;

  // Cache policy data management for !mem.cache_hint metadata.
  // These methods are const but modify mutable state.
  NVPTX::FunctionCachePolicyData &getCachePolicyData(const Function *F) const {
    auto &Data = CachePolicyData[F];
    if (!Data)
      Data = std::make_unique<NVPTX::FunctionCachePolicyData>();
    return *Data;
  }

  void clearCachePolicyData(const Function *F) const {
    CachePolicyData.erase(F);
  }
}; // NVPTXTargetMachine.

class NVPTXTargetMachine32 : public NVPTXTargetMachine {
  virtual void anchor();

public:
  NVPTXTargetMachine32(const Target &T, const Triple &TT, StringRef CPU,
                       StringRef FS, const TargetOptions &Options,
                       std::optional<Reloc::Model> RM,
                       std::optional<CodeModel::Model> CM, CodeGenOptLevel OL,
                       bool JIT);
};

class NVPTXTargetMachine64 : public NVPTXTargetMachine {
  virtual void anchor();

public:
  NVPTXTargetMachine64(const Target &T, const Triple &TT, StringRef CPU,
                       StringRef FS, const TargetOptions &Options,
                       std::optional<Reloc::Model> RM,
                       std::optional<CodeModel::Model> CM, CodeGenOptLevel OL,
                       bool JIT);
};

} // end namespace llvm

#endif
