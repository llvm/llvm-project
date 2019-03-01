// WebAssemblyTargetMachine.h - Define TargetMachine for WebAssembly -*- C++ -*-
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the WebAssembly-specific subclass of
/// TargetMachine.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYTARGETMACHINE_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYTARGETMACHINE_H

#include "WebAssemblySubtarget.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class WebAssemblyTargetMachine final : public LLVMTargetMachine {
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  mutable StringMap<std::unique_ptr<WebAssemblySubtarget>> SubtargetMap;
  mutable FeatureBitset UsedFeatures;

public:
  WebAssemblyTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                           StringRef FS, const TargetOptions &Options,
                           Optional<Reloc::Model> RM,
                           Optional<CodeModel::Model> CM, CodeGenOpt::Level OL,
                           bool JIT);

  ~WebAssemblyTargetMachine() override;

  const WebAssemblySubtarget *getSubtargetImpl(std::string CPU,
                                               std::string FS) const;
  const WebAssemblySubtarget *
  getSubtargetImpl(const Function &F) const override;

  // Pass Pipeline Configuration
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }

  FeatureBitset getUsedFeatures() const { return UsedFeatures; }

  TargetTransformInfo getTargetTransformInfo(const Function &F) override;

  bool usesPhysRegsForPEI() const override { return false; }
};

} // end namespace llvm

#endif
