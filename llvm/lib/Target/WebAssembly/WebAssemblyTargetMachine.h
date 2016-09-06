// WebAssemblyTargetMachine.h - Define TargetMachine for WebAssembly -*- C++ -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file declares the WebAssembly-specific subclass of
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

public:
  WebAssemblyTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                           StringRef FS, const TargetOptions &Options,
                           Optional<Reloc::Model> RM, CodeModel::Model CM,
                           CodeGenOpt::Level OL);

  ~WebAssemblyTargetMachine() override;
  const WebAssemblySubtarget *
  getSubtargetImpl(const Function &F) const override;

  // Pass Pipeline Configuration
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }

  /// \brief Get the TargetIRAnalysis for this target.
  TargetIRAnalysis getTargetIRAnalysis() override;

  bool usesPhysRegsForPEI() const override { return false; }
};

} // end namespace llvm

#endif
