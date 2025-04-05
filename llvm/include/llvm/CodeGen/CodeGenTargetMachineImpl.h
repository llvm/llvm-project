//===-- CodeGenTargetMachineImpl.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file describes the CodeGenTargetMachineImpl class, which
/// implements a set of functionality used by \c TargetMachine classes in
/// LLVM that make use of the target-independent code generator.
//===----------------------------------------------------------------------===//
#ifndef LLVM_CODEGEN_CODEGENTARGETMACHINEIMPL_H
#define LLVM_CODEGEN_CODEGENTARGETMACHINEIMPL_H
#include "llvm/Target/TargetMachine.h"

namespace llvm {

/// \brief implements a set of functionality in the \c TargetMachine class
/// for targets that make use of the independent code generator (CodeGen)
/// library. Must not be used directly in code unless to inherit its
/// implementation.
class CodeGenTargetMachineImpl : public TargetMachine {
protected: // Can only create subclasses.
  CodeGenTargetMachineImpl(const Target &T, StringRef DataLayoutString,
                           const Triple &TT, StringRef CPU, StringRef FS,
                           const TargetOptions &Options, Reloc::Model RM,
                           CodeModel::Model CM, CodeGenOptLevel OL);

  void initAsmInfo();

  /// Reset internal state.
  virtual void reset() {};

public:
  /// Get a TargetTransformInfo implementation for the target.
  ///
  /// The TTI returned uses the common code generator to answer queries about
  /// the IR.
  TargetTransformInfo getTargetTransformInfo(const Function &F) const override;

  /// Create a pass configuration object to be used by addPassToEmitX methods
  /// for generating a pipeline of CodeGen passes.
  virtual TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  /// Creates a new \c MachineModuleInfo from this \c TargetMachine
  std::unique_ptr<MachineModuleInfo> createMachineModuleInfo() const override;

  /// Add passes to the specified pass manager to get the specified file
  /// emitted. Typically, this will involve several steps of code generation.
  /// This method should return true if emission of this file type is not
  /// supported, or false on success.
  /// \p MMI a \c MachineModuleInfo that holds the generated machine code
  /// throughout the code generation process
  bool addPassesToEmitFile(PassManagerBase &PM, raw_pwrite_stream &Out,
                           raw_pwrite_stream *DwoOut, CodeGenFileType FileType,
                           MachineModuleInfo *MMI,
                           bool DisableVerify = true) override;

  /// Add passes to the specified pass manager to get machine code emitted with
  /// the MCJIT. This method returns true if machine code is not supported.
  /// \p MMI a \c MachineModuleInfo that holds the generated machine code
  /// throughout the code generation process
  bool addPassesToEmitMC(PassManagerBase &PM,
                         raw_pwrite_stream &Out,
                         MachineModuleInfo *MMI,
                         bool DisableVerify = true) override;

  /// Adds an AsmPrinter pass to the pipeline that prints assembly or
  /// machine code from the MI representation.
  bool addAsmPrinter(PassManagerBase &PM, raw_pwrite_stream &Out,
                     raw_pwrite_stream *DwoOut, CodeGenFileType FileType,
                     MCContext &Context) override;

  Expected<std::unique_ptr<MCStreamer>>
  createMCStreamer(raw_pwrite_stream &Out, raw_pwrite_stream *DwoOut,
                   CodeGenFileType FileType, MCContext &Ctx) override;
};

/// Helper method for getting the code model, returning Default if
/// CM does not have a value. The tiny and kernel models will produce
/// an error, so targets that support them or require more complex codemodel
/// selection logic should implement and call their own getEffectiveCodeModel.
inline CodeModel::Model
getEffectiveCodeModel(std::optional<CodeModel::Model> CM,
                      CodeModel::Model Default) {
  if (CM) {
    // By default, targets do not support the tiny and kernel models.
    if (*CM == CodeModel::Tiny)
      report_fatal_error("Target does not support the tiny CodeModel", false);
    if (*CM == CodeModel::Kernel)
      report_fatal_error("Target does not support the kernel CodeModel", false);
    return *CM;
  }
  return Default;
}

} // namespace llvm

#endif
