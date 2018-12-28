//===------ SPIRVWriterPass.h - SPIRV writing pass --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file provides a SPIRV writing pass.
///
//===----------------------------------------------------------------------===//

#ifndef SPIRV_SPIRVWRITERPASS_H
#define SPIRV_SPIRVWRITERPASS_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
class Module;
class ModulePass;
class raw_ostream;
class PreservedAnalyses;

/// \brief Create and return a pass that writes the module to the specified
/// ostream. Note that this pass is designed for use with the legacy pass
/// manager.
ModulePass *createSPIRVWriterPass(raw_ostream &Str);

/// \brief Pass for writing a module of IR out to a SPIRV file.
///
/// Note that this is intended for use with the new pass manager. To construct
/// a pass for the legacy pass manager, use the function above.
class SPIRVWriterPass {
  raw_ostream &OS;

public:
  /// \brief Construct a SPIRV writer pass around a particular output stream.
  explicit SPIRVWriterPass(raw_ostream &OS) : OS(OS) {}

  /// \brief Run the SPIRV writer pass, and output the module to the selected
  /// output stream.
  PreservedAnalyses run(Module &M);

  static StringRef name() { return "SPIRVWriterPass"; }
};

} // namespace llvm

#endif // SPIRV_SPIRVWRITERPASS_H
