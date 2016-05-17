//===-- llvm/MC/MCCodeGenInfo.h - Target CodeGen Info -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tracks information about the target which can affect codegen,
// asm parsing, and asm printing. For example, relocation model.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCCODEGENINFO_H
#define LLVM_MC_MCCODEGENINFO_H

#include "llvm/Support/CodeGen.h"

namespace llvm {

class MCCodeGenInfo {
  /// Relocation model: static, pic, etc.
  ///
  Reloc::Model RelocationModel;

  /// Code model.
  ///
  CodeModel::Model CMModel;

  /// Optimization level.
  ///
  CodeGenOpt::Level OptLevel;

public:
  void initMCCodeGenInfo(Reloc::Model RM, CodeModel::Model CM,
                         CodeGenOpt::Level OL);

  Reloc::Model getRelocationModel() const { return RelocationModel; }

  CodeModel::Model getCodeModel() const { return CMModel; }

  CodeGenOpt::Level getOptLevel() const { return OptLevel; }

  // Allow overriding OptLevel on a per-function basis.
  void setOptLevel(CodeGenOpt::Level Level) { OptLevel = Level; }
};
} // namespace llvm

#endif
