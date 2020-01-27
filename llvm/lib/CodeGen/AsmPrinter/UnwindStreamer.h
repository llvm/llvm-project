//===- UnwindStreamer.h - Unwind Directive Streamer -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing unwind info into assembly files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_UNWINDSTREAMER_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_UNWINDSTREAMER_H

#include "llvm/CodeGen/AsmPrinterHandler.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class AsmPrinter;
class MachineInstr;
class MCSymbol;

/// Emits unwind info directives.
class LLVM_LIBRARY_VISIBILITY UnwindStreamer : public AsmPrinterHandler {
protected:
  /// Target of directive emission.
  AsmPrinter *Asm;

  /// Per-module flag to indicate if .debug_frame has been emitted yet.
  bool HasEmittedDebugFrame = false;

public:
  UnwindStreamer(AsmPrinter *A);
  ~UnwindStreamer() override;

  // Unused.
  void setSymbolSize(const MCSymbol *Sym, uint64_t Size) override {}
  void endModule() override {}
  void beginInstruction(const MachineInstr *MI) override {}
  void endInstruction() override {}

  void beginFunction(const MachineFunction *MF) override;
  void endFunction(const MachineFunction *MF) override;
};

} // end namespace llvm

#endif // LLVM_LIB_CODEGEN_ASMPRINTER_UNWINDSTREAMER_H
