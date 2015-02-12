//===-- Win64Exception.h - Windows Exception Handling ----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing windows exception info into asm files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_WIN64EXCEPTION_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_WIN64EXCEPTION_H

#include "EHStreamer.h"

namespace llvm {
class MachineFunction;

class Win64Exception : public EHStreamer {
  /// Per-function flag to indicate if personality info should be emitted.
  bool shouldEmitPersonality;

  /// Per-function flag to indicate if the LSDA should be emitted.
  bool shouldEmitLSDA;

  /// Per-function flag to indicate if frame moves info should be emitted.
  bool shouldEmitMoves;

public:
  //===--------------------------------------------------------------------===//
  // Main entry points.
  //
  Win64Exception(AsmPrinter *A);
  virtual ~Win64Exception();

  /// Emit all exception information that should come after the content.
  void endModule() override;

  /// Gather pre-function exception information.  Assumes being emitted
  /// immediately after the function entry point.
  void beginFunction(const MachineFunction *MF) override;

  /// Gather and emit post-function exception information.
  void endFunction(const MachineFunction *) override;
};
}

#endif

