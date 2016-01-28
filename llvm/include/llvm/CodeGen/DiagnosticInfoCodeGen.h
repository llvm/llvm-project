//===- llvm/Support/DiagnosticInfoCodeGen.h - Diagnostic Declaration ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the different classes involved in codegen diagnostics.
//
// Diagnostics reporting is still done as part of the LLVMContext.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_DIAGNOSTICINFOCODEGEN_H
#define LLVM_CODEGEN_DIAGNOSTICINFOCODEGEN_H

#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"

namespace llvm {

/// Diagnostic information for unsupported feature in backend.
class DiagnosticInfoUnsupported
    : public DiagnosticInfoWithDebugLocBase {
private:
  const Twine &Msg;
  const SDValue Value;

public:
  /// \p Fn is the function where the diagnostic is being emitted. \p DLoc is
  /// the location information to use in the diagnostic. If line table
  /// information is available, the diagnostic will include the source code
  /// location. \p Msg is the message to show. Note that this class does not
  /// copy this message, so this reference must be valid for the whole life time
  /// of the diagnostic.
  DiagnosticInfoUnsupported(const Function &Fn, const Twine &Msg,
                            SDLoc DLoc = SDLoc(), SDValue Value = SDValue())
      : DiagnosticInfoWithDebugLocBase(DK_Unsupported, DS_Error, Fn,
                                       DLoc.getDebugLoc()),
        Msg(Msg), Value(Value) {}

  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == DK_Unsupported;
  }

  const Twine &getMessage() const { return Msg; }

  void print(DiagnosticPrinter &DP) const;
};

}

#endif
