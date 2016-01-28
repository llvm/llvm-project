//===-- AMDGPUDiagnosticInfoUnsupported.h - Error reporting -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUDIAGNOSTICINFOUNSUPPORTED_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUDIAGNOSTICINFOUNSUPPORTED_H

#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"

namespace llvm {

/// Diagnostic information for unimplemented or unsupported feature reporting.
class DiagnosticInfoUnsupported : public DiagnosticInfo {
private:
  const Twine &Description;
  const Function &Fn;

  static int KindID;

  static int getKindID() {
    if (KindID == 0)
      KindID = llvm::getNextAvailablePluginDiagnosticKind();
    return KindID;
  }

public:
  DiagnosticInfoUnsupported(const Function &Fn, const Twine &Desc,
                            DiagnosticSeverity Severity = DS_Error);

  const Function &getFunction() const { return Fn; }
  const Twine &getDescription() const { return Description; }

  void print(DiagnosticPrinter &DP) const override;

  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == getKindID();
  }
};

}

#endif
