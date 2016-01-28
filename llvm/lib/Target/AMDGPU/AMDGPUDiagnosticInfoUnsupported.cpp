//===-- AMDGPUDiagnosticInfoUnsupported.cpp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUDiagnosticInfoUnsupported.h"

using namespace llvm;

DiagnosticInfoUnsupported::DiagnosticInfoUnsupported(
  const Function &Fn,
  const Twine &Desc,
  DiagnosticSeverity Severity)
  : DiagnosticInfo(getKindID(), Severity),
    Description(Desc),
    Fn(Fn) { }

int DiagnosticInfoUnsupported::KindID = 0;

void DiagnosticInfoUnsupported::print(DiagnosticPrinter &DP) const {
  DP << "unsupported " << getDescription() << " in " << Fn.getName();
}
