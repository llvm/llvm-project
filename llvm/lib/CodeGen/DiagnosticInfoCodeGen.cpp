//===- llvm/Support/DiagnosticInfoCodeGen.cpp - Diagnostic Definitions -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the different classes involved in codegen diagnostics.
//
// Diagnostics reporting is still done as part of the LLVMContext.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/DiagnosticInfoCodeGen.h"
#include "llvm/IR/DiagnosticPrinter.h"

namespace llvm {

void DiagnosticInfoUnsupported::print(DiagnosticPrinter &DP) const {
  std::string Str;
  raw_string_ostream OS(Str);

  OS << getLocationStr() << ": in function " << getFunction().getName() << ' '
     << *getFunction().getFunctionType() << ": " << Msg;
  if (Value)
    Value->print(OS);
  OS << '\n';
  OS.flush();
  DP << Str;
}

}
