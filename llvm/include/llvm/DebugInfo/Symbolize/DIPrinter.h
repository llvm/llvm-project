//===- llvm/DebugInfo/Symbolize/DIPrinter.h ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the DIPrinter class, which is responsible for printing
// structures defined in DebugInfo/DIContext.h
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_SYMBOLIZE_DIPRINTER_H
#define LLVM_DEBUGINFO_SYMBOLIZE_DIPRINTER_H

#include "llvm/Support/raw_ostream.h"

namespace llvm {
struct DILineInfo;
class DIInliningInfo;
struct DIGlobal;

namespace symbolize {

class DIPrinter {
  raw_ostream &OS;
  bool PrintFunctionNames;

public:
  DIPrinter(raw_ostream &OS, bool PrintFunctionNames = true)
      : OS(OS), PrintFunctionNames(PrintFunctionNames) {}

  DIPrinter &operator<<(const DILineInfo &Info);
  DIPrinter &operator<<(const DIInliningInfo &Info);
  DIPrinter &operator<<(const DIGlobal &Global);
};
}
}

#endif

