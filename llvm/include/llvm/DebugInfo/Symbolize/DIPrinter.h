//===- llvm/DebugInfo/Symbolize/DIPrinter.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
public:
  enum class OutputStyle { LLVM, GNU };

private:
  raw_ostream &OS;
  bool PrintFunctionNames;
  bool PrintPretty;
  int PrintSourceContext;
  bool Verbose;
  bool Basenames;
  OutputStyle Style;

  void print(const DILineInfo &Info, bool Inlined);
  void printContext(const std::string &FileName, int64_t Line);

public:
  DIPrinter(raw_ostream &OS, bool PrintFunctionNames = true,
            bool PrintPretty = false, int PrintSourceContext = 0,
            bool Verbose = false, bool Basenames = false,
            OutputStyle Style = OutputStyle::LLVM)
      : OS(OS), PrintFunctionNames(PrintFunctionNames),
        PrintPretty(PrintPretty), PrintSourceContext(PrintSourceContext),
        Verbose(Verbose), Basenames(Basenames), Style(Style) {}

  DIPrinter &operator<<(const DILineInfo &Info);
  DIPrinter &operator<<(const DIInliningInfo &Info);
  DIPrinter &operator<<(const DIGlobal &Global);
};
}
}

#endif

