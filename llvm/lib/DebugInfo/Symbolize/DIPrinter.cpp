//===- lib/DebugInfo/Symbolize/DIPrinter.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the DIPrinter class, which is responsible for printing
// structures defined in DebugInfo/DIContext.h
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/Symbolize/DIPrinter.h"

#include "llvm/DebugInfo/DIContext.h"

namespace llvm {
namespace symbolize {

// By default, DILineInfo contains "<invalid>" for function/filename it
// cannot fetch. We replace it to "??" to make our output closer to addr2line.
static const char kDILineInfoBadString[] = "<invalid>";
static const char kBadString[] = "??";

void DIPrinter::printName(const DILineInfo &Info, bool Inlined) {
  if (PrintFunctionNames) {
    std::string FunctionName = Info.FunctionName;
    if (FunctionName == kDILineInfoBadString)
      FunctionName = kBadString;

    StringRef Delimiter = (PrintPretty == true) ? " at " : "\n";
    StringRef Prefix = (PrintPretty && Inlined) ? " (inlined by) " : "";
    OS << Prefix << FunctionName << Delimiter;
  }
  std::string Filename = Info.FileName;
  if (Filename == kDILineInfoBadString)
    Filename = kBadString;
  OS << Filename << ":" << Info.Line << ":" << Info.Column << "\n";
}

DIPrinter &DIPrinter::operator<<(const DILineInfo &Info) {
  printName(Info, false);
  return *this;
}

DIPrinter &DIPrinter::operator<<(const DIInliningInfo &Info) {
  uint32_t FramesNum = Info.getNumberOfFrames();
  if (FramesNum == 0) {
    printName(DILineInfo(), false);
    return *this;
  }
  for (uint32_t i = 0; i < FramesNum; i++)
    printName(Info.getFrame(i), i > 0);
  return *this;
}

DIPrinter &DIPrinter::operator<<(const DIGlobal &Global) {
  std::string Name = Global.Name;
  if (Name == kDILineInfoBadString)
    Name = kBadString;
  OS << Name << "\n";
  OS << Global.Start << " " << Global.Size << "\n";
  return *this;
}

}
}
