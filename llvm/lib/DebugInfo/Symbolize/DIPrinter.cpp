//===- lib/DebugInfo/Symbolize/DIPrinter.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the DIPrinter class, which is responsible for printing
// structures defined in DebugInfo/DIContext.h
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/Symbolize/DIPrinter.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace llvm {
namespace symbolize {

void PlainPrinterBase::printHeader(uint64_t Address) {
  if (Config.PrintAddress) {
    OS << "0x";
    OS.write_hex(Address);
    StringRef Delimiter = Config.Pretty ? ": " : "\n";
    OS << Delimiter;
  }
}

// Prints source code around in the FileName the Line.
void PlainPrinterBase::printContext(StringRef FileName, int64_t Line) {
  if (Config.SourceContextLines <= 0)
    return;

  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFile(FileName);
  if (!BufOrErr)
    return;

  std::unique_ptr<MemoryBuffer> Buf = std::move(BufOrErr.get());
  int64_t FirstLine =
      std::max(static_cast<int64_t>(1), Line - Config.SourceContextLines / 2);
  int64_t LastLine = FirstLine + Config.SourceContextLines;
  size_t MaxLineNumberWidth = std::ceil(std::log10(LastLine));

  for (line_iterator I = line_iterator(*Buf, false);
       !I.is_at_eof() && I.line_number() <= LastLine; ++I) {
    int64_t L = I.line_number();
    if (L >= FirstLine && L <= LastLine) {
      OS << format_decimal(L, MaxLineNumberWidth);
      if (L == Line)
        OS << " >: ";
      else
        OS << "  : ";
      OS << *I << "\n";
    }
  }
}

void PlainPrinterBase::printFunctionName(StringRef FunctionName, bool Inlined) {
  if (Config.PrintFunctions) {
    if (FunctionName == DILineInfo::BadString)
      FunctionName = DILineInfo::Addr2LineBadString;
    StringRef Delimiter = Config.Pretty ? " at " : "\n";
    StringRef Prefix = (Config.Pretty && Inlined) ? " (inlined by) " : "";
    OS << Prefix << FunctionName << Delimiter;
  }
}

void LLVMPrinter::printSimpleLocation(StringRef Filename,
                                      const DILineInfo &Info) {
  OS << Filename << ':' << Info.Line << ':' << Info.Column << '\n';
  printContext(Filename, Info.Line);
}

void GNUPrinter::printSimpleLocation(StringRef Filename,
                                     const DILineInfo &Info) {
  OS << Filename << ':' << Info.Line;
  if (Info.Discriminator)
    OS << " (discriminator " << Info.Discriminator << ')';
  OS << '\n';
  printContext(Filename, Info.Line);
}

void PlainPrinterBase::printVerbose(StringRef Filename,
                                    const DILineInfo &Info) {
  OS << "  Filename: " << Filename << '\n';
  if (Info.StartLine) {
    OS << "  Function start filename: " << Info.StartFileName << '\n';
    OS << "  Function start line: " << Info.StartLine << '\n';
  }
  OS << "  Line: " << Info.Line << '\n';
  OS << "  Column: " << Info.Column << '\n';
  if (Info.Discriminator)
    OS << "  Discriminator: " << Info.Discriminator << '\n';
}

void LLVMPrinter::printFooter() { OS << '\n'; }

void PlainPrinterBase::print(const DILineInfo &Info, bool Inlined) {
  printFunctionName(Info.FunctionName, Inlined);
  StringRef Filename = Info.FileName;
  if (Filename == DILineInfo::BadString)
    Filename = DILineInfo::Addr2LineBadString;
  if (Config.Verbose)
    printVerbose(Filename, Info);
  else
    printSimpleLocation(Filename, Info);
}

void PlainPrinterBase::print(const Request &Request, const DILineInfo &Info) {
  printHeader(Request.Address);
  print(Info, false);
  printFooter();
}

void PlainPrinterBase::print(const Request &Request,
                             const DIInliningInfo &Info) {
  printHeader(Request.Address);
  uint32_t FramesNum = Info.getNumberOfFrames();
  if (FramesNum == 0)
    print(DILineInfo(), false);
  else
    for (uint32_t I = 0; I < FramesNum; ++I)
      print(Info.getFrame(I), I > 0);
  printFooter();
}

void PlainPrinterBase::print(const Request &Request, const DIGlobal &Global) {
  printHeader(Request.Address);
  StringRef Name = Global.Name;
  if (Name == DILineInfo::BadString)
    Name = DILineInfo::Addr2LineBadString;
  OS << Name << "\n";
  OS << Global.Start << " " << Global.Size << "\n";
  printFooter();
}

void PlainPrinterBase::print(const Request &Request,
                             const std::vector<DILocal> &Locals) {
  printHeader(Request.Address);
  if (Locals.empty())
    OS << DILineInfo::Addr2LineBadString << '\n';
  else
    for (const DILocal &L : Locals) {
      if (L.FunctionName.empty())
        OS << DILineInfo::Addr2LineBadString;
      else
        OS << L.FunctionName;
      OS << '\n';

      if (L.Name.empty())
        OS << DILineInfo::Addr2LineBadString;
      else
        OS << L.Name;
      OS << '\n';

      if (L.DeclFile.empty())
        OS << DILineInfo::Addr2LineBadString;
      else
        OS << L.DeclFile;

      OS << ':' << L.DeclLine << '\n';

      if (L.FrameOffset)
        OS << *L.FrameOffset;
      else
        OS << DILineInfo::Addr2LineBadString;
      OS << ' ';

      if (L.Size)
        OS << *L.Size;
      else
        OS << DILineInfo::Addr2LineBadString;
      OS << ' ';

      if (L.TagOffset)
        OS << *L.TagOffset;
      else
        OS << DILineInfo::Addr2LineBadString;
      OS << '\n';
    }
  printFooter();
}

void PlainPrinterBase::printInvalidCommand(const Request &Request,
                                           const ErrorInfoBase &ErrorInfo) {
  OS << ErrorInfo.message() << '\n';
}

bool PlainPrinterBase::printError(const Request &Request,
                                  const ErrorInfoBase &ErrorInfo,
                                  StringRef ErrorBanner) {
  ES << ErrorBanner;
  ErrorInfo.log(ES);
  ES << '\n';
  // Print an empty struct too.
  return true;
}

} // end namespace symbolize
} // end namespace llvm
