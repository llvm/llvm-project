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

class SourceCode {
  std::unique_ptr<MemoryBuffer> MemBuf;

  const Optional<StringRef> load(StringRef FileName,
                                 const Optional<StringRef> &EmbeddedSource) {
    if (Lines <= 0)
      return None;

    if (EmbeddedSource)
      return EmbeddedSource;
    else {
      ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
          MemoryBuffer::getFile(FileName);
      if (!BufOrErr)
        return None;
      MemBuf = std::move(*BufOrErr);
      return MemBuf->getBuffer();
    }
  }

  const Optional<StringRef> pruneSource(const Optional<StringRef> &Source) {
    if (!Source)
      return None;
    size_t FirstLinePos = StringRef::npos, Pos = 0;
    for (int64_t L = 1; L <= LastLine; ++L, ++Pos) {
      if (L == FirstLine)
        FirstLinePos = Pos;
      Pos = Source->find('\n', Pos);
      if (Pos == StringRef::npos)
        break;
    }
    if (FirstLinePos == StringRef::npos)
      return None;
    return Source->substr(FirstLinePos, (Pos == StringRef::npos)
                                            ? StringRef::npos
                                            : Pos - FirstLinePos);
  }

public:
  const int64_t Line;
  const int Lines;
  const int64_t FirstLine;
  const int64_t LastLine;
  const Optional<StringRef> PrunedSource;

  SourceCode(
      StringRef FileName, int64_t Line, int Lines,
      const Optional<StringRef> &EmbeddedSource = Optional<StringRef>(None))
      : Line(Line), Lines(Lines),
        FirstLine(std::max(static_cast<int64_t>(1), Line - Lines / 2)),
        LastLine(FirstLine + Lines - 1),
        PrunedSource(pruneSource(load(FileName, EmbeddedSource))) {}
};

void PlainPrinterBase::printHeader(uint64_t Address) {
  if (Config.PrintAddress) {
    OS << "0x";
    OS.write_hex(Address);
    StringRef Delimiter = Config.Pretty ? ": " : "\n";
    OS << Delimiter;
  }
}

// Prints source code around in the FileName the Line.
void PlainPrinterBase::printContext(SourceCode SourceCode) {
  if (!SourceCode.PrunedSource)
    return;

  StringRef Source = *SourceCode.PrunedSource;
  std::string SourceCopy;
  if (*Source.end() != '\0') {
    SourceCopy = Source.str();
    Source = SourceCopy;
  }

  size_t MaxLineNumberWidth = std::ceil(std::log10(SourceCode.LastLine));
  for (line_iterator I = line_iterator(MemoryBufferRef(Source, ""), false);
       !I.is_at_eof(); ++I) {
    int64_t L = SourceCode.FirstLine + I.line_number() - 1;
    OS << format_decimal(L, MaxLineNumberWidth);
    if (L == SourceCode.Line)
      OS << " >: ";
    else
      OS << "  : ";
    OS << *I << '\n';
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
  printContext(SourceCode(Filename, Info.Line, Config.SourceContextLines));
}

void GNUPrinter::printSimpleLocation(StringRef Filename,
                                     const DILineInfo &Info) {
  OS << Filename << ':' << Info.Line;
  if (Info.Discriminator)
    OS << " (discriminator " << Info.Discriminator << ')';
  OS << '\n';
  printContext(SourceCode(Filename, Info.Line, Config.SourceContextLines));
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
  printHeader(*Request.Address);
  print(Info, false);
  printFooter();
}

void PlainPrinterBase::print(const Request &Request,
                             const DIInliningInfo &Info) {
  printHeader(*Request.Address);
  uint32_t FramesNum = Info.getNumberOfFrames();
  if (FramesNum == 0)
    print(DILineInfo(), false);
  else
    for (uint32_t I = 0; I < FramesNum; ++I)
      print(Info.getFrame(I), I > 0);
  printFooter();
}

void PlainPrinterBase::print(const Request &Request, const DIGlobal &Global) {
  printHeader(*Request.Address);
  StringRef Name = Global.Name;
  if (Name == DILineInfo::BadString)
    Name = DILineInfo::Addr2LineBadString;
  OS << Name << "\n";
  OS << Global.Start << " " << Global.Size << "\n";
  printFooter();
}

void PlainPrinterBase::print(const Request &Request,
                             const std::vector<DILocal> &Locals) {
  printHeader(*Request.Address);
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
                                           StringRef Command) {
  OS << Command << '\n';
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

static std::string toHex(uint64_t V) {
  return ("0x" + Twine::utohexstr(V)).str();
}

static json::Object toJSON(const Request &Request, StringRef ErrorMsg = "") {
  json::Object Json({{"ModuleName", Request.ModuleName.str()}});
  if (Request.Address)
    Json["Address"] = toHex(*Request.Address);
  if (!ErrorMsg.empty())
    Json["Error"] = json::Object({{"Message", ErrorMsg.str()}});
  return Json;
}

void JSONPrinter::print(const Request &Request, const DILineInfo &Info) {
  DIInliningInfo InliningInfo;
  InliningInfo.addFrame(Info);
  print(Request, InliningInfo);
}

void JSONPrinter::print(const Request &Request, const DIInliningInfo &Info) {
  json::Array Array;
  for (uint32_t I = 0, N = Info.getNumberOfFrames(); I < N; ++I) {
    const DILineInfo &LineInfo = Info.getFrame(I);
    Array.push_back(json::Object(
        {{"FunctionName", LineInfo.FunctionName != DILineInfo::BadString
                              ? LineInfo.FunctionName
                              : ""},
         {"StartFileName", LineInfo.StartFileName != DILineInfo::BadString
                               ? LineInfo.StartFileName
                               : ""},
         {"StartLine", LineInfo.StartLine},
         {"FileName",
          LineInfo.FileName != DILineInfo::BadString ? LineInfo.FileName : ""},
         {"Line", LineInfo.Line},
         {"Column", LineInfo.Column},
         {"Discriminator", LineInfo.Discriminator}}));
  }
  json::Object Json = toJSON(Request);
  Json["Symbol"] = std::move(Array);
  if (ObjectList)
    ObjectList->push_back(std::move(Json));
  else
    printJSON(std::move(Json));
}

void JSONPrinter::print(const Request &Request, const DIGlobal &Global) {
  json::Object Data(
      {{"Name", Global.Name != DILineInfo::BadString ? Global.Name : ""},
       {"Start", toHex(Global.Start)},
       {"Size", toHex(Global.Size)}});
  json::Object Json = toJSON(Request);
  Json["Data"] = std::move(Data);
  if (ObjectList)
    ObjectList->push_back(std::move(Json));
  else
    printJSON(std::move(Json));
}

void JSONPrinter::print(const Request &Request,
                        const std::vector<DILocal> &Locals) {
  json::Array Frame;
  for (const DILocal &Local : Locals) {
    json::Object FrameObject(
        {{"FunctionName", Local.FunctionName},
         {"Name", Local.Name},
         {"DeclFile", Local.DeclFile},
         {"DeclLine", int64_t(Local.DeclLine)},
         {"Size", Local.Size ? toHex(*Local.Size) : ""},
         {"TagOffset", Local.TagOffset ? toHex(*Local.TagOffset) : ""}});
    if (Local.FrameOffset)
      FrameObject["FrameOffset"] = *Local.FrameOffset;
    Frame.push_back(std::move(FrameObject));
  }
  json::Object Json = toJSON(Request);
  Json["Frame"] = std::move(Frame);
  if (ObjectList)
    ObjectList->push_back(std::move(Json));
  else
    printJSON(std::move(Json));
}

void JSONPrinter::printInvalidCommand(const Request &Request,
                                      StringRef Command) {
  printError(Request,
             StringError("unable to parse arguments: " + Command,
                         std::make_error_code(std::errc::invalid_argument)),
             "");
}

bool JSONPrinter::printError(const Request &Request,
                             const ErrorInfoBase &ErrorInfo,
                             StringRef ErrorBanner) {
  json::Object Json = toJSON(Request, ErrorInfo.message());
  if (ObjectList)
    ObjectList->push_back(std::move(Json));
  else
    printJSON(std::move(Json));
  return false;
}

void JSONPrinter::listBegin() {
  assert(!ObjectList);
  ObjectList = std::make_unique<json::Array>();
}

void JSONPrinter::listEnd() {
  assert(ObjectList);
  printJSON(std::move(*ObjectList));
  ObjectList.reset();
}

} // end namespace symbolize
} // end namespace llvm
