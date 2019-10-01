//===- RemarkParser.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utility methods used by clients that want to use the
// parser for remark diagnostics in LLVM.
//
//===----------------------------------------------------------------------===//

#include "llvm/Remarks/RemarkParser.h"
#include "YAMLRemarkParser.h"
#include "llvm-c/Remarks.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CBindingWrapping.h"

using namespace llvm;
using namespace llvm::remarks;

Parser::Parser(StringRef Buf) : Impl(llvm::make_unique<YAMLParserImpl>(Buf)) {}

Parser::Parser(StringRef Buf, StringRef StrTabBuf)
    : Impl(llvm::make_unique<YAMLParserImpl>(Buf, StrTabBuf)) {}

Parser::~Parser() = default;

static Expected<const Remark *> getNextYAML(YAMLParserImpl &Impl) {
  YAMLRemarkParser &YAMLParser = Impl.YAMLParser;
  // Check for EOF.
  if (Impl.YAMLIt == Impl.YAMLParser.Stream.end())
    return nullptr;

  auto CurrentIt = Impl.YAMLIt;

  // Try to parse an entry.
  if (Error E = YAMLParser.parseYAMLElement(*CurrentIt)) {
    // Set the iterator to the end, in case the user calls getNext again.
    Impl.YAMLIt = Impl.YAMLParser.Stream.end();
    return std::move(E);
  }

  // Move on.
  ++Impl.YAMLIt;

  // Return the just-parsed remark.
  if (const Optional<YAMLRemarkParser::ParseState> &State = YAMLParser.State)
    return &State->TheRemark;
  else
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "unexpected error while parsing.");
}

Expected<const Remark *> Parser::getNext() const {
  if (auto *Impl = dyn_cast<YAMLParserImpl>(this->Impl.get()))
    return getNextYAML(*Impl);
  llvm_unreachable("Get next called with an unknown parsing implementation.");
}

ParsedStringTable::ParsedStringTable(StringRef InBuffer) : Buffer(InBuffer) {
  while (!InBuffer.empty()) {
    // Strings are separated by '\0' bytes.
    std::pair<StringRef, StringRef> Split = InBuffer.split('\0');
    // We only store the offset from the beginning of the buffer.
    Offsets.push_back(Split.first.data() - Buffer.data());
    InBuffer = Split.second;
  }
}

Expected<StringRef> ParsedStringTable::operator[](size_t Index) {
  if (Index >= Offsets.size())
    return createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "String with index %u is out of bounds (size = %u).", Index,
        Offsets.size());

  size_t Offset = Offsets[Index];
  // If it's the last offset, we can't use the next offset to know the size of
  // the string.
  size_t NextOffset =
      (Index == Offsets.size() - 1) ? Buffer.size() : Offsets[Index + 1];
  return StringRef(Buffer.data() + Offset, NextOffset - Offset - 1);
}

// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(remarks::Parser, LLVMRemarkParserRef)

extern "C" LLVMRemarkParserRef LLVMRemarkParserCreateYAML(const void *Buf,
                                                          uint64_t Size) {
  return wrap(
      new remarks::Parser(StringRef(static_cast<const char *>(Buf), Size)));
}

static void handleYAMLError(remarks::YAMLParserImpl &Impl, Error E) {
  handleAllErrors(
      std::move(E),
      [&](const YAMLParseError &PE) {
        Impl.YAMLParser.Stream.printError(&PE.getNode(),
                                          Twine(PE.getMessage()) + Twine('\n'));
      },
      [&](const ErrorInfoBase &EIB) { EIB.log(Impl.YAMLParser.ErrorStream); });
  Impl.HasErrors = true;
}

extern "C" LLVMRemarkEntryRef
LLVMRemarkParserGetNext(LLVMRemarkParserRef Parser) {
  remarks::Parser &TheParser = *unwrap(Parser);

  Expected<const remarks::Remark *> RemarkOrErr = TheParser.getNext();
  if (!RemarkOrErr) {
    // Error during parsing.
    if (auto *Impl = dyn_cast<remarks::YAMLParserImpl>(TheParser.Impl.get()))
      handleYAMLError(*Impl, RemarkOrErr.takeError());
    else
      llvm_unreachable("unkown parser implementation.");
    return nullptr;
  }

  if (*RemarkOrErr == nullptr)
    return nullptr;
  // Valid remark.
  return wrap(*RemarkOrErr);
}

extern "C" LLVMBool LLVMRemarkParserHasError(LLVMRemarkParserRef Parser) {
  if (auto *Impl =
          dyn_cast<remarks::YAMLParserImpl>(unwrap(Parser)->Impl.get()))
    return Impl->HasErrors;
  llvm_unreachable("unkown parser implementation.");
}

extern "C" const char *
LLVMRemarkParserGetErrorMessage(LLVMRemarkParserRef Parser) {
  if (auto *Impl =
          dyn_cast<remarks::YAMLParserImpl>(unwrap(Parser)->Impl.get()))
    return Impl->YAMLParser.ErrorStream.str().c_str();
  llvm_unreachable("unkown parser implementation.");
}

extern "C" void LLVMRemarkParserDispose(LLVMRemarkParserRef Parser) {
  delete unwrap(Parser);
}
