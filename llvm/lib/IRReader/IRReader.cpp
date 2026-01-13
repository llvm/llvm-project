//===---- IRReader.cpp - Reader for LLVM IR files -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IRReader/IRReader.h"
#include "llvm-c/IRReader.h"
#include "llvm/AsmParser/AsmParserContext.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <optional>
#include <system_error>

using namespace llvm;

namespace llvm {
  extern bool TimePassesIsEnabled;
}

const char TimeIRParsingGroupName[] = "irparse";
const char TimeIRParsingGroupDescription[] = "LLVM IR Parsing";
const char TimeIRParsingName[] = "parse";
const char TimeIRParsingDescription[] = "Parse IR";

std::unique_ptr<Module>
llvm::getLazyIRModule(std::unique_ptr<MemoryBuffer> Buffer, SMDiagnostic &Err,
                      LLVMContext &Context, bool ShouldLazyLoadMetadata) {
  if (isBitcode((const unsigned char *)Buffer->getBufferStart(),
                (const unsigned char *)Buffer->getBufferEnd())) {
    Expected<std::unique_ptr<Module>> ModuleOrErr = getOwningLazyBitcodeModule(
        std::move(Buffer), Context, ShouldLazyLoadMetadata);
    if (Error E = ModuleOrErr.takeError()) {
      handleAllErrors(std::move(E), [&](ErrorInfoBase &EIB) {
        Err = SMDiagnostic(Buffer->getBufferIdentifier(), SourceMgr::DK_Error,
                           EIB.message());
      });
      return nullptr;
    }
    return std::move(ModuleOrErr.get());
  }

  return parseAssembly(Buffer->getMemBufferRef(), Err, Context);
}

std::unique_ptr<Module> llvm::getLazyIRFileModule(StringRef Filename,
                                                  SMDiagnostic &Err,
                                                  LLVMContext &Context,
                                                  bool ShouldLazyLoadMetadata) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
      MemoryBuffer::getFileOrSTDIN(Filename);
  if (std::error_code EC = FileOrErr.getError()) {
    Err = SMDiagnostic(Filename, SourceMgr::DK_Error,
                       "Could not open input file: " + EC.message());
    return nullptr;
  }

  return getLazyIRModule(std::move(FileOrErr.get()), Err, Context,
                         ShouldLazyLoadMetadata);
}

std::unique_ptr<Module> llvm::parseIR(MemoryBufferRef Buffer, SMDiagnostic &Err,
                                      LLVMContext &Context,
                                      ParserCallbacks Callbacks,
                                      llvm::AsmParserContext *ParserContext) {
  NamedRegionTimer T(TimeIRParsingName, TimeIRParsingDescription,
                     TimeIRParsingGroupName, TimeIRParsingGroupDescription,
                     TimePassesIsEnabled);
  if (isBitcode((const unsigned char *)Buffer.getBufferStart(),
                (const unsigned char *)Buffer.getBufferEnd())) {
    Expected<std::unique_ptr<Module>> ModuleOrErr =
        parseBitcodeFile(Buffer, Context, Callbacks);
    if (Error E = ModuleOrErr.takeError()) {
      handleAllErrors(std::move(E), [&](ErrorInfoBase &EIB) {
        Err = SMDiagnostic(Buffer.getBufferIdentifier(), SourceMgr::DK_Error,
                           EIB.message());
      });
      return nullptr;
    }
    return std::move(ModuleOrErr.get());
  }

  return parseAssembly(Buffer, Err, Context, nullptr,
                       Callbacks.DataLayout.value_or(
                           [](StringRef, StringRef) { return std::nullopt; }),
                       ParserContext);
}

std::unique_ptr<Module> llvm::parseIRFile(StringRef Filename, SMDiagnostic &Err,
                                          LLVMContext &Context,
                                          ParserCallbacks Callbacks,
                                          AsmParserContext *ParserContext) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
      MemoryBuffer::getFileOrSTDIN(Filename, /*IsText=*/true);
  if (std::error_code EC = FileOrErr.getError()) {
    Err = SMDiagnostic(Filename, SourceMgr::DK_Error,
                       "Could not open input file: " + EC.message());
    return nullptr;
  }

  return parseIR(FileOrErr.get()->getMemBufferRef(), Err, Context, Callbacks,
                 ParserContext);
}

//===----------------------------------------------------------------------===//
// C API.
//===----------------------------------------------------------------------===//

LLVMBool LLVMParseIRInContext(LLVMContextRef ContextRef,
                              LLVMMemoryBufferRef MemBuf, LLVMModuleRef *OutM,
                              char **OutMessage) {
  std::unique_ptr<MemoryBuffer> MB(unwrap(MemBuf));
  return LLVMParseIRInContext2(ContextRef, wrap(MB.get()), OutM, OutMessage);
}

LLVMBool LLVMParseIRInContext2(LLVMContextRef ContextRef,
                               LLVMMemoryBufferRef MemBuf, LLVMModuleRef *OutM,
                               char **OutMessage) {
  SMDiagnostic Diag;

  *OutM = wrap(parseIR(*unwrap(MemBuf), Diag, *unwrap(ContextRef)).release());

  if (*OutM)
    return 0;

  if (OutMessage) {
    std::string Buf;
    raw_string_ostream OS(Buf);
    Diag.print(nullptr, OS, /*ShowColors=*/false);
    *OutMessage = strdup(Buf.c_str());
  }

  return 1;
}
