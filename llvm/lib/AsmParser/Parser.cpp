//===- Parser.cpp - Main dispatch module for the Parser library -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This library implements the functionality defined in llvm/Assembly/Parser.h
//
//===----------------------------------------------------------------------===//

#include "llvm/Assembly/Parser.h"
#include "LLParser.h"
#include "llvm/Module.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include <cstring>
using namespace llvm;

Module *llvm::ParseAssembly(MemoryBuffer *F,
                            Module *M,
                            SMDiagnostic &Err,
                            LLVMContext &Context) {
  SourceMgr SM;
  SM.AddNewSourceBuffer(F, SMLoc());

  // If we are parsing into an existing module, do it.
  if (M)
    return LLParser(F, SM, Err, M).Run() ? 0 : M;

  // Otherwise create a new module.
  OwningPtr<Module> M2(new Module(F->getBufferIdentifier(), Context));
  if (LLParser(F, SM, Err, M2.get()).Run())
    return 0;
  return M2.take();
}

Module *llvm::ParseAssemblyFile(const std::string &Filename, SMDiagnostic &Err,
                                LLVMContext &Context) {
  OwningPtr<MemoryBuffer> File;
  if (error_code ec = MemoryBuffer::getFileOrSTDIN(Filename.c_str(), File)) {
    Err = SMDiagnostic(Filename, SourceMgr::DK_Error,
                       "Could not open input file: " + ec.message());
    return 0;
  }

  return ParseAssembly(File.take(), 0, Err, Context);
}

Module *llvm::ParseAssemblyString(const char *AsmString, Module *M,
                                  SMDiagnostic &Err, LLVMContext &Context) {
  MemoryBuffer *F =
    MemoryBuffer::getMemBuffer(StringRef(AsmString, strlen(AsmString)),
                               "<string>");

  return ParseAssembly(F, M, Err, Context);
}
