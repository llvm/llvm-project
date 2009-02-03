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
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
using namespace llvm;

Module *llvm::ParseAssemblyFile(const std::string &Filename, ParseError &Err) {
  Err.setFilename(Filename);

  std::string ErrorStr;
  OwningPtr<MemoryBuffer>
    F(MemoryBuffer::getFileOrSTDIN(Filename.c_str(), &ErrorStr));
  if (F == 0) {
    Err.setError("Could not open input file '" + Filename + "'");
    return 0;
  }

  OwningPtr<Module> M(new Module(Filename));
  if (LLParser(F.get(), Err, M.get()).Run())
    return 0;
  return M.take();
}

Module *llvm::ParseAssemblyString(const char *AsmString, Module *M,
                                  ParseError &Err) {
  Err.setFilename("<string>");

  OwningPtr<MemoryBuffer>
    F(MemoryBuffer::getMemBuffer(AsmString, AsmString+strlen(AsmString),
                                 "<string>"));
  
  // If we are parsing into an existing module, do it.
  if (M)
    return LLParser(F.get(), Err, M).Run() ? 0 : M;

  // Otherwise create a new module.
  OwningPtr<Module> M2(new Module("<string>"));
  if (LLParser(F.get(), Err, M2.get()).Run())
    return 0;
  return M2.take();
}


//===------------------------------------------------------------------------===
//                              ParseError Class
//===------------------------------------------------------------------------===

void ParseError::PrintError(const char *ProgName, raw_ostream &S) {
  errs() << ProgName << ": ";
  if (Filename == "-")
    errs() << "<stdin>";
  else
    errs() << Filename;

  if (LineNo != -1) {
    errs() << ':' << LineNo;
    if (ColumnNo != -1)
      errs() << ':' << (ColumnNo+1);
  }

  errs() << ": " << Message << '\n';

  if (LineNo != -1 && ColumnNo != -1) {
    errs() << LineContents << '\n';

    // Print out spaces/tabs before the caret.
    for (unsigned i = 0; i != unsigned(ColumnNo); ++i)
      errs() << (LineContents[i] == '\t' ? '\t' : ' ');
    errs() << "^\n";
  }
}
