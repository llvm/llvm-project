//===-- llvm-nm.cpp - Symbol table dumping utility for llvm ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program is a utility that works like traditional Unix "nm",
// that is, it prints out the names of symbols in a bitcode file,
// along with some information about each symbol.
//
// This "nm" does not print symbols' addresses. It supports many of
// the features of GNU "nm", including its different output formats.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Bitcode/Archive.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/System/Signals.h"
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <iostream>
using namespace llvm;

namespace {
  enum OutputFormatTy { bsd, sysv, posix };
  cl::opt<OutputFormatTy>
  OutputFormat("format",
       cl::desc("Specify output format"),
         cl::values(clEnumVal(bsd,   "BSD format"),
                    clEnumVal(sysv,  "System V format"),
                    clEnumVal(posix, "POSIX.2 format"),
                    clEnumValEnd), cl::init(bsd));
  cl::alias OutputFormat2("f", cl::desc("Alias for --format"),
                          cl::aliasopt(OutputFormat));

  cl::list<std::string>
  InputFilenames(cl::Positional, cl::desc("<input bitcode files>"),
                 cl::ZeroOrMore);

  cl::opt<bool> UndefinedOnly("undefined-only",
                              cl::desc("Show only undefined symbols"));
  cl::alias UndefinedOnly2("u", cl::desc("Alias for --undefined-only"),
                           cl::aliasopt(UndefinedOnly));

  cl::opt<bool> DefinedOnly("defined-only",
                            cl::desc("Show only defined symbols"));

  cl::opt<bool> ExternalOnly("extern-only",
                             cl::desc("Show only external symbols"));
  cl::alias ExternalOnly2("g", cl::desc("Alias for --extern-only"),
                          cl::aliasopt(ExternalOnly));

  cl::opt<bool> BSDFormat("B", cl::desc("Alias for --format=bsd"));
  cl::opt<bool> POSIXFormat("P", cl::desc("Alias for --format=posix"));

  bool MultipleFiles = false;

  std::string ToolName;
}

static char TypeCharForSymbol(GlobalValue &GV) {
  /* FIXME: what to do with private linkage? */
  if (GV.isDeclaration())                                  return 'U';
  if (GV.hasLinkOnceLinkage())                             return 'C';
  if (GV.hasCommonLinkage())                               return 'C';
  if (GV.hasWeakLinkage())                                 return 'W';
  if (isa<Function>(GV) && GV.hasInternalLinkage())        return 't';
  if (isa<Function>(GV))                                   return 'T';
  if (isa<GlobalVariable>(GV) && GV.hasInternalLinkage())  return 'd';
  if (isa<GlobalVariable>(GV))                             return 'D';
  if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(&GV)) {
    const GlobalValue *AliasedGV = GA->getAliasedGlobal();
    if (isa<Function>(AliasedGV))                          return 'T';
    if (isa<GlobalVariable>(AliasedGV))                    return 'D';
  }
                                                           return '?';
}

static void DumpSymbolNameForGlobalValue(GlobalValue &GV) {
  const std::string SymbolAddrStr = "        "; // Not used yet...
  char TypeChar = TypeCharForSymbol (GV);
  if ((TypeChar != 'U') && UndefinedOnly)
    return;
  if ((TypeChar == 'U') && DefinedOnly)
    return;
  if (GV.hasLocalLinkage () && ExternalOnly)
    return;
  if (OutputFormat == posix) {
    std::cout << GV.getName () << " " << TypeCharForSymbol (GV) << " "
              << SymbolAddrStr << "\n";
  } else if (OutputFormat == bsd) {
    std::cout << SymbolAddrStr << " " << TypeCharForSymbol (GV) << " "
              << GV.getName () << "\n";
  } else if (OutputFormat == sysv) {
    std::string PaddedName (GV.getName ());
    while (PaddedName.length () < 20)
      PaddedName += " ";
    std::cout << PaddedName << "|" << SymbolAddrStr << "|   "
              << TypeCharForSymbol (GV)
              << "  |                  |      |     |\n";
  }
}

static void DumpSymbolNamesFromModule(Module *M) {
  const std::string &Filename = M->getModuleIdentifier ();
  if (OutputFormat == posix && MultipleFiles) {
    std::cout << Filename << ":\n";
  } else if (OutputFormat == bsd && MultipleFiles) {
    std::cout << "\n" << Filename << ":\n";
  } else if (OutputFormat == sysv) {
    std::cout << "\n\nSymbols from " << Filename << ":\n\n"
              << "Name                  Value   Class        Type"
              << "         Size   Line  Section\n";
  }
  std::for_each (M->begin (), M->end (), DumpSymbolNameForGlobalValue);
  std::for_each (M->global_begin (), M->global_end (),
                 DumpSymbolNameForGlobalValue);
  std::for_each (M->alias_begin (), M->alias_end (),
                 DumpSymbolNameForGlobalValue);
}

static void DumpSymbolNamesFromFile(std::string &Filename) {
  std::string ErrorMessage;
  sys::Path aPath(Filename);
  // Note: Currently we do not support reading an archive from stdin.
  if (Filename == "-" || aPath.isBitcodeFile()) {
    std::auto_ptr<MemoryBuffer> Buffer(
                   MemoryBuffer::getFileOrSTDIN(Filename, &ErrorMessage));
    Module *Result = 0;
    if (Buffer.get())
      Result = ParseBitcodeFile(Buffer.get(), &ErrorMessage);
    
    if (Result)
      DumpSymbolNamesFromModule(Result);
    else {
      std::cerr << ToolName << ": " << Filename << ": " << ErrorMessage << "\n";
      return;
    }
    
  } else if (aPath.isArchive()) {
    std::string ErrMsg;
    Archive* archive = Archive::OpenAndLoad(sys::Path(Filename), &ErrorMessage);
    if (!archive)
      std::cerr << ToolName << ": " << Filename << ": " << ErrorMessage << "\n";
    std::vector<Module *> Modules;
    if (archive->getAllModules(Modules, &ErrorMessage)) {
      std::cerr << ToolName << ": " << Filename << ": " << ErrorMessage << "\n";
      return;
    }
    MultipleFiles = true;
    std::for_each (Modules.begin(), Modules.end(), DumpSymbolNamesFromModule);
  } else {
    std::cerr << ToolName << ": " << Filename << ": "
              << "unrecognizable file type\n";
    return;
  }
}

int main(int argc, char **argv) {
  llvm_shutdown_obj X;  // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv, "llvm symbol table dumper\n");
  sys::PrintStackTraceOnErrorSignal();

  ToolName = argv[0];
  if (BSDFormat) OutputFormat = bsd;
  if (POSIXFormat) OutputFormat = posix;

  switch (InputFilenames.size()) {
  case 0: InputFilenames.push_back("-");
  case 1: break;
  default: MultipleFiles = true;
  }

  std::for_each(InputFilenames.begin(), InputFilenames.end(),
                DumpSymbolNamesFromFile);
  return 0;
}
