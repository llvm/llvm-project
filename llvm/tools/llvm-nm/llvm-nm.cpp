//===-- llvm-nm.cpp - Symbol table dumping utility for llvm ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program is a utility that works like traditional Unix "nm", that is, it
// prints out the names of symbols in a bitcode or object file, along with some
// information about each symbol.
//
// This "nm" supports many of the features of GNU "nm", including its different
// output formats.
//
//===----------------------------------------------------------------------===//

#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Bitcode/Archive.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/system_error.h"
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <vector>
using namespace llvm;
using namespace object;

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

  cl::opt<bool> DynamicSyms("dynamic",
                             cl::desc("Display the dynamic symbols instead "
                                      "of normal symbols."));
  cl::alias DynamicSyms2("D", cl::desc("Alias for --dynamic"),
                         cl::aliasopt(DynamicSyms));

  cl::opt<bool> DefinedOnly("defined-only",
                            cl::desc("Show only defined symbols"));

  cl::opt<bool> ExternalOnly("extern-only",
                             cl::desc("Show only external symbols"));
  cl::alias ExternalOnly2("g", cl::desc("Alias for --extern-only"),
                          cl::aliasopt(ExternalOnly));

  cl::opt<bool> BSDFormat("B", cl::desc("Alias for --format=bsd"));
  cl::opt<bool> POSIXFormat("P", cl::desc("Alias for --format=posix"));

  cl::opt<bool> PrintFileName("print-file-name",
    cl::desc("Precede each symbol with the object file it came from"));

  cl::alias PrintFileNameA("A", cl::desc("Alias for --print-file-name"),
                                cl::aliasopt(PrintFileName));
  cl::alias PrintFileNameo("o", cl::desc("Alias for --print-file-name"),
                                cl::aliasopt(PrintFileName));

  cl::opt<bool> DebugSyms("debug-syms",
    cl::desc("Show all symbols, even debugger only"));
  cl::alias DebugSymsa("a", cl::desc("Alias for --debug-syms"),
                            cl::aliasopt(DebugSyms));

  cl::opt<bool> NumericSort("numeric-sort",
    cl::desc("Sort symbols by address"));
  cl::alias NumericSortn("n", cl::desc("Alias for --numeric-sort"),
                              cl::aliasopt(NumericSort));
  cl::alias NumericSortv("v", cl::desc("Alias for --numeric-sort"),
                              cl::aliasopt(NumericSort));

  cl::opt<bool> NoSort("no-sort",
    cl::desc("Show symbols in order encountered"));
  cl::alias NoSortp("p", cl::desc("Alias for --no-sort"),
                         cl::aliasopt(NoSort));

  cl::opt<bool> PrintSize("print-size",
    cl::desc("Show symbol size instead of address"));
  cl::alias PrintSizeS("S", cl::desc("Alias for --print-size"),
                            cl::aliasopt(PrintSize));

  cl::opt<bool> SizeSort("size-sort", cl::desc("Sort symbols by size"));

  cl::opt<bool> WithoutAliases("without-aliases", cl::Hidden,
                               cl::desc("Exclude aliases from output"));

  bool PrintAddress = true;

  bool MultipleFiles = false;

  std::string ToolName;
}


static void error(Twine message, Twine path = Twine()) {
  errs() << ToolName << ": " << path << ": " << message << ".\n";
}

static bool error(error_code ec, Twine path = Twine()) {
  if (ec) {
    error(ec.message(), path);
    return true;
  }
  return false;
}

namespace {
  struct NMSymbol {
    uint64_t  Address;
    uint64_t  Size;
    char      TypeChar;
    StringRef Name;
  };

  static bool CompareSymbolAddress(const NMSymbol &a, const NMSymbol &b) {
    if (a.Address < b.Address)
      return true;
    else if (a.Address == b.Address && a.Name < b.Name)
      return true;
    else
      return false;

  }

  static bool CompareSymbolSize(const NMSymbol &a, const NMSymbol &b) {
    if (a.Size < b.Size)
      return true;
    else if (a.Size == b.Size && a.Name < b.Name)
      return true;
    else
      return false;
  }

  static bool CompareSymbolName(const NMSymbol &a, const NMSymbol &b) {
    return a.Name < b.Name;
  }

  StringRef CurrentFilename;
  typedef std::vector<NMSymbol> SymbolListT;
  SymbolListT SymbolList;
}

static void SortAndPrintSymbolList() {
  if (!NoSort) {
    if (NumericSort)
      std::sort(SymbolList.begin(), SymbolList.end(), CompareSymbolAddress);
    else if (SizeSort)
      std::sort(SymbolList.begin(), SymbolList.end(), CompareSymbolSize);
    else
      std::sort(SymbolList.begin(), SymbolList.end(), CompareSymbolName);
  }

  if (OutputFormat == posix && MultipleFiles) {
    outs() << '\n' << CurrentFilename << ":\n";
  } else if (OutputFormat == bsd && MultipleFiles) {
    outs() << "\n" << CurrentFilename << ":\n";
  } else if (OutputFormat == sysv) {
    outs() << "\n\nSymbols from " << CurrentFilename << ":\n\n"
           << "Name                  Value   Class        Type"
           << "         Size   Line  Section\n";
  }

  for (SymbolListT::iterator i = SymbolList.begin(),
                             e = SymbolList.end(); i != e; ++i) {
    if ((i->TypeChar != 'U') && UndefinedOnly)
      continue;
    if ((i->TypeChar == 'U') && DefinedOnly)
      continue;
    if (SizeSort && !PrintAddress && i->Size == UnknownAddressOrSize)
      continue;

    char SymbolAddrStr[10] = "";
    char SymbolSizeStr[10] = "";

    if (OutputFormat == sysv || i->Address == object::UnknownAddressOrSize)
      strcpy(SymbolAddrStr, "        ");
    if (OutputFormat == sysv)
      strcpy(SymbolSizeStr, "        ");

    if (i->Address != object::UnknownAddressOrSize)
      format("%08" PRIx64, i->Address).print(SymbolAddrStr,
                                             sizeof(SymbolAddrStr));
    if (i->Size != object::UnknownAddressOrSize)
      format("%08" PRIx64, i->Size).print(SymbolSizeStr, sizeof(SymbolSizeStr));

    if (OutputFormat == posix) {
      outs() << i->Name << " " << i->TypeChar << " "
             << SymbolAddrStr << SymbolSizeStr << "\n";
    } else if (OutputFormat == bsd) {
      if (PrintAddress)
        outs() << SymbolAddrStr << ' ';
      if (PrintSize) {
        outs() << SymbolSizeStr;
        if (i->Size != object::UnknownAddressOrSize)
          outs() << ' ';
      }
      outs() << i->TypeChar << " " << i->Name  << "\n";
    } else if (OutputFormat == sysv) {
      std::string PaddedName (i->Name);
      while (PaddedName.length () < 20)
        PaddedName += " ";
      outs() << PaddedName << "|" << SymbolAddrStr << "|   "
             << i->TypeChar
             << "  |                  |" << SymbolSizeStr << "|     |\n";
    }
  }

  SymbolList.clear();
}

static char TypeCharForSymbol(GlobalValue &GV) {
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
  // Private linkage and available_externally linkage don't exist in symtab.
  if (GV.hasPrivateLinkage() ||
      GV.hasLinkerPrivateLinkage() ||
      GV.hasLinkerPrivateWeakLinkage() ||
      GV.hasAvailableExternallyLinkage())
    return;
  char TypeChar = TypeCharForSymbol(GV);
  if (GV.hasLocalLinkage () && ExternalOnly)
    return;

  NMSymbol s;
  s.Address = object::UnknownAddressOrSize;
  s.Size = object::UnknownAddressOrSize;
  s.TypeChar = TypeChar;
  s.Name     = GV.getName();
  SymbolList.push_back(s);
}

static void DumpSymbolNamesFromModule(Module *M) {
  CurrentFilename = M->getModuleIdentifier();
  std::for_each (M->begin(), M->end(), DumpSymbolNameForGlobalValue);
  std::for_each (M->global_begin(), M->global_end(),
                 DumpSymbolNameForGlobalValue);
  if (!WithoutAliases)
    std::for_each (M->alias_begin(), M->alias_end(),
		   DumpSymbolNameForGlobalValue);

  SortAndPrintSymbolList();
}

static void DumpSymbolNamesFromObject(ObjectFile *obj) {
  error_code ec;
  symbol_iterator ibegin = obj->begin_symbols();
  symbol_iterator iend = obj->end_symbols();
  if (DynamicSyms) {
    ibegin = obj->begin_dynamic_symbols();
    iend = obj->end_dynamic_symbols();
  }
  for (symbol_iterator i = ibegin; i != iend; i.increment(ec)) {
    if (error(ec)) break;
    uint32_t symflags;
    if (error(i->getFlags(symflags))) break;
    if (!DebugSyms && (symflags & SymbolRef::SF_FormatSpecific))
      continue;
    NMSymbol s;
    s.Size = object::UnknownAddressOrSize;
    s.Address = object::UnknownAddressOrSize;
    if (PrintSize || SizeSort) {
      if (error(i->getSize(s.Size))) break;
    }
    if (PrintAddress)
      if (error(i->getAddress(s.Address))) break;
    if (error(i->getNMTypeChar(s.TypeChar))) break;
    if (error(i->getName(s.Name))) break;
    SymbolList.push_back(s);
  }

  CurrentFilename = obj->getFileName();
  SortAndPrintSymbolList();
}

static void DumpSymbolNamesFromFile(std::string &Filename) {
  if (Filename != "-" && !sys::fs::exists(Filename)) {
    errs() << ToolName << ": '" << Filename << "': " << "No such file\n";
    return;
  }

  OwningPtr<MemoryBuffer> Buffer;
  if (error(MemoryBuffer::getFileOrSTDIN(Filename, Buffer), Filename))
    return;

  sys::fs::file_magic magic = sys::fs::identify_magic(Buffer->getBuffer());

  LLVMContext &Context = getGlobalContext();
  std::string ErrorMessage;
  if (magic == sys::fs::file_magic::bitcode) {
    Module *Result = 0;
    Result = ParseBitcodeFile(Buffer.get(), Context, &ErrorMessage);
    if (Result) {
      DumpSymbolNamesFromModule(Result);
      delete Result;
    } else {
      error(ErrorMessage, Filename);
      return;
    }
  } else if (magic == sys::fs::file_magic::archive) {
    OwningPtr<Binary> arch;
    if (error(object::createBinary(Buffer.take(), arch), Filename))
      return;

    if (object::Archive *a = dyn_cast<object::Archive>(arch.get())) {
      for (object::Archive::child_iterator i = a->begin_children(),
                                           e = a->end_children(); i != e; ++i) {
        OwningPtr<Binary> child;
        if (i->getAsBinary(child)) {
          // Try opening it as a bitcode file.
          OwningPtr<MemoryBuffer> buff(i->getBuffer());
          Module *Result = 0;
          if (buff)
            Result = ParseBitcodeFile(buff.get(), Context, &ErrorMessage);

          if (Result) {
            DumpSymbolNamesFromModule(Result);
            delete Result;
          }
          continue;
        }
        if (object::ObjectFile *o = dyn_cast<ObjectFile>(child.get())) {
          outs() << o->getFileName() << ":\n";
          DumpSymbolNamesFromObject(o);
        }
      }
    }
  } else if (magic.is_object()) {
    OwningPtr<Binary> obj;
    if (error(object::createBinary(Buffer.take(), obj), Filename))
      return;
    if (object::ObjectFile *o = dyn_cast<ObjectFile>(obj.get()))
      DumpSymbolNamesFromObject(o);
  } else {
    errs() << ToolName << ": " << Filename << ": "
           << "unrecognizable file type\n";
    return;
  }
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);

  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv, "llvm symbol table dumper\n");

  // llvm-nm only reads binary files.
  if (error(sys::Program::ChangeStdinToBinary()))
    return 1;

  ToolName = argv[0];
  if (BSDFormat) OutputFormat = bsd;
  if (POSIXFormat) OutputFormat = posix;

  // The relative order of these is important. If you pass --size-sort it should
  // only print out the size. However, if you pass -S --size-sort, it should
  // print out both the size and address.
  if (SizeSort && !PrintSize) PrintAddress = false;
  if (OutputFormat == sysv || SizeSort) PrintSize = true;

  switch (InputFilenames.size()) {
  case 0: InputFilenames.push_back("-");
  case 1: break;
  default: MultipleFiles = true;
  }

  std::for_each(InputFilenames.begin(), InputFilenames.end(),
                DumpSymbolNamesFromFile);
  return 0;
}
