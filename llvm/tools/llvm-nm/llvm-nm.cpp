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

#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <system_error>
#include <vector>

using namespace llvm;
using namespace object;

namespace {
enum OutputFormatTy { bsd, sysv, posix, darwin };
cl::opt<OutputFormatTy> OutputFormat(
    "format", cl::desc("Specify output format"),
    cl::values(clEnumVal(bsd, "BSD format"), clEnumVal(sysv, "System V format"),
               clEnumVal(posix, "POSIX.2 format"),
               clEnumVal(darwin, "Darwin -m format"), clEnumValEnd),
    cl::init(bsd));
cl::alias OutputFormat2("f", cl::desc("Alias for --format"),
                        cl::aliasopt(OutputFormat));

cl::list<std::string> InputFilenames(cl::Positional, cl::desc("<input files>"),
                                     cl::ZeroOrMore);

cl::opt<bool> UndefinedOnly("undefined-only",
                            cl::desc("Show only undefined symbols"));
cl::alias UndefinedOnly2("u", cl::desc("Alias for --undefined-only"),
                         cl::aliasopt(UndefinedOnly), cl::Grouping);

cl::opt<bool> DynamicSyms("dynamic",
                          cl::desc("Display the dynamic symbols instead "
                                   "of normal symbols."));
cl::alias DynamicSyms2("D", cl::desc("Alias for --dynamic"),
                       cl::aliasopt(DynamicSyms), cl::Grouping);

cl::opt<bool> DefinedOnly("defined-only",
                          cl::desc("Show only defined symbols"));
cl::alias DefinedOnly2("U", cl::desc("Alias for --defined-only"),
                       cl::aliasopt(DefinedOnly), cl::Grouping);

cl::opt<bool> ExternalOnly("extern-only",
                           cl::desc("Show only external symbols"));
cl::alias ExternalOnly2("g", cl::desc("Alias for --extern-only"),
                        cl::aliasopt(ExternalOnly), cl::Grouping);

cl::opt<bool> BSDFormat("B", cl::desc("Alias for --format=bsd"),
                        cl::Grouping);
cl::opt<bool> POSIXFormat("P", cl::desc("Alias for --format=posix"),
                          cl::Grouping);
cl::opt<bool> DarwinFormat("m", cl::desc("Alias for --format=darwin"),
                           cl::Grouping);

static cl::list<std::string>
    ArchFlags("arch", cl::desc("architecture(s) from a Mach-O file to dump"),
              cl::ZeroOrMore);
bool ArchAll = false;

cl::opt<bool> PrintFileName(
    "print-file-name",
    cl::desc("Precede each symbol with the object file it came from"));

cl::alias PrintFileNameA("A", cl::desc("Alias for --print-file-name"),
                         cl::aliasopt(PrintFileName), cl::Grouping);
cl::alias PrintFileNameo("o", cl::desc("Alias for --print-file-name"),
                         cl::aliasopt(PrintFileName), cl::Grouping);

cl::opt<bool> DebugSyms("debug-syms",
                        cl::desc("Show all symbols, even debugger only"));
cl::alias DebugSymsa("a", cl::desc("Alias for --debug-syms"),
                     cl::aliasopt(DebugSyms), cl::Grouping);

cl::opt<bool> NumericSort("numeric-sort", cl::desc("Sort symbols by address"));
cl::alias NumericSortn("n", cl::desc("Alias for --numeric-sort"),
                       cl::aliasopt(NumericSort), cl::Grouping);
cl::alias NumericSortv("v", cl::desc("Alias for --numeric-sort"),
                       cl::aliasopt(NumericSort), cl::Grouping);

cl::opt<bool> NoSort("no-sort", cl::desc("Show symbols in order encountered"));
cl::alias NoSortp("p", cl::desc("Alias for --no-sort"), cl::aliasopt(NoSort),
                  cl::Grouping);

cl::opt<bool> ReverseSort("reverse-sort", cl::desc("Sort in reverse order"));
cl::alias ReverseSortr("r", cl::desc("Alias for --reverse-sort"),
                       cl::aliasopt(ReverseSort), cl::Grouping);

cl::opt<bool> PrintSize("print-size",
                        cl::desc("Show symbol size instead of address"));
cl::alias PrintSizeS("S", cl::desc("Alias for --print-size"),
                     cl::aliasopt(PrintSize), cl::Grouping);

cl::opt<bool> SizeSort("size-sort", cl::desc("Sort symbols by size"));

cl::opt<bool> WithoutAliases("without-aliases", cl::Hidden,
                             cl::desc("Exclude aliases from output"));

cl::opt<bool> ArchiveMap("print-armap", cl::desc("Print the archive map"));
cl::alias ArchiveMaps("M", cl::desc("Alias for --print-armap"),
                      cl::aliasopt(ArchiveMap), cl::Grouping);

cl::opt<bool> JustSymbolName("just-symbol-name",
                             cl::desc("Print just the symbol's name"));
cl::alias JustSymbolNames("j", cl::desc("Alias for --just-symbol-name"),
                          cl::aliasopt(JustSymbolName), cl::Grouping);

// FIXME: This option takes exactly two strings and should be allowed anywhere
// on the command line.  Such that "llvm-nm -s __TEXT __text foo.o" would work.
// But that does not as the CommandLine Library does not have a way to make
// this work.  For now the "-s __TEXT __text" has to be last on the command
// line.
cl::list<std::string> SegSect("s", cl::Positional, cl::ZeroOrMore,
                              cl::desc("Dump only symbols from this segment "
                                       "and section name, Mach-O only"));

cl::opt<bool> FormatMachOasHex("x", cl::desc("Print symbol entry in hex, "
                                             "Mach-O only"), cl::Grouping);

cl::opt<bool> NoLLVMBitcode("no-llvm-bc",
                            cl::desc("Disable LLVM bitcode reader"));

bool PrintAddress = true;

bool MultipleFiles = false;

bool HadError = false;

std::string ToolName;
} // anonymous namespace

static void error(Twine Message, Twine Path = Twine()) {
  HadError = true;
  errs() << ToolName << ": " << Path << ": " << Message << ".\n";
}

static bool error(std::error_code EC, Twine Path = Twine()) {
  if (EC) {
    error(EC.message(), Path);
    return true;
  }
  return false;
}

namespace {
struct NMSymbol {
  uint64_t Address;
  uint64_t Size;
  char TypeChar;
  StringRef Name;
  BasicSymbolRef Sym;
};
} // anonymous namespace

static bool compareSymbolAddress(const NMSymbol &A, const NMSymbol &B) {
  bool ADefined = !(A.Sym.getFlags() & SymbolRef::SF_Undefined);
  bool BDefined = !(B.Sym.getFlags() & SymbolRef::SF_Undefined);
  return std::make_tuple(ADefined, A.Address, A.Name, A.Size) <
         std::make_tuple(BDefined, B.Address, B.Name, B.Size);
}

static bool compareSymbolSize(const NMSymbol &A, const NMSymbol &B) {
  return std::make_tuple(A.Size, A.Name, A.Address) <
         std::make_tuple(B.Size, B.Name, B.Address);
}

static bool compareSymbolName(const NMSymbol &A, const NMSymbol &B) {
  return std::make_tuple(A.Name, A.Size, A.Address) <
         std::make_tuple(B.Name, B.Size, B.Address);
}

static char isSymbolList64Bit(SymbolicFile &Obj) {
  if (isa<IRObjectFile>(Obj)) {
    IRObjectFile *IRobj = dyn_cast<IRObjectFile>(&Obj);
    Module &M = IRobj->getModule();
    if (M.getTargetTriple().empty())
      return false;
    Triple T(M.getTargetTriple());
    return T.isArch64Bit();
  }
  if (isa<COFFObjectFile>(Obj))
    return false;
  if (MachOObjectFile *MachO = dyn_cast<MachOObjectFile>(&Obj))
    return MachO->is64Bit();
  return cast<ELFObjectFileBase>(Obj).getBytesInAddress() == 8;
}

static StringRef CurrentFilename;
typedef std::vector<NMSymbol> SymbolListT;
static SymbolListT SymbolList;

static char getSymbolNMTypeChar(IRObjectFile &Obj, basic_symbol_iterator I);

// darwinPrintSymbol() is used to print a symbol from a Mach-O file when the
// the OutputFormat is darwin or we are printing Mach-O symbols in hex.  For
// the darwin format it produces the same output as darwin's nm(1) -m output
// and when printing Mach-O symbols in hex it produces the same output as
// darwin's nm(1) -x format.
static void darwinPrintSymbol(SymbolicFile &Obj, SymbolListT::iterator I,
                              char *SymbolAddrStr, const char *printBlanks,
                              const char *printDashes, const char *printFormat) {
  MachO::mach_header H;
  MachO::mach_header_64 H_64;
  uint32_t Filetype = MachO::MH_OBJECT;
  uint32_t Flags = 0;
  uint8_t NType = 0;
  uint8_t NSect = 0;
  uint16_t NDesc = 0;
  uint32_t NStrx = 0;
  uint64_t NValue = 0;
  MachOObjectFile *MachO = dyn_cast<MachOObjectFile>(&Obj);
  if (Obj.isIR()) {
    uint32_t SymFlags = I->Sym.getFlags();
    if (SymFlags & SymbolRef::SF_Global)
      NType |= MachO::N_EXT;
    if (SymFlags & SymbolRef::SF_Hidden)
      NType |= MachO::N_PEXT;
    if (SymFlags & SymbolRef::SF_Undefined)
      NType |= MachO::N_EXT | MachO::N_UNDF;
    else {
      // Here we have a symbol definition.  So to fake out a section name we
      // use 1, 2 and 3 for section numbers.  See below where they are used to
      // print out fake section names.
      NType |= MachO::N_SECT;
      if(SymFlags & SymbolRef::SF_Const)
        NSect = 3;
      else {
        IRObjectFile *IRobj = dyn_cast<IRObjectFile>(&Obj);
        char c = getSymbolNMTypeChar(*IRobj, I->Sym);
        if (c == 't')
          NSect = 1;
        else
          NSect = 2;
      }
    }
    if (SymFlags & SymbolRef::SF_Weak)
      NDesc |= MachO::N_WEAK_DEF;
  } else {
    DataRefImpl SymDRI = I->Sym.getRawDataRefImpl();
    if (MachO->is64Bit()) {
      H_64 = MachO->MachOObjectFile::getHeader64();
      Filetype = H_64.filetype;
      Flags = H_64.flags;
      MachO::nlist_64 STE_64 = MachO->getSymbol64TableEntry(SymDRI);
      NType = STE_64.n_type;
      NSect = STE_64.n_sect;
      NDesc = STE_64.n_desc;
      NStrx = STE_64.n_strx;
      NValue = STE_64.n_value;
    } else {
      H = MachO->MachOObjectFile::getHeader();
      Filetype = H.filetype;
      Flags = H.flags;
      MachO::nlist STE = MachO->getSymbolTableEntry(SymDRI);
      NType = STE.n_type;
      NSect = STE.n_sect;
      NDesc = STE.n_desc;
      NStrx = STE.n_strx;
      NValue = STE.n_value;
    }
  }

  // If we are printing Mach-O symbols in hex do that and return.
  if (FormatMachOasHex) {
    char Str[18] = "";
    format(printFormat, NValue).print(Str, sizeof(Str));
    outs() << Str << ' ';
    format("%02x", NType).print(Str, sizeof(Str));
    outs() << Str << ' ';
    format("%02x", NSect).print(Str, sizeof(Str));
    outs() << Str << ' ';
    format("%04x", NDesc).print(Str, sizeof(Str));
    outs() << Str << ' ';
    format("%08x", NStrx).print(Str, sizeof(Str));
    outs() << Str << ' ';
    outs() << I->Name << "\n";
    return;
  }

  if (PrintAddress) {
    if ((NType & MachO::N_TYPE) == MachO::N_INDR)
      strcpy(SymbolAddrStr, printBlanks);
    if (Obj.isIR() && (NType & MachO::N_TYPE) == MachO::N_TYPE)
      strcpy(SymbolAddrStr, printDashes);
    outs() << SymbolAddrStr << ' ';
  }

  switch (NType & MachO::N_TYPE) {
  case MachO::N_UNDF:
    if (NValue != 0) {
      outs() << "(common) ";
      if (MachO::GET_COMM_ALIGN(NDesc) != 0)
        outs() << "(alignment 2^" << (int)MachO::GET_COMM_ALIGN(NDesc) << ") ";
    } else {
      if ((NType & MachO::N_TYPE) == MachO::N_PBUD)
        outs() << "(prebound ";
      else
        outs() << "(";
      if ((NDesc & MachO::REFERENCE_TYPE) ==
          MachO::REFERENCE_FLAG_UNDEFINED_LAZY)
        outs() << "undefined [lazy bound]) ";
      else if ((NDesc & MachO::REFERENCE_TYPE) ==
               MachO::REFERENCE_FLAG_UNDEFINED_LAZY)
        outs() << "undefined [private lazy bound]) ";
      else if ((NDesc & MachO::REFERENCE_TYPE) ==
               MachO::REFERENCE_FLAG_PRIVATE_UNDEFINED_NON_LAZY)
        outs() << "undefined [private]) ";
      else
        outs() << "undefined) ";
    }
    break;
  case MachO::N_ABS:
    outs() << "(absolute) ";
    break;
  case MachO::N_INDR:
    outs() << "(indirect) ";
    break;
  case MachO::N_SECT: {
    if (Obj.isIR()) {
      // For llvm bitcode files print out a fake section name using the values
      // use 1, 2 and 3 for section numbers as set above.
      if (NSect == 1)
        outs() << "(LTO,CODE) ";
      else if (NSect == 2)
        outs() << "(LTO,DATA) ";
      else if (NSect == 3)
        outs() << "(LTO,RODATA) ";
      else
        outs() << "(?,?) ";
      break;
    }
    ErrorOr<section_iterator> SecOrErr =
      MachO->getSymbolSection(I->Sym.getRawDataRefImpl());
    if (SecOrErr.getError()) {
      outs() << "(?,?) ";
      break;
    }
    section_iterator Sec = *SecOrErr;
    DataRefImpl Ref = Sec->getRawDataRefImpl();
    StringRef SectionName;
    MachO->getSectionName(Ref, SectionName);
    StringRef SegmentName = MachO->getSectionFinalSegmentName(Ref);
    outs() << "(" << SegmentName << "," << SectionName << ") ";
    break;
  }
  default:
    outs() << "(?) ";
    break;
  }

  if (NType & MachO::N_EXT) {
    if (NDesc & MachO::REFERENCED_DYNAMICALLY)
      outs() << "[referenced dynamically] ";
    if (NType & MachO::N_PEXT) {
      if ((NDesc & MachO::N_WEAK_DEF) == MachO::N_WEAK_DEF)
        outs() << "weak private external ";
      else
        outs() << "private external ";
    } else {
      if ((NDesc & MachO::N_WEAK_REF) == MachO::N_WEAK_REF ||
          (NDesc & MachO::N_WEAK_DEF) == MachO::N_WEAK_DEF) {
        if ((NDesc & (MachO::N_WEAK_REF | MachO::N_WEAK_DEF)) ==
            (MachO::N_WEAK_REF | MachO::N_WEAK_DEF))
          outs() << "weak external automatically hidden ";
        else
          outs() << "weak external ";
      } else
        outs() << "external ";
    }
  } else {
    if (NType & MachO::N_PEXT)
      outs() << "non-external (was a private external) ";
    else
      outs() << "non-external ";
  }

  if (Filetype == MachO::MH_OBJECT &&
      (NDesc & MachO::N_NO_DEAD_STRIP) == MachO::N_NO_DEAD_STRIP)
    outs() << "[no dead strip] ";

  if (Filetype == MachO::MH_OBJECT &&
      ((NType & MachO::N_TYPE) != MachO::N_UNDF) &&
      (NDesc & MachO::N_SYMBOL_RESOLVER) == MachO::N_SYMBOL_RESOLVER)
    outs() << "[symbol resolver] ";

  if (Filetype == MachO::MH_OBJECT &&
      ((NType & MachO::N_TYPE) != MachO::N_UNDF) &&
      (NDesc & MachO::N_ALT_ENTRY) == MachO::N_ALT_ENTRY)
    outs() << "[alt entry] ";

  if ((NDesc & MachO::N_ARM_THUMB_DEF) == MachO::N_ARM_THUMB_DEF)
    outs() << "[Thumb] ";

  if ((NType & MachO::N_TYPE) == MachO::N_INDR) {
    outs() << I->Name << " (for ";
    StringRef IndirectName;
    if (!MachO ||
        MachO->getIndirectName(I->Sym.getRawDataRefImpl(), IndirectName))
      outs() << "?)";
    else
      outs() << IndirectName << ")";
  } else
    outs() << I->Name;

  if ((Flags & MachO::MH_TWOLEVEL) == MachO::MH_TWOLEVEL &&
      (((NType & MachO::N_TYPE) == MachO::N_UNDF && NValue == 0) ||
       (NType & MachO::N_TYPE) == MachO::N_PBUD)) {
    uint32_t LibraryOrdinal = MachO::GET_LIBRARY_ORDINAL(NDesc);
    if (LibraryOrdinal != 0) {
      if (LibraryOrdinal == MachO::EXECUTABLE_ORDINAL)
        outs() << " (from executable)";
      else if (LibraryOrdinal == MachO::DYNAMIC_LOOKUP_ORDINAL)
        outs() << " (dynamically looked up)";
      else {
        StringRef LibraryName;
        if (!MachO ||
            MachO->getLibraryShortNameByIndex(LibraryOrdinal - 1, LibraryName))
          outs() << " (from bad library ordinal " << LibraryOrdinal << ")";
        else
          outs() << " (from " << LibraryName << ")";
      }
    }
  }

  outs() << "\n";
}

// Table that maps Darwin's Mach-O stab constants to strings to allow printing.
struct DarwinStabName {
  uint8_t NType;
  const char *Name;
};
static const struct DarwinStabName DarwinStabNames[] = {
    {MachO::N_GSYM, "GSYM"},
    {MachO::N_FNAME, "FNAME"},
    {MachO::N_FUN, "FUN"},
    {MachO::N_STSYM, "STSYM"},
    {MachO::N_LCSYM, "LCSYM"},
    {MachO::N_BNSYM, "BNSYM"},
    {MachO::N_PC, "PC"},
    {MachO::N_AST, "AST"},
    {MachO::N_OPT, "OPT"},
    {MachO::N_RSYM, "RSYM"},
    {MachO::N_SLINE, "SLINE"},
    {MachO::N_ENSYM, "ENSYM"},
    {MachO::N_SSYM, "SSYM"},
    {MachO::N_SO, "SO"},
    {MachO::N_OSO, "OSO"},
    {MachO::N_LSYM, "LSYM"},
    {MachO::N_BINCL, "BINCL"},
    {MachO::N_SOL, "SOL"},
    {MachO::N_PARAMS, "PARAM"},
    {MachO::N_VERSION, "VERS"},
    {MachO::N_OLEVEL, "OLEV"},
    {MachO::N_PSYM, "PSYM"},
    {MachO::N_EINCL, "EINCL"},
    {MachO::N_ENTRY, "ENTRY"},
    {MachO::N_LBRAC, "LBRAC"},
    {MachO::N_EXCL, "EXCL"},
    {MachO::N_RBRAC, "RBRAC"},
    {MachO::N_BCOMM, "BCOMM"},
    {MachO::N_ECOMM, "ECOMM"},
    {MachO::N_ECOML, "ECOML"},
    {MachO::N_LENG, "LENG"},
    {0, nullptr}};

static const char *getDarwinStabString(uint8_t NType) {
  for (unsigned i = 0; DarwinStabNames[i].Name; i++) {
    if (DarwinStabNames[i].NType == NType)
      return DarwinStabNames[i].Name;
  }
  return nullptr;
}

// darwinPrintStab() prints the n_sect, n_desc along with a symbolic name of
// a stab n_type value in a Mach-O file.
static void darwinPrintStab(MachOObjectFile *MachO, SymbolListT::iterator I) {
  MachO::nlist_64 STE_64;
  MachO::nlist STE;
  uint8_t NType;
  uint8_t NSect;
  uint16_t NDesc;
  DataRefImpl SymDRI = I->Sym.getRawDataRefImpl();
  if (MachO->is64Bit()) {
    STE_64 = MachO->getSymbol64TableEntry(SymDRI);
    NType = STE_64.n_type;
    NSect = STE_64.n_sect;
    NDesc = STE_64.n_desc;
  } else {
    STE = MachO->getSymbolTableEntry(SymDRI);
    NType = STE.n_type;
    NSect = STE.n_sect;
    NDesc = STE.n_desc;
  }

  char Str[18] = "";
  format("%02x", NSect).print(Str, sizeof(Str));
  outs() << ' ' << Str << ' ';
  format("%04x", NDesc).print(Str, sizeof(Str));
  outs() << Str << ' ';
  if (const char *stabString = getDarwinStabString(NType))
    format("%5.5s", stabString).print(Str, sizeof(Str));
  else
    format("   %02x", NType).print(Str, sizeof(Str));
  outs() << Str;
}

static void sortAndPrintSymbolList(SymbolicFile &Obj, bool printName,
                                   std::string ArchiveName,
                                   std::string ArchitectureName) {
  if (!NoSort) {
    std::function<bool(const NMSymbol &, const NMSymbol &)> Cmp;
    if (NumericSort)
      Cmp = compareSymbolAddress;
    else if (SizeSort)
      Cmp = compareSymbolSize;
    else
      Cmp = compareSymbolName;

    if (ReverseSort)
      Cmp = [=](const NMSymbol &A, const NMSymbol &B) { return Cmp(B, A); };
    std::sort(SymbolList.begin(), SymbolList.end(), Cmp);
  }

  if (!PrintFileName) {
    if (OutputFormat == posix && MultipleFiles && printName) {
      outs() << '\n' << CurrentFilename << ":\n";
    } else if (OutputFormat == bsd && MultipleFiles && printName) {
      outs() << "\n" << CurrentFilename << ":\n";
    } else if (OutputFormat == sysv) {
      outs() << "\n\nSymbols from " << CurrentFilename << ":\n\n"
             << "Name                  Value   Class        Type"
             << "         Size   Line  Section\n";
    }
  }

  const char *printBlanks, *printDashes, *printFormat;
  if (isSymbolList64Bit(Obj)) {
    printBlanks = "                ";
    printDashes = "----------------";
    printFormat = "%016" PRIx64;
  } else {
    printBlanks = "        ";
    printDashes = "--------";
    printFormat = "%08" PRIx64;
  }

  for (SymbolListT::iterator I = SymbolList.begin(), E = SymbolList.end();
       I != E; ++I) {
    uint32_t SymFlags = I->Sym.getFlags();
    bool Undefined = SymFlags & SymbolRef::SF_Undefined;
    bool Global = SymFlags & SymbolRef::SF_Global;
    if ((!Undefined && UndefinedOnly) || (Undefined && DefinedOnly) ||
        (!Global && ExternalOnly) || (SizeSort && !PrintAddress))
      continue;
    if (PrintFileName) {
      if (!ArchitectureName.empty())
        outs() << "(for architecture " << ArchitectureName << "):";
      if (!ArchiveName.empty())
        outs() << ArchiveName << ":";
      outs() << CurrentFilename << ": ";
    }
    if ((JustSymbolName || (UndefinedOnly && isa<MachOObjectFile>(Obj) &&
                            OutputFormat != darwin)) && OutputFormat != posix) {
      outs() << I->Name << "\n";
      continue;
    }

    char SymbolAddrStr[18] = "";
    char SymbolSizeStr[18] = "";

    if (OutputFormat == sysv || I->TypeChar == 'U')
      strcpy(SymbolAddrStr, printBlanks);
    if (OutputFormat == sysv)
      strcpy(SymbolSizeStr, printBlanks);

    if (I->TypeChar != 'U') {
      if (Obj.isIR())
        strcpy(SymbolAddrStr, printDashes);
      else
        format(printFormat, I->Address)
          .print(SymbolAddrStr, sizeof(SymbolAddrStr));
    }
    format(printFormat, I->Size).print(SymbolSizeStr, sizeof(SymbolSizeStr));

    // If OutputFormat is darwin or we are printing Mach-O symbols in hex and
    // we have a MachOObjectFile, call darwinPrintSymbol to print as darwin's
    // nm(1) -m output or hex, else if OutputFormat is darwin or we are
    // printing Mach-O symbols in hex and not a Mach-O object fall back to
    // OutputFormat bsd (see below).
    MachOObjectFile *MachO = dyn_cast<MachOObjectFile>(&Obj);
    if ((OutputFormat == darwin || FormatMachOasHex) && (MachO || Obj.isIR())) {
      darwinPrintSymbol(Obj, I, SymbolAddrStr, printBlanks, printDashes,
                        printFormat);
    } else if (OutputFormat == posix) {
      outs() << I->Name << " " << I->TypeChar << " ";
      if (MachO)
        outs() << I->Address << " " << "0" /* SymbolSizeStr */ << "\n";
      else
        outs() << SymbolAddrStr << SymbolSizeStr << "\n";
    } else if (OutputFormat == bsd || (OutputFormat == darwin && !MachO)) {
      if (PrintAddress)
        outs() << SymbolAddrStr << ' ';
      if (PrintSize) {
        outs() << SymbolSizeStr;
        outs() << ' ';
      }
      outs() << I->TypeChar;
      if (I->TypeChar == '-' && MachO)
        darwinPrintStab(MachO, I);
      outs() << " " << I->Name << "\n";
    } else if (OutputFormat == sysv) {
      std::string PaddedName(I->Name);
      while (PaddedName.length() < 20)
        PaddedName += " ";
      outs() << PaddedName << "|" << SymbolAddrStr << "|   " << I->TypeChar
             << "  |                  |" << SymbolSizeStr << "|     |\n";
    }
  }

  SymbolList.clear();
}

static char getSymbolNMTypeChar(ELFObjectFileBase &Obj,
                                basic_symbol_iterator I) {
  // OK, this is ELF
  elf_symbol_iterator SymI(I);

  ErrorOr<elf_section_iterator> SecIOrErr = SymI->getSection();
  if (error(SecIOrErr.getError()))
    return '?';

  elf_section_iterator SecI = *SecIOrErr;
  if (SecI != Obj.section_end()) {
    switch (SecI->getType()) {
    case ELF::SHT_PROGBITS:
    case ELF::SHT_DYNAMIC:
      switch (SecI->getFlags()) {
      case (ELF::SHF_ALLOC | ELF::SHF_EXECINSTR):
        return 't';
      case (ELF::SHF_TLS | ELF::SHF_ALLOC | ELF::SHF_WRITE):
      case (ELF::SHF_ALLOC | ELF::SHF_WRITE):
        return 'd';
      case ELF::SHF_ALLOC:
      case (ELF::SHF_ALLOC | ELF::SHF_MERGE):
      case (ELF::SHF_ALLOC | ELF::SHF_MERGE | ELF::SHF_STRINGS):
        return 'r';
      }
      break;
    case ELF::SHT_NOBITS:
      return 'b';
    }
  }

  if (SymI->getELFType() == ELF::STT_SECTION) {
    ErrorOr<StringRef> Name = SymI->getName();
    if (error(Name.getError()))
      return '?';
    return StringSwitch<char>(*Name)
        .StartsWith(".debug", 'N')
        .StartsWith(".note", 'n')
        .Default('?');
  }

  return 'n';
}

static char getSymbolNMTypeChar(COFFObjectFile &Obj, symbol_iterator I) {
  COFFSymbolRef Symb = Obj.getCOFFSymbol(*I);
  // OK, this is COFF.
  symbol_iterator SymI(I);

  ErrorOr<StringRef> Name = SymI->getName();
  if (error(Name.getError()))
    return '?';

  char Ret = StringSwitch<char>(*Name)
                 .StartsWith(".debug", 'N')
                 .StartsWith(".sxdata", 'N')
                 .Default('?');

  if (Ret != '?')
    return Ret;

  uint32_t Characteristics = 0;
  if (!COFF::isReservedSectionNumber(Symb.getSectionNumber())) {
    ErrorOr<section_iterator> SecIOrErr = SymI->getSection();
    if (error(SecIOrErr.getError()))
      return '?';
    section_iterator SecI = *SecIOrErr;
    const coff_section *Section = Obj.getCOFFSection(*SecI);
    Characteristics = Section->Characteristics;
  }

  switch (Symb.getSectionNumber()) {
  case COFF::IMAGE_SYM_DEBUG:
    return 'n';
  default:
    // Check section type.
    if (Characteristics & COFF::IMAGE_SCN_CNT_CODE)
      return 't';
    if (Characteristics & COFF::IMAGE_SCN_CNT_INITIALIZED_DATA)
      return Characteristics & COFF::IMAGE_SCN_MEM_WRITE ? 'd' : 'r';
    if (Characteristics & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA)
      return 'b';
    if (Characteristics & COFF::IMAGE_SCN_LNK_INFO)
      return 'i';
    // Check for section symbol.
    if (Symb.isSectionDefinition())
      return 's';
  }

  return '?';
}

static uint8_t getNType(MachOObjectFile &Obj, DataRefImpl Symb) {
  if (Obj.is64Bit()) {
    MachO::nlist_64 STE = Obj.getSymbol64TableEntry(Symb);
    return STE.n_type;
  }
  MachO::nlist STE = Obj.getSymbolTableEntry(Symb);
  return STE.n_type;
}

static char getSymbolNMTypeChar(MachOObjectFile &Obj, basic_symbol_iterator I) {
  DataRefImpl Symb = I->getRawDataRefImpl();
  uint8_t NType = getNType(Obj, Symb);

  if (NType & MachO::N_STAB)
    return '-';

  switch (NType & MachO::N_TYPE) {
  case MachO::N_ABS:
    return 's';
  case MachO::N_INDR:
    return 'i';
  case MachO::N_SECT: {
    ErrorOr<section_iterator> SecOrErr = Obj.getSymbolSection(Symb);
    if (SecOrErr.getError())
      return 's';
    section_iterator Sec = *SecOrErr;
    DataRefImpl Ref = Sec->getRawDataRefImpl();
    StringRef SectionName;
    Obj.getSectionName(Ref, SectionName);
    StringRef SegmentName = Obj.getSectionFinalSegmentName(Ref);
    if (SegmentName == "__TEXT" && SectionName == "__text")
      return 't';
    if (SegmentName == "__DATA" && SectionName == "__data")
      return 'd';
    if (SegmentName == "__DATA" && SectionName == "__bss")
      return 'b';
    return 's';
  }
  }

  return '?';
}

static char getSymbolNMTypeChar(const GlobalValue &GV) {
  if (GV.getValueType()->isFunctionTy())
    return 't';
  // FIXME: should we print 'b'? At the IR level we cannot be sure if this
  // will be in bss or not, but we could approximate.
  return 'd';
}

static char getSymbolNMTypeChar(IRObjectFile &Obj, basic_symbol_iterator I) {
  const GlobalValue *GV = Obj.getSymbolGV(I->getRawDataRefImpl());
  if (!GV)
    return 't';
  return getSymbolNMTypeChar(*GV);
}

static bool isObject(SymbolicFile &Obj, basic_symbol_iterator I) {
  auto *ELF = dyn_cast<ELFObjectFileBase>(&Obj);
  if (!ELF)
    return false;

  return elf_symbol_iterator(I)->getELFType() == ELF::STT_OBJECT;
}

static char getNMTypeChar(SymbolicFile &Obj, basic_symbol_iterator I) {
  uint32_t Symflags = I->getFlags();
  if ((Symflags & object::SymbolRef::SF_Weak) && !isa<MachOObjectFile>(Obj)) {
    char Ret = isObject(Obj, I) ? 'v' : 'w';
    if (!(Symflags & object::SymbolRef::SF_Undefined))
      Ret = toupper(Ret);
    return Ret;
  }

  if (Symflags & object::SymbolRef::SF_Undefined)
    return 'U';

  if (Symflags & object::SymbolRef::SF_Common)
    return 'C';

  char Ret = '?';
  if (Symflags & object::SymbolRef::SF_Absolute)
    Ret = 'a';
  else if (IRObjectFile *IR = dyn_cast<IRObjectFile>(&Obj)) {
    Ret = getSymbolNMTypeChar(*IR, I);
    Triple Host(sys::getDefaultTargetTriple());
    if (Ret == 'd' && Host.isOSDarwin()) {
      if(Symflags & SymbolRef::SF_Const)
        Ret = 's';
    }
  }
  else if (COFFObjectFile *COFF = dyn_cast<COFFObjectFile>(&Obj))
    Ret = getSymbolNMTypeChar(*COFF, I);
  else if (MachOObjectFile *MachO = dyn_cast<MachOObjectFile>(&Obj))
    Ret = getSymbolNMTypeChar(*MachO, I);
  else
    Ret = getSymbolNMTypeChar(cast<ELFObjectFileBase>(Obj), I);

  if (Symflags & object::SymbolRef::SF_Global)
    Ret = toupper(Ret);

  return Ret;
}

// getNsectForSegSect() is used to implement the Mach-O "-s segname sectname"
// option to dump only those symbols from that section in a Mach-O file.
// It is called once for each Mach-O file from dumpSymbolNamesFromObject()
// to get the section number for that named section from the command line
// arguments. It returns the section number for that section in the Mach-O
// file or zero it is not present.
static unsigned getNsectForSegSect(MachOObjectFile *Obj) {
  unsigned Nsect = 1;
  for (section_iterator I = Obj->section_begin(), E = Obj->section_end();
       I != E; ++I) {
    DataRefImpl Ref = I->getRawDataRefImpl();
    StringRef SectionName;
    Obj->getSectionName(Ref, SectionName);
    StringRef SegmentName = Obj->getSectionFinalSegmentName(Ref);
    if (SegmentName == SegSect[0] && SectionName == SegSect[1])
      return Nsect;
    Nsect++;
  }
  return 0;
}

// getNsectInMachO() is used to implement the Mach-O "-s segname sectname"
// option to dump only those symbols from that section in a Mach-O file.
// It is called once for each symbol in a Mach-O file from
// dumpSymbolNamesFromObject() and returns the section number for that symbol
// if it is in a section, else it returns 0.
static unsigned getNsectInMachO(MachOObjectFile &Obj, BasicSymbolRef Sym) {
  DataRefImpl Symb = Sym.getRawDataRefImpl();
  if (Obj.is64Bit()) {
    MachO::nlist_64 STE = Obj.getSymbol64TableEntry(Symb);
    if ((STE.n_type & MachO::N_TYPE) == MachO::N_SECT)
      return STE.n_sect;
    return 0;
  }
  MachO::nlist STE = Obj.getSymbolTableEntry(Symb);
  if ((STE.n_type & MachO::N_TYPE) == MachO::N_SECT)
    return STE.n_sect;
  return 0;
}

static void dumpSymbolNamesFromObject(SymbolicFile &Obj, bool printName,
                                      std::string ArchiveName = std::string(),
                                      std::string ArchitectureName =
                                        std::string()) {
  auto Symbols = Obj.symbols();
  if (DynamicSyms) {
    const auto *E = dyn_cast<ELFObjectFileBase>(&Obj);
    if (!E) {
      error("File format has no dynamic symbol table", Obj.getFileName());
      return;
    }
    auto DynSymbols = E->getDynamicSymbolIterators();
    Symbols =
        make_range<basic_symbol_iterator>(DynSymbols.begin(), DynSymbols.end());
  }
  std::string NameBuffer;
  raw_string_ostream OS(NameBuffer);
  // If a "-s segname sectname" option was specified and this is a Mach-O
  // file get the section number for that section in this object file.
  unsigned int Nsect = 0;
  MachOObjectFile *MachO = dyn_cast<MachOObjectFile>(&Obj);
  if (SegSect.size() != 0 && MachO) {
    Nsect = getNsectForSegSect(MachO);
    // If this section is not in the object file no symbols are printed.
    if (Nsect == 0)
      return;
  }
  for (BasicSymbolRef Sym : Symbols) {
    uint32_t SymFlags = Sym.getFlags();
    if (!DebugSyms && (SymFlags & SymbolRef::SF_FormatSpecific))
      continue;
    if (WithoutAliases) {
      if (IRObjectFile *IR = dyn_cast<IRObjectFile>(&Obj)) {
        const GlobalValue *GV = IR->getSymbolGV(Sym.getRawDataRefImpl());
        if (GV && isa<GlobalAlias>(GV))
          continue;
      }
    }
    // If a "-s segname sectname" option was specified and this is a Mach-O
    // file and this section appears in this file, Nsect will be non-zero then
    // see if this symbol is a symbol from that section and if not skip it.
    if (Nsect && Nsect != getNsectInMachO(*MachO, Sym))
      continue;
    NMSymbol S;
    S.Size = 0;
    S.Address = 0;
    if (PrintSize) {
      if (isa<ELFObjectFileBase>(&Obj))
        S.Size = ELFSymbolRef(Sym).getSize();
    }
    if (PrintAddress && isa<ObjectFile>(Obj)) {
      SymbolRef SymRef(Sym);
      ErrorOr<uint64_t> AddressOrErr = SymRef.getAddress();
      if (error(AddressOrErr.getError()))
        break;
      S.Address = *AddressOrErr;
    }
    S.TypeChar = getNMTypeChar(Obj, Sym);
    std::error_code EC = Sym.printName(OS);
    if (EC && MachO)
      OS << "bad string index";
    else 
      error(EC);
    OS << '\0';
    S.Sym = Sym;
    SymbolList.push_back(S);
  }

  OS.flush();
  const char *P = NameBuffer.c_str();
  for (unsigned I = 0; I < SymbolList.size(); ++I) {
    SymbolList[I].Name = P;
    P += strlen(P) + 1;
  }

  CurrentFilename = Obj.getFileName();
  sortAndPrintSymbolList(Obj, printName, ArchiveName, ArchitectureName);
}

// checkMachOAndArchFlags() checks to see if the SymbolicFile is a Mach-O file
// and if it is and there is a list of architecture flags is specified then
// check to make sure this Mach-O file is one of those architectures or all
// architectures was specificed.  If not then an error is generated and this
// routine returns false.  Else it returns true.
static bool checkMachOAndArchFlags(SymbolicFile *O, std::string &Filename) {
  MachOObjectFile *MachO = dyn_cast<MachOObjectFile>(O);

  if (!MachO || ArchAll || ArchFlags.size() == 0)
    return true;

  MachO::mach_header H;
  MachO::mach_header_64 H_64;
  Triple T;
  if (MachO->is64Bit()) {
    H_64 = MachO->MachOObjectFile::getHeader64();
    T = MachOObjectFile::getArch(H_64.cputype, H_64.cpusubtype);
  } else {
    H = MachO->MachOObjectFile::getHeader();
    T = MachOObjectFile::getArch(H.cputype, H.cpusubtype);
  }
  if (std::none_of(
          ArchFlags.begin(), ArchFlags.end(),
          [&](const std::string &Name) { return Name == T.getArchName(); })) {
    error("No architecture specified", Filename);
    return false;
  }
  return true;
}

static void dumpSymbolNamesFromFile(std::string &Filename) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFileOrSTDIN(Filename);
  if (error(BufferOrErr.getError(), Filename))
    return;

  LLVMContext &Context = getGlobalContext();
  ErrorOr<std::unique_ptr<Binary>> BinaryOrErr = createBinary(
      BufferOrErr.get()->getMemBufferRef(), NoLLVMBitcode ? nullptr : &Context);
  if (error(BinaryOrErr.getError(), Filename))
    return;
  Binary &Bin = *BinaryOrErr.get();

  if (Archive *A = dyn_cast<Archive>(&Bin)) {
    if (ArchiveMap) {
      Archive::symbol_iterator I = A->symbol_begin();
      Archive::symbol_iterator E = A->symbol_end();
      if (I != E) {
        outs() << "Archive map\n";
        for (; I != E; ++I) {
          ErrorOr<Archive::Child> C = I->getMember();
          if (error(C.getError()))
            return;
          ErrorOr<StringRef> FileNameOrErr = C->getName();
          if (error(FileNameOrErr.getError()))
            return;
          StringRef SymName = I->getName();
          outs() << SymName << " in " << FileNameOrErr.get() << "\n";
        }
        outs() << "\n";
      }
    }

    for (Archive::child_iterator I = A->child_begin(), E = A->child_end();
         I != E; ++I) {
      if (error(I->getError()))
        return;
      auto &C = I->get();
      ErrorOr<std::unique_ptr<Binary>> ChildOrErr = C.getAsBinary(&Context);
      if (ChildOrErr.getError())
        continue;
      if (SymbolicFile *O = dyn_cast<SymbolicFile>(&*ChildOrErr.get())) {
        if (!checkMachOAndArchFlags(O, Filename))
          return;
        if (!PrintFileName) {
          outs() << "\n";
          if (isa<MachOObjectFile>(O)) {
            outs() << Filename << "(" << O->getFileName() << ")";
          } else
            outs() << O->getFileName();
          outs() << ":\n";
        }
        dumpSymbolNamesFromObject(*O, false, Filename);
      }
    }
    return;
  }
  if (MachOUniversalBinary *UB = dyn_cast<MachOUniversalBinary>(&Bin)) {
    // If we have a list of architecture flags specified dump only those.
    if (!ArchAll && ArchFlags.size() != 0) {
      // Look for a slice in the universal binary that matches each ArchFlag.
      bool ArchFound;
      for (unsigned i = 0; i < ArchFlags.size(); ++i) {
        ArchFound = false;
        for (MachOUniversalBinary::object_iterator I = UB->begin_objects(),
                                                   E = UB->end_objects();
             I != E; ++I) {
          if (ArchFlags[i] == I->getArchTypeName()) {
            ArchFound = true;
            ErrorOr<std::unique_ptr<ObjectFile>> ObjOrErr =
                I->getAsObjectFile();
            std::string ArchiveName;
            std::string ArchitectureName;
            ArchiveName.clear();
            ArchitectureName.clear();
            if (ObjOrErr) {
              ObjectFile &Obj = *ObjOrErr.get();
              if (ArchFlags.size() > 1) {
                if (PrintFileName)
                  ArchitectureName = I->getArchTypeName();
                else
                  outs() << "\n" << Obj.getFileName() << " (for architecture "
                         << I->getArchTypeName() << ")"
                         << ":\n";
              }
              dumpSymbolNamesFromObject(Obj, false, ArchiveName,
                                        ArchitectureName);
            } else if (ErrorOr<std::unique_ptr<Archive>> AOrErr =
                           I->getAsArchive()) {
              std::unique_ptr<Archive> &A = *AOrErr;
              for (Archive::child_iterator AI = A->child_begin(),
                                           AE = A->child_end();
                   AI != AE; ++AI) {
                if (error(AI->getError()))
                  return;
                auto &C = AI->get();
                ErrorOr<std::unique_ptr<Binary>> ChildOrErr =
                    C.getAsBinary(&Context);
                if (ChildOrErr.getError())
                  continue;
                if (SymbolicFile *O =
                        dyn_cast<SymbolicFile>(&*ChildOrErr.get())) {
                  if (PrintFileName) {
                    ArchiveName = A->getFileName();
                    if (ArchFlags.size() > 1)
                      ArchitectureName = I->getArchTypeName();
                  } else {
                    outs() << "\n" << A->getFileName();
                    outs() << "(" << O->getFileName() << ")";
                    if (ArchFlags.size() > 1) {
                      outs() << " (for architecture " << I->getArchTypeName()
                             << ")";
                    }
                    outs() << ":\n";
                  }
                  dumpSymbolNamesFromObject(*O, false, ArchiveName,
                                            ArchitectureName);
                }
              }
            }
          }
        }
        if (!ArchFound) {
          error(ArchFlags[i],
                "file: " + Filename + " does not contain architecture");
          return;
        }
      }
      return;
    }
    // No architecture flags were specified so if this contains a slice that
    // matches the host architecture dump only that.
    if (!ArchAll) {
      StringRef HostArchName = MachOObjectFile::getHostArch().getArchName();
      for (MachOUniversalBinary::object_iterator I = UB->begin_objects(),
                                                 E = UB->end_objects();
           I != E; ++I) {
        if (HostArchName == I->getArchTypeName()) {
          ErrorOr<std::unique_ptr<ObjectFile>> ObjOrErr = I->getAsObjectFile();
          std::string ArchiveName;
          ArchiveName.clear();
          if (ObjOrErr) {
            ObjectFile &Obj = *ObjOrErr.get();
            dumpSymbolNamesFromObject(Obj, false);
          } else if (ErrorOr<std::unique_ptr<Archive>> AOrErr =
                         I->getAsArchive()) {
            std::unique_ptr<Archive> &A = *AOrErr;
            for (Archive::child_iterator AI = A->child_begin(),
                                         AE = A->child_end();
                 AI != AE; ++AI) {
              if (error(AI->getError()))
                return;
              auto &C = AI->get();
              ErrorOr<std::unique_ptr<Binary>> ChildOrErr =
                  C.getAsBinary(&Context);
              if (ChildOrErr.getError())
                continue;
              if (SymbolicFile *O =
                      dyn_cast<SymbolicFile>(&*ChildOrErr.get())) {
                if (PrintFileName)
                  ArchiveName = A->getFileName();
                else
                  outs() << "\n" << A->getFileName() << "(" << O->getFileName()
                         << ")"
                         << ":\n";
                dumpSymbolNamesFromObject(*O, false, ArchiveName);
              }
            }
          }
          return;
        }
      }
    }
    // Either all architectures have been specified or none have been specified
    // and this does not contain the host architecture so dump all the slices.
    bool moreThanOneArch = UB->getNumberOfObjects() > 1;
    for (MachOUniversalBinary::object_iterator I = UB->begin_objects(),
                                               E = UB->end_objects();
         I != E; ++I) {
      ErrorOr<std::unique_ptr<ObjectFile>> ObjOrErr = I->getAsObjectFile();
      std::string ArchiveName;
      std::string ArchitectureName;
      ArchiveName.clear();
      ArchitectureName.clear();
      if (ObjOrErr) {
        ObjectFile &Obj = *ObjOrErr.get();
        if (PrintFileName) {
          if (isa<MachOObjectFile>(Obj) && moreThanOneArch)
            ArchitectureName = I->getArchTypeName();
        } else {
          if (moreThanOneArch)
            outs() << "\n";
          outs() << Obj.getFileName();
          if (isa<MachOObjectFile>(Obj) && moreThanOneArch)
            outs() << " (for architecture " << I->getArchTypeName() << ")";
          outs() << ":\n";
        }
        dumpSymbolNamesFromObject(Obj, false, ArchiveName, ArchitectureName);
      } else if (ErrorOr<std::unique_ptr<Archive>> AOrErr = I->getAsArchive()) {
        std::unique_ptr<Archive> &A = *AOrErr;
        for (Archive::child_iterator AI = A->child_begin(), AE = A->child_end();
             AI != AE; ++AI) {
          if (error(AI->getError()))
            return;
          auto &C = AI->get();
          ErrorOr<std::unique_ptr<Binary>> ChildOrErr = C.getAsBinary(&Context);
          if (ChildOrErr.getError())
            continue;
          if (SymbolicFile *O = dyn_cast<SymbolicFile>(&*ChildOrErr.get())) {
            if (PrintFileName) {
              ArchiveName = A->getFileName();
              if (isa<MachOObjectFile>(O) && moreThanOneArch)
                ArchitectureName = I->getArchTypeName();
            } else {
              outs() << "\n" << A->getFileName();
              if (isa<MachOObjectFile>(O)) {
                outs() << "(" << O->getFileName() << ")";
                if (moreThanOneArch)
                  outs() << " (for architecture " << I->getArchTypeName()
                         << ")";
              } else
                outs() << ":" << O->getFileName();
              outs() << ":\n";
            }
            dumpSymbolNamesFromObject(*O, false, ArchiveName, ArchitectureName);
          }
        }
      }
    }
    return;
  }
  if (SymbolicFile *O = dyn_cast<SymbolicFile>(&Bin)) {
    if (!checkMachOAndArchFlags(O, Filename))
      return;
    dumpSymbolNamesFromObject(*O, true);
    return;
  }
  error("unrecognizable file type", Filename);
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);

  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv, "llvm symbol table dumper\n");

  // llvm-nm only reads binary files.
  if (error(sys::ChangeStdinToBinary()))
    return 1;

  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();

  ToolName = argv[0];
  if (BSDFormat)
    OutputFormat = bsd;
  if (POSIXFormat)
    OutputFormat = posix;
  if (DarwinFormat)
    OutputFormat = darwin;

  // The relative order of these is important. If you pass --size-sort it should
  // only print out the size. However, if you pass -S --size-sort, it should
  // print out both the size and address.
  if (SizeSort && !PrintSize)
    PrintAddress = false;
  if (OutputFormat == sysv || SizeSort)
    PrintSize = true;

  switch (InputFilenames.size()) {
  case 0:
    InputFilenames.push_back("a.out");
  case 1:
    break;
  default:
    MultipleFiles = true;
  }

  for (unsigned i = 0; i < ArchFlags.size(); ++i) {
    if (ArchFlags[i] == "all") {
      ArchAll = true;
    } else {
      if (!MachOObjectFile::isValidArch(ArchFlags[i]))
        error("Unknown architecture named '" + ArchFlags[i] + "'",
              "for the -arch option");
    }
  }

  if (SegSect.size() != 0 && SegSect.size() != 2)
    error("bad number of arguments (must be two arguments)",
          "for the -s option");

  std::for_each(InputFilenames.begin(), InputFilenames.end(),
                dumpSymbolNamesFromFile);

  if (HadError)
    return 1;

  return 0;
}
