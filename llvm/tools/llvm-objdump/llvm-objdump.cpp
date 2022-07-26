//===-- llvm-objdump.cpp - Object file dumping utility for llvm -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This program is a utility that works like binutils "objdump", that is, it
// dumps out a plethora of information about an object file depending on the
// flags.
//
// The flags and output of this program should be near identical to those of
// binutils objdump.
//
//===----------------------------------------------------------------------===//

#include "llvm-objdump.h"
#include "COFFDump.h"
#include "ELFDump.h"
#include "MachODump.h"
#include "ObjdumpOptID.h"
#include "OffloadDump.h"
#include "SourcePrinter.h"
#include "WasmDump.h"
#include "XCOFFDump.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/Twine.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/Symbolize/SymbolizableModule.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCDisassembler/MCRelocationInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/COFFImportFile.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/FaultMapParser.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Object/Wasm.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <system_error>
#include <unordered_map>
#include <utility>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::objdump;
using namespace llvm::opt;

namespace {

class CommonOptTable : public opt::OptTable {
public:
  CommonOptTable(ArrayRef<Info> OptionInfos, const char *Usage,
                 const char *Description)
      : OptTable(OptionInfos), Usage(Usage), Description(Description) {
    setGroupedShortOptions(true);
  }

  void printHelp(StringRef Argv0, bool ShowHidden = false) const {
    Argv0 = sys::path::filename(Argv0);
    opt::OptTable::printHelp(outs(), (Argv0 + Usage).str().c_str(), Description,
                             ShowHidden, ShowHidden);
    // TODO Replace this with OptTable API once it adds extrahelp support.
    outs() << "\nPass @FILE as argument to read options from FILE.\n";
  }

private:
  const char *Usage;
  const char *Description;
};

// ObjdumpOptID is in ObjdumpOptID.h

#define PREFIX(NAME, VALUE) const char *const OBJDUMP_##NAME[] = VALUE;
#include "ObjdumpOpts.inc"
#undef PREFIX

static constexpr opt::OptTable::Info ObjdumpInfoTable[] = {
#define OBJDUMP_nullptr nullptr
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  {OBJDUMP_##PREFIX, NAME,         HELPTEXT,                                   \
   METAVAR,          OBJDUMP_##ID, opt::Option::KIND##Class,                   \
   PARAM,            FLAGS,        OBJDUMP_##GROUP,                            \
   OBJDUMP_##ALIAS,  ALIASARGS,    VALUES},
#include "ObjdumpOpts.inc"
#undef OPTION
#undef OBJDUMP_nullptr
};

class ObjdumpOptTable : public CommonOptTable {
public:
  ObjdumpOptTable()
      : CommonOptTable(ObjdumpInfoTable, " [options] <input object files>",
                       "llvm object file dumper") {}
};

enum OtoolOptID {
  OTOOL_INVALID = 0, // This is not an option ID.
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  OTOOL_##ID,
#include "OtoolOpts.inc"
#undef OPTION
};

#define PREFIX(NAME, VALUE) const char *const OTOOL_##NAME[] = VALUE;
#include "OtoolOpts.inc"
#undef PREFIX

static constexpr opt::OptTable::Info OtoolInfoTable[] = {
#define OTOOL_nullptr nullptr
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  {OTOOL_##PREFIX, NAME,       HELPTEXT,                                       \
   METAVAR,        OTOOL_##ID, opt::Option::KIND##Class,                       \
   PARAM,          FLAGS,      OTOOL_##GROUP,                                  \
   OTOOL_##ALIAS,  ALIASARGS,  VALUES},
#include "OtoolOpts.inc"
#undef OPTION
#undef OTOOL_nullptr
};

class OtoolOptTable : public CommonOptTable {
public:
  OtoolOptTable()
      : CommonOptTable(OtoolInfoTable, " [option...] [file...]",
                       "Mach-O object file displaying tool") {}
};

} // namespace

#define DEBUG_TYPE "objdump"

static uint64_t AdjustVMA;
static bool AllHeaders;
static std::string ArchName;
bool objdump::ArchiveHeaders;
bool objdump::Demangle;
bool objdump::Disassemble;
bool objdump::DisassembleAll;
bool objdump::SymbolDescription;
static std::vector<std::string> DisassembleSymbols;
static bool DisassembleZeroes;
static std::vector<std::string> DisassemblerOptions;
DIDumpType objdump::DwarfDumpType;
static bool DynamicRelocations;
static bool FaultMapSection;
static bool FileHeaders;
bool objdump::SectionContents;
static std::vector<std::string> InputFilenames;
bool objdump::PrintLines;
static bool MachOOpt;
std::string objdump::MCPU;
std::vector<std::string> objdump::MAttrs;
bool objdump::ShowRawInsn;
bool objdump::LeadingAddr;
static bool Offloading;
static bool RawClangAST;
bool objdump::Relocations;
bool objdump::PrintImmHex;
bool objdump::PrivateHeaders;
std::vector<std::string> objdump::FilterSections;
bool objdump::SectionHeaders;
static bool ShowLMA;
bool objdump::PrintSource;

static uint64_t StartAddress;
static bool HasStartAddressFlag;
static uint64_t StopAddress = UINT64_MAX;
static bool HasStopAddressFlag;

bool objdump::SymbolTable;
static bool SymbolizeOperands;
static bool DynamicSymbolTable;
std::string objdump::TripleName;
bool objdump::UnwindInfo;
static bool Wide;
std::string objdump::Prefix;
uint32_t objdump::PrefixStrip;

DebugVarsFormat objdump::DbgVariables = DVDisabled;

int objdump::DbgIndent = 52;

static StringSet<> DisasmSymbolSet;
StringSet<> objdump::FoundSectionSet;
static StringRef ToolName;

namespace {
struct FilterResult {
  // True if the section should not be skipped.
  bool Keep;

  // True if the index counter should be incremented, even if the section should
  // be skipped. For example, sections may be skipped if they are not included
  // in the --section flag, but we still want those to count toward the section
  // count.
  bool IncrementIndex;
};
} // namespace

static FilterResult checkSectionFilter(object::SectionRef S) {
  if (FilterSections.empty())
    return {/*Keep=*/true, /*IncrementIndex=*/true};

  Expected<StringRef> SecNameOrErr = S.getName();
  if (!SecNameOrErr) {
    consumeError(SecNameOrErr.takeError());
    return {/*Keep=*/false, /*IncrementIndex=*/false};
  }
  StringRef SecName = *SecNameOrErr;

  // StringSet does not allow empty key so avoid adding sections with
  // no name (such as the section with index 0) here.
  if (!SecName.empty())
    FoundSectionSet.insert(SecName);

  // Only show the section if it's in the FilterSections list, but always
  // increment so the indexing is stable.
  return {/*Keep=*/is_contained(FilterSections, SecName),
          /*IncrementIndex=*/true};
}

SectionFilter objdump::ToolSectionFilter(object::ObjectFile const &O,
                                         uint64_t *Idx) {
  // Start at UINT64_MAX so that the first index returned after an increment is
  // zero (after the unsigned wrap).
  if (Idx)
    *Idx = UINT64_MAX;
  return SectionFilter(
      [Idx](object::SectionRef S) {
        FilterResult Result = checkSectionFilter(S);
        if (Idx != nullptr && Result.IncrementIndex)
          *Idx += 1;
        return Result.Keep;
      },
      O);
}

std::string objdump::getFileNameForError(const object::Archive::Child &C,
                                         unsigned Index) {
  Expected<StringRef> NameOrErr = C.getName();
  if (NameOrErr)
    return std::string(NameOrErr.get());
  // If we have an error getting the name then we print the index of the archive
  // member. Since we are already in an error state, we just ignore this error.
  consumeError(NameOrErr.takeError());
  return "<file index: " + std::to_string(Index) + ">";
}

void objdump::reportWarning(const Twine &Message, StringRef File) {
  // Output order between errs() and outs() matters especially for archive
  // files where the output is per member object.
  outs().flush();
  WithColor::warning(errs(), ToolName)
      << "'" << File << "': " << Message << "\n";
}

[[noreturn]] void objdump::reportError(StringRef File, const Twine &Message) {
  outs().flush();
  WithColor::error(errs(), ToolName) << "'" << File << "': " << Message << "\n";
  exit(1);
}

[[noreturn]] void objdump::reportError(Error E, StringRef FileName,
                                       StringRef ArchiveName,
                                       StringRef ArchitectureName) {
  assert(E);
  outs().flush();
  WithColor::error(errs(), ToolName);
  if (ArchiveName != "")
    errs() << ArchiveName << "(" << FileName << ")";
  else
    errs() << "'" << FileName << "'";
  if (!ArchitectureName.empty())
    errs() << " (for architecture " << ArchitectureName << ")";
  errs() << ": ";
  logAllUnhandledErrors(std::move(E), errs());
  exit(1);
}

static void reportCmdLineWarning(const Twine &Message) {
  WithColor::warning(errs(), ToolName) << Message << "\n";
}

[[noreturn]] static void reportCmdLineError(const Twine &Message) {
  WithColor::error(errs(), ToolName) << Message << "\n";
  exit(1);
}

static void warnOnNoMatchForSections() {
  SetVector<StringRef> MissingSections;
  for (StringRef S : FilterSections) {
    if (FoundSectionSet.count(S))
      return;
    // User may specify a unnamed section. Don't warn for it.
    if (!S.empty())
      MissingSections.insert(S);
  }

  // Warn only if no section in FilterSections is matched.
  for (StringRef S : MissingSections)
    reportCmdLineWarning("section '" + S +
                         "' mentioned in a -j/--section option, but not "
                         "found in any input file");
}

static const Target *getTarget(const ObjectFile *Obj) {
  // Figure out the target triple.
  Triple TheTriple("unknown-unknown-unknown");
  if (TripleName.empty()) {
    TheTriple = Obj->makeTriple();
  } else {
    TheTriple.setTriple(Triple::normalize(TripleName));
    auto Arch = Obj->getArch();
    if (Arch == Triple::arm || Arch == Triple::armeb)
      Obj->setARMSubArch(TheTriple);
  }

  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(ArchName, TheTriple,
                                                         Error);
  if (!TheTarget)
    reportError(Obj->getFileName(), "can't find target: " + Error);

  // Update the triple name and return the found target.
  TripleName = TheTriple.getTriple();
  return TheTarget;
}

bool objdump::isRelocAddressLess(RelocationRef A, RelocationRef B) {
  return A.getOffset() < B.getOffset();
}

static Error getRelocationValueString(const RelocationRef &Rel,
                                      SmallVectorImpl<char> &Result) {
  const ObjectFile *Obj = Rel.getObject();
  if (auto *ELF = dyn_cast<ELFObjectFileBase>(Obj))
    return getELFRelocationValueString(ELF, Rel, Result);
  if (auto *COFF = dyn_cast<COFFObjectFile>(Obj))
    return getCOFFRelocationValueString(COFF, Rel, Result);
  if (auto *Wasm = dyn_cast<WasmObjectFile>(Obj))
    return getWasmRelocationValueString(Wasm, Rel, Result);
  if (auto *MachO = dyn_cast<MachOObjectFile>(Obj))
    return getMachORelocationValueString(MachO, Rel, Result);
  if (auto *XCOFF = dyn_cast<XCOFFObjectFile>(Obj))
    return getXCOFFRelocationValueString(*XCOFF, Rel, Result);
  llvm_unreachable("unknown object file format");
}

/// Indicates whether this relocation should hidden when listing
/// relocations, usually because it is the trailing part of a multipart
/// relocation that will be printed as part of the leading relocation.
static bool getHidden(RelocationRef RelRef) {
  auto *MachO = dyn_cast<MachOObjectFile>(RelRef.getObject());
  if (!MachO)
    return false;

  unsigned Arch = MachO->getArch();
  DataRefImpl Rel = RelRef.getRawDataRefImpl();
  uint64_t Type = MachO->getRelocationType(Rel);

  // On arches that use the generic relocations, GENERIC_RELOC_PAIR
  // is always hidden.
  if (Arch == Triple::x86 || Arch == Triple::arm || Arch == Triple::ppc)
    return Type == MachO::GENERIC_RELOC_PAIR;

  if (Arch == Triple::x86_64) {
    // On x86_64, X86_64_RELOC_UNSIGNED is hidden only when it follows
    // an X86_64_RELOC_SUBTRACTOR.
    if (Type == MachO::X86_64_RELOC_UNSIGNED && Rel.d.a > 0) {
      DataRefImpl RelPrev = Rel;
      RelPrev.d.a--;
      uint64_t PrevType = MachO->getRelocationType(RelPrev);
      if (PrevType == MachO::X86_64_RELOC_SUBTRACTOR)
        return true;
    }
  }

  return false;
}

namespace {

/// Get the column at which we want to start printing the instruction
/// disassembly, taking into account anything which appears to the left of it.
unsigned getInstStartColumn(const MCSubtargetInfo &STI) {
  return !ShowRawInsn ? 16 : STI.getTargetTriple().isX86() ? 40 : 24;
}

static bool isAArch64Elf(const ObjectFile &Obj) {
  const auto *Elf = dyn_cast<ELFObjectFileBase>(&Obj);
  return Elf && Elf->getEMachine() == ELF::EM_AARCH64;
}

static bool isArmElf(const ObjectFile &Obj) {
  const auto *Elf = dyn_cast<ELFObjectFileBase>(&Obj);
  return Elf && Elf->getEMachine() == ELF::EM_ARM;
}

static bool isCSKYElf(const ObjectFile &Obj) {
  const auto *Elf = dyn_cast<ELFObjectFileBase>(&Obj);
  return Elf && Elf->getEMachine() == ELF::EM_CSKY;
}

static bool hasMappingSymbols(const ObjectFile &Obj) {
  return isArmElf(Obj) || isAArch64Elf(Obj) || isCSKYElf(Obj) ;
}

static void printRelocation(formatted_raw_ostream &OS, StringRef FileName,
                            const RelocationRef &Rel, uint64_t Address,
                            bool Is64Bits) {
  StringRef Fmt = Is64Bits ? "\t\t%016" PRIx64 ":  " : "\t\t\t%08" PRIx64 ":  ";
  SmallString<16> Name;
  SmallString<32> Val;
  Rel.getTypeName(Name);
  if (Error E = getRelocationValueString(Rel, Val))
    reportError(std::move(E), FileName);
  OS << format(Fmt.data(), Address) << Name << "\t" << Val;
}

static void AlignToInstStartColumn(size_t Start, const MCSubtargetInfo &STI,
                                   raw_ostream &OS) {
  // The output of printInst starts with a tab. Print some spaces so that
  // the tab has 1 column and advances to the target tab stop.
  unsigned TabStop = getInstStartColumn(STI);
  unsigned Column = OS.tell() - Start;
  OS.indent(Column < TabStop - 1 ? TabStop - 1 - Column : 7 - Column % 8);
}

class PrettyPrinter {
public:
  virtual ~PrettyPrinter() = default;
  virtual void
  printInst(MCInstPrinter &IP, const MCInst *MI, ArrayRef<uint8_t> Bytes,
            object::SectionedAddress Address, formatted_raw_ostream &OS,
            StringRef Annot, MCSubtargetInfo const &STI, SourcePrinter *SP,
            StringRef ObjectFilename, std::vector<RelocationRef> *Rels,
            LiveVariablePrinter &LVP) {
    if (SP && (PrintSource || PrintLines))
      SP->printSourceLine(OS, Address, ObjectFilename, LVP);
    LVP.printBetweenInsts(OS, false);

    size_t Start = OS.tell();
    if (LeadingAddr)
      OS << format("%8" PRIx64 ":", Address.Address);
    if (ShowRawInsn) {
      OS << ' ';
      dumpBytes(Bytes, OS);
    }

    AlignToInstStartColumn(Start, STI, OS);

    if (MI) {
      // See MCInstPrinter::printInst. On targets where a PC relative immediate
      // is relative to the next instruction and the length of a MCInst is
      // difficult to measure (x86), this is the address of the next
      // instruction.
      uint64_t Addr =
          Address.Address + (STI.getTargetTriple().isX86() ? Bytes.size() : 0);
      IP.printInst(MI, Addr, "", STI, OS);
    } else
      OS << "\t<unknown>";
  }
};
PrettyPrinter PrettyPrinterInst;

class HexagonPrettyPrinter : public PrettyPrinter {
public:
  void printLead(ArrayRef<uint8_t> Bytes, uint64_t Address,
                 formatted_raw_ostream &OS) {
    uint32_t opcode =
      (Bytes[3] << 24) | (Bytes[2] << 16) | (Bytes[1] << 8) | Bytes[0];
    if (LeadingAddr)
      OS << format("%8" PRIx64 ":", Address);
    if (ShowRawInsn) {
      OS << "\t";
      dumpBytes(Bytes.slice(0, 4), OS);
      OS << format("\t%08" PRIx32, opcode);
    }
  }
  void printInst(MCInstPrinter &IP, const MCInst *MI, ArrayRef<uint8_t> Bytes,
                 object::SectionedAddress Address, formatted_raw_ostream &OS,
                 StringRef Annot, MCSubtargetInfo const &STI, SourcePrinter *SP,
                 StringRef ObjectFilename, std::vector<RelocationRef> *Rels,
                 LiveVariablePrinter &LVP) override {
    if (SP && (PrintSource || PrintLines))
      SP->printSourceLine(OS, Address, ObjectFilename, LVP, "");
    if (!MI) {
      printLead(Bytes, Address.Address, OS);
      OS << " <unknown>";
      return;
    }
    std::string Buffer;
    {
      raw_string_ostream TempStream(Buffer);
      IP.printInst(MI, Address.Address, "", STI, TempStream);
    }
    StringRef Contents(Buffer);
    // Split off bundle attributes
    auto PacketBundle = Contents.rsplit('\n');
    // Split off first instruction from the rest
    auto HeadTail = PacketBundle.first.split('\n');
    auto Preamble = " { ";
    auto Separator = "";

    // Hexagon's packets require relocations to be inline rather than
    // clustered at the end of the packet.
    std::vector<RelocationRef>::const_iterator RelCur = Rels->begin();
    std::vector<RelocationRef>::const_iterator RelEnd = Rels->end();
    auto PrintReloc = [&]() -> void {
      while ((RelCur != RelEnd) && (RelCur->getOffset() <= Address.Address)) {
        if (RelCur->getOffset() == Address.Address) {
          printRelocation(OS, ObjectFilename, *RelCur, Address.Address, false);
          return;
        }
        ++RelCur;
      }
    };

    while (!HeadTail.first.empty()) {
      OS << Separator;
      Separator = "\n";
      if (SP && (PrintSource || PrintLines))
        SP->printSourceLine(OS, Address, ObjectFilename, LVP, "");
      printLead(Bytes, Address.Address, OS);
      OS << Preamble;
      Preamble = "   ";
      StringRef Inst;
      auto Duplex = HeadTail.first.split('\v');
      if (!Duplex.second.empty()) {
        OS << Duplex.first;
        OS << "; ";
        Inst = Duplex.second;
      }
      else
        Inst = HeadTail.first;
      OS << Inst;
      HeadTail = HeadTail.second.split('\n');
      if (HeadTail.first.empty())
        OS << " } " << PacketBundle.second;
      PrintReloc();
      Bytes = Bytes.slice(4);
      Address.Address += 4;
    }
  }
};
HexagonPrettyPrinter HexagonPrettyPrinterInst;

class AMDGCNPrettyPrinter : public PrettyPrinter {
public:
  void printInst(MCInstPrinter &IP, const MCInst *MI, ArrayRef<uint8_t> Bytes,
                 object::SectionedAddress Address, formatted_raw_ostream &OS,
                 StringRef Annot, MCSubtargetInfo const &STI, SourcePrinter *SP,
                 StringRef ObjectFilename, std::vector<RelocationRef> *Rels,
                 LiveVariablePrinter &LVP) override {
    if (SP && (PrintSource || PrintLines))
      SP->printSourceLine(OS, Address, ObjectFilename, LVP);

    if (MI) {
      SmallString<40> InstStr;
      raw_svector_ostream IS(InstStr);

      IP.printInst(MI, Address.Address, "", STI, IS);

      OS << left_justify(IS.str(), 60);
    } else {
      // an unrecognized encoding - this is probably data so represent it
      // using the .long directive, or .byte directive if fewer than 4 bytes
      // remaining
      if (Bytes.size() >= 4) {
        OS << format("\t.long 0x%08" PRIx32 " ",
                     support::endian::read32<support::little>(Bytes.data()));
        OS.indent(42);
      } else {
          OS << format("\t.byte 0x%02" PRIx8, Bytes[0]);
          for (unsigned int i = 1; i < Bytes.size(); i++)
            OS << format(", 0x%02" PRIx8, Bytes[i]);
          OS.indent(55 - (6 * Bytes.size()));
      }
    }

    OS << format("// %012" PRIX64 ":", Address.Address);
    if (Bytes.size() >= 4) {
      // D should be casted to uint32_t here as it is passed by format to
      // snprintf as vararg.
      for (uint32_t D : makeArrayRef(
               reinterpret_cast<const support::little32_t *>(Bytes.data()),
               Bytes.size() / 4))
        OS << format(" %08" PRIX32, D);
    } else {
      for (unsigned char B : Bytes)
        OS << format(" %02" PRIX8, B);
    }

    if (!Annot.empty())
      OS << " // " << Annot;
  }
};
AMDGCNPrettyPrinter AMDGCNPrettyPrinterInst;

class BPFPrettyPrinter : public PrettyPrinter {
public:
  void printInst(MCInstPrinter &IP, const MCInst *MI, ArrayRef<uint8_t> Bytes,
                 object::SectionedAddress Address, formatted_raw_ostream &OS,
                 StringRef Annot, MCSubtargetInfo const &STI, SourcePrinter *SP,
                 StringRef ObjectFilename, std::vector<RelocationRef> *Rels,
                 LiveVariablePrinter &LVP) override {
    if (SP && (PrintSource || PrintLines))
      SP->printSourceLine(OS, Address, ObjectFilename, LVP);
    if (LeadingAddr)
      OS << format("%8" PRId64 ":", Address.Address / 8);
    if (ShowRawInsn) {
      OS << "\t";
      dumpBytes(Bytes, OS);
    }
    if (MI)
      IP.printInst(MI, Address.Address, "", STI, OS);
    else
      OS << "\t<unknown>";
  }
};
BPFPrettyPrinter BPFPrettyPrinterInst;

class ARMPrettyPrinter : public PrettyPrinter {
public:
  void printInst(MCInstPrinter &IP, const MCInst *MI, ArrayRef<uint8_t> Bytes,
                 object::SectionedAddress Address, formatted_raw_ostream &OS,
                 StringRef Annot, MCSubtargetInfo const &STI, SourcePrinter *SP,
                 StringRef ObjectFilename, std::vector<RelocationRef> *Rels,
                 LiveVariablePrinter &LVP) override {
    if (SP && (PrintSource || PrintLines))
      SP->printSourceLine(OS, Address, ObjectFilename, LVP);
    LVP.printBetweenInsts(OS, false);

    size_t Start = OS.tell();
    if (LeadingAddr)
      OS << format("%8" PRIx64 ":", Address.Address);
    if (ShowRawInsn) {
      size_t Pos = 0, End = Bytes.size();
      if (STI.checkFeatures("+thumb-mode")) {
        for (; Pos + 2 <= End; Pos += 2)
          OS << ' '
             << format_hex_no_prefix(
                    llvm::support::endian::read<uint16_t>(
                        Bytes.data() + Pos, llvm::support::little),
                    4);
      } else {
        for (; Pos + 4 <= End; Pos += 4)
          OS << ' '
             << format_hex_no_prefix(
                    llvm::support::endian::read<uint32_t>(
                        Bytes.data() + Pos, llvm::support::little),
                    8);
      }
      if (Pos < End) {
        OS << ' ';
        dumpBytes(Bytes.slice(Pos), OS);
      }
    }

    AlignToInstStartColumn(Start, STI, OS);

    if (MI) {
      IP.printInst(MI, Address.Address, "", STI, OS);
    } else
      OS << "\t<unknown>";
  }
};
ARMPrettyPrinter ARMPrettyPrinterInst;

class AArch64PrettyPrinter : public PrettyPrinter {
public:
  void printInst(MCInstPrinter &IP, const MCInst *MI, ArrayRef<uint8_t> Bytes,
                 object::SectionedAddress Address, formatted_raw_ostream &OS,
                 StringRef Annot, MCSubtargetInfo const &STI, SourcePrinter *SP,
                 StringRef ObjectFilename, std::vector<RelocationRef> *Rels,
                 LiveVariablePrinter &LVP) override {
    if (SP && (PrintSource || PrintLines))
      SP->printSourceLine(OS, Address, ObjectFilename, LVP);
    LVP.printBetweenInsts(OS, false);

    size_t Start = OS.tell();
    if (LeadingAddr)
      OS << format("%8" PRIx64 ":", Address.Address);
    if (ShowRawInsn) {
      size_t Pos = 0, End = Bytes.size();
      for (; Pos + 4 <= End; Pos += 4)
        OS << ' '
           << format_hex_no_prefix(
                  llvm::support::endian::read<uint32_t>(Bytes.data() + Pos,
                                                        llvm::support::little),
                  8);
      if (Pos < End) {
        OS << ' ';
        dumpBytes(Bytes.slice(Pos), OS);
      }
    }

    AlignToInstStartColumn(Start, STI, OS);

    if (MI) {
      IP.printInst(MI, Address.Address, "", STI, OS);
    } else
      OS << "\t<unknown>";
  }
};
AArch64PrettyPrinter AArch64PrettyPrinterInst;

PrettyPrinter &selectPrettyPrinter(Triple const &Triple) {
  switch(Triple.getArch()) {
  default:
    return PrettyPrinterInst;
  case Triple::hexagon:
    return HexagonPrettyPrinterInst;
  case Triple::amdgcn:
    return AMDGCNPrettyPrinterInst;
  case Triple::bpfel:
  case Triple::bpfeb:
    return BPFPrettyPrinterInst;
  case Triple::arm:
  case Triple::armeb:
  case Triple::thumb:
  case Triple::thumbeb:
    return ARMPrettyPrinterInst;
  case Triple::aarch64:
  case Triple::aarch64_be:
  case Triple::aarch64_32:
    return AArch64PrettyPrinterInst;
  }
}
}

static uint8_t getElfSymbolType(const ObjectFile &Obj, const SymbolRef &Sym) {
  assert(Obj.isELF());
  if (auto *Elf32LEObj = dyn_cast<ELF32LEObjectFile>(&Obj))
    return unwrapOrError(Elf32LEObj->getSymbol(Sym.getRawDataRefImpl()),
                         Obj.getFileName())
        ->getType();
  if (auto *Elf64LEObj = dyn_cast<ELF64LEObjectFile>(&Obj))
    return unwrapOrError(Elf64LEObj->getSymbol(Sym.getRawDataRefImpl()),
                         Obj.getFileName())
        ->getType();
  if (auto *Elf32BEObj = dyn_cast<ELF32BEObjectFile>(&Obj))
    return unwrapOrError(Elf32BEObj->getSymbol(Sym.getRawDataRefImpl()),
                         Obj.getFileName())
        ->getType();
  if (auto *Elf64BEObj = cast<ELF64BEObjectFile>(&Obj))
    return unwrapOrError(Elf64BEObj->getSymbol(Sym.getRawDataRefImpl()),
                         Obj.getFileName())
        ->getType();
  llvm_unreachable("Unsupported binary format");
}

template <class ELFT>
static void
addDynamicElfSymbols(const ELFObjectFile<ELFT> &Obj,
                     std::map<SectionRef, SectionSymbolsTy> &AllSymbols) {
  for (auto Symbol : Obj.getDynamicSymbolIterators()) {
    uint8_t SymbolType = Symbol.getELFType();
    if (SymbolType == ELF::STT_SECTION)
      continue;

    uint64_t Address = unwrapOrError(Symbol.getAddress(), Obj.getFileName());
    // ELFSymbolRef::getAddress() returns size instead of value for common
    // symbols which is not desirable for disassembly output. Overriding.
    if (SymbolType == ELF::STT_COMMON)
      Address = unwrapOrError(Obj.getSymbol(Symbol.getRawDataRefImpl()),
                              Obj.getFileName())
                    ->st_value;

    StringRef Name = unwrapOrError(Symbol.getName(), Obj.getFileName());
    if (Name.empty())
      continue;

    section_iterator SecI =
        unwrapOrError(Symbol.getSection(), Obj.getFileName());
    if (SecI == Obj.section_end())
      continue;

    AllSymbols[*SecI].emplace_back(Address, Name, SymbolType);
  }
}

static void
addDynamicElfSymbols(const ELFObjectFileBase &Obj,
                     std::map<SectionRef, SectionSymbolsTy> &AllSymbols) {
  if (auto *Elf32LEObj = dyn_cast<ELF32LEObjectFile>(&Obj))
    addDynamicElfSymbols(*Elf32LEObj, AllSymbols);
  else if (auto *Elf64LEObj = dyn_cast<ELF64LEObjectFile>(&Obj))
    addDynamicElfSymbols(*Elf64LEObj, AllSymbols);
  else if (auto *Elf32BEObj = dyn_cast<ELF32BEObjectFile>(&Obj))
    addDynamicElfSymbols(*Elf32BEObj, AllSymbols);
  else if (auto *Elf64BEObj = cast<ELF64BEObjectFile>(&Obj))
    addDynamicElfSymbols(*Elf64BEObj, AllSymbols);
  else
    llvm_unreachable("Unsupported binary format");
}

static Optional<SectionRef> getWasmCodeSection(const WasmObjectFile &Obj) {
  for (auto SecI : Obj.sections()) {
    const WasmSection &Section = Obj.getWasmSection(SecI);
    if (Section.Type == wasm::WASM_SEC_CODE)
      return SecI;
  }
  return None;
}

static void
addMissingWasmCodeSymbols(const WasmObjectFile &Obj,
                          std::map<SectionRef, SectionSymbolsTy> &AllSymbols) {
  Optional<SectionRef> Section = getWasmCodeSection(Obj);
  if (!Section)
    return;
  SectionSymbolsTy &Symbols = AllSymbols[*Section];

  std::set<uint64_t> SymbolAddresses;
  for (const auto &Sym : Symbols)
    SymbolAddresses.insert(Sym.Addr);

  for (const wasm::WasmFunction &Function : Obj.functions()) {
    uint64_t Address = Function.CodeSectionOffset;
    // Only add fallback symbols for functions not already present in the symbol
    // table.
    if (SymbolAddresses.count(Address))
      continue;
    // This function has no symbol, so it should have no SymbolName.
    assert(Function.SymbolName.empty());
    // We use DebugName for the name, though it may be empty if there is no
    // "name" custom section, or that section is missing a name for this
    // function.
    StringRef Name = Function.DebugName;
    Symbols.emplace_back(Address, Name, ELF::STT_NOTYPE);
  }
}

static void addPltEntries(const ObjectFile &Obj,
                          std::map<SectionRef, SectionSymbolsTy> &AllSymbols,
                          StringSaver &Saver) {
  Optional<SectionRef> Plt = None;
  for (const SectionRef &Section : Obj.sections()) {
    Expected<StringRef> SecNameOrErr = Section.getName();
    if (!SecNameOrErr) {
      consumeError(SecNameOrErr.takeError());
      continue;
    }
    if (*SecNameOrErr == ".plt")
      Plt = Section;
  }
  if (!Plt)
    return;
  if (auto *ElfObj = dyn_cast<ELFObjectFileBase>(&Obj)) {
    for (auto PltEntry : ElfObj->getPltAddresses()) {
      if (PltEntry.first) {
        SymbolRef Symbol(*PltEntry.first, ElfObj);
        uint8_t SymbolType = getElfSymbolType(Obj, Symbol);
        if (Expected<StringRef> NameOrErr = Symbol.getName()) {
          if (!NameOrErr->empty())
            AllSymbols[*Plt].emplace_back(
                PltEntry.second, Saver.save((*NameOrErr + "@plt").str()),
                SymbolType);
          continue;
        } else {
          // The warning has been reported in disassembleObject().
          consumeError(NameOrErr.takeError());
        }
      }
      reportWarning("PLT entry at 0x" + Twine::utohexstr(PltEntry.second) +
                        " references an invalid symbol",
                    Obj.getFileName());
    }
  }
}

// Normally the disassembly output will skip blocks of zeroes. This function
// returns the number of zero bytes that can be skipped when dumping the
// disassembly of the instructions in Buf.
static size_t countSkippableZeroBytes(ArrayRef<uint8_t> Buf) {
  // Find the number of leading zeroes.
  size_t N = 0;
  while (N < Buf.size() && !Buf[N])
    ++N;

  // We may want to skip blocks of zero bytes, but unless we see
  // at least 8 of them in a row.
  if (N < 8)
    return 0;

  // We skip zeroes in multiples of 4 because do not want to truncate an
  // instruction if it starts with a zero byte.
  return N & ~0x3;
}

// Returns a map from sections to their relocations.
static std::map<SectionRef, std::vector<RelocationRef>>
getRelocsMap(object::ObjectFile const &Obj) {
  std::map<SectionRef, std::vector<RelocationRef>> Ret;
  uint64_t I = (uint64_t)-1;
  for (SectionRef Sec : Obj.sections()) {
    ++I;
    Expected<section_iterator> RelocatedOrErr = Sec.getRelocatedSection();
    if (!RelocatedOrErr)
      reportError(Obj.getFileName(),
                  "section (" + Twine(I) +
                      "): failed to get a relocated section: " +
                      toString(RelocatedOrErr.takeError()));

    section_iterator Relocated = *RelocatedOrErr;
    if (Relocated == Obj.section_end() || !checkSectionFilter(*Relocated).Keep)
      continue;
    std::vector<RelocationRef> &V = Ret[*Relocated];
    append_range(V, Sec.relocations());
    // Sort relocations by address.
    llvm::stable_sort(V, isRelocAddressLess);
  }
  return Ret;
}

// Used for --adjust-vma to check if address should be adjusted by the
// specified value for a given section.
// For ELF we do not adjust non-allocatable sections like debug ones,
// because they are not loadable.
// TODO: implement for other file formats.
static bool shouldAdjustVA(const SectionRef &Section) {
  const ObjectFile *Obj = Section.getObject();
  if (Obj->isELF())
    return ELFSectionRef(Section).getFlags() & ELF::SHF_ALLOC;
  return false;
}


typedef std::pair<uint64_t, char> MappingSymbolPair;
static char getMappingSymbolKind(ArrayRef<MappingSymbolPair> MappingSymbols,
                                 uint64_t Address) {
  auto It =
      partition_point(MappingSymbols, [Address](const MappingSymbolPair &Val) {
        return Val.first <= Address;
      });
  // Return zero for any address before the first mapping symbol; this means
  // we should use the default disassembly mode, depending on the target.
  if (It == MappingSymbols.begin())
    return '\x00';
  return (It - 1)->second;
}

static uint64_t dumpARMELFData(uint64_t SectionAddr, uint64_t Index,
                               uint64_t End, const ObjectFile &Obj,
                               ArrayRef<uint8_t> Bytes,
                               ArrayRef<MappingSymbolPair> MappingSymbols,
                               const MCSubtargetInfo &STI, raw_ostream &OS) {
  support::endianness Endian =
      Obj.isLittleEndian() ? support::little : support::big;
  size_t Start = OS.tell();
  OS << format("%8" PRIx64 ": ", SectionAddr + Index);
  if (Index + 4 <= End) {
    dumpBytes(Bytes.slice(Index, 4), OS);
    AlignToInstStartColumn(Start, STI, OS);
    OS << "\t.word\t"
           << format_hex(support::endian::read32(Bytes.data() + Index, Endian),
                         10);
    return 4;
  }
  if (Index + 2 <= End) {
    dumpBytes(Bytes.slice(Index, 2), OS);
    AlignToInstStartColumn(Start, STI, OS);
    OS << "\t.short\t"
       << format_hex(support::endian::read16(Bytes.data() + Index, Endian), 6);
    return 2;
  }
  dumpBytes(Bytes.slice(Index, 1), OS);
  AlignToInstStartColumn(Start, STI, OS);
  OS << "\t.byte\t" << format_hex(Bytes[Index], 4);
  return 1;
}

static void dumpELFData(uint64_t SectionAddr, uint64_t Index, uint64_t End,
                        ArrayRef<uint8_t> Bytes) {
  // print out data up to 8 bytes at a time in hex and ascii
  uint8_t AsciiData[9] = {'\0'};
  uint8_t Byte;
  int NumBytes = 0;

  for (; Index < End; ++Index) {
    if (NumBytes == 0)
      outs() << format("%8" PRIx64 ":", SectionAddr + Index);
    Byte = Bytes.slice(Index)[0];
    outs() << format(" %02x", Byte);
    AsciiData[NumBytes] = isPrint(Byte) ? Byte : '.';

    uint8_t IndentOffset = 0;
    NumBytes++;
    if (Index == End - 1 || NumBytes > 8) {
      // Indent the space for less than 8 bytes data.
      // 2 spaces for byte and one for space between bytes
      IndentOffset = 3 * (8 - NumBytes);
      for (int Excess = NumBytes; Excess < 8; Excess++)
        AsciiData[Excess] = '\0';
      NumBytes = 8;
    }
    if (NumBytes == 8) {
      AsciiData[8] = '\0';
      outs() << std::string(IndentOffset, ' ') << "         ";
      outs() << reinterpret_cast<char *>(AsciiData);
      outs() << '\n';
      NumBytes = 0;
    }
  }
}

SymbolInfoTy objdump::createSymbolInfo(const ObjectFile &Obj,
                                       const SymbolRef &Symbol) {
  const StringRef FileName = Obj.getFileName();
  const uint64_t Addr = unwrapOrError(Symbol.getAddress(), FileName);
  const StringRef Name = unwrapOrError(Symbol.getName(), FileName);

  if (Obj.isXCOFF() && SymbolDescription) {
    const auto &XCOFFObj = cast<XCOFFObjectFile>(Obj);
    DataRefImpl SymbolDRI = Symbol.getRawDataRefImpl();

    const uint32_t SymbolIndex = XCOFFObj.getSymbolIndex(SymbolDRI.p);
    Optional<XCOFF::StorageMappingClass> Smc =
        getXCOFFSymbolCsectSMC(XCOFFObj, Symbol);
    return SymbolInfoTy(Addr, Name, Smc, SymbolIndex,
                        isLabel(XCOFFObj, Symbol));
  } else if (Obj.isXCOFF()) {
    const SymbolRef::Type SymType = unwrapOrError(Symbol.getType(), FileName);
    return SymbolInfoTy(Addr, Name, SymType, true);
  } else
    return SymbolInfoTy(Addr, Name,
                        Obj.isELF() ? getElfSymbolType(Obj, Symbol)
                                    : (uint8_t)ELF::STT_NOTYPE);
}

static SymbolInfoTy createDummySymbolInfo(const ObjectFile &Obj,
                                          const uint64_t Addr, StringRef &Name,
                                          uint8_t Type) {
  if (Obj.isXCOFF() && SymbolDescription)
    return SymbolInfoTy(Addr, Name, None, None, false);
  else
    return SymbolInfoTy(Addr, Name, Type);
}

static void
collectBBAddrMapLabels(const std::unordered_map<uint64_t, BBAddrMap> &AddrToBBAddrMap,
                       uint64_t SectionAddr, uint64_t Start, uint64_t End,
                       std::unordered_map<uint64_t, std::vector<std::string>> &Labels) {
  if (AddrToBBAddrMap.empty())
    return;
  Labels.clear();
  uint64_t StartAddress = SectionAddr + Start;
  uint64_t EndAddress = SectionAddr + End;
  auto Iter = AddrToBBAddrMap.find(StartAddress);
  if (Iter == AddrToBBAddrMap.end())
    return;
  for (unsigned I = 0, Size = Iter->second.BBEntries.size(); I < Size; ++I) {
    uint64_t BBAddress = Iter->second.BBEntries[I].Offset + Iter->second.Addr;
    if (BBAddress >= EndAddress)
      continue;
    Labels[BBAddress].push_back(("BB" + Twine(I)).str());
  }
}

static void collectLocalBranchTargets(
    ArrayRef<uint8_t> Bytes, const MCInstrAnalysis *MIA, MCDisassembler *DisAsm,
    MCInstPrinter *IP, const MCSubtargetInfo *STI, uint64_t SectionAddr,
    uint64_t Start, uint64_t End, std::unordered_map<uint64_t, std::string> &Labels) {
  // So far only supports PowerPC and X86.
  if (!STI->getTargetTriple().isPPC() && !STI->getTargetTriple().isX86())
    return;

  Labels.clear();
  unsigned LabelCount = 0;
  Start += SectionAddr;
  End += SectionAddr;
  uint64_t Index = Start;
  while (Index < End) {
    // Disassemble a real instruction and record function-local branch labels.
    MCInst Inst;
    uint64_t Size;
    ArrayRef<uint8_t> ThisBytes = Bytes.slice(Index - SectionAddr);
    bool Disassembled =
        DisAsm->getInstruction(Inst, Size, ThisBytes, Index, nulls());
    if (Size == 0)
      Size = std::min(ThisBytes.size(),
                      DisAsm->suggestBytesToSkip(ThisBytes, Index));

    if (Disassembled && MIA) {
      uint64_t Target;
      bool TargetKnown = MIA->evaluateBranch(Inst, Index, Size, Target);
      // On PowerPC, if the address of a branch is the same as the target, it
      // means that it's a function call. Do not mark the label for this case.
      if (TargetKnown && (Target >= Start && Target < End) &&
          !Labels.count(Target) &&
          !(STI->getTargetTriple().isPPC() && Target == Index))
        Labels[Target] = ("L" + Twine(LabelCount++)).str();
    }
    Index += Size;
  }
}

// Create an MCSymbolizer for the target and add it to the MCDisassembler.
// This is currently only used on AMDGPU, and assumes the format of the
// void * argument passed to AMDGPU's createMCSymbolizer.
static void addSymbolizer(
    MCContext &Ctx, const Target *Target, StringRef TripleName,
    MCDisassembler *DisAsm, uint64_t SectionAddr, ArrayRef<uint8_t> Bytes,
    SectionSymbolsTy &Symbols,
    std::vector<std::unique_ptr<std::string>> &SynthesizedLabelNames) {

  std::unique_ptr<MCRelocationInfo> RelInfo(
      Target->createMCRelocationInfo(TripleName, Ctx));
  if (!RelInfo)
    return;
  std::unique_ptr<MCSymbolizer> Symbolizer(Target->createMCSymbolizer(
      TripleName, nullptr, nullptr, &Symbols, &Ctx, std::move(RelInfo)));
  MCSymbolizer *SymbolizerPtr = &*Symbolizer;
  DisAsm->setSymbolizer(std::move(Symbolizer));

  if (!SymbolizeOperands)
    return;

  // Synthesize labels referenced by branch instructions by
  // disassembling, discarding the output, and collecting the referenced
  // addresses from the symbolizer.
  for (size_t Index = 0; Index != Bytes.size();) {
    MCInst Inst;
    uint64_t Size;
    ArrayRef<uint8_t> ThisBytes = Bytes.slice(Index - SectionAddr);
    DisAsm->getInstruction(Inst, Size, ThisBytes, Index, nulls());
    if (Size == 0)
      Size = std::min(ThisBytes.size(),
                      DisAsm->suggestBytesToSkip(ThisBytes, Index));
    Index += Size;
  }
  ArrayRef<uint64_t> LabelAddrsRef = SymbolizerPtr->getReferencedAddresses();
  // Copy and sort to remove duplicates.
  std::vector<uint64_t> LabelAddrs;
  LabelAddrs.insert(LabelAddrs.end(), LabelAddrsRef.begin(),
                    LabelAddrsRef.end());
  llvm::sort(LabelAddrs);
  LabelAddrs.resize(std::unique(LabelAddrs.begin(), LabelAddrs.end()) -
                    LabelAddrs.begin());
  // Add the labels.
  for (unsigned LabelNum = 0; LabelNum != LabelAddrs.size(); ++LabelNum) {
    auto Name = std::make_unique<std::string>();
    *Name = (Twine("L") + Twine(LabelNum)).str();
    SynthesizedLabelNames.push_back(std::move(Name));
    Symbols.push_back(SymbolInfoTy(
        LabelAddrs[LabelNum], *SynthesizedLabelNames.back(), ELF::STT_NOTYPE));
  }
  llvm::stable_sort(Symbols);
  // Recreate the symbolizer with the new symbols list.
  RelInfo.reset(Target->createMCRelocationInfo(TripleName, Ctx));
  Symbolizer.reset(Target->createMCSymbolizer(
      TripleName, nullptr, nullptr, &Symbols, &Ctx, std::move(RelInfo)));
  DisAsm->setSymbolizer(std::move(Symbolizer));
}

static StringRef getSegmentName(const MachOObjectFile *MachO,
                                const SectionRef &Section) {
  if (MachO) {
    DataRefImpl DR = Section.getRawDataRefImpl();
    StringRef SegmentName = MachO->getSectionFinalSegmentName(DR);
    return SegmentName;
  }
  return "";
}

static void emitPostInstructionInfo(formatted_raw_ostream &FOS,
                                    const MCAsmInfo &MAI,
                                    const MCSubtargetInfo &STI,
                                    StringRef Comments,
                                    LiveVariablePrinter &LVP) {
  do {
    if (!Comments.empty()) {
      // Emit a line of comments.
      StringRef Comment;
      std::tie(Comment, Comments) = Comments.split('\n');
      // MAI.getCommentColumn() assumes that instructions are printed at the
      // position of 8, while getInstStartColumn() returns the actual position.
      unsigned CommentColumn =
          MAI.getCommentColumn() - 8 + getInstStartColumn(STI);
      FOS.PadToColumn(CommentColumn);
      FOS << MAI.getCommentString() << ' ' << Comment;
    }
    LVP.printAfterInst(FOS);
    FOS << '\n';
  } while (!Comments.empty());
  FOS.flush();
}

static void createFakeELFSections(ObjectFile &Obj) {
  assert(Obj.isELF());
  if (auto *Elf32LEObj = dyn_cast<ELF32LEObjectFile>(&Obj))
    Elf32LEObj->createFakeSections();
  else if (auto *Elf64LEObj = dyn_cast<ELF64LEObjectFile>(&Obj))
    Elf64LEObj->createFakeSections();
  else if (auto *Elf32BEObj = dyn_cast<ELF32BEObjectFile>(&Obj))
    Elf32BEObj->createFakeSections();
  else if (auto *Elf64BEObj = cast<ELF64BEObjectFile>(&Obj))
    Elf64BEObj->createFakeSections();
  else
    llvm_unreachable("Unsupported binary format");
}

static void disassembleObject(const Target *TheTarget, ObjectFile &Obj,
                              MCContext &Ctx, MCDisassembler *PrimaryDisAsm,
                              MCDisassembler *SecondaryDisAsm,
                              const MCInstrAnalysis *MIA, MCInstPrinter *IP,
                              const MCSubtargetInfo *PrimarySTI,
                              const MCSubtargetInfo *SecondarySTI,
                              PrettyPrinter &PIP, SourcePrinter &SP,
                              bool InlineRelocs) {
  const MCSubtargetInfo *STI = PrimarySTI;
  MCDisassembler *DisAsm = PrimaryDisAsm;
  bool PrimaryIsThumb = false;
  if (isArmElf(Obj))
    PrimaryIsThumb = STI->checkFeatures("+thumb-mode");

  std::map<SectionRef, std::vector<RelocationRef>> RelocMap;
  if (InlineRelocs)
    RelocMap = getRelocsMap(Obj);
  bool Is64Bits = Obj.getBytesInAddress() > 4;

  // Create a mapping from virtual address to symbol name.  This is used to
  // pretty print the symbols while disassembling.
  std::map<SectionRef, SectionSymbolsTy> AllSymbols;
  SectionSymbolsTy AbsoluteSymbols;
  const StringRef FileName = Obj.getFileName();
  const MachOObjectFile *MachO = dyn_cast<const MachOObjectFile>(&Obj);
  for (const SymbolRef &Symbol : Obj.symbols()) {
    Expected<StringRef> NameOrErr = Symbol.getName();
    if (!NameOrErr) {
      reportWarning(toString(NameOrErr.takeError()), FileName);
      continue;
    }
    if (NameOrErr->empty() && !(Obj.isXCOFF() && SymbolDescription))
      continue;

    if (Obj.isELF() && getElfSymbolType(Obj, Symbol) == ELF::STT_SECTION)
      continue;

    if (MachO) {
      // __mh_(execute|dylib|dylinker|bundle|preload|object)_header are special
      // symbols that support MachO header introspection. They do not bind to
      // code locations and are irrelevant for disassembly.
      if (NameOrErr->startswith("__mh_") && NameOrErr->endswith("_header"))
        continue;
      // Don't ask a Mach-O STAB symbol for its section unless you know that
      // STAB symbol's section field refers to a valid section index. Otherwise
      // the symbol may error trying to load a section that does not exist.
      DataRefImpl SymDRI = Symbol.getRawDataRefImpl();
      uint8_t NType = (MachO->is64Bit() ?
                       MachO->getSymbol64TableEntry(SymDRI).n_type:
                       MachO->getSymbolTableEntry(SymDRI).n_type);
      if (NType & MachO::N_STAB)
        continue;
    }

    section_iterator SecI = unwrapOrError(Symbol.getSection(), FileName);
    if (SecI != Obj.section_end())
      AllSymbols[*SecI].push_back(createSymbolInfo(Obj, Symbol));
    else
      AbsoluteSymbols.push_back(createSymbolInfo(Obj, Symbol));
  }

  if (AllSymbols.empty() && Obj.isELF())
    addDynamicElfSymbols(cast<ELFObjectFileBase>(Obj), AllSymbols);

  if (Obj.isWasm())
    addMissingWasmCodeSymbols(cast<WasmObjectFile>(Obj), AllSymbols);

  if (Obj.isELF() && Obj.sections().empty())
    createFakeELFSections(Obj);

  BumpPtrAllocator A;
  StringSaver Saver(A);
  addPltEntries(Obj, AllSymbols, Saver);

  // Create a mapping from virtual address to section. An empty section can
  // cause more than one section at the same address. Sort such sections to be
  // before same-addressed non-empty sections so that symbol lookups prefer the
  // non-empty section.
  std::vector<std::pair<uint64_t, SectionRef>> SectionAddresses;
  for (SectionRef Sec : Obj.sections())
    SectionAddresses.emplace_back(Sec.getAddress(), Sec);
  llvm::stable_sort(SectionAddresses, [](const auto &LHS, const auto &RHS) {
    if (LHS.first != RHS.first)
      return LHS.first < RHS.first;
    return LHS.second.getSize() < RHS.second.getSize();
  });

  // Linked executables (.exe and .dll files) typically don't include a real
  // symbol table but they might contain an export table.
  if (const auto *COFFObj = dyn_cast<COFFObjectFile>(&Obj)) {
    for (const auto &ExportEntry : COFFObj->export_directories()) {
      StringRef Name;
      if (Error E = ExportEntry.getSymbolName(Name))
        reportError(std::move(E), Obj.getFileName());
      if (Name.empty())
        continue;

      uint32_t RVA;
      if (Error E = ExportEntry.getExportRVA(RVA))
        reportError(std::move(E), Obj.getFileName());

      uint64_t VA = COFFObj->getImageBase() + RVA;
      auto Sec = partition_point(
          SectionAddresses, [VA](const std::pair<uint64_t, SectionRef> &O) {
            return O.first <= VA;
          });
      if (Sec != SectionAddresses.begin()) {
        --Sec;
        AllSymbols[Sec->second].emplace_back(VA, Name, ELF::STT_NOTYPE);
      } else
        AbsoluteSymbols.emplace_back(VA, Name, ELF::STT_NOTYPE);
    }
  }

  // Sort all the symbols, this allows us to use a simple binary search to find
  // Multiple symbols can have the same address. Use a stable sort to stabilize
  // the output.
  StringSet<> FoundDisasmSymbolSet;
  for (std::pair<const SectionRef, SectionSymbolsTy> &SecSyms : AllSymbols)
    llvm::stable_sort(SecSyms.second);
  llvm::stable_sort(AbsoluteSymbols);

  std::unique_ptr<DWARFContext> DICtx;
  LiveVariablePrinter LVP(*Ctx.getRegisterInfo(), *STI);

  if (DbgVariables != DVDisabled) {
    DICtx = DWARFContext::create(Obj);
    for (const std::unique_ptr<DWARFUnit> &CU : DICtx->compile_units())
      LVP.addCompileUnit(CU->getUnitDIE(false));
  }

  LLVM_DEBUG(LVP.dump());

  std::unordered_map<uint64_t, BBAddrMap> AddrToBBAddrMap;
  auto ReadBBAddrMap = [&](Optional<unsigned> SectionIndex = None) {
    AddrToBBAddrMap.clear();
    if (const auto *Elf = dyn_cast<ELFObjectFileBase>(&Obj)) {
      auto BBAddrMapsOrErr = Elf->readBBAddrMap(SectionIndex);
      if (!BBAddrMapsOrErr)
          reportWarning(toString(BBAddrMapsOrErr.takeError()),
                        Obj.getFileName());
      for (auto &FunctionBBAddrMap : *BBAddrMapsOrErr)
        AddrToBBAddrMap.emplace(FunctionBBAddrMap.Addr,
                                std::move(FunctionBBAddrMap));
    }
  };

  // For non-relocatable objects, Read all LLVM_BB_ADDR_MAP sections into a
  // single mapping, since they don't have any conflicts.
  if (SymbolizeOperands && !Obj.isRelocatableObject())
    ReadBBAddrMap();

  for (const SectionRef &Section : ToolSectionFilter(Obj)) {
    if (FilterSections.empty() && !DisassembleAll &&
        (!Section.isText() || Section.isVirtual()))
      continue;

    uint64_t SectionAddr = Section.getAddress();
    uint64_t SectSize = Section.getSize();
    if (!SectSize)
      continue;

    // For relocatable object files, read the LLVM_BB_ADDR_MAP section
    // corresponding to this section, if present.
    if (SymbolizeOperands && Obj.isRelocatableObject())
      ReadBBAddrMap(Section.getIndex());

    // Get the list of all the symbols in this section.
    SectionSymbolsTy &Symbols = AllSymbols[Section];
    std::vector<MappingSymbolPair> MappingSymbols;
    if (hasMappingSymbols(Obj)) {
      for (const auto &Symb : Symbols) {
        uint64_t Address = Symb.Addr;
        StringRef Name = Symb.Name;
        if (Name.startswith("$d"))
          MappingSymbols.emplace_back(Address - SectionAddr, 'd');
        if (Name.startswith("$x"))
          MappingSymbols.emplace_back(Address - SectionAddr, 'x');
        if (Name.startswith("$a"))
          MappingSymbols.emplace_back(Address - SectionAddr, 'a');
        if (Name.startswith("$t"))
          MappingSymbols.emplace_back(Address - SectionAddr, 't');
      }
    }

    llvm::sort(MappingSymbols);

    ArrayRef<uint8_t> Bytes = arrayRefFromStringRef(
        unwrapOrError(Section.getContents(), Obj.getFileName()));

    std::vector<std::unique_ptr<std::string>> SynthesizedLabelNames;
    if (Obj.isELF() && Obj.getArch() == Triple::amdgcn) {
      // AMDGPU disassembler uses symbolizer for printing labels
      addSymbolizer(Ctx, TheTarget, TripleName, DisAsm, SectionAddr, Bytes,
                    Symbols, SynthesizedLabelNames);
    }

    StringRef SegmentName = getSegmentName(MachO, Section);
    StringRef SectionName = unwrapOrError(Section.getName(), Obj.getFileName());
    // If the section has no symbol at the start, just insert a dummy one.
    if (Symbols.empty() || Symbols[0].Addr != 0) {
      Symbols.insert(Symbols.begin(),
                     createDummySymbolInfo(Obj, SectionAddr, SectionName,
                                           Section.isText() ? ELF::STT_FUNC
                                                            : ELF::STT_OBJECT));
    }

    SmallString<40> Comments;
    raw_svector_ostream CommentStream(Comments);

    uint64_t VMAAdjustment = 0;
    if (shouldAdjustVA(Section))
      VMAAdjustment = AdjustVMA;

    // In executable and shared objects, r_offset holds a virtual address.
    // Subtract SectionAddr from the r_offset field of a relocation to get
    // the section offset.
    uint64_t RelAdjustment = Obj.isRelocatableObject() ? 0 : SectionAddr;
    uint64_t Size;
    uint64_t Index;
    bool PrintedSection = false;
    std::vector<RelocationRef> Rels = RelocMap[Section];
    std::vector<RelocationRef>::const_iterator RelCur = Rels.begin();
    std::vector<RelocationRef>::const_iterator RelEnd = Rels.end();
    // Disassemble symbol by symbol.
    for (unsigned SI = 0, SE = Symbols.size(); SI != SE; ++SI) {
      std::string SymbolName = Symbols[SI].Name.str();
      if (Demangle)
        SymbolName = demangle(SymbolName);

      // Skip if --disassemble-symbols is not empty and the symbol is not in
      // the list.
      if (!DisasmSymbolSet.empty() && !DisasmSymbolSet.count(SymbolName))
        continue;

      uint64_t Start = Symbols[SI].Addr;
      if (Start < SectionAddr || StopAddress <= Start)
        continue;
      else
        FoundDisasmSymbolSet.insert(SymbolName);

      // The end is the section end, the beginning of the next symbol, or
      // --stop-address.
      uint64_t End = std::min<uint64_t>(SectionAddr + SectSize, StopAddress);
      if (SI + 1 < SE)
        End = std::min(End, Symbols[SI + 1].Addr);
      if (Start >= End || End <= StartAddress)
        continue;
      Start -= SectionAddr;
      End -= SectionAddr;

      if (!PrintedSection) {
        PrintedSection = true;
        outs() << "\nDisassembly of section ";
        if (!SegmentName.empty())
          outs() << SegmentName << ",";
        outs() << SectionName << ":\n";
      }

      outs() << '\n';
      if (LeadingAddr)
        outs() << format(Is64Bits ? "%016" PRIx64 " " : "%08" PRIx64 " ",
                         SectionAddr + Start + VMAAdjustment);
      if (Obj.isXCOFF() && SymbolDescription) {
        outs() << getXCOFFSymbolDescription(Symbols[SI], SymbolName) << ":\n";
      } else
        outs() << '<' << SymbolName << ">:\n";

      // Don't print raw contents of a virtual section. A virtual section
      // doesn't have any contents in the file.
      if (Section.isVirtual()) {
        outs() << "...\n";
        continue;
      }

      auto Status = DisAsm->onSymbolStart(Symbols[SI], Size,
                                          Bytes.slice(Start, End - Start),
                                          SectionAddr + Start, CommentStream);
      // To have round trippable disassembly, we fall back to decoding the
      // remaining bytes as instructions.
      //
      // If there is a failure, we disassemble the failed region as bytes before
      // falling back. The target is expected to print nothing in this case.
      //
      // If there is Success or SoftFail i.e no 'real' failure, we go ahead by
      // Size bytes before falling back.
      // So if the entire symbol is 'eaten' by the target:
      //   Start += Size  // Now Start = End and we will never decode as
      //                  // instructions
      //
      // Right now, most targets return None i.e ignore to treat a symbol
      // separately. But WebAssembly decodes preludes for some symbols.
      //
      if (Status) {
        if (Status.value() == MCDisassembler::Fail) {
          outs() << "// Error in decoding " << SymbolName
                 << " : Decoding failed region as bytes.\n";
          for (uint64_t I = 0; I < Size; ++I) {
            outs() << "\t.byte\t " << format_hex(Bytes[I], 1, /*Upper=*/true)
                   << "\n";
          }
        }
      } else {
        Size = 0;
      }

      Start += Size;

      Index = Start;
      if (SectionAddr < StartAddress)
        Index = std::max<uint64_t>(Index, StartAddress - SectionAddr);

      // If there is a data/common symbol inside an ELF text section and we are
      // only disassembling text (applicable all architectures), we are in a
      // situation where we must print the data and not disassemble it.
      if (Obj.isELF() && !DisassembleAll && Section.isText()) {
        uint8_t SymTy = Symbols[SI].Type;
        if (SymTy == ELF::STT_OBJECT || SymTy == ELF::STT_COMMON) {
          dumpELFData(SectionAddr, Index, End, Bytes);
          Index = End;
        }
      }

      bool CheckARMELFData = hasMappingSymbols(Obj) &&
                             Symbols[SI].Type != ELF::STT_OBJECT &&
                             !DisassembleAll;
      bool DumpARMELFData = false;
      formatted_raw_ostream FOS(outs());

      std::unordered_map<uint64_t, std::string> AllLabels;
      std::unordered_map<uint64_t, std::vector<std::string>> BBAddrMapLabels;
      if (SymbolizeOperands) {
        collectLocalBranchTargets(Bytes, MIA, DisAsm, IP, PrimarySTI,
                                  SectionAddr, Index, End, AllLabels);
        collectBBAddrMapLabels(AddrToBBAddrMap, SectionAddr, Index, End,
                               BBAddrMapLabels);
      }

      while (Index < End) {
        // ARM and AArch64 ELF binaries can interleave data and text in the
        // same section. We rely on the markers introduced to understand what
        // we need to dump. If the data marker is within a function, it is
        // denoted as a word/short etc.
        if (CheckARMELFData) {
          char Kind = getMappingSymbolKind(MappingSymbols, Index);
          DumpARMELFData = Kind == 'd';
          if (SecondarySTI) {
            if (Kind == 'a') {
              STI = PrimaryIsThumb ? SecondarySTI : PrimarySTI;
              DisAsm = PrimaryIsThumb ? SecondaryDisAsm : PrimaryDisAsm;
            } else if (Kind == 't') {
              STI = PrimaryIsThumb ? PrimarySTI : SecondarySTI;
              DisAsm = PrimaryIsThumb ? PrimaryDisAsm : SecondaryDisAsm;
            }
          }
        }

        if (DumpARMELFData) {
          Size = dumpARMELFData(SectionAddr, Index, End, Obj, Bytes,
                                MappingSymbols, *STI, FOS);
        } else {
          // When -z or --disassemble-zeroes are given we always dissasemble
          // them. Otherwise we might want to skip zero bytes we see.
          if (!DisassembleZeroes) {
            uint64_t MaxOffset = End - Index;
            // For --reloc: print zero blocks patched by relocations, so that
            // relocations can be shown in the dump.
            if (RelCur != RelEnd)
              MaxOffset = std::min(RelCur->getOffset() - RelAdjustment - Index,
                                   MaxOffset);

            if (size_t N =
                    countSkippableZeroBytes(Bytes.slice(Index, MaxOffset))) {
              FOS << "\t\t..." << '\n';
              Index += N;
              continue;
            }
          }

          // Print local label if there's any.
          auto Iter1 = BBAddrMapLabels.find(SectionAddr + Index);
          if (Iter1 != BBAddrMapLabels.end()) {
            for (StringRef Label : Iter1->second)
              FOS << "<" << Label << ">:\n";
          } else {
            auto Iter2 = AllLabels.find(SectionAddr + Index);
            if (Iter2 != AllLabels.end())
              FOS << "<" << Iter2->second << ">:\n";
          }

          // Disassemble a real instruction or a data when disassemble all is
          // provided
          MCInst Inst;
          ArrayRef<uint8_t> ThisBytes = Bytes.slice(Index);
          uint64_t ThisAddr = SectionAddr + Index;
          bool Disassembled = DisAsm->getInstruction(Inst, Size, ThisBytes,
                                                     ThisAddr, CommentStream);
          if (Size == 0)
            Size = std::min(ThisBytes.size(),
                            DisAsm->suggestBytesToSkip(ThisBytes, ThisAddr));

          LVP.update({Index, Section.getIndex()},
                     {Index + Size, Section.getIndex()}, Index + Size != End);

          IP->setCommentStream(CommentStream);

          PIP.printInst(
              *IP, Disassembled ? &Inst : nullptr, Bytes.slice(Index, Size),
              {SectionAddr + Index + VMAAdjustment, Section.getIndex()}, FOS,
              "", *STI, &SP, Obj.getFileName(), &Rels, LVP);

          IP->setCommentStream(llvm::nulls());

          // If disassembly has failed, avoid analysing invalid/incomplete
          // instruction information. Otherwise, try to resolve the target
          // address (jump target or memory operand address) and print it on the
          // right of the instruction.
          if (Disassembled && MIA) {
            // Branch targets are printed just after the instructions.
            llvm::raw_ostream *TargetOS = &FOS;
            uint64_t Target;
            bool PrintTarget =
                MIA->evaluateBranch(Inst, SectionAddr + Index, Size, Target);
            if (!PrintTarget)
              if (Optional<uint64_t> MaybeTarget =
                      MIA->evaluateMemoryOperandAddress(
                          Inst, STI, SectionAddr + Index, Size)) {
                Target = *MaybeTarget;
                PrintTarget = true;
                // Do not print real address when symbolizing.
                if (!SymbolizeOperands) {
                  // Memory operand addresses are printed as comments.
                  TargetOS = &CommentStream;
                  *TargetOS << "0x" << Twine::utohexstr(Target);
                }
              }
            if (PrintTarget) {
              // In a relocatable object, the target's section must reside in
              // the same section as the call instruction or it is accessed
              // through a relocation.
              //
              // In a non-relocatable object, the target may be in any section.
              // In that case, locate the section(s) containing the target
              // address and find the symbol in one of those, if possible.
              //
              // N.B. We don't walk the relocations in the relocatable case yet.
              std::vector<const SectionSymbolsTy *> TargetSectionSymbols;
              if (!Obj.isRelocatableObject()) {
                auto It = llvm::partition_point(
                    SectionAddresses,
                    [=](const std::pair<uint64_t, SectionRef> &O) {
                      return O.first <= Target;
                    });
                uint64_t TargetSecAddr = 0;
                while (It != SectionAddresses.begin()) {
                  --It;
                  if (TargetSecAddr == 0)
                    TargetSecAddr = It->first;
                  if (It->first != TargetSecAddr)
                    break;
                  TargetSectionSymbols.push_back(&AllSymbols[It->second]);
                }
              } else {
                TargetSectionSymbols.push_back(&Symbols);
              }
              TargetSectionSymbols.push_back(&AbsoluteSymbols);

              // Find the last symbol in the first candidate section whose
              // offset is less than or equal to the target. If there are no
              // such symbols, try in the next section and so on, before finally
              // using the nearest preceding absolute symbol (if any), if there
              // are no other valid symbols.
              const SymbolInfoTy *TargetSym = nullptr;
              for (const SectionSymbolsTy *TargetSymbols :
                   TargetSectionSymbols) {
                auto It = llvm::partition_point(
                    *TargetSymbols,
                    [=](const SymbolInfoTy &O) { return O.Addr <= Target; });
                if (It != TargetSymbols->begin()) {
                  TargetSym = &*(It - 1);
                  break;
                }
              }

              // Print the labels corresponding to the target if there's any.
              bool BBAddrMapLabelAvailable = BBAddrMapLabels.count(Target);
              bool LabelAvailable = AllLabels.count(Target);
              if (TargetSym != nullptr) {
                uint64_t TargetAddress = TargetSym->Addr;
                uint64_t Disp = Target - TargetAddress;
                std::string TargetName = TargetSym->Name.str();
                if (Demangle)
                  TargetName = demangle(TargetName);

                *TargetOS << " <";
                if (!Disp) {
                  // Always Print the binary symbol precisely corresponding to
                  // the target address.
                  *TargetOS << TargetName;
                } else if (BBAddrMapLabelAvailable) {
                  *TargetOS << BBAddrMapLabels[Target].front();
                } else if (LabelAvailable) {
                  *TargetOS << AllLabels[Target];
                } else {
                  // Always Print the binary symbol plus an offset if there's no
                  // local label corresponding to the target address.
                  *TargetOS << TargetName << "+0x" << Twine::utohexstr(Disp);
                }
                *TargetOS << ">";
              } else if (BBAddrMapLabelAvailable) {
                *TargetOS << " <" << BBAddrMapLabels[Target].front() << ">";
              } else if (LabelAvailable) {
                *TargetOS << " <" << AllLabels[Target] << ">";
              }
              // By convention, each record in the comment stream should be
              // terminated.
              if (TargetOS == &CommentStream)
                *TargetOS << "\n";
            }
          }
        }

        assert(Ctx.getAsmInfo());
        emitPostInstructionInfo(FOS, *Ctx.getAsmInfo(), *STI,
                                CommentStream.str(), LVP);
        Comments.clear();

        // Hexagon does this in pretty printer
        if (Obj.getArch() != Triple::hexagon) {
          // Print relocation for instruction and data.
          while (RelCur != RelEnd) {
            uint64_t Offset = RelCur->getOffset() - RelAdjustment;
            // If this relocation is hidden, skip it.
            if (getHidden(*RelCur) || SectionAddr + Offset < StartAddress) {
              ++RelCur;
              continue;
            }

            // Stop when RelCur's offset is past the disassembled
            // instruction/data. Note that it's possible the disassembled data
            // is not the complete data: we might see the relocation printed in
            // the middle of the data, but this matches the binutils objdump
            // output.
            if (Offset >= Index + Size)
              break;

            // When --adjust-vma is used, update the address printed.
            if (RelCur->getSymbol() != Obj.symbol_end()) {
              Expected<section_iterator> SymSI =
                  RelCur->getSymbol()->getSection();
              if (SymSI && *SymSI != Obj.section_end() &&
                  shouldAdjustVA(**SymSI))
                Offset += AdjustVMA;
            }

            printRelocation(FOS, Obj.getFileName(), *RelCur,
                            SectionAddr + Offset, Is64Bits);
            LVP.printAfterOtherLine(FOS, true);
            ++RelCur;
          }
        }

        Index += Size;
      }
    }
  }
  StringSet<> MissingDisasmSymbolSet =
      set_difference(DisasmSymbolSet, FoundDisasmSymbolSet);
  for (StringRef Sym : MissingDisasmSymbolSet.keys())
    reportWarning("failed to disassemble missing symbol " + Sym, FileName);
}

static void disassembleObject(ObjectFile *Obj, bool InlineRelocs) {
  const Target *TheTarget = getTarget(Obj);

  // Package up features to be passed to target/subtarget
  SubtargetFeatures Features = Obj->getFeatures();
  if (!MAttrs.empty()) {
    for (unsigned I = 0; I != MAttrs.size(); ++I)
      Features.AddFeature(MAttrs[I]);
  } else if (MCPU.empty() && Obj->getArch() == llvm::Triple::aarch64) {
    Features.AddFeature("+all");
  }

  std::unique_ptr<const MCRegisterInfo> MRI(
      TheTarget->createMCRegInfo(TripleName));
  if (!MRI)
    reportError(Obj->getFileName(),
                "no register info for target " + TripleName);

  // Set up disassembler.
  MCTargetOptions MCOptions;
  std::unique_ptr<const MCAsmInfo> AsmInfo(
      TheTarget->createMCAsmInfo(*MRI, TripleName, MCOptions));
  if (!AsmInfo)
    reportError(Obj->getFileName(),
                "no assembly info for target " + TripleName);

  if (MCPU.empty())
    MCPU = Obj->tryGetCPUName().value_or("").str();

  std::unique_ptr<const MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TripleName, MCPU, Features.getString()));
  if (!STI)
    reportError(Obj->getFileName(),
                "no subtarget info for target " + TripleName);
  std::unique_ptr<const MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  if (!MII)
    reportError(Obj->getFileName(),
                "no instruction info for target " + TripleName);
  MCContext Ctx(Triple(TripleName), AsmInfo.get(), MRI.get(), STI.get());
  // FIXME: for now initialize MCObjectFileInfo with default values
  std::unique_ptr<MCObjectFileInfo> MOFI(
      TheTarget->createMCObjectFileInfo(Ctx, /*PIC=*/false));
  Ctx.setObjectFileInfo(MOFI.get());

  std::unique_ptr<MCDisassembler> DisAsm(
      TheTarget->createMCDisassembler(*STI, Ctx));
  if (!DisAsm)
    reportError(Obj->getFileName(), "no disassembler for target " + TripleName);

  // If we have an ARM object file, we need a second disassembler, because
  // ARM CPUs have two different instruction sets: ARM mode, and Thumb mode.
  // We use mapping symbols to switch between the two assemblers, where
  // appropriate.
  std::unique_ptr<MCDisassembler> SecondaryDisAsm;
  std::unique_ptr<const MCSubtargetInfo> SecondarySTI;
  if (isArmElf(*Obj) && !STI->checkFeatures("+mclass")) {
    if (STI->checkFeatures("+thumb-mode"))
      Features.AddFeature("-thumb-mode");
    else
      Features.AddFeature("+thumb-mode");
    SecondarySTI.reset(TheTarget->createMCSubtargetInfo(TripleName, MCPU,
                                                        Features.getString()));
    SecondaryDisAsm.reset(TheTarget->createMCDisassembler(*SecondarySTI, Ctx));
  }

  std::unique_ptr<const MCInstrAnalysis> MIA(
      TheTarget->createMCInstrAnalysis(MII.get()));

  int AsmPrinterVariant = AsmInfo->getAssemblerDialect();
  std::unique_ptr<MCInstPrinter> IP(TheTarget->createMCInstPrinter(
      Triple(TripleName), AsmPrinterVariant, *AsmInfo, *MII, *MRI));
  if (!IP)
    reportError(Obj->getFileName(),
                "no instruction printer for target " + TripleName);
  IP->setPrintImmHex(PrintImmHex);
  IP->setPrintBranchImmAsAddress(true);
  IP->setSymbolizeOperands(SymbolizeOperands);
  IP->setMCInstrAnalysis(MIA.get());

  PrettyPrinter &PIP = selectPrettyPrinter(Triple(TripleName));
  SourcePrinter SP(Obj, TheTarget->getName());

  for (StringRef Opt : DisassemblerOptions)
    if (!IP->applyTargetSpecificCLOption(Opt))
      reportError(Obj->getFileName(),
                  "Unrecognized disassembler option: " + Opt);

  disassembleObject(TheTarget, *Obj, Ctx, DisAsm.get(), SecondaryDisAsm.get(),
                    MIA.get(), IP.get(), STI.get(), SecondarySTI.get(), PIP, SP,
                    InlineRelocs);
}

void objdump::printRelocations(const ObjectFile *Obj) {
  StringRef Fmt = Obj->getBytesInAddress() > 4 ? "%016" PRIx64 :
                                                 "%08" PRIx64;

  // Build a mapping from relocation target to a vector of relocation
  // sections. Usually, there is an only one relocation section for
  // each relocated section.
  MapVector<SectionRef, std::vector<SectionRef>> SecToRelSec;
  uint64_t Ndx;
  for (const SectionRef &Section : ToolSectionFilter(*Obj, &Ndx)) {
    if (Obj->isELF() && (ELFSectionRef(Section).getFlags() & ELF::SHF_ALLOC))
      continue;
    if (Section.relocation_begin() == Section.relocation_end())
      continue;
    Expected<section_iterator> SecOrErr = Section.getRelocatedSection();
    if (!SecOrErr)
      reportError(Obj->getFileName(),
                  "section (" + Twine(Ndx) +
                      "): unable to get a relocation target: " +
                      toString(SecOrErr.takeError()));
    SecToRelSec[**SecOrErr].push_back(Section);
  }

  for (std::pair<SectionRef, std::vector<SectionRef>> &P : SecToRelSec) {
    StringRef SecName = unwrapOrError(P.first.getName(), Obj->getFileName());
    outs() << "\nRELOCATION RECORDS FOR [" << SecName << "]:\n";
    uint32_t OffsetPadding = (Obj->getBytesInAddress() > 4 ? 16 : 8);
    uint32_t TypePadding = 24;
    outs() << left_justify("OFFSET", OffsetPadding) << " "
           << left_justify("TYPE", TypePadding) << " "
           << "VALUE\n";

    for (SectionRef Section : P.second) {
      for (const RelocationRef &Reloc : Section.relocations()) {
        uint64_t Address = Reloc.getOffset();
        SmallString<32> RelocName;
        SmallString<32> ValueStr;
        if (Address < StartAddress || Address > StopAddress || getHidden(Reloc))
          continue;
        Reloc.getTypeName(RelocName);
        if (Error E = getRelocationValueString(Reloc, ValueStr))
          reportError(std::move(E), Obj->getFileName());

        outs() << format(Fmt.data(), Address) << " "
               << left_justify(RelocName, TypePadding) << " " << ValueStr
               << "\n";
      }
    }
  }
}

void objdump::printDynamicRelocations(const ObjectFile *Obj) {
  // For the moment, this option is for ELF only
  if (!Obj->isELF())
    return;

  const auto *Elf = dyn_cast<ELFObjectFileBase>(Obj);
  if (!Elf || !any_of(Elf->sections(), [](const ELFSectionRef Sec) {
        return Sec.getType() == ELF::SHT_DYNAMIC;
      })) {
    reportError(Obj->getFileName(), "not a dynamic object");
    return;
  }

  std::vector<SectionRef> DynRelSec = Obj->dynamic_relocation_sections();
  if (DynRelSec.empty())
    return;

  outs() << "\nDYNAMIC RELOCATION RECORDS\n";
  const uint32_t OffsetPadding = (Obj->getBytesInAddress() > 4 ? 16 : 8);
  const uint32_t TypePadding = 24;
  outs() << left_justify("OFFSET", OffsetPadding) << ' '
         << left_justify("TYPE", TypePadding) << " VALUE\n";

  StringRef Fmt = Obj->getBytesInAddress() > 4 ? "%016" PRIx64 : "%08" PRIx64;
  for (const SectionRef &Section : DynRelSec)
    for (const RelocationRef &Reloc : Section.relocations()) {
      uint64_t Address = Reloc.getOffset();
      SmallString<32> RelocName;
      SmallString<32> ValueStr;
      Reloc.getTypeName(RelocName);
      if (Error E = getRelocationValueString(Reloc, ValueStr))
        reportError(std::move(E), Obj->getFileName());
      outs() << format(Fmt.data(), Address) << ' '
             << left_justify(RelocName, TypePadding) << ' ' << ValueStr << '\n';
    }
}

// Returns true if we need to show LMA column when dumping section headers. We
// show it only when the platform is ELF and either we have at least one section
// whose VMA and LMA are different and/or when --show-lma flag is used.
static bool shouldDisplayLMA(const ObjectFile &Obj) {
  if (!Obj.isELF())
    return false;
  for (const SectionRef &S : ToolSectionFilter(Obj))
    if (S.getAddress() != getELFSectionLMA(S))
      return true;
  return ShowLMA;
}

static size_t getMaxSectionNameWidth(const ObjectFile &Obj) {
  // Default column width for names is 13 even if no names are that long.
  size_t MaxWidth = 13;
  for (const SectionRef &Section : ToolSectionFilter(Obj)) {
    StringRef Name = unwrapOrError(Section.getName(), Obj.getFileName());
    MaxWidth = std::max(MaxWidth, Name.size());
  }
  return MaxWidth;
}

void objdump::printSectionHeaders(ObjectFile &Obj) {
  size_t NameWidth = getMaxSectionNameWidth(Obj);
  size_t AddressWidth = 2 * Obj.getBytesInAddress();
  bool HasLMAColumn = shouldDisplayLMA(Obj);
  outs() << "\nSections:\n";
  if (HasLMAColumn)
    outs() << "Idx " << left_justify("Name", NameWidth) << " Size     "
           << left_justify("VMA", AddressWidth) << " "
           << left_justify("LMA", AddressWidth) << " Type\n";
  else
    outs() << "Idx " << left_justify("Name", NameWidth) << " Size     "
           << left_justify("VMA", AddressWidth) << " Type\n";

  if (Obj.isELF() && Obj.sections().empty())
    createFakeELFSections(Obj);

  uint64_t Idx;
  for (const SectionRef &Section : ToolSectionFilter(Obj, &Idx)) {
    StringRef Name = unwrapOrError(Section.getName(), Obj.getFileName());
    uint64_t VMA = Section.getAddress();
    if (shouldAdjustVA(Section))
      VMA += AdjustVMA;

    uint64_t Size = Section.getSize();

    std::string Type = Section.isText() ? "TEXT" : "";
    if (Section.isData())
      Type += Type.empty() ? "DATA" : ", DATA";
    if (Section.isBSS())
      Type += Type.empty() ? "BSS" : ", BSS";
    if (Section.isDebugSection())
      Type += Type.empty() ? "DEBUG" : ", DEBUG";

    if (HasLMAColumn)
      outs() << format("%3" PRIu64 " %-*s %08" PRIx64 " ", Idx, NameWidth,
                       Name.str().c_str(), Size)
             << format_hex_no_prefix(VMA, AddressWidth) << " "
             << format_hex_no_prefix(getELFSectionLMA(Section), AddressWidth)
             << " " << Type << "\n";
    else
      outs() << format("%3" PRIu64 " %-*s %08" PRIx64 " ", Idx, NameWidth,
                       Name.str().c_str(), Size)
             << format_hex_no_prefix(VMA, AddressWidth) << " " << Type << "\n";
  }
}

void objdump::printSectionContents(const ObjectFile *Obj) {
  const MachOObjectFile *MachO = dyn_cast<const MachOObjectFile>(Obj);

  for (const SectionRef &Section : ToolSectionFilter(*Obj)) {
    StringRef Name = unwrapOrError(Section.getName(), Obj->getFileName());
    uint64_t BaseAddr = Section.getAddress();
    uint64_t Size = Section.getSize();
    if (!Size)
      continue;

    outs() << "Contents of section ";
    StringRef SegmentName = getSegmentName(MachO, Section);
    if (!SegmentName.empty())
      outs() << SegmentName << ",";
    outs() << Name << ":\n";
    if (Section.isBSS()) {
      outs() << format("<skipping contents of bss section at [%04" PRIx64
                       ", %04" PRIx64 ")>\n",
                       BaseAddr, BaseAddr + Size);
      continue;
    }

    StringRef Contents = unwrapOrError(Section.getContents(), Obj->getFileName());

    // Dump out the content as hex and printable ascii characters.
    for (std::size_t Addr = 0, End = Contents.size(); Addr < End; Addr += 16) {
      outs() << format(" %04" PRIx64 " ", BaseAddr + Addr);
      // Dump line of hex.
      for (std::size_t I = 0; I < 16; ++I) {
        if (I != 0 && I % 4 == 0)
          outs() << ' ';
        if (Addr + I < End)
          outs() << hexdigit((Contents[Addr + I] >> 4) & 0xF, true)
                 << hexdigit(Contents[Addr + I] & 0xF, true);
        else
          outs() << "  ";
      }
      // Print ascii.
      outs() << "  ";
      for (std::size_t I = 0; I < 16 && Addr + I < End; ++I) {
        if (isPrint(static_cast<unsigned char>(Contents[Addr + I]) & 0xFF))
          outs() << Contents[Addr + I];
        else
          outs() << ".";
      }
      outs() << "\n";
    }
  }
}

void objdump::printSymbolTable(const ObjectFile &O, StringRef ArchiveName,
                               StringRef ArchitectureName, bool DumpDynamic) {
  if (O.isCOFF() && !DumpDynamic) {
    outs() << "\nSYMBOL TABLE:\n";
    printCOFFSymbolTable(cast<const COFFObjectFile>(O));
    return;
  }

  const StringRef FileName = O.getFileName();

  if (!DumpDynamic) {
    outs() << "\nSYMBOL TABLE:\n";
    for (auto I = O.symbol_begin(); I != O.symbol_end(); ++I)
      printSymbol(O, *I, {}, FileName, ArchiveName, ArchitectureName,
                  DumpDynamic);
    return;
  }

  outs() << "\nDYNAMIC SYMBOL TABLE:\n";
  if (!O.isELF()) {
    reportWarning(
        "this operation is not currently supported for this file format",
        FileName);
    return;
  }

  const ELFObjectFileBase *ELF = cast<const ELFObjectFileBase>(&O);
  auto Symbols = ELF->getDynamicSymbolIterators();
  Expected<std::vector<VersionEntry>> SymbolVersionsOrErr =
      ELF->readDynsymVersions();
  if (!SymbolVersionsOrErr) {
    reportWarning(toString(SymbolVersionsOrErr.takeError()), FileName);
    SymbolVersionsOrErr = std::vector<VersionEntry>();
    (void)!SymbolVersionsOrErr;
  }
  for (auto &Sym : Symbols)
    printSymbol(O, Sym, *SymbolVersionsOrErr, FileName, ArchiveName,
                ArchitectureName, DumpDynamic);
}

void objdump::printSymbol(const ObjectFile &O, const SymbolRef &Symbol,
                          ArrayRef<VersionEntry> SymbolVersions,
                          StringRef FileName, StringRef ArchiveName,
                          StringRef ArchitectureName, bool DumpDynamic) {
  const MachOObjectFile *MachO = dyn_cast<const MachOObjectFile>(&O);
  uint64_t Address = unwrapOrError(Symbol.getAddress(), FileName, ArchiveName,
                                   ArchitectureName);
  if ((Address < StartAddress) || (Address > StopAddress))
    return;
  SymbolRef::Type Type =
      unwrapOrError(Symbol.getType(), FileName, ArchiveName, ArchitectureName);
  uint32_t Flags =
      unwrapOrError(Symbol.getFlags(), FileName, ArchiveName, ArchitectureName);

  // Don't ask a Mach-O STAB symbol for its section unless you know that
  // STAB symbol's section field refers to a valid section index. Otherwise
  // the symbol may error trying to load a section that does not exist.
  bool IsSTAB = false;
  if (MachO) {
    DataRefImpl SymDRI = Symbol.getRawDataRefImpl();
    uint8_t NType =
        (MachO->is64Bit() ? MachO->getSymbol64TableEntry(SymDRI).n_type
                          : MachO->getSymbolTableEntry(SymDRI).n_type);
    if (NType & MachO::N_STAB)
      IsSTAB = true;
  }
  section_iterator Section = IsSTAB
                                 ? O.section_end()
                                 : unwrapOrError(Symbol.getSection(), FileName,
                                                 ArchiveName, ArchitectureName);

  StringRef Name;
  if (Type == SymbolRef::ST_Debug && Section != O.section_end()) {
    if (Expected<StringRef> NameOrErr = Section->getName())
      Name = *NameOrErr;
    else
      consumeError(NameOrErr.takeError());

  } else {
    Name = unwrapOrError(Symbol.getName(), FileName, ArchiveName,
                         ArchitectureName);
  }

  bool Global = Flags & SymbolRef::SF_Global;
  bool Weak = Flags & SymbolRef::SF_Weak;
  bool Absolute = Flags & SymbolRef::SF_Absolute;
  bool Common = Flags & SymbolRef::SF_Common;
  bool Hidden = Flags & SymbolRef::SF_Hidden;

  char GlobLoc = ' ';
  if ((Section != O.section_end() || Absolute) && !Weak)
    GlobLoc = Global ? 'g' : 'l';
  char IFunc = ' ';
  if (O.isELF()) {
    if (ELFSymbolRef(Symbol).getELFType() == ELF::STT_GNU_IFUNC)
      IFunc = 'i';
    if (ELFSymbolRef(Symbol).getBinding() == ELF::STB_GNU_UNIQUE)
      GlobLoc = 'u';
  }

  char Debug = ' ';
  if (DumpDynamic)
    Debug = 'D';
  else if (Type == SymbolRef::ST_Debug || Type == SymbolRef::ST_File)
    Debug = 'd';

  char FileFunc = ' ';
  if (Type == SymbolRef::ST_File)
    FileFunc = 'f';
  else if (Type == SymbolRef::ST_Function)
    FileFunc = 'F';
  else if (Type == SymbolRef::ST_Data)
    FileFunc = 'O';

  const char *Fmt = O.getBytesInAddress() > 4 ? "%016" PRIx64 : "%08" PRIx64;

  outs() << format(Fmt, Address) << " "
         << GlobLoc            // Local -> 'l', Global -> 'g', Neither -> ' '
         << (Weak ? 'w' : ' ') // Weak?
         << ' '                // Constructor. Not supported yet.
         << ' '                // Warning. Not supported yet.
         << IFunc              // Indirect reference to another symbol.
         << Debug              // Debugging (d) or dynamic (D) symbol.
         << FileFunc           // Name of function (F), file (f) or object (O).
         << ' ';
  if (Absolute) {
    outs() << "*ABS*";
  } else if (Common) {
    outs() << "*COM*";
  } else if (Section == O.section_end()) {
    if (O.isXCOFF()) {
      XCOFFSymbolRef XCOFFSym = cast<const XCOFFObjectFile>(O).toSymbolRef(
          Symbol.getRawDataRefImpl());
      if (XCOFF::N_DEBUG == XCOFFSym.getSectionNumber())
        outs() << "*DEBUG*";
      else
        outs() << "*UND*";
    } else
      outs() << "*UND*";
  } else {
    StringRef SegmentName = getSegmentName(MachO, *Section);
    if (!SegmentName.empty())
      outs() << SegmentName << ",";
    StringRef SectionName = unwrapOrError(Section->getName(), FileName);
    outs() << SectionName;
    if (O.isXCOFF()) {
      Optional<SymbolRef> SymRef =
          getXCOFFSymbolContainingSymbolRef(cast<XCOFFObjectFile>(O), Symbol);
      if (SymRef) {

        Expected<StringRef> NameOrErr = SymRef->getName();

        if (NameOrErr) {
          outs() << " (csect:";
          std::string SymName(NameOrErr.get());

          if (Demangle)
            SymName = demangle(SymName);

          if (SymbolDescription)
            SymName = getXCOFFSymbolDescription(
                createSymbolInfo(O, SymRef.value()), SymName);

          outs() << ' ' << SymName;
          outs() << ") ";
        } else
          reportWarning(toString(NameOrErr.takeError()), FileName);
      }
    }
  }

  if (Common)
    outs() << '\t' << format(Fmt, static_cast<uint64_t>(Symbol.getAlignment()));
  else if (O.isXCOFF())
    outs() << '\t'
           << format(Fmt, cast<XCOFFObjectFile>(O).getSymbolSize(
                              Symbol.getRawDataRefImpl()));
  else if (O.isELF())
    outs() << '\t' << format(Fmt, ELFSymbolRef(Symbol).getSize());

  if (O.isELF()) {
    if (!SymbolVersions.empty()) {
      const VersionEntry &Ver =
          SymbolVersions[Symbol.getRawDataRefImpl().d.b - 1];
      std::string Str;
      if (!Ver.Name.empty())
        Str = Ver.IsVerDef ? ' ' + Ver.Name : '(' + Ver.Name + ')';
      outs() << ' ' << left_justify(Str, 12);
    }

    uint8_t Other = ELFSymbolRef(Symbol).getOther();
    switch (Other) {
    case ELF::STV_DEFAULT:
      break;
    case ELF::STV_INTERNAL:
      outs() << " .internal";
      break;
    case ELF::STV_HIDDEN:
      outs() << " .hidden";
      break;
    case ELF::STV_PROTECTED:
      outs() << " .protected";
      break;
    default:
      outs() << format(" 0x%02x", Other);
      break;
    }
  } else if (Hidden) {
    outs() << " .hidden";
  }

  std::string SymName(Name);
  if (Demangle)
    SymName = demangle(SymName);

  if (O.isXCOFF() && SymbolDescription)
    SymName = getXCOFFSymbolDescription(createSymbolInfo(O, Symbol), SymName);

  outs() << ' ' << SymName << '\n';
}

static void printUnwindInfo(const ObjectFile *O) {
  outs() << "Unwind info:\n\n";

  if (const COFFObjectFile *Coff = dyn_cast<COFFObjectFile>(O))
    printCOFFUnwindInfo(Coff);
  else if (const MachOObjectFile *MachO = dyn_cast<MachOObjectFile>(O))
    printMachOUnwindInfo(MachO);
  else
    // TODO: Extract DWARF dump tool to objdump.
    WithColor::error(errs(), ToolName)
        << "This operation is only currently supported "
           "for COFF and MachO object files.\n";
}

/// Dump the raw contents of the __clangast section so the output can be piped
/// into llvm-bcanalyzer.
static void printRawClangAST(const ObjectFile *Obj) {
  if (outs().is_displayed()) {
    WithColor::error(errs(), ToolName)
        << "The -raw-clang-ast option will dump the raw binary contents of "
           "the clang ast section.\n"
           "Please redirect the output to a file or another program such as "
           "llvm-bcanalyzer.\n";
    return;
  }

  StringRef ClangASTSectionName("__clangast");
  if (Obj->isCOFF()) {
    ClangASTSectionName = "clangast";
  }

  Optional<object::SectionRef> ClangASTSection;
  for (auto Sec : ToolSectionFilter(*Obj)) {
    StringRef Name;
    if (Expected<StringRef> NameOrErr = Sec.getName())
      Name = *NameOrErr;
    else
      consumeError(NameOrErr.takeError());

    if (Name == ClangASTSectionName) {
      ClangASTSection = Sec;
      break;
    }
  }
  if (!ClangASTSection)
    return;

  StringRef ClangASTContents =
      unwrapOrError(ClangASTSection.value().getContents(), Obj->getFileName());
  outs().write(ClangASTContents.data(), ClangASTContents.size());
}

static void printFaultMaps(const ObjectFile *Obj) {
  StringRef FaultMapSectionName;

  if (Obj->isELF()) {
    FaultMapSectionName = ".llvm_faultmaps";
  } else if (Obj->isMachO()) {
    FaultMapSectionName = "__llvm_faultmaps";
  } else {
    WithColor::error(errs(), ToolName)
        << "This operation is only currently supported "
           "for ELF and Mach-O executable files.\n";
    return;
  }

  Optional<object::SectionRef> FaultMapSection;

  for (auto Sec : ToolSectionFilter(*Obj)) {
    StringRef Name;
    if (Expected<StringRef> NameOrErr = Sec.getName())
      Name = *NameOrErr;
    else
      consumeError(NameOrErr.takeError());

    if (Name == FaultMapSectionName) {
      FaultMapSection = Sec;
      break;
    }
  }

  outs() << "FaultMap table:\n";

  if (!FaultMapSection) {
    outs() << "<not found>\n";
    return;
  }

  StringRef FaultMapContents =
      unwrapOrError(FaultMapSection->getContents(), Obj->getFileName());
  FaultMapParser FMP(FaultMapContents.bytes_begin(),
                     FaultMapContents.bytes_end());

  outs() << FMP;
}

static void printPrivateFileHeaders(const ObjectFile *O, bool OnlyFirst) {
  if (O->isELF()) {
    printELFFileHeader(O);
    printELFDynamicSection(O);
    printELFSymbolVersionInfo(O);
    return;
  }
  if (O->isCOFF())
    return printCOFFFileHeader(cast<object::COFFObjectFile>(*O));
  if (O->isWasm())
    return printWasmFileHeader(O);
  if (O->isMachO()) {
    printMachOFileHeader(O);
    if (!OnlyFirst)
      printMachOLoadCommands(O);
    return;
  }
  reportError(O->getFileName(), "Invalid/Unsupported object file format");
}

static void printFileHeaders(const ObjectFile *O) {
  if (!O->isELF() && !O->isCOFF())
    reportError(O->getFileName(), "Invalid/Unsupported object file format");

  Triple::ArchType AT = O->getArch();
  outs() << "architecture: " << Triple::getArchTypeName(AT) << "\n";
  uint64_t Address = unwrapOrError(O->getStartAddress(), O->getFileName());

  StringRef Fmt = O->getBytesInAddress() > 4 ? "%016" PRIx64 : "%08" PRIx64;
  outs() << "start address: "
         << "0x" << format(Fmt.data(), Address) << "\n";
}

static void printArchiveChild(StringRef Filename, const Archive::Child &C) {
  Expected<sys::fs::perms> ModeOrErr = C.getAccessMode();
  if (!ModeOrErr) {
    WithColor::error(errs(), ToolName) << "ill-formed archive entry.\n";
    consumeError(ModeOrErr.takeError());
    return;
  }
  sys::fs::perms Mode = ModeOrErr.get();
  outs() << ((Mode & sys::fs::owner_read) ? "r" : "-");
  outs() << ((Mode & sys::fs::owner_write) ? "w" : "-");
  outs() << ((Mode & sys::fs::owner_exe) ? "x" : "-");
  outs() << ((Mode & sys::fs::group_read) ? "r" : "-");
  outs() << ((Mode & sys::fs::group_write) ? "w" : "-");
  outs() << ((Mode & sys::fs::group_exe) ? "x" : "-");
  outs() << ((Mode & sys::fs::others_read) ? "r" : "-");
  outs() << ((Mode & sys::fs::others_write) ? "w" : "-");
  outs() << ((Mode & sys::fs::others_exe) ? "x" : "-");

  outs() << " ";

  outs() << format("%d/%d %6" PRId64 " ", unwrapOrError(C.getUID(), Filename),
                   unwrapOrError(C.getGID(), Filename),
                   unwrapOrError(C.getRawSize(), Filename));

  StringRef RawLastModified = C.getRawLastModified();
  unsigned Seconds;
  if (RawLastModified.getAsInteger(10, Seconds))
    outs() << "(date: \"" << RawLastModified
           << "\" contains non-decimal chars) ";
  else {
    // Since ctime(3) returns a 26 character string of the form:
    // "Sun Sep 16 01:03:52 1973\n\0"
    // just print 24 characters.
    time_t t = Seconds;
    outs() << format("%.24s ", ctime(&t));
  }

  StringRef Name = "";
  Expected<StringRef> NameOrErr = C.getName();
  if (!NameOrErr) {
    consumeError(NameOrErr.takeError());
    Name = unwrapOrError(C.getRawName(), Filename);
  } else {
    Name = NameOrErr.get();
  }
  outs() << Name << "\n";
}

// For ELF only now.
static bool shouldWarnForInvalidStartStopAddress(ObjectFile *Obj) {
  if (const auto *Elf = dyn_cast<ELFObjectFileBase>(Obj)) {
    if (Elf->getEType() != ELF::ET_REL)
      return true;
  }
  return false;
}

static void checkForInvalidStartStopAddress(ObjectFile *Obj,
                                            uint64_t Start, uint64_t Stop) {
  if (!shouldWarnForInvalidStartStopAddress(Obj))
    return;

  for (const SectionRef &Section : Obj->sections())
    if (ELFSectionRef(Section).getFlags() & ELF::SHF_ALLOC) {
      uint64_t BaseAddr = Section.getAddress();
      uint64_t Size = Section.getSize();
      if ((Start < BaseAddr + Size) && Stop > BaseAddr)
        return;
    }

  if (!HasStartAddressFlag)
    reportWarning("no section has address less than 0x" +
                      Twine::utohexstr(Stop) + " specified by --stop-address",
                  Obj->getFileName());
  else if (!HasStopAddressFlag)
    reportWarning("no section has address greater than or equal to 0x" +
                      Twine::utohexstr(Start) + " specified by --start-address",
                  Obj->getFileName());
  else
    reportWarning("no section overlaps the range [0x" +
                      Twine::utohexstr(Start) + ",0x" + Twine::utohexstr(Stop) +
                      ") specified by --start-address/--stop-address",
                  Obj->getFileName());
}

static void dumpObject(ObjectFile *O, const Archive *A = nullptr,
                       const Archive::Child *C = nullptr) {
  // Avoid other output when using a raw option.
  if (!RawClangAST) {
    outs() << '\n';
    if (A)
      outs() << A->getFileName() << "(" << O->getFileName() << ")";
    else
      outs() << O->getFileName();
    outs() << ":\tfile format " << O->getFileFormatName().lower() << "\n";
  }

  if (HasStartAddressFlag || HasStopAddressFlag)
    checkForInvalidStartStopAddress(O, StartAddress, StopAddress);

  // Note: the order here matches GNU objdump for compatability.
  StringRef ArchiveName = A ? A->getFileName() : "";
  if (ArchiveHeaders && !MachOOpt && C)
    printArchiveChild(ArchiveName, *C);
  if (FileHeaders)
    printFileHeaders(O);
  if (PrivateHeaders || FirstPrivateHeader)
    printPrivateFileHeaders(O, FirstPrivateHeader);
  if (SectionHeaders)
    printSectionHeaders(*O);
  if (SymbolTable)
    printSymbolTable(*O, ArchiveName);
  if (DynamicSymbolTable)
    printSymbolTable(*O, ArchiveName, /*ArchitectureName=*/"",
                     /*DumpDynamic=*/true);
  if (DwarfDumpType != DIDT_Null) {
    std::unique_ptr<DIContext> DICtx = DWARFContext::create(*O);
    // Dump the complete DWARF structure.
    DIDumpOptions DumpOpts;
    DumpOpts.DumpType = DwarfDumpType;
    DICtx->dump(outs(), DumpOpts);
  }
  if (Relocations && !Disassemble)
    printRelocations(O);
  if (DynamicRelocations)
    printDynamicRelocations(O);
  if (SectionContents)
    printSectionContents(O);
  if (Disassemble)
    disassembleObject(O, Relocations);
  if (UnwindInfo)
    printUnwindInfo(O);

  // Mach-O specific options:
  if (ExportsTrie)
    printExportsTrie(O);
  if (Rebase)
    printRebaseTable(O);
  if (Bind)
    printBindTable(O);
  if (LazyBind)
    printLazyBindTable(O);
  if (WeakBind)
    printWeakBindTable(O);

  // Other special sections:
  if (RawClangAST)
    printRawClangAST(O);
  if (FaultMapSection)
    printFaultMaps(O);
  if (Offloading)
    dumpOffloadBinary(*O);
}

static void dumpObject(const COFFImportFile *I, const Archive *A,
                       const Archive::Child *C = nullptr) {
  StringRef ArchiveName = A ? A->getFileName() : "";

  // Avoid other output when using a raw option.
  if (!RawClangAST)
    outs() << '\n'
           << ArchiveName << "(" << I->getFileName() << ")"
           << ":\tfile format COFF-import-file"
           << "\n\n";

  if (ArchiveHeaders && !MachOOpt && C)
    printArchiveChild(ArchiveName, *C);
  if (SymbolTable)
    printCOFFSymbolTable(*I);
}

/// Dump each object file in \a a;
static void dumpArchive(const Archive *A) {
  Error Err = Error::success();
  unsigned I = -1;
  for (auto &C : A->children(Err)) {
    ++I;
    Expected<std::unique_ptr<Binary>> ChildOrErr = C.getAsBinary();
    if (!ChildOrErr) {
      if (auto E = isNotObjectErrorInvalidFileType(ChildOrErr.takeError()))
        reportError(std::move(E), getFileNameForError(C, I), A->getFileName());
      continue;
    }
    if (ObjectFile *O = dyn_cast<ObjectFile>(&*ChildOrErr.get()))
      dumpObject(O, A, &C);
    else if (COFFImportFile *I = dyn_cast<COFFImportFile>(&*ChildOrErr.get()))
      dumpObject(I, A, &C);
    else
      reportError(errorCodeToError(object_error::invalid_file_type),
                  A->getFileName());
  }
  if (Err)
    reportError(std::move(Err), A->getFileName());
}

/// Open file and figure out how to dump it.
static void dumpInput(StringRef file) {
  // If we are using the Mach-O specific object file parser, then let it parse
  // the file and process the command line options.  So the -arch flags can
  // be used to select specific slices, etc.
  if (MachOOpt) {
    parseInputMachO(file);
    return;
  }

  // Attempt to open the binary.
  OwningBinary<Binary> OBinary = unwrapOrError(createBinary(file), file);
  Binary &Binary = *OBinary.getBinary();

  if (Archive *A = dyn_cast<Archive>(&Binary))
    dumpArchive(A);
  else if (ObjectFile *O = dyn_cast<ObjectFile>(&Binary))
    dumpObject(O);
  else if (MachOUniversalBinary *UB = dyn_cast<MachOUniversalBinary>(&Binary))
    parseInputMachO(UB);
  else if (OffloadBinary *OB = dyn_cast<OffloadBinary>(&Binary))
    dumpOffloadSections(*OB);
  else
    reportError(errorCodeToError(object_error::invalid_file_type), file);
}

template <typename T>
static void parseIntArg(const llvm::opt::InputArgList &InputArgs, int ID,
                        T &Value) {
  if (const opt::Arg *A = InputArgs.getLastArg(ID)) {
    StringRef V(A->getValue());
    if (!llvm::to_integer(V, Value, 0)) {
      reportCmdLineError(A->getSpelling() +
                         ": expected a non-negative integer, but got '" + V +
                         "'");
    }
  }
}

static void invalidArgValue(const opt::Arg *A) {
  reportCmdLineError("'" + StringRef(A->getValue()) +
                     "' is not a valid value for '" + A->getSpelling() + "'");
}

static std::vector<std::string>
commaSeparatedValues(const llvm::opt::InputArgList &InputArgs, int ID) {
  std::vector<std::string> Values;
  for (StringRef Value : InputArgs.getAllArgValues(ID)) {
    llvm::SmallVector<StringRef, 2> SplitValues;
    llvm::SplitString(Value, SplitValues, ",");
    for (StringRef SplitValue : SplitValues)
      Values.push_back(SplitValue.str());
  }
  return Values;
}

static void parseOtoolOptions(const llvm::opt::InputArgList &InputArgs) {
  MachOOpt = true;
  FullLeadingAddr = true;
  PrintImmHex = true;

  ArchName = InputArgs.getLastArgValue(OTOOL_arch).str();
  LinkOptHints = InputArgs.hasArg(OTOOL_C);
  if (InputArgs.hasArg(OTOOL_d))
    FilterSections.push_back("__DATA,__data");
  DylibId = InputArgs.hasArg(OTOOL_D);
  UniversalHeaders = InputArgs.hasArg(OTOOL_f);
  DataInCode = InputArgs.hasArg(OTOOL_G);
  FirstPrivateHeader = InputArgs.hasArg(OTOOL_h);
  IndirectSymbols = InputArgs.hasArg(OTOOL_I);
  ShowRawInsn = InputArgs.hasArg(OTOOL_j);
  PrivateHeaders = InputArgs.hasArg(OTOOL_l);
  DylibsUsed = InputArgs.hasArg(OTOOL_L);
  MCPU = InputArgs.getLastArgValue(OTOOL_mcpu_EQ).str();
  ObjcMetaData = InputArgs.hasArg(OTOOL_o);
  DisSymName = InputArgs.getLastArgValue(OTOOL_p).str();
  InfoPlist = InputArgs.hasArg(OTOOL_P);
  Relocations = InputArgs.hasArg(OTOOL_r);
  if (const Arg *A = InputArgs.getLastArg(OTOOL_s)) {
    auto Filter = (A->getValue(0) + StringRef(",") + A->getValue(1)).str();
    FilterSections.push_back(Filter);
  }
  if (InputArgs.hasArg(OTOOL_t))
    FilterSections.push_back("__TEXT,__text");
  Verbose = InputArgs.hasArg(OTOOL_v) || InputArgs.hasArg(OTOOL_V) ||
            InputArgs.hasArg(OTOOL_o);
  SymbolicOperands = InputArgs.hasArg(OTOOL_V);
  if (InputArgs.hasArg(OTOOL_x))
    FilterSections.push_back(",__text");
  LeadingAddr = LeadingHeaders = !InputArgs.hasArg(OTOOL_X);

  InputFilenames = InputArgs.getAllArgValues(OTOOL_INPUT);
  if (InputFilenames.empty())
    reportCmdLineError("no input file");

  for (const Arg *A : InputArgs) {
    const Option &O = A->getOption();
    if (O.getGroup().isValid() && O.getGroup().getID() == OTOOL_grp_obsolete) {
      reportCmdLineWarning(O.getPrefixedName() +
                           " is obsolete and not implemented");
    }
  }
}

static void parseObjdumpOptions(const llvm::opt::InputArgList &InputArgs) {
  parseIntArg(InputArgs, OBJDUMP_adjust_vma_EQ, AdjustVMA);
  AllHeaders = InputArgs.hasArg(OBJDUMP_all_headers);
  ArchName = InputArgs.getLastArgValue(OBJDUMP_arch_name_EQ).str();
  ArchiveHeaders = InputArgs.hasArg(OBJDUMP_archive_headers);
  Demangle = InputArgs.hasArg(OBJDUMP_demangle);
  Disassemble = InputArgs.hasArg(OBJDUMP_disassemble);
  DisassembleAll = InputArgs.hasArg(OBJDUMP_disassemble_all);
  SymbolDescription = InputArgs.hasArg(OBJDUMP_symbol_description);
  DisassembleSymbols =
      commaSeparatedValues(InputArgs, OBJDUMP_disassemble_symbols_EQ);
  DisassembleZeroes = InputArgs.hasArg(OBJDUMP_disassemble_zeroes);
  if (const opt::Arg *A = InputArgs.getLastArg(OBJDUMP_dwarf_EQ)) {
    DwarfDumpType = StringSwitch<DIDumpType>(A->getValue())
                        .Case("frames", DIDT_DebugFrame)
                        .Default(DIDT_Null);
    if (DwarfDumpType == DIDT_Null)
      invalidArgValue(A);
  }
  DynamicRelocations = InputArgs.hasArg(OBJDUMP_dynamic_reloc);
  FaultMapSection = InputArgs.hasArg(OBJDUMP_fault_map_section);
  Offloading = InputArgs.hasArg(OBJDUMP_offloading);
  FileHeaders = InputArgs.hasArg(OBJDUMP_file_headers);
  SectionContents = InputArgs.hasArg(OBJDUMP_full_contents);
  PrintLines = InputArgs.hasArg(OBJDUMP_line_numbers);
  InputFilenames = InputArgs.getAllArgValues(OBJDUMP_INPUT);
  MachOOpt = InputArgs.hasArg(OBJDUMP_macho);
  MCPU = InputArgs.getLastArgValue(OBJDUMP_mcpu_EQ).str();
  MAttrs = commaSeparatedValues(InputArgs, OBJDUMP_mattr_EQ);
  ShowRawInsn = !InputArgs.hasArg(OBJDUMP_no_show_raw_insn);
  LeadingAddr = !InputArgs.hasArg(OBJDUMP_no_leading_addr);
  RawClangAST = InputArgs.hasArg(OBJDUMP_raw_clang_ast);
  Relocations = InputArgs.hasArg(OBJDUMP_reloc);
  PrintImmHex =
      InputArgs.hasFlag(OBJDUMP_print_imm_hex, OBJDUMP_no_print_imm_hex, false);
  PrivateHeaders = InputArgs.hasArg(OBJDUMP_private_headers);
  FilterSections = InputArgs.getAllArgValues(OBJDUMP_section_EQ);
  SectionHeaders = InputArgs.hasArg(OBJDUMP_section_headers);
  ShowLMA = InputArgs.hasArg(OBJDUMP_show_lma);
  PrintSource = InputArgs.hasArg(OBJDUMP_source);
  parseIntArg(InputArgs, OBJDUMP_start_address_EQ, StartAddress);
  HasStartAddressFlag = InputArgs.hasArg(OBJDUMP_start_address_EQ);
  parseIntArg(InputArgs, OBJDUMP_stop_address_EQ, StopAddress);
  HasStopAddressFlag = InputArgs.hasArg(OBJDUMP_stop_address_EQ);
  SymbolTable = InputArgs.hasArg(OBJDUMP_syms);
  SymbolizeOperands = InputArgs.hasArg(OBJDUMP_symbolize_operands);
  DynamicSymbolTable = InputArgs.hasArg(OBJDUMP_dynamic_syms);
  TripleName = InputArgs.getLastArgValue(OBJDUMP_triple_EQ).str();
  UnwindInfo = InputArgs.hasArg(OBJDUMP_unwind_info);
  Wide = InputArgs.hasArg(OBJDUMP_wide);
  Prefix = InputArgs.getLastArgValue(OBJDUMP_prefix).str();
  parseIntArg(InputArgs, OBJDUMP_prefix_strip, PrefixStrip);
  if (const opt::Arg *A = InputArgs.getLastArg(OBJDUMP_debug_vars_EQ)) {
    DbgVariables = StringSwitch<DebugVarsFormat>(A->getValue())
                       .Case("ascii", DVASCII)
                       .Case("unicode", DVUnicode)
                       .Default(DVInvalid);
    if (DbgVariables == DVInvalid)
      invalidArgValue(A);
  }
  parseIntArg(InputArgs, OBJDUMP_debug_vars_indent_EQ, DbgIndent);

  parseMachOOptions(InputArgs);

  // Parse -M (--disassembler-options) and deprecated
  // --x86-asm-syntax={att,intel}.
  //
  // Note, for x86, the asm dialect (AssemblerDialect) is initialized when the
  // MCAsmInfo is constructed. MCInstPrinter::applyTargetSpecificCLOption is
  // called too late. For now we have to use the internal cl::opt option.
  const char *AsmSyntax = nullptr;
  for (const auto *A : InputArgs.filtered(OBJDUMP_disassembler_options_EQ,
                                          OBJDUMP_x86_asm_syntax_att,
                                          OBJDUMP_x86_asm_syntax_intel)) {
    switch (A->getOption().getID()) {
    case OBJDUMP_x86_asm_syntax_att:
      AsmSyntax = "--x86-asm-syntax=att";
      continue;
    case OBJDUMP_x86_asm_syntax_intel:
      AsmSyntax = "--x86-asm-syntax=intel";
      continue;
    }

    SmallVector<StringRef, 2> Values;
    llvm::SplitString(A->getValue(), Values, ",");
    for (StringRef V : Values) {
      if (V == "att")
        AsmSyntax = "--x86-asm-syntax=att";
      else if (V == "intel")
        AsmSyntax = "--x86-asm-syntax=intel";
      else
        DisassemblerOptions.push_back(V.str());
    }
  }
  if (AsmSyntax) {
    const char *Argv[] = {"llvm-objdump", AsmSyntax};
    llvm::cl::ParseCommandLineOptions(2, Argv);
  }

  // objdump defaults to a.out if no filenames specified.
  if (InputFilenames.empty())
    InputFilenames.push_back("a.out");
}

int main(int argc, char **argv) {
  using namespace llvm;
  InitLLVM X(argc, argv);

  ToolName = argv[0];
  std::unique_ptr<CommonOptTable> T;
  OptSpecifier Unknown, HelpFlag, HelpHiddenFlag, VersionFlag;

  StringRef Stem = sys::path::stem(ToolName);
  auto Is = [=](StringRef Tool) {
    // We need to recognize the following filenames:
    //
    // llvm-objdump -> objdump
    // llvm-otool-10.exe -> otool
    // powerpc64-unknown-freebsd13-objdump -> objdump
    auto I = Stem.rfind_insensitive(Tool);
    return I != StringRef::npos &&
           (I + Tool.size() == Stem.size() || !isAlnum(Stem[I + Tool.size()]));
  };
  if (Is("otool")) {
    T = std::make_unique<OtoolOptTable>();
    Unknown = OTOOL_UNKNOWN;
    HelpFlag = OTOOL_help;
    HelpHiddenFlag = OTOOL_help_hidden;
    VersionFlag = OTOOL_version;
  } else {
    T = std::make_unique<ObjdumpOptTable>();
    Unknown = OBJDUMP_UNKNOWN;
    HelpFlag = OBJDUMP_help;
    HelpHiddenFlag = OBJDUMP_help_hidden;
    VersionFlag = OBJDUMP_version;
  }

  BumpPtrAllocator A;
  StringSaver Saver(A);
  opt::InputArgList InputArgs =
      T->parseArgs(argc, argv, Unknown, Saver,
                   [&](StringRef Msg) { reportCmdLineError(Msg); });

  if (InputArgs.size() == 0 || InputArgs.hasArg(HelpFlag)) {
    T->printHelp(ToolName);
    return 0;
  }
  if (InputArgs.hasArg(HelpHiddenFlag)) {
    T->printHelp(ToolName, /*ShowHidden=*/true);
    return 0;
  }

  // Initialize targets and assembly printers/parsers.
  InitializeAllTargetInfos();
  InitializeAllTargetMCs();
  InitializeAllDisassemblers();

  if (InputArgs.hasArg(VersionFlag)) {
    cl::PrintVersionMessage();
    if (!Is("otool")) {
      outs() << '\n';
      TargetRegistry::printRegisteredTargetsForVersion(outs());
    }
    return 0;
  }

  if (Is("otool"))
    parseOtoolOptions(InputArgs);
  else
    parseObjdumpOptions(InputArgs);

  if (StartAddress >= StopAddress)
    reportCmdLineError("start address should be less than stop address");

  // Removes trailing separators from prefix.
  while (!Prefix.empty() && sys::path::is_separator(Prefix.back()))
    Prefix.pop_back();

  if (AllHeaders)
    ArchiveHeaders = FileHeaders = PrivateHeaders = Relocations =
        SectionHeaders = SymbolTable = true;

  if (DisassembleAll || PrintSource || PrintLines ||
      !DisassembleSymbols.empty())
    Disassemble = true;

  if (!ArchiveHeaders && !Disassemble && DwarfDumpType == DIDT_Null &&
      !DynamicRelocations && !FileHeaders && !PrivateHeaders && !RawClangAST &&
      !Relocations && !SectionHeaders && !SectionContents && !SymbolTable &&
      !DynamicSymbolTable && !UnwindInfo && !FaultMapSection && !Offloading &&
      !(MachOOpt && (Bind || DataInCode || DyldInfo || DylibId || DylibsUsed ||
                     ExportsTrie || FirstPrivateHeader || FunctionStarts ||
                     IndirectSymbols || InfoPlist || LazyBind || LinkOptHints ||
                     ObjcMetaData || Rebase || Rpaths || UniversalHeaders ||
                     WeakBind || !FilterSections.empty()))) {
    T->printHelp(ToolName);
    return 2;
  }

  DisasmSymbolSet.insert(DisassembleSymbols.begin(), DisassembleSymbols.end());

  llvm::for_each(InputFilenames, dumpInput);

  warnOnNoMatchForSections();

  return EXIT_SUCCESS;
}
