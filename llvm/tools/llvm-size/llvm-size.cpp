//===-- llvm-size.cpp - Print the size of each object section ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This program is a utility that works like traditional Unix "size",
// that is, it prints out the size of each section, and the total size of all
// sections.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APInt.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Driver.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <memory>
#include <string>

using namespace llvm;
using namespace object;

namespace {
using namespace llvm::opt; // for HelpHidden in Opts.inc
enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

#define OPTTABLE_CODE
#include "Opts.inc"

class SizeOptTable : public opt::OptTable {
public:
  SizeOptTable() : OptTable(optionTables()) { setGroupedShortOptions(true); }
};

enum OutputFormatTy { berkeley, sysv, darwin };
enum RadixTy { octal = 8, decimal = 10, hexadecimal = 16 };
enum class SliceKind { None, ExplicitArch, HostArch, AllArchs };

struct SummarySizes {
  uint64_t Text = 0;
  uint64_t Data = 0;
  uint64_t Bss = 0;
};

struct MachOSegmentSizes {
  uint64_t Text = 0;
  uint64_t Data = 0;
  uint64_t Objc = 0;
  uint64_t Others = 0;
};

struct ObjectContext {
  StringRef ArchiveName;
  StringRef ArchName;
  SliceKind Slice = SliceKind::None;
  bool ShowMultiple = false;
  bool ShowArchInArchive = false;
};
} // namespace

static bool ArchAll = false;
static std::vector<StringRef> ArchFlags;
static bool ELFCommons;
static bool DarwinLongFormat;
static RadixTy Radix = RadixTy::decimal;
static bool TotalSizes;
static bool ExcludePageZero = false;

static std::string ToolName;

// States
static bool HadError = false;
static bool MoreThanOneFile = false;

static void error(const Twine &Message, StringRef File = "") {
  HadError = true;
  if (File.empty())
    WithColor::error(errs(), ToolName) << Message << '\n';
  else
    WithColor::error(errs(), ToolName)
        << "'" << File << "': " << Message << '\n';
}

// This version of error() prints the archive name and member name, for example:
// "libx.a(foo.o)" after the ToolName before the error message.  It sets
// HadError but returns allowing the code to move on to other archive members.
static void error(llvm::Error E, StringRef FileName, const Archive::Child &C,
                  StringRef ArchitectureName = StringRef()) {
  HadError = true;
  WithColor::error(errs(), ToolName) << "'" << FileName << "'";

  Expected<StringRef> NameOrErr = C.getName();
  // TODO: if we have a error getting the name then it would be nice to print
  // the index of which archive member this is and or its offset in the
  // archive instead of "???" as the name.
  if (!NameOrErr) {
    consumeError(NameOrErr.takeError());
    errs() << "(" << "???" << ")";
  } else
    errs() << "(" << NameOrErr.get() << ")";

  if (!ArchitectureName.empty())
    errs() << " (for architecture " << ArchitectureName << ") ";

  std::string Buf;
  raw_string_ostream OS(Buf);
  logAllUnhandledErrors(std::move(E), OS);
  errs() << ": " << Buf << "\n";
}

// This version of error() prints the file name and which architecture slice it // is from, for example: "foo.o (for architecture i386)" after the ToolName
// before the error message.  It sets HadError but returns allowing the code to
// move on to other architecture slices.
static void error(llvm::Error E, StringRef FileName,
                  StringRef ArchitectureName = StringRef()) {
  HadError = true;
  WithColor::error(errs(), ToolName) << "'" << FileName << "'";

  if (!ArchitectureName.empty())
    errs() << " (for architecture " << ArchitectureName << ") ";

  std::string Buf;
  raw_string_ostream OS(Buf);
  logAllUnhandledErrors(std::move(E), OS);
  errs() << ": " << Buf << "\n";
}

/// Get the length of the string that represents @p num in Radix including the
/// leading 0x or 0 for hexadecimal and octal respectively.
static size_t getNumLengthAsString(uint64_t num) {
  APInt conv(64, num);
  SmallString<32> result;
  conv.toString(result, Radix, false, true);
  return result.size();
}

// Print Value right-aligned in Width characters using the configured Radix.
static void printRadixNumber(raw_ostream &OS, uint64_t Value, unsigned Width) {
  int W = static_cast<int>(Width);
  switch (Radix) {
  case octal:
    OS << format("%#*" PRIo64, W, Value);
    break;
  case decimal:
    OS << format("%*" PRIu64, W, Value);
    break;
  case hexadecimal:
    OS << format("%#*" PRIx64, W, Value);
    break;
  }
}

/// Remove unneeded ELF sections from calculation
static bool considerForSize(const ObjectFile &Obj, SectionRef Section) {
  if (!Obj.isELF())
    return true;
  switch (static_cast<ELFSectionRef>(Section).getType()) {
  case ELF::SHT_NULL:
  case ELF::SHT_SYMTAB:
    return false;
  case ELF::SHT_STRTAB:
  case ELF::SHT_REL:
  case ELF::SHT_RELA:
    return static_cast<ELFSectionRef>(Section).getFlags() & ELF::SHF_ALLOC;
  }
  return true;
}

/// Total size of all ELF common symbols
static Expected<uint64_t> getCommonSize(const ObjectFile &Obj) {
  uint64_t TotalCommons = 0;
  for (const auto &Sym : Obj.symbols()) {
    Expected<uint32_t> SymFlagsOrErr =
        Obj.getSymbolFlags(Sym.getRawDataRefImpl());
    if (!SymFlagsOrErr)
      return SymFlagsOrErr.takeError();
    if (*SymFlagsOrErr & SymbolRef::SF_Common)
      TotalCommons += Obj.getCommonSymbolSize(Sym.getRawDataRefImpl());
  }
  return TotalCommons;
}

// Calculate cumulative text, data, and bss sizes using Berkeley classification.
static SummarySizes getBerkeleySectionSizes(const ObjectFile &Obj) {
  SummarySizes Sizes;
  for (SectionRef Section : Obj.sections()) {
    uint64_t Size = Section.getSize();
    if (Section.isBerkeleyText())
      Sizes.Text += Size;
    else if (Section.isBerkeleyData())
      Sizes.Data += Size;
    else if (Section.isBSS())
      Sizes.Bss += Size;
  }
  return Sizes;
}

// Calculate standard Mach-O segment summary sizes.
static MachOSegmentSizes getMachOSegmentSizes(const MachOObjectFile &MachO) {
  MachOSegmentSizes Sizes;
  bool IsObject = MachO.getHeader().filetype == MachO::MH_OBJECT;
  auto AddSize = [&](StringRef SegmentName, uint64_t Size) {
    if (SegmentName == "__TEXT")
      Sizes.Text += Size;
    else if (SegmentName == "__DATA")
      Sizes.Data += Size;
    else if (SegmentName == "__OBJC")
      Sizes.Objc += Size;
    else if (IsObject || !ExcludePageZero || SegmentName != "__PAGEZERO")
      Sizes.Others += Size;
  };

  auto AddSegment = [&](const auto &Seg, auto GetSection) {
    if (IsObject) {
      for (unsigned J = 0; J < Seg.nsects; ++J) {
        auto Sec = GetSection(J);
        AddSize(StringRef(Sec.segname), Sec.size);
      }
    } else {
      AddSize(StringRef(Seg.segname), Seg.vmsize);
    }
  };

  for (const auto &Load : MachO.load_commands()) {
    if (Load.C.cmd == MachO::LC_SEGMENT_64) {
      AddSegment(MachO.getSegment64LoadCommand(Load),
                 [&](unsigned J) { return MachO.getSection64(Load, J); });
    } else if (Load.C.cmd == MachO::LC_SEGMENT) {
      AddSegment(MachO.getSegmentLoadCommand(Load),
                 [&](unsigned J) { return MachO.getSection(Load, J); });
    }
  }
  return Sizes;
}

namespace {
class SizePrinter {
protected:
  raw_ostream &OS;

public:
  explicit SizePrinter(raw_ostream &OS) : OS(OS) {}
  virtual ~SizePrinter() = default;

  virtual void printObject(const ObjectFile &Obj, const ObjectContext &Ctx) = 0;
  virtual void finish() {}
};

// Base printer for tabular summary formats (such as Berkeley).
class SummarySizePrinter : public SizePrinter {
protected:
  bool HeaderPrinted = false;
  uint64_t TotalText = 0;
  uint64_t TotalData = 0;
  uint64_t TotalBss = 0;
  uint64_t TotalSize = 0;

  virtual SummarySizes getSectionSizes(const ObjectFile &Obj) const = 0;
  virtual void printHeader() = 0;
  virtual void printRow(uint64_t Text, uint64_t Data, uint64_t Bss,
                        uint64_t Total) = 0;
  virtual void printTotals() {
    printRow(TotalText, TotalData, TotalBss, TotalSize);
    OS << "(TOTALS)\n";
  }

  void printObjectName(const ObjectFile &Obj, const ObjectContext &Ctx) {
    if (!Ctx.ArchiveName.empty()) {
      OS << Obj.getFileName() << " (ex " << Ctx.ArchiveName << ")\n";
    } else {
      OS << Obj.getFileName();
      if (!Ctx.ArchName.empty())
        OS << " (for architecture " << Ctx.ArchName << ')';
      OS << '\n';
    }
  }

public:
  using SizePrinter::SizePrinter;

  void printSummaryRow(const ObjectFile &Obj) {
    SummarySizes Sizes = getSectionSizes(Obj);
    if (ELFCommons) {
      if (Expected<uint64_t> CommonSizeOrErr = getCommonSize(Obj))
        Sizes.Bss += *CommonSizeOrErr;
      else {
        error(CommonSizeOrErr.takeError(), Obj.getFileName());
        return;
      }
    }

    uint64_t Total = Sizes.Text + Sizes.Data + Sizes.Bss;
    if (TotalSizes) {
      TotalText += Sizes.Text;
      TotalData += Sizes.Data;
      TotalBss += Sizes.Bss;
      TotalSize += Total;
    }

    if (!HeaderPrinted) {
      printHeader();
      HeaderPrinted = true;
    }

    printRow(Sizes.Text, Sizes.Data, Sizes.Bss, Total);
  }

  void printObject(const ObjectFile &Obj, const ObjectContext &Ctx) override {
    printSummaryRow(Obj);
    printObjectName(Obj, Ctx);
  }

  void finish() override {
    if (TotalSizes)
      printTotals();
  }
};

// Printer for Berkeley output format, including Darwin's default segment
// summary output for Mach-O files.
class BerkeleySizePrinter final : public SummarySizePrinter {
  bool HasMachOFiles = false;
  uint64_t TotalObjc = 0;
  uint64_t TotalOthers = 0;

  void printMachOObject(const MachOObjectFile &MachO,
                        const ObjectContext &Ctx) {
    MachOSegmentSizes Sizes = getMachOSegmentSizes(MachO);
    uint64_t Total = Sizes.Text + Sizes.Data + Sizes.Objc + Sizes.Others;
    HasMachOFiles = true;

    if (TotalSizes) {
      TotalText += Sizes.Text;
      TotalData += Sizes.Data;
      TotalObjc += Sizes.Objc;
      TotalOthers += Sizes.Others;
      TotalSize += Total;
    }

    if (!HeaderPrinted) {
      OS << "__TEXT\t__DATA\t__OBJC\tothers\tdec\thex\n";
      HeaderPrinted = true;
    }

    OS << Sizes.Text << '\t' << Sizes.Data << '\t' << Sizes.Objc << '\t'
       << Sizes.Others << '\t' << Total << '\t' << format("%" PRIx64, Total)
       << '\t';

    if (!Ctx.ArchiveName.empty()) {
      OS << Ctx.ArchiveName << '(' << MachO.getFileName() << ')';
      if (Ctx.ShowArchInArchive && !Ctx.ArchName.empty())
        OS << " (for architecture " << Ctx.ArchName << ')';
    } else if (Ctx.ShowMultiple) {
      OS << MachO.getFileName();
      if (!Ctx.ArchName.empty())
        OS << " (for architecture " << Ctx.ArchName << ')';
    }
    OS << '\n';
  }

protected:
  SummarySizes getSectionSizes(const ObjectFile &Obj) const override {
    return getBerkeleySectionSizes(Obj);
  }

  void printHeader() override {
    OS << "   text\t"
          "   data\t"
          "    bss\t"
          "    "
       << (Radix == octal ? "oct" : "dec")
       << "\t"
          "    hex\t"
          "filename\n";
  }

  void printRow(uint64_t Text, uint64_t Data, uint64_t Bss,
                uint64_t Total) override {
    printRadixNumber(OS, Text, 7);
    OS << '\t';
    printRadixNumber(OS, Data, 7);
    OS << '\t';
    printRadixNumber(OS, Bss, 7);
    OS << '\t';
    if (Radix == octal)
      OS << format("%7" PRIo64 "\t", Total);
    else
      OS << format("%7" PRIu64 "\t", Total);
    OS << format("%7" PRIx64 "\t", Total);
  }

  void printTotals() override {
    if (HasMachOFiles) {
      // Darwin format totals: __TEXT __DATA __OBJC others dec hex
      OS << TotalText << '\t' << TotalData << '\t' << TotalObjc << '\t'
         << TotalOthers << '\t' << TotalSize << '\t'
         << format("%" PRIx64, TotalSize) << "\t(TOTALS)\n";
    } else {
      SummarySizePrinter::printTotals();
    }
  }

public:
  using SummarySizePrinter::SummarySizePrinter;

  void printObject(const ObjectFile &Obj, const ObjectContext &Ctx) override {
    if (const auto *MachO = dyn_cast<MachOObjectFile>(&Obj)) {
      printMachOObject(*MachO, Ctx);
      return;
    }
    SummarySizePrinter::printObject(Obj, Ctx);
  }
};

// Printer for System V output format.
class SysVSizePrinter final : public SizePrinter {
public:
  using SizePrinter::SizePrinter;

  void printObject(const ObjectFile &Obj, const ObjectContext &Ctx) override {
    if (!Ctx.ArchiveName.empty())
      OS << Obj.getFileName() << "   (ex " << Ctx.ArchiveName << "):\n";
    else
      OS << Obj.getFileName() << "  :\n";

    // Run two passes over all sections. The first gets the lengths needed for
    // formatting the output. The second actually does the output.
    uint64_t Total = 0;
    size_t MaxNameLen = strlen("section");
    size_t MaxSizeLen = strlen("size");
    size_t MaxAddrLen = strlen("addr");
    for (SectionRef Section : Obj.sections()) {
      if (!considerForSize(Obj, Section))
        continue;
      uint64_t Size = Section.getSize();
      Total += Size;

      Expected<StringRef> NameOrErr = Section.getName();
      if (!NameOrErr) {
        error(NameOrErr.takeError(), Obj.getFileName());
        return;
      }

      uint64_t Addr = Section.getAddress();
      MaxNameLen = std::max(MaxNameLen, NameOrErr->size());
      MaxSizeLen = std::max(MaxSizeLen, getNumLengthAsString(Size));
      MaxAddrLen = std::max(MaxAddrLen, getNumLengthAsString(Addr));
    }

    // Add extra padding.
    MaxNameLen += 2;
    MaxSizeLen += 2;
    MaxAddrLen += 2;

    // Print header.
    OS << left_justify("section", MaxNameLen) << ' '
       << right_justify("size", MaxSizeLen) << ' '
       << right_justify("addr", MaxAddrLen) << '\n';

    // Print each section.
    for (SectionRef Section : Obj.sections()) {
      if (!considerForSize(Obj, Section))
        continue;

      Expected<StringRef> NameOrErr = Section.getName();
      if (!NameOrErr) {
        error(NameOrErr.takeError(), Obj.getFileName());
        return;
      }

      OS << left_justify(*NameOrErr, MaxNameLen) << ' ';
      printRadixNumber(OS, Section.getSize(), MaxSizeLen);
      OS << ' ';
      printRadixNumber(OS, Section.getAddress(), MaxAddrLen);
      OS << '\n';
    }

    if (ELFCommons) {
      if (Expected<uint64_t> CommonSizeOrErr = getCommonSize(Obj)) {
        Total += *CommonSizeOrErr;
        OS << left_justify("*COM*", MaxNameLen) << ' ';
        printRadixNumber(OS, *CommonSizeOrErr, MaxSizeLen);
        OS << ' ';
        printRadixNumber(OS, 0, MaxAddrLen);
        OS << '\n';
      } else {
        error(CommonSizeOrErr.takeError(), Obj.getFileName());
        return;
      }
    }

    // Print total.
    OS << left_justify("Total", MaxNameLen) << ' ';
    printRadixNumber(OS, Total, MaxSizeLen);
    OS << "\n\n\n";
  }
};

// Printer for Darwin output format (darwin's size(1) -m output), falling back
// to Berkeley format for non-Mach-O inputs.
class DarwinSizePrinter final : public SizePrinter {
  BerkeleySizePrinter FallbackPrinter;

  static void printSize(raw_ostream &OS, uint64_t Value) {
    switch (Radix) {
    case octal:
      OS << format("%" PRIo64, Value);
      break;
    case decimal:
      OS << format("%" PRIu64, Value);
      break;
    case hexadecimal:
      OS << format("0x%" PRIx64, Value);
      break;
    }
  }

  void printMachOSectionSizes(const MachOObjectFile &MachO) {
    uint32_t Filetype = MachO.getHeader().filetype;
    uint64_t Total = 0;

    auto PrintSegment = [&](const auto &Seg, auto GetSection) {
      OS << "Segment " << Seg.segname << ": ";
      printSize(OS, Seg.vmsize);
      if (DarwinLongFormat)
        OS << " (vmaddr 0x"
           << format("%" PRIx64, static_cast<uint64_t>(Seg.vmaddr))
           << " fileoff " << Seg.fileoff << ')';
      OS << '\n';
      Total += Seg.vmsize;

      uint64_t SecTotal = 0;
      for (unsigned J = 0; J < Seg.nsects; ++J) {
        auto Sec = GetSection(J);
        if (Filetype == MachO::MH_OBJECT)
          OS << "\tSection (" << format("%.16s", Sec.segname) << ", "
             << format("%.16s", Sec.sectname) << "): ";
        else
          OS << "\tSection " << format("%.16s", Sec.sectname) << ": ";
        printSize(OS, Sec.size);
        if (DarwinLongFormat)
          OS << " (addr 0x"
             << format("%" PRIx64, static_cast<uint64_t>(Sec.addr))
             << " offset " << Sec.offset << ')';
        OS << '\n';
        SecTotal += Sec.size;
      }
      if (Seg.nsects != 0) {
        OS << "\ttotal ";
        printSize(OS, SecTotal);
        OS << '\n';
      }
    };

    for (const auto &Load : MachO.load_commands()) {
      if (Load.C.cmd == MachO::LC_SEGMENT_64) {
        PrintSegment(MachO.getSegment64LoadCommand(Load),
                     [&](unsigned J) { return MachO.getSection64(Load, J); });
      } else if (Load.C.cmd == MachO::LC_SEGMENT) {
        PrintSegment(MachO.getSegmentLoadCommand(Load),
                     [&](unsigned J) { return MachO.getSection(Load, J); });
      }
    }
    OS << "total ";
    printSize(OS, Total);
    OS << '\n';
  }

public:
  explicit DarwinSizePrinter(raw_ostream &OS)
      : SizePrinter(OS), FallbackPrinter(OS) {}

  void printObject(const ObjectFile &Obj, const ObjectContext &Ctx) override {
    const auto *MachO = dyn_cast<MachOObjectFile>(&Obj);
    if (!MachO) {
      if (Ctx.Slice == SliceKind::None)
        FallbackPrinter.printObject(Obj, Ctx);
      else
        FallbackPrinter.printSummaryRow(Obj);
      return;
    }

    if (!Ctx.ArchiveName.empty()) {
      OS << Ctx.ArchiveName << '(' << MachO->getFileName() << ')';
      if (!Ctx.ArchName.empty())
        OS << " (for architecture " << Ctx.ArchName << ')';
      OS << ":\n";
    } else if (Ctx.Slice == SliceKind::AllArchs) {
      if (Ctx.ShowMultiple)
        OS << MachO->getFileName() << " (for architecture " << Ctx.ArchName
           << "):";
      OS << '\n';
    } else if (Ctx.ShowMultiple) {
      OS << MachO->getFileName();
      if (!Ctx.ArchName.empty())
        OS << " (for architecture " << Ctx.ArchName << ')';
      OS << (Ctx.Slice == SliceKind::ExplicitArch ? ": \n" : ":\n");
    }

    printMachOSectionSizes(*MachO);
  }
};
} // namespace

static std::unique_ptr<SizePrinter> createSizePrinter(OutputFormatTy Format,
                                                      raw_ostream &OS) {
  switch (Format) {
  case berkeley:
    return std::make_unique<BerkeleySizePrinter>(OS);
  case sysv:
    return std::make_unique<SysVSizePrinter>(OS);
  case darwin:
    return std::make_unique<DarwinSizePrinter>(OS);
  }
  llvm_unreachable("Unknown output format");
}

/// Checks to see if the @p Obj ObjectFile is a Mach-O file and if it is and
/// there is a list of architecture flags specified then check to make sure this
/// Mach-O file is one of those architectures or all architectures was
/// specified.  If not then an error is generated and this routine returns
/// false.  Else it returns true.
static bool checkMachOAndArchFlags(const ObjectFile &Obj, StringRef Filename) {
  const auto *MachO = dyn_cast<MachOObjectFile>(&Obj);

  if (!MachO || ArchAll || ArchFlags.empty())
    return true;

  Triple T;
  if (MachO->is64Bit()) {
    MachO::mach_header_64 H64 = MachO->MachOObjectFile::getHeader64();
    T = MachOObjectFile::getArchTriple(H64.cputype, H64.cpusubtype);
  } else {
    MachO::mach_header H = MachO->MachOObjectFile::getHeader();
    T = MachOObjectFile::getArchTriple(H.cputype, H.cpusubtype);
  }
  if (!is_contained(ArchFlags, T.getArchName())) {
    error("no architecture specified", Filename);
    return false;
  }
  return true;
}

// Iterate over each member of Archive A and print its sizes.
static void printArchiveMembers(Archive &A, StringRef File,
                                SizePrinter &Printer, const ObjectContext &Ctx,
                                StringRef ErrorArchName = StringRef(),
                                bool CheckArchFlags = false) {
  Error Err = Error::success();
  for (const Archive::Child &C : A.children(Err)) {
    Expected<std::unique_ptr<Binary>> ChildOrErr = C.getAsBinary();
    if (!ChildOrErr) {
      if (auto E = isNotObjectErrorInvalidFileType(ChildOrErr.takeError()))
        error(std::move(E), A.getFileName(), C, ErrorArchName);
      continue;
    }
    if (const auto *Obj = dyn_cast<ObjectFile>(ChildOrErr->get())) {
      if (CheckArchFlags && !checkMachOAndArchFlags(*Obj, File))
        break;
      Printer.printObject(*Obj, Ctx);
    }
  }
  if (Err)
    error(std::move(Err), A.getFileName());
}

// Print a single slice of a Mach-O universal binary. Returns false if an
// unrecoverable error occurred and processing of the file should stop.
static bool
printUniversalSlice(const MachOUniversalBinary::ObjectForArch &Slice,
                    StringRef File, SizePrinter &Printer, SliceKind Kind,
                    bool MultipleArchs) {
  std::string ArchName = Slice.getArchFlagName();
  StringRef ErrorArchName = MultipleArchs ? StringRef(ArchName) : StringRef();
  bool ShowMultiple = MoreThanOneFile || MultipleArchs;

  Expected<std::unique_ptr<MachOObjectFile>> UO = Slice.getAsObjectFile();
  if (UO) {
    ObjectContext Ctx;
    Ctx.ArchName = ArchName;
    Ctx.Slice = Kind;
    Ctx.ShowMultiple = ShowMultiple;
    Printer.printObject(**UO, Ctx);
    return true;
  }
  if (auto E = isNotObjectErrorInvalidFileType(UO.takeError())) {
    error(std::move(E), File, ErrorArchName);
    return false;
  }
  Expected<std::unique_ptr<Archive>> AOrErr = Slice.getAsArchive();
  if (!AOrErr) {
    consumeError(AOrErr.takeError());
    error("mach-o universal file for architecture " + StringRef(ArchName) +
              " is not a mach-o file or an archive file",
          File);
    return true;
  }

  ObjectContext Ctx;
  Ctx.ArchiveName = (*AOrErr)->getFileName();
  Ctx.ArchName = ArchName;
  Ctx.Slice = Kind;
  Ctx.ShowMultiple = ShowMultiple;
  Ctx.ShowArchInArchive = (Kind == SliceKind::AllArchs) || MultipleArchs;
  printArchiveMembers(**AOrErr, File, Printer, Ctx, ErrorArchName,
                      /*CheckArchFlags=*/false);
  return true;
}

/// Print the section sizes for @p File. If @p File is an archive, print the
/// section sizes for each archive member.
static void printFileSectionSizes(StringRef File, SizePrinter &Printer) {
  // Attempt to open the binary.
  Expected<OwningBinary<Binary>> BinaryOrErr = createBinary(File);
  if (!BinaryOrErr) {
    error(BinaryOrErr.takeError(), File);
    return;
  }
  Binary &Bin = *BinaryOrErr.get().getBinary();

  if (auto *A = dyn_cast<Archive>(&Bin)) {
    ObjectContext Ctx;
    Ctx.ArchiveName = A->getFileName();
    Ctx.ShowMultiple = MoreThanOneFile;
    printArchiveMembers(*A, File, Printer, Ctx, /*ErrorArchName=*/StringRef(),
                        /*CheckArchFlags=*/true);
  } else if (auto *UB = dyn_cast<MachOUniversalBinary>(&Bin)) {
    // If we have a list of architecture flags specified dump only those.
    if (!ArchAll && !ArchFlags.empty()) {
      bool MultipleArchs = ArchFlags.size() > 1;
      for (StringRef ArchFlag : ArchFlags) {
        bool ArchFound = false;
        for (const auto &Slice : UB->objects()) {
          if (ArchFlag == Slice.getArchFlagName()) {
            ArchFound = true;
            if (!printUniversalSlice(Slice, File, Printer,
                                     SliceKind::ExplicitArch, MultipleArchs))
              return;
          }
        }
        if (!ArchFound) {
          error("file does not contain architecture " + ArchFlag, File);
          return;
        }
      }
      return;
    }
    // No architecture flags were specified so if this contains a slice that
    // matches the host architecture dump only that.
    if (!ArchAll) {
      StringRef HostArchName = MachOObjectFile::getHostArch().getArchName();
      for (const auto &Slice : UB->objects()) {
        if (HostArchName == Slice.getArchFlagName()) {
          printUniversalSlice(Slice, File, Printer, SliceKind::HostArch,
                              /*MultipleArchs=*/false);
          return;
        }
      }
    }
    // Either all architectures have been specified or none have been specified
    // and this does not contain the host architecture so dump all the slices.
    bool MoreThanOneArch = UB->getNumberOfObjects() > 1;
    for (const auto &Slice : UB->objects()) {
      if (!printUniversalSlice(Slice, File, Printer, SliceKind::AllArchs,
                               /*MultipleArchs=*/MoreThanOneArch))
        return;
    }
  } else if (const auto *Obj = dyn_cast<ObjectFile>(&Bin)) {
    if (!checkMachOAndArchFlags(*Obj, File))
      return;
    ObjectContext Ctx;
    Ctx.ShowMultiple = MoreThanOneFile;
    Printer.printObject(*Obj, Ctx);
  } else {
    error("unsupported file type", File);
  }
}

int llvm_size_main(int argc, char **argv, const llvm::ToolContext &) {
  BumpPtrAllocator A;
  StringSaver Saver(A);
  SizeOptTable Tbl;
  ToolName = argv[0];
  opt::InputArgList Args =
      Tbl.parseArgs(argc, argv, OPT_UNKNOWN, Saver, [&](StringRef Msg) {
        error(Msg);
        exit(1);
      });
  if (Args.hasArg(OPT_help)) {
    Tbl.printHelp(
        outs(),
        (Twine(ToolName) + " [options] <input object files>").str().c_str(),
        "LLVM object size dumper");
    // TODO Replace this with OptTable API once it adds extrahelp support.
    outs() << "\nPass @FILE as argument to read options from FILE.\n";
    return 0;
  }
  if (Args.hasArg(OPT_version)) {
    outs() << ToolName << '\n';
    cl::PrintVersionMessage();
    return 0;
  }

  ELFCommons = Args.hasArg(OPT_common);
  DarwinLongFormat = Args.hasArg(OPT_l);
  ExcludePageZero = Args.hasArg(OPT_exclude_pagezero);
  TotalSizes = Args.hasArg(OPT_totals);

  OutputFormatTy OutputFormat = berkeley;
  StringRef V = Args.getLastArgValue(OPT_format_EQ, "berkeley");
  if (V == "berkeley")
    OutputFormat = berkeley;
  else if (V == "darwin")
    OutputFormat = darwin;
  else if (V == "sysv")
    OutputFormat = sysv;
  else
    error("--format value should be one of: 'berkeley', 'darwin', 'sysv'");
  V = Args.getLastArgValue(OPT_radix_EQ, "10");
  if (V == "8")
    Radix = RadixTy::octal;
  else if (V == "10")
    Radix = RadixTy::decimal;
  else if (V == "16")
    Radix = RadixTy::hexadecimal;
  else
    error("--radix value should be one of: 8, 10, 16 ");

  for (const auto *A : Args.filtered(OPT_arch_EQ)) {
    SmallVector<StringRef, 2> Values;
    llvm::SplitString(A->getValue(), Values, ",");
    for (StringRef V : Values) {
      if (V == "all")
        ArchAll = true;
      else if (MachOObjectFile::isValidArch(V))
        ArchFlags.push_back(V);
      else {
        outs() << ToolName << ": for the -arch option: Unknown architecture "
               << "named '" << V << "'";
        return 1;
      }
    }
  }

  std::vector<std::string> InputFilenames = Args.getAllArgValues(OPT_INPUT);
  if (InputFilenames.empty())
    InputFilenames.push_back("a.out");

  MoreThanOneFile = InputFilenames.size() > 1;
  std::unique_ptr<SizePrinter> Printer =
      createSizePrinter(OutputFormat, outs());
  for (StringRef File : InputFilenames)
    printFileSectionSizes(File, *Printer);
  Printer->finish();

  if (HadError)
    return 1;
  return 0;
}
