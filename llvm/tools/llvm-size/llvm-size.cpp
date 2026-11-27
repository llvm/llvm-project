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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
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
static OutputFormatTy OutputFormat;
static bool DarwinLongFormat;
static RadixTy Radix = RadixTy::decimal;
static bool TotalSizes;
static bool HasMachOFiles = false;
static bool ExcludePageZero = false;

static std::vector<std::string> InputFilenames;

static std::string ToolName;

// States
static bool HadError = false;
static bool BerkeleyHeaderPrinted = false;
static bool MoreThanOneFile = false;
static uint64_t TotalObjectText = 0;
static uint64_t TotalObjectData = 0;
static uint64_t TotalObjectBss = 0;
static uint64_t TotalObjectTotal = 0;

// Darwin-specific totals
static uint64_t TotalObjectObjc = 0;
static uint64_t TotalObjectOthers = 0;

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

// Print a Berkeley summary row (text, data, bss, dec/oct, hex).
static void printBerkeleyRow(raw_ostream &OS, uint64_t Text, uint64_t Data,
                             uint64_t Bss, uint64_t Total) {
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

/// Print the size of each Mach-O segment and section in @p MachO.
///
/// This is when used when @c OutputFormat is darwin and produces the same
/// output as darwin's size(1) -m output.
static void printDarwinSectionSizes(const MachOObjectFile &MachO) {
  auto PrintSize = [](raw_ostream &OS, uint64_t Value) {
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
  };

  uint32_t Filetype = MachO.getHeader().filetype;
  uint64_t Total = 0;

  auto PrintSegment = [&](const auto &Seg, auto GetSection) {
    outs() << "Segment " << Seg.segname << ": ";
    PrintSize(outs(), Seg.vmsize);
    if (DarwinLongFormat)
      outs() << " (vmaddr 0x"
             << format("%" PRIx64, static_cast<uint64_t>(Seg.vmaddr))
             << " fileoff " << Seg.fileoff << ')';
    outs() << '\n';
    Total += Seg.vmsize;

    uint64_t SecTotal = 0;
    for (unsigned J = 0; J < Seg.nsects; ++J) {
      auto Sec = GetSection(J);
      if (Filetype == MachO::MH_OBJECT)
        outs() << "\tSection (" << format("%.16s", Sec.segname) << ", "
               << format("%.16s", Sec.sectname) << "): ";
      else
        outs() << "\tSection " << format("%.16s", Sec.sectname) << ": ";
      PrintSize(outs(), Sec.size);
      if (DarwinLongFormat)
        outs() << " (addr 0x"
               << format("%" PRIx64, static_cast<uint64_t>(Sec.addr))
               << " offset " << Sec.offset << ')';
      outs() << '\n';
      SecTotal += Sec.size;
    }
    if (Seg.nsects != 0) {
      outs() << "\ttotal ";
      PrintSize(outs(), SecTotal);
      outs() << '\n';
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
  outs() << "total ";
  PrintSize(outs(), Total);
  outs() << '\n';
}

/// Print the summary sizes of the standard Mach-O segments in @p MachO.
///
/// This is when used when @c OutputFormat is berkeley with a Mach-O file and
/// produces the same output as darwin's size(1) default output.
static void printDarwinSegmentSizes(const MachOObjectFile &MachO) {
  MachOSegmentSizes Sizes = getMachOSegmentSizes(MachO);
  uint64_t Total = Sizes.Text + Sizes.Data + Sizes.Objc + Sizes.Others;
  HasMachOFiles = true;

  if (TotalSizes) {
    TotalObjectText += Sizes.Text;
    TotalObjectData += Sizes.Data;
    TotalObjectObjc += Sizes.Objc;
    TotalObjectOthers += Sizes.Others;
    TotalObjectTotal += Total;
  }

  if (!BerkeleyHeaderPrinted) {
    outs() << "__TEXT\t__DATA\t__OBJC\tothers\tdec\thex\n";
    BerkeleyHeaderPrinted = true;
  }
  outs() << Sizes.Text << '\t' << Sizes.Data << '\t' << Sizes.Objc << '\t'
         << Sizes.Others << '\t' << Total << '\t' << format("%" PRIx64, Total)
         << '\t';
}

/// Print the size of each section in @p Obj.
///
/// The format used is determined by @c OutputFormat and @c Radix.
static void printObjectSectionSizes(const ObjectFile &Obj) {
  // If OutputFormat is darwin and we have a MachOObjectFile print as darwin's
  // size(1) -m output, else if OutputFormat is darwin and not a Mach-O object
  // let it fall through to OutputFormat berkeley.
  const auto *MachO = dyn_cast<MachOObjectFile>(&Obj);
  if (OutputFormat == darwin && MachO) {
    printDarwinSectionSizes(*MachO);
  } else if (MachO && OutputFormat == berkeley) {
    // Print as darwin's default berkeley format for Mach-O files.
    printDarwinSegmentSizes(*MachO);
  } else if (OutputFormat == sysv) {
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
    outs() << left_justify("section", MaxNameLen) << ' '
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

      outs() << left_justify(*NameOrErr, MaxNameLen) << ' ';
      printRadixNumber(outs(), Section.getSize(), MaxSizeLen);
      outs() << ' ';
      printRadixNumber(outs(), Section.getAddress(), MaxAddrLen);
      outs() << '\n';
    }

    if (ELFCommons) {
      if (Expected<uint64_t> CommonSizeOrErr = getCommonSize(Obj)) {
        Total += *CommonSizeOrErr;
        outs() << left_justify("*COM*", MaxNameLen) << ' ';
        printRadixNumber(outs(), *CommonSizeOrErr, MaxSizeLen);
        outs() << ' ';
        printRadixNumber(outs(), 0, MaxAddrLen);
        outs() << '\n';
      } else {
        error(CommonSizeOrErr.takeError(), Obj.getFileName());
        return;
      }
    }

    // Print total.
    outs() << left_justify("Total", MaxNameLen) << ' ';
    printRadixNumber(outs(), Total, MaxSizeLen);
    outs() << "\n\n\n";
  } else {
    SummarySizes Sizes = getBerkeleySectionSizes(Obj);
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
      TotalObjectText += Sizes.Text;
      TotalObjectData += Sizes.Data;
      TotalObjectBss += Sizes.Bss;
      TotalObjectTotal += Total;
    }

    if (!BerkeleyHeaderPrinted) {
      outs() << "   text\t"
                "   data\t"
                "    bss\t"
                "    "
             << (Radix == octal ? "oct" : "dec")
             << "\t"
                "    hex\t"
                "filename\n";
      BerkeleyHeaderPrinted = true;
    }

    printBerkeleyRow(outs(), Sizes.Text, Sizes.Data, Sizes.Bss, Total);
  }
}

// Print Obj along with any format-specific prefix or suffix headers described
// by Ctx.
static void printObject(const ObjectFile &Obj, const ObjectContext &Ctx) {
  const auto *MachO = dyn_cast<MachOObjectFile>(&Obj);
  if (OutputFormat == sysv) {
    if (!Ctx.ArchiveName.empty())
      outs() << Obj.getFileName() << "   (ex " << Ctx.ArchiveName << "):\n";
    else
      outs() << Obj.getFileName() << "  :\n";
  } else if (MachO && OutputFormat == darwin) {
    if (!Ctx.ArchiveName.empty()) {
      outs() << Ctx.ArchiveName << '(' << MachO->getFileName() << ')';
      if (!Ctx.ArchName.empty())
        outs() << " (for architecture " << Ctx.ArchName << ')';
      outs() << ":\n";
    } else if (Ctx.Slice == SliceKind::AllArchs) {
      if (Ctx.ShowMultiple)
        outs() << MachO->getFileName() << " (for architecture " << Ctx.ArchName
               << "):";
      outs() << '\n';
    } else if (Ctx.ShowMultiple) {
      outs() << MachO->getFileName();
      if (!Ctx.ArchName.empty())
        outs() << " (for architecture " << Ctx.ArchName << ')';
      outs() << (Ctx.Slice == SliceKind::ExplicitArch ? ": \n" : ":\n");
    }
  }

  printObjectSectionSizes(Obj);

  if (!MachO && OutputFormat == darwin && Ctx.Slice == SliceKind::None) {
    if (!Ctx.ArchiveName.empty())
      outs() << Obj.getFileName() << " (ex " << Ctx.ArchiveName << ")\n";
    else
      outs() << Obj.getFileName() << '\n';
  }
  if (OutputFormat == berkeley) {
    if (MachO) {
      if (!Ctx.ArchiveName.empty()) {
        outs() << Ctx.ArchiveName << '(' << MachO->getFileName() << ')';
        if (Ctx.ShowArchInArchive && !Ctx.ArchName.empty())
          outs() << " (for architecture " << Ctx.ArchName << ')';
      } else if (Ctx.ShowMultiple) {
        outs() << MachO->getFileName();
        if (!Ctx.ArchName.empty())
          outs() << " (for architecture " << Ctx.ArchName << ')';
      }
      outs() << '\n';
    } else if (!Ctx.ArchiveName.empty()) {
      outs() << Obj.getFileName() << " (ex " << Ctx.ArchiveName << ")\n";
    } else {
      outs() << Obj.getFileName();
      if (!Ctx.ArchName.empty())
        outs() << " (for architecture " << Ctx.ArchName << ')';
      outs() << '\n';
    }
  }
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
                                const ObjectContext &Ctx,
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
      printObject(*Obj, Ctx);
    }
  }
  if (Err)
    error(std::move(Err), A.getFileName());
}

// Print a single slice of a Mach-O universal binary. Returns false if an
// unrecoverable error occurred and processing of the file should stop.
static bool
printUniversalSlice(const MachOUniversalBinary::ObjectForArch &Slice,
                    StringRef File, SliceKind Kind, bool MultipleArchs) {
  std::string ArchName = Slice.getArchFlagName();
  StringRef ErrorArchName = MultipleArchs ? StringRef(ArchName) : StringRef();
  bool ShowMultiple = MoreThanOneFile || MultipleArchs;

  Expected<std::unique_ptr<MachOObjectFile>> UO = Slice.getAsObjectFile();
  if (UO) {
    ObjectContext Ctx;
    Ctx.ArchName = ArchName;
    Ctx.Slice = Kind;
    Ctx.ShowMultiple = ShowMultiple;
    printObject(**UO, Ctx);
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
  printArchiveMembers(**AOrErr, File, Ctx, ErrorArchName,
                      /*CheckArchFlags=*/false);
  return true;
}

/// Print the section sizes for @p File. If @p File is an archive, print the
/// section sizes for each archive member.
static void printFileSectionSizes(StringRef File) {
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
    printArchiveMembers(*A, File, Ctx, /*ErrorArchName=*/StringRef(),
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
            if (!printUniversalSlice(Slice, File, SliceKind::ExplicitArch,
                                     MultipleArchs))
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
          printUniversalSlice(Slice, File, SliceKind::HostArch,
                              /*MultipleArchs=*/false);
          return;
        }
      }
    }
    // Either all architectures have been specified or none have been specified
    // and this does not contain the host architecture so dump all the slices.
    bool MoreThanOneArch = UB->getNumberOfObjects() > 1;
    for (const auto &Slice : UB->objects()) {
      if (!printUniversalSlice(Slice, File, SliceKind::AllArchs,
                               /*MultipleArchs=*/MoreThanOneArch))
        return;
    }
  } else if (const auto *Obj = dyn_cast<ObjectFile>(&Bin)) {
    if (!checkMachOAndArchFlags(*Obj, File))
      return;
    ObjectContext Ctx;
    Ctx.ShowMultiple = MoreThanOneFile;
    printObject(*Obj, Ctx);
  } else {
    error("unsupported file type", File);
  }
}

static void printBerkeleyTotals() {
  if (HasMachOFiles) {
    // Darwin format totals: __TEXT __DATA __OBJC others dec hex
    outs() << TotalObjectText << '\t' << TotalObjectData << '\t'
           << TotalObjectObjc << '\t' << TotalObjectOthers << '\t'
           << TotalObjectTotal << '\t' << format("%" PRIx64, TotalObjectTotal)
           << "\t(TOTALS)\n";
  } else {
    printBerkeleyRow(outs(), TotalObjectText, TotalObjectData, TotalObjectBss,
                     TotalObjectTotal);
    outs() << "(TOTALS)\n";
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

  InputFilenames = Args.getAllArgValues(OPT_INPUT);
  if (InputFilenames.empty())
    InputFilenames.push_back("a.out");

  MoreThanOneFile = InputFilenames.size() > 1;
  llvm::for_each(InputFilenames, printFileSectionSizes);
  if (OutputFormat == berkeley && TotalSizes)
    printBerkeleyTotals();

  if (HadError)
    return 1;
  return 0;
}
