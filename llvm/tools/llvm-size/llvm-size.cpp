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

/// Return the printing format for the Radix.
static const char *getRadixFmt() {
  switch (Radix) {
  case octal:
    return PRIo64;
  case decimal:
    return PRIu64;
  case hexadecimal:
    return PRIx64;
  }
  return nullptr;
}

/// Remove unneeded ELF sections from calculation
static bool considerForSize(ObjectFile *Obj, SectionRef Section) {
  if (!Obj->isELF())
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
static Expected<uint64_t> getCommonSize(ObjectFile *Obj) {
  uint64_t TotalCommons = 0;
  for (auto &Sym : Obj->symbols()) {
    Expected<uint32_t> SymFlagsOrErr =
        Obj->getSymbolFlags(Sym.getRawDataRefImpl());
    if (!SymFlagsOrErr)
      return SymFlagsOrErr.takeError();
    if (*SymFlagsOrErr & SymbolRef::SF_Common)
      TotalCommons += Obj->getCommonSymbolSize(Sym.getRawDataRefImpl());
  }
  return TotalCommons;
}

/// Print the size of each Mach-O segment and section in @p MachO.
///
/// This is when used when @c OutputFormat is darwin and produces the same
/// output as darwin's size(1) -m output.
static void printDarwinSectionSizes(MachOObjectFile *MachO) {
  std::string fmtbuf;
  raw_string_ostream fmt(fmtbuf);
  const char *radix_fmt = getRadixFmt();
  if (Radix == hexadecimal)
    fmt << "0x";
  fmt << "%" << radix_fmt;

  uint32_t Filetype = MachO->getHeader().filetype;

  uint64_t total = 0;
  for (const auto &Load : MachO->load_commands()) {
    if (Load.C.cmd == MachO::LC_SEGMENT_64) {
      MachO::segment_command_64 Seg = MachO->getSegment64LoadCommand(Load);
      outs() << "Segment " << Seg.segname << ": "
             << format(fmtbuf.c_str(), Seg.vmsize);
      if (DarwinLongFormat)
        outs() << " (vmaddr 0x" << format("%" PRIx64, Seg.vmaddr) << " fileoff "
               << Seg.fileoff << ")";
      outs() << "\n";
      total += Seg.vmsize;
      uint64_t sec_total = 0;
      for (unsigned J = 0; J < Seg.nsects; ++J) {
        MachO::section_64 Sec = MachO->getSection64(Load, J);
        if (Filetype == MachO::MH_OBJECT)
          outs() << "\tSection (" << format("%.16s", &Sec.segname) << ", "
                 << format("%.16s", &Sec.sectname) << "): ";
        else
          outs() << "\tSection " << format("%.16s", &Sec.sectname) << ": ";
        outs() << format(fmtbuf.c_str(), Sec.size);
        if (DarwinLongFormat)
          outs() << " (addr 0x" << format("%" PRIx64, Sec.addr) << " offset "
                 << Sec.offset << ")";
        outs() << "\n";
        sec_total += Sec.size;
      }
      if (Seg.nsects != 0)
        outs() << "\ttotal " << format(fmtbuf.c_str(), sec_total) << "\n";
    } else if (Load.C.cmd == MachO::LC_SEGMENT) {
      MachO::segment_command Seg = MachO->getSegmentLoadCommand(Load);
      uint64_t Seg_vmsize = Seg.vmsize;
      outs() << "Segment " << Seg.segname << ": "
             << format(fmtbuf.c_str(), Seg_vmsize);
      if (DarwinLongFormat)
        outs() << " (vmaddr 0x" << format("%" PRIx32, Seg.vmaddr) << " fileoff "
               << Seg.fileoff << ")";
      outs() << "\n";
      total += Seg.vmsize;
      uint64_t sec_total = 0;
      for (unsigned J = 0; J < Seg.nsects; ++J) {
        MachO::section Sec = MachO->getSection(Load, J);
        if (Filetype == MachO::MH_OBJECT)
          outs() << "\tSection (" << format("%.16s", &Sec.segname) << ", "
                 << format("%.16s", &Sec.sectname) << "): ";
        else
          outs() << "\tSection " << format("%.16s", &Sec.sectname) << ": ";
        uint64_t Sec_size = Sec.size;
        outs() << format(fmtbuf.c_str(), Sec_size);
        if (DarwinLongFormat)
          outs() << " (addr 0x" << format("%" PRIx32, Sec.addr) << " offset "
                 << Sec.offset << ")";
        outs() << "\n";
        sec_total += Sec.size;
      }
      if (Seg.nsects != 0)
        outs() << "\ttotal " << format(fmtbuf.c_str(), sec_total) << "\n";
    }
  }
  outs() << "total " << format(fmtbuf.c_str(), total) << "\n";
}

/// Print the summary sizes of the standard Mach-O segments in @p MachO.
///
/// This is when used when @c OutputFormat is berkeley with a Mach-O file and
/// produces the same output as darwin's size(1) default output.
static void printDarwinSegmentSizes(MachOObjectFile *MachO) {
  uint64_t total_text = 0;
  uint64_t total_data = 0;
  uint64_t total_objc = 0;
  uint64_t total_others = 0;
  HasMachOFiles = true;
  for (const auto &Load : MachO->load_commands()) {
    if (Load.C.cmd == MachO::LC_SEGMENT_64) {
      MachO::segment_command_64 Seg = MachO->getSegment64LoadCommand(Load);
      if (MachO->getHeader().filetype == MachO::MH_OBJECT) {
        for (unsigned J = 0; J < Seg.nsects; ++J) {
          MachO::section_64 Sec = MachO->getSection64(Load, J);
          StringRef SegmentName = StringRef(Sec.segname);
          if (SegmentName == "__TEXT")
            total_text += Sec.size;
          else if (SegmentName == "__DATA")
            total_data += Sec.size;
          else if (SegmentName == "__OBJC")
            total_objc += Sec.size;
          else
            total_others += Sec.size;
        }
      } else {
        StringRef SegmentName = StringRef(Seg.segname);
        if (SegmentName == "__TEXT")
          total_text += Seg.vmsize;
        else if (SegmentName == "__DATA")
          total_data += Seg.vmsize;
        else if (SegmentName == "__OBJC")
          total_objc += Seg.vmsize;
        else if (!ExcludePageZero || SegmentName != "__PAGEZERO")
          total_others += Seg.vmsize;
      }
    } else if (Load.C.cmd == MachO::LC_SEGMENT) {
      MachO::segment_command Seg = MachO->getSegmentLoadCommand(Load);
      if (MachO->getHeader().filetype == MachO::MH_OBJECT) {
        for (unsigned J = 0; J < Seg.nsects; ++J) {
          MachO::section Sec = MachO->getSection(Load, J);
          StringRef SegmentName = StringRef(Sec.segname);
          if (SegmentName == "__TEXT")
            total_text += Sec.size;
          else if (SegmentName == "__DATA")
            total_data += Sec.size;
          else if (SegmentName == "__OBJC")
            total_objc += Sec.size;
          else
            total_others += Sec.size;
        }
      } else {
        StringRef SegmentName = StringRef(Seg.segname);
        if (SegmentName == "__TEXT")
          total_text += Seg.vmsize;
        else if (SegmentName == "__DATA")
          total_data += Seg.vmsize;
        else if (SegmentName == "__OBJC")
          total_objc += Seg.vmsize;
        else if (!ExcludePageZero || SegmentName != "__PAGEZERO")
          total_others += Seg.vmsize;
      }
    }
  }
  uint64_t total = total_text + total_data + total_objc + total_others;

  if (TotalSizes) {
    TotalObjectText += total_text;
    TotalObjectData += total_data;
    TotalObjectObjc += total_objc;
    TotalObjectOthers += total_others;
    TotalObjectTotal += total;
  }

  if (!BerkeleyHeaderPrinted) {
    outs() << "__TEXT\t__DATA\t__OBJC\tothers\tdec\thex\n";
    BerkeleyHeaderPrinted = true;
  }
  outs() << total_text << "\t" << total_data << "\t" << total_objc << "\t"
         << total_others << "\t" << total << "\t" << format("%" PRIx64, total)
         << "\t";
}

/// Print the size of each section in @p Obj.
///
/// The format used is determined by @c OutputFormat and @c Radix.
static void printObjectSectionSizes(ObjectFile *Obj) {
  uint64_t total = 0;
  std::string fmtbuf;
  raw_string_ostream fmt(fmtbuf);
  const char *radix_fmt = getRadixFmt();

  // If OutputFormat is darwin and we have a MachOObjectFile print as darwin's
  // size(1) -m output, else if OutputFormat is darwin and not a Mach-O object
  // let it fall through to OutputFormat berkeley.
  MachOObjectFile *MachO = dyn_cast<MachOObjectFile>(Obj);
  if (OutputFormat == darwin && MachO)
    printDarwinSectionSizes(MachO);
  // If we have a MachOObjectFile and the OutputFormat is berkeley print as
  // darwin's default berkeley format for Mach-O files.
  else if (MachO && OutputFormat == berkeley)
    printDarwinSegmentSizes(MachO);
  else if (OutputFormat == sysv) {
    // Run two passes over all sections. The first gets the lengths needed for
    // formatting the output. The second actually does the output.
    std::size_t max_name_len = strlen("section");
    std::size_t max_size_len = strlen("size");
    std::size_t max_addr_len = strlen("addr");
    for (const SectionRef &Section : Obj->sections()) {
      if (!considerForSize(Obj, Section))
        continue;
      uint64_t size = Section.getSize();
      total += size;

      Expected<StringRef> name_or_err = Section.getName();
      if (!name_or_err) {
        error(name_or_err.takeError(), Obj->getFileName());
        return;
      }

      uint64_t addr = Section.getAddress();
      max_name_len = std::max(max_name_len, name_or_err->size());
      max_size_len = std::max(max_size_len, getNumLengthAsString(size));
      max_addr_len = std::max(max_addr_len, getNumLengthAsString(addr));
    }

    // Add extra padding.
    max_name_len += 2;
    max_size_len += 2;
    max_addr_len += 2;

    // Setup header format.
    fmt << "%-" << max_name_len << "s "
        << "%" << max_size_len << "s "
        << "%" << max_addr_len << "s\n";

    // Print header
    outs() << format(fmtbuf.c_str(), static_cast<const char *>("section"),
                     static_cast<const char *>("size"),
                     static_cast<const char *>("addr"));
    fmtbuf.clear();

    // Setup per section format.
    fmt << "%-" << max_name_len << "s "
        << "%#" << max_size_len << radix_fmt << " "
        << "%#" << max_addr_len << radix_fmt << "\n";

    // Print each section.
    for (const SectionRef &Section : Obj->sections()) {
      if (!considerForSize(Obj, Section))
        continue;

      Expected<StringRef> name_or_err = Section.getName();
      if (!name_or_err) {
        error(name_or_err.takeError(), Obj->getFileName());
        return;
      }

      uint64_t size = Section.getSize();
      uint64_t addr = Section.getAddress();
      outs() << format(fmtbuf.c_str(), name_or_err->str().c_str(), size, addr);
    }

    if (ELFCommons) {
      if (Expected<uint64_t> CommonSizeOrErr = getCommonSize(Obj)) {
        total += *CommonSizeOrErr;
        outs() << format(fmtbuf.c_str(), std::string("*COM*").c_str(),
                         *CommonSizeOrErr, static_cast<uint64_t>(0));
      } else {
        error(CommonSizeOrErr.takeError(), Obj->getFileName());
        return;
      }
    }

    // Print total.
    fmtbuf.clear();
    fmt << "%-" << max_name_len << "s "
        << "%#" << max_size_len << radix_fmt << "\n";
    outs() << format(fmtbuf.c_str(), static_cast<const char *>("Total"), total)
           << "\n\n";
  } else {
    // The Berkeley format does not display individual section sizes. It
    // displays the cumulative size for each section type.
    uint64_t total_text = 0;
    uint64_t total_data = 0;
    uint64_t total_bss = 0;

    // Make one pass over the section table to calculate sizes.
    for (const SectionRef &Section : Obj->sections()) {
      uint64_t size = Section.getSize();
      bool isText = Section.isBerkeleyText();
      bool isData = Section.isBerkeleyData();
      bool isBSS = Section.isBSS();
      if (isText)
        total_text += size;
      else if (isData)
        total_data += size;
      else if (isBSS)
        total_bss += size;
    }

    if (ELFCommons) {
      if (Expected<uint64_t> CommonSizeOrErr = getCommonSize(Obj))
        total_bss += *CommonSizeOrErr;
      else {
        error(CommonSizeOrErr.takeError(), Obj->getFileName());
        return;
      }
    }

    total = total_text + total_data + total_bss;

    if (TotalSizes) {
      TotalObjectText += total_text;
      TotalObjectData += total_data;
      TotalObjectBss += total_bss;
      TotalObjectTotal += total;
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

    // Print result.
    fmt << "%#7" << radix_fmt << "\t"
        << "%#7" << radix_fmt << "\t"
        << "%#7" << radix_fmt << "\t";
    outs() << format(fmtbuf.c_str(), total_text, total_data, total_bss);
    fmtbuf.clear();
    fmt << "%7" << (Radix == octal ? PRIo64 : PRIu64) << "\t"
        << "%7" PRIx64 "\t";
    outs() << format(fmtbuf.c_str(), total, total);
  }
}

// Print Obj along with any format-specific prefix or suffix headers described
// by Ctx.
static void printObject(ObjectFile &Obj, const ObjectContext &Ctx) {
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

  printObjectSectionSizes(&Obj);

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
    if (auto *Obj = dyn_cast<ObjectFile>(ChildOrErr->get())) {
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
  } else if (auto *Obj = dyn_cast<ObjectFile>(&Bin)) {
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
  std::string fmtbuf;
  raw_string_ostream fmt(fmtbuf);
  const char *radix_fmt = getRadixFmt();

  if (HasMachOFiles) {
    // Darwin format totals: __TEXT __DATA __OBJC others dec hex
    outs() << TotalObjectText << "\t" << TotalObjectData << "\t"
           << TotalObjectObjc << "\t" << TotalObjectOthers << "\t"
           << TotalObjectTotal << "\t" << format("%" PRIx64, TotalObjectTotal)
           << "\t(TOTALS)\n";
  } else {
    fmt << "%#7" << radix_fmt << "\t"
        << "%#7" << radix_fmt << "\t"
        << "%#7" << radix_fmt << "\t";
    outs() << format(fmtbuf.c_str(), TotalObjectText, TotalObjectData,
                     TotalObjectBss);
    fmtbuf.clear();
    fmt << "%7" << (Radix == octal ? PRIo64 : PRIu64) << "\t"
        << "%7" PRIx64 "\t";
    outs() << format(fmtbuf.c_str(), TotalObjectTotal, TotalObjectTotal)
           << "(TOTALS)\n";
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
