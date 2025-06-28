//===- llvm-elf2bin.cpp - Convert ELF image to binary formats -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a tool that converts images into binary and hex formats, similar to
// some of llvm-objcopy's functionality, but specialized to ELF, using only the
// 'load view' of an ELF image, that is, the PT_LOAD segments in the program
// header table. The output can be written to plain binary files or various hex
// formats. An additional option allows the output to be split into multiple
// 'banks' to be loaded into separate ROMs, e.g. with the first 2 bytes out of
// every 4 going into one ROM and the other 2 bytes going into another.
//
//===----------------------------------------------------------------------===//

#include "llvm-elf2bin.h"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/bit.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/LLVMDriver.h"

using namespace llvm;
using llvm::object::ELFObjectFileBase;

[[noreturn]] static void fatal_common(std::optional<StringRef> filename,
                                      Twine message,
                                      std::optional<llvm::Error> err,
                                      bool suggest_help) {
  llvm::errs() << "llvm-elf2bin: ";
  if (filename)
    llvm::errs() << *filename << ": ";
  llvm::errs() << message;
  if (err) {
    handleAllErrors(std::move(*err), [](const ErrorInfoBase &einfo) {
      llvm::errs() << ": " << einfo.message();
    });
  }
  llvm::errs() << "\n";
  if (suggest_help)
    llvm::errs() << "(try 'llvm-elf2bin --help' for help)\n";
  exit(1);
}

[[noreturn]] void fatal(InputObject &inobj, Twine message, llvm::Error err) {
  fatal_common(inobj.filename, message, std::move(err), false);
}
[[noreturn]] void fatal(InputObject &inobj, Twine message) {
  fatal_common(inobj.filename, message, std::nullopt, false);
}
[[noreturn]] void fatal(StringRef filename, Twine message, llvm::Error err) {
  fatal_common(filename, message, std::move(err), false);
}
[[noreturn]] void fatal(StringRef filename, Twine message) {
  fatal_common(filename, message, std::nullopt, false);
}
[[noreturn]] void fatal(Twine message) {
  fatal_common(std::nullopt, message, std::nullopt, false);
}
// Just like fatal() but also suggests --help
[[noreturn]] void fatal_suggest_help(Twine message) {
  fatal_common(std::nullopt, message, std::nullopt, true);
}

/*
 * Format an output file name, according to the format string
 * description provided by the user and documented in the help above.
 *
 * (So, unlike sprintf, this function doesn't take an arbitrary
 * variadic argument list. Its input data consists of the details of
 * an output file that's about to be written, and the format
 * directives refer to particular pieces of data like 'input file
 * name' and 'bank number' rather than to data types.)
 */
std::string format_outfile(std::string pattern, std::string inpath,
                           std::optional<uint64_t> baseaddr, uint64_t bank) {
  std::string infile;
  size_t slash = inpath.find_last_of("/"
#ifdef _WIN32
                                     "\\:"
#endif
  );
  infile = inpath.substr(slash == std::string::npos ? 0 : slash + 1);

  std::string outstring;
  llvm::raw_string_ostream outstream(outstring);

  for (auto it = pattern.begin(); it != pattern.end();) {
    char c = *it++;
    if (c == '%') {
      if (it == pattern.end())
        fatal(Twine("output pattern '") + pattern +
              "' ends with incomplete % escape");
      char d = *it++;
      switch (d) {
      case 'F':
        outstream << infile;
        break;
      case 'f':
        outstream << infile.substr(
            0, std::min(infile.size(), infile.find_last_of('.')));
        break;
      case 'a':
      case 'A': {
        if (!baseaddr)
          fatal(Twine("output pattern '") + pattern + "' contains '%" +
                Twine(d) + "' but no base address is available");
        Twine hex = Twine::utohexstr(baseaddr.value());
        if (d == 'a') {
          outstream << hex;
        } else {
          SmallVector<char, 16> hexdata;
          outstream << hex.toStringRef(hexdata).upper();
        }
        break;
      }
      case 'b':
        outstream << bank;
        break;
      case '%':
        outstream << '%';
        break;
      default:
        fatal(Twine("output pattern '") + pattern +
              "' contains unrecognized % escape '%" + Twine(d) + "'");
      }
    } else {
      outstream << c;
    }
  }

  return outstring;
}

enum class Format {
  IHex,
  SRec,
  BinMultifile,
  BinCombined,
  VhxMultifile,
  VhxCombined
};

namespace {
using namespace llvm::opt; // for HelpHidden in Opts.inc
enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

#define PREFIX(NAME, VALUE)                                                    \
  static constexpr StringLiteral NAME##_init[] = VALUE;                        \
  static constexpr ArrayRef<StringLiteral> NAME(NAME##_init,                   \
                                                std::size(NAME##_init) - 1);
#include "Opts.inc"
#undef PREFIX

static constexpr opt::OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

class Elf2BinOptTable : public opt::GenericOptTable {
public:
  Elf2BinOptTable() : opt::GenericOptTable(InfoTable) {}
};
} // namespace

int llvm_elf2bin_main(int argc, char **argv, const llvm::ToolContext &) {
  InitLLVM X(argc, argv);
  BumpPtrAllocator A;
  StringSaver Saver(A);
  Elf2BinOptTable table;
  opt::InputArgList Args =
      table.parseArgs(argc, argv, OPT_UNKNOWN, Saver, fatal_suggest_help);

  if (Args.hasArg(OPT_help)) {
    table.printHelp(outs(), "llvm-elf2bin [options] <input ELF images>",
                    "LLVM ELF-to-binary converter");
    return 0;
  }
  if (Args.hasArg(OPT_version)) {
    llvm::cl::PrintVersionMessage();
    return 0;
  }

  std::vector<std::string> infiles = Args.getAllArgValues(OPT_INPUT);

  std::optional<std::string> outfile, outpattern;
  if (Arg *A = Args.getLastArg(OPT_output_file_EQ))
    outfile = A->getValue();
  if (Arg *A = Args.getLastArg(OPT_output_pattern_EQ))
    outpattern = A->getValue();

  std::optional<Format> format;
  if (Arg *A = Args.getLastArg(OPT_ihex, OPT_srec, OPT_bin, OPT_bincombined,
                               OPT_vhx, OPT_vhxcombined)) {
    auto Option = A->getOption();
    if (Option.matches(OPT_ihex))
      format = Format::IHex;
    if (Option.matches(OPT_srec))
      format = Format::SRec;
    if (Option.matches(OPT_bin))
      format = Format::BinMultifile;
    if (Option.matches(OPT_bincombined))
      format = Format::BinCombined;
    if (Option.matches(OPT_vhx))
      format = Format::VhxMultifile;
    if (Option.matches(OPT_vhxcombined))
      format = Format::VhxCombined;
  }

  std::optional<uint64_t> baseaddr;
  if (Arg *A = Args.getLastArg(OPT_base_EQ)) {
    uint64_t value;
    StringRef str = A->getValue();
    if (str.getAsInteger(0, value))
      fatal(Twine("cannot parse base address '") + str + "'");
    baseaddr = value;
  }

  std::optional<std::set<uint64_t>> segments_wanted;
  if (Arg *A = Args.getLastArg(OPT_segments_EQ)) {
    std::set<uint64_t> bases;
    SmallVector<StringRef, 4> fields;
    SplitString(A->getValue(), fields, ",");
    for (auto field : fields) {
      uint64_t base;
      if (field.getAsInteger(0, base))
        fatal(Twine("cannot parse segment base address '") + field + "'");
      bases.insert(base);
    }
    segments_wanted = bases;
  }

  uint64_t bankwidth = 1, nbanks = 1;
  if (Arg *A = Args.getLastArg(OPT_banks_EQ)) {
    StringRef str = A->getValue();
    size_t xpos = str.find_first_of("x");
    size_t xpos2 = str.find_last_of("x");
    if (!(xpos == xpos2 && !str.substr(0, xpos).getAsInteger(0, bankwidth) &&
          !str.substr(xpos + 1).getAsInteger(0, nbanks) &&
          llvm::has_single_bit(bankwidth) && llvm::has_single_bit(nbanks) &&
          bankwidth * nbanks != 0))
      fatal(Twine("cannot parse bank specification '") + str + "'");
  }

  uint64_t datareclen = 16;
  if (Arg *A = Args.getLastArg(OPT_datareclen_EQ)) {
    StringRef str = A->getValue();
    if (str.getAsInteger(0, datareclen))
      fatal(Twine("cannot parse base address '") + str + "'");
  }

  bool include_zi = Args.hasArg(OPT_zi);
  bool physical = true;
  if (Arg *A = Args.getLastArg(OPT_physical, OPT_virtual))
    physical = A->getOption().matches(OPT_physical);

  if (infiles.empty())
    fatal_suggest_help("no input file specified");
  if (!outfile && !outpattern)
    fatal_suggest_help("no output filename or pattern specified");
  if (!format)
    fatal_suggest_help("no output format specified");
  if (outfile && outpattern)
    fatal_suggest_help("output filename and pattern both specified");

  if ((format != Format::BinCombined && format != Format::VhxCombined) &&
      baseaddr)
    fatal("--base only applies to --bincombined and --vhxcombined");

  if ((format != Format::BinMultifile && format != Format::BinCombined &&
       format != Format::VhxMultifile && format != Format::VhxCombined) &&
      (bankwidth != 1 || nbanks != 1))
    fatal("--banks only applies to binary and VHX output");

  /*
   * Open the input files.
   */
  std::vector<InputObject> objects;

  for (const std::string &infile : infiles) {
    objects.emplace_back();
    InputObject &inobj = objects.back();
    inobj.filename = infile;

    ErrorOr<std::unique_ptr<MemoryBuffer>> membuf_or_err =
        MemoryBuffer::getFileOrSTDIN(infile, false, false);
    if (std::error_code error = membuf_or_err.getError())
      fatal(infile, "unable to open", errorCodeToError(error));
    inobj.membuf = std::move(membuf_or_err.get());

    Expected<std::unique_ptr<llvm::object::Binary>> binary_or_err =
        llvm::object::createBinary(inobj.membuf->getMemBufferRef(), nullptr,
                                   false);
    if (!binary_or_err)
      fatal(infile, "unable to process", binary_or_err.takeError());

    std::unique_ptr<ELFObjectFileBase> elf =
        dyn_cast<ELFObjectFileBase>(*binary_or_err);
    if (!elf)
      fatal(infile, "unable to process: not an ELF file");
    inobj.elf = std::move(elf);
  }

  /*
   * Helper function for listing the segments of a file, paying
   * attention to the --segments option to restrict to a subset.
   */
  auto segments = [&](InputObject &inobj) {
    std::vector<Segment> allsegs = inobj.segments(physical);
    if (!segments_wanted)
      return allsegs;

    std::vector<Segment> segs;
    auto &keep = segments_wanted.value();
    for (auto seg : allsegs)
      if (keep.find(seg.baseaddr) != keep.end())
        segs.push_back(seg);
    return segs;
  };

  /*
   * Make a list of all the conversions we want to do.
   */

  struct Conversion {
    InputObject *inobj;
    std::string outfile;
    std::optional<uint64_t> baseaddr, fileoffset, size, zisize;
    uint64_t bank;
  };
  std::vector<Conversion> convs;

  for (auto &inobj : objects) {
    // Helper function to fill in infile and outfile
    auto add_conv = [&](Conversion conv) {
      conv.inobj = &inobj;
      if (outfile)
        conv.outfile = outfile.value();
      else
        conv.outfile = format_outfile(outpattern.value(), conv.inobj->filename,
                                      conv.baseaddr, conv.bank);
      convs.push_back(conv);
    };

    switch (format.value()) {
    case Format::BinMultifile:
    case Format::VhxMultifile:
      /*
       * Separate output file per segment and per bank, so go
       * through this input file and list its segments.
       */
      for (auto seg : segments(inobj)) {
        for (uint64_t bank = 0; bank < nbanks; bank++) {
          Conversion conv;
          conv.baseaddr = seg.baseaddr;
          conv.fileoffset = seg.fileoffset;
          conv.size = seg.filesize;
          conv.bank = bank;
          if (include_zi && seg.memsize > seg.filesize)
            conv.zisize = seg.memsize - seg.filesize;
          else
            conv.zisize = 0;
          add_conv(conv);
        }
      }
      break;
    case Format::BinCombined:
    case Format::VhxCombined:
      /*
       * Separate output file per bank, but each one contains
       * the whole input file.
       */
      for (uint64_t bank = 0; bank < nbanks; bank++) {
        Conversion conv;
        conv.bank = bank;
        add_conv(conv);
      }
      break;
    default:
      /*
       * One output file per input file.
       */
      add_conv(Conversion{});
      break;
    }
  }

  std::set<std::string> outfiles;
  for (const auto &conv : convs) {
    if (outfiles.find(conv.outfile) != outfiles.end()) {
      fatal(Twine("output file '") + conv.outfile +
            "' would be written more than once by this command");
      std::exit(1);
    }
    outfiles.insert(conv.outfile);
  }

  uint64_t bankmod = nbanks * bankwidth;

  for (const auto &conv : convs) {
    switch (format.value()) {
    case Format::BinMultifile:
      bin_write(*conv.inobj, conv.outfile, conv.fileoffset.value(),
                conv.size.value(), conv.zisize.value(), bankmod,
                conv.bank * bankwidth, bankwidth);
      break;

    case Format::BinCombined:
      bincombined_write(*conv.inobj, conv.outfile, segments(*conv.inobj),
                        include_zi, baseaddr, bankmod, conv.bank * bankwidth,
                        bankwidth);
      break;

    case Format::VhxMultifile:
      vhx_write(*conv.inobj, conv.outfile, conv.fileoffset.value(),
                conv.size.value(), conv.zisize.value(), bankmod,
                conv.bank * bankwidth, bankwidth);
      break;

    case Format::VhxCombined:
      vhxcombined_write(*conv.inobj, conv.outfile, segments(*conv.inobj),
                        include_zi, baseaddr, bankmod, conv.bank * bankwidth,
                        bankwidth);
      break;

    case Format::IHex:
      ihex_write(*conv.inobj, conv.outfile, segments(*conv.inobj), include_zi,
                 datareclen);
      break;

    case Format::SRec:
      srec_write(*conv.inobj, conv.outfile, segments(*conv.inobj), include_zi,
                 datareclen);
      break;
    }
  }

  return 0;
}
