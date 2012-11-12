//===-- llvm-size.cpp - Print the size of each object section -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/system_error.h"
#include <algorithm>
#include <string>
using namespace llvm;
using namespace object;

enum OutputFormatTy {berkeley, sysv};
static cl::opt<OutputFormatTy>
       OutputFormat("format",
         cl::desc("Specify output format"),
         cl::values(clEnumVal(sysv, "System V format"),
                    clEnumVal(berkeley, "Berkeley format"),
                    clEnumValEnd),
         cl::init(berkeley));

static cl::opt<OutputFormatTy>
       OutputFormatShort(cl::desc("Specify output format"),
         cl::values(clEnumValN(sysv, "A", "System V format"),
                    clEnumValN(berkeley, "B", "Berkeley format"),
                    clEnumValEnd),
         cl::init(berkeley));

enum RadixTy {octal = 8, decimal = 10, hexadecimal = 16};
static cl::opt<unsigned int>
       Radix("-radix",
         cl::desc("Print size in radix. Only 8, 10, and 16 are valid"),
         cl::init(decimal));

static cl::opt<RadixTy>
       RadixShort(cl::desc("Print size in radix:"),
         cl::values(clEnumValN(octal, "o", "Print size in octal"),
                    clEnumValN(decimal, "d", "Print size in decimal"),
                    clEnumValN(hexadecimal, "x", "Print size in hexadecimal"),
                    clEnumValEnd),
         cl::init(decimal));

static cl::list<std::string>
       InputFilenames(cl::Positional, cl::desc("<input files>"),
                      cl::ZeroOrMore);

static std::string ToolName;

///  @brief If ec is not success, print the error and return true.
static bool error(error_code ec) {
  if (!ec) return false;

  outs() << ToolName << ": error reading file: " << ec.message() << ".\n";
  outs().flush();
  return true;
}

/// @brief Get the length of the string that represents @p num in Radix
///        including the leading 0x or 0 for hexadecimal and octal respectively.
static size_t getNumLengthAsString(uint64_t num) {
  APInt conv(64, num);
  SmallString<32> result;
  conv.toString(result, Radix, false, true);
  return result.size();
}

/// @brief Print the size of each section in @p o.
///
/// The format used is determined by @c OutputFormat and @c Radix.
static void PrintObjectSectionSizes(ObjectFile *o) {
  uint64_t total = 0;
  std::string fmtbuf;
  raw_string_ostream fmt(fmtbuf);

  const char *radix_fmt = 0;
  switch (Radix) {
  case octal:
    radix_fmt = PRIo64;
    break;
  case decimal:
    radix_fmt = PRIu64;
    break;
  case hexadecimal:
    radix_fmt = PRIx64;
    break;
  }
  if (OutputFormat == sysv) {
    // Run two passes over all sections. The first gets the lengths needed for
    // formatting the output. The second actually does the output.
    std::size_t max_name_len = strlen("section");
    std::size_t max_size_len = strlen("size");
    std::size_t max_addr_len = strlen("addr");
    error_code ec;
    for (section_iterator i = o->begin_sections(),
                          e = o->end_sections(); i != e;
                          i.increment(ec)) {
      if (error(ec))
        return;
      uint64_t size = 0;
      if (error(i->getSize(size)))
        return;
      total += size;

      StringRef name;
      uint64_t addr = 0;
      if (error(i->getName(name))) return;
      if (error(i->getAddress(addr))) return;
      max_name_len = std::max(max_name_len, name.size());
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
    outs() << format(fmt.str().c_str(),
                     static_cast<const char*>("section"),
                     static_cast<const char*>("size"),
                     static_cast<const char*>("addr"));
    fmtbuf.clear();

    // Setup per section format.
    fmt << "%-" << max_name_len << "s "
        << "%#" << max_size_len << radix_fmt << " "
        << "%#" << max_addr_len << radix_fmt << "\n";

    // Print each section.
    for (section_iterator i = o->begin_sections(),
                          e = o->end_sections(); i != e;
                          i.increment(ec)) {
      if (error(ec))
        return;

      StringRef name;
      uint64_t size = 0;
      uint64_t addr = 0;
      if (error(i->getName(name))) return;
      if (error(i->getSize(size))) return;
      if (error(i->getAddress(addr))) return;
      std::string namestr = name;

      outs() << format(fmt.str().c_str(),
                       namestr.c_str(),
                       size,
                       addr);
    }

    // Print total.
    fmtbuf.clear();
    fmt << "%-" << max_name_len << "s "
        << "%#" << max_size_len << radix_fmt << "\n";
    outs() << format(fmt.str().c_str(),
                     static_cast<const char*>("Total"),
                     total);
  } else {
    // The Berkeley format does not display individual section sizes. It
    // displays the cumulative size for each section type.
    uint64_t total_text = 0;
    uint64_t total_data = 0;
    uint64_t total_bss = 0;

    // Make one pass over the section table to calculate sizes.
    error_code ec;
    for (section_iterator i = o->begin_sections(),
                          e = o->end_sections(); i != e;
                          i.increment(ec)) {
      if (error(ec))
        return;

      uint64_t size = 0;
      bool isText = false;
      bool isData = false;
      bool isBSS = false;
      if (error(i->getSize(size))) return;
      if (error(i->isText(isText))) return;
      if (error(i->isData(isData))) return;
      if (error(i->isBSS(isBSS))) return;
      if (isText)
        total_text += size;
      else if (isData)
        total_data += size;
      else if (isBSS)
        total_bss += size;
    }

    total = total_text + total_data + total_bss;

    // Print result.
    fmt << "%#7" << radix_fmt << " "
        << "%#7" << radix_fmt << " "
        << "%#7" << radix_fmt << " ";
    outs() << format(fmt.str().c_str(),
                     total_text,
                     total_data,
                     total_bss);
    fmtbuf.clear();
    fmt << "%7" << (Radix == octal ? PRIo64 : PRIu64) << " "
        << "%7" PRIx64 " ";
    outs() << format(fmt.str().c_str(),
                     total,
                     total);
  }
}

/// @brief Print the section sizes for @p file. If @p file is an archive, print
///        the section sizes for each archive member.
static void PrintFileSectionSizes(StringRef file) {
  // If file is not stdin, check that it exists.
  if (file != "-") {
    bool exists;
    if (sys::fs::exists(file, exists) || !exists) {
      errs() << ToolName << ": '" << file << "': " << "No such file\n";
      return;
    }
  }

  // Attempt to open the binary.
  OwningPtr<Binary> binary;
  if (error_code ec = createBinary(file, binary)) {
    errs() << ToolName << ": " << file << ": " << ec.message() << ".\n";
    return;
  }

  if (Archive *a = dyn_cast<Archive>(binary.get())) {
    // This is an archive. Iterate over each member and display its sizes.
    for (object::Archive::child_iterator i = a->begin_children(),
                                         e = a->end_children(); i != e; ++i) {
      OwningPtr<Binary> child;
      if (error_code ec = i->getAsBinary(child)) {
        errs() << ToolName << ": " << file << ": " << ec.message() << ".\n";
        continue;
      }
      if (ObjectFile *o = dyn_cast<ObjectFile>(child.get())) {
        if (OutputFormat == sysv)
          outs() << o->getFileName() << "   (ex " << a->getFileName()
                  << "):\n";
        PrintObjectSectionSizes(o);
        if (OutputFormat == berkeley)
          outs() << o->getFileName() << " (ex " << a->getFileName() << ")\n";
      }
    }
  } else if (ObjectFile *o = dyn_cast<ObjectFile>(binary.get())) {
    if (OutputFormat == sysv)
      outs() << o->getFileName() << "  :\n";
    PrintObjectSectionSizes(o);
    if (OutputFormat == berkeley)
      outs() << o->getFileName() << "\n";
  } else {
    errs() << ToolName << ": " << file << ": " << "Unrecognized file type.\n";
  }
  // System V adds an extra newline at the end of each file.
  if (OutputFormat == sysv)
    outs() << "\n";
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);

  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv, "llvm object size dumper\n");

  ToolName = argv[0];
  if (OutputFormatShort.getNumOccurrences())
    OutputFormat = OutputFormatShort;
  if (RadixShort.getNumOccurrences())
    Radix = RadixShort;

  if (InputFilenames.size() == 0)
    InputFilenames.push_back("a.out");

  if (OutputFormat == berkeley)
    outs() << "   text    data     bss     "
           << (Radix == octal ? "oct" : "dec")
           << "     hex filename\n";

  std::for_each(InputFilenames.begin(), InputFilenames.end(),
                PrintFileSectionSizes);

  return 0;
}
