//===-- llvm-dwarfdump.cpp - Debug info dumping utility for llvm ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program is a utility that works like "dwarfdump".
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/RelocVisitor.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include <algorithm>
#include <cstring>
#include <list>
#include <string>

using namespace llvm;
using namespace object;

static cl::list<std::string>
InputFilenames(cl::Positional, cl::desc("<input object files>"),
               cl::ZeroOrMore);

static cl::opt<unsigned long long>
Address("address", cl::init(-1ULL),
        cl::desc("Print line information for a given address"));

static cl::opt<bool>
PrintFunctions("functions", cl::init(false),
               cl::desc("Print function names as well as line information "
                        "for a given address"));

static cl::opt<bool>
PrintInlining("inlining", cl::init(false),
              cl::desc("Print all inlined frames for a given address"));

static void PrintDILineInfo(DILineInfo dli) {
  if (PrintFunctions)
    outs() << (dli.getFunctionName() ? dli.getFunctionName() : "<unknown>")
           << "\n";
  outs() << (dli.getFileName() ? dli.getFileName() : "<unknown>") << ':'
         << dli.getLine() << ':' << dli.getColumn() << '\n';
}

static void DumpInput(const StringRef &Filename) {
  OwningPtr<MemoryBuffer> Buff;

  if (error_code ec = MemoryBuffer::getFileOrSTDIN(Filename, Buff)) {
    errs() << Filename << ": " << ec.message() << "\n";
    return;
  }

  OwningPtr<ObjectFile> Obj(ObjectFile::createObjectFile(Buff.take()));

  StringRef DebugInfoSection;
  RelocAddrMap RelocMap;
  StringRef DebugAbbrevSection;
  StringRef DebugLineSection;
  StringRef DebugArangesSection;
  StringRef DebugStringSection;
  StringRef DebugRangesSection;

  error_code ec;
  for (section_iterator i = Obj->begin_sections(),
                        e = Obj->end_sections();
                        i != e; i.increment(ec)) {
    StringRef name;
    i->getName(name);
    StringRef data;
    i->getContents(data);

    if (name.startswith("__DWARF,"))
      name = name.substr(8); // Skip "__DWARF," prefix.
    name = name.substr(name.find_first_not_of("._")); // Skip . and _ prefixes.
    if (name == "debug_info")
      DebugInfoSection = data;
    else if (name == "debug_abbrev")
      DebugAbbrevSection = data;
    else if (name == "debug_line")
      DebugLineSection = data;
    else if (name == "debug_aranges")
      DebugArangesSection = data;
    else if (name == "debug_str")
      DebugStringSection = data;
    else if (name == "debug_ranges")
      DebugRangesSection = data;
    // Any more debug info sections go here.
    else
      continue;

    // TODO: For now only handle relocations for the debug_info section.
    if (name != "debug_info")
      continue;

    if (i->begin_relocations() != i->end_relocations()) {
      uint64_t SectionSize;
      i->getSize(SectionSize);
      for (relocation_iterator reloc_i = i->begin_relocations(),
                               reloc_e = i->end_relocations();
                               reloc_i != reloc_e; reloc_i.increment(ec)) {
        uint64_t Address;
        reloc_i->getAddress(Address);
        uint64_t Type;
        reloc_i->getType(Type);

        RelocVisitor V(Obj->getFileFormatName());
        // The section address is always 0 for debug sections.
        RelocToApply R(V.visit(Type, *reloc_i));
        if (V.error()) {
          SmallString<32> Name;
          error_code ec(reloc_i->getTypeName(Name));
          if (ec) {
            errs() << "Aaaaaa! Nameless relocation! Aaaaaa!\n";
          }
          errs() << "error: failed to compute relocation: "
                 << Name << "\n";
          continue;
        }

        if (Address + R.Width > SectionSize) {
          errs() << "error: " << R.Width << "-byte relocation starting "
                 << Address << " bytes into section " << name << " which is "
                 << SectionSize << " bytes long.\n";
          continue;
        }
        if (R.Width > 8) {
          errs() << "error: can't handle a relocation of more than 8 bytes at "
                    "a time.\n";
          continue;
        }
        DEBUG(dbgs() << "Writing " << format("%p", R.Value)
                     << " at " << format("%p", Address)
                     << " with width " << format("%d", R.Width)
                     << "\n");
        RelocMap[Address] = std::make_pair(R.Width, R.Value);
      }
    }
  }

  OwningPtr<DIContext> dictx(DIContext::getDWARFContext(/*FIXME*/true,
                                                        DebugInfoSection,
                                                        DebugAbbrevSection,
                                                        DebugArangesSection,
                                                        DebugLineSection,
                                                        DebugStringSection,
                                                        DebugRangesSection,
                                                        RelocMap));
  if (Address == -1ULL) {
    outs() << Filename
           << ":\tfile format " << Obj->getFileFormatName() << "\n\n";
    // Dump the complete DWARF structure.
    dictx->dump(outs());
  } else {
    // Print line info for the specified address.
    int SpecFlags = DILineInfoSpecifier::FileLineInfo |
                    DILineInfoSpecifier::AbsoluteFilePath;
    if (PrintFunctions)
      SpecFlags |= DILineInfoSpecifier::FunctionName;
    if (PrintInlining) {
      DIInliningInfo InliningInfo =
        dictx->getInliningInfoForAddress(Address, SpecFlags);
      uint32_t n = InliningInfo.getNumberOfFrames();
      if (n == 0) {
        // Print one empty debug line info in any case.
        PrintDILineInfo(DILineInfo());
      } else {
        for (uint32_t i = 0; i < n; i++) {
          DILineInfo dli = InliningInfo.getFrame(i);
          PrintDILineInfo(dli);
        }
      }
    } else {
      DILineInfo dli = dictx->getLineInfoForAddress(Address, SpecFlags);
      PrintDILineInfo(dli);
    }
  }
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "llvm dwarf dumper\n");

  // Defaults to a.out if no filenames specified.
  if (InputFilenames.size() == 0)
    InputFilenames.push_back("a.out");

  std::for_each(InputFilenames.begin(), InputFilenames.end(), DumpInput);

  return 0;
}
