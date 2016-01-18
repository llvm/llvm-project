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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/RelocVisitor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstring>
#include <list>
#include <string>
#include <system_error>

using namespace llvm;
using namespace object;

static cl::list<std::string>
InputFilenames(cl::Positional, cl::desc("<input object files>"),
               cl::ZeroOrMore);

static cl::opt<DIDumpType>
DumpType("debug-dump", cl::init(DIDT_All),
  cl::desc("Dump of debug sections:"),
  cl::values(
        clEnumValN(DIDT_All, "all", "Dump all debug sections"),
        clEnumValN(DIDT_Abbrev, "abbrev", ".debug_abbrev"),
        clEnumValN(DIDT_AbbrevDwo, "abbrev.dwo", ".debug_abbrev.dwo"),
        clEnumValN(DIDT_AppleNames, "apple_names", ".apple_names"),
        clEnumValN(DIDT_AppleTypes, "apple_types", ".apple_types"),
        clEnumValN(DIDT_AppleNamespaces, "apple_namespaces", ".apple_namespaces"),
        clEnumValN(DIDT_AppleObjC, "apple_objc", ".apple_objc"),
        clEnumValN(DIDT_Aranges, "aranges", ".debug_aranges"),
        clEnumValN(DIDT_Info, "info", ".debug_info"),
        clEnumValN(DIDT_InfoDwo, "info.dwo", ".debug_info.dwo"),
        clEnumValN(DIDT_Types, "types", ".debug_types"),
        clEnumValN(DIDT_TypesDwo, "types.dwo", ".debug_types.dwo"),
        clEnumValN(DIDT_Line, "line", ".debug_line"),
        clEnumValN(DIDT_LineDwo, "line.dwo", ".debug_line.dwo"),
        clEnumValN(DIDT_Loc, "loc", ".debug_loc"),
        clEnumValN(DIDT_LocDwo, "loc.dwo", ".debug_loc.dwo"),
        clEnumValN(DIDT_Frames, "frames", ".debug_frame"),
        clEnumValN(DIDT_Ranges, "ranges", ".debug_ranges"),
        clEnumValN(DIDT_Pubnames, "pubnames", ".debug_pubnames"),
        clEnumValN(DIDT_Pubtypes, "pubtypes", ".debug_pubtypes"),
        clEnumValN(DIDT_GnuPubnames, "gnu_pubnames", ".debug_gnu_pubnames"),
        clEnumValN(DIDT_GnuPubtypes, "gnu_pubtypes", ".debug_gnu_pubtypes"),
        clEnumValN(DIDT_Str, "str", ".debug_str"),
        clEnumValN(DIDT_StrDwo, "str.dwo", ".debug_str.dwo"),
        clEnumValN(DIDT_StrOffsetsDwo, "str_offsets.dwo", ".debug_str_offsets.dwo"),
        clEnumValEnd));

static void error(StringRef Filename, std::error_code EC) {
  if (!EC)
    return;
  errs() << Filename << ": " << EC.message() << "\n";
  exit(1);
}

static void DumpObjectFile(ObjectFile &Obj, Twine Filename) {
  std::unique_ptr<DIContext> DICtx(new DWARFContextInMemory(Obj));

  outs() << Filename.str() << ":\tfile format " << Obj.getFileFormatName()
         << "\n\n";
  // Dump the complete DWARF structure.
  DICtx->dump(outs(), DumpType);
}

static void DumpInput(StringRef Filename) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BuffOrErr =
      MemoryBuffer::getFileOrSTDIN(Filename);
  error(Filename, BuffOrErr.getError());
  std::unique_ptr<MemoryBuffer> Buff = std::move(BuffOrErr.get());

  ErrorOr<std::unique_ptr<Binary>> BinOrErr =
      object::createBinary(Buff->getMemBufferRef());
  error(Filename, BinOrErr.getError());

  if (auto *Obj = dyn_cast<ObjectFile>(BinOrErr->get()))
    DumpObjectFile(*Obj, Filename);
  else if (auto *Fat = dyn_cast<MachOUniversalBinary>(BinOrErr->get()))
    for (auto &ObjForArch : Fat->objects()) {
      auto MachOOrErr = ObjForArch.getAsObjectFile();
      error(Filename, MachOOrErr.getError());
      DumpObjectFile(**MachOOrErr,
                     Filename + " (" + ObjForArch.getArchTypeName() + ")");
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

  return EXIT_SUCCESS;
}
