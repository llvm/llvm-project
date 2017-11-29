//===- llvm-objcopy.cpp ---------------------------------------------------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm-objcopy.h"
#include "Object.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/Error.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

using namespace llvm;
using namespace object;
using namespace ELF;

// The name this program was invoked as.
static StringRef ToolName;

namespace llvm {

LLVM_ATTRIBUTE_NORETURN void error(Twine Message) {
  errs() << ToolName << ": " << Message << ".\n";
  errs().flush();
  exit(1);
}

LLVM_ATTRIBUTE_NORETURN void reportError(StringRef File, std::error_code EC) {
  assert(EC);
  errs() << ToolName << ": '" << File << "': " << EC.message() << ".\n";
  exit(1);
}

LLVM_ATTRIBUTE_NORETURN void reportError(StringRef File, Error E) {
  assert(E);
  std::string Buf;
  raw_string_ostream OS(Buf);
  logAllUnhandledErrors(std::move(E), OS, "");
  OS.flush();
  errs() << ToolName << ": '" << File << "': " << Buf;
  exit(1);
}

} // end namespace llvm

static cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input>"));
static cl::opt<std::string> OutputFilename(cl::Positional, cl::desc("<output>"),
                                    cl::init("-"));
static cl::opt<std::string>
    OutputFormat("O", cl::desc("Set output format to one of the following:"
                               "\n\tbinary"));
static cl::list<std::string> ToRemove("remove-section",
                                      cl::desc("Remove <section>"),
                                      cl::value_desc("section"));
static cl::alias ToRemoveA("R", cl::desc("Alias for remove-section"),
                           cl::aliasopt(ToRemove));
static cl::opt<bool> StripAll(
    "strip-all",
    cl::desc(
        "Removes non-allocated sections other than .gnu.warning* sections"));
static cl::opt<bool>
    StripAllGNU("strip-all-gnu",
                cl::desc("Removes symbol, relocation, and debug information"));
static cl::opt<bool> StripDebug("strip-debug",
                                cl::desc("Removes all debug information"));
static cl::opt<bool> StripSections("strip-sections",
                                   cl::desc("Remove all section headers"));
static cl::opt<bool> StripNonAlloc("strip-non-alloc",
                                   cl::desc("Remove all non-allocated sections"));
static cl::opt<bool>
    StripDWO("strip-dwo", cl::desc("Remove all DWARF .dwo sections from file"));
static cl::opt<bool> ExtractDWO(
    "extract-dwo",
    cl::desc("Remove all sections that are not DWARF .dwo sections from file"));
static cl::opt<std::string>
    SplitDWO("split-dwo",
             cl::desc("Equivalent to extract-dwo on the input file to "
                      "<dwo-file>, then strip-dwo on the input file"),
             cl::value_desc("dwo-file"));

using SectionPred = std::function<bool(const SectionBase &Sec)>;

bool IsDWOSection(const SectionBase &Sec) {
  return Sec.Name.endswith(".dwo");
}

template <class ELFT>
bool OnlyKeepDWOPred(const Object<ELFT> &Obj, const SectionBase &Sec) {
  // We can't remove the section header string table.
  if (&Sec == Obj.getSectionHeaderStrTab())
    return false;
  // Short of keeping the string table we want to keep everything that is a DWO
  // section and remove everything else.
  return !IsDWOSection(Sec);
}

template <class ELFT>
void WriteObjectFile(const Object<ELFT> &Obj, StringRef File) {
  std::unique_ptr<FileOutputBuffer> Buffer;
  Expected<std::unique_ptr<FileOutputBuffer>> BufferOrErr =
      FileOutputBuffer::create(File, Obj.totalSize(),
                               FileOutputBuffer::F_executable);
  handleAllErrors(BufferOrErr.takeError(), [](const ErrorInfoBase &) {
    error("failed to open " + OutputFilename);
  });
  Buffer = std::move(*BufferOrErr);

  Obj.write(*Buffer);
  if (auto E = Buffer->commit())
    reportError(File, errorToErrorCode(std::move(E)));
}

template <class ELFT>
void SplitDWOToFile(const ELFObjectFile<ELFT> &ObjFile, StringRef File) {
  // Construct a second output file for the DWO sections.
  ELFObject<ELFT> DWOFile(ObjFile);

  DWOFile.removeSections([&](const SectionBase &Sec) {
    return OnlyKeepDWOPred<ELFT>(DWOFile, Sec);
  });
  DWOFile.finalize();
  WriteObjectFile(DWOFile, File);
}

template <class ELFT>
void CopyBinary(const ELFObjectFile<ELFT> &ObjFile) {
  std::unique_ptr<Object<ELFT>> Obj;

  if (!OutputFormat.empty() && OutputFormat != "binary")
    error("invalid output format '" + OutputFormat + "'");
  if (!OutputFormat.empty() && OutputFormat == "binary")
    Obj = llvm::make_unique<BinaryObject<ELFT>>(ObjFile);
  else
    Obj = llvm::make_unique<ELFObject<ELFT>>(ObjFile);

  if (!SplitDWO.empty())
    SplitDWOToFile<ELFT>(ObjFile, SplitDWO.getValue());

  SectionPred RemovePred = [](const SectionBase &) { return false; };

  if (!ToRemove.empty()) {
    RemovePred = [&](const SectionBase &Sec) {
      return std::find(std::begin(ToRemove), std::end(ToRemove), Sec.Name) !=
             std::end(ToRemove);
    };
  }

  if (StripDWO || !SplitDWO.empty())
    RemovePred = [RemovePred](const SectionBase &Sec) {
      return IsDWOSection(Sec) || RemovePred(Sec);
    };

  if (ExtractDWO)
    RemovePred = [RemovePred, &Obj](const SectionBase &Sec) {
      return OnlyKeepDWOPred(*Obj, Sec) || RemovePred(Sec);
    };

  if (StripAllGNU)
    RemovePred = [RemovePred, &Obj](const SectionBase &Sec) {
      if (RemovePred(Sec))
        return true;
      if ((Sec.Flags & SHF_ALLOC) != 0)
        return false;
      if (&Sec == Obj->getSectionHeaderStrTab())
        return false;
      switch(Sec.Type) {
      case SHT_SYMTAB:
      case SHT_REL:
      case SHT_RELA:
      case SHT_STRTAB:
        return true;
      }
      return Sec.Name.startswith(".debug");
    };

  if (StripSections) {
    RemovePred = [RemovePred](const SectionBase &Sec) {
      return RemovePred(Sec) || (Sec.Flags & SHF_ALLOC) == 0;
    };
    Obj->WriteSectionHeaders = false;
  }

  if (StripDebug) {
    RemovePred = [RemovePred](const SectionBase &Sec) {
      return RemovePred(Sec) || Sec.Name.startswith(".debug");
    };
  }

  if (StripNonAlloc)
    RemovePred = [RemovePred, &Obj](const SectionBase &Sec) {
      if (RemovePred(Sec))
        return true;
      if (&Sec == Obj->getSectionHeaderStrTab())
        return false;
      return (Sec.Flags & SHF_ALLOC) == 0;
    };

  if (StripAll)
    RemovePred = [RemovePred, &Obj](const SectionBase &Sec) {
      if (RemovePred(Sec))
        return true;
      if (&Sec == Obj->getSectionHeaderStrTab())
        return false;
      if (Sec.Name.startswith(".gnu.warning"))
        return false;
      return (Sec.Flags & SHF_ALLOC) == 0;
    };

  Obj->removeSections(RemovePred);
  Obj->finalize();
  WriteObjectFile(*Obj, OutputFilename.getValue());
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv, "llvm objcopy utility\n");
  ToolName = argv[0];
  if (InputFilename.empty()) {
    cl::PrintHelpMessage();
    return 2;
  }
  Expected<OwningBinary<Binary>> BinaryOrErr = createBinary(InputFilename);
  if (!BinaryOrErr)
    reportError(InputFilename, BinaryOrErr.takeError());
  Binary &Binary = *BinaryOrErr.get().getBinary();
  if (auto *o = dyn_cast<ELFObjectFile<ELF64LE>>(&Binary)) {
    CopyBinary(*o);
    return 0;
  }
  if (auto *o = dyn_cast<ELFObjectFile<ELF32LE>>(&Binary)) {
    CopyBinary(*o);
    return 0;
  }
  if (auto *o = dyn_cast<ELFObjectFile<ELF64BE>>(&Binary)) {
    CopyBinary(*o);
    return 0;
  }
  if (auto *o = dyn_cast<ELFObjectFile<ELF32BE>>(&Binary)) {
    CopyBinary(*o);
    return 0;
  }
  reportError(InputFilename, object_error::invalid_file_type);
}
