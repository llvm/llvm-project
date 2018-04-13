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
#include "llvm/Support/InitLLVM.h"
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
static cl::opt<std::string> OutputFilename(cl::Positional, cl::desc("[ <output> ]"));

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
static cl::list<std::string> Keep("keep", cl::desc("Keep <section>"),
                                  cl::value_desc("section"));
static cl::list<std::string> OnlyKeep("only-keep",
                                      cl::desc("Remove all but <section>"),
                                      cl::value_desc("section"));
static cl::alias OnlyKeepA("j", cl::desc("Alias for only-keep"),
                           cl::aliasopt(OnlyKeep));
static cl::opt<bool> StripDebug("strip-debug",
                                cl::desc("Removes all debug information"));
static cl::opt<bool> StripSections("strip-sections",
                                   cl::desc("Remove all section headers"));
static cl::opt<bool>
    StripNonAlloc("strip-non-alloc",
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
static cl::list<std::string> AddSection(
    "add-section",
    cl::desc("Make a section named <section> with the contents of <file>."),
    cl::value_desc("section=file"));
static cl::opt<bool> LocalizeHidden(
    "localize-hidden",
    cl::desc(
        "Mark all symbols that have hidden or internal visibility as local"));
static cl::opt<std::string>
    AddGnuDebugLink("add-gnu-debuglink",
                    cl::desc("adds a .gnu_debuglink for <debug-file>"),
                    cl::value_desc("debug-file"));

using SectionPred = std::function<bool(const SectionBase &Sec)>;

bool IsDWOSection(const SectionBase &Sec) { return Sec.Name.endswith(".dwo"); }

bool OnlyKeepDWOPred(const Object &Obj, const SectionBase &Sec) {
  // We can't remove the section header string table.
  if (&Sec == Obj.SectionNames)
    return false;
  // Short of keeping the string table we want to keep everything that is a DWO
  // section and remove everything else.
  return !IsDWOSection(Sec);
}

static ElfType OutputElfType;

std::unique_ptr<Writer> CreateWriter(Object &Obj, StringRef File) {
  if (OutputFormat == "binary") {
    return llvm::make_unique<BinaryWriter>(OutputFilename, Obj);
  }
  // Depending on the initial ELFT and OutputFormat we need a different Writer.
  switch (OutputElfType) {
  case ELFT_ELF32LE:
    return llvm::make_unique<ELFWriter<ELF32LE>>(File, Obj, !StripSections);
  case ELFT_ELF64LE:
    return llvm::make_unique<ELFWriter<ELF64LE>>(File, Obj, !StripSections);
  case ELFT_ELF32BE:
    return llvm::make_unique<ELFWriter<ELF32BE>>(File, Obj, !StripSections);
  case ELFT_ELF64BE:
    return llvm::make_unique<ELFWriter<ELF64BE>>(File, Obj, !StripSections);
  }
  llvm_unreachable("Invalid output format");
}

void SplitDWOToFile(const Reader &Reader, StringRef File) {
  auto DWOFile = Reader.create();
  DWOFile->removeSections(
      [&](const SectionBase &Sec) { return OnlyKeepDWOPred(*DWOFile, Sec); });
  auto Writer = CreateWriter(*DWOFile, File);
  Writer->finalize();
  Writer->write();
}

// This function handles the high level operations of GNU objcopy including
// handling command line options. It's important to outline certain properties
// we expect to hold of the command line operations. Any operation that "keeps"
// should keep regardless of a remove. Additionally any removal should respect
// any previous removals. Lastly whether or not something is removed shouldn't
// depend a) on the order the options occur in or b) on some opaque priority
// system. The only priority is that keeps/copies overrule removes.
void HandleArgs(Object &Obj, const Reader &Reader) {

  if (!SplitDWO.empty()) {
    SplitDWOToFile(Reader, SplitDWO);
  }

  // Localize:

  if (LocalizeHidden) {
    Obj.SymbolTable->localize([](const Symbol &Sym) {
      return Sym.Visibility == STV_HIDDEN || Sym.Visibility == STV_INTERNAL;
    });
  }

  SectionPred RemovePred = [](const SectionBase &) { return false; };

  // Removes:

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
      return OnlyKeepDWOPred(Obj, Sec) || RemovePred(Sec);
    };

  if (StripAllGNU)
    RemovePred = [RemovePred, &Obj](const SectionBase &Sec) {
      if (RemovePred(Sec))
        return true;
      if ((Sec.Flags & SHF_ALLOC) != 0)
        return false;
      if (&Sec == Obj.SectionNames)
        return false;
      switch (Sec.Type) {
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
      if (&Sec == Obj.SectionNames)
        return false;
      return (Sec.Flags & SHF_ALLOC) == 0;
    };

  if (StripAll)
    RemovePred = [RemovePred, &Obj](const SectionBase &Sec) {
      if (RemovePred(Sec))
        return true;
      if (&Sec == Obj.SectionNames)
        return false;
      if (Sec.Name.startswith(".gnu.warning"))
        return false;
      return (Sec.Flags & SHF_ALLOC) == 0;
    };

  // Explicit copies:

  if (!OnlyKeep.empty()) {
    RemovePred = [RemovePred, &Obj](const SectionBase &Sec) {
      // Explicitly keep these sections regardless of previous removes.
      if (std::find(std::begin(OnlyKeep), std::end(OnlyKeep), Sec.Name) !=
          std::end(OnlyKeep))
        return false;

      // Allow all implicit removes.
      if (RemovePred(Sec))
        return true;

      // Keep special sections.
      if (Obj.SectionNames == &Sec)
        return false;
      if (Obj.SymbolTable == &Sec || Obj.SymbolTable->getStrTab() == &Sec)
        return false;

      // Remove everything else.
      return true;
    };
  }

  if (!Keep.empty()) {
    RemovePred = [RemovePred](const SectionBase &Sec) {
      // Explicitly keep these sections regardless of previous removes.
      if (std::find(std::begin(Keep), std::end(Keep), Sec.Name) !=
          std::end(Keep))
        return false;
      // Otherwise defer to RemovePred.
      return RemovePred(Sec);
    };
  }

  Obj.removeSections(RemovePred);

  if (!AddSection.empty()) {
    for (const auto &Flag : AddSection) {
      auto SecPair = StringRef(Flag).split("=");
      auto SecName = SecPair.first;
      auto File = SecPair.second;
      auto BufOrErr = MemoryBuffer::getFile(File);
      if (!BufOrErr)
        reportError(File, BufOrErr.getError());
      auto Buf = std::move(*BufOrErr);
      auto BufPtr = reinterpret_cast<const uint8_t *>(Buf->getBufferStart());
      auto BufSize = Buf->getBufferSize();
      Obj.addSection<OwnedDataSection>(SecName,
                                       ArrayRef<uint8_t>(BufPtr, BufSize));
    }
  }

  if (!AddGnuDebugLink.empty()) {
    Obj.addSection<GnuDebugLinkSection>(StringRef(AddGnuDebugLink));
  }
}

std::unique_ptr<Reader> CreateReader() {
  // Right now we can only read ELF files so there's only one reader;
  auto Out = llvm::make_unique<ELFReader>(StringRef(InputFilename));
  // We need to set the default ElfType for output.
  OutputElfType = Out->getElfType();
  return std::move(Out);
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "llvm objcopy utility\n");
  ToolName = argv[0];
  if (InputFilename.empty()) {
    cl::PrintHelpMessage();
    return 2;
  }

  auto Reader = CreateReader();
  auto Obj = Reader->create();
  StringRef Output =
      OutputFilename.getNumOccurrences() ? OutputFilename : InputFilename;
  auto Writer = CreateWriter(*Obj, Output);
  HandleArgs(*Obj, *Reader);
  Writer->finalize();
  Writer->write();
}
