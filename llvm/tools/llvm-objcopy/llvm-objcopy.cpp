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
    OutputFormat("O", cl::desc("set output format to one of the following:"
                               "\n\tbinary"));
static cl::list<std::string> ToRemove("remove-section",
                                      cl::desc("Remove a specific section"));
static cl::alias ToRemoveA("R", cl::desc("Alias for remove-section"),
                           cl::aliasopt(ToRemove));
static cl::opt<bool> StripSections("strip-sections",
                                   cl::desc("Remove all section headers"));
static cl::opt<bool>
    StripDWO("strip-dwo", cl::desc("remove all DWARF .dwo sections from file"));
static cl::opt<bool> ExtractDWO(
    "extract-dwo",
    cl::desc("remove all sections that are not DWARF .dwo sections from file"));
static cl::opt<std::string>
    SplitDWO("split-dwo",
             cl::desc("equivalent to extract-dwo on the input file to "
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
  ErrorOr<std::unique_ptr<FileOutputBuffer>> BufferOrErr =
      FileOutputBuffer::create(File, Obj.totalSize(),
                               FileOutputBuffer::F_executable);
  if (BufferOrErr.getError())
    error("failed to open " + OutputFilename);
  else
    Buffer = std::move(*BufferOrErr);
  Obj.write(*Buffer);
  if (auto EC = Buffer->commit())
    reportError(File, EC);
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

void CopyBinary(const ELFObjectFile<ELF64LE> &ObjFile) {
  std::unique_ptr<Object<ELF64LE>> Obj;

  if (!OutputFormat.empty() && OutputFormat != "binary")
    error("invalid output format '" + OutputFormat + "'");
  if (!OutputFormat.empty() && OutputFormat == "binary")
    Obj = llvm::make_unique<BinaryObject<ELF64LE>>(ObjFile);
  else
    Obj = llvm::make_unique<ELFObject<ELF64LE>>(ObjFile);

  if (!SplitDWO.empty())
    SplitDWOToFile<ELF64LE>(ObjFile, SplitDWO.getValue());

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

  if (StripSections) {
    RemovePred = [RemovePred](const SectionBase &Sec) {
      return RemovePred(Sec) || (Sec.Flags & SHF_ALLOC) == 0;
    };
    Obj->WriteSectionHeaders = false;
  }

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
  if (ELFObjectFile<ELF64LE> *o = dyn_cast<ELFObjectFile<ELF64LE>>(&Binary)) {
    CopyBinary(*o);
    return 0;
  }
  reportError(InputFilename, object_error::invalid_file_type);
}
