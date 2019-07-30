//===-- llvm-lipo.cpp - a tool for manipulating universal binaries --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A utility for creating / splitting / inspecting universal binaries.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Triple.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;
using namespace llvm::object;

static const StringRef ToolName = "llvm-lipo";

LLVM_ATTRIBUTE_NORETURN static void reportError(Twine Message) {
  WithColor::error(errs(), ToolName) << Message << "\n";
  errs().flush();
  exit(EXIT_FAILURE);
}

LLVM_ATTRIBUTE_NORETURN static void reportError(StringRef File, Error E) {
  assert(E);
  std::string Buf;
  raw_string_ostream OS(Buf);
  logAllUnhandledErrors(std::move(E), OS);
  OS.flush();
  WithColor::error(errs(), ToolName) << "'" << File << "': " << Buf;
  exit(EXIT_FAILURE);
}

namespace {
enum LipoID {
  LIPO_INVALID = 0, // This is not an option ID.
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  LIPO_##ID,
#include "LipoOpts.inc"
#undef OPTION
};

// LipoInfoTable below references LIPO_##PREFIX. OptionGroup has prefix nullptr.
const char *const *LIPO_nullptr = nullptr;
#define PREFIX(NAME, VALUE) const char *const LIPO_##NAME[] = VALUE;
#include "LipoOpts.inc"
#undef PREFIX

static const opt::OptTable::Info LipoInfoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  {LIPO_##PREFIX, NAME,      HELPTEXT,                                         \
   METAVAR,       LIPO_##ID, opt::Option::KIND##Class,                         \
   PARAM,         FLAGS,     LIPO_##GROUP,                                     \
   LIPO_##ALIAS,  ALIASARGS, VALUES},
#include "LipoOpts.inc"
#undef OPTION
};

class LipoOptTable : public opt::OptTable {
public:
  LipoOptTable() : OptTable(LipoInfoTable) {}
};

enum class LipoAction {
  PrintArchs,
  PrintInfo,
  VerifyArch,
  ThinArch,
  CreateUniversal,
  ReplaceArch,
};

struct Replacement {
  StringRef ArchType;
  StringRef FileName;
};

struct Config {
  SmallVector<std::string, 1> InputFiles;
  SmallVector<std::string, 1> VerifyArchList;
  SmallVector<Replacement, 1> Replacements;
  std::string ThinArchType;
  std::string OutputFile;
  LipoAction ActionToPerform;
};

struct Slice {
  const MachOObjectFile *ObjectFile;
  // Requires Alignment field to store slice alignment values from universal
  // binaries. Also needed to order the slices using compareSlices, so the total
  // file size can be calculated before creating the output buffer.
  uint32_t Alignment;
};

} // end namespace

static void validateArchitectureName(StringRef ArchitectureName) {
  if (!MachOObjectFile::isValidArch(ArchitectureName)) {
    std::string Buf;
    raw_string_ostream OS(Buf);
    OS << "Invalid architecture: " << ArchitectureName
       << "\nValid architecture names are:";
    for (auto arch : MachOObjectFile::getValidArchs())
      OS << " " << arch;
    reportError(OS.str());
  }
}

static Config parseLipoOptions(ArrayRef<const char *> ArgsArr) {
  Config C;
  LipoOptTable T;
  unsigned MissingArgumentIndex, MissingArgumentCount;
  opt::InputArgList InputArgs =
      T.ParseArgs(ArgsArr, MissingArgumentIndex, MissingArgumentCount);

  if (MissingArgumentCount)
    reportError("missing argument to " +
                StringRef(InputArgs.getArgString(MissingArgumentIndex)) +
                " option");

  if (InputArgs.size() == 0) {
    // PrintHelp does not accept Twine.
    T.PrintHelp(errs(), "llvm-lipo input[s] option[s]", "llvm-lipo");
    exit(EXIT_FAILURE);
  }

  if (InputArgs.hasArg(LIPO_help)) {
    // PrintHelp does not accept Twine.
    T.PrintHelp(outs(), "llvm-lipo input[s] option[s]", "llvm-lipo");
    exit(EXIT_SUCCESS);
  }

  if (InputArgs.hasArg(LIPO_version)) {
    outs() << ToolName + "\n";
    cl::PrintVersionMessage();
    exit(EXIT_SUCCESS);
  }

  for (auto Arg : InputArgs.filtered(LIPO_UNKNOWN))
    reportError("unknown argument '" + Arg->getAsString(InputArgs) + "'");

  for (auto Arg : InputArgs.filtered(LIPO_INPUT))
    C.InputFiles.push_back(Arg->getValue());
  if (C.InputFiles.empty())
    reportError("at least one input file should be specified");

  if (InputArgs.hasArg(LIPO_output))
    C.OutputFile = InputArgs.getLastArgValue(LIPO_output);

  SmallVector<opt::Arg *, 1> ActionArgs(InputArgs.filtered(LIPO_action_group));
  if (ActionArgs.empty())
    reportError("at least one action should be specified");
  // errors if multiple actions specified other than replace
  // multiple replace flags may be specified, as long as they are not mixed with
  // other action flags
  auto ReplacementArgsRange = InputArgs.filtered(LIPO_replace);
  if (ActionArgs.size() > 1 &&
      ActionArgs.size() !=
          static_cast<size_t>(std::distance(ReplacementArgsRange.begin(),
                                            ReplacementArgsRange.end()))) {
    std::string Buf;
    raw_string_ostream OS(Buf);
    OS << "only one of the following actions can be specified:";
    for (auto Arg : ActionArgs)
      OS << " " << Arg->getSpelling();
    reportError(OS.str());
  }

  switch (ActionArgs[0]->getOption().getID()) {
  case LIPO_verify_arch:
    for (auto A : InputArgs.getAllArgValues(LIPO_verify_arch))
      C.VerifyArchList.push_back(A);
    if (C.VerifyArchList.empty())
      reportError(
          "verify_arch requires at least one architecture to be specified");
    if (C.InputFiles.size() > 1)
      reportError("verify_arch expects a single input file");
    C.ActionToPerform = LipoAction::VerifyArch;
    return C;

  case LIPO_archs:
    if (C.InputFiles.size() > 1)
      reportError("archs expects a single input file");
    C.ActionToPerform = LipoAction::PrintArchs;
    return C;

  case LIPO_info:
    C.ActionToPerform = LipoAction::PrintInfo;
    return C;

  case LIPO_thin:
    if (C.InputFiles.size() > 1)
      reportError("thin expects a single input file");
    C.ThinArchType = ActionArgs[0]->getValue();
    validateArchitectureName(C.ThinArchType);
    if (C.OutputFile.empty())
      reportError("thin expects a single output file");

    C.ActionToPerform = LipoAction::ThinArch;
    return C;

  case LIPO_create:
    if (C.OutputFile.empty())
      reportError("create expects a single output file to be specified");
    C.ActionToPerform = LipoAction::CreateUniversal;
    return C;

  case LIPO_replace:
    for (auto Action : ActionArgs) {
      if (!Action->getValue(1))
        reportError(
            "replace is missing an argument: expects -replace arch_type "
            "file_name");
      C.Replacements.push_back(
          Replacement{Action->getValue(0), Action->getValue(1)});
    }

    if (C.OutputFile.empty())
      reportError("replace expects a single output file to be specified");
    if (C.InputFiles.size() > 1)
      reportError("replace expects a single input file");
    C.ActionToPerform = LipoAction::ReplaceArch;
    return C;

  default:
    reportError("llvm-lipo action unspecified");
  }
}

static SmallVector<OwningBinary<Binary>, 1>
readInputBinaries(ArrayRef<std::string> InputFiles) {
  SmallVector<OwningBinary<Binary>, 1> InputBinaries;
  for (StringRef InputFile : InputFiles) {
    Expected<OwningBinary<Binary>> BinaryOrErr = createBinary(InputFile);
    if (!BinaryOrErr)
      reportError(InputFile, BinaryOrErr.takeError());
    // TODO: Add compatibility for archive files
    if (BinaryOrErr->getBinary()->isArchive())
      reportError("File " + InputFile +
                  " is an archive file and is not yet supported.");
    if (!BinaryOrErr->getBinary()->isMachO() &&
        !BinaryOrErr->getBinary()->isMachOUniversalBinary())
      reportError("File " + InputFile + " has unsupported binary format");
    InputBinaries.push_back(std::move(*BinaryOrErr));
  }
  return InputBinaries;
}

LLVM_ATTRIBUTE_NORETURN
static void verifyArch(ArrayRef<OwningBinary<Binary>> InputBinaries,
                       ArrayRef<std::string> VerifyArchList) {
  assert(!VerifyArchList.empty() &&
         "The list of architectures should be non-empty");
  assert(InputBinaries.size() == 1 && "Incorrect number of input binaries");

  for (StringRef Arch : VerifyArchList)
    validateArchitectureName(Arch);

  if (auto UO =
          dyn_cast<MachOUniversalBinary>(InputBinaries.front().getBinary())) {
    for (StringRef Arch : VerifyArchList) {
      Expected<std::unique_ptr<MachOObjectFile>> Obj =
          UO->getObjectForArch(Arch);
      if (!Obj)
        exit(EXIT_FAILURE);
    }
  } else if (auto O =
                 dyn_cast<MachOObjectFile>(InputBinaries.front().getBinary())) {
    const Triple::ArchType ObjectArch = O->getArch();
    for (StringRef Arch : VerifyArchList)
      if (ObjectArch != Triple(Arch).getArch())
        exit(EXIT_FAILURE);
  } else {
    llvm_unreachable("Unexpected binary format");
  }
  exit(EXIT_SUCCESS);
}

// Returns a string of the given Object file's architecture type
// Unknown architectures formatted unknown(CPUType,CPUSubType) for compatibility
// with cctools lipo
static std::string getArchString(const MachOObjectFile &ObjectFile) {
  const Triple T = ObjectFile.getArchTriple();
  const StringRef ObjectArch = T.getArchName();
  if (!ObjectArch.empty())
    return ObjectArch;
  return ("unknown(" + Twine(ObjectFile.getHeader().cputype) + "," +
          Twine(ObjectFile.getHeader().cpusubtype & ~MachO::CPU_SUBTYPE_MASK) +
          ")")
      .str();
}

static void printBinaryArchs(const Binary *Binary, raw_ostream &OS) {
  // Prints trailing space for compatibility with cctools lipo.
  if (auto UO = dyn_cast<MachOUniversalBinary>(Binary)) {
    for (const auto &O : UO->objects()) {
      Expected<std::unique_ptr<MachOObjectFile>> BinaryOrError =
          O.getAsObjectFile();
      if (!BinaryOrError)
        reportError(Binary->getFileName(), BinaryOrError.takeError());
      OS << getArchString(*BinaryOrError.get().get()) << " ";
    }
    OS << "\n";
    return;
  }
  OS << getArchString(*cast<MachOObjectFile>(Binary)) << " \n";
}

LLVM_ATTRIBUTE_NORETURN
static void printArchs(ArrayRef<OwningBinary<Binary>> InputBinaries) {
  assert(InputBinaries.size() == 1 && "Incorrect number of input binaries");
  printBinaryArchs(InputBinaries.front().getBinary(), outs());
  exit(EXIT_SUCCESS);
}

LLVM_ATTRIBUTE_NORETURN
static void printInfo(ArrayRef<OwningBinary<Binary>> InputBinaries) {
  // Group universal and thin files together for compatibility with cctools lipo
  for (auto &IB : InputBinaries) {
    const Binary *Binary = IB.getBinary();
    if (Binary->isMachOUniversalBinary()) {
      outs() << "Architectures in the fat file: " << Binary->getFileName()
             << " are: ";
      printBinaryArchs(Binary, outs());
    }
  }
  for (auto &IB : InputBinaries) {
    const Binary *Binary = IB.getBinary();
    if (!Binary->isMachOUniversalBinary()) {
      assert(Binary->isMachO() && "expected MachO binary");
      outs() << "Non-fat file: " << Binary->getFileName()
             << " is architecture: ";
      printBinaryArchs(Binary, outs());
    }
  }
  exit(EXIT_SUCCESS);
}

LLVM_ATTRIBUTE_NORETURN
static void extractSlice(ArrayRef<OwningBinary<Binary>> InputBinaries,
                         StringRef ThinArchType, StringRef OutputFileName) {
  assert(!ThinArchType.empty() && "The architecture type should be non-empty");
  assert(InputBinaries.size() == 1 && "Incorrect number of input binaries");
  assert(!OutputFileName.empty() && "Thin expects a single output file");

  if (InputBinaries.front().getBinary()->isMachO()) {
    reportError("input file " +
                InputBinaries.front().getBinary()->getFileName() +
                " must be a fat file when the -thin option is specified");
    exit(EXIT_FAILURE);
  }

  auto *UO = cast<MachOUniversalBinary>(InputBinaries.front().getBinary());
  Expected<std::unique_ptr<MachOObjectFile>> Obj =
      UO->getObjectForArch(ThinArchType);
  if (!Obj)
    reportError("fat input file " + UO->getFileName() +
                " does not contain the specified architecture " + ThinArchType +
                " to thin it to");

  Expected<std::unique_ptr<FileOutputBuffer>> OutFileOrError =
      FileOutputBuffer::create(OutputFileName,
                               Obj.get()->getMemoryBufferRef().getBufferSize(),
                               sys::fs::can_execute(UO->getFileName())
                                   ? FileOutputBuffer::F_executable
                                   : 0);
  if (!OutFileOrError)
    reportError(OutputFileName, OutFileOrError.takeError());
  std::copy(Obj.get()->getMemoryBufferRef().getBufferStart(),
            Obj.get()->getMemoryBufferRef().getBufferEnd(),
            OutFileOrError.get()->getBufferStart());
  if (Error E = OutFileOrError.get()->commit())
    reportError(OutputFileName, std::move(E));
  exit(EXIT_SUCCESS);
}

static void checkArchDuplicates(const ArrayRef<Slice> &Slices) {
  DenseMap<uint64_t, const MachOObjectFile *> CPUIds;
  auto CPUIDForSlice = [](const Slice &S) {
    return static_cast<uint64_t>(S.ObjectFile->getHeader().cputype) << 32 |
           S.ObjectFile->getHeader().cpusubtype;
  };
  for (const auto &S : Slices) {
    auto Entry = CPUIds.try_emplace(CPUIDForSlice(S), S.ObjectFile);
    if (!Entry.second)
      reportError(Entry.first->second->getFileName() + " and " +
                  S.ObjectFile->getFileName() + " have the same architecture " +
                  getArchString(*S.ObjectFile) +
                  " and therefore cannot be in the same universal binary");
  }
}

// For compatibility with cctools lipo, alignment is calculated as the minimum
// aligment of all segments. Each segments's alignment is the maximum alignment
// from its sections
static uint32_t calculateSegmentAlignment(const MachOObjectFile &O) {
  uint32_t P2CurrentAlignment;
  uint32_t P2MinAlignment = MachOUniversalBinary::MaxSectionAlignment;
  const bool Is64Bit = O.is64Bit();

  for (const auto &LC : O.load_commands()) {
    if (LC.C.cmd != (Is64Bit ? MachO::LC_SEGMENT_64 : MachO::LC_SEGMENT))
      continue;
    if (O.getHeader().filetype == MachO::MH_OBJECT) {
      unsigned NumberOfSections =
          (Is64Bit ? O.getSegment64LoadCommand(LC).nsects
                   : O.getSegmentLoadCommand(LC).nsects);
      P2CurrentAlignment = NumberOfSections ? 2 : P2MinAlignment;
      for (unsigned SI = 0; SI < NumberOfSections; ++SI) {
        P2CurrentAlignment = std::max(P2CurrentAlignment,
                                      (Is64Bit ? O.getSection64(LC, SI).align
                                               : O.getSection(LC, SI).align));
      }
    } else {
      P2CurrentAlignment =
          countTrailingZeros(Is64Bit ? O.getSegment64LoadCommand(LC).vmaddr
                                     : O.getSegmentLoadCommand(LC).vmaddr);
    }
    P2MinAlignment = std::min(P2MinAlignment, P2CurrentAlignment);
  }
  // return a value >= 4 byte aligned, and less than MachO MaxSectionAlignment
  return std::max(
      static_cast<uint32_t>(2),
      std::min(P2MinAlignment, static_cast<uint32_t>(
                                   MachOUniversalBinary::MaxSectionAlignment)));
}

static uint32_t calculateAlignment(const MachOObjectFile *ObjectFile) {
  switch (ObjectFile->getHeader().cputype) {
  case MachO::CPU_TYPE_I386:
  case MachO::CPU_TYPE_X86_64:
  case MachO::CPU_TYPE_POWERPC:
  case MachO::CPU_TYPE_POWERPC64:
    return 12; // log2 value of page size(4k) for x86 and PPC
  case MachO::CPU_TYPE_ARM:
  case MachO::CPU_TYPE_ARM64:
  case MachO::CPU_TYPE_ARM64_32:
    return 14; // log2 value of page size(16k) for Darwin ARM
  default:
    return calculateSegmentAlignment(*ObjectFile);
  }
}

// This function replicates ordering from cctools lipo for consistency
static bool compareSlices(const Slice &Lhs, const Slice &Rhs) {
  if (Lhs.ObjectFile->getHeader().cputype ==
      Rhs.ObjectFile->getHeader().cputype)
    return Lhs.ObjectFile->getHeader().cpusubtype <
           Rhs.ObjectFile->getHeader().cpusubtype;

  // force arm64-family to follow after all other slices for compatibility
  // with cctools lipo
  if (Lhs.ObjectFile->getHeader().cputype == MachO::CPU_TYPE_ARM64)
    return false;
  if (Rhs.ObjectFile->getHeader().cputype == MachO::CPU_TYPE_ARM64)
    return true;

  // Sort by alignment to minimize file size
  return Lhs.Alignment < Rhs.Alignment;
}

// Updates vector ExtractedObjects with the MachOObjectFiles extracted from
// Universal Binary files to transfer ownership.
static SmallVector<Slice, 2> buildSlices(
    ArrayRef<OwningBinary<Binary>> InputBinaries,
    SmallVectorImpl<std::unique_ptr<MachOObjectFile>> &ExtractedObjects) {
  SmallVector<Slice, 2> Slices;
  for (auto &IB : InputBinaries) {
    const Binary *InputBinary = IB.getBinary();
    if (auto UO = dyn_cast<MachOUniversalBinary>(InputBinary)) {
      for (const auto &O : UO->objects()) {
        Expected<std::unique_ptr<MachOObjectFile>> BinaryOrError =
            O.getAsObjectFile();
        if (!BinaryOrError)
          reportError(InputBinary->getFileName(), BinaryOrError.takeError());
        ExtractedObjects.push_back(std::move(BinaryOrError.get()));
        Slices.push_back(Slice{ExtractedObjects.back().get(), O.getAlign()});
      }
    } else if (auto O = dyn_cast<MachOObjectFile>(InputBinary)) {
      Slices.push_back(Slice{O, calculateAlignment(O)});
    } else {
      llvm_unreachable("Unexpected binary format");
    }
  }
  return Slices;
}

static SmallVector<MachO::fat_arch, 2>
buildFatArchList(ArrayRef<Slice> Slices) {
  SmallVector<MachO::fat_arch, 2> FatArchList;
  uint64_t Offset =
      sizeof(MachO::fat_header) + Slices.size() * sizeof(MachO::fat_arch);

  for (size_t Index = 0, Size = Slices.size(); Index < Size; ++Index) {
    Offset = alignTo(Offset, 1ull << Slices[Index].Alignment);
    const MachOObjectFile *ObjectFile = Slices[Index].ObjectFile;
    if (Offset > UINT32_MAX)
      reportError("fat file too large to be created because the offset "
                  "field in struct fat_arch is only 32-bits and the offset " +
                  Twine(Offset) + " for " + ObjectFile->getFileName() +
                  " for architecture " + getArchString(*ObjectFile) +
                  "exceeds that.");

    MachO::fat_arch FatArch;
    FatArch.cputype = ObjectFile->getHeader().cputype;
    FatArch.cpusubtype = ObjectFile->getHeader().cpusubtype;
    FatArch.offset = Offset;
    FatArch.size = ObjectFile->getMemoryBufferRef().getBufferSize();
    FatArch.align = Slices[Index].Alignment;
    Offset += FatArch.size;
    FatArchList.push_back(FatArch);
  }
  return FatArchList;
}

static void createUniversalBinary(SmallVectorImpl<Slice> &Slices,
                                  StringRef OutputFileName) {
  MachO::fat_header FatHeader;
  FatHeader.magic = MachO::FAT_MAGIC;
  FatHeader.nfat_arch = Slices.size();

  stable_sort(Slices, compareSlices);
  SmallVector<MachO::fat_arch, 2> FatArchList = buildFatArchList(Slices);

  const bool IsExecutable = any_of(Slices, [](Slice S) {
    return sys::fs::can_execute(S.ObjectFile->getFileName());
  });
  const uint64_t OutputFileSize =
      FatArchList.back().offset + FatArchList.back().size;
  Expected<std::unique_ptr<FileOutputBuffer>> OutFileOrError =
      FileOutputBuffer::create(OutputFileName, OutputFileSize,
                               IsExecutable ? FileOutputBuffer::F_executable
                                            : 0);
  if (!OutFileOrError)
    reportError(OutputFileName, OutFileOrError.takeError());
  std::unique_ptr<FileOutputBuffer> OutFile = std::move(OutFileOrError.get());
  std::memset(OutFile->getBufferStart(), 0, OutputFileSize);

  if (sys::IsLittleEndianHost)
    MachO::swapStruct(FatHeader);
  std::memcpy(OutFile->getBufferStart(), &FatHeader, sizeof(MachO::fat_header));

  for (size_t Index = 0, Size = Slices.size(); Index < Size; ++Index) {
    MemoryBufferRef BufferRef = Slices[Index].ObjectFile->getMemoryBufferRef();
    std::copy(BufferRef.getBufferStart(), BufferRef.getBufferEnd(),
              OutFile->getBufferStart() + FatArchList[Index].offset);
  }

  // FatArchs written after Slices in order to reduce the number of swaps for
  // the LittleEndian case
  if (sys::IsLittleEndianHost)
    for (MachO::fat_arch &FA : FatArchList)
      MachO::swapStruct(FA);
  std::memcpy(OutFile->getBufferStart() + sizeof(MachO::fat_header),
              FatArchList.begin(),
              sizeof(MachO::fat_arch) * FatArchList.size());

  if (Error E = OutFile->commit())
    reportError(OutputFileName, std::move(E));
}

LLVM_ATTRIBUTE_NORETURN
static void createUniversalBinary(ArrayRef<OwningBinary<Binary>> InputBinaries,
                                  StringRef OutputFileName) {
  assert(InputBinaries.size() >= 1 && "Incorrect number of input binaries");
  assert(!OutputFileName.empty() && "Create expects a single output file");

  SmallVector<std::unique_ptr<MachOObjectFile>, 1> ExtractedObjects;
  SmallVector<Slice, 1> Slices = buildSlices(InputBinaries, ExtractedObjects);
  checkArchDuplicates(Slices);
  createUniversalBinary(Slices, OutputFileName);

  exit(EXIT_SUCCESS);
}

static StringMap<Slice>
buildReplacementSlices(ArrayRef<OwningBinary<Binary>> ReplacementBinaries,
                       ArrayRef<Replacement> Replacements) {
  assert(ReplacementBinaries.size() == Replacements.size() &&
         "Number of replacment binaries does not match the number of "
         "replacements");
  StringMap<Slice> Slices;
  // populates StringMap of slices to replace with; error checks for mismatched
  // replace flag args, fat files, and duplicate arch_types
  for (size_t Index = 0, Size = Replacements.size(); Index < Size; ++Index) {
    StringRef ReplacementArch = Replacements[Index].ArchType;
    const Binary *ReplacementBinary = ReplacementBinaries[Index].getBinary();
    validateArchitectureName(ReplacementArch);

    auto O = dyn_cast<MachOObjectFile>(ReplacementBinary);
    if (!O)
      reportError("replacement file: " + ReplacementBinary->getFileName() +
                  " is a fat file (must be a thin file)");

    if (O->getArch() != Triple(ReplacementArch).getArch())
      reportError("specified architecture: " + ReplacementArch +
                  " for replacement file: " + ReplacementBinary->getFileName() +
                  " does not match the file's architecture");

    auto Entry =
        Slices.try_emplace(ReplacementArch, Slice{O, calculateAlignment(O)});
    if (!Entry.second)
      reportError("-replace " + ReplacementArch +
                  " <file_name> specified multiple times: " +
                  Entry.first->second.ObjectFile->getFileName() + ", " +
                  O->getFileName());
  }
  return Slices;
}

LLVM_ATTRIBUTE_NORETURN
static void replaceSlices(ArrayRef<OwningBinary<Binary>> InputBinaries,
                          StringRef OutputFileName,
                          ArrayRef<Replacement> Replacements) {
  assert(InputBinaries.size() == 1 && "Incorrect number of input binaries");
  assert(!OutputFileName.empty() && "Replace expects a single output file");

  if (InputBinaries.front().getBinary()->isMachO())
    reportError("input file " +
                InputBinaries.front().getBinary()->getFileName() +
                " must be a fat file when the -replace option is specified");

  SmallVector<std::string, 1> ReplacementFiles;
  for (const auto &R : Replacements)
    ReplacementFiles.push_back(R.FileName);
  SmallVector<OwningBinary<Binary>, 1> ReplacementBinaries =
      readInputBinaries(ReplacementFiles);

  StringMap<Slice> ReplacementSlices =
      buildReplacementSlices(ReplacementBinaries, Replacements);
  SmallVector<std::unique_ptr<MachOObjectFile>, 2> ExtractedObjects;
  SmallVector<Slice, 2> Slices = buildSlices(InputBinaries, ExtractedObjects);

  for (auto &Slice : Slices) {
    auto It = ReplacementSlices.find(getArchString(*Slice.ObjectFile));
    if (It != ReplacementSlices.end()) {
      Slice = It->second;
      ReplacementSlices.erase(It); // only keep remaining replacing arch_types
    }
  }

  if (!ReplacementSlices.empty())
    reportError("-replace " + ReplacementSlices.begin()->first() +
                " <file_name> specified but fat file: " +
                InputBinaries.front().getBinary()->getFileName() +
                " does not contain that architecture");
  createUniversalBinary(Slices, OutputFileName);
  exit(EXIT_SUCCESS);
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  Config C = parseLipoOptions(makeArrayRef(argv + 1, argc));
  SmallVector<OwningBinary<Binary>, 1> InputBinaries =
      readInputBinaries(C.InputFiles);

  switch (C.ActionToPerform) {
  case LipoAction::VerifyArch:
    verifyArch(InputBinaries, C.VerifyArchList);
    break;
  case LipoAction::PrintArchs:
    printArchs(InputBinaries);
    break;
  case LipoAction::PrintInfo:
    printInfo(InputBinaries);
    break;
  case LipoAction::ThinArch:
    extractSlice(InputBinaries, C.ThinArchType, C.OutputFile);
    break;
  case LipoAction::CreateUniversal:
    createUniversalBinary(InputBinaries, C.OutputFile);
    break;
  case LipoAction::ReplaceArch:
    replaceSlices(InputBinaries, C.OutputFile, C.Replacements);
    break;
  }
  return EXIT_SUCCESS;
}
