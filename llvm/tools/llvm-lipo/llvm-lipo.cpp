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

#include "llvm/ADT/STLExtras.h"
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
#include "llvm/TextAPI/MachO/Architecture.h"

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
  ExtractArch,
  CreateUniversal,
  ReplaceArch,
};

struct InputFile {
  Optional<StringRef> ArchType;
  StringRef FileName;
};

struct Config {
  SmallVector<InputFile, 1> InputFiles;
  SmallVector<std::string, 1> VerifyArchList;
  SmallVector<InputFile, 1> ReplacementFiles;
  StringMap<const uint32_t> SegmentAlignments;
  std::string ArchType;
  std::string OutputFile;
  LipoAction ActionToPerform;
};

// For compatibility with cctools lipo, a file's alignment is calculated as the
// minimum aligment of all segments. For object files, the file's alignment is
// the maximum alignment of its sections.
static uint32_t calculateFileAlignment(const MachOObjectFile &O) {
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
    return calculateFileAlignment(*ObjectFile);
  }
}

class Slice {
  const Binary *B;
  uint32_t CPUType;
  uint32_t CPUSubType;
  std::string ArchName;

  // P2Alignment field stores slice alignment values from universal
  // binaries. This is also needed to order the slices so the total
  // file size can be calculated before creating the output buffer.
  uint32_t P2Alignment;

public:
  Slice(const MachOObjectFile *O, uint32_t Align)
      : B(O), CPUType(O->getHeader().cputype),
        CPUSubType(O->getHeader().cpusubtype),
        ArchName(std::string(O->getArchTriple().getArchName())),
        P2Alignment(Align) {}

  explicit Slice(const MachOObjectFile *O) : Slice(O, calculateAlignment(O)){};

  explicit Slice(const Archive *A) : B(A) {
    Error Err = Error::success();
    std::unique_ptr<MachOObjectFile> FO = nullptr;
    for (const Archive::Child &Child : A->children(Err)) {
      Expected<std::unique_ptr<Binary>> ChildOrErr = Child.getAsBinary();
      if (!ChildOrErr)
        reportError(A->getFileName(), ChildOrErr.takeError());
      Binary *Bin = ChildOrErr.get().get();
      if (Bin->isMachOUniversalBinary())
        reportError(("archive member " + Bin->getFileName() +
                     " is a fat file (not allowed in an archive)")
                        .str());
      if (!Bin->isMachO())
        reportError(("archive member " + Bin->getFileName() +
                     " is not a MachO file (not allowed in an archive)"));
      MachOObjectFile *O = cast<MachOObjectFile>(Bin);
      if (FO &&
          std::tie(FO->getHeader().cputype, FO->getHeader().cpusubtype) !=
              std::tie(O->getHeader().cputype, O->getHeader().cpusubtype)) {
        reportError(("archive member " + O->getFileName() + " cputype (" +
                     Twine(O->getHeader().cputype) + ") and cpusubtype(" +
                     Twine(O->getHeader().cpusubtype) +
                     ") does not match previous archive members cputype (" +
                     Twine(FO->getHeader().cputype) + ") and cpusubtype(" +
                     Twine(FO->getHeader().cpusubtype) +
                     ") (all members must match) " + FO->getFileName())
                        .str());
      }
      if (!FO) {
        ChildOrErr.get().release();
        FO.reset(O);
      }
    }
    if (Err)
      reportError(A->getFileName(), std::move(Err));
    if (!FO)
      reportError(("empty archive with no architecture specification: " +
                   A->getFileName() + " (can't determine architecture for it)")
                      .str());
    CPUType = FO->getHeader().cputype;
    CPUSubType = FO->getHeader().cpusubtype;
    ArchName = std::string(FO->getArchTriple().getArchName());
    // Replicate the behavior of cctools lipo.
    P2Alignment = FO->is64Bit() ? 3 : 2;
  }

  void setP2Alignment(uint32_t Align) { P2Alignment = Align; }

  const Binary *getBinary() const { return B; }

  uint32_t getCPUType() const { return CPUType; }

  uint32_t getCPUSubType() const { return CPUSubType; }

  uint32_t getP2Alignment() const { return P2Alignment; }

  uint64_t getCPUID() const {
    return static_cast<uint64_t>(CPUType) << 32 | CPUSubType;
  }

  std::string getArchString() const {
    if (!ArchName.empty())
      return ArchName;
    return ("unknown(" + Twine(CPUType) + "," +
            Twine(CPUSubType & ~MachO::CPU_SUBTYPE_MASK) + ")")
        .str();
  }

  friend bool operator<(const Slice &Lhs, const Slice &Rhs) {
    if (Lhs.CPUType == Rhs.CPUType)
      return Lhs.CPUSubType < Rhs.CPUSubType;
    // force arm64-family to follow after all other slices for
    // compatibility with cctools lipo
    if (Lhs.CPUType == MachO::CPU_TYPE_ARM64)
      return false;
    if (Rhs.CPUType == MachO::CPU_TYPE_ARM64)
      return true;
    // Sort by alignment to minimize file size
    return Lhs.P2Alignment < Rhs.P2Alignment;
  }
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
    C.InputFiles.push_back({None, Arg->getValue()});
  for (auto Arg : InputArgs.filtered(LIPO_arch)) {
    validateArchitectureName(Arg->getValue(0));
    if (!Arg->getValue(1))
      reportError(
          "arch is missing an argument: expects -arch arch_type file_name");
    C.InputFiles.push_back({StringRef(Arg->getValue(0)), Arg->getValue(1)});
  }

  if (C.InputFiles.empty())
    reportError("at least one input file should be specified");

  if (InputArgs.hasArg(LIPO_output))
    C.OutputFile = std::string(InputArgs.getLastArgValue(LIPO_output));

  for (auto Segalign : InputArgs.filtered(LIPO_segalign)) {
    if (!Segalign->getValue(1))
      reportError("segalign is missing an argument: expects -segalign "
                  "arch_type alignment_value");

    validateArchitectureName(Segalign->getValue(0));

    uint32_t AlignmentValue;
    if (!to_integer<uint32_t>(Segalign->getValue(1), AlignmentValue, 16))
      reportError("argument to -segalign <arch_type> " +
                  Twine(Segalign->getValue(1)) +
                  " (hex) is not a proper hexadecimal number");
    if (!isPowerOf2_32(AlignmentValue))
      reportError("argument to -segalign <arch_type> " +
                  Twine(Segalign->getValue(1)) +
                  " (hex) must be a non-zero power of two");
    if (Log2_32(AlignmentValue) > MachOUniversalBinary::MaxSectionAlignment)
      reportError(
          "argument to -segalign <arch_type> " + Twine(Segalign->getValue(1)) +
          " (hex) must be less than or equal to the maximum section align 2^" +
          Twine(MachOUniversalBinary::MaxSectionAlignment));
    auto Entry = C.SegmentAlignments.try_emplace(Segalign->getValue(0),
                                                 Log2_32(AlignmentValue));
    if (!Entry.second)
      reportError("-segalign " + Twine(Segalign->getValue(0)) +
                  " <alignment_value> specified multiple times: " +
                  Twine(1 << Entry.first->second) + ", " +
                  Twine(AlignmentValue));
  }

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
    if (C.OutputFile.empty())
      reportError("thin expects a single output file");
    C.ArchType = ActionArgs[0]->getValue();
    validateArchitectureName(C.ArchType);
    C.ActionToPerform = LipoAction::ThinArch;
    return C;

  case LIPO_extract:
    if (C.InputFiles.size() > 1)
      reportError("extract expects a single input file");
    if (C.OutputFile.empty())
      reportError("extract expects a single output file");
    C.ArchType = ActionArgs[0]->getValue();
    validateArchitectureName(C.ArchType);
    C.ActionToPerform = LipoAction::ExtractArch;
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
      validateArchitectureName(Action->getValue(0));
      C.ReplacementFiles.push_back(
          {StringRef(Action->getValue(0)), Action->getValue(1)});
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
readInputBinaries(ArrayRef<InputFile> InputFiles) {
  SmallVector<OwningBinary<Binary>, 1> InputBinaries;
  for (const InputFile &IF : InputFiles) {
    Expected<OwningBinary<Binary>> BinaryOrErr = createBinary(IF.FileName);
    if (!BinaryOrErr)
      reportError(IF.FileName, BinaryOrErr.takeError());
    const Binary *B = BinaryOrErr->getBinary();
    if (!B->isArchive() && !B->isMachO() && !B->isMachOUniversalBinary())
      reportError("File " + IF.FileName + " has unsupported binary format");
    if (IF.ArchType && (B->isMachO() || B->isArchive())) {
      const auto S = B->isMachO() ? Slice(cast<MachOObjectFile>(B))
                                  : Slice(cast<Archive>(B));
      const auto SpecifiedCPUType = MachO::getCPUTypeFromArchitecture(
                                        MachO::getArchitectureFromName(
                                            Triple(*IF.ArchType).getArchName()))
                                        .first;
      // For compatibility with cctools' lipo the comparison is relaxed just to
      // checking cputypes.
      if (S.getCPUType() != SpecifiedCPUType)
        reportError("specified architecture: " + *IF.ArchType +
                    " for file: " + B->getFileName() +
                    " does not match the file's architecture (" +
                    S.getArchString() + ")");
    }
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
      Expected<MachOUniversalBinary::ObjectForArch> Obj =
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

static void printBinaryArchs(const Binary *Binary, raw_ostream &OS) {
  // Prints trailing space for compatibility with cctools lipo.
  if (auto UO = dyn_cast<MachOUniversalBinary>(Binary)) {
    for (const auto &O : UO->objects()) {
      Expected<std::unique_ptr<MachOObjectFile>> MachOObjOrError =
          O.getAsObjectFile();
      if (MachOObjOrError) {
        OS << Slice(MachOObjOrError->get()).getArchString() << " ";
        continue;
      }
      Expected<std::unique_ptr<Archive>> ArchiveOrError = O.getAsArchive();
      if (ArchiveOrError) {
        consumeError(MachOObjOrError.takeError());
        OS << Slice(ArchiveOrError->get()).getArchString() << " ";
        continue;
      }
      consumeError(ArchiveOrError.takeError());
      reportError(Binary->getFileName(), MachOObjOrError.takeError());
    }
    OS << "\n";
    return;
  }
  OS << Slice(cast<MachOObjectFile>(Binary)).getArchString() << " \n";
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
static void thinSlice(ArrayRef<OwningBinary<Binary>> InputBinaries,
                      StringRef ArchType, StringRef OutputFileName) {
  assert(!ArchType.empty() && "The architecture type should be non-empty");
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
      UO->getMachOObjectForArch(ArchType);
  Expected<std::unique_ptr<Archive>> Ar = UO->getArchiveForArch(ArchType);
  if (!Obj && !Ar)
    reportError("fat input file " + UO->getFileName() +
                " does not contain the specified architecture " + ArchType +
                " to thin it to");
  Binary *B = Obj ? static_cast<Binary *>(Obj->get())
                  : static_cast<Binary *>(Ar->get());
  Expected<std::unique_ptr<FileOutputBuffer>> OutFileOrError =
      FileOutputBuffer::create(OutputFileName,
                               B->getMemoryBufferRef().getBufferSize(),
                               sys::fs::can_execute(UO->getFileName())
                                   ? FileOutputBuffer::F_executable
                                   : 0);
  if (!OutFileOrError)
    reportError(OutputFileName, OutFileOrError.takeError());
  std::copy(B->getMemoryBufferRef().getBufferStart(),
            B->getMemoryBufferRef().getBufferEnd(),
            OutFileOrError.get()->getBufferStart());
  if (Error E = OutFileOrError.get()->commit())
    reportError(OutputFileName, std::move(E));
  exit(EXIT_SUCCESS);
}

static void checkArchDuplicates(ArrayRef<Slice> Slices) {
  DenseMap<uint64_t, const Binary *> CPUIds;
  for (const auto &S : Slices) {
    auto Entry = CPUIds.try_emplace(S.getCPUID(), S.getBinary());
    if (!Entry.second)
      reportError(Entry.first->second->getFileName() + " and " +
                  S.getBinary()->getFileName() +
                  " have the same architecture " + S.getArchString() +
                  " and therefore cannot be in the same universal binary");
  }
}

template <typename Range>
static void updateAlignments(Range &Slices,
                             const StringMap<const uint32_t> &Alignments) {
  for (auto &Slice : Slices) {
    auto Alignment = Alignments.find(Slice.getArchString());
    if (Alignment != Alignments.end())
      Slice.setP2Alignment(Alignment->second);
  }
}

static void checkUnusedAlignments(ArrayRef<Slice> Slices,
                                  const StringMap<const uint32_t> &Alignments) {
  auto HasArch = [&](StringRef Arch) {
    return llvm::find_if(Slices, [Arch](Slice S) {
             return S.getArchString() == Arch;
           }) != Slices.end();
  };
  for (StringRef Arch : Alignments.keys())
    if (!HasArch(Arch))
      reportError("-segalign " + Arch +
                  " <value> specified but resulting fat file does not contain "
                  "that architecture ");
}

// Updates vector ExtractedObjects with the MachOObjectFiles extracted from
// Universal Binary files to transfer ownership.
static SmallVector<Slice, 2> buildSlices(
    ArrayRef<OwningBinary<Binary>> InputBinaries,
    const StringMap<const uint32_t> &Alignments,
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
        Slices.emplace_back(ExtractedObjects.back().get(), O.getAlign());
      }
    } else if (auto O = dyn_cast<MachOObjectFile>(InputBinary)) {
      Slices.emplace_back(O);
    } else if (auto A = dyn_cast<Archive>(InputBinary)) {
      Slices.emplace_back(A);
    } else {
      llvm_unreachable("Unexpected binary format");
    }
  }
  updateAlignments(Slices, Alignments);
  return Slices;
}

static SmallVector<MachO::fat_arch, 2>
buildFatArchList(ArrayRef<Slice> Slices) {
  SmallVector<MachO::fat_arch, 2> FatArchList;
  uint64_t Offset =
      sizeof(MachO::fat_header) + Slices.size() * sizeof(MachO::fat_arch);

  for (const auto &S : Slices) {
    Offset = alignTo(Offset, 1ull << S.getP2Alignment());
    if (Offset > UINT32_MAX)
      reportError("fat file too large to be created because the offset "
                  "field in struct fat_arch is only 32-bits and the offset " +
                  Twine(Offset) + " for " + S.getBinary()->getFileName() +
                  " for architecture " + S.getArchString() + "exceeds that.");

    MachO::fat_arch FatArch;
    FatArch.cputype = S.getCPUType();
    FatArch.cpusubtype = S.getCPUSubType();
    FatArch.offset = Offset;
    FatArch.size = S.getBinary()->getMemoryBufferRef().getBufferSize();
    FatArch.align = S.getP2Alignment();
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

  stable_sort(Slices);
  SmallVector<MachO::fat_arch, 2> FatArchList = buildFatArchList(Slices);

  const bool IsExecutable = any_of(Slices, [](Slice S) {
    return sys::fs::can_execute(S.getBinary()->getFileName());
  });
  const uint64_t OutputFileSize =
      static_cast<uint64_t>(FatArchList.back().offset) +
      FatArchList.back().size;
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
    MemoryBufferRef BufferRef = Slices[Index].getBinary()->getMemoryBufferRef();
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
                                  const StringMap<const uint32_t> &Alignments,
                                  StringRef OutputFileName) {
  assert(InputBinaries.size() >= 1 && "Incorrect number of input binaries");
  assert(!OutputFileName.empty() && "Create expects a single output file");

  SmallVector<std::unique_ptr<MachOObjectFile>, 1> ExtractedObjects;
  SmallVector<Slice, 1> Slices =
      buildSlices(InputBinaries, Alignments, ExtractedObjects);
  checkArchDuplicates(Slices);
  checkUnusedAlignments(Slices, Alignments);
  createUniversalBinary(Slices, OutputFileName);

  exit(EXIT_SUCCESS);
}

LLVM_ATTRIBUTE_NORETURN
static void extractSlice(ArrayRef<OwningBinary<Binary>> InputBinaries,
                         const StringMap<const uint32_t> &Alignments,
                         StringRef ArchType, StringRef OutputFileName) {
  assert(!ArchType.empty() &&
         "The architecture type should be non-empty");
  assert(InputBinaries.size() == 1 && "Incorrect number of input binaries");
  assert(!OutputFileName.empty() && "Thin expects a single output file");

  if (InputBinaries.front().getBinary()->isMachO()) {
    reportError("input file " +
                InputBinaries.front().getBinary()->getFileName() +
                " must be a fat file when the -extract option is specified");
  }

  SmallVector<std::unique_ptr<MachOObjectFile>, 2> ExtractedObjects;
  SmallVector<Slice, 2> Slices =
      buildSlices(InputBinaries, Alignments, ExtractedObjects);
  erase_if(Slices, [ArchType](const Slice &S) {
    return ArchType != S.getArchString();
  });

  if (Slices.empty())
    reportError(
        "fat input file " + InputBinaries.front().getBinary()->getFileName() +
        " does not contain the specified architecture " + ArchType);
  createUniversalBinary(Slices, OutputFileName);
  exit(EXIT_SUCCESS);
}

static StringMap<Slice>
buildReplacementSlices(ArrayRef<OwningBinary<Binary>> ReplacementBinaries,
                       const StringMap<const uint32_t> &Alignments) {
  StringMap<Slice> Slices;
  // populates StringMap of slices to replace with; error checks for mismatched
  // replace flag args, fat files, and duplicate arch_types
  for (const auto &OB : ReplacementBinaries) {
    const Binary *ReplacementBinary = OB.getBinary();
    auto O = dyn_cast<MachOObjectFile>(ReplacementBinary);
    if (!O)
      reportError("replacement file: " + ReplacementBinary->getFileName() +
                  " is a fat file (must be a thin file)");
    Slice S(O);
    auto Entry = Slices.try_emplace(S.getArchString(), S);
    if (!Entry.second)
      reportError("-replace " + S.getArchString() +
                  " <file_name> specified multiple times: " +
                  Entry.first->second.getBinary()->getFileName() + ", " +
                  O->getFileName());
  }
  auto SlicesMapRange = map_range(
      Slices, [](StringMapEntry<Slice> &E) -> Slice & { return E.getValue(); });
  updateAlignments(SlicesMapRange, Alignments);
  return Slices;
}

LLVM_ATTRIBUTE_NORETURN
static void replaceSlices(ArrayRef<OwningBinary<Binary>> InputBinaries,
                          const StringMap<const uint32_t> &Alignments,
                          StringRef OutputFileName,
                          ArrayRef<InputFile> ReplacementFiles) {
  assert(InputBinaries.size() == 1 && "Incorrect number of input binaries");
  assert(!OutputFileName.empty() && "Replace expects a single output file");

  if (InputBinaries.front().getBinary()->isMachO())
    reportError("input file " +
                InputBinaries.front().getBinary()->getFileName() +
                " must be a fat file when the -replace option is specified");

  SmallVector<OwningBinary<Binary>, 1> ReplacementBinaries =
      readInputBinaries(ReplacementFiles);

  StringMap<Slice> ReplacementSlices =
      buildReplacementSlices(ReplacementBinaries, Alignments);
  SmallVector<std::unique_ptr<MachOObjectFile>, 2> ExtractedObjects;
  SmallVector<Slice, 2> Slices =
      buildSlices(InputBinaries, Alignments, ExtractedObjects);

  for (auto &Slice : Slices) {
    auto It = ReplacementSlices.find(Slice.getArchString());
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

  checkUnusedAlignments(Slices, Alignments);
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
    thinSlice(InputBinaries, C.ArchType, C.OutputFile);
    break;
  case LipoAction::ExtractArch:
    extractSlice(InputBinaries, C.SegmentAlignments, C.ArchType, C.OutputFile);
    break;
  case LipoAction::CreateUniversal:
    createUniversalBinary(InputBinaries, C.SegmentAlignments, C.OutputFile);
    break;
  case LipoAction::ReplaceArch:
    replaceSlices(InputBinaries, C.SegmentAlignments, C.OutputFile,
                  C.ReplacementFiles);
    break;
  }
  return EXIT_SUCCESS;
}
