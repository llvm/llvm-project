//===- xray-extract.cc - XRay Instrumentation Map Extraction --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the xray-extract.h interface.
//
// FIXME: Support other XRay-instrumented binary formats other than ELF.
//
//===----------------------------------------------------------------------===//

#include <type_traits>
#include <utility>

#include "xray-extract.h"

#include "xray-registry.h"
#include "xray-sleds.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::xray;
using namespace llvm::yaml;

// llvm-xray extract
// ----------------------------------------------------------------------------
static cl::SubCommand Extract("extract", "Extract instrumentation maps");
static cl::opt<std::string> ExtractInput(cl::Positional,
                                         cl::desc("<input file>"), cl::Required,
                                         cl::sub(Extract));
static cl::opt<std::string>
    ExtractOutput("output", cl::value_desc("output file"), cl::init("-"),
                  cl::desc("output file; use '-' for stdout"),
                  cl::sub(Extract));
static cl::alias ExtractOutput2("o", cl::aliasopt(ExtractOutput),
                                cl::desc("Alias for -output"),
                                cl::sub(Extract));

struct YAMLXRaySledEntry {
  int32_t FuncId;
  Hex64 Address;
  Hex64 Function;
  SledEntry::FunctionKinds Kind;
  bool AlwaysInstrument;
};

namespace llvm {
namespace yaml {

template <> struct ScalarEnumerationTraits<SledEntry::FunctionKinds> {
  static void enumeration(IO &IO, SledEntry::FunctionKinds &Kind) {
    IO.enumCase(Kind, "function-enter", SledEntry::FunctionKinds::ENTRY);
    IO.enumCase(Kind, "function-exit", SledEntry::FunctionKinds::EXIT);
    IO.enumCase(Kind, "tail-exit", SledEntry::FunctionKinds::TAIL);
  }
};

template <> struct MappingTraits<YAMLXRaySledEntry> {
  static void mapping(IO &IO, YAMLXRaySledEntry &Entry) {
    IO.mapRequired("id", Entry.FuncId);
    IO.mapRequired("address", Entry.Address);
    IO.mapRequired("function", Entry.Function);
    IO.mapRequired("kind", Entry.Kind);
    IO.mapRequired("always-instrument", Entry.AlwaysInstrument);
  }

  static constexpr bool flow = true;
};
}
}

LLVM_YAML_IS_SEQUENCE_VECTOR(YAMLXRaySledEntry)

namespace {

llvm::Error LoadBinaryInstrELF(
    StringRef Filename, std::deque<SledEntry> &OutputSleds,
    InstrumentationMapExtractor::FunctionAddressMap &InstrMap,
    InstrumentationMapExtractor::FunctionAddressReverseMap &FunctionIds) {
  auto ObjectFile = object::ObjectFile::createObjectFile(Filename);

  if (!ObjectFile)
    return ObjectFile.takeError();

  // FIXME: Maybe support other ELF formats. For now, 64-bit Little Endian only.
  if (!ObjectFile->getBinary()->isELF())
    return make_error<StringError>(
        "File format not supported (only does ELF).",
        std::make_error_code(std::errc::not_supported));
  if (ObjectFile->getBinary()->getArch() != Triple::x86_64)
    return make_error<StringError>(
        "File format not supported (only does ELF little endian 64-bit).",
        std::make_error_code(std::errc::not_supported));

  // Find the section named "xray_instr_map".
  StringRef Contents = "";
  const auto &Sections = ObjectFile->getBinary()->sections();
  auto I = find_if(Sections, [&](object::SectionRef Section) {
    StringRef Name = "";
    if (Section.getName(Name))
      return false;
    return Name == "xray_instr_map";
  });
  if (I == Sections.end())
    return make_error<StringError>(
        "Failed to find XRay instrumentation map.",
        std::make_error_code(std::errc::not_supported));
  if (I->getContents(Contents))
    return make_error<StringError>(
        "Failed to get contents of 'xray_instr_map' section.",
        std::make_error_code(std::errc::executable_format_error));

  // Copy the instrumentation map data into the Sleds data structure.
  auto C = Contents.bytes_begin();
  static constexpr size_t ELF64SledEntrySize = 32;

  if ((C - Contents.bytes_end()) % ELF64SledEntrySize != 0)
    return make_error<StringError>(
        "Instrumentation map entries not evenly divisible by size of an XRay "
        "sled entry in ELF64.",
        std::make_error_code(std::errc::executable_format_error));

  int32_t FuncId = 1;
  uint64_t CurFn = 0;
  std::deque<SledEntry> Sleds;
  for (; C != Contents.bytes_end(); C += ELF64SledEntrySize) {
    DataExtractor Extractor(
        StringRef(reinterpret_cast<const char *>(C), ELF64SledEntrySize), true,
        8);
    Sleds.push_back({});
    auto &Entry = Sleds.back();
    uint32_t OffsetPtr = 0;
    Entry.Address = Extractor.getU64(&OffsetPtr);
    Entry.Function = Extractor.getU64(&OffsetPtr);
    auto Kind = Extractor.getU8(&OffsetPtr);
    switch (Kind) {
    case 0: // ENTRY
      Entry.Kind = SledEntry::FunctionKinds::ENTRY;
      break;
    case 1: // EXIT
      Entry.Kind = SledEntry::FunctionKinds::EXIT;
      break;
    case 2: // TAIL
      Entry.Kind = SledEntry::FunctionKinds::TAIL;
      break;
    default:
      return make_error<StringError>(
          Twine("Encountered unknown sled type ") + "'" + Twine(int32_t{Kind}) +
              "'.",
          std::make_error_code(std::errc::executable_format_error));
    }
    Entry.AlwaysInstrument = Extractor.getU8(&OffsetPtr) != 0;

    // We replicate the function id generation scheme implemented in the runtime
    // here. Ideally we should be able to break it out, or output this map from
    // the runtime, but that's a design point we can discuss later on. For now,
    // we replicate the logic and move on.
    if (CurFn == 0) {
      CurFn = Entry.Function;
      InstrMap[FuncId] = Entry.Function;
      FunctionIds[Entry.Function] = FuncId;
    }
    if (Entry.Function != CurFn) {
      ++FuncId;
      CurFn = Entry.Function;
      InstrMap[FuncId] = Entry.Function;
      FunctionIds[Entry.Function] = FuncId;
    }
  }
  OutputSleds = std::move(Sleds);
  return llvm::Error::success();
}

Error LoadYAMLInstrMap(
    StringRef Filename, std::deque<SledEntry> &Sleds,
    InstrumentationMapExtractor::FunctionAddressMap &InstrMap,
    InstrumentationMapExtractor::FunctionAddressReverseMap &FunctionIds) {
  int Fd;
  if (auto EC = sys::fs::openFileForRead(Filename, Fd))
    return make_error<StringError>(
        Twine("Failed opening file '") + Filename + "' for reading.", EC);

  uint64_t FileSize;
  if (auto EC = sys::fs::file_size(Filename, FileSize))
    return make_error<StringError>(
        Twine("Failed getting size of file '") + Filename + "'.", EC);

  std::error_code EC;
  sys::fs::mapped_file_region MappedFile(
      Fd, sys::fs::mapped_file_region::mapmode::readonly, FileSize, 0, EC);
  if (EC)
    return make_error<StringError>(
        Twine("Failed memory-mapping file '") + Filename + "'.", EC);

  std::vector<YAMLXRaySledEntry> YAMLSleds;
  Input In(StringRef(MappedFile.data(), MappedFile.size()));
  In >> YAMLSleds;
  if (In.error())
    return make_error<StringError>(
        Twine("Failed loading YAML document from '") + Filename + "'.",
        In.error());

  for (const auto &Y : YAMLSleds) {
    InstrMap[Y.FuncId] = Y.Function;
    FunctionIds[Y.Function] = Y.FuncId;
    Sleds.push_back(
        SledEntry{Y.Address, Y.Function, Y.Kind, Y.AlwaysInstrument});
  }
  return Error::success();
}

} // namespace

InstrumentationMapExtractor::InstrumentationMapExtractor(std::string Filename,
                                                         InputFormats Format,
                                                         Error &EC) {
  ErrorAsOutParameter ErrAsOutputParam(&EC);
  if (Filename.empty()) {
    EC = Error::success();
    return;
  }
  switch (Format) {
  case InputFormats::ELF: {
    EC = handleErrors(
        LoadBinaryInstrELF(Filename, Sleds, FunctionAddresses, FunctionIds),
        [&](std::unique_ptr<ErrorInfoBase> E) {
          return joinErrors(
              make_error<StringError>(
                  Twine("Cannot extract instrumentation map from '") +
                      Filename + "'.",
                  std::make_error_code(std::errc::executable_format_error)),
              std::move(E));
        });
    break;
  }
  case InputFormats::YAML: {
    EC = handleErrors(
        LoadYAMLInstrMap(Filename, Sleds, FunctionAddresses, FunctionIds),
        [&](std::unique_ptr<ErrorInfoBase> E) {
          return joinErrors(
              make_error<StringError>(
                  Twine("Cannot load YAML instrumentation map from '") +
                      Filename + "'.",
                  std::make_error_code(std::errc::executable_format_error)),
              std::move(E));
        });
    break;
  }
  }
}

void InstrumentationMapExtractor::exportAsYAML(raw_ostream &OS) {
  // First we translate the sleds into the YAMLXRaySledEntry objects in a deque.
  std::vector<YAMLXRaySledEntry> YAMLSleds;
  YAMLSleds.reserve(Sleds.size());
  for (const auto &Sled : Sleds) {
    YAMLSleds.push_back({FunctionIds[Sled.Function], Sled.Address,
                         Sled.Function, Sled.Kind, Sled.AlwaysInstrument});
  }
  Output Out(OS);
  Out << YAMLSleds;
}

static CommandRegistration Unused(&Extract, []() -> Error {
  Error Err = Error::success();
  xray::InstrumentationMapExtractor Extractor(
      ExtractInput, InstrumentationMapExtractor::InputFormats::ELF, Err);
  if (Err)
    return Err;

  std::error_code EC;
  raw_fd_ostream OS(ExtractOutput, EC, sys::fs::OpenFlags::F_Text);
  if (EC)
    return make_error<StringError>(
        Twine("Cannot open file '") + ExtractOutput + "' for writing.", EC);
  Extractor.exportAsYAML(OS);
  return Error::success();
});
