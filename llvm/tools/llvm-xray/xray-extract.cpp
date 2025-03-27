//===- xray-extract.cpp: XRay Instrumentation Map Extraction --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the xray-extract.h interface.
//
// FIXME: Support other XRay-instrumented binary formats other than ELF.
//
//===----------------------------------------------------------------------===//


#include "func-id-helper.h"
#include "xray-registry.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/XRay/InstrumentationMap.h"

#include <assert.h>

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
                                cl::desc("Alias for -output"));
static cl::opt<bool> ExtractSymbolize("symbolize", cl::value_desc("symbolize"),
                                      cl::init(false),
                                      cl::desc("symbolize functions"),
                                      cl::sub(Extract));
static cl::alias ExtractSymbolize2("s", cl::aliasopt(ExtractSymbolize),
                                   cl::desc("alias for -symbolize"));
static cl::opt<bool> Demangle("demangle",
                              cl::desc("demangle symbols (default)"),
                              cl::sub(Extract));
static cl::opt<bool> NoDemangle("no-demangle",
                                cl::desc("don't demangle symbols"),
                                cl::sub(Extract));
static cl::opt<bool> FromMapping("mapping", cl::init(false),
                             cl::desc("Create instrumentation map from object map YAML"),
                             cl::sub(Extract));

namespace {

struct YAMLXRayObjectMapEntry {
  int32_t ObjId;
  std::string Path;
};

struct YAMLXRayObjectMapping {
  int NumObjBits;
  std::vector<YAMLXRayObjectMapEntry> Objects;
};

}

namespace llvm{
namespace yaml {
template <> struct MappingTraits<YAMLXRayObjectMapEntry> {
  static void mapping(IO &IO, YAMLXRayObjectMapEntry &Entry) {
    IO.mapRequired("id", Entry.ObjId);
    IO.mapRequired("path", Entry.Path);
  }
};

template <> struct MappingTraits<YAMLXRayObjectMapping> {
  static void mapping(IO &IO, YAMLXRayObjectMapping &Mapping) {
    IO.mapRequired("num_object_bits", Mapping.NumObjBits);
    IO.mapRequired("objects", Mapping.Objects);
  }
};
} // end namespace yaml
} // end namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(YAMLXRayObjectMapEntry)

namespace {

Error ReadObjectMappingYAML(StringRef Filename, YAMLXRayObjectMapping& Mapping) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> FileBufferOrErr =
      llvm::MemoryBuffer::getFile(ExtractInput);
  if (!FileBufferOrErr) {
    return joinErrors(make_error<StringError>(
                          Twine("Cannot read object mapping YAML from '") +
                              ExtractInput + "'.",
                          std::make_error_code(std::errc::invalid_argument)),
                      errorCodeToError(FileBufferOrErr.getError()));
  }

  yaml::Input In((*FileBufferOrErr)->getBuffer());
  In >> Mapping;
  if (In.error())
    return make_error<StringError>(
        Twine("Failed loading YAML document from '") + Filename + "'.",
        In.error());
  return Error::success();
}

struct IdMappingHelper {
  IdMappingHelper(int NumObjBits) : NumObjBits(NumObjBits) {
    assert(NumObjBits >= 0 && NumObjBits < 32 && "Invalid NumObjBits");
    NumFnBits = 32 - NumObjBits;
    ObjBitMask = (1l << NumObjBits) - 1;
    FnBitMask = (1l << NumFnBits) - 1;
  }

  int32_t MapId(int32_t FnId, int32_t ObjId) const {
    return ((ObjId & ObjBitMask) << NumFnBits) | (FnId & FnBitMask);
  }
private:
  int NumObjBits;
  int NumFnBits;
  int32_t ObjBitMask;
  int32_t FnBitMask;
};


void TranslateAndAppendSleds(const InstrumentationMap &Map,
                             FuncIdConversionHelper &FH,
                             int ObjId, const IdMappingHelper& IdMapping,
                             std::vector<YAMLXRaySledEntry>& YAMLSleds) {
  auto Sleds = Map.sleds();
  auto SledCount = std::distance(Sleds.begin(), Sleds.end());
  YAMLSleds.reserve(YAMLSleds.size() + SledCount);
  for (const auto &Sled : Sleds) {
    auto FuncId = Map.getFunctionId(Sled.Function);
    if (!FuncId)
      return;
    auto MappedId = IdMapping.MapId(*FuncId, ObjId);
    YAMLSleds.push_back(
        {MappedId, Sled.Address, Sled.Function, Sled.Kind, Sled.AlwaysInstrument,
         ExtractSymbolize ? FH.SymbolOrNumber(*FuncId) : "", Sled.Version});
  }
}

} // namespace

static CommandRegistration Unused(&Extract, []() -> Error {
  int NumObjBits{0};
  std::unordered_map<int, std::string> Inputs;
  if (FromMapping) {
    YAMLXRayObjectMapping ObjMapping;

    auto Err = ReadObjectMappingYAML(ExtractInput, ObjMapping);
    if (Err) {
      return Err;
    }
    NumObjBits = ObjMapping.NumObjBits;
    for (auto& Obj : ObjMapping.Objects) {
      Inputs[Obj.ObjId] = Obj.Path;
    }
  } else {
    Inputs[0] = ExtractInput;
  }

  IdMappingHelper IdMapping(NumObjBits);

  symbolize::LLVMSymbolizer::Options opts;
  if (Demangle.getPosition() < NoDemangle.getPosition())
    opts.Demangle = false;
  symbolize::LLVMSymbolizer Symbolizer(opts);

  std::vector<YAMLXRaySledEntry> YAMLSleds;

  for (auto& [ObjId, Path] : Inputs) {
    auto InstrumentationMapOrError = loadInstrumentationMap(Path);
    if (!InstrumentationMapOrError)
      return joinErrors(make_error<StringError>(
                            Twine("Cannot extract instrumentation map from '") +
                                Path + "'.",
                            std::make_error_code(std::errc::invalid_argument)),
                        InstrumentationMapOrError.takeError());

    const auto &FunctionAddresses =
        InstrumentationMapOrError->getFunctionAddresses();

    llvm::xray::FuncIdConversionHelper FuncIdHelper(Path, Symbolizer,
                                                    FunctionAddresses);
    TranslateAndAppendSleds(*InstrumentationMapOrError, FuncIdHelper,
                            ObjId, IdMapping, YAMLSleds);
  }

  std::error_code EC;
  raw_fd_ostream OS(ExtractOutput, EC, sys::fs::OpenFlags::OF_TextWithCRLF);
  if (EC)
    return make_error<StringError>(
        Twine("Cannot open file '") + ExtractOutput + "' for writing.", EC);
  Output Out(OS, nullptr, 0);
  Out << YAMLSleds;
  return Error::success();
});


