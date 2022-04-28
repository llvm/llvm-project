//===- TensorSpec.cpp - tensor type abstraction ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation file for the abstraction of a tensor type, and JSON loading
// utils.
//
//===----------------------------------------------------------------------===//
#include "llvm/Config/config.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/TensorSpec.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <numeric>

using namespace llvm;

namespace llvm {

#define TFUTILS_GETDATATYPE_IMPL(T, E)                                         \
  template <> TensorType TensorSpec::getDataType<T>() { return TensorType::E; }

SUPPORTED_TENSOR_TYPES(TFUTILS_GETDATATYPE_IMPL)

#undef TFUTILS_GETDATATYPE_IMPL

TensorSpec::TensorSpec(const std::string &Name, int Port, TensorType Type,
                       size_t ElementSize, const std::vector<int64_t> &Shape)
    : Name(Name), Port(Port), Type(Type), Shape(Shape),
      ElementCount(std::accumulate(Shape.begin(), Shape.end(), 1,
                                   std::multiplies<int64_t>())),
      ElementSize(ElementSize) {}

Optional<TensorSpec> getTensorSpecFromJSON(LLVMContext &Ctx,
                                           const json::Value &Value) {
  auto EmitError = [&](const llvm::Twine &Message) -> Optional<TensorSpec> {
    std::string S;
    llvm::raw_string_ostream OS(S);
    OS << Value;
    Ctx.emitError("Unable to parse JSON Value as spec (" + Message + "): " + S);
    return None;
  };
  // FIXME: accept a Path as a parameter, and use it for error reporting.
  json::Path::Root Root("tensor_spec");
  json::ObjectMapper Mapper(Value, Root);
  if (!Mapper)
    return EmitError("Value is not a dict");

  std::string TensorName;
  int TensorPort = -1;
  std::string TensorType;
  std::vector<int64_t> TensorShape;

  if (!Mapper.map<std::string>("name", TensorName))
    return EmitError("'name' property not present or not a string");
  if (!Mapper.map<std::string>("type", TensorType))
    return EmitError("'type' property not present or not a string");
  if (!Mapper.map<int>("port", TensorPort))
    return EmitError("'port' property not present or not an int");
  if (!Mapper.map<std::vector<int64_t>>("shape", TensorShape))
    return EmitError("'shape' property not present or not an int array");

#define PARSE_TYPE(T, E)                                                       \
  if (TensorType == #T)                                                        \
    return TensorSpec::createSpec<T>(TensorName, TensorShape, TensorPort);
  SUPPORTED_TENSOR_TYPES(PARSE_TYPE)
#undef PARSE_TYPE
  return None;
}

Optional<std::vector<LoggedFeatureSpec>>
loadOutputSpecs(LLVMContext &Ctx, StringRef ExpectedDecisionName,
                StringRef ModelPath, StringRef SpecFileOverride) {
  SmallVector<char, 128> OutputSpecsPath;
  StringRef FileName = SpecFileOverride;
  if (FileName.empty()) {
    llvm::sys::path::append(OutputSpecsPath, ModelPath, "output_spec.json");
    FileName = {OutputSpecsPath.data(), OutputSpecsPath.size()};
  }

  auto BufferOrError = MemoryBuffer::getFileOrSTDIN(FileName);
  if (!BufferOrError) {
    Ctx.emitError("Error opening output specs file: " + FileName + " : " +
                  BufferOrError.getError().message());
    return None;
  }
  auto ParsedJSONValues = json::parse(BufferOrError.get()->getBuffer());
  if (!ParsedJSONValues) {
    Ctx.emitError("Could not parse specs file: " + FileName);
    return None;
  }
  auto ValuesArray = ParsedJSONValues->getAsArray();
  if (!ValuesArray) {
    Ctx.emitError("Expected an array of {tensor_spec:<TensorSpec>, "
                  "logging_name:<name>} dictionaries");
    return None;
  }
  std::vector<LoggedFeatureSpec> Ret;
  for (const auto &Value : *ValuesArray)
    if (const auto *Obj = Value.getAsObject())
      if (const auto *SpecPart = Obj->get("tensor_spec"))
        if (auto TensorSpec = getTensorSpecFromJSON(Ctx, *SpecPart))
          if (auto LoggingName = Obj->getString("logging_name")) {
            if (!TensorSpec->isElementType<int64_t>() &&
                !TensorSpec->isElementType<int32_t>() &&
                !TensorSpec->isElementType<float>()) {
              Ctx.emitError(
                  "Only int64, int32, and float tensors are supported. "
                  "Found unsupported type for tensor named " +
                  TensorSpec->name());
              return None;
            }
            Ret.push_back({*TensorSpec, LoggingName->str()});
          }

  if (ValuesArray->size() != Ret.size()) {
    Ctx.emitError(
        "Unable to parse output spec. It should be a json file containing an "
        "array of dictionaries. Each dictionary must have a 'tensor_spec' key, "
        "with a json object describing a TensorSpec; and a 'logging_name' key, "
        "which is a string to use as name when logging this tensor in the "
        "training log.");
    return None;
  }
  if (Ret.empty() || *Ret[0].LoggingName != ExpectedDecisionName) {
    Ctx.emitError("The first output spec must describe the decision tensor, "
                  "and must have the logging_name " +
                  StringRef(ExpectedDecisionName));
    return None;
  }
  return Ret;
}
} // namespace llvm
