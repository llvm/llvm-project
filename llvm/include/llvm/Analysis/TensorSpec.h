//===- TensorSpec.h - type descriptor for a tensor --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_ANALYSIS_TENSORSPEC_H
#define LLVM_ANALYSIS_TENSORSPEC_H

#include "llvm/Config/llvm-config.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/JSON.h"

#include <memory>
#include <vector>

namespace llvm {
/// TensorSpec encapsulates the specification of a tensor: its dimensions, or
/// "shape" (row-major), its type (see TensorSpec::getDataType specializations
/// for supported types), its name and port (see "TensorFlow: Large-Scale
/// Machine Learning on Heterogeneous Distributed Systems", section 4.2, para 2:
/// https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45166.pdf)
///
/// Known tensor types. The left part is the C type, the right is a name we
/// can use to identify the type (to implement TensorSpec equality checks), and
/// to use, if needed, when mapping to an underlying evaluator's type system.
/// The main requirement is that the C type we use has the same size and
/// encoding (e.g. endian-ness) as the one used by the evaluator.
#define SUPPORTED_TENSOR_TYPES(M)                                              \
  M(float, Float)                                                              \
  M(double, Double)                                                            \
  M(int8_t, Int8)                                                              \
  M(uint8_t, UInt8)                                                            \
  M(int16_t, Int16)                                                            \
  M(uint16_t, UInt16)                                                          \
  M(int32_t, Int32)                                                            \
  M(uint32_t, UInt32)                                                          \
  M(int64_t, Int64)                                                            \
  M(uint64_t, UInt64)

enum class TensorType {
  Invalid,
#define _TENSOR_TYPE_ENUM_MEMBERS(_, Name) Name,
  SUPPORTED_TENSOR_TYPES(_TENSOR_TYPE_ENUM_MEMBERS)
#undef _TENSOR_TYPE_ENUM_MEMBERS
};

class TensorSpec final {
public:
  template <typename T>
  static TensorSpec createSpec(const std::string &Name,
                               const std::vector<int64_t> &Shape,
                               int Port = 0) {
    return TensorSpec(Name, Port, getDataType<T>(), sizeof(T), Shape);
  }

  const std::string &name() const { return Name; }
  int port() const { return Port; }
  TensorType type() const { return Type; }
  const std::vector<int64_t> &shape() const { return Shape; }

  bool operator==(const TensorSpec &Other) const {
    return Name == Other.Name && Port == Other.Port && Type == Other.Type &&
           Shape == Other.Shape;
  }

  bool operator!=(const TensorSpec &Other) const { return !(*this == Other); }

  /// Get the number of elements in a tensor with this shape.
  size_t getElementCount() const { return ElementCount; }
  /// Get the size, in bytes, of one element.
  size_t getElementByteSize() const { return ElementSize; }

  template <typename T> bool isElementType() const {
    return getDataType<T>() == Type;
  }

private:
  TensorSpec(const std::string &Name, int Port, TensorType Type,
             size_t ElementSize, const std::vector<int64_t> &Shape);

  template <typename T> static TensorType getDataType();

  std::string Name;
  int Port = 0;
  TensorType Type = TensorType::Invalid;
  std::vector<int64_t> Shape;
  size_t ElementCount = 0;
  size_t ElementSize = 0;
};

/// Construct a TensorSpec from a JSON dictionary of the form:
/// { "name": <string>,
///   "port": <int>,
///   "type": <string. Use LLVM's types, e.g. float, double, int64_t>,
///   "shape": <array of ints> }
/// For the "type" field, see the C++ primitive types used in
/// TFUTILS_SUPPORTED_TYPES.
Optional<TensorSpec> getTensorSpecFromJSON(LLVMContext &Ctx,
                                           const json::Value &Value);

struct LoggedFeatureSpec {
  TensorSpec Spec;
  Optional<std::string> LoggingName;
  const std::string &getLoggingName() const {
    return LoggingName ? *LoggingName : Spec.name();
  }
};

/// Load the output specs. If SpecFileOverride is not empty, that path is used.
/// Otherwise, the file is assumed to be called 'output_spec.json' and be found
/// under ModelPath (the model directory).
/// The first output tensor name must match ExpectedDecisionName.
/// In case of error, the return is None and the error is logged.
Optional<std::vector<LoggedFeatureSpec>>
loadOutputSpecs(LLVMContext &Ctx, StringRef ExpectedDecisionName,
                StringRef ModelPath, StringRef SpecFileOverride = StringRef());

#define TFUTILS_GETDATATYPE_DEF(T, Name)                                       \
  template <> TensorType TensorSpec::getDataType<T>();
SUPPORTED_TENSOR_TYPES(TFUTILS_GETDATATYPE_DEF)

#undef TFUTILS_GETDATATYPE_DEF
} // namespace llvm

#endif // LLVM_ANALYSIS_TENSORSPEC_H
