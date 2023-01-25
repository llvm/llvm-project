//===- BytecodeImplementation.h - MLIR Bytecode Implementation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines various interfaces and utilities necessary for dialects
// to hook into bytecode serialization.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BYTECODE_BYTECODEIMPLEMENTATION_H
#define MLIR_BYTECODE_BYTECODEIMPLEMENTATION_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/Twine.h"

namespace mlir {
//===----------------------------------------------------------------------===//
// DialectBytecodeReader
//===----------------------------------------------------------------------===//

/// This class defines a virtual interface for reading a bytecode stream,
/// providing hooks into the bytecode reader. As such, this class should only be
/// derived and defined by the main bytecode reader, users (i.e. dialects)
/// should generally only interact with this class via the
/// BytecodeDialectInterface below.
class DialectBytecodeReader {
public:
  virtual ~DialectBytecodeReader() = default;

  /// Emit an error to the reader.
  virtual InFlightDiagnostic emitError(const Twine &msg = {}) = 0;

  /// Read out a list of elements, invoking the provided callback for each
  /// element. The callback function may be in any of the following forms:
  ///   * LogicalResult(T &)
  ///   * FailureOr<T>()
  template <typename T, typename CallbackFn>
  LogicalResult readList(SmallVectorImpl<T> &result, CallbackFn &&callback) {
    uint64_t size;
    if (failed(readVarInt(size)))
      return failure();
    result.reserve(size);

    for (uint64_t i = 0; i < size; ++i) {
      // Check if the callback uses FailureOr, or populates the result by
      // reference.
      if constexpr (llvm::function_traits<std::decay_t<CallbackFn>>::num_args) {
        T element = {};
        if (failed(callback(element)))
          return failure();
        result.emplace_back(std::move(element));
      } else {
        FailureOr<T> element = callback();
        if (failed(element))
          return failure();
        result.emplace_back(std::move(*element));
      }
    }
    return success();
  }

  //===--------------------------------------------------------------------===//
  // IR
  //===--------------------------------------------------------------------===//

  /// Read a reference to the given attribute.
  virtual LogicalResult readAttribute(Attribute &result) = 0;
  template <typename T>
  LogicalResult readAttributes(SmallVectorImpl<T> &attrs) {
    return readList(attrs, [this](T &attr) { return readAttribute(attr); });
  }
  template <typename T>
  LogicalResult readAttribute(T &result) {
    Attribute baseResult;
    if (failed(readAttribute(baseResult)))
      return failure();
    if ((result = baseResult.dyn_cast<T>()))
      return success();
    return emitError() << "expected " << llvm::getTypeName<T>()
                       << ", but got: " << baseResult;
  }

  /// Read a reference to the given type.
  virtual LogicalResult readType(Type &result) = 0;
  template <typename T>
  LogicalResult readTypes(SmallVectorImpl<T> &types) {
    return readList(types, [this](T &type) { return readType(type); });
  }
  template <typename T>
  LogicalResult readType(T &result) {
    Type baseResult;
    if (failed(readType(baseResult)))
      return failure();
    if ((result = baseResult.dyn_cast<T>()))
      return success();
    return emitError() << "expected " << llvm::getTypeName<T>()
                       << ", but got: " << baseResult;
  }

  /// Read a handle to a dialect resource.
  template <typename ResourceT>
  FailureOr<ResourceT> readResourceHandle() {
    FailureOr<AsmDialectResourceHandle> handle = readResourceHandle();
    if (failed(handle))
      return failure();
    if (auto *result = dyn_cast<ResourceT>(&*handle))
      return std::move(*result);
    return emitError() << "provided resource handle differs from the "
                          "expected resource type";
  }

  //===--------------------------------------------------------------------===//
  // Primitives
  //===--------------------------------------------------------------------===//

  /// Read a variable width integer.
  virtual LogicalResult readVarInt(uint64_t &result) = 0;

  /// Read a signed variable width integer.
  virtual LogicalResult readSignedVarInt(int64_t &result) = 0;
  LogicalResult readSignedVarInts(SmallVectorImpl<int64_t> &result) {
    return readList(result,
                    [this](int64_t &value) { return readSignedVarInt(value); });
  }

  /// Read an APInt that is known to have been encoded with the given width.
  virtual FailureOr<APInt> readAPIntWithKnownWidth(unsigned bitWidth) = 0;

  /// Read an APFloat that is known to have been encoded with the given
  /// semantics.
  virtual FailureOr<APFloat>
  readAPFloatWithKnownSemantics(const llvm::fltSemantics &semantics) = 0;

  /// Read a string from the bytecode.
  virtual LogicalResult readString(StringRef &result) = 0;

  /// Read a blob from the bytecode.
  virtual LogicalResult readBlob(ArrayRef<char> &result) = 0;

private:
  /// Read a handle to a dialect resource.
  virtual FailureOr<AsmDialectResourceHandle> readResourceHandle() = 0;
};

//===----------------------------------------------------------------------===//
// DialectBytecodeWriter
//===----------------------------------------------------------------------===//

/// This class defines a virtual interface for writing to a bytecode stream,
/// providing hooks into the bytecode writer. As such, this class should only be
/// derived and defined by the main bytecode writer, users (i.e. dialects)
/// should generally only interact with this class via the
/// BytecodeDialectInterface below.
class DialectBytecodeWriter {
public:
  virtual ~DialectBytecodeWriter() = default;

  //===--------------------------------------------------------------------===//
  // IR
  //===--------------------------------------------------------------------===//

  /// Write out a list of elements, invoking the provided callback for each
  /// element.
  template <typename RangeT, typename CallbackFn>
  void writeList(RangeT &&range, CallbackFn &&callback) {
    writeVarInt(llvm::size(range));
    for (auto &element : range)
      callback(element);
  }

  /// Write a reference to the given attribute.
  virtual void writeAttribute(Attribute attr) = 0;
  template <typename T>
  void writeAttributes(ArrayRef<T> attrs) {
    writeList(attrs, [this](T attr) { writeAttribute(attr); });
  }

  /// Write a reference to the given type.
  virtual void writeType(Type type) = 0;
  template <typename T>
  void writeTypes(ArrayRef<T> types) {
    writeList(types, [this](T type) { writeType(type); });
  }

  /// Write the given handle to a dialect resource.
  virtual void
  writeResourceHandle(const AsmDialectResourceHandle &resource) = 0;

  //===--------------------------------------------------------------------===//
  // Primitives
  //===--------------------------------------------------------------------===//

  /// Write a variable width integer to the output stream. This should be the
  /// preferred method for emitting integers whenever possible.
  virtual void writeVarInt(uint64_t value) = 0;

  /// Write a signed variable width integer to the output stream. This should be
  /// the preferred method for emitting signed integers whenever possible.
  virtual void writeSignedVarInt(int64_t value) = 0;
  void writeSignedVarInts(ArrayRef<int64_t> value) {
    writeList(value, [this](int64_t value) { writeSignedVarInt(value); });
  }

  /// Write an APInt to the bytecode stream whose bitwidth will be known
  /// externally at read time. This method is useful for encoding APInt values
  /// when the width is known via external means, such as via a type. This
  /// method should generally only be invoked if you need an APInt, otherwise
  /// use the varint methods above. APInt values are generally encoded using
  /// zigzag encoding, to enable more efficient encodings for negative values.
  virtual void writeAPIntWithKnownWidth(const APInt &value) = 0;

  /// Write an APFloat to the bytecode stream whose semantics will be known
  /// externally at read time. This method is useful for encoding APFloat values
  /// when the semantics are known via external means, such as via a type.
  virtual void writeAPFloatWithKnownSemantics(const APFloat &value) = 0;

  /// Write a string to the bytecode, which is owned by the caller and is
  /// guaranteed to not die before the end of the bytecode process. This should
  /// only be called if such a guarantee can be made, such as when the string is
  /// owned by an attribute or type.
  virtual void writeOwnedString(StringRef str) = 0;

  /// Write a blob to the bytecode, which is owned by the caller and is
  /// guaranteed to not die before the end of the bytecode process. The blob is
  /// written as-is, with no additional compression or compaction.
  virtual void writeOwnedBlob(ArrayRef<char> blob) = 0;
};

//===----------------------------------------------------------------------===//
// BytecodeDialectInterface
//===----------------------------------------------------------------------===//

class BytecodeDialectInterface
    : public DialectInterface::Base<BytecodeDialectInterface> {
public:
  using Base::Base;

  //===--------------------------------------------------------------------===//
  // Reading
  //===--------------------------------------------------------------------===//

  /// Read an attribute belonging to this dialect from the given reader. This
  /// method should return null in the case of failure.
  virtual Attribute readAttribute(DialectBytecodeReader &reader) const {
    reader.emitError() << "dialect " << getDialect()->getNamespace()
                       << " does not support reading attributes from bytecode";
    return Attribute();
  }

  /// Read a type belonging to this dialect from the given reader. This method
  /// should return null in the case of failure.
  virtual Type readType(DialectBytecodeReader &reader) const {
    reader.emitError() << "dialect " << getDialect()->getNamespace()
                       << " does not support reading types from bytecode";
    return Type();
  }

  //===--------------------------------------------------------------------===//
  // Writing
  //===--------------------------------------------------------------------===//

  /// Write the given attribute, which belongs to this dialect, to the given
  /// writer. This method may return failure to indicate that the given
  /// attribute could not be encoded, in which case the textual format will be
  /// used to encode this attribute instead.
  virtual LogicalResult writeAttribute(Attribute attr,
                                       DialectBytecodeWriter &writer) const {
    return failure();
  }

  /// Write the given type, which belongs to this dialect, to the given writer.
  /// This method may return failure to indicate that the given type could not
  /// be encoded, in which case the textual format will be used to encode this
  /// type instead.
  virtual LogicalResult writeType(Type type,
                                  DialectBytecodeWriter &writer) const {
    return failure();
  }
};

} // namespace mlir

#endif // MLIR_BYTECODE_BYTECODEIMPLEMENTATION_H
