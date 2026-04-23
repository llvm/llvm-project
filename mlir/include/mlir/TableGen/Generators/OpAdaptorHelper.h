//===- OpAdaptorHelper.h - Helper for Op/OpAdaptor code gen -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares AttributeMetadata and OpOrAdaptorHelper, which are used
// to share attribute-access code generation between Op and OpAdaptor emitters.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_GENERATORS_OPADAPTORHELPER_H
#define MLIR_TABLEGEN_GENERATORS_OPADAPTORHELPER_H

#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Property.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <functional>
#include <optional>
#include <string>

namespace mlir {
namespace tblgen {

//===----------------------------------------------------------------------===//
// AttributeMetadata
//===----------------------------------------------------------------------===//

/// Metadata on a registered attribute. Given that attributes are stored in
/// sorted order on operations, we can use information from ODS to deduce the
/// number of required attributes less than and greater than each attribute,
/// allowing us to search only a subrange of the attributes in ODS-generated
/// getters.
struct AttributeMetadata {
  /// The attribute name.
  llvm::StringRef attrName;
  /// Whether the attribute is required.
  bool isRequired;
  /// The ODS attribute constraint. Not present for implicit attributes.
  std::optional<Attribute> constraint;
  /// The number of required attributes less than this attribute.
  unsigned lowerBound = 0;
  /// The number of required attributes greater than this attribute.
  unsigned upperBound = 0;
};

//===----------------------------------------------------------------------===//
// OpOrAdaptorHelper
//===----------------------------------------------------------------------===//

/// Helper class to select between OpAdaptor and Op code templates for
/// attribute-access code generation.
class OpOrAdaptorHelper {
public:
  OpOrAdaptorHelper(const Operator &op, bool emitForOp)
      : op(op), emitForOp(emitForOp) {
    computeAttrMetadata();
  }

  /// Object that wraps a functor in a stream operator for interop with
  /// llvm::formatv.
  class Formatter {
  public:
    template <typename Functor>
    Formatter(Functor &&func) : func(std::forward<Functor>(func)) {}

    std::string str() const {
      std::string result;
      llvm::raw_string_ostream os(result);
      os << *this;
      return os.str();
    }

  private:
    std::function<llvm::raw_ostream &(llvm::raw_ostream &)> func;

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                         const Formatter &fmt) {
      return fmt.func(os);
    }
  };

  /// Generate code for getting an attribute. The definition is in the .cpp
  /// file because it references a file-local format string constant.
  Formatter getAttr(llvm::StringRef attrName, bool isNamed = false) const;

  /// Generate code for getting the name of an attribute.
  Formatter getAttrName(llvm::StringRef attrName) const {
    return [this, attrName](llvm::raw_ostream &os) -> llvm::raw_ostream & {
      if (emitForOp)
        return os << op.getGetterName(attrName) << "AttrName()";
      return os << llvm::formatv("{0}::{1}AttrName(*odsOpName)",
                                 op.getCppClassName(),
                                 op.getGetterName(attrName));
    };
  }

  /// Get the code snippet for getting the named attribute range.
  llvm::StringRef getAttrRange() const {
    return emitForOp ? "(*this)->getAttrs()" : "odsAttrs";
  }

  /// Get the prefix code for emitting an error.
  Formatter emitErrorPrefix() const {
    return [this](llvm::raw_ostream &os) -> llvm::raw_ostream & {
      if (emitForOp)
        return os << "emitOpError(\"";
      return os << llvm::formatv("emitError(loc, \"'{0}' op ",
                                 op.getOperationName());
    };
  }

  /// Get the call to get an operand or segment of operands.
  Formatter getOperand(unsigned index) const {
    return [this, index](llvm::raw_ostream &os) -> llvm::raw_ostream & {
      return os << llvm::formatv(op.getOperand(index).isVariadic()
                                     ? "this->getODSOperands({0})"
                                     : "(*this->getODSOperands({0}).begin())",
                                 index);
    };
  }

  /// Get the call to get a result or segment of results.
  Formatter getResult(unsigned index) const {
    return [this, index](llvm::raw_ostream &os) -> llvm::raw_ostream & {
      if (!emitForOp)
        return os << "<no results should be generated>";
      return os << llvm::formatv(op.getResult(index).isVariadic()
                                     ? "this->getODSResults({0})"
                                     : "(*this->getODSResults({0}).begin())",
                                 index);
    };
  }

  /// Return whether an op instance is available.
  bool isEmittingForOp() const { return emitForOp; }

  /// Return the ODS operation wrapper.
  const Operator &getOp() const { return op; }

  /// Get the attribute metadata sorted by name.
  const llvm::MapVector<llvm::StringRef, AttributeMetadata> &
  getAttrMetadata() const {
    return attrMetadata;
  }

  /// Returns whether to emit a Properties struct for this operation or not.
  bool hasProperties() const {
    if (!op.getProperties().empty())
      return true;
    return true;
  }

  /// Returns whether the operation will have a non-empty Properties struct.
  bool hasNonEmptyPropertiesStruct() const {
    if (!op.getProperties().empty())
      return true;
    if (!hasProperties())
      return false;
    if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments") ||
        op.getTrait("::mlir::OpTrait::AttrSizedResultSegments"))
      return true;
    return llvm::any_of(
        getAttrMetadata(),
        [](const std::pair<llvm::StringRef, AttributeMetadata> &it) {
          return !it.second.constraint ||
                 !it.second.constraint->isDerivedAttr();
        });
  }

  std::optional<NamedProperty> &getOperandSegmentsSize() {
    return operandSegmentsSize;
  }

  std::optional<NamedProperty> &getResultSegmentsSize() {
    return resultSegmentsSize;
  }

  uint32_t getOperandSegmentSizesLegacyIndex() {
    return operandSegmentSizesLegacyIndex;
  }

  uint32_t getResultSegmentSizesLegacyIndex() {
    return resultSegmentSizesLegacyIndex;
  }

private:
  /// Compute the attribute metadata.
  void computeAttrMetadata();

  /// The operation ODS wrapper.
  const Operator &op;
  /// True if code is being generated for an op, false for an adaptor.
  const bool emitForOp;

  /// The attribute metadata, mapped by name.
  llvm::MapVector<llvm::StringRef, AttributeMetadata> attrMetadata;

  std::optional<NamedProperty> operandSegmentsSize;
  std::string operandSegmentsSizeStorage;
  std::string operandSegmentsSizeParser;
  std::optional<NamedProperty> resultSegmentsSize;
  std::string resultSegmentsSizeStorage;
  std::string resultSegmentsSizeParser;

  /// Indices storing the position in the emission order of the operand/result
  /// segment sizes attribute if emitted as part of the properties for legacy
  /// bytecode encodings (versions less than 6).
  uint32_t operandSegmentSizesLegacyIndex = 0;
  uint32_t resultSegmentSizesLegacyIndex = 0;

  /// The number of required attributes.
  unsigned numRequired;
};

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_GENERATORS_OPADAPTORHELPER_H
