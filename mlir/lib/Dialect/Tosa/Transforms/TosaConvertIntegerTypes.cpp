//===- TosaConvertIntegerTypes.cpp - Convert TOSA integers ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/DialectResourceBlobManager.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace llvm::cl {
template <>
class parser<std::pair<unsigned, unsigned>>
    : public basic_parser<std::pair<unsigned, unsigned>> {
public:
  parser(Option &option) : basic_parser(option) {}

  bool parse(Option &option, StringRef argName, StringRef arg,
             std::pair<unsigned, unsigned> &value) {
    auto [source, destination] = arg.split(':');
    if (source.getAsInteger(10, value.first) ||
        destination.getAsInteger(10, value.second))
      return option.error("expected <source>:<destination> integer bitwidths",
                          argName);
    return false;
  }

  StringRef getValueName() const override { return "source:destination"; }

  static void print(raw_ostream &os,
                    const std::pair<unsigned, unsigned> &value) {
    os << value.first << ":" << value.second;
  }
};
} // namespace llvm::cl

namespace mlir::tosa {
#define GEN_PASS_DEF_TOSACONVERTINTEGERTYPESPASS
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace mlir::tosa

using namespace mlir;

namespace {
// Preserve inherent dense arrays and known structural scalars at the operation
// boundary. Extend this policy for additional storage constraints; all other
// attributes and operations participate in generic conversion. Identical values
// in discardable or nested metadata still follow the bitwidth mapping.
static bool shouldPreserveAttribute(Operation *op, NamedAttribute attr) {
  StringRef name = attr.getName().getValue();
  if (!op->getInherentAttr(name))
    return false;
  if (isa<DenseArrayAttr>(attr.getValue()))
    return true;
  if (isa<tensor::ConcatOp>(op) && name == "dim")
    return true;
  if (op->getName().getDialectNamespace() !=
      tosa::TosaDialect::getDialectNamespace())
    return false;

  // TOSA axes and boolean options have operation-defined storage types.
  // Identify these by name: a semantic i1 value is also a BoolAttr, and
  // scalar payloads such as clamp bounds must still be converted.
  return llvm::StringSwitch<bool>(name)
      .Cases({"axis", "local_bound", "inverse", "round", "input_unsigned",
              "output_unsigned", "scale32", "per_channel"},
             true)
      .Default(false);
}

static APInt convertValue(const APInt &value, Type type, unsigned width) {
  return type.isUnsignedInteger() ? value.zextOrTrunc(width)
                                  : value.sextOrTrunc(width);
}

// Convert a validated byte-aligned payload, retaining storage for every
// element. In particular, each i4 element occupies one byte, not half a byte.
static SmallVector<char> convertRawValues(ArrayRef<char> rawData,
                                          IntegerType type,
                                          IntegerType convertedType) {
  const unsigned inputBytes = llvm::divideCeil(type.getWidth(), 8);
  const unsigned outputBytes = llvm::divideCeil(convertedType.getWidth(), 8);
  const size_t numElements = rawData.size() / inputBytes;
  SmallVector<char> data(numElements * outputBytes);
  const auto *input = reinterpret_cast<const uint8_t *>(rawData.data());
  auto *output = reinterpret_cast<uint8_t *>(data.data());
  for (size_t i = 0; i < numElements; ++i) {
    APInt value(inputBytes * 8, 0);
    llvm::LoadIntFromMemory(value, input + i * inputBytes, inputBytes);
    // Discard padding bits before interpreting the source sign bit.
    value = value.trunc(type.getWidth());
    llvm::StoreIntToMemory(convertValue(value, type, convertedType.getWidth()),
                           output + i * outputBytes, outputBytes);
  }
  return data;
}

class IntegerTypeReplacer : public AttrTypeReplacer {
public:
  // The mappings must outlive this replacer.
  explicit IntegerTypeReplacer(const DenseMap<unsigned, unsigned> &mappings) {
    addReplacement([&mappings](IntegerType type) -> std::optional<Type> {
      auto mapping = mappings.find(type.getWidth());
      if (mapping == mappings.end())
        return std::nullopt;
      return IntegerType::get(type.getContext(), mapping->second,
                              type.getSignedness());
    });
    // Reject unsupported typed payloads rather than changing just their type.
    // Specific payload handlers below take precedence over this fallback.
    addReplacement([this](TypedAttr attr) -> std::optional<Attribute> {
      if (replace(attr.getType()) != attr.getType())
        return nullptr;
      return std::nullopt;
    });
    addReplacement(
        [this](IntegerAttr attr) { return skip(convertInteger(attr)); });
    addReplacement(
        [this](DenseArrayAttr attr) { return skip(convertDenseArray(attr)); });
    addReplacement([this](DenseTypedElementsAttr attr) {
      return skip(convertDenseElements(attr));
    });
    addReplacement([this](SparseElementsAttr attr) {
      return skip(convertSparseElements(attr));
    });
    addReplacement([this](DenseResourceElementsAttr attr) {
      return skip(convertDenseResource(attr));
    });
  }

private:
  Attribute convertInteger(IntegerAttr attr) {
    Type type = attr.getType();
    Type convertedType = replace(type);
    if (!convertedType)
      return nullptr;
    if (type == convertedType)
      return attr;
    return IntegerAttr::get(
        convertedType,
        convertValue(attr.getValue(), type,
                     cast<IntegerType>(convertedType).getWidth()));
  }

  Attribute convertDenseArray(DenseArrayAttr attr) {
    Type type = attr.getElementType();
    Type convertedType = replace(type);
    if (!convertedType)
      return nullptr;
    if (type == convertedType)
      return attr;
    // Dense array assembly only supports byte-sized integers and i1.
    unsigned width = cast<IntegerType>(convertedType).getWidth();
    if (width != 1 && width % 8 != 0)
      return nullptr;
    // Unlike dense elements, arrays always store every element, even splats.
    SmallVector<char> data =
        convertRawValues(attr.getRawData(), cast<IntegerType>(type),
                         cast<IntegerType>(convertedType));
    return DenseArrayAttr::get(convertedType, attr.size(), data);
  }

  Attribute convertDenseElements(DenseTypedElementsAttr attr) {
    auto convertedType = dyn_cast_or_null<ShapedType>(replace(attr.getType()));
    if (!convertedType)
      return nullptr;

    Type elementType = attr.getElementType();
    DenseElementsAttr convertedAttr = attr;
    if (elementType != convertedType.getElementType()) {
      auto convertedElementType =
          dyn_cast<IntegerType>(convertedType.getElementType());
      if (!isa<IntegerType>(elementType) || !convertedElementType)
        return nullptr;
      convertedAttr = attr.mapValues(
          convertedType.getElementType(),
          [elementType, convertedElementType](const APInt &value) {
            return convertValue(value, elementType,
                                convertedElementType.getWidth());
          });
    }
    return convertedAttr.reshape(convertedType);
  }

  Attribute convertSparseElements(SparseElementsAttr attr) {
    auto type = dyn_cast_or_null<ShapedType>(replace(attr.getType()));
    auto values =
        dyn_cast_or_null<DenseElementsAttr>(replace(attr.getValues()));
    if (!type || !values)
      return nullptr;
    // Sparse indices must remain i64 even when i64 values are converted.
    return SparseElementsAttr::get(type, attr.getIndices(), values);
  }

  Attribute convertDenseResource(DenseResourceElementsAttr attr) {
    ShapedType type = attr.getType();
    Type elementType = type.getElementType();
    auto convertedType = dyn_cast_or_null<ShapedType>(replace(type));
    if (!convertedType)
      return nullptr;
    Type convertedElementType = convertedType.getElementType();
    if (elementType == convertedElementType)
      return DenseResourceElementsAttr::get(convertedType, attr.getRawHandle());
    auto integerType = dyn_cast<IntegerType>(elementType);
    auto convertedIntegerType =
        dyn_cast_or_null<IntegerType>(convertedElementType);
    if (!integerType || !convertedIntegerType)
      return nullptr;
    const AsmResourceBlob *blob = attr.getRawHandle().getBlob();
    if (!blob)
      return nullptr;
    ArrayRef<char> rawData = blob->getData();
    const unsigned inputBytes = llvm::divideCeil(integerType.getWidth(), 8);
    if (rawData.size() % inputBytes != 0)
      return nullptr;
    if (type.hasStaticShape() &&
        rawData.size() !=
            inputBytes * static_cast<uint64_t>(type.getNumElements()))
      return nullptr;
    SmallVector<char> data =
        convertRawValues(rawData, integerType, convertedIntegerType);
    // Alignment must be a power of two, including for sub-byte and
    // non-power-of-two destination widths.
    unsigned alignment =
        std::min<uint64_t>(llvm::PowerOf2Ceil(llvm::divideCeil(
                               convertedIntegerType.getWidth(), 8)),
                           alignof(uint64_t));
    return DenseResourceElementsAttr::get(
        convertedType, attr.getRawHandle().getKey(),
        HeapAsmResourceBlob::allocateAndCopyWithAlign(data, alignment));
  }

  // Payload handlers already converted their subelements. Do not apply a
  // second mapping when a destination width is also a source width.
  static ReplaceFnResult<Attribute> skip(Attribute attr) {
    return std::make_pair(attr, WalkResult::skip());
  }
};

struct TosaConvertIntegerTypes
    : tosa::impl::TosaConvertIntegerTypesPassBase<TosaConvertIntegerTypes> {
  using Base::Base;

  void runOnOperation() override {
    if (bitwidthMappings.empty()) {
      getOperation().emitError(
          "The 'bitwidth-mappings' option must be specified and non-empty.");
      return signalPassFailure();
    }
    DenseMap<unsigned, unsigned> mappings;
    for (auto [source, destination] : bitwidthMappings) {
      for (auto [kind, width] : {std::make_pair("source", source),
                                 std::make_pair("destination", destination)}) {
        if (width == 0 || width > IntegerType::kMaxWidth) {
          getOperation().emitError("invalid ")
              << kind << " integer bitwidth " << width
              << ": expected a value between 1 and " << IntegerType::kMaxWidth;
          return signalPassFailure();
        }
      }
      auto [it, inserted] = mappings.try_emplace(source, destination);
      if (!inserted && it->second != destination) {
        getOperation().emitError(
            "conflicting mappings for source integer bitwidth ")
            << source << ": " << it->second << " and " << destination;
        return signalPassFailure();
      }
    }
    IntegerTypeReplacer replacer(mappings);
    WalkResult result = getOperation()->walk([&](Operation *op) -> WalkResult {
      bool invalid = false;
      SmallVector<NamedAttribute> convertedAttrs;
      for (NamedAttribute attr : op->getAttrs()) {
        // Preserve fixed-storage attributes at the operation boundary so the
        // same attribute can still convert as discardable or nested metadata.
        Attribute value = attr.getValue();
        if (!shouldPreserveAttribute(op, attr))
          value = replacer.replace(value);
        if (!value)
          invalid = true;
        else
          convertedAttrs.emplace_back(attr.getName(), value);
      }
      auto replaceValueType = [&](Value value) {
        if (Type type = replacer.replace(value.getType()))
          value.setType(type);
        else
          invalid = true;
      };
      for (Value value : op->getResults())
        replaceValueType(value);
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument arg : block.getArguments())
            replaceValueType(arg);

      if (invalid) {
        op->emitError("failed to legalize operation '")
            << op->getName()
            << "': failed to convert integer types or attributes";
        return WalkResult::interrupt();
      }
      op->setAttrs(convertedAttrs);
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};
} // namespace
