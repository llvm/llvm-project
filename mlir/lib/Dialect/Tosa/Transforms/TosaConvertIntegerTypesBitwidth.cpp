//===- TosaConvertIntegerTypesBitwidth.cpp - Convert TOSA integers --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/DialectResourceBlobManager.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"

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
#define GEN_PASS_DEF_TOSACONVERTINTEGERTYPESBITWIDTHPASS
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace mlir::tosa

using namespace mlir;

namespace {
APInt convertValue(const APInt &value, Type type, unsigned width) {
  return value.sextOrTrunc(width);
}

// Convert a validated byte-aligned payload, retaining storage for every
// element. In particular, each i4 element occupies one byte, not half a byte.
SmallVector<char> convertRawValues(ArrayRef<char> rawData,
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
      return IntegerType::get(type.getContext(), mapping->second);
    });
    addReplacement(
        [this](IntegerAttr attr) { return skip(convertInteger(attr)); });
    addReplacement(
        [this](DenseArrayAttr attr) { return skip(convertDenseArray(attr)); });
    addReplacement([this](DenseTypedElementsAttr attr) {
      return skip(convertDenseElements(attr));
    });
    addReplacement([this](DenseResourceElementsAttr attr) {
      return skip(convertDenseResource(attr));
    });
  }

  bool hasFailed() const { return conversionFailed; }

private:
  Attribute convertInteger(IntegerAttr attr) {
    Type type = attr.getType();
    Type convertedType = replace(type);
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
      if (!convertedElementType)
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

  Attribute convertDenseResource(DenseResourceElementsAttr attr) {
    ShapedType type = attr.getType();
    Type elementType = type.getElementType();
    auto convertedType = dyn_cast_or_null<ShapedType>(replace(type));
    if (!convertedType)
      return nullptr;
    Type convertedElementType = convertedType.getElementType();
    if (elementType == convertedElementType)
      return DenseResourceElementsAttr::get(convertedType, attr.getRawHandle());
    auto convertedIntegerType = dyn_cast<IntegerType>(convertedElementType);
    if (!convertedIntegerType)
      return nullptr;
    auto integerType = cast<IntegerType>(elementType);
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
    unsigned alignment = llvm::PowerOf2Ceil(
        llvm::divideCeil(convertedIntegerType.getWidth(), 8));
    return DenseResourceElementsAttr::get(
        convertedType, attr.getRawHandle().getKey(),
        HeapAsmResourceBlob::allocateAndCopyWithAlign(data, alignment));
  }

  // Payload handlers already converted their subelements. Do not apply a
  // second mapping when a destination width is also a source width.
  ReplaceFnResult<Attribute> skip(Attribute attr) {
    // replaceElementsIn ignores null replacements, so track failures here.
    conversionFailed |= !attr;
    return std::make_pair(attr, WalkResult::skip());
  }

  bool conversionFailed = false;
};

struct TosaConvertIntegerTypesBitwidth
    : tosa::impl::TosaConvertIntegerTypesBitwidthPassBase<TosaConvertIntegerTypesBitwidth> {
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
      // These widths are fixed by boolean flags, axes, structural arrays and
      // sparse indices. Reject these sources rather than exempting those
      // attributes.
      if (source == 1 || source == 32 || source == 64) {
        getOperation().emitError("unsupported source integer bitwidth ")
            << source;
        return signalPassFailure();
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
    replacer.recursivelyReplaceElementsIn(getOperation(), /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);
    if (replacer.hasFailed()) {
      getOperation().emitError("failed to convert integer types or attributes");
      signalPassFailure();
    }
  }
};
} // namespace
