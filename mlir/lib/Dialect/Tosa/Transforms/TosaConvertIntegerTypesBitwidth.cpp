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
// Convert a validated byte-aligned payload, retaining storage for every
// element. In particular, each i4 element occupies one byte, not half a byte.
SmallVector<char> convertRawValues(ArrayRef<char> rawData, IntegerType type,
                                   IntegerType convertedType) {
  const unsigned inputElementSizeInBytes = llvm::divideCeil(type.getWidth(), 8);
  const unsigned outputElementSizeInBytes = convertedType.getWidth() / 8;
  const size_t numElements = rawData.size() / inputElementSizeInBytes;
  SmallVector<char> data(numElements * outputElementSizeInBytes);
  const auto *input = reinterpret_cast<const uint8_t *>(rawData.data());
  auto *output = reinterpret_cast<uint8_t *>(data.data());
  for (size_t i = 0; i < numElements; ++i) {
    APInt value(inputElementSizeInBytes * 8, 0);
    llvm::LoadIntFromMemory(value, input + i * inputElementSizeInBytes,
                            inputElementSizeInBytes);
    // Discard padding bits before interpreting the source sign bit.
    value = value.trunc(type.getWidth());
    llvm::StoreIntToMemory(value.sext(convertedType.getWidth()),
                           output + i * outputElementSizeInBytes,
                           outputElementSizeInBytes);
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
        attr.getValue().sext(cast<IntegerType>(convertedType).getWidth()));
  }

  Attribute convertDenseArray(DenseArrayAttr attr) {
    Type type = attr.getElementType();
    Type convertedType = replace(type);
    if (type == convertedType)
      return attr;
    // Unlike dense elements, arrays always store every element, even splats.
    const SmallVector<char> data =
        convertRawValues(attr.getRawData(), cast<IntegerType>(type),
                         cast<IntegerType>(convertedType));
    return DenseArrayAttr::get(convertedType, attr.size(), data);
  }

  Attribute convertDenseElements(DenseTypedElementsAttr attr) {
    auto convertedType = dyn_cast_or_null<ShapedType>(replace(attr.getType()));
    if (!convertedType)
      return nullptr;

    Type elementType = attr.getElementType();
    Type convertedElementType = convertedType.getElementType();
    if (elementType == convertedElementType) {
      // reshape is still necessary to handle any encoding changes.
      return attr.reshape(convertedType);
    }

    auto convertedIntegerType = dyn_cast<IntegerType>(convertedElementType);
    if (!convertedIntegerType)
      return nullptr;
    return attr
        .mapValues(convertedElementType,
                   [convertedIntegerType](const APInt &value) {
                     return value.sext(convertedIntegerType.getWidth());
                   })
        .reshape(convertedType);
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
    const unsigned elementSizeInBytes =
        llvm::divideCeil(integerType.getWidth(), 8);
    if (rawData.size() % elementSizeInBytes != 0)
      return nullptr;
    // Check resource is correct size.
    if (type.hasStaticShape() &&
        rawData.size() !=
            elementSizeInBytes * static_cast<uint64_t>(type.getNumElements()))
      return nullptr;
    const SmallVector<char> data =
        convertRawValues(rawData, integerType, convertedIntegerType);
    const unsigned alignment = convertedIntegerType.getWidth() / 8;
    return DenseResourceElementsAttr::get(
        convertedType, attr.getRawHandle().getKey(),
        HeapAsmResourceBlob::allocateAndCopyWithAlign(data, alignment));
  }

  // Payload handlers already converted their subelements.
  ReplaceFnResult<Attribute> skip(Attribute attr) {
    // replaceElementsIn ignores null replacements, so track failures here.
    conversionFailed |= !attr;
    return std::make_pair(attr, WalkResult::skip());
  }

  bool conversionFailed = false;
};

struct TosaConvertIntegerTypesBitwidth
    : tosa::impl::TosaConvertIntegerTypesBitwidthPassBase<
          TosaConvertIntegerTypesBitwidth> {
  using Base::Base;

  void runOnOperation() override {
    if (bitwidthMappings.empty()) {
      getOperation().emitError(
          "The 'bitwidth-mappings' option must be specified and non-empty.");
      return signalPassFailure();
    }
    DenseMap<unsigned, unsigned> mappings;
    for (auto [source, destination] : bitwidthMappings) {
      if (!((source == 4 && destination == 8) ||
            (source == 48 && destination == 64))) {
        getOperation().emitError("unsupported integer bitwidth mapping ")
            << source << ":" << destination << ": expected 4:8 or 48:64";
        return signalPassFailure();
      }
      mappings.try_emplace(source, destination);
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
