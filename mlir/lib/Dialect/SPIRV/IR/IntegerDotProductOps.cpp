//===- IntegerDotProductOps.cpp - MLIR SPIR-V Integer Dot Product Ops  ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the Integer Dot Product operations in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

#include "SPIRVOpUtils.h"
#include "SPIRVParsingUtils.h"

#include "llvm/Support/FormatVariadic.h"

using namespace mlir::spirv::AttrNames;

namespace mlir::spirv {

//===----------------------------------------------------------------------===//
// Integer Dot Product ops
//===----------------------------------------------------------------------===//

static LogicalResult verifyIntegerDotProduct(Operation *op) {
  assert(llvm::is_contained({2u, 3u}, op->getNumOperands()) &&
         "Not an integer dot product op?");
  assert(op->getNumResults() == 1 && "Expected a single result");

  Type factorTy = op->getOperand(0).getType();
  if (op->getOperand(1).getType() != factorTy)
    return op->emitOpError("requires the same type for both vector operands");

  unsigned expectedNumAttrs = 0;
  if (auto intTy = llvm::dyn_cast<IntegerType>(factorTy)) {
    ++expectedNumAttrs;
    auto packedVectorFormat =
        llvm::dyn_cast_or_null<spirv::PackedVectorFormatAttr>(
            op->getAttr(kPackedVectorFormatAttrName));
    if (!packedVectorFormat)
      return op->emitOpError("requires Packed Vector Format attribute for "
                             "integer vector operands");

    assert(packedVectorFormat.getValue() ==
               spirv::PackedVectorFormat::PackedVectorFormat4x8Bit &&
           "Unknown Packed Vector Format");
    if (intTy.getWidth() != 32)
      return op->emitOpError(
          llvm::formatv("with specified Packed Vector Format ({0}) requires "
                        "integer vector operands to be 32-bits wide",
                        packedVectorFormat.getValue()));
  } else {
    if (op->hasAttr(kPackedVectorFormatAttrName))
      return op->emitOpError(llvm::formatv(
          "with invalid format attribute for vector operands of type '{0}'",
          factorTy));
  }

  if (op->getAttrs().size() > expectedNumAttrs)
    return op->emitError(
        "op only supports the 'format' #spirv.packed_vector_format attribute");

  Type resultTy = op->getResultTypes().front();
  bool hasAccumulator = op->getNumOperands() == 3;
  if (hasAccumulator && op->getOperand(2).getType() != resultTy)
    return op->emitOpError(
        "requires the same accumulator operand and result types");

  unsigned factorBitWidth = getBitWidth(factorTy);
  unsigned resultBitWidth = getBitWidth(resultTy);
  if (factorBitWidth > resultBitWidth)
    return op->emitOpError(
        llvm::formatv("result type has insufficient bit-width ({0} bits) "
                      "for the specified vector operand type ({1} bits)",
                      resultBitWidth, factorBitWidth));

  return success();
}

static std::optional<spirv::Version> getIntegerDotProductMinVersion() {
  return spirv::Version::V_1_0; // Available in SPIR-V >= 1.0.
}

static std::optional<spirv::Version> getIntegerDotProductMaxVersion() {
  return spirv::Version::V_1_6; // Available in SPIR-V <= 1.6.
}

static SmallVector<ArrayRef<spirv::Extension>, 1>
getIntegerDotProductExtensions() {
  // Requires the SPV_KHR_integer_dot_product extension, specified either
  // explicitly or implied by target env's SPIR-V version >= 1.6.
  static const auto extension = spirv::Extension::SPV_KHR_integer_dot_product;
  return {extension};
}

static SmallVector<ArrayRef<spirv::Capability>, 1>
getIntegerDotProductCapabilities(Operation *op) {
  // Requires the the DotProduct capability and capabilities that depend on
  // exact op types.
  static const auto dotProductCap = spirv::Capability::DotProduct;
  static const auto dotProductInput4x8BitPackedCap =
      spirv::Capability::DotProductInput4x8BitPacked;
  static const auto dotProductInput4x8BitCap =
      spirv::Capability::DotProductInput4x8Bit;
  static const auto dotProductInputAllCap =
      spirv::Capability::DotProductInputAll;

  SmallVector<ArrayRef<spirv::Capability>, 1> capabilities = {dotProductCap};

  Type factorTy = op->getOperand(0).getType();
  if (auto intTy = llvm::dyn_cast<IntegerType>(factorTy)) {
    auto formatAttr = llvm::cast<spirv::PackedVectorFormatAttr>(
        op->getAttr(kPackedVectorFormatAttrName));
    if (formatAttr.getValue() ==
        spirv::PackedVectorFormat::PackedVectorFormat4x8Bit)
      capabilities.push_back(dotProductInput4x8BitPackedCap);

    return capabilities;
  }

  auto vecTy = llvm::cast<VectorType>(factorTy);
  if (vecTy.getElementTypeBitWidth() == 8) {
    capabilities.push_back(dotProductInput4x8BitCap);
    return capabilities;
  }

  capabilities.push_back(dotProductInputAllCap);
  return capabilities;
}

#define SPIRV_IMPL_INTEGER_DOT_PRODUCT_OP(OpName)                              \
  LogicalResult OpName::verify() { return verifyIntegerDotProduct(*this); }    \
  SmallVector<ArrayRef<spirv::Extension>, 1> OpName::getExtensions() {         \
    return getIntegerDotProductExtensions();                                   \
  }                                                                            \
  SmallVector<ArrayRef<spirv::Capability>, 1> OpName::getCapabilities() {      \
    return getIntegerDotProductCapabilities(*this);                            \
  }                                                                            \
  std::optional<spirv::Version> OpName::getMinVersion() {                      \
    return getIntegerDotProductMinVersion();                                   \
  }                                                                            \
  std::optional<spirv::Version> OpName::getMaxVersion() {                      \
    return getIntegerDotProductMaxVersion();                                   \
  }

SPIRV_IMPL_INTEGER_DOT_PRODUCT_OP(SDotOp)
SPIRV_IMPL_INTEGER_DOT_PRODUCT_OP(SUDotOp)
SPIRV_IMPL_INTEGER_DOT_PRODUCT_OP(UDotOp)
SPIRV_IMPL_INTEGER_DOT_PRODUCT_OP(SDotAccSatOp)
SPIRV_IMPL_INTEGER_DOT_PRODUCT_OP(SUDotAccSatOp)
SPIRV_IMPL_INTEGER_DOT_PRODUCT_OP(UDotAccSatOp)

#undef SPIRV_IMPL_INTEGER_DOT_PRODUCT_OP

} // namespace mlir::spirv
