//===- QuantUtils.cpp ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TOSA numerical support functions and quantization attribute builders
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

namespace mlir {
namespace tosa {

std::string tosa_quantized_type = "quint8";

namespace {

void computeMultiplierAndShiftTosaScale16(double scale, int32_t &multiplier,
                                          int32_t &shift) {

  /* Generates mantissa and shift values where mantissa is in [-1.0,-0.5] or
     [0.5, 1.0] such that
     multiplier = mantissa*2^shift */
  const double mantissa = std::frexp(scale, &shift);
  auto shifted_m = std::round(mantissa * (int64_t(1) << 15));

  assert(shifted_m <= (int64_t(1) << 15)); // can't be greater that 1.0
  if (shifted_m == (int64_t(1) << 15)) {
    shifted_m /= 2;
    shift++;
  }
  // TOSA expect right shift to be positive, and embed (1 << 15) into right
  // shift bits
  shift = (-shift) + 15;

  assert(shifted_m <= std::numeric_limits<int32_t>::max());

  multiplier = static_cast<int32_t>(shifted_m);
}

void computeMultiplierAndShiftTosaScale32(double scale, int32_t &multiplier,
                                          int32_t &shift) {

  /* Generates mantissa and shift values where mantissa is in [-1.0,-0.5] or
     [0.5, 1.0] such that
     multiplier = mantissa*2^shift */
  const double mantissa = std::frexp(scale, &shift);
  auto shifted_m = std::round(mantissa * (int64_t(1) << 31));

  assert(shifted_m <= (int64_t(1) << 31)); // can't be greater that 1.0
  if (shifted_m == (int64_t(1) << 31)) {
    shifted_m /= 2;
    shift++;
  }
  // TOSA expect right shift to be positive, and embed (1 << 31) into right
  // shift bits
  shift = (-shift) + 31;

  assert(shifted_m <= std::numeric_limits<int32_t>::max());

  multiplier = static_cast<int32_t>(shifted_m);
}

} // namespace

/* Generates a quantized multiplier / shift from double */
void computeMultiplierAndShift(double scale, int32_t &multiplier,
                               int32_t &shift, int32_t scale_width) {

  switch (scale_width) {
  case 16:
    computeMultiplierAndShiftTosaScale16(scale, multiplier, shift);
    return;
  case 32:
    computeMultiplierAndShiftTosaScale32(scale, multiplier, shift);
    return;
  default:
    assert(0 && "Unsupported Tosa quantized_scale regime specified!");
  }
}

void computeMultiplierAndShiftGtOne(double scale, int32_t &multiplier,
                                    int32_t &shift, int32_t scale_width) {
  assert(scale > double(1.0));
  computeMultiplierAndShift(scale, multiplier, shift, scale_width);
  assert(shift >= 0);
}

void computeMultiplierAndShiftLtOneExp(double scale, int32_t &multiplier,
                                       int32_t &shift, int32_t scale_width) {
  assert(scale < double(1.0));
  assert(scale > double(0.0));
  computeMultiplierAndShift(scale, multiplier, shift, scale_width);
  assert(shift <= 0);
}

#define GET_UQTYPE(input)                                                      \
  ((input)                                                                     \
       .getType()                                                              \
       .dyn_cast<RankedTensorType>()                                           \
       .getElementType()                                                       \
       .dyn_cast<mlir::quant::UniformQuantizedType>())

/* method to build ConvOpQuantizationAttr, called from
 * ConvOpQuantInfoBuilder/TransConvOpQuantInfoBuilder: input_zp: input zeropoint
 * weight_zp: weight zeropoint
 */
ConvOpQuantizationAttr buildConvOpQuantizationAttr(mlir::OpBuilder &builder,
                                                   Value input, Value weight) {

  auto input_type = input.getType().dyn_cast<RankedTensorType>();
  auto weight_type = weight.getType().dyn_cast<RankedTensorType>();

  if (!input_type || !weight_type)
    return nullptr;

  bool input_is_qtype =
      input_type.getElementType().isa<mlir::quant::QuantizedType>();
  bool weight_is_qtype =
      weight_type.getElementType().isa<mlir::quant::QuantizedType>();

  // Either all quantized or all not quantized
  assert(!(input_is_qtype ^ weight_is_qtype));

  if (input_is_qtype) {

    auto input_qtype = input_type.getElementType()
                           .dyn_cast<mlir::quant::UniformQuantizedType>();
    assert(input_qtype); // We don't support any other kind of input
                         // quantization here

    int64_t input_zp = input_qtype.getZeroPoint();
    int64_t weight_zp = 0;

    // per tensor quantization
    if (auto weight_qtype =
            weight_type.getElementType()
                .dyn_cast<mlir::quant::UniformQuantizedType>()) {
      weight_zp = weight_qtype.getZeroPoint();
      // per channel quantization
    } else if (auto weight_qtype =
                   weight_type.getElementType()
                       .dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
      weight_zp = weight_qtype.getZeroPoints().front();
    }

    auto quantattr = mlir::tosa::ConvOpQuantizationAttr::get(
        builder.getI32IntegerAttr(input_zp),
        builder.getI32IntegerAttr(weight_zp), builder.getContext());

    return quantattr;
  }

  return nullptr;
}

/* method to build MatMulOpQuantizationAttr, called from
 * MatMulOpQuantInfoBuilder: a_zp: input a zeropoint b_zp: input b zeropoint
 */
MatMulOpQuantizationAttr buildMatMulOpQuantizationAttr(mlir::OpBuilder &builder,
                                                       Value a, Value b) {

  auto a_type = a.getType().dyn_cast<RankedTensorType>();
  auto b_type = b.getType().dyn_cast<RankedTensorType>();

  if (!a_type || !b_type)
    return nullptr;

  bool a_is_qtype =
      a_type.getElementType().isa<mlir::quant::UniformQuantizedType>();
  bool b_is_qtype =
      b_type.getElementType().isa<mlir::quant::UniformQuantizedType>();

  // Either all quantized or all not quantized
  assert(!(a_is_qtype ^ b_is_qtype));

  if (a_is_qtype) {

    auto a_qtype = GET_UQTYPE(a);
    auto b_qtype = GET_UQTYPE(b);

    assert(a_qtype && b_qtype);

    int64_t a_zp = a_qtype.getZeroPoint();
    int64_t b_zp = b_qtype.getZeroPoint();

    auto quantattr = mlir::tosa::MatMulOpQuantizationAttr::get(
        builder.getI32IntegerAttr(a_zp), builder.getI32IntegerAttr(b_zp),
        builder.getContext());

    return quantattr;
  }

  return nullptr;
}

/* method to build UnaryOpQuantizationAttr, called from
 * UnaryOpQuantInfoBuilder: input_zp: input zeropoint output_zp: output
 * zeropoint
 */
UnaryOpQuantizationAttr buildUnaryOpQuantizationAttr(mlir::OpBuilder &builder,
                                                     Value input,
                                                     Type output_raw_type) {

  auto input_type = input.getType().dyn_cast<RankedTensorType>();
  auto output_type = output_raw_type.dyn_cast<RankedTensorType>();

  if (!input_type || !output_type)
    return nullptr;

  bool input_is_qtype =
      input_type.getElementType().isa<mlir::quant::QuantizedType>();
  bool output_is_qtype =
      output_type.getElementType().isa<mlir::quant::QuantizedType>();

  // Either all quantized or all not quantized
  assert(!(input_is_qtype ^ output_is_qtype));

  if (input_is_qtype) {

    auto input_qtype = input_type.getElementType()
                           .dyn_cast<mlir::quant::UniformQuantizedType>();
    auto output_qtype = output_type.getElementType()
                            .dyn_cast<mlir::quant::UniformQuantizedType>();
    assert(input_qtype && output_qtype);

    int64_t input_zp = input_qtype.getZeroPoint();
    int64_t output_zp = output_qtype.getZeroPoint();

    auto quantattr = mlir::tosa::UnaryOpQuantizationAttr::get(
        builder.getI32IntegerAttr(input_zp),
        builder.getI32IntegerAttr(output_zp), builder.getContext());

    return quantattr;
  }

  return nullptr;
}

/* method to build PadOpQuantizationAttr, called from PadOpQuantInfoBuilder:
 * input_zp: input zeropoint
 */
PadOpQuantizationAttr buildPadOpQuantizationAttr(mlir::OpBuilder &builder,
                                                 Value input) {

  auto input_type = input.getType().dyn_cast<RankedTensorType>();

  if (!input_type)
    return nullptr;

  bool input_is_qtype =
      input_type.getElementType().isa<mlir::quant::UniformQuantizedType>();

  if (input_is_qtype) {

    auto input_qtype = input_type.getElementType()
                           .dyn_cast<mlir::quant::UniformQuantizedType>();
    assert(input_qtype);

    int64_t input_zp = input_qtype.getZeroPoint();

    auto quantattr = mlir::tosa::PadOpQuantizationAttr::get(
        builder.getI32IntegerAttr(input_zp), builder.getContext());

    return quantattr;
  }

  return nullptr;
}

Type buildQTypeFromMinMax(OpBuilder builder, Type input_dtype,
                          Attribute minattr, Attribute maxattr,
                          IntegerAttr quant_bits, int filter_quantdim,
                          bool issigned, BoolAttr narrow_range) {

  quant::QuantizedType rettype;

  auto convfunc =
      quant::ExpressedToQuantizedConverter::forInputType(input_dtype);

  auto minelems = minattr.dyn_cast<DenseFPElementsAttr>();
  auto maxelems = maxattr.dyn_cast<DenseFPElementsAttr>();

  SmallVector<double, 2> min, max;

  if (minelems || maxelems) { // at least one is per-axis quantized elementsattr

    // must have the same number of elements
    if (minelems.getNumElements() != maxelems.getNumElements())
      return {};

    min.reserve(minelems.getNumElements());
    max.reserve(maxelems.getNumElements());
    for (auto i : minelems) {
      min.push_back(FloatAttr::getValueAsDouble(i));
    }
    for (auto i : maxelems) {
      max.push_back(FloatAttr::getValueAsDouble(i));
    }
  } else { // Just a single FP value

    auto minval = minattr.dyn_cast<FloatAttr>();
    if (minval)
      min.push_back(minval.getValueAsDouble());
    else
      return {};
    auto maxval = maxattr.dyn_cast<FloatAttr>();
    if (maxval)
      max.push_back(maxval.getValueAsDouble());
    else
      return {};
  }

  if (min.size() == max.size()) {

    if (min.size() == 1) { // Per-tensor quantization with one min/max pair

      rettype = quant::fakeQuantAttrsToType(
          builder.getUnknownLoc(), quant_bits.getInt(), min[0], max[0],
          narrow_range.getValue(), convfunc.expressedType, issigned);

    } else if (min.size() > 1) { // per-axis quant on filter_quantdim

      auto shape = input_dtype.dyn_cast<ShapedType>();
      if (!shape)
        return {};
      if ((filter_quantdim) >= 0 && (shape.getRank() > filter_quantdim)) {

        rettype = quant::fakeQuantAttrsToType(
            builder.getUnknownLoc(), quant_bits.getInt(), filter_quantdim,
            min[0], max[0], narrow_range.getValue(), convfunc.expressedType,
            issigned);
      }

    } else {
      return {};
    }
  } else {
    return {};
  }

  if (!rettype)
    return {};

  return convfunc.convert(rettype);
}

TypeAttr buildQTypeAttrFromMinMax(OpBuilder builder, Type input_dtype,
                                  Attribute minattr, Attribute maxattr,
                                  IntegerAttr quant_bits, int filter_quantdim,
                                  bool issigned, BoolAttr narrow_range) {

  return TypeAttr::get(
      buildQTypeFromMinMax(builder, input_dtype, minattr, maxattr, quant_bits,
                           filter_quantdim, issigned, narrow_range));
}

} // namespace tosa
} // namespace mlir
