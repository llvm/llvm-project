//===-- TosaOpBase.td - TOSA dialect op builders -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the common definitions for the TOSA dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TOSA_OP_BASE
#define TOSA_OP_BASE

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// The TOSA Dialect.
//===----------------------------------------------------------------------===//

def Tosa_Dialect : Dialect {
  let name = "tosa";

  let description = [{
    The Tensor Operator Set Architecture (TOSA) dialect.

    This dialect implements the TOSA standard described at
    https://developer.mlplatform.org/w/tosa/ .

    Tensor Operator Set Architecture (TOSA) provides a set of whole-tensor
    operations commonly employed by Deep Neural Networks. The intent is to
    enable a variety of implementations running on a diverse range of
    processors, with the results at the TOSA level consistent across those
    implementations. Applications or frameworks which target TOSA can therefore
    be deployed on a wide range of different processors, such as CPUs or GPUs,
    with defined accuracy and compatibility constraints. Most operators from the
    common ML frameworks should be expressible in TOSA. It is expected that
    there will be tools to lower from the ML frameworks into TOSA.
  }];

  let dependentDialects = ["tensor::TensorDialect"];

  let cppNamespace = "mlir::tosa";
  let hasConstantMaterializer = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// TOSA Attributes.
//===----------------------------------------------------------------------===//

class Tosa_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Tosa_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// TOSA Operator Quantization Attributes.
//===----------------------------------------------------------------------===//

// Quantization attributes used across TOSA operators. Quantization attributes
// feed numerical precision parameters to the functional implementation of TOSA
// operators.
// The functional behavior is defined in the TOSA specification maintained at
// https://developer.mlplatform.org/w/tosa/. TOSA leverages MLIR's built in
// quantization support: https://mlir.llvm.org/docs/Quantization/, and supports
// uniform quantization. Depending on datatype, asymmetric and symmetric
// quantization are supported. The types themselves are described in
// TosaTypesBase.td .

// This quantization attribute expresses numerical behavior of operators where
// the operator has a numerical relationship between a single input and output.
// For example: tosa.negate.
def Tosa_UnaryOpQuantizationAttr
    : Tosa_Attr<"UnaryOpQuantization", "unary_quant"> {
  let summary = "Attribute for UnaryOp quantization information.";
  let parameters = (ins "int64_t":$input_zp, "int64_t":$output_zp);
  let assemblyFormat = "`<` struct(params) `>`";
}

// There is no explicit BinaryOpQuantizationAttr for 2-input/1-output ops. In
// this case, a tosa.rescale is used to express the inputs to the same scale.
// TODO: Upload WIP legalization document describing this construction by
// example.

// This quantization attribute holds input and weight zero point. Both the
// ConvOp and MatMulOp QuantizationAttrs follow a common design semantic where
// their ownquantization attribute only expresses the numerical behavior at
// the inputs.
// The scaling of their accumulator output is done using an explicit
// tosa.rescale operator that scales the accumulator result to output scale.
def Tosa_ConvOpQuantizationAttr
    : Tosa_Attr<"ConvOpQuantization", "conv_quant"> {
  let summary = "Attribute for Conv type op quantization information.";
  let parameters = (ins "int64_t":$input_zp, "int64_t":$weight_zp);
  let assemblyFormat = "`<` struct(params) `>`";
}

def Tosa_MatMulOpQuantizationAttr
    : Tosa_Attr< "MatMulOpQuantization", "matmul_quant"> {
  let summary = "Attribute for MatMulOp quantization information.";
  let parameters = (ins "int64_t":$a_zp, "int64_t":$b_zp);
  let assemblyFormat = "`<` struct(params) `>`";
}

// This attribute holds input zero point correction applied to the padding
// zeros to ensure numerical accuracy in the subsequent TOSA operations.
// Its functional application is described in the tosa.pad() operator
// description in the specification.
def Tosa_PadOpQuantizationAttr : Tosa_Attr<"PadOpQuantization", "pad_quant"> {
  let summary = "Attribute for PadOp quantization information.";
  let parameters = (ins "int64_t":$input_zp);
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// TOSA Operator Quantization Builders.
//===----------------------------------------------------------------------===//

// This builder is called on all convolution operators except for TransposeConv,
// which has specialized output shape semantics. The builder also defines the
// bitwidth of the output given the bit width of the input & weight content.
def Tosa_ConvOpQuantInfoBuilder : OpBuilder<
  (ins "Type":$outputType, "Value":$input, "Value":$weight, "Value":$bias,
       "ArrayAttr":$pad, "ArrayAttr":$stride, "ArrayAttr":$dilation),
  [{
    buildConvOpWithQuantInfo($_builder, $_state, outputType,
                             input, weight, bias,
                             pad, stride, dilation);
  }]>;

// Handles tosa.transpose_conv2d which has an outpad and output shape attribute.
def Tosa_TransConvOpQuantInfoBuilder : OpBuilder<
  (ins "Type":$outputType, "Value":$input, "Value":$weight, "Value":$bias,
       "ArrayAttr":$outpad, "ArrayAttr":$stride, "ArrayAttr":$outputShape),
  [{
    buildTransConvOpWithQuantInfo($_builder, $_state, outputType,
                                  input, weight, bias,
                                  outpad, stride,
                                  outputShape);
  }]>;

// The tosa.fully_connected op has its own builder as it does not have
// strides/dilation/padding.
def Tosa_FCOpQuantInfoBuilder : OpBuilder<
  (ins "Type":$outputType, "Value":$input, "Value":$weight, "Value":$bias),
  [{
    buildFCOpWithQuantInfo($_builder, $_state, outputType,
                           input, weight, bias);
  }]>;

// The tosa.matmul op is also intended to be generated where a fully_connected
// op must be constructed where the weight is not a constant. In this case,
// the fully_connected op must be expressed using matmul.
// TODO: Add link to the leglization document explaining this.
def Tosa_MatMulOpQuantInfoBuilder : OpBuilder<
  (ins "Type":$outputType, "Value":$a, "Value":$b),
  [{
    buildMatMulOpWithQuantInfo($_builder, $_state, outputType,
                               a, b);
  }]>;

// Both the tosa.avg_pool2d and unary ops use the same
// UnaruOpQuantizationAttr but the avg_pool operator has its own builder as it
// has additional parameters not part of the unary ops.
def Tosa_AvgPool2dOpQuantInfoBuilder : OpBuilder<
  (ins "Type":$outputType, "Value":$input, "ArrayAttr":$kernel,
       "ArrayAttr":$stride, "ArrayAttr":$pad),
  [{
    buildAvgPool2dOpWithQuantInfo($_builder, $_state, outputType,
                                  input, kernel, stride, pad);
  }]>;

// This builder is called on single-parameter unary operators that have a scale
// relationship between their input and output, expressed by the
// UnaryOpQuantizationAttr.
def Tosa_UnaryOpQuantInfoBuilder : OpBuilder<
  (ins "Type":$outputType, "Value":$input),
  [{
    buildUnaryOpWithQuantInfo($_builder, $_state, outputType, input);
  }]>;

// These builders are called on the TOSA pad operator that needs to create its
// own OptionalAttr quantization_attr parameter to scale the padding values
// correctly.
def Tosa_PadOpQuantInfoBuilder : OpBuilder<
  (ins "Type":$outputType, "Value":$input, "Value":$paddings),
  [{
    buildPadOpWithQuantInfo($_builder, $_state, outputType,
                            input, paddings);
  }]>;

def Tosa_ExplicitValuePadOpQuantInfoBuilder : OpBuilder<
  (ins "Type":$outputType, "Value":$input, "Value":$paddings,
       "Value":$pad_value),
  [{
    buildExplicitValuePadOpWithQuantInfo($_builder, $_state, outputType,
                                         input, paddings, pad_value);
  }]>;

//===----------------------------------------------------------------------===//
// TOSA Operator.
//===----------------------------------------------------------------------===//

class Tosa_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tosa_Dialect, mnemonic, !listconcat(traits, [TosaOpInterface])> {
}

#endif // TOSA_OP_BASE
