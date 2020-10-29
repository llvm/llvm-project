//===-- TosaOpBase.td - TOSA dialect op builders *- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operation builders for the TOSA dialect.
//
//===----------------------------------------------------------------------===//


#ifdef TOSA_OP_BASE
#else
#define TOSA_OP_BASE

// Quantization attributes

def Tosa_UnaryOpQuantizationAttr : StructAttr<"UnaryOpQuantizationAttr", Tosa_Dialect, [
                StructFieldAttr<"input_zp",         I32Attr>,
                StructFieldAttr<"output_zp",        I32Attr>] > {
  let description = "Attribute holding quantization information for Unary Ops.";
}

def Tosa_ConvOpQuantizationAttr : StructAttr<"ConvOpQuantizationAttr", Tosa_Dialect, [
                StructFieldAttr<"input_zp",         I32Attr>,
                StructFieldAttr<"weight_zp",        I32Attr>] > {
  let description = "Attribute holding quantization information for Convolution Ops.";
}

def Tosa_MatMulOpQuantizationAttr : StructAttr<"MatMulOpQuantizationAttr", Tosa_Dialect, [
                StructFieldAttr<"a_zp",             I32Attr>,
                StructFieldAttr<"b_zp",             I32Attr>] > {
  let description = "Attribute holding quantization information for Convolution Ops.";
}

def Tosa_PadOpQuantizationAttr : StructAttr<"PadOpQuantizationAttr", Tosa_Dialect, [
                StructFieldAttr<"input_zp",         I32Attr>] > {
  let description = "Attribute holding quantization information for Pad Ops.";
}

///////////////////////////////////
// Tosa QuantizationInfo Builders
///////////////////////////////////

// ConvOp Quantization Info Builders

// This builder is called on convolution operation types that need to create their
// OptionalAttr quantization_attr parameter. It happens transparently when legalize_something.cc calls
// the rewriter.replaceOpWithNewOp() or similar builder, with just the standard arguments and not the
// additional quantization_attr option. It may be explicitly specified, but is not necessary.
// If it is explicitly specified, a different auto-generated builder will handle it.
def Tosa_ConvOpQuantInfoBuilder : OpBuilderDAG<
  (ins "Type":$output_type, "Value":$input, "Value":$filter, "Value":$bias, "ArrayAttr":$strides, "ArrayAttr":$dilations, "ArrayAttr":$padding),
  [{
    $_state.addOperands(input);
    $_state.addOperands(filter);
    $_state.addOperands(bias);
    $_state.addAttribute("strides", strides);
    $_state.addAttribute("dilations", dilations);
    $_state.addAttribute("padding", padding);

    auto quantattr = mlir::tosa::buildConvOpQuantizationAttr($_builder,
                                                        input,
                                                        filter);
    if ( quantattr ) {
      $_state.addAttribute("quantization_info", quantattr);
      unsigned input_bits = input.getType().dyn_cast<RankedTensorType>()
                            .getElementType().dyn_cast<mlir::quant::QuantizedType>()
                            .getStorageTypeIntegralWidth();
      unsigned weight_bits = filter.getType().dyn_cast<RankedTensorType>()
                            .getElementType().dyn_cast<mlir::quant::QuantizedType>()
                            .getStorageTypeIntegralWidth();
      auto output_shape = output_type.dyn_cast<RankedTensorType>().getShape();
      IntegerType acc_element_type;
      if(input_bits == 16 && weight_bits == 8) {
        acc_element_type = $_builder.getIntegerType(48);
      }
      else {
        acc_element_type = $_builder.getI32Type();
      }
      auto acc_type = RankedTensorType::get(output_shape, acc_element_type);
      $_state.addTypes(acc_type);
    }
    else {
      $_state.addTypes(output_type);
    }
  }]>;

// A variant of ConvOpQuantInfo builder for transpose_conv op which has no bias parameter.
def Tosa_TransConvOpQuantInfoBuilder : OpBuilderDAG<
  (ins "Type":$output_type, "Value":$input, "Value":$filter, "Value":$bias, "ArrayAttr":$strides, "ArrayAttr":$dilations, "ArrayAttr":$outpad, "ArrayAttr":$output_shape),
  [{
    $_state.addOperands(input);
    $_state.addOperands(filter);
    $_state.addOperands(bias);
    $_state.addAttribute("strides", strides);
    $_state.addAttribute("dilations", dilations);
    $_state.addAttribute("outpad", outpad);
    $_state.addAttribute("output_shape", output_shape);
    auto quantattr = mlir::tosa::buildConvOpQuantizationAttr($_builder,
                                                        input,
                                                        filter);

    if ( quantattr ) {
      $_state.addAttribute("quantization_info", quantattr);
      unsigned input_bits = input.getType().dyn_cast<RankedTensorType>()
                            .getElementType().dyn_cast<mlir::quant::QuantizedType>()
                            .getStorageTypeIntegralWidth();
      unsigned weight_bits = filter.getType().dyn_cast<RankedTensorType>()
                            .getElementType().dyn_cast<mlir::quant::QuantizedType>()
                            .getStorageTypeIntegralWidth();
      auto output_shape = output_type.dyn_cast<RankedTensorType>().getShape();
      IntegerType acc_element_type;
      if(input_bits == 16 && weight_bits == 8) {
        acc_element_type = $_builder.getIntegerType(48);
      }
      else {
        acc_element_type = $_builder.getI32Type();
      }
      auto acc_type = RankedTensorType::get(output_shape, acc_element_type);
      $_state.addTypes(acc_type);
    }
    else {
      $_state.addTypes(output_type);
    }
  }]>;

// All Conv legalizations are done in C++ so no TableGen builder

// FullyConnectedOp Quantization Info Builder

// This builder is called on FC operation that needs to create their
// OptionalAttr quantization_attr parameter. It happens transparently when legalize_something.cc calls
// the rewriter.replaceOpWithNewOp() or similar builder, with just the standard arguments and not the
// additional quantization_attr option. It may be explicitly specified, but is not necessary.
// If it is explicitly specified, a different auto-generated builder will handle it.
def Tosa_FCOpQuantInfoBuilder : OpBuilderDAG<
  (ins "Type":$output_type, "Value":$input, "Value":$filter, "Value":$bias),
  [{
    $_state.addOperands(input);
    $_state.addOperands(filter);
    $_state.addOperands(bias);
    auto quantattr = mlir::tosa::buildConvOpQuantizationAttr($_builder,
                                                        input,
                                                        filter);
    if ( quantattr ) {
      $_state.addAttribute("quantization_info", quantattr);
      unsigned input_bits = input.getType().dyn_cast<RankedTensorType>()
                            .getElementType().dyn_cast<mlir::quant::QuantizedType>()
                            .getStorageTypeIntegralWidth();
      unsigned weight_bits = filter.getType().dyn_cast<RankedTensorType>()
                            .getElementType().dyn_cast<mlir::quant::QuantizedType>()
                            .getStorageTypeIntegralWidth();
      auto output_shape = output_type.dyn_cast<RankedTensorType>().getShape();
      IntegerType acc_element_type;
      if(input_bits == 16 && weight_bits == 8) {
        acc_element_type = $_builder.getIntegerType(48);
      }
      else {
        acc_element_type = $_builder.getI32Type();
      }
      auto acc_type = RankedTensorType::get(output_shape, acc_element_type);
      $_state.addTypes(acc_type);
    }
    else {
      $_state.addTypes(output_type);
    }
  }]>;

// Similar to FCOpQuantInfoBuilder, but drop bias
def Tosa_MatMulOpQuantInfoBuilder : OpBuilderDAG<
  (ins "Type":$output_type, "Value":$a, "Value":$b),
  [{
    $_state.addOperands(a);
    $_state.addOperands(b);
    auto quantattr = mlir::tosa::buildMatMulOpQuantizationAttr($_builder, a, b);

    if ( quantattr ) {
      $_state.addAttribute("quantization_info", quantattr);
      unsigned input_bits = a.getType().dyn_cast<RankedTensorType>()
                            .getElementType().dyn_cast<mlir::quant::QuantizedType>()
                            .getStorageTypeIntegralWidth();
      auto output_shape = output_type.dyn_cast<RankedTensorType>().getShape();
      IntegerType acc_element_type;
      if(input_bits == 16) {
        acc_element_type = $_builder.getIntegerType(48);
      }
      else {
        acc_element_type = $_builder.getI32Type();
      }
      auto acc_type = RankedTensorType::get(output_shape, acc_element_type);
      $_state.addTypes(acc_type);
    }
    else {
      $_state.addTypes(output_type);
    }
  }]>;

// FC legalization done in C++ so no TableGen builder

// This builder is called on pool operation types that need to create their
// OptionalAttr quantization_attr parameter. It works like earlier builders, except that pool has
// additional cmdline parameters that must be accommodated by the builder even though only
// the input and output parameters matter to the quantization op creation.
def Tosa_AvgPool2dOpQuantInfoBuilder : OpBuilderDAG<
  (ins "Type":$output_type, "Value":$input, "ArrayAttr":$kernel_size, "ArrayAttr":$strides, "ArrayAttr":$padding),
  [{
    $_state.addOperands(input);
    $_state.addAttribute("kernel_size", kernel_size);
    $_state.addAttribute("strides", strides);
    $_state.addAttribute("padding", padding);
    auto quantattr = mlir::tosa::buildUnaryOpQuantizationAttr($_builder,
                                                                input,
                                                                output_type);
    if ( quantattr )
      $_state.addAttribute("quantization_info", quantattr);
    $_state.types.push_back(output_type);
  }]>;

// This builder is called on single-parameter unary types that need to create their
// OptionalAttr quantization_attr parameter.
def Tosa_UnaryOpQuantInfoBuilder : OpBuilderDAG<
  (ins "Type":$output_type, "Value":$input),
  [{
    $_state.addOperands(input);
    auto quantattr = mlir::tosa::buildUnaryOpQuantizationAttr($_builder,
                                                                input,
                                                                output_type);
    if ( quantattr )
      $_state.addAttribute("quantization_info", quantattr);
    $_state.types.push_back(output_type);
  }]>;

// This builder is called on single-parameter unary types that need to create their
// OptionalAttr quantization_attr parameter.
def Tosa_PadOpQuantInfoBuilder : OpBuilderDAG<
  (ins "Type":$output_type, "Value":$input, "Value":$paddings),
  [{
    $_state.addOperands(input);
    $_state.addOperands(paddings);
    auto quantattr = mlir::tosa::buildPadOpQuantizationAttr($_builder,
                                                              input);
    if ( quantattr )
      $_state.addAttribute("quantization_info", quantattr);
    $_state.types.push_back(output_type);
  }]>;

def Tosa_BroadcastableBinaryBuilder : OpBuilderDAG<
  (ins "Value":$lhs, "Value":$rhs),
  [{
    auto result_type =
      OpTrait::util::getBroadcastedType(lhs.getType(), rhs.getType());
    if (!result_type)
      mlir::emitError($_state.location, "Operands are not broadcastable");
    $_state.addOperands(lhs);
    $_state.addOperands(rhs);
    $_state.types.push_back(result_type);
  }]>;

///////////////////////////////
// Tosa Operator Definitions
//////////////////////////////

class Tosa_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Tosa_Dialect, mnemonic, !listconcat(traits, [TosaOpInterface])> {

    let extraClassDeclaration = [{
       static StringRef getTOSAOpType() { return ""; } // TBD
       static StringRef getTOSAOpSubtype() { return ""; } // TBD
    }];

}

class Tosa_ElemwiseUnaryOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Tosa_Dialect, mnemonic, !listconcat(traits, [TosaOpInterface])> {

    let extraClassDeclaration = [{
       static StringRef getTOSAOpType() { return "Elemwise"; }
       static StringRef getTOSAOpSubtype() { return "Unary"; }
    }];

}

class Tosa_ElemwiseBinaryOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Tosa_Dialect, mnemonic, !listconcat(traits, [TosaOpInterface])> {

    let extraClassDeclaration = [{
       static StringRef getTOSAOpType() { return "Elemwise"; }
       static StringRef getTOSAOpSubtype() { return "Binary"; }
    }];

}

class Tosa_ElemwiseCompareOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Tosa_Dialect, mnemonic, !listconcat(traits, [TosaOpInterface])> {

    let extraClassDeclaration = [{
       static StringRef getTOSAOpType() { return "Elemwise"; }
       static StringRef getTOSAOpSubtype() { return "Compare"; }
    }];

}

class Tosa_ElemwiseTernaryOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Tosa_Dialect, mnemonic, !listconcat(traits, [TosaOpInterface])> {

    let extraClassDeclaration = [{
       static StringRef getTOSAOpType() { return "Elemwise"; }
       static StringRef getTOSAOpSubtype() { return "Ternary"; }
    }];

}

class Tosa_DataLayoutOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Tosa_Dialect, mnemonic, !listconcat(traits, [TosaOpInterface])> {

    let extraClassDeclaration = [{
       static StringRef getTOSAOpType() { return "DataLayout"; }
       static StringRef getTOSAOpSubtype() { return ""; } // TBD
    }];

}

class Tosa_DataNodeOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Tosa_Dialect, mnemonic, !listconcat(traits, [TosaOpInterface])> {

    let extraClassDeclaration = [{
       static StringRef getTOSAOpType() { return "DataNode"; }
       static StringRef getTOSAOpSubtype() { return ""; } // TBD
    }];

}

class Tosa_AggregationOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Tosa_Dialect, mnemonic, !listconcat(traits, [TosaOpInterface])> {

    let extraClassDeclaration = [{
       static StringRef getTOSAOpType() { return "Aggregation"; }
       static StringRef getTOSAOpSubtype() { return ""; } // TBD
    }];

}

class Tosa_TensorArgOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Tosa_Dialect, mnemonic, !listconcat(traits, [TosaOpInterface])> {

    let extraClassDeclaration = [{
       static StringRef getTOSAOpType() { return "Tensor"; }
       static StringRef getTOSAOpSubtype() { return "Arg"; }
    }];

}

class Tosa_TensorConvOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Tosa_Dialect, mnemonic, !listconcat(traits, [TosaOpInterface])> {

    let extraClassDeclaration = [{
       static StringRef getTOSAOpType() { return "Tensor"; }
       static StringRef getTOSAOpSubtype() { return "Conv"; }
    }];

}

class Tosa_TensorPoolOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Tosa_Dialect, mnemonic, !listconcat(traits, [TosaOpInterface])> {

    let extraClassDeclaration = [{
       static StringRef getTOSAOpType() { return "Tensor"; }
       static StringRef getTOSAOpSubtype() { return "Pool"; }
    }];

}

class Tosa_TensorImageOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Tosa_Dialect, mnemonic, !listconcat(traits, [TosaOpInterface])> {

    let extraClassDeclaration = [{
       static StringRef getTOSAOpType() { return "Image"; }
       static StringRef getTOSAOpSubtype() { return ""; } // TBD
    }];

}

class Tosa_ActivationOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Tosa_Dialect, mnemonic, !listconcat(traits, [TosaOpInterface])> {

    let extraClassDeclaration = [{
       static StringRef getTOSAOpType() { return "Activation"; }
       static StringRef getTOSAOpSubtype() { return ""; } // TBD
    }];

}

class Tosa_ReductionOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Tosa_Dialect, mnemonic, !listconcat(traits, [TosaOpInterface])> {

    let extraClassDeclaration = [{
       static StringRef getTOSAOpType() { return "Reduction"; }
       static StringRef getTOSAOpSubtype() { return ""; } // TBD
    }];

}

class Tosa_ImageOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Tosa_Dialect, mnemonic, !listconcat(traits, [TosaOpInterface])> {

    let extraClassDeclaration = [{
       static StringRef getTOSAOpType() { return "Reduction"; }
       static StringRef getTOSAOpSubtype() { return ""; } // TBD
    }];

}

class Tosa_ConversionOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Tosa_Dialect, mnemonic, !listconcat(traits, [TosaOpInterface])> {

    let extraClassDeclaration = [{
       static StringRef getTOSAOpType() { return "Conversion"; }
       static StringRef getTOSAOpSubtype() { return ""; } // TBD
    }];

}

// Specify traits of operators.

#endif // TOSA_OP_BASE
