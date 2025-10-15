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
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"

include "mlir/Dialect/Tosa/IR/TosaInterfaces.td"

//===----------------------------------------------------------------------===//
// The TOSA Dialect.
//===----------------------------------------------------------------------===//

def Tosa_Dialect : Dialect {
  let name = "tosa";

  let description = [{
    The Tensor Operator Set Architecture (TOSA) dialect.

    This dialect implements the TOSA standard described at
    https://www.mlplatform.org/tosa/tosa_spec.html .

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

  let dependentDialects = ["tensor::TensorDialect", "quant::QuantDialect"];

  let cppNamespace = "mlir::tosa";
  let hasConstantMaterializer = 1;
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
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
// https://www.mlplatform.org/tosa/tosa_spec.html. TOSA leverages MLIR's built in
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
  (ins "::mlir::Type":$outputType, "::mlir::Value":$input,
       "::mlir::Value":$weight, "::mlir::Value":$bias,
       "::mlir::DenseI64ArrayAttr":$pad, "::mlir::DenseI64ArrayAttr":$stride,
       "::mlir::DenseI64ArrayAttr":$dilation,
       "::mlir::TypeAttr":$acc_type),
  [{
    buildConvOpWithQuantInfo($_builder, $_state, outputType,
                             input, weight, bias,
                             pad, stride, dilation, acc_type);
  }]>;

// Handles tosa.transpose_conv2d which has an outpad and output shape attribute.
def Tosa_TransConvOpQuantInfoBuilder : OpBuilder<
  (ins "::mlir::Type":$outputType, "::mlir::Value":$input,
       "::mlir::Value":$weight, "mlir::Value":$bias,
       "::mlir::DenseI64ArrayAttr":$outpad,
       "::mlir::DenseI64ArrayAttr":$stride,
       "::mlir::TypeAttr":$acc_type),
  [{
    buildTransConvOpWithQuantInfo($_builder, $_state, outputType,
                                  input, weight, bias,
                                  outpad, stride, acc_type);
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
  (ins "::mlir::Type":$outputType, "::mlir::Value":$input,
       "::mlir::DenseI64ArrayAttr":$kernel, "::mlir::DenseI64ArrayAttr":$stride,
       "::mlir::DenseI64ArrayAttr":$pad, "::mlir::TypeAttr":$acc_type),
  [{
    buildAvgPool2dOpWithQuantInfo($_builder, $_state, outputType,
                                  input, kernel, stride, pad, acc_type);
  }]>;

// This builder is called on single-parameter negate operators that have a scale
// relationship between their input and output, expressed by the
// UnaryOpQuantizationAttr.
def Tosa_NegateOpQuantInfoBuilder : OpBuilder<
  (ins "Type":$outputType, "Value":$input),
  [{
    buildNegateOpWithQuantInfo($_builder, $_state, outputType, input);
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

// This builder is called on the TOSA variable operator with a variable type
// and optional initial value. The builder will extract var_shape and element type
// attributes from variable type.
def Tosa_VariableOpBuilder : OpBuilder<
  (ins "StringRef":$sym_name, "Type":$variable_type, "Attribute":$initial_value),
  [{
    buildVariableOp($_builder, $_state, sym_name, variable_type, initial_value);
  }]>;


// Wrapper over base I32EnumAttr to set common fields.
class Tosa_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
     : I32EnumAttr<name, description, cases> {
   let genSpecializedAttr = 0;
   let cppNamespace = "::mlir::tosa";
}

class Tosa_I32EnumAttr<string name, string description, string mnemonic,
                         list<I32EnumAttrCase> cases>
    : EnumAttr<Tosa_Dialect, Tosa_I32Enum<name, description, cases>, mnemonic> {
   let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// TOSA Spec Section 1.5.
//
// Profile:
// INT : Integer Inference. Integer operations, primarily 8 and 32-bit values.
// FP  : Floating-Point Inference. Primarily FP16 and FP32 operations.
//
// Extension:
// INT16        : 16-bit integer operations.
// INT4         : 4-bit integer weights.
// BF16         : BFloat16 operations.
// FP8E4M3      : 8-bit floating-point operations E4M3.
// FP8E5M2      : 8-bit floating-point operations E5M2.
// FFT          : Fast Fourier Transform operations.
// VARIABLE     : Stateful variable operations.
// CONTROLFLOW  : Control Flow operations.
// DOUBLEROUND  : Adds double rounding support to the RESCALE operator.
// INEXACTROUND : Adds inexact rounding support to the RESCALE operator.
// DYNAMIC      : Removes all Compile Time Constant state for CTC inputs.
//===----------------------------------------------------------------------===//

def Tosa_NONE : I32EnumAttrCase<"none", 0>;
def Tosa_PRO_INT   : I32EnumAttrCase<"pro_int", 1>;
def Tosa_PRO_FP   : I32EnumAttrCase<"pro_fp", 2>;

def Tosa_ProfileAttr
    : Tosa_I32EnumAttr<"Profile", "supported TOSA profiles", "prof",
                       [Tosa_PRO_INT, Tosa_PRO_FP, Tosa_NONE]> {
  let extraClassDeclaration = [{
    static llvm::SmallVector<Profile, 2> getAllValues() {
      return {Profile::pro_int, Profile::pro_fp};
    }
  }];
}

def Tosa_ProfileArrayAttr
    : TypedArrayAttrBase<Tosa_ProfileAttr, "TOSA profile array attribute">;

def Tosa_EXT_NONE         : I32EnumAttrCase<"none", 0>;
def Tosa_EXT_INT16        : I32EnumAttrCase<"int16", 1>;
def Tosa_EXT_INT4         : I32EnumAttrCase<"int4", 2>;
def Tosa_EXT_BF16         : I32EnumAttrCase<"bf16", 3>;
def Tosa_EXT_FP8E4M3      : I32EnumAttrCase<"fp8e4m3", 4>;
def Tosa_EXT_FP8E5M2      : I32EnumAttrCase<"fp8e5m2", 5>;
def Tosa_EXT_FFT          : I32EnumAttrCase<"fft", 6>;
def Tosa_EXT_VARIABLE     : I32EnumAttrCase<"variable", 7>;
def Tosa_EXT_CONTROLFLOW  : I32EnumAttrCase<"controlflow", 8>;
def Tosa_EXT_DOUBLEROUND  : I32EnumAttrCase<"doubleround", 9>;
def Tosa_EXT_INEXACTROUND : I32EnumAttrCase<"inexactround", 10>;
def Tosa_EXT_DYNAMIC      : I32EnumAttrCase<"dynamic", 11>;

def Tosa_ExtensionAttr
    : Tosa_I32EnumAttr<"Extension", "supported TOSA extensions", "ext", [
      Tosa_EXT_NONE, Tosa_EXT_INT16, Tosa_EXT_INT4, Tosa_EXT_BF16,
      Tosa_EXT_FP8E4M3, Tosa_EXT_FP8E5M2, Tosa_EXT_FFT, Tosa_EXT_VARIABLE,
      Tosa_EXT_CONTROLFLOW, Tosa_EXT_DOUBLEROUND, Tosa_EXT_INEXACTROUND,
      Tosa_EXT_DYNAMIC
    ]> {
  let extraClassDeclaration = [{
    static llvm::SmallVector<Extension, 11> getAllValues() {
      return {
        Extension::int16, Extension::int4, Extension::bf16,
        Extension::fp8e4m3, Extension::fp8e5m2, Extension::fft,
        Extension::variable, Extension::controlflow, Extension::doubleround,
        Extension::inexactround, Extension::dynamic
      };
    }
  }];
}

def Tosa_ExtensionArrayAttr
    : TypedArrayAttrBase<Tosa_ExtensionAttr, "TOSA extension array attribute">;

def Tosa_LVL_NONE : I32EnumAttrCase<"none", 0>;
def Tosa_LVL_8K   : I32EnumAttrCase<"eightK", 1, "8k">;

def Tosa_LevelAttr
    : Tosa_I32EnumAttr<"Level", "supported TOSA levels", "level", [Tosa_LVL_NONE, Tosa_LVL_8K]>;

// The base class for defining op availability dimensions.
class Availability {
  // The following are fields for controlling the generated C++ OpInterface.

  // The namespace for the generated C++ OpInterface subclass.
  string cppNamespace = "::mlir::tosa";

  // The name for the generated C++ OpInterface subclass.
  string interfaceName = ?;

  // The description for the generated C++ OpInterface subclass.
  string interfaceDescription = "";

  // The query function's return type in the generated C++ OpInterface subclass.
  string queryFnRetType = ?;

  // The query function's name in the generated C++ OpInterface subclass.
  string queryFnName = ?;

  // The logic for merging two availability requirements.
  code mergeAction = ?;

  // The initializer for the final availability requirement.
  string initializer = ?;

  // An availability instance's type.
  string instanceType = ?;

  // The following are fields for a concrete availability instance.

  // The code for preparing a concrete instance. This should be C++ statements
  // and will be generated before the `mergeAction` logic.
  code instancePreparation = "";

  // The availability requirement carried by a concrete instance.
  string instance = ?;
}


class Profile<list<I32EnumAttrCase> profiles> : Availability {
  let interfaceName = "QueryProfileInterface";
  let interfaceDescription = [{
    Querying interface for the supported set of Tosa profile.

    This interface provides a `getProfiles()` method to query
    the supported set of Tosa profile. The returned value is a
    list of `mlir::Tosa::Profile` enum number.
  }];

  let queryFnRetType = "::llvm::SmallVector<::llvm::ArrayRef<"
                          "::mlir::tosa::Profile>, 1>";
  let queryFnName = "getProfiles";

  let mergeAction = !if(
      !empty(profiles), "", "$overall.emplace_back($instance)");

  let initializer = "{}";

  let instanceType = "::llvm::ArrayRef<::mlir::tosa::Profile>";

  // Pack all profiles as a static array and get its reference.
  let instancePreparation = !if(!empty(profiles), "",
    "static const ::mlir::tosa::Profile profs[] = {" #
    !interleave(!foreach(prof, profiles,
                         "::mlir::tosa::Profile::" # prof.symbol), ", ") #
    "}; " #
    "ArrayRef<::mlir::tosa::Profile> " #
      "ref(profs, std::size(profs));");

  let instance = "ref";
}

class Extension<list<I32EnumAttrCase> extensions> : Availability {
  let interfaceName = "QueryExtensionInterface";
  let interfaceDescription = [{
    Querying interface for the supported set of TOSA extension.

    This interface provides a `getExtensions()` method to query
    the supported set of Tosa extension. The returned value is a
    list of `mlir::Tosa::Extension` enum number.
  }];

  let queryFnRetType = "::llvm::SmallVector<::llvm::ArrayRef<"
                          "::mlir::tosa::Extension>, 1>";
  let queryFnName = "getExtensions";

  let mergeAction = !if(
      !empty(extensions), "", "$overall.emplace_back($instance)");

  let initializer = "{}";

  let instanceType = "::llvm::ArrayRef<::mlir::tosa::Extension>";

  // Pack all extensions as a static array and get its reference.
  let instancePreparation = !if(!empty(extensions), "",
    "static const ::mlir::tosa::Extension exts[] = {" #
    !interleave(!foreach(ext, extensions,
                         "::mlir::tosa::Extension::" # ext.symbol), ", ") #
    "}; " #
    "ArrayRef<::mlir::tosa::Extension> " #
      "ref(exts, std::size(exts));");

  let instance = "ref";
}

//===----------------------------------------------------------------------===//
// TOSA target environment.
//===----------------------------------------------------------------------===//
def Tosa_TargetEnv : Tosa_Attr<"TargetEnv", "target_env"> {
  let summary = "Target environment information.";
  let parameters = ( ins
    "Level": $level,
    ArrayRefParameter<"Profile">: $profiles,
    ArrayRefParameter<"Extension">: $extensions
  );

  let assemblyFormat = "`<` `level` `=` $level `,` `profiles` `=` `[` $profiles `]` `,` "
                       "`extensions` `=` `[` $extensions `]` `>`";
}

//===----------------------------------------------------------------------===//
// Iterable attributes.
//===----------------------------------------------------------------------===//
// Defined in `section 3. Enumerations` of the TOSA specification.

def Tosa_RESIZE_NEAREST_NEIGHBOR          : I32EnumAttrCase<"NEAREST_NEIGHBOR", 1>;
def Tosa_RESIZE_BILINEAR                  : I32EnumAttrCase<"BILINEAR", 2>;

def Tosa_ResizeModeAttr
    : Tosa_I32EnumAttr<"ResizeMode", "Supported resize/upsampling strategies", "resize_mode",
                    [Tosa_RESIZE_NEAREST_NEIGHBOR, Tosa_RESIZE_BILINEAR]>;

def Tosa_NANPROPAGATION_PROPAGATE : I32EnumAttrCase<"PROPAGATE", 1>;
def Tosa_NANPROPAGATION_IGNORE    : I32EnumAttrCase<"IGNORE", 2>;

def Tosa_NanPropagationModeAttr
    : Tosa_I32EnumAttr<"NanPropagationMode", "Supported NaN propagation strategies", "nan_mode",
                    [Tosa_NANPROPAGATION_PROPAGATE, Tosa_NANPROPAGATION_IGNORE]>;

def Tosa_ROUNDING_SINGLE_ROUND    : I32EnumAttrCase<"SINGLE_ROUND", 1>;
def Tosa_ROUNDING_INEXACT_ROUND   : I32EnumAttrCase<"INEXACT_ROUND", 2>;
def Tosa_ROUNDING_DOUBLE_ROUND    : I32EnumAttrCase<"DOUBLE_ROUND", 3>;

def Tosa_RoundingModeAttr
    : Tosa_I32EnumAttr<"RoundingMode", "Supported rounding modes", "rounding_mode",
                    [Tosa_ROUNDING_SINGLE_ROUND, Tosa_ROUNDING_INEXACT_ROUND, Tosa_ROUNDING_DOUBLE_ROUND]>;


//===----------------------------------------------------------------------===//
// TOSA Interfaces.
//===----------------------------------------------------------------------===//

def QueryProfileInterface : OpInterface<"QueryProfileInterface"> {
  let cppNamespace = "::mlir::tosa";
  let methods = [InterfaceMethod<
    "get supported profiles",
    "::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1>",
    "getProfiles">];
}

def QueryExtensionInterface : OpInterface<"QueryExtensionInterface"> {
  let cppNamespace = "::mlir::tosa";
  let methods = [InterfaceMethod<
    "get supported extensions",
    "::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1>",
    "getExtensions">];
}

//===----------------------------------------------------------------------===//
// TOSA Operator Trait.
//===----------------------------------------------------------------------===//

// Permits broadcasting. Elementwise trait is too strict.
def TosaElementwiseOperator : NativeOpTrait<"TosaElementwiseOperator"> {
  let cppNamespace = "mlir::OpTrait::tosa";
}

//===----------------------------------------------------------------------===//
// TOSA Operator Trait.
//===----------------------------------------------------------------------===//
// Op operands with TOSA shape types must be compile time resolvable
def TosaResolvableShapeOperands : NativeOpTrait<"TosaResolvableShapeOperands"> {
  let cppNamespace = "mlir::OpTrait::tosa";
}

//===----------------------------------------------------------------------===//
// TOSA Operator Class.
//===----------------------------------------------------------------------===//

class Tosa_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tosa_Dialect, mnemonic, !listconcat(traits, [TosaOpInterface,
    DeclareOpInterfaceMethods<QueryProfileInterface>,
    DeclareOpInterfaceMethods<QueryExtensionInterface>,
    TosaResolvableShapeOperands])> {

  // Default availability specification.
  list<Availability> availability = [
    Profile<[]>,
    Extension<[]>];

  // When not set, manual implementation of these methods is required.
  bit autogenAvailability = 1;
}

class Tosa_ElementwiseOp<string mnemonic, list<Trait> traits = []> :
    Tosa_Op<mnemonic, !listconcat(traits, [
              DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                                        ["inferReturnTypeComponents"]>,
              ResultsBroadcastableShape,
              TosaElementwiseOperator,
              SameOperandsAndResultRank,
              Pure])> {}

class Tosa_ElementwiseUnaryOp<string mnemonic, list<Trait> traits = []> :
    Tosa_ElementwiseOp<mnemonic, !listconcat(traits, [
              SameOperandsAndResultShape,
              SameOperandsAndResultElementType])> {}

class Tosa_InferTensorTypeOp<string mnemonic, list<Trait> traits = []>
    : Tosa_Op<mnemonic, !listconcat(traits, [InferTensorTypeAdaptor, Pure])> {}

class Tosa_InferShapedTypeOp<string mnemonic, list<Trait> traits = []>
    : Tosa_Op<mnemonic, !listconcat(traits, [InferShapedTypeOpAdaptor, Pure])> {}

// The "SameVariadicOperandSize" trait allows us to pass optional arguments
// for multiple zero points in convolution ops.
class Tosa_ConvOp<string mnemonic, list<Trait> traits = []>
    : Tosa_InferShapedTypeOp<mnemonic, !listconcat(traits,
      [SameVariadicOperandSize])> {
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

#endif // TOSA_OP_BASE
