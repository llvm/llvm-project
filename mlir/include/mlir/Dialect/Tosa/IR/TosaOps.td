//===-- TosaOps.td - TOSA dialect operation definitions *- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operation set for the TOSA dialect.
//
//===----------------------------------------------------------------------===//

#ifdef TOSA_OPS
#else
#define TOSA_OPS

include "mlir/IR/OpBase.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Dialect/Tosa/IR/TosaInterfaces.td"

include "mlir/Dialect/Tosa/IR/TosaTypesBase.td"

def Tosa_Dialect : Dialect {
  let name = "tosa";

  let description = [{
    The Tosa dialect.

    Invariants:

    * All values are of Tensor type (in particular, scalars are
      represented using zero-dimentional tensors);
  }];

  let cppNamespace = "mlir::tosa";
}

#ifdef TOSA_OP_BASE
#else
include "mlir/Dialect/Tosa/IR/TosaOpBase.td"
#endif

/* TOSA Spec Section 2.2 */
/* Operator Class: Tensor Data Engine Operators */

/* Operator: argmax */

def Tosa_ArgMaxOp : Tosa_TensorArgOp<"argmax", [NoSideEffect]> {
  let summary = "Perform argmax on the input.";

  let description = [{
    This returns the index with the largest value across the given axis of the input tensor.
  }];

  let arguments = (ins
              Tosa_Tensor: $input,
              I64Attr: $axis);

  let results = (outs Tosa_Tensor: $output);

}

/* Operator: avg_pool2d */

def Tosa_AvgPool2dOp : Tosa_TensorPoolOp<"avg_pool2d", [NoSideEffect]> {
  let summary = "Performs max pooling on the input.";

  let description = [{
    This performs an average pooling over the given input tensor. A sliding window of size
    given by <kernel size> is passed over the input tensor, with the mean value being placed
    in the output tensor.
  }];

  let arguments = (ins
    Tosa_Tensor:$input,

    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$kernel_size,
    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$strides,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$padding,
    OptionalAttr<Tosa_UnaryOpQuantizationAttr>:$quantization_info
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  let builders = [Tosa_AvgPool2dOpQuantInfoBuilder];
}

/* Operator: conv2d */

def Tosa_Conv2DOp : Tosa_TensorConvOp<"conv2d", [NoSideEffect]> {
  let summary = [{
    2D Convolution Operator
  }];

  let description = [{
    Performs a 2D convolution over the given tensor input, using the weight tensor.
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    Tosa_Tensor:$filter,
    Tosa_Tensor:$bias,

    DefaultValuedAttr<I64ArrayAttr, "{1, 1, 1, 1}">:$strides,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1, 1, 1}">:$dilations,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$padding,
    OptionalAttr<Tosa_ConvOpQuantizationAttr>:$quantization_info
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  let builders = [Tosa_ConvOpQuantInfoBuilder];

}

/* Operator: conv3d */

def Tosa_Conv3DOp : Tosa_TensorConvOp<"conv3d", [NoSideEffect]> {
  let summary = [{
    3D Convolution operator
  }];

  let description = [{
    Performs a 3D convolution over the given input tensor.
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    Tosa_Tensor:$filter,
    Tosa_Tensor:$bias,

    DefaultValuedAttr<Confined<I64ArrayAttr, [ArrayMinCount<5>]>, "{1, 1, 1, 1, 1}">:$strides,
    DefaultValuedAttr<Confined<I64ArrayAttr, [ArrayMinCount<5>]>, "{1, 1, 1, 1, 1}">:$dilations,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$padding,
    OptionalAttr<Tosa_ConvOpQuantizationAttr>:$quantization_info
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  let builders = [Tosa_ConvOpQuantInfoBuilder];

}

def Tosa_DepthwiseConv2DOp : Tosa_TensorConvOp<"depthwise_conv2d", [NoSideEffect]> {
  let summary = [{
    Depthwise 2D Convolution operator
  }];

  let description = [{
    Performs 2D convolutions separately over each channel of the given tensor input, using the weight tensor.
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    Tosa_Tensor:$filter,
    Tosa_Tensor:$bias,

    DefaultValuedAttr<I64ArrayAttr, "{1, 1, 1, 1}">:$strides,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1, 1, 1}">:$dilations,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$padding,
    OptionalAttr<Tosa_ConvOpQuantizationAttr>:$quantization_info
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  let builders = [Tosa_ConvOpQuantInfoBuilder];

}

/* Operator: fully_connected */

def Tosa_FullyConnectedOp : Tosa_TensorConvOp<"fully_connected", [NoSideEffect]> {
  let summary = "Fully Connected operator";

  let description = [{
    Performs a fully connected network.
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    Tosa_Tensor:$filter,
    Tosa_TensorOfOrNone<[Tosa_AnyNumber]>:$bias,
    OptionalAttr<Tosa_ConvOpQuantizationAttr>:$quantization_info
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  let builders = [Tosa_FCOpQuantInfoBuilder];

}

/* Operator: matmul */

def Tosa_MatMulOp : Tosa_TensorConvOp<"matmul", [NoSideEffect]> {
  let summary = "Matrix multiplication with bias";

  let description = [{
    Performs a two dimensional matrix multiplication. This allows both inputs to be activations,
    rather than reserving weights as an attribute in the FULLY_CONNECTED operator.
  }];

  let arguments = (ins
    Tosa_Tensor:$a,
    Tosa_Tensor:$b,
    OptionalAttr<Tosa_MatMulOpQuantizationAttr>:$quantization_info
  );

  let results = (outs
    Tosa_Tensor:$c
  );

  let builders = [Tosa_MatMulOpQuantInfoBuilder];

}

/* Operator: max_pool2d */

def Tosa_MaxPool2dOp : Tosa_TensorPoolOp<"max_pool2d", [NoSideEffect]> {
  let summary = "Performs max pooling on the input.";

  let description = [{
    This performs a max pooling over the given input tensor. A sliding window of size given by
    <kernel size> is passed over the input tensor, with the maximum value being placed in the
    output tensor.
  }];

  let arguments = (ins
    Tosa_Tensor:$input,

    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$kernel_size,
    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$strides,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$padding
  );

  let results = (outs
    Tosa_Tensor:$output
  );
}

/* Operator: transpose_conv2d */

def Tosa_TransposeConv2DOp : Tosa_TensorConvOp<"transpose_conv2d", [NoSideEffect]> {
  let summary = [{
    Transpose 2D Convolution operator.
  }];

  let description = [{
    Performs a 2D transposed convolution over the given tensor input, using the weights tensor.
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    Tosa_Tensor:$filter,
    Tosa_Tensor:$bias,

    I64ArrayAttr:$strides,
    I64ArrayAttr:$dilations,
    I64ArrayAttr:$outpad,
    I64ArrayAttr:$output_shape,
    OptionalAttr<Tosa_ConvOpQuantizationAttr>:$quantization_info
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  let builders = [Tosa_TransConvOpQuantInfoBuilder];

}

/* TOSA Spec Section 2.3 */
/* Operator Class: Activation Functions */

/* Operator: clamp */

def Tosa_ClampOp : Tosa_ActivationOp<"clamp", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes clamp(features, min, max).";

  let description = [{
    Clamp to an arbitrary minimum and maximum value. Note that the maximum and minimum values are
    specified as signed quantized values, no scaling happens before or after this operation.
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    I64Attr:$min_int,
    I64Attr:$max_int,
    F32Attr:$min_fp,
    F32Attr:$max_fp
  );

  let results = (outs
    Tosa_Tensor:$output
  );

}

/* Operator: reluN */

def Tosa_ReluNOp : Tosa_ActivationOp<"reluN", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes rectified linear: `max(features, N)`.";

  let description = [{
     ReLU with a scalar maximum value.
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    I64Attr:$max_int,
    F32Attr:$max_fp
  );

  let results = (outs
    Tosa_Tensor:$output
  );

}

/* Operator: sigmoid */

def Tosa_SigmoidOp : Tosa_ActivationOp<"sigmoid", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes elementwise sigmoid of input.";

  let description = [{
    Sigmoid function: output = 1 / (1 + exp(-input))
    For quantized integer data types, the TABLE operator should be used instead with the following definition.
    The sigmoid table has 513 entries each of 16-bit precision and covering the input range -16.0 to +16.0
    in steps of 1/16.
  }];

  let arguments = (ins
    Tosa_Tensor:$input
  );

  let results = (outs
    Tosa_Tensor:$output
  );

}

/* Operator: tanh */

def Tosa_TanhOp : Tosa_ActivationOp<"tanh", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes elementwise hyperbolic tangent of input";

  let description = [{
    Parameterized hyperbolic tangent.
    For quantized integer data types, the TABLE operator should be used instead with the following definition.
    The tanh_table has 513 entries each of 16-bit precision and covering the input range -8.0 to +8.0 in steps of 1/32.
  }];

  let arguments = (ins
    Tosa_Tensor:$input
  );

  let results = (outs
    Tosa_Tensor:$output
  );

}

/* TOSA Spec Section 2.4 */
/* Operator Class: Elementwise unary/binary/ternary operators */
/* Operator Subclass: Elementwise binary ops */

/* Operator: add */

def Tosa_AddOp : Tosa_ElemwiseBinaryOp<"add", [ResultsBroadcastableShape, NoSideEffect, Commutative]> {
  let summary = "Elementwise addition operator";

  let description = [{
    Elementwise addition of input1 and input2. Axis of size 1 will be broadcast, as necessary.
    Rank of input tensors must match.
  }];

  let arguments = (
    ins Tosa_Tensor:$lhs,
    Tosa_Tensor:$rhs);

  let results = (outs Tosa_Tensor:$output);

  let builders = [Tosa_BroadcastableBinaryBuilder];
}

/* Operator: arithmetic_right_shift */

def Tosa_ArithmeticRightShiftOp : Tosa_ElemwiseBinaryOp<"arithmetic_right_shift", [ResultsBroadcastableShape, NoSideEffect]> {
  let summary = "Elementwise Arithmetic Right Shift";

  let description = [{
    Elementwise arithmetic right shift of input1 by the amount specified in input2. Axis of size 1 will be broadcast, as necessary.
    Rank of input tensors must match.
  }];

  let arguments = (
    ins Tosa_Tensor:$lhs,
    Tosa_Tensor:$rhs);

  let results = (outs Tosa_Tensor:$output);

  let builders = [Tosa_BroadcastableBinaryBuilder];
}

/* Operator: bitwise_and */

def Tosa_BitwiseAndOp : Tosa_ElemwiseBinaryOp<"bitwise_and", [ResultsBroadcastableShape, NoSideEffect, Commutative]> {
  let summary = "Bitwise AND operator";

  let description = [{
    Elementwise bitwise AND of input tensor 0 and input tensor 1. Axis of size 1 will be broadcast as necessary.
    Rank of input tensors must match.
  }];

  let arguments = (
    ins Tosa_Tensor:$lhs,
    Tosa_Tensor:$rhs);

  let results = (outs Tosa_Tensor:$output);

  let builders = [Tosa_BroadcastableBinaryBuilder];
}

/* Operator: bitwise_or */

def Tosa_BitwiseOrOp : Tosa_ElemwiseBinaryOp<"bitwise_or", [ResultsBroadcastableShape, NoSideEffect, Commutative]> {
  let summary = "Bitwise OR operator";

  let description = [{
    Elementwise bitwise OR of input1 and input2. Axis of size 1 will be broadcast as necessary.
    Rank of input tensors must match.
  }];

  let arguments = (
    ins Tosa_Tensor:$lhs,
    Tosa_Tensor:$rhs);

  let results = (outs Tosa_Tensor:$output);

  let builders = [Tosa_BroadcastableBinaryBuilder];
}

/* Operator: bitwise_xor */

def Tosa_BitwiseXorOp : Tosa_ElemwiseBinaryOp<"bitwise_xor", [ResultsBroadcastableShape, NoSideEffect, Commutative]> {
  let summary = "Bitwise XOR operator";

  let description = [{
    Elementwise bitwise XOR of input1 and input2. Axis of size 1 will be broadcast as necessary.
    Rank of input tensors must match.
  }];

  let arguments = (
    ins Tosa_Tensor:$lhs,
    Tosa_Tensor:$rhs);

  let results = (outs Tosa_Tensor:$output);

  let builders = [Tosa_BroadcastableBinaryBuilder];
}

/* Operator: logical_and */

def Tosa_LogicalAndOp : Tosa_ElemwiseBinaryOp<"logical_and", [ResultsBroadcastableShape, Commutative, NoSideEffect]> {
  let summary = "Returns the truth value of x AND y element-wise.";

  let description = [{
    Elementwise logical AND of input1 and input2. Axis of size 1 will be broadcast, as necessary.
    Rank of input tensors must match.
  }];

  let arguments = (ins
    I1Tensor:$lhs,
    I1Tensor:$rhs
  );

  let results = (outs
    I1Tensor:$z
  );

  let builders = [Tosa_BroadcastableBinaryBuilder];
}

/* Operator: logical_left_shift */

def Tosa_LogicalLeftShiftOp : Tosa_ElemwiseBinaryOp<"logical_left_shift", [ResultsBroadcastableShape, NoSideEffect]> {
  let summary = "Elementwise Logical Left Shift";

  let description = [{
    Elementwise left shift of input1 and input2. Axis of size 1 will be broadcast, as necessary.
    Rank of input tensors must match.
  }];

  let arguments = (
    ins Tosa_Tensor:$lhs,
    Tosa_Tensor:$rhs);

  let results = (outs Tosa_Tensor:$output);

  let builders = [Tosa_BroadcastableBinaryBuilder];
}

/* Operator: logical_right_shift */

def Tosa_LogicalRightShiftOp : Tosa_ElemwiseBinaryOp<"logical_right_shift", [ResultsBroadcastableShape, NoSideEffect]> {
  let summary = "Elementwise Logical Right Shift";

  let description = [{
    Elementwise logical right shift of input1 by the amount specified in input2.
    Axis of size 1 will be broadcast, as necessary.
    Rank of input tensors must match.
  }];

  let arguments = (
    ins Tosa_Tensor:$lhs,
    Tosa_Tensor:$rhs);

  let results = (outs Tosa_Tensor:$output);

  let builders = [Tosa_BroadcastableBinaryBuilder];
}

/* Operator: logical_or */

def Tosa_LogicalOrOp : Tosa_ElemwiseBinaryOp<"logical_or", [ResultsBroadcastableShape, Commutative, NoSideEffect]> {
  let summary = "Returns the truth value of x OR y element-wise.";

  let description = [{
    Elementwise logical OR of input1 and input2. Axis of size 1 will be broadcast as necessary.
    Rank of input tensors must match.
  }];

  let arguments = (ins
    I1Tensor:$lhs,
    I1Tensor:$rhs
  );

  let results = (outs
    I1Tensor:$z
  );

  let builders = [Tosa_BroadcastableBinaryBuilder];
}

/* Operator: logical_xor */

def Tosa_LogicalXorOp : Tosa_ElemwiseBinaryOp<"logical_xor", [ResultsBroadcastableShape, Commutative, NoSideEffect]> {
  let summary = "Returns the truth value of x XOR y element-wise.";

  let description = [{
    Elementwise logical XOR of input tensor 0 and input tensor 1.
    Axis of size 1 will be broadcast as necessary.
    Rank of input tensors must match.
  }];

  let arguments = (ins
    I1Tensor:$lhs,
    I1Tensor:$rhs
  );

  let results = (outs
    I1Tensor:$z
  );

  let builders = [Tosa_BroadcastableBinaryBuilder];
}

/* Operator: maximum */

def Tosa_MaximumOp : Tosa_ElemwiseBinaryOp<"maximum", [ResultsBroadcastableShape, NoSideEffect, Commutative]> {
  let summary = "Elementwise Maximum";

  let description = [{
    Elementwise max of input1 and input2. Axis of size 1 will be broadcast, as necessary.
    Rank of input tensors must match.
  }];

  let arguments = (
    ins Tosa_Tensor:$lhs,
    Tosa_Tensor:$rhs);

  let results = (outs Tosa_Tensor:$output);

  let builders = [Tosa_BroadcastableBinaryBuilder];
}

/* Operator: minimum */

def Tosa_MinimumOp : Tosa_ElemwiseBinaryOp<"minimum", [ResultsBroadcastableShape, NoSideEffect, Commutative]> {
  let summary = "Elementwise Minimum";

  let description = [{
    Elementwise minimum of input tensor 0 and input tensor 1. Axis of size 1 will be broadcast, as necessary.
    Rank of input tensors must match.
  }];

  let arguments = (
    ins Tosa_Tensor:$lhs,
    Tosa_Tensor:$rhs);

  let results = (outs Tosa_Tensor:$output);

  let builders = [Tosa_BroadcastableBinaryBuilder];
}

/* Operator: mul */

def Tosa_MulOp : Tosa_ElemwiseBinaryOp<"mul", [ResultsBroadcastableShape, NoSideEffect, Commutative]> {
  let summary = "Multiplication operator";

  let description = [{
    Elementwise multiplication (Hadamard product) of input tensor 0 and input tensor 1.
    Axis of size 1 will be broadcast, as necessary.
    Rank of input tensors must match.
  }];

  let arguments = (
    ins Tosa_Tensor:$lhs,
    Tosa_Tensor:$rhs);

  let results = (outs Tosa_Tensor:$output);

  let builders = [Tosa_BroadcastableBinaryBuilder];
}

/* Operator: pow */

def Tosa_PowOp : Tosa_ElemwiseBinaryOp<"pow", [ResultsBroadcastableShape, NoSideEffect]> {
  let summary = "Computes the power of one value to another.";

  let description = [{
    Elementwise input tensor 0 value raised to the power of input 1 tensor.
    Axis of size 1 will be broadcast, as necessary.
    Rank of input tensors must match.
  }];

  let arguments = (ins
      Tosa_Tensor:$lhs,
      Tosa_Tensor:$rhs
      );

  let results = (outs
      Tosa_Tensor:$z
      );

  let builders = [Tosa_BroadcastableBinaryBuilder];
}

/* Operator: sub */

def Tosa_SubOp : Tosa_ElemwiseBinaryOp<"sub", [ResultsBroadcastableShape, NoSideEffect]> {
  let summary = "Elementwise subtraction operator";

  let description = [{
    Elementwise subtraction of input tensor 0 and input tensor 1.
    Axis of size 1 will be broadcast as necessary.
    Rank of input tensors must match.
  }];

  let arguments = (
    ins Tosa_Tensor:$lhs,
    Tosa_Tensor:$rhs);

  let results = (outs Tosa_Tensor:$output);

  let builders = [Tosa_BroadcastableBinaryBuilder];
}

def Tosa_TableOp : Tosa_ElemwiseBinaryOp<"table", [NoSideEffect]> {
  let summary = "Table lookup op";

  let description = [{
    Interpolated table lookup operation. Input values are scaled to create a fixed-point 9.7 value.
    The high 9 bits are used to index into the table. The fractional bits are used to interpolate
    based on the looked up value and the index+1 value in the table. The TABLE operator then returns
    a 16.7 interpolated value. Note that there must be 513 values to handle the full range of inputs.

    The TABLE operator is expected to be used as follows:
    • A RECALE node is expected before the TABLE operator to scale the input to a full int16_t range
    for the table lookup
    • If an int16_t result is required then follow the TABLE operator with a RESCALE with a right
    shift of 7
    • If an int8_t result is required then follow the TABLE operator with a RESCALE with a right
    shift of 15
  }];

  let arguments = ( ins
    Tosa_Tensor: $input,
    Tosa_Tensor: $lut
  );

  let results = (outs
    Tosa_Tensor:$output
  );

}

/* TOSA Spec Section 2.5 */
/* Operator Class: Elementwise unary/binary/ternary operators */
/* Operator Subclass: Elementwise unary ops */

def Tosa_AbsOp : Tosa_ElemwiseUnaryOp<"abs", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Elementwise abs op";

  let description = [{
    Elementwise absolute value operation
  }];

  let arguments = (ins Tosa_Tensor:$input
                  );

  let results = (outs Tosa_Tensor:$output
                );
}

/* Operator: bitwise_not */

def Tosa_BitwiseNotOp : Tosa_ElemwiseUnaryOp<"bitwise_not", [ResultsBroadcastableShape, NoSideEffect]> {
  let summary = "Bitwise NOT operator";

  let description = [{
    Elementwise bitwise NOT of input tensor.
  }];

  let arguments = (
    ins Tosa_Tensor:$input
  );

  let results = (outs Tosa_Tensor:$output);

}

def Tosa_CeilOp : Tosa_ElemwiseUnaryOp<"ceil", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Elementwise ceil op";

  let description = [{
    Elementwise ceiling operation
  }];

  let arguments = (ins Tosa_Tensor:$input
                  );

  let results = (outs Tosa_Tensor:$output
                );
}

def Tosa_ClzOp : Tosa_ElemwiseUnaryOp<"clz", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Elementwise count leading zero op";

  let description = [{
    Elementwise count leading zeros operation
  }];

  let arguments = (ins Tosa_Tensor:$input
                  );
  let results = (outs Tosa_Tensor:$output
                );
}

def Tosa_ExpOp : Tosa_ElemwiseUnaryOp<"exp", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Elementwise exp op";

  let description = [{
    Elementwise e to the x operation
  }];

  let arguments = (ins Tosa_Tensor:$input
                  );

  let results = (outs Tosa_Tensor:$output
                );
}

def Tosa_FloorOp : Tosa_ElemwiseUnaryOp<"floor", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Elementwise floor op";

  let description = [{
    Elementwise floor operation
  }];

  let arguments = (ins Tosa_Tensor:$input
                  );

  let results = (outs Tosa_Tensor:$output
                );
}

def Tosa_LogOp : Tosa_ElemwiseUnaryOp<"log", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Elementwise log op";

  let description = [{
    Elementwise natural logarithm operation
  }];

  let arguments = (ins Tosa_Tensor:$input
                  );

  let results = (outs Tosa_Tensor:$output
                );
}

/* Operator: logical_not */

def Tosa_LogicalNotOp : Tosa_ElemwiseBinaryOp<"logical_not", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Returns the truth value of NOT x element-wise.";

  let description = [{
    Elementwise logical NOT of input.
  }];

  let arguments = (ins
    I1Tensor:$x
  );

  let results = (outs
    I1Tensor:$y
  );

}

def Tosa_NegateOp : Tosa_ElemwiseUnaryOp<"negate", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Elementwise negate op";

  let description = [{
    Elementwise negation operation
  }];

  let arguments = (ins
      Tosa_Tensor:$input,
      OptionalAttr<Tosa_UnaryOpQuantizationAttr>:$quantization_info
  );
  let results = (outs Tosa_Tensor:$output
                );

  let builders = [Tosa_UnaryOpQuantInfoBuilder];
}

def Tosa_ReciprocalOp : Tosa_ElemwiseUnaryOp<"reciprocal", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Elementwise reciprocal op";

  let description = [{
    Elementwise reciprocal operation. For integer operation, a TABLE should be used
    with the appropriate ranges.
  }];

  let arguments = (ins Tosa_Tensor:$input
                  );
  let results = (outs Tosa_Tensor:$output
                );
}

def Tosa_RsqrtOp : Tosa_ElemwiseUnaryOp<"rsqrt", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Elementwise 1/sqrt op";

  let description = [{
    Elementwise reciprocal square root operation. For integer operation, a TABLE should be
    used with the appropriate ranges.
  }];

  let arguments = (ins Tosa_Tensor:$input
                  );
  let results = (outs Tosa_Tensor:$output
                );
}

/* TOSA Spec Section 2.6 */
/* Operator Class: Elementwise unary/binary/ternary operators */
/* Operator Subclass: Elementwise ternary ops */

/* Operator: select */

def Tosa_SelectOp : Tosa_ElemwiseTernaryOp<"select", [NoSideEffect]> {
  let summary = "Elementwise select operator";

  let description = [{
    Elementwise select of the output based on a condition.
  }];

  let arguments = (ins
      Tosa_Tensor:$condition,
      Tosa_Tensor:$a,
      Tosa_Tensor:$b
  );

  let results = (outs
      Tosa_Tensor:$output
  );

}

/* TOSA Spec Section 2.7 */
/* Operator Class: Logical Operations */

/* Operator: equal */

def Tosa_EqualOp : Tosa_ElemwiseCompareOp<"equal", [ResultsBroadcastableShape, Commutative, NoSideEffect]> {
  let summary = "Returns the truth value of (x == y) element-wise.";

  let description = [{
     Elementwise comparison operation
  }];

  let arguments = (ins
    Tosa_Tensor:$lhs,
    Tosa_Tensor:$rhs
  );

  let results = (outs
    I1Tensor:$z
  );

  let builders = [Tosa_BroadcastableBinaryBuilder];
}

/* Operator: greater */

def Tosa_GreaterOp : Tosa_ElemwiseCompareOp<"greater", [ResultsBroadcastableShape, NoSideEffect]> {
  let summary = "Returns the truth value of (x > y) element-wise.";

  let description = [{
    Elementwise greater than comparison operation
  }];

  let arguments = (ins
    Tosa_Tensor:$lhs,
    Tosa_Tensor:$rhs
  );

  let results = (outs
    I1Tensor:$z
  );

  let builders = [Tosa_BroadcastableBinaryBuilder];
}

/* Operator: greater_equal */

def Tosa_GreaterEqualOp : Tosa_ElemwiseCompareOp<"greater_equal", [ResultsBroadcastableShape, NoSideEffect]> {
  let summary = "Returns the truth value of (x >= y) element-wise.";

  let description = [{
    Elementwise comparison operation
  }];

  let arguments = (ins
    Tosa_Tensor:$lhs,
    Tosa_Tensor:$rhs
  );

  let results = (outs
    I1Tensor:$z
  );

  let builders = [Tosa_BroadcastableBinaryBuilder];
}

/* TOSA Spec Section 2.8 */
/* Operator Class: Reduction Ops */

/* Operator: reduce_all */

def Tosa_ReduceAllOp : Tosa_ReductionOp<"reduce_all", [NoSideEffect]> {
  let summary = [{
    Reduce All operator
  }];

  let description = [{
    Reduce a tensor along the given axis with a logical AND operation
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    I64Attr:$axis
  );

  let results = (outs
    Tosa_Tensor:$output
  );

}

/* Operator: reduce_any */

def Tosa_ReduceAnyOp : Tosa_ReductionOp<"reduce_any", [NoSideEffect]> {
  let summary = [{
    Reduce Any operator
  }];

  let description = [{
    Reduce a tensor along the given axis with a logical OR operation
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    I64Attr:$axis
  );

  let results = (outs
    Tosa_Tensor:$output
  );

}

/* Operator: reduce_max */

def Tosa_ReduceMaxOp : Tosa_ReductionOp<"reduce_max", [NoSideEffect]> {
  let summary = [{
    Reduce Max operator
  }];

  let description = [{
    Reduce a tensor along the given axis with a maximum operation
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    I64Attr:$axis
  );

  let results = (outs
    Tosa_Tensor:$output
  );

}

/* Operator: reduce_min */

def Tosa_ReduceMinOp : Tosa_ReductionOp<"reduce_min", [NoSideEffect]> {
  let summary = [{
    Reduce Min operator
  }];

  let description = [{
    Reduce a tensor along the given axis with a minimum operation
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    I64Attr:$axis
  );

  let results = (outs
    Tosa_Tensor:$output
  );

}

/* Operator: reduce_prod */

def Tosa_ReduceProdOp : Tosa_ReductionOp<"reduce_prod", [NoSideEffect]> {
  let summary = [{
    Reduce Prod operator
  }];

  let description = [{
    Reduce a tensor along the given axis by computing the product of the axis.
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    I64Attr:$axis
  );

  let results = (outs
    Tosa_Tensor:$output
  );

}

/* Operator: reduce_sum */

def Tosa_ReduceSumOp : Tosa_ReductionOp<"reduce_sum", [NoSideEffect]> {
  let summary = [{
    Reduce Sum operator
  }];

  let description = [{
    Reduce a tensor along the given axis by computing the sum of the axis.
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    I64Attr:$axis
  );

  let results = (outs
    Tosa_Tensor:$output
  );

}

/* TOSA Spec Section 2.9 */
/* Operator Class: Data Layout / Memory Reinterpretation */

/* Operator: concat */

def Tosa_ConcatOp : Tosa_DataLayoutOp<"concat", [NoSideEffect]> {
  let summary = "Concatenates tensors along one dimension.";

  let description = [{
    Concatenate two tensors along a given axis. No data conversion happens during a concat operation.
  }];

  let arguments = (ins
    Tosa_Tensor:$a,
    Tosa_Tensor:$b,
    I64Attr:$axis
  );

  let results = (outs
    Tosa_Tensor:$output
  );

}

/* Operator: pad */

def Tosa_PadOp : Tosa_DataLayoutOp<"pad", [NoSideEffect]> {
  let summary = "Pads a tensor with zeros.";

  let description = [{
    Zero-pads a tensor along borders of each dimension.
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    Tosa_Int32Or64Tensor:$paddings,
    OptionalAttr<Tosa_PadOpQuantizationAttr>:$quantization_info
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  let builders = [Tosa_PadOpQuantInfoBuilder];

}

/* Operator: reshape */

def Tosa_ReshapeOp: Tosa_DataLayoutOp<"reshape", [
   NoSideEffect]> {
  let summary = "Reshape operator";

  let description = [{
    Returns a tensor with the same type/values as the input, with a new shape specified by the shape
    argument. Reshape may operate on tensors of any rank. No data conversion happens during a reshape
    operation.
  }];

  let arguments = (
    ins Tosa_Tensor:$input,
    I64ArrayAttr:$shape);

  let results = (outs Tosa_Tensor:$output);
}

/* Operator: reverse */

def Tosa_ReverseOp: Tosa_DataLayoutOp<"reverse", [
   NoSideEffect]> {
  let summary = "Reverse operator";

  let description = [{
    Returns a tensor with the same type/values as the input, with the data reversed along the given
    axis. No data conversion happens during a reverse operation.
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    I64Attr:$axis);

  let results = (outs
    Tosa_Tensor:$output);
}

/* Operator: slice */

def Tosa_SliceOp: Tosa_DataLayoutOp<"slice", [
   NoSideEffect]> {
  let summary = "Slice operator";

  let description = [{
    Extracts a slice of the input tensor 0 on the given axis, beginning at the start coordinates,
    and extending for size elements in each direction. No data conversion happens during a slice operation.
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    I64ArrayAttr:$begin,
    I64ArrayAttr:$size
  );

  let results = (outs
    Tosa_Tensor:$output);
}

/* Operator: tile */

def Tosa_TileOp: Tosa_DataLayoutOp<"tile", [NoSideEffect]> {
  let summary = "Tile operator";

  let description = [{
    Replicates input 0 multiplies times along each dimension.
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    I64ArrayAttr:$multiples);

  let results = (outs
    Tosa_Tensor:$output);
}

/* Operator: transpose */

def Tosa_TransposeOp : Tosa_DataLayoutOp<"transpose", [NoSideEffect]> {
  let summary = "Transpose operator";

  let description = [{
    Permutes the dimensions based on perm.
  }];

  let arguments = (ins
    Tosa_Tensor:$x,
    Tosa_Int32Or64Tensor:$perm
  );

  let results = (
    outs Tosa_Tensor:$y
  );

}

/* TOSA Spec Section 2.10 */
/* Operator Class: Scatter/gather Operations */

/* Operator: gather */

def Tosa_GatherOp : Tosa_AggregationOp<"gather", [NoSideEffect]> {
  let summary = [{
    Gather operation
  }];

  let description = [{
    Generate a tensor for which each element in the output is a subtensor of the values tensor along
    the given axis, based on the value of indices.
  }];

  let arguments = (ins
    Tosa_Tensor:$params,
    Tosa_Int32Or64Tensor:$indices,
    I64Attr:$axis,

    DefaultValuedAttr<I64Attr, "0">:$batch_dims
  );

  let results = (outs
    Tosa_Tensor:$z
  );

}

/* TOSA Spec Section 2.11 */
/* Operator Class: Image Frontend Functions */

/* Operator: resize */

def Tosa_ResizeOp : Tosa_ImageOp<"resize", [NoSideEffect]> {

  let summary = "Resize operation, supports various resize/upsample modes";

  let description = [{
    Resizes a tensor. Resize is only allowed in the H and W dimensions. In expected use,
    stride_y is approximately (IH<<shift)/OH and stride_x is approximately (IW<<shift)/OW.
    OH and OW are also supplied as inputs since there may be off by one errors if calculating
    OH and OW from the strides.
  }];

  let arguments = (ins
      Tosa_Tensor:$input,
      I32ArrayAttr:$output_size,
      I32ArrayAttr:$stride,
      I32ArrayAttr:$offset,
      I32Attr:$shift,
      Tosa_ResizeTypeAttr:$mode);

  let results = (outs
      Tosa_Tensor:$output);

}

/* TOSA Spec Section 2.12 */
/* Operator Class: Type Conversion */

/* Operator: cast */

def Tosa_CastOp: Tosa_ConversionOp<"cast", [NoSideEffect, SameOperandsAndResultShape]> {

    let summary = "Cast operation";

    let description = [{
      Performs a set of permissible cast operations
        Mode                    Input   Output
        ---------------------------------------
        signed 8 to bool        int8    Boolean
        signed 16 to bool       int16   Boolean
        signed 32 to bool       int32   Boolean
        bool to 8               Boolean int8
        bool to 16              Boolean int16
        bool to 32              Boolean int32
        signed 8 to signed 16   int8    int16
        signed 8 to signed 32   int8    int32
        signed 16 to signed 8   int16   int8
        signed 16 to signed 32  int16   int32
        signed 32 to signed 8   int32   int8
        signed 32 to signed 16  int32   int16
        float to signed 8       float   int8
        float to signed 16      float   int16
        signed 8 to float       int8    float
        signed 16 to float      int16   float
     }];

  let arguments = (
    ins Tosa_Tensor:$input
  );

  let results = (outs Tosa_Tensor:$output);

}

/* Operator: rescale */

def Tosa_RescaleOp: Tosa_ConversionOp<"rescale", [NoSideEffect]> {
  let summary = "Tosa rescale operator";

  let description = [{
    Rescale quantized values into a new domain. Supported rescalings are:
    Mode                    Input   Output
    signed 8 to 8           aint8   aint8
    signed 8 to 16          aint8   int16
    signed 8 to 32          aint8   int32
    signed 16 to 8          int16   aint8
    signed 16 to 16         int16   int16
    signed 16 to 32         int16   int32
    signed 32 to 8          int32   aint8
    signed 32 to 16         int32   int16
    signed 32 to 32         int32   int32
    signed 48 to 8          int48   aint8
    signed 48 to 16         int48   int16
    signed 48 to 32         int48   int32
    unsigned 8 to signed 8  uint8   aint8
    signed 8 to unsigned 8  aint8   uint8
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    I32Attr:$input_zp,
    I32Attr:$output_zp,
    I32ArrayAttr:$multiplier,
    I32ArrayAttr:$shift,
    BoolAttr:$scale32,
    BoolAttr:$double_round,
    BoolAttr:$per_channel
  );

  let results = (outs
    Tosa_Tensor:$output
  );

}

/* TOSA Spec Section 2.13 */
/* Operator Class: Data Node Ops */

/* Operator: const */

def Tosa_ConstOp : Tosa_DataNodeOp<"const", [ConstantLike, NoSideEffect, FirstAttrDerivedResultType]> {
  let summary = "Constant op.";

  let description = [{
    A node containing constant data for use as the input to an operation. May hold data
    in any of the supported data formats.
  }];

  let arguments = (ins ElementsAttr:$value);

  let results = (outs AnyTensor:$output);

  let builders = [
        OpBuilderDAG<(ins "Type":$type, "Attribute":$value)>,
  ];

}

/* Operator: identity */

def Tosa_IdentityOp: Tosa_DataNodeOp<"identity", [NoSideEffect]> {
    let summary = "Identity operator";
    let description = [{
        Returns a tensor with the same shape, size, type
        and content as the input.
    }];

   let arguments = (ins
      Tosa_Tensor:$input
    );

   let results = (outs
      Tosa_Tensor:$output);
}

/* Operator: identityn */

def Tosa_IdentityNOp: Tosa_DataNodeOp<"identityn", [NoSideEffect]> {
    let summary = "IdentityN operator";
    let description = [{
      Returns a list of tensors with the same shape, type, and contents as the
      input list of tensors.
    }];

   let arguments = (ins
    Variadic<Tosa_Tensor>:$input
    );

   let results = (outs
      Variadic<Tosa_Tensor>:$output);
}

/* Operator: placeholder */

def Tosa_PlaceholderOp : Tosa_DataNodeOp<"placeholder", [NoSideEffect]> {
  let summary = "Placeholder op";

  let description = [{
    A node where data will be inserted into the network at runtime. Generally used for inputs to the network.
  }];

  let arguments = (ins
  );

  let results = (outs
    Tosa_Tensor:$output
  );

}

/* TOSA Spec Section 2.14 */
/* Operator Class: Custom Operators */

/* Operator: custom */

def Tosa_CustomOp : Tosa_Op<"custom"> {

  let summary = "Custom operator wrapper for Tosa";

  let description = [{
    Hardware implementing TOSA may choose to add additional custom operators that are not expressed in
    the existing TOSA operations. These operators are not expected to be portable across TOSA
    implementations. The input and output signatures must be expressed in the corresponding TOSA node.
  }];

  let arguments = (ins
    StrAttr:$identifier,
    Variadic<Tosa_Tensor>:$inputs
  );

  let results = (outs
    Variadic<Tosa_Tensor>:$outputs
  );

}

/* TOSA Spec Section 2.15 */
/* Operator Class: Control Flow Operators */

/* Operator: cond_if */

def Tosa_IfOp : Tosa_Op<"cond_if", [
       SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Conditional if operator";

  let description = [{
    Evaluates a Boolean condition and then takes one of two distinct execution paths. This
    implements the semantic If-then-else structure.
  }];

  let arguments = (ins
    Tosa_Tensor:$cond,
    Variadic<Tosa_Tensor>:$inputs
  );

  let results = (outs
    Variadic<Tosa_Tensor>:$output
  );

  let regions = (region
    SizedRegion<1>:$then_branch,
    SizedRegion<1>:$else_branch
  );

}

/* Operator: while_loop */

def Tosa_WhileOp : Tosa_Op<"while_loop", [
       DeclareOpInterfaceMethods<LoopLikeOpInterface>,
       SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = [{
    output = input; While (Cond(output)) { output = Body(output) }
  }];

  let description = [{
    Generates and evaluates a Bool condition and either executes a loop body or exits to
    another control point. This action is performed repeatedly after updating and re-evaluating
    the Boolean condition every iteration. This implements the semantic foreach or while
    iterative loop structure.
  }];

  let arguments = (ins
    Variadic<Tosa_Tensor>:$inputs
  );

  let results = (outs
    Variadic<Tosa_Tensor>:$output);

  let regions = (region
    SizedRegion<1>:$cond,
    SizedRegion<1>:$body
  );

}

def Tosa_YieldOp : Tosa_Op<"yield", [Terminator]> {
  let summary = "yield operator";

  let description = [{
    return operation within the conditional abd body of
    structured control flow. Operation takes variadic operands
    but produces no results of its own.
  }];

  let arguments = (ins
    Variadic<Tosa_Tensor>:$inputs
  );

}

#endif // TOSA_OPS
