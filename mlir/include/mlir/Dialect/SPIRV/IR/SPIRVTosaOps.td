//===- SPIRVTosaOps.td - TOSA extended insts spec file -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the op definition spec of TOSA extension ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_TOSA_OPS
#define MLIR_DIALECT_SPIRV_IR_TOSA_OPS

include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"
include "mlir/Dialect/SPIRV/IR/SPIRVGraphOps.td"
include "mlir/Dialect/SPIRV/IR/SPIRVTosaTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// SPIR-V TOSA opcode specification.
//===----------------------------------------------------------------------===//

// Base class for all TOSA ops.
class SPIRV_TosaOp<string mnemonic, int opcode, list<Trait> traits = []> :
  SPIRV_ExtInstOp<mnemonic, "Tosa", "TOSA.001000.1", opcode,
  !listconcat(traits, [InGraphScope])> {

  let availability = [
    MinVersion<SPIRV_V_1_5>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_ARM_graph]>,
    Capability<[SPIRV_C_GraphARM]>
  ];
}


def SPIRV_TosaArgMaxOp : SPIRV_TosaOp<"ArgMax", 0, [Pure]> {
  let summary = "Perform argmax on the input.";

  let description = [{
    Returns the index with the largest value across the given axis of the
    input tensor. If multiple locations have equal values, returns the first
    match along the search axis.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_argmax
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_argmax

    #### Example:
    ```mlir
    %2 = spirv.Tosa.ArgMax axis = 3, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<3x28x17x17xi8> -> !spirv.arm.tensor<3x28x17xi32>
    %2 = spirv.Tosa.ArgMax axis = 3, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<2x2x7x14xf32> -> !spirv.arm.tensor<2x2x14xi32>
    ```
  }];

  let arguments = (ins
    SPIRV_TensorArmAxisAttr: $axis,
    SPIRV_TosaExtNaNPropagationModeAttr: $nan_mode,
    SPIRV_TosaNumerical_TensorArm: $input
  );

  let results = (outs
    SPIRV_Int32_TensorArmUpTo5D: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `axis` `=` $axis `,` `nan_mode` `=` $nan_mode `,`
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}

#endif // MLIR_DIALECT_SPIRV_IR_TOSA_OPS
