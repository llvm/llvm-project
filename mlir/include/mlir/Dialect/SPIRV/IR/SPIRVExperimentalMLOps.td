//===- SPIRVExperimentalMLOps.td - Experimental ML ops ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_EXPERIMENTAL_ML_OPS
#define MLIR_DIALECT_SPIRV_IR_EXPERIMENTAL_ML_OPS

include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"

class SPIRV_ExperimentalMLOp<string mnemonic, int opcode,
                             list<Trait> traits = []> :
  SPIRV_ExtInstOp<mnemonic, "ExperimentalML", "Arm.ExperimentalMLOperations.1",
                  opcode, traits> {
  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[]>
  ];

  let hasVerifier = 0;
}

def SPIRV_ExperimentalMLCallOp : SPIRV_ExperimentalMLOp<"Call", 0> {
  let summary = "Call an Arm experimental ML operation.";

  let description = [{
    Calls an operation encoded using the Arm.ExperimentalMLOperations extended
    instruction set. The `opcode` attribute is serialized as the operation
    opcode literal integer operand of CALL.
  }];

  let arguments = (ins
    I32Attr:$opcode,
    Variadic<AnyType>:$parameters
  );

  let results = (outs
    AnyType:$output
  );

  let assemblyFormat = [{
    `opcode` `=` $opcode `,`
    $parameters attr-dict `:` functional-type($parameters, $output)
  }];
}

#endif // MLIR_DIALECT_SPIRV_IR_EXPERIMENTAL_ML_OPS
