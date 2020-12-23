//===- SPIRVOCLOps.td - OpenCL extended insts spec file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the op definition spec of OpenCL extension ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_OCL_OPS
#define MLIR_DIALECT_SPIRV_IR_OCL_OPS

include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"

//===----------------------------------------------------------------------===//
// SPIR-V OpenCL opcode specification.
//===----------------------------------------------------------------------===//

// Base class for all OpenCL ops.
class SPV_OCLOp<string mnemonic, int opcode, list<OpTrait> traits = []> :
  SPV_ExtInstOp<mnemonic, "OCL", "OpenCL.std", opcode, traits>;

// Base class for OpenCL unary ops.
class SPV_OCLUnaryOp<string mnemonic, int opcode, Type resultType,
                     Type operandType, list<OpTrait> traits = []> :
  SPV_OCLOp<mnemonic, opcode, !listconcat([NoSideEffect], traits)> {

  let arguments = (ins
    SPV_ScalarOrVectorOf<operandType>:$operand
  );

  let results = (outs
    SPV_ScalarOrVectorOf<resultType>:$result
  );

  let parser = [{ return parseUnaryOp(parser, result); }];

  let printer = [{ return printUnaryOp(getOperation(), p); }];

  let verifier = [{ return success(); }];
}

// Base class for OpenCL Unary arithmetic ops where return type matches
// the operand type.
class SPV_OCLUnaryArithmeticOp<string mnemonic, int opcode, Type type,
                               list<OpTrait> traits = []> :
  SPV_OCLUnaryOp<mnemonic, opcode, type, type, traits>;

// Base class for OpenCL binary ops.
class SPV_OCLBinaryOp<string mnemonic, int opcode, Type resultType,
                      Type operandType, list<OpTrait> traits = []> :
  SPV_OCLOp<mnemonic, opcode, !listconcat([NoSideEffect], traits)> {

  let arguments = (ins
    SPV_ScalarOrVectorOf<operandType>:$lhs,
    SPV_ScalarOrVectorOf<operandType>:$rhs
  );

  let results = (outs
    SPV_ScalarOrVectorOf<resultType>:$result
  );

  let parser = [{ return impl::parseOneResultSameOperandTypeOp(parser, result); }];

  let printer = [{ return impl::printOneResultOp(getOperation(), p); }];

  let verifier = [{ return success(); }];
}

// Base class for OpenCL Binary arithmetic ops where operand types and
// return type matches.
class SPV_OCLBinaryArithmeticOp<string mnemonic, int opcode, Type type,
                                list<OpTrait> traits = []> :
  SPV_OCLBinaryOp<mnemonic, opcode, type, type, traits>;

// -----

def SPV_OCLExpOp : SPV_OCLUnaryArithmeticOp<"exp", 19, SPV_Float> {
  let summary = "Exponentiation of Operand 1";

  let description = [{
    Compute the base-e exponential of x. (i.e. ex)

    Result Type and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand,
    must be of the same type.

    <!-- End of AutoGen section -->
    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    exp-op ::= ssa-id `=` `spv.OCL.exp` ssa-use `:`
               float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.OCL.exp %0 : f32
    %3 = spv.OCL.exp %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_OCLFAbsOp : SPV_OCLUnaryArithmeticOp<"fabs", 23, SPV_Float> {
  let summary = "Absolute value of operand";

  let description = [{
    Compute the absolute value of x.

    Result Type and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand,
    must be of the same type.

    <!-- End of AutoGen section -->
    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    abs-op ::= ssa-id `=` `spv.OCL.fabs` ssa-use `:`
               float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.OCL.fabs %0 : f32
    %3 = spv.OCL.fabs %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_OCLSAbsOp : SPV_OCLUnaryArithmeticOp<"s_abs", 141, SPV_Integer> {
  let summary = "Absolute value of operand";

  let description = [{
    Returns |x|, where x is treated as signed integer.

    Result Type and x must be integer or vector(2,3,4,8,16) of
    integer values.

    All of the operands, including the Result Type operand,
    must be of the same type.

    <!-- End of AutoGen section -->
    ```
    integer-scalar-vector-type ::= integer-type |
                                   `vector<` integer-literal `x` integer-type `>`
    abs-op ::= ssa-id `=` `spv.OCL.s_abs` ssa-use `:`
               integer-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.OCL.s_abs %0 : i32
    %3 = spv.OCL.s_abs %1 : vector<3xi16>
    ```
  }];
}

#endif // MLIR_DIALECT_SPIRV_IR_OCL_OPS
