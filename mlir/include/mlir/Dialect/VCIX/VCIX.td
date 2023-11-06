//===-- VCIX.td - VCIX dialect operation definitions *- tablegen -*--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The file defines the basic operations for the VCIX dialect.
//
// The SiFive Vector Coprocessor Interface (VCIX) provides a flexible mechanism
// to extend application processors with custom coprocessors and
// variable-latency arithmetic units. The interface offers throughput comparable
// to that of standard RISC-V vector instructions. To accelerate performance,
// system designers may use VCIX as a low-latency, high-throughput interface to
// a coprocessor
//
// https://www.sifive.com/document-file/sifive-vector-coprocessor-interface-vcix-software
//
//===----------------------------------------------------------------------===//
#ifndef VCIX
#define VCIX

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/Dialect/VCIX/VCIXAttrs.td"

//===----------------------------------------------------------------------===//
// VCIX dialect definition.
//===----------------------------------------------------------------------===//

def VCIX_Dialect : Dialect {
  let name = "vcix";
  let cppNamespace = "::mlir::vcix";
  let description = [{
     The SiFive Vector Coprocessor Interface (VCIX) provides a flexible mechanism
     to extend application processors with custom coprocessors and
     variable-latency arithmetic units. The interface offers throughput comparable
     to that of standard RISC-V vector instructions. To accelerate performance,
     system designers may use VCIX as a low-latency, high-throughput interface to
     a coprocessor

     https://www.sifive.com/document-file/sifive-vector-coprocessor-interface-vcix-software
  }];

  let usePropertiesForAttributes = 1;
}

//===----------------------------------------------------------------------===//
// VCIX Ops
//===----------------------------------------------------------------------===//
class VCIX_Op<string mnemonic, list<Trait> traits = []>
    : Op<VCIX_Dialect, mnemonic, traits> {}

class VCIX_IntrinOp<string mnemonic, list<Trait> traits = []>
    : LLVM_OpBase<VCIX_Dialect, "intrin." #mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// Unary VCIX operations
//===----------------------------------------------------------------------===//
def VCIX_UnaryROOp : VCIX_Op<"unary.ro", []> {
  let summary = "Unary VCIX operation with side effects and without result";
  let description = [{
    Unary VCIX operation that has some side effects and does not produce result

    Correponds to
    ```
    Mnemonic    funct6 vm  rs2   rs1  funct3  rd     Destination   Sources
    sf.vc.x     0000-- 1  -----  xs1   100   -----   none          scalar xs1
    sf.vc.i     0000-- 1  ----- simm   011   -----   none          simm[4:0]
    ```
  }];

  let arguments = (ins AnyTypeOf<[I<64>, I<32>, I<5>]>: $op,
                       RVLType: $rvl,
                       OpcodeIntAttr: $opcode,
                       VCIX_SewLmulAttr: $sew_lmul,
                       RAttr: $rs2,
                       RAttr: $rd);

  let assemblyFormat = [{
    $sew_lmul $op `,` $rvl  attr-dict `:` `(` type($op) `,` type($rvl) `)`
  }];

  let hasVerifier = 1;
}

def VCIX_UnaryOp : VCIX_Op<"unary", []> {
  let summary = "unary VCIX operation";
  let description = [{
    Unary VCIX operation that produces result

    Correponds to
    ```
    Mnemonic    funct6 vm  rs2   rs1  funct3  rd     Destination   Sources
    sf.vc.v.x   0000-- 0  -----  xs1   100    vd     vector vd     scalar xs1
    sf.vc.v.i   0000-- 0  ----- simm   011    vd     vector vd     simm[4:0]
    ```
  }];

  let arguments = (ins AnyTypeOf<[I<64>, I<32>, I<5>]>: $op,
                       Optional<RVLType>: $rvl,
                       OpcodeIntAttr: $opcode,
                       RAttr: $rs2);

  let results = (outs VectorOfRank1: $result);

  let assemblyFormat = [{
    $op (`,` $rvl^)?  attr-dict
      `:` `(` type($op) (`,` type($rvl)^)? `)` `->` type($result)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Binary VCIX operations
//===----------------------------------------------------------------------===//
def VCIX_BinaryROOp : VCIX_Op<"binary.ro", []> {
  let summary = "Read-only binary VCIX operation";
  let description = [{
    Read-only binary VCIX operation that does not produce result

    Correponds to
    ```
    Mnemonic    funct6 vm  rs2   rs1  funct3  rd     Destination   Sources
    sf.vc.v.vv  0010-- 1   vs2   vs1   000    -----     none       vector vs1, vector vs2
    sf.vc.v.xv  0010-- 1   vs2   xs1   100    -----     none       scalar xs1, vector vs2
    sf.vc.v.iv  0010-- 1   vs2  simm   011    -----     none       simm[4:0],  vector vs2
    sf.vc.v.fv  0010-- 1   vs2   fs1   101    -----     none       scalar fs1, vector vs2
    ```
  }];

  let arguments = (ins VectorOfRank1OrScalar: $op1,
                       VectorOfRank1: $op2,
                       Optional<RVLType>: $rvl,
                       OpcodeIntAttr: $opcode,
                       RAttr: $rd);

  let assemblyFormat = [{
    $op1 `,` $op2 (`,` $rvl^)? attr-dict `:`
      `(` type($op1) `,` type($op2) (`,` type($rvl)^)? `)`
  }];

  let hasVerifier = 1;
}

def VCIX_BinaryOp : VCIX_Op<"binary", []> {
  let summary = "binary VCIX operation";
  let description = [{
    Binary VCIX operation that produces result

    Correponds to
    ```
    Mnemonic    funct6 vm  rs2   rs1  funct3  rd     Destination   Sources
    sf.vc.v.vv  0010-- 0   vs2   vs1   000    vd     vector vd     vector vs1, vector vs2
    sf.vc.v.xv  0010-- 0   vs2   xs1   100    vd     vector vd     scalar xs1, vector vs2
    sf.vc.v.iv  0010-- 0   vs2  simm   011    vd     vector vd     simm[4:0],  vector vs2
    sf.vc.v.fv  0010-- 0   vs2   fs1   101    vd     vector vd     scalar fs1, vector vs2
    ```
  }];

  let arguments = (ins VectorOfRank1OrScalar: $op1,
                       VectorOfRank1: $op2,
                       Optional<RVLType>: $rvl,
                       OpcodeIntAttr: $opcode);

  let results = (outs VectorOfRank1: $result);

  let assemblyFormat = [{
    $op1 `,` $op2 (`,` $rvl^)? attr-dict `:`
      `(` type($op1) `,` type($op2) (`,` type($rvl)^)? `)` `->` type($result)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Ternary VCIX operations
//===----------------------------------------------------------------------===//
def VCIX_TernaryROOp : VCIX_Op<"ternary.ro", []> {
  let summary = "Ternary VCIX operation";
  let description = [{
    Ternary VCIX operation that does not generate result

    Correponds to
    ```
    Mnemonic    funct6 vm  rs2   rs1  funct3  rd     Destination   Sources
    sf.vc.vvv   1010-- 1   vs2   vs1  000     vd        none       vector vs1, vector vs2, vector vd
    sf.vc.xvv   1010-- 1   vs2   xs1  100     vd        none       scalar xs1, vector vs2, vector vd
    sf.vc.ivv   1010-- 1   vs2   simm 011     vd        none       simm[4:0], vector vs2, vector vd
    sf.vc.fvv   10101- 1   vs2   fs1  101     vd        none       scalar fs1, vector vs2, vector vd
    ```
  }];

  let arguments = (ins VectorOfRank1OrScalar: $op1,
                       VectorOfRank1: $op2,
                       VectorOfRank1: $op3,
                       Optional<RVLType>: $rvl,
                       OpcodeIntAttr: $opcode);

  let assemblyFormat = [{
    $op1 `,` $op2 `,` $op3 (`,` $rvl^)? attr-dict `:`
      `(` type($op1) `,` type($op2) `,` type($op3) (`,` type($rvl)^)? `)`
  }];

  let hasVerifier = 1;
}

def VCIX_TernaryOp : VCIX_Op<"ternary", []> {
  let summary = "Ternary VCIX operation";
  let description = [{
    Ternary VCIX operation that produces result

    Correponds to
    ```
    Mnemonic    funct6 vm  rs2   rs1  funct3  rd     Destination   Sources
    sf.vc.v.vvv 1010-- 0   vs2   vs1  000     vd     vector vd     vector vs1, vector vs2, vector vd
    sf.vc.v.xvv 1010-- 0   vs2   xs1  100     vd     vector vd     scalar xs1, vector vs2, vector vd
    sf.vc.v.ivv 1010-- 0   vs2   simm 011     vd     vector vd     simm[4:0], vector vs2, vector vd
    sf.vc.v.fvv 10101- 0   vs2   fs1  101     vd     vector vd     scalar fs1, vector vs2, vector vd
    ```
  }];

  let arguments = (ins VectorOfRank1OrScalar: $op1,
                       VectorOfRank1: $op2,
                       VectorOfRank1: $op3,
                       Optional<RVLType>: $rvl,
                       OpcodeIntAttr: $opcode);

  let results = (outs VectorOfRank1: $result);

  let assemblyFormat = [{
    $op1 `,` $op2 `,` $op3 (`,` $rvl^)? attr-dict `:`
      `(` type($op1) `,` type($op2) `,` type($op3) (`,` type($rvl)^)? `)` `->` type($result)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Wide ternary VCIX operations
//===----------------------------------------------------------------------===//
def VCIX_WideTernaryROOp : VCIX_Op<"wide.ternary.ro", []> {
  let summary = "Ternary VCIX operation";
  let description = [{
    Wide Ternary VCIX operation that does not produce result

    Correponds to
    ```
    Mnemonic    funct6 vm  rs2   rs1  funct3  rd     Destination   Sources
    sf.vc.vvw   1111-- 1   vs2   vs1  000     vd      none         vector vs1, vector vs2, wide vd
    sf.vc.xvw   1111-- 1   vs2   xs1  100     vd      none         scalar xs1, vector vs2, wide vd
    sf.vc.ivw   1111-- 1   vs2   simm 011     vd      none         simm[4:0], vector vs2, wide vd
    sf.vc.fvw   11111- 1   vs2   fs1  101     vd      none         scalar fs1, vector vs2, wide vd
    ```
  }];

  let arguments = (ins VectorOfRank1OrScalar: $op1,
                       VectorOfRank1: $op2,
                       VectorOfRank1: $op3,
                       Optional<RVLType>: $rvl,
                       OpcodeIntAttr: $opcode);

  let assemblyFormat = [{
    $op1 `,` $op2 `,` $op3 (`,` $rvl^)? attr-dict `:`
      `(` type($op1) `,` type($op2) `,` type($op3) (`,` type($rvl)^)? `)`
  }];

  let hasVerifier = 1;
}

def VCIX_WideTernaryOp : VCIX_Op<"wide.ternary", []> {
  let summary = "Ternary VCIX operation";
  let description = [{
    Wide Ternary VCIX operation that produces result

    Correponds to
    ```
    Mnemonic    funct6 vm  rs2   rs1  funct3  rd     Destination   Sources
    sf.vc.v.vvw 1111-- 0   vs2   vs1  000     vd     wide vd       vector vs1, vector vs2, wide vd
    sf.vc.v.xvw 1111-- 0   vs2   xs1  100     vd     wide vd       scalar xs1, vector vs2, wide vd
    sf.vc.v.ivw 1111-- 0   vs2   simm 011     vd     wide vd       simm[4:0], vector vs2, wide vd
    sf.vc.v.fvw 11111- 0   vs2   fs1  101     vd     wide vd       scalar fs1, vector vs2, wide vd
    ```
  }];

  let arguments = (ins VectorOfRank1OrScalar: $op1,
                       VectorOfRank1: $op2,
                       VectorOfRank1: $op3,
                       Optional<RVLType>: $rvl,
                       OpcodeIntAttr: $opcode);

  let results = (outs VectorOfRank1: $result);

  let assemblyFormat = [{
    $op1 `,` $op2 `,` $op3 (`,` $rvl^)? attr-dict `:`
      `(` type($op1) `,` type($op2) `,` type($op3) (`,` type($rvl)^)? `)` `->` type($result)
  }];

  let hasVerifier = 1;
}
#endif // VCIX
