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

#ifndef VCIXIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

//===----------------------------------------------------------------------===//
// VCIX dialect definition.
//===----------------------------------------------------------------------===//

def VCIX_Dialect : Dialect {
  let name = "vcix";
  let cppNamespace = "::mlir::vcix";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let description = [{
     The SiFive Vector Coprocessor Interface (VCIX) provides a flexible mechanism
     to extend application processors with custom coprocessors and
     variable-latency arithmetic units. The interface offers throughput comparable
     to that of standard RISC-V vector instructions. To accelerate performance,
     system designers may use VCIX as a low-latency, high-throughput interface to
     a coprocessor

     https://www.sifive.com/document-file/sifive-vector-coprocessor-interface-vcix-software
  }];
}

// Special version for intrinsic version where int attr is zext to i32 or i64
// depending on xlen of the target.
def VCIX_VectorOrScalar
    : AnyTypeOf<[LLVM_AnyVector, I<64>, I<32>, F<16>, F<32>, F<64>]>;
def VCIX_OpcodeAttr : AnyAttrOf<[I32Attr, I64Attr]>;
def VCIX_Register : AnyTypeOf<[I32, I64]>;
def VCIX_ImmAttr : AnyAttrOf<[I32Attr, I64Attr]>;
def VCIX_VL : AnyTypeOf<[I<64>, I<32>]>;

class VCIX_Op<string mnemonic, list<Trait> traits = []>
    : LLVM_OpBase<VCIX_Dialect, mnemonic, traits> {
}

def VCIX_BinaryImmOp : VCIX_Op<"v.iv">,
    Results<(outs LLVM_AnyVector: $res)>,
    Arguments<(ins VCIX_OpcodeAttr: $opcode,
                   LLVM_AnyVector: $vs2,
                   VCIX_ImmAttr: $imm,
                   Optional<VCIX_VL>: $vl)> {
  let summary = "Binary VCIX operation with an immediate second operand";
  let description = [{
    Binary VCIX operation with an immediate second operand.

    Correponds to:
    |Mnemonic|funct6|vm|rs2|rs1|funct3|rd|Destination|Sources|
    |--|--|--|--|--|--|--|--|--|
    |sf.vc.v.iv|0010--|0|vs2|simm|011|vd|vector vd| simm[4:0]  vector vs2|
  }];

  string llvmBuilder = [{
      llvm::Type *xlen =getXlenType($opcode, moduleTranslation);
      llvm::Value *opcodeConst = mlir::LLVM::detail::getLLVMConstant(
          xlen, $opcode, $_location, moduleTranslation);
      llvm::Value *immConst = mlir::LLVM::detail::getLLVMConstant(
          xlen, $imm, $_location, moduleTranslation);
      VectorType vt = op.getResult().getType().cast<VectorType>();
      llvm::Value *vl =
          createVL(builder, $vl, vt, xlen, $_location, moduleTranslation);
      $res = createIntrinsicCall(
          builder, llvm::Intrinsic::riscv_sf_vc_v_iv_se,
          {opcodeConst, $vs2, immConst, vl},
          {$_resultType, xlen, $vs2->getType(), xlen, xlen});
  }];
}

def VCIX_BinaryOp : VCIX_Op<"v.sv">,
    Results<(outs LLVM_AnyVector: $res)>,
    Arguments<(ins VCIX_OpcodeAttr: $opcode,
                   LLVM_AnyVector: $vs2,
                   VCIX_VectorOrScalar: $op,
                   Optional<VCIX_VL>: $vl)> {
  let summary = "Binary VCIX operation";
  let description = [{
    Binary VCIX operation with an integer scalar, or floating pointer scalar or
    vector second operand.

    Correponds to:
    |Mnemonic|funct6|vm|rs2|rs1|funct3|rd|Destination| Sources|
    |--|--|--|--|--|--|--|--|--|--|
    |sf.vc.v.vv|0010--|0|vs2|vs1|000|vd|vector vd|vector vs1, vector vs|
    |sf.vc.v.xv|0010--|0|vs2|xs1|100|vd|vector vd|scalar xs1, vector vs2|
    |sf.vc.v.fv|0010--|0|vs2|fs1|101|vd|vector vd|scalar fs1, vector vs2|
  }];

  string llvmBuilder = [{
      llvm::Type *xlen = getXlenType($opcode, moduleTranslation);
      llvm::Value *opcodeConst = mlir::LLVM::detail::getLLVMConstant(
          xlen, $opcode, $_location, moduleTranslation);
      llvm::Intrinsic::ID id;
      llvm::Type *opType = $op->getType();
      if (opType->isVectorTy()) {
        id = llvm::Intrinsic::riscv_sf_vc_v_vv_se;
      } else if (opType->isIntegerTy()) {
        id = llvm::Intrinsic::riscv_sf_vc_v_xv_se;
      } else {
        id = llvm::Intrinsic::riscv_sf_vc_v_fv_se;
      }
      VectorType vt = op.getResult().getType().cast<VectorType>();
      llvm::Value *vl =
          createVL(builder, $vl, vt, xlen, $_location, moduleTranslation);
      $res = createIntrinsicCall(
          builder, id, {opcodeConst, $vs2, $op, vl},
          {$_resultType, xlen, $vs2->getType(), $op->getType(), xlen});
  }];
}

#endif // VCIXIR_OPS
