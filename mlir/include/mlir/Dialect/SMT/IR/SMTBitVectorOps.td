//===- SMTBitVectorOps.td - SMT bit-vector dialect ops -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SMT_IR_SMTBITVECTOROPS_TD
#define MLIR_DIALECT_SMT_IR_SMTBITVECTOROPS_TD

include "mlir/Dialect/SMT/IR/SMTDialect.td"
include "mlir/Dialect/SMT/IR/SMTAttributes.td"
include "mlir/Dialect/SMT/IR/SMTTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class SMTBVOp<string mnemonic, list<Trait> traits = []> :
  Op<SMTDialect, "bv." # mnemonic, traits>;

def BVConstantOp : SMTBVOp<"constant", [
  Pure,
  ConstantLike,
  FirstAttrDerivedResultType,
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
  let summary = "produce a constant bit-vector";
  let description = [{
    This operation produces an SSA value equal to the bit-vector constant
    specified by the 'value' attribute.
    Refer to the `BitVectorAttr` documentation for more information about
    the semantics of bit-vector constants, their format, and associated sort.
    The result type always matches the attribute's type.

    Examples:
    ```mlir
    %c92_bv8 = smt.bv.constant #smt.bv<92> : !smt.bv<8>
    %c5_bv4 = smt.bv.constant #smt.bv<5> : !smt.bv<4>
    ```
  }];

  let arguments = (ins BitVectorAttr:$value);
  let results = (outs BitVectorType:$result);

  let assemblyFormat = "qualified($value) attr-dict";

  let builders = [
    OpBuilder<(ins "const llvm::APInt &":$value), [{
      build($_builder, $_state,
            BitVectorAttr::get($_builder.getContext(), value));
    }]>,
    OpBuilder<(ins "uint64_t":$value, "unsigned":$width), [{
      build($_builder, $_state,
            BitVectorAttr::get($_builder.getContext(), value, width));
    }]>,
  ];

  let hasFolder = true;
}

class BVArithmeticOrBitwiseOp<string mnemonic, string desc> :
    SMTBVOp<mnemonic, [Pure, SameOperandsAndResultType]> {
  let summary = "equivalent to bv" # mnemonic # " in SMT-LIB";
  let description = "This operation performs " # desc # [{. The semantics are
    equivalent to the `bv}] # mnemonic # [{` operator defined in the SMT-LIB 2.7
    standard. More precisely in the [theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
    and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
    describing closed quantifier-free formulas over the theory of fixed-size
    bit-vectors.
  }];

  let results = (outs BitVectorType:$result);
}

class BinaryBVOp<string mnemonic, string desc> :
     BVArithmeticOrBitwiseOp<mnemonic, desc> {
  let arguments = (ins BitVectorType:$lhs, BitVectorType:$rhs);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($result))";
}

class UnaryBVOp<string mnemonic, string desc> :
    BVArithmeticOrBitwiseOp<mnemonic, desc> {
  let arguments = (ins BitVectorType:$input);
  let assemblyFormat = "$input attr-dict `:` qualified(type($result))";
}

def BVNotOp  : UnaryBVOp<"not", "bitwise negation">;
def BVNegOp  : UnaryBVOp<"neg", "two's complement unary minus">;

def BVAndOp  : BinaryBVOp<"and", "bitwise AND">;
def BVOrOp   : BinaryBVOp<"or", "bitwise OR">;
def BVXOrOp  : BinaryBVOp<"xor", "bitwise exclusive OR">;

def BVAddOp  : BinaryBVOp<"add", "addition">;
def BVMulOp  : BinaryBVOp<"mul", "multiplication">;
def BVUDivOp : BinaryBVOp<"udiv", "unsigned division (rounded towards zero)">;
def BVSDivOp : BinaryBVOp<"sdiv", "two's complement signed division">;
def BVURemOp : BinaryBVOp<"urem", "unsigned remainder">;
def BVSRemOp : BinaryBVOp<"srem",
  "two's complement signed remainder (sign follows dividend)">;
def BVSModOp : BinaryBVOp<"smod",
  "two's complement signed remainder (sign follows divisor)">;
def BVShlOp  : BinaryBVOp<"shl", "shift left">;
def BVLShrOp : BinaryBVOp<"lshr", "logical shift right">;
def BVAShrOp : BinaryBVOp<"ashr", "arithmetic shift right">;

def PredicateSLT : I64EnumAttrCase<"slt", 0>;
def PredicateSLE : I64EnumAttrCase<"sle", 1>;
def PredicateSGT : I64EnumAttrCase<"sgt", 2>;
def PredicateSGE : I64EnumAttrCase<"sge", 3>;
def PredicateULT : I64EnumAttrCase<"ult", 4>;
def PredicateULE : I64EnumAttrCase<"ule", 5>;
def PredicateUGT : I64EnumAttrCase<"ugt", 6>;
def PredicateUGE : I64EnumAttrCase<"uge", 7>;
let cppNamespace = "mlir::smt" in
def BVCmpPredicate : I64EnumAttr<
    "BVCmpPredicate",
    "smt bit-vector comparison predicate",
    [PredicateSLT, PredicateSLE, PredicateSGT, PredicateSGE,
     PredicateULT, PredicateULE, PredicateUGT, PredicateUGE]>;

def BVCmpOp : SMTBVOp<"cmp", [Pure, SameTypeOperands]> {
  let summary = "compare bit-vectors interpreted as signed or unsigned";
  let description = [{
    This operation compares bit-vector values, interpreting them as signed or
    unsigned values depending on the predicate. The semantics are equivalent to
    the `bvslt`, `bvsle`, `bvsgt`, `bvsge`, `bvult`, `bvule`, `bvugt`, or
    `bvuge` operator defined in the SMT-LIB 2.7 standard depending on the
    specified predicate. More precisely in the
    [theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
    and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
    describing closed quantifier-free formulas over the theory of fixed-size
    bit-vectors.
  }];

  let arguments = (ins BVCmpPredicate:$pred,
                       BitVectorType:$lhs,
                       BitVectorType:$rhs);
  let results = (outs BoolType:$result);

  let assemblyFormat = [{
    $pred $lhs `,` $rhs attr-dict `:` qualified(type($lhs))
  }];
}

def ConcatOp : SMTBVOp<"concat", [
  Pure,
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
]> {
  let summary = "bit-vector concatenation";
  let description = [{
    This operation concatenates bit-vector values with semantics equivalent to
    the `concat` operator defined in the SMT-LIB 2.7 standard. More precisely in
    the [theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
    and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
    describing closed quantifier-free formulas over the theory of fixed-size
    bit-vectors.

    Note that the following equivalences hold:
    * `smt.bv.concat %a, %b : !smt.bv<4>, !smt.bv<4>` is equivalent to
      `(concat a b)` in SMT-LIB
    * `(= (concat #xf #x0) #xf0)`
  }];

  let arguments = (ins BitVectorType:$lhs, BitVectorType:$rhs);
  let results = (outs BitVectorType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type(operands))";
}

def ExtractOp : SMTBVOp<"extract", [Pure]> {
  let summary = "bit-vector extraction";
  let description = [{
    This operation extracts the range of bits starting at the 'lowBit' index
    (inclusive) up to the 'lowBit' + result-width index (exclusive). The
    semantics are equivalent to the `extract` operator defined in the SMT-LIB
    2.7 standard. More precisely in the
    [theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
    and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
    describing closed quantifier-free formulas over the theory of fixed-size
    bit-vectors.

    Note that `smt.bv.extract %bv from 2 : (!smt.bv<32>) -> !smt.bv<16>` is
    equivalent to `((_ extract 17 2) bv)`, i.e., the SMT-LIB operator takes the
    low and high indices where both are inclusive. The following equivalence
    holds: `(= ((_ extract 3 0) #x0f) #xf)`
  }];

  let arguments = (ins I32Attr:$lowBit, BitVectorType:$input);
  let results = (outs BitVectorType:$result);

  let assemblyFormat = [{
    $input `from` $lowBit attr-dict `:` functional-type($input, $result)
  }];

  let hasVerifier = true;
}

def RepeatOp : SMTBVOp<"repeat", [Pure]> {
  let summary = "repeated bit-vector concatenation of one value";
  let description = [{
    This operation is a shorthand for repeated concatenation of the same
    bit-vector value, i.e.,
    ```mlir
    smt.bv.repeat 5 times %a : !smt.bv<4>
    // is the same as
    %0 = smt.bv.repeat 4 times %a : !smt.bv<4>
    smt.bv.concat %a, %0 : !smt.bv<4>, !smt.bv<16>
    // or also 
    %0 = smt.bv.repeat 4 times %a : !smt.bv<4>
    smt.bv.concat %0, %a : !smt.bv<16>, !smt.bv<4>
    ```
    
    The semantics are equivalent to the `repeat` operator defined in the SMT-LIB
    2.7 standard. More precisely in the
    [theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
    and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
    describing closed quantifier-free formulas over the theory of fixed-size
    bit-vectors.
  }];

  let arguments = (ins BitVectorType:$input);
  let results = (outs BitVectorType:$result);

  let hasCustomAssemblyFormat = true;
  let hasVerifier = true;

  let builders = [
    OpBuilder<(ins "unsigned":$count, "mlir::Value":$input)>,
  ];

  let extraClassDeclaration = [{
    /// Get the number of times the input operand is repeated.
    unsigned getCount();
  }];
}

def BV2IntOp : SMTOp<"bv2int", [Pure]> {
  let summary = "Convert an SMT bit-vector to an SMT integer.";
  let description = [{
    Create an integer from the bit-vector argument `input`. If `is_signed` is
    present, the bit-vector is treated as two's complement signed.  Otherwise,
    it is treated as an unsigned integer in the range [0..2^N-1], where N is
    the number of bits in `input`.
  }];
  let arguments = (ins BitVectorType:$input, UnitAttr:$is_signed);
  let results = (outs IntType:$result);
  let assemblyFormat = [{$input (`signed` $is_signed^)? attr-dict `:`
    qualified(type($input))}];
}

#endif // MLIR_DIALECT_SMT_IR_SMTBITVECTOROPS_TD
