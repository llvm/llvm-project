//===-- ArmSVE.td - ArmSVE dialect operation definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the ArmSVE dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ARMSVE_OPS
#define ARMSVE_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/Arithmetic/IR/ArithmeticBase.td"
include "mlir/Dialect/ArmSVE/ArmSVEOpBase.td"

//===----------------------------------------------------------------------===//
// ArmSVE dialect definition
//===----------------------------------------------------------------------===//

def ArmSVE_Dialect : Dialect {
  let name = "arm_sve";
  let cppNamespace = "::mlir::arm_sve";
  let summary = "Basic dialect to target Arm SVE architectures";
  let description = [{
    This dialect contains the definitions necessary to target Arm SVE scalable
    vector operations, including a scalable vector type and intrinsics for
    some Arm SVE instructions.
  }];
}

//===----------------------------------------------------------------------===//
// ArmSVE type definitions
//===----------------------------------------------------------------------===//

def ArmSVE_ScalableVectorType : DialectType<ArmSVE_Dialect,
    CPred<"$_self.isa<ScalableVectorType>()">,
              "scalable vector type">,
    BuildableType<"$_builder.getType<ScalableVectorType>()"> {
  let description = [{
    `arm_sve.vector` represents vectors that will be processed by a scalable
    vector architecture.
  }];
}

class ArmSVE_Type<string name> : TypeDef<ArmSVE_Dialect, name> { }

def ScalableVectorType : ArmSVE_Type<"ScalableVector"> {
  let mnemonic = "vector";

  let summary = "Scalable vector type";

  let description = [{
    A type representing scalable length SIMD vectors. Unlike fixed-length SIMD
    vectors, whose size is constant and known at compile time, scalable
    vectors' length is constant but determined by the specific hardware at
    run time.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t", "Vector shape">:$shape,
    "Type":$elementType
  );

  let printer = [{
    $_printer << "vector<";
    for (int64_t dim : getShape())
      $_printer << dim << 'x';
    $_printer << getElementType() << '>';
  }];

  let parser = [{
    VectorType vector;
    if ($_parser.parseType(vector))
      return Type();
    return get($_ctxt, vector.getShape(), vector.getElementType());
  }];

  let extraClassDeclaration = [{
    bool hasStaticShape() const {
      return llvm::none_of(getShape(), ShapedType::isDynamic);
    }
    int64_t getNumElements() const {
      assert(hasStaticShape() &&
             "cannot get element count of dynamic shaped type");
      ArrayRef<int64_t> shape = getShape();
      int64_t num = 1;
      for (auto dim : shape)
        num *= dim;
      return num;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Additional LLVM type constraints
//===----------------------------------------------------------------------===//
def LLVMScalableVectorType :
  Type<CPred<"$_self.isa<::mlir::LLVM::LLVMScalableVectorType>()">,
       "LLVM dialect scalable vector type">;

//===----------------------------------------------------------------------===//
// ArmSVE op definitions
//===----------------------------------------------------------------------===//

class ArmSVE_Op<string mnemonic, list<OpTrait> traits = []> :
  Op<ArmSVE_Dialect, mnemonic, traits> {}

class ArmSVE_NonSVEIntrUnaryOverloadedOp<string mnemonic,
                                         list<OpTrait> traits =[]> :
  LLVM_IntrOpBase</*Dialect dialect=*/ArmSVE_Dialect,
                  /*string opName=*/mnemonic,
                  /*string enumName=*/mnemonic,
                  /*list<int> overloadedResults=*/[0],
                  /*list<int> overloadedOperands=*/[], // defined by result overload
                  /*list<OpTrait> traits=*/traits,
                  /*int numResults=*/1>;

class ArmSVE_IntrBinaryOverloadedOp<string mnemonic,
                                    list<OpTrait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/ArmSVE_Dialect,
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"aarch64_sve_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[0],
                  /*list<int> overloadedOperands=*/[], // defined by result overload
                  /*list<OpTrait> traits=*/traits,
                  /*int numResults=*/1>;

class ScalableFOp<string mnemonic, string op_description,
                  list<OpTrait> traits = []> :
  ArmSVE_Op<mnemonic, !listconcat(traits,
                       [AllTypesMatch<["src1", "src2", "dst"]>])> {
  let summary = op_description # " for scalable vectors of floats";
  let description = [{
    The `arm_sve.}] # mnemonic # [{` operations takes two scalable vectors and
    returns one scalable vector with the result of the }] # op_description # [{.
  }];
  let arguments = (ins
          ScalableVectorOf<[AnyFloat]>:$src1,
          ScalableVectorOf<[AnyFloat]>:$src2
  );
  let results = (outs ScalableVectorOf<[AnyFloat]>:$dst);
  let assemblyFormat =
    "$src1 `,` $src2 attr-dict `:` type($src1)";
}

class ScalableIOp<string mnemonic, string op_description,
                  list<OpTrait> traits = []> :
  ArmSVE_Op<mnemonic, !listconcat(traits,
                       [AllTypesMatch<["src1", "src2", "dst"]>])> {
  let summary = op_description # " for scalable vectors of integers";
  let description = [{
    The `arm_sve.}] # mnemonic # [{` operation takes two scalable vectors and
    returns one scalable vector with the result of the }] # op_description # [{.
  }];
  let arguments = (ins
          ScalableVectorOf<[I8, I16, I32, I64]>:$src1,
          ScalableVectorOf<[I8, I16, I32, I64]>:$src2
  );
  let results = (outs ScalableVectorOf<[I8, I16, I32, I64]>:$dst);
  let assemblyFormat =
    "$src1 `,` $src2 attr-dict `:` type($src1)";
}

class ScalableMaskedFOp<string mnemonic, string op_description,
                        list<OpTrait> traits = []> :
  ArmSVE_Op<mnemonic, !listconcat(traits,
                       [AllTypesMatch<["src1", "src2", "res"]>,
                        TypesMatchWith<
                          "mask has i1 element type and same shape as operands",
                          "src1", "mask", "getI1SameShape($_self)">])> {
  let summary = "masked " # op_description # " for scalable vectors of floats";
  let description = [{
    The `arm_sve.}] # mnemonic # [{` operation takes one scalable vector mask
    and two scalable vector operands, and perform floating point }] #
    op_description # [{ on active lanes. Inactive lanes will keep the value of
    the first operand.}];
  let arguments = (ins
          ScalableVectorOf<[I1]>:$mask,
          ScalableVectorOf<[AnyFloat]>:$src1,
          ScalableVectorOf<[AnyFloat]>:$src2
  );
  let results = (outs ScalableVectorOf<[AnyFloat]>:$res);
  let assemblyFormat =
    "$mask `,` $src1 `,` $src2 attr-dict `:` type($mask) `,` type($res)";
}

class ScalableMaskedIOp<string mnemonic, string op_description,
                        list<OpTrait> traits = []> :
  ArmSVE_Op<mnemonic, !listconcat(traits,
                       [AllTypesMatch<["src1", "src2", "res"]>,
                        TypesMatchWith<
                          "mask has i1 element type and same shape as operands",
                          "src1", "mask", "getI1SameShape($_self)">])> {
  let summary = "masked " # op_description # " for scalable vectors of integers";
  let description = [{
    The `arm_sve.}] # mnemonic # [{` operation takes one scalable vector mask
    and two scalable vector operands, and perform integer }] #
    op_description # [{ on active lanes. Inactive lanes will keep the value of
    the first operand.}];
  let arguments = (ins
          ScalableVectorOf<[I1]>:$mask,
          ScalableVectorOf<[I8, I16, I32, I64]>:$src1,
          ScalableVectorOf<[I8, I16, I32, I64]>:$src2
  );
  let results = (outs ScalableVectorOf<[I8, I16, I32, I64]>:$res);
  let assemblyFormat =
    "$mask `,` $src1 `,` $src2 attr-dict `:` type($mask) `,` type($res)";
}

def SdotOp : ArmSVE_Op<"sdot",
               [NoSideEffect,
               AllTypesMatch<["src1", "src2"]>,
               AllTypesMatch<["acc", "dst"]>,
             ]> {
  let summary = "Vector-vector dot product and accumulate op";
  let description = [{
    SDOT: Signed integer addition of dot product.

    This function maps to the SDOT instruction, and it takes signless integer
    operands that the operation interprets as signed. It partitions the second
    and third vector inputs into groups of four elements. They calculate the dot
    product of each group (without loss of precision) and then add each result
    to the overlapping element of the first vector input.

    Source:
    https://developer.arm.com/documentation/100987/0000
  }];
  // Supports either:
  //   (vector<16xi8>, vector<16xi8>) -> (vector<4xi32>)
  //   (vector<8xi16>. vector<8xi16>) -> (vector<2xi64>)
  let arguments = (ins
          ScalableVectorOfLengthAndType<[4, 2], [I32, I64]>:$acc,
          ScalableVectorOfLengthAndType<[16, 8], [I8, I16]>:$src1,
          ScalableVectorOfLengthAndType<[16, 8], [I8, I16]>:$src2
  );
  let results = (outs ScalableVectorOfLengthAndType<[4, 2], [I32, I64]>:$dst);
  let assemblyFormat =
    "$acc `,` $src1 `,` $src2 attr-dict `:` type($src1) `to` type($dst)";
}

def SmmlaOp : ArmSVE_Op<"smmla",
                [NoSideEffect,
                AllTypesMatch<["src1", "src2"]>,
                AllTypesMatch<["acc", "dst"]>,
              ]> {
  let summary = "Matrix-matrix multiply and accumulate op";
  let description = [{
    SMMLA: Signed integer matrix multiply-accumulate.

    This function maps to the SMMLA instruction, and it takes signless integer
    operands that the operation interprets as signed. It partitions the inputs
    into 128-bit quadwords, with the first input containing a row-by-row 2×2
    matrix of 32-bit integers, the second input containing a row-by-row 2×8
    matrix of 8-bit integers, and the third input containing a column-by-column
    8×2 matrix of 8-bit integers. For each quadword, they multiply the second
    input matrix by the third input matrix using natural arithmetic and then add
    the result to the first input using modular arithmetic.

    Source:
    https://developer.arm.com/documentation/100987/0000
  }];
  // Supports (vector<16xi8>, vector<16xi8>) -> (vector<4xi32>)
  let arguments = (ins
          ScalableVectorOfLengthAndType<[4], [I32]>:$acc,
          ScalableVectorOfLengthAndType<[16], [I8]>:$src1,
          ScalableVectorOfLengthAndType<[16], [I8]>:$src2
  );
  let results = (outs ScalableVectorOfLengthAndType<[4], [I32]>:$dst);
  let assemblyFormat =
    "$acc `,` $src1 `,` $src2 attr-dict `:` type($src1) `to` type($dst)";
}

def UdotOp : ArmSVE_Op<"udot",
               [NoSideEffect,
               AllTypesMatch<["src1", "src2"]>,
               AllTypesMatch<["acc", "dst"]>,
             ]> {
  let summary = "Vector-vector dot product and accumulate op";
  let description = [{
    UDOT: Unsigned integer addition of dot product.

    This function maps to the UDOT instruction, and it takes signless integer
    operands that the operation interprets as unsigned. It partitions the second
    and third vector inputs into groups of four elements. They calculate the dot
    product of each group (without loss of precision) and then add each result
    to the overlapping element of the first vector input.

    Source:
    https://developer.arm.com/documentation/100987/0000
  }];
  // Supports either:
  //   (vector<16xi8>, vector<16xi8>) -> (vector<4xi32>)
  //   (vector<8xi16>. vector<8xi16>) -> (vector<2xi64>)
  let arguments = (ins
          ScalableVectorOfLengthAndType<[4, 2], [I32, I64]>:$acc,
          ScalableVectorOfLengthAndType<[16, 8], [I8, I16]>:$src1,
          ScalableVectorOfLengthAndType<[16, 8], [I8, I16]>:$src2
  );
  let results = (outs ScalableVectorOfLengthAndType<[4, 2], [I32, I64]>:$dst);
  let assemblyFormat =
    "$acc `,` $src1 `,` $src2 attr-dict `:` type($src1) `to` type($dst)";
}

def UmmlaOp : ArmSVE_Op<"ummla",
                [NoSideEffect,
                AllTypesMatch<["src1", "src2"]>,
                AllTypesMatch<["acc", "dst"]>,
              ]> {
  let summary = "Matrix-matrix multiply and accumulate op";
  let description = [{
    UMMLA: Unsigned integer matrix multiply-accumulate.

    This function maps to the UMMLA instruction, and it takes signless integer
    operands that the operation interprets as unsigned. It partitions the inputs
    into 128-bit quadwords, with the first input containing a row-by-row 2×2
    matrix of 32-bit integers, the second input containing a row-by-row 2×8
    matrix of 8-bit integers, and the third input containing a column-by-column
    8×2 matrix of 8-bit integers. For each quadword, they multiply the second
    input matrix by the third input matrix using natural arithmetic and then add
    the result to the first input using modular arithmetic.

    Source:
    https://developer.arm.com/documentation/100987/0000
  }];
  // Supports (vector<16xi8>, vector<16xi8>) -> (vector<4xi32>)
  let arguments = (ins
          ScalableVectorOfLengthAndType<[4], [I32]>:$acc,
          ScalableVectorOfLengthAndType<[16], [I8]>:$src1,
          ScalableVectorOfLengthAndType<[16], [I8]>:$src2
  );
  let results = (outs ScalableVectorOfLengthAndType<[4], [I32]>:$dst);
  let assemblyFormat =
    "$acc `,` $src1 `,` $src2 attr-dict `:` type($src1) `to` type($dst)";
}

def VectorScaleOp : ArmSVE_Op<"vector_scale",
                 [NoSideEffect]> {
  let summary = "Load vector scale size";
  let description = [{
    The vector_scale op returns the scale of the scalable vectors, a positive
    integer value that is constant at runtime but unknown at compile time.
    The scale of the vector indicates the multiplicity of the vectors and
    vector operations. I.e.: an !arm_sve.vector<4xi32> is equivalent to
    vector_scale consecutive vector<4xi32>; and an operation on an
    !arm_sve.vector<4xi32> is equivalent to performing that operation vector_scale
    times, once on each <4xi32> segment of the scalable vector. The vector_scale
    op can be used to calculate the step in vector-length agnostic (VLA) loops.
  }];
  let results = (outs Index:$res);
  let assemblyFormat =
    "attr-dict `:` type($res)";
}

def ScalableLoadOp : ArmSVE_Op<"load">,
    Arguments<(ins Arg<AnyMemRef, "", [MemRead]>:$base, Index:$index)>,
    Results<(outs ScalableVectorOf<[AnyType]>:$result)> {
  let summary = "Load scalable vector from memory";
  let description = [{
    Load a slice of memory into a scalable vector.
  }];
  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return base().getType().cast<MemRefType>();
    }
  }];
  let assemblyFormat = "$base `[` $index `]` attr-dict `:` "
    "type($result) `from` type($base)";
}

def ScalableStoreOp : ArmSVE_Op<"store">,
    Arguments<(ins Arg<AnyMemRef, "", [MemWrite]>:$base, Index:$index,
               ScalableVectorOf<[AnyType]>:$value)> {
  let summary = "Store scalable vector into memory";
  let description = [{
    Store a scalable vector on a slice of memory.
  }];
  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return base().getType().cast<MemRefType>();
    }
  }];
  let assemblyFormat = "$value `,` $base `[` $index `]` attr-dict `:` "
    "type($value) `to` type($base)";
}

def ScalableAddIOp : ScalableIOp<"addi", "addition", [Commutative]>;

def ScalableAddFOp : ScalableFOp<"addf", "addition", [Commutative]>;

def ScalableSubIOp : ScalableIOp<"subi", "subtraction">;

def ScalableSubFOp : ScalableFOp<"subf", "subtraction">;

def ScalableMulIOp : ScalableIOp<"muli", "multiplication", [Commutative]>;

def ScalableMulFOp : ScalableFOp<"mulf", "multiplication", [Commutative]>;

def ScalableSDivIOp : ScalableIOp<"divi_signed", "signed division">;

def ScalableUDivIOp : ScalableIOp<"divi_unsigned", "unsigned division">;

def ScalableDivFOp : ScalableFOp<"divf", "division">;

def ScalableMaskedAddIOp : ScalableMaskedIOp<"masked.addi", "addition",
                                             [Commutative]>;

def ScalableMaskedAddFOp : ScalableMaskedFOp<"masked.addf", "addition",
                            [Commutative]>;

def ScalableMaskedSubIOp : ScalableMaskedIOp<"masked.subi", "subtraction">;

def ScalableMaskedSubFOp : ScalableMaskedFOp<"masked.subf", "subtraction">;

def ScalableMaskedMulIOp : ScalableMaskedIOp<"masked.muli", "multiplication",
                            [Commutative]>;

def ScalableMaskedMulFOp : ScalableMaskedFOp<"masked.mulf", "multiplication",
                            [Commutative]>;

def ScalableMaskedSDivIOp : ScalableMaskedIOp<"masked.divi_signed",
                                              "signed division">;

def ScalableMaskedUDivIOp : ScalableMaskedIOp<"masked.divi_unsigned",
                                              "unsigned division">;

def ScalableMaskedDivFOp : ScalableMaskedFOp<"masked.divf", "division">;

//===----------------------------------------------------------------------===//
// ScalableCmpFOp
//===----------------------------------------------------------------------===//

def ScalableCmpFOp : ArmSVE_Op<"cmpf", [NoSideEffect, SameTypeOperands,
    TypesMatchWith<"result type has i1 element type and same shape as operands",
    "lhs", "result", "getI1SameShape($_self)">]> {
  let summary = "floating-point comparison operation for scalable vectors";
  let description = [{
    The `arm_sve.cmpf` operation compares two scalable vectors of floating point
    elements according to the float comparison rules and the predicate specified
    by the respective attribute. The predicate defines the type of comparison:
    (un)orderedness, (in)equality and signed less/greater than (or equal to) as
    well as predicates that are always true or false. The result is a scalable
    vector of i1 elements. Unlike `arm_sve.cmpi`, the operands are always
    treated as signed. The u prefix indicates *unordered* comparison, not
    unsigned comparison, so "une" means unordered not equal. For the sake of
    readability by humans, custom assembly form for the operation uses a
    string-typed attribute for the predicate.  The value of this attribute
    corresponds to lower-cased name of the predicate constant, e.g., "one" means
    "ordered not equal".  The string representation of the attribute is merely a
    syntactic sugar and is converted to an integer attribute by the parser.

    Example:

    ```mlir
    %r = arm_sve.cmpf oeq, %0, %1 : !arm_sve.vector<4xf32>
    ```
  }];
  let arguments = (ins
    Arith_CmpFPredicateAttr:$predicate,
    ScalableVectorOf<[AnyFloat]>:$lhs,
    ScalableVectorOf<[AnyFloat]>:$rhs // TODO: This should support a simple scalar
  );
  let results = (outs ScalableVectorOf<[I1]>:$result);

  let builders = [
    OpBuilder<(ins "arith::CmpFPredicate":$predicate, "Value":$lhs,
                  "Value":$rhs), [{
      buildScalableCmpFOp($_builder, $_state, predicate, lhs, rhs);
    }]>];

  let extraClassDeclaration = [{
    static StringRef getPredicateAttrName() { return "predicate"; }
    static arith::CmpFPredicate getPredicateByName(StringRef name);

    arith::CmpFPredicate getPredicate() {
      return (arith::CmpFPredicate) (*this)->getAttrOfType<IntegerAttr>(
          getPredicateAttrName()).getInt();
    }
  }];

  let verifier = [{ return success(); }];

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
}

//===----------------------------------------------------------------------===//
// ScalableCmpIOp
//===----------------------------------------------------------------------===//

def ScalableCmpIOp : ArmSVE_Op<"cmpi", [NoSideEffect, SameTypeOperands,
    TypesMatchWith<"result type has i1 element type and same shape as operands",
    "lhs", "result", "getI1SameShape($_self)">]> {
  let summary = "integer comparison operation for scalable vectors";
  let description = [{
    The `arm_sve.cmpi` operation compares two scalable vectors of integer
    elements according to the predicate specified by the respective attribute.

    The predicate defines the type of comparison:

    -   equal (mnemonic: `"eq"`; integer value: `0`)
    -   not equal (mnemonic: `"ne"`; integer value: `1`)
    -   signed less than (mnemonic: `"slt"`; integer value: `2`)
    -   signed less than or equal (mnemonic: `"sle"`; integer value: `3`)
    -   signed greater than (mnemonic: `"sgt"`; integer value: `4`)
    -   signed greater than or equal (mnemonic: `"sge"`; integer value: `5`)
    -   unsigned less than (mnemonic: `"ult"`; integer value: `6`)
    -   unsigned less than or equal (mnemonic: `"ule"`; integer value: `7`)
    -   unsigned greater than (mnemonic: `"ugt"`; integer value: `8`)
    -   unsigned greater than or equal (mnemonic: `"uge"`; integer value: `9`)

    Example:

    ```mlir
    %r = arm_sve.cmpi uge, %0, %1 : !arm_sve.vector<4xi32>
    ```
  }];

  let arguments = (ins
      Arith_CmpIPredicateAttr:$predicate,
      ScalableVectorOf<[I8, I16, I32, I64]>:$lhs,
      ScalableVectorOf<[I8, I16, I32, I64]>:$rhs
  );
  let results = (outs ScalableVectorOf<[I1]>:$result);

  let builders = [
    OpBuilder<(ins "arith::CmpIPredicate":$predicate, "Value":$lhs,
                 "Value":$rhs), [{
      buildScalableCmpIOp($_builder, $_state, predicate, lhs, rhs);
    }]>];

  let extraClassDeclaration = [{
    static StringRef getPredicateAttrName() { return "predicate"; }
    static arith::CmpIPredicate getPredicateByName(StringRef name);

    arith::CmpIPredicate getPredicate() {
      return (arith::CmpIPredicate) (*this)->getAttrOfType<IntegerAttr>(
          getPredicateAttrName()).getInt();
    }
  }];

  let verifier = [{ return success(); }];

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
}

def UmmlaIntrOp :
  ArmSVE_IntrBinaryOverloadedOp<"ummla">,
  Arguments<(ins LLVMScalableVectorType, LLVMScalableVectorType,
             LLVMScalableVectorType)>;

def SmmlaIntrOp :
  ArmSVE_IntrBinaryOverloadedOp<"smmla">,
  Arguments<(ins LLVMScalableVectorType, LLVMScalableVectorType,
             LLVMScalableVectorType)>;

def SdotIntrOp :
  ArmSVE_IntrBinaryOverloadedOp<"sdot">,
  Arguments<(ins LLVMScalableVectorType, LLVMScalableVectorType,
             LLVMScalableVectorType)>;

def UdotIntrOp :
  ArmSVE_IntrBinaryOverloadedOp<"udot">,
  Arguments<(ins LLVMScalableVectorType, LLVMScalableVectorType,
             LLVMScalableVectorType)>;

def ScalableMaskedAddIIntrOp :
  ArmSVE_IntrBinaryOverloadedOp<"add">,
  Arguments<(ins LLVMScalableVectorType, LLVMScalableVectorType,
             LLVMScalableVectorType)>;

def ScalableMaskedAddFIntrOp :
  ArmSVE_IntrBinaryOverloadedOp<"fadd">,
  Arguments<(ins LLVMScalableVectorType, LLVMScalableVectorType,
             LLVMScalableVectorType)>;

def ScalableMaskedMulIIntrOp :
  ArmSVE_IntrBinaryOverloadedOp<"mul">,
  Arguments<(ins LLVMScalableVectorType, LLVMScalableVectorType,
             LLVMScalableVectorType)>;

def ScalableMaskedMulFIntrOp :
  ArmSVE_IntrBinaryOverloadedOp<"fmul">,
  Arguments<(ins LLVMScalableVectorType, LLVMScalableVectorType,
             LLVMScalableVectorType)>;

def ScalableMaskedSubIIntrOp :
  ArmSVE_IntrBinaryOverloadedOp<"sub">,
  Arguments<(ins LLVMScalableVectorType, LLVMScalableVectorType,
             LLVMScalableVectorType)>;

def ScalableMaskedSubFIntrOp :
  ArmSVE_IntrBinaryOverloadedOp<"fsub">,
  Arguments<(ins LLVMScalableVectorType, LLVMScalableVectorType,
             LLVMScalableVectorType)>;

def ScalableMaskedSDivIIntrOp :
  ArmSVE_IntrBinaryOverloadedOp<"sdiv">,
  Arguments<(ins LLVMScalableVectorType, LLVMScalableVectorType,
             LLVMScalableVectorType)>;

def ScalableMaskedUDivIIntrOp :
  ArmSVE_IntrBinaryOverloadedOp<"udiv">,
  Arguments<(ins LLVMScalableVectorType, LLVMScalableVectorType,
             LLVMScalableVectorType)>;

def ScalableMaskedDivFIntrOp :
  ArmSVE_IntrBinaryOverloadedOp<"fdiv">,
  Arguments<(ins LLVMScalableVectorType, LLVMScalableVectorType,
             LLVMScalableVectorType)>;

def VectorScaleIntrOp:
  ArmSVE_NonSVEIntrUnaryOverloadedOp<"vscale">;

#endif // ARMSVE_OPS
