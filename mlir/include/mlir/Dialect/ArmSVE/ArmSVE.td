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
// ArmSVE type traits
//===----------------------------------------------------------------------===//

def IsScalableVectorTypePred :
    CPred<"$_self.isa<::mlir::arm_sve::ScalableVectorType>()">;

class ScalableVectorOf<list<Type> allowedTypes> :
    ContainerType<AnyTypeOf<allowedTypes>, IsScalableVectorTypePred,
          "$_self.cast<::mlir::arm_sve::ScalableVectorType>().getElementType()",
          "scalable vector">;

class IsScalableVectorOfLengthPred<list<int> allowedLengths> :
  And<[IsScalableVectorTypePred,
       Or<!foreach(allowedlength, allowedLengths, CPred<
          [{$_self.cast<::mlir::arm_sve::ScalableVectorType>().getNumElements() == }]
          # allowedlength>)>]>;

class ScalableVectorOfLength<list<int> allowedLengths> : Type<
  IsScalableVectorOfLengthPred<allowedLengths>,
  " of length " # !interleave(allowedLengths, "/")>;

class ScalableVectorOfLengthAndType<list<int> allowedLengths,
                                    list<Type> allowedTypes> : Type<
  And<[ScalableVectorOf<allowedTypes>.predicate,
       ScalableVectorOfLength<allowedLengths>.predicate]>,
  ScalableVectorOf<allowedTypes>.summary #
  ScalableVectorOfLength<allowedLengths>.summary>;

//===----------------------------------------------------------------------===//
// ArmSVE op definitions
//===----------------------------------------------------------------------===//

class ArmSVE_Op<string mnemonic, list<OpTrait> traits = []> :
  Op<ArmSVE_Dialect, mnemonic, traits> {}

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

#endif // ARMSVE_OPS
