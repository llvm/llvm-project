//===-- ArmSVEOpBase.td - Base op definitions for ArmSVE ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the base operation definition file for ArmSVE scalable vector types.
//
//===----------------------------------------------------------------------===//

#ifndef ARMSVE_OP_BASE
#define ARMSVE_OP_BASE

//===----------------------------------------------------------------------===//
// ArmSVE scalable vector type constraints
//===----------------------------------------------------------------------===//

def IsScalableVectorTypePred :
    CPred<"$_self.isa<::mlir::arm_sve::ScalableVectorType>()">;

class ScalableVectorOf<list<Type> allowedTypes> :
    ContainerType<AnyTypeOf<allowedTypes>, IsScalableVectorTypePred,
          "$_self.cast<::mlir::arm_sve::ScalableVectorType>().getElementType()",
          "scalable vector">;

// Whether the number of elements of a scalable vector is from the given
// `allowedLengths` list
class IsScalableVectorOfLengthPred<list<int> allowedLengths> :
  And<[IsScalableVectorTypePred,
       Or<!foreach(allowedlength, allowedLengths, CPred<
          [{$_self.cast<::mlir::arm_sve::ScalableVectorType>().getNumElements() == }]
          # allowedlength>)>]>;

// Any scalable vector where the number of elements is from the given
// `allowedLengths` list
class ScalableVectorOfLength<list<int> allowedLengths> : Type<
  IsScalableVectorOfLengthPred<allowedLengths>,
  " of length " # !interleave(allowedLengths, "/"),
  "::mlir::arm_sve::ScalableVectorType">;

// Any scalable vector where the number of elements is from the given
// `allowedLengths` list and the type is from the given `allowedTypes` list
class ScalableVectorOfLengthAndType<list<int> allowedLengths,
                                    list<Type> allowedTypes> : Type<
  And<[ScalableVectorOf<allowedTypes>.predicate,
       ScalableVectorOfLength<allowedLengths>.predicate]>,
  ScalableVectorOf<allowedTypes>.summary #
  ScalableVectorOfLength<allowedLengths>.summary,
  "::mlir::arm_sve::ScalableVectorType">;

#endif // ARMSVE_OP_BASE
