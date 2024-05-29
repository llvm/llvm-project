//===- OpenMPOpBase.td - OpenMP dialect shared definitions -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains shared definitions for the OpenMP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_OP_BASE
#define OPENMP_OP_BASE

include "mlir/Dialect/OpenMP/OpenMPAttrDefs.td"
include "mlir/Dialect/OpenMP/OpenMPDialect.td"
include "mlir/Dialect/OpenMP/OpenMPOpsInterfaces.td"
include "mlir/Dialect/OpenMP/OpenMPTypeInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// OpenMP dialect type constraints.
//===----------------------------------------------------------------------===//

class OpenMP_Type<string name, string typeMnemonic> :
      TypeDef<OpenMP_Dialect, name> {
  let mnemonic = typeMnemonic;
}

// Type which can be constraint accepting standard integers and indices.
def IntLikeType : AnyTypeOf<[AnyInteger, Index]>;

def OpenMP_PointerLikeType : TypeAlias<OpenMP_PointerLikeTypeInterface,
	"OpenMP-compatible variable type">;

def OpenMP_MapBoundsType : OpenMP_Type<"MapBounds", "map_bounds_ty"> {
  let summary = "Type for representing omp map clause bounds information";
}

//===----------------------------------------------------------------------===//
// Base classes for OpenMP dialect operations.
//===----------------------------------------------------------------------===//

class OpenMP_Op<string mnemonic, list<Trait> traits = []> :
      Op<OpenMP_Dialect, mnemonic, traits>;

#endif  // OPENMP_OP_BASE
