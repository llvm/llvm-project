//===- MaskingInterfaces.td - Masking Interfaces Decls === -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for vector masking related interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_INTERFACES_MASKINGINTERFACES
#define MLIR_DIALECT_VECTOR_INTERFACES_MASKINGINTERFACES

include "mlir/IR/OpBase.td"

def MaskableOpInterface : OpInterface<"MaskableOpInterface"> {
  let description = [{
    The 'MaskableOpInterface' define an operation that can be masked using the
    `vector.mask` operation and provides information about its masking
    constraints and semantics.
  }];
  let cppNamespace = "::mlir::vector";
  let methods = [
    InterfaceMethod<
    /*desc=*/"Returns true if the operation may have a passthru argument when"
             " masked.",
    /*retTy=*/"bool",
    /*methodName=*/"supportsPassthru",
    /*args=*/(ins),
    /*methodBody=*/"",
    /*defaultImplementation=*/[{
      return false;
    }]>,
    InterfaceMethod<
    /*desc=*/"Returns the mask type expected by this operation. It requires the"
             " operation to be vectorized.",
    /*retTy=*/"mlir::VectorType",
    /*methodName=*/"getExpectedMaskType",
    /*args=*/(ins),
    /*methodBody=*/"",
    /*defaultImplementation=*/[{
    // Default implementation is only aimed for operations that implement the
    // `getVectorType()` method.
      return $_op.getVectorType().cloneWith(
        /*shape=*/llvm::None, IntegerType::get($_op.getContext(), /*width=*/1));
    }]>,
  ];
}

#endif // MLIR_DIALECT_VECTOR_INTERFACES_MASKINGINTERFACES
