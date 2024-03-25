//===- AffineMemoryOpInterfaces.td -------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces for affine memory ops.
//
//===----------------------------------------------------------------------===//

#ifndef AFFINEMEMORYOPINTERFACES
#define AFFINEMEMORYOPINTERFACES

include "mlir/IR/OpBase.td"

def AffineReadOpInterface : OpInterface<"AffineReadOpInterface"> {
  let description = [{
      Interface to query characteristics of read-like ops with affine
      restrictions.
  }];
  let cppNamespace = "::mlir::affine";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the memref operand to read from.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getMemRef",
      /*args=*/(ins),
      /*methodBody*/[{}],
      /*defaultImplementation=*/ [{
        return $_op.getOperand($_op.getMemRefOperandIndex());
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the type of the memref operand.",
      /*retTy=*/"::mlir::MemRefType",
      /*methodName=*/"getMemRefType",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return ::llvm::cast<::mlir::MemRefType>($_op.getMemRef().getType());
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns affine map operands.",
      /*retTy=*/"::mlir::Operation::operand_range",
      /*methodName=*/"getMapOperands",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return llvm::drop_begin($_op.getOperands(), 1);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the affine map used to index the memref for this operation.
      }],
      /*retTy=*/"::mlir::AffineMap",
      /*methodName=*/"getAffineMap",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getAffineMapAttr().getValue();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the value read by this operation.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getValue",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op;
      }]
    >,
  ];
}

def AffineWriteOpInterface : OpInterface<"AffineWriteOpInterface"> {
  let description = [{
      Interface to query characteristics of write-like ops with affine
      restrictions.
  }];
  let cppNamespace = "::mlir::affine";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the memref operand to write to.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getMemRef",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getOperand($_op.getMemRefOperandIndex());
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the type of the memref operand.",
      /*retTy=*/"::mlir::MemRefType",
      /*methodName=*/"getMemRefType",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return ::llvm::cast<::mlir::MemRefType>($_op.getMemRef().getType());
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns affine map operands.",
      /*retTy=*/"::mlir::Operation::operand_range",
      /*methodName=*/"getMapOperands",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return llvm::drop_begin($_op.getOperands(), 2);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the affine map used to index the memref for this operation.
      }],
      /*retTy=*/"::mlir::AffineMap",
      /*methodName=*/"getAffineMap",
      /*args=*/(ins),
      /*methodName=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getAffineMapAttr().getValue();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the value to store.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getValueToStore",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getOperand($_op.getStoredValOperandIndex());
      }]
    >,
  ];
}

def AffineMapAccessInterface : OpInterface<"AffineMapAccessInterface"> {
  let description = [{
      Interface to query the AffineMap used to dereference and access a given
      memref. Implementers of this interface must operate on at least one
      memref operand.  The memref argument given to this interface much match
      one of those memref operands.
  }];
  let cppNamespace = "::mlir::affine";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the AffineMapAttr associated with 'memref'.",
      /*retTy=*/"::mlir::NamedAttribute",
      /*methodName=*/"getAffineMapAttrForMemRef",
      /*args=*/(ins "::mlir::Value":$memref),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        assert(memref == $_op.getMemRef() &&
               "Expected memref argument to match memref operand");
        return {::mlir::StringAttr::get(
                    $_op.getContext(), $_op.getMapAttrStrName()),
                    $_op.getAffineMapAttr()};
      }]
    >,
  ];
}

#endif // AFFINEMEMORYOPINTERFACES
