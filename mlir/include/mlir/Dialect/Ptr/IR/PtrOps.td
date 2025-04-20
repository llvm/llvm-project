//===- PtrOps.td - Pointer dialect ops ---------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PTR_OPS
#define PTR_OPS

include "mlir/Dialect/Ptr/IR/PtrDialect.td"
include "mlir/Dialect/Ptr/IR/PtrAttrDefs.td"
include "mlir/Dialect/Ptr/IR/PtrEnums.td"
include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/OpAsmInterface.td"

//===----------------------------------------------------------------------===//
// PtrAddOp
//===----------------------------------------------------------------------===//

def Ptr_PtrAddOp : Pointer_Op<"ptradd", [
    Pure, AllTypesMatch<["base", "result"]>,
    DeclareOpInterfaceMethods<ViewLikeOpInterface>
  ]> {
  let summary = "Pointer add operation";
  let description = [{
    The `ptradd` operation adds an integer offset to a pointer to produce a new
    pointer. The input and output pointer types are always the same.

    Example:

    ```mlir
    %x_off  = ptr.ptradd %x, %off : !ptr.ptr<0>, i32
    %x_off0 = ptr.ptradd nusw %x, %off : !ptr.ptr<0>, i32
    ```
  }];

  let arguments = (ins
    Ptr_PtrType:$base,
    AnySignlessIntegerOrIndex:$offset,
    DefaultValuedAttr<Ptr_PtrAddFlags,
        "::mlir::ptr::PtrAddFlags::none">:$flags);
  let results = (outs Ptr_PtrType:$result);
  let assemblyFormat = [{
    ($flags^)? $base `,` $offset attr-dict `:` type($base) `,` type($offset)
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// TypeOffsetOp
//===----------------------------------------------------------------------===//

def Ptr_TypeOffsetOp : Pointer_Op<"type_offset", [ConstantLike, Pure]> {
  let summary = "Type offset operation";
  let description = [{
    The `type_offset` operation produces an int or index-typed SSA value
    equal to a target-specific constant representing the offset of a single
    element of the given type.

    Example:

    ```mlir
    %0 = ptr.type_offset f32 : index
    %1 = ptr.type_offset memref<12 x f64> : i32
    ```
  }];

  let arguments = (ins TypeAttr:$element_type);
  let results = (outs AnySignlessIntegerOrIndex:$result);
  let builders = [
    OpBuilder<(ins "TypeAttr":$element_type)>
  ];
  let assemblyFormat = [{
    $element_type attr-dict `:` type($result)
  }];
  let extraClassDeclaration = [{
    /// Returns the type offset according to `maybeLayout`. If `maybeLayout` is
    /// `nullopt` the nearest layout the op will be used for the computation.
    llvm::TypeSize getTypeSize(std::optional<DataLayout> layout = std::nullopt);
  }];
  let hasFolder = 1;
}

#endif // PTR_OPS
