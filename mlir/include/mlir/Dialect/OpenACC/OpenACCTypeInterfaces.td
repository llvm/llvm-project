//===-- OpenACCTypeInterfaces.td - OpenACC type interfaces ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPENACC_TYPE_INTERFACES
#define OPENACC_TYPE_INTERFACES

include "mlir/IR/OpBase.td"

def OpenACC_PointerLikeTypeInterface : TypeInterface<"PointerLikeType"> {
  let cppNamespace = "::mlir::acc";

  let description = [{
    An interface for pointer-like types that point to an OpenACC var.
  }];

  // By convention, any of the pointer types associated with this interface
  // will need to provide getElementType.
  let methods = [
    InterfaceMethod<
      /*description=*/[{
        Returns the pointee type or null if the pointer has no pointee type
      }],
      /*retTy=*/"::mlir::Type",
      /*methodName=*/"getElementType"
    >,
  ];
}

def OpenACC_MappableTypeInterface : TypeInterface<"MappableType"> {
  let cppNamespace = "::mlir::acc";

  let description = [{
    An interface to capture type-based semantics for mapping in a manner that
    makes it convertible to size-based semantics.
  }];

  let methods = [
    InterfaceMethod<
      /*description=*/[{
        Returns the pointer to the `var` if recoverable (such as in cases
        where the current operation is a load from a memory slot).
      }],
      /*retTy=*/"::mlir::TypedValue<::mlir::acc::PointerLikeType>",
      /*methodName=*/"getVarPtr",
      /*args=*/(ins "::mlir::Value":$var),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if (auto ptr = mlir::dyn_cast<mlir::TypedValue<mlir::acc::PointerLikeType>>(
              var))
          return ptr;
        return {};
      }]
    >,
    InterfaceMethod<
      /*description=*/[{
        Returns the size in bytes when computable. If this is an array-like
        type, avoiding passing `accBounds` ensures a computation of the size
        of whole type.
      }],
      /*retTy=*/"::std::optional<::llvm::TypeSize>",
      /*methodName=*/"getSizeInBytes",
      /*args=*/(ins "::mlir::Value":$var,
                    "::mlir::ValueRange":$accBounds,
                    "const ::mlir::DataLayout &":$dataLayout),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        // Bounds operations are typically created for array types. In the
        // generic implementation, it is not straightforward to distinguish
        // between array types and ensure the size and offset take into account
        // just the slice requested. Thus return not-computable for now.
        if (!accBounds.empty())
          return {};
        return {dataLayout.getTypeSize($_type)};
      }]
    >,
    InterfaceMethod<
      /*description=*/[{
        Returns the offset in bytes when computable.
      }],
      /*retTy=*/"::std::optional<::int64_t>",
      /*methodName=*/"getOffsetInBytes",
      /*args=*/(ins "::mlir::Value":$var,
                    "::mlir::ValueRange":$accBounds,
                    "const ::mlir::DataLayout &":$dataLayout),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        // Bounds operations are typically created for array types. In the
        // generic implementation, it is not straightforward to distinguish
        // between array types and ensure the size and offset take into account
        // just the slice requested. Thus return not-computable for now.
        if (!accBounds.empty())
          return {};

        // If the type size is computable, it means it is trivial. Assume
        // offset of 0.
        if (::mlir::cast<::mlir::acc::MappableType>($_type).getSizeInBytes(
              var, accBounds, dataLayout).has_value()) {
          return {0};
        }

        return {};
      }]
    >,
      InterfaceMethod<
      /*description=*/[{
        Returns explicit `acc.bounds` operations that envelop the whole
        data structure. These operations are inserted using the provided builder
        at the location set before calling this API.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::Value>",
      /*methodName=*/"generateAccBounds",
      /*args=*/(ins "::mlir::Value":$var,
                    "::mlir::OpBuilder &":$builder),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return {};
      }]
    >,
  ];
}

#endif // OPENACC_TYPE_INTERFACES
