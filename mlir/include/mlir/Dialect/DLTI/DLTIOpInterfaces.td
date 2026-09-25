//===- DLTIOpInterfaces.td - DLTI operation interfaces -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_DLTI_DLTIOPINTERFACES
#define MLIR_DIALECT_DLTI_DLTIOPINTERFACES

include "mlir/Interfaces/DataLayoutInterfaces.td"

def DLTIQueryOpInterface : OpInterface<"DLTIQueryOpInterface"> {
  let cppNamespace = "::mlir";

  let description = [{
    Operation interface for scoped DLTI key-value associations.

    Queries and updates operate on one key at a time. The interface does not
    prescribe how an operation stores values. For example, an operation with
    an inherent `architecture = "sm_90"` attribute could return that string
    attribute for the key `"architecture"`, without having a `dlti` attribute
    or a `#dlti.map`.

    The default implementations use the operation's inherent `dlti` attribute.
    An update preserves other entries in a `#dlti.map`; setting a null value
    removes an entry. Updating a non-map `dlti` attribute or using a null or
    empty string key fails.
  }];

  let methods = [
    InterfaceMethod<
      /*description=*/"Returns the DLTI value associated with the key.",
      /*retTy=*/"::mlir::FailureOr<::mlir::Attribute>",
      /*methodName=*/"queryDlti",
      /*args=*/(ins "::mlir::DataLayoutEntryKey":$key),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::dlti::detail::queryDlti($_op.getOperation(), key);
      }]
    >,
    InterfaceMethod<
      /*description=*/"Creates, replaces, or removes a DLTI key-value "
                      "association.",
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"setDlti",
      /*args=*/(ins "::mlir::DataLayoutEntryKey":$key,
                    "::mlir::Attribute":$value),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::dlti::detail::setDlti($_op.getOperation(), key, value);
      }]
    >
  ];
}

#endif // MLIR_DIALECT_DLTI_DLTIOPINTERFACES
