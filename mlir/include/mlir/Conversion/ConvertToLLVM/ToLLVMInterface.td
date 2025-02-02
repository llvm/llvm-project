
//===- ToLLVMInterface.td - Conversion to LLVM interfaces -----*- tablegen -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines interfaces for managing transformations, including populating
// pattern rewrites.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_CONVERTTOLLVM_TOLLVMINTERFACE_TD
#define MLIR_CONVERSION_CONVERTTOLLVM_TOLLVMINTERFACE_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Attribute interface
//===----------------------------------------------------------------------===//

def ConvertToLLVMAttrInterface :
    AttrInterface<"ConvertToLLVMAttrInterface"> {
  let description = [{
    The `ConvertToLLVMAttrInterface` attribute interfaces allows using
    attributes to configure the convert to LLVM infrastructure, this includes:
     - The conversion target.
     - The LLVM type converter.
     - The pattern set.

    This interface permits fined grained configuration of the `convert-to-llvm`
    process. For example, attributes with target information like
    `#nvvm.target` or `#rodcl.target` can leverage this interface for populating
    patterns specific to a particular target.
  }];
  let cppNamespace = "::mlir";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Populate the dialect conversion target, type converter and pattern set.
      }],
      /*retTy=*/"void",
      /*methodName=*/"populateConvertToLLVMConversionPatterns",
      /*args=*/(ins "::mlir::ConversionTarget&":$target,
                    "::mlir::LLVMTypeConverter&":$typeConverter,
                    "::mlir::RewritePatternSet&":$patternSet)>
  ];
}

//===----------------------------------------------------------------------===//
// Op interface
//===----------------------------------------------------------------------===//

def ConvertToLLVMOpInterface : OpInterface<"ConvertToLLVMOpInterface"> {
  let description = [{
    Interface for collecting all convert to LLVM attributes stored in an
    operation. See `ConvertToLLVMAttrInterface` for more information on these
    attributes.
  }];
  let cppNamespace = "::mlir";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Populate the provided vector with a list of convert to LLVM attributes
        to apply.
      }],
      /*retTy=*/"void",
      /*methodName=*/"getConvertToLLVMConversionAttrs",
      /*args=*/(ins
        "::llvm::SmallVectorImpl<::mlir::ConvertToLLVMAttrInterface>&":$attrs)
    >
  ];
}

#endif // MLIR_CONVERSION_CONVERTTOLLVM_TOLLVMINTERFACE_TD
