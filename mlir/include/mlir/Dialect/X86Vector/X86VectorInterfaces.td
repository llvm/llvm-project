//===- X86VectorInterfaces.td - X86Vector interfaces -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces for the X86Vector dialect.
//
//===----------------------------------------------------------------------===//

#ifndef X86VECTOR_INTERFACES
#define X86VECTOR_INTERFACES

include "mlir/IR/Interfaces.td"

//===----------------------------------------------------------------------===//
// One-to-One Intrinsic Interface
//===----------------------------------------------------------------------===//

def OneToOneIntrinsicOpInterface : OpInterface<"OneToOneIntrinsicOp"> {
  let description = [{
    Interface for 1-to-1 conversion of an operation into LLVM intrinsics.

    An op implementing this interface can be simply replaced by a call
    to a matching intrinsic function.
    The op must ensure that the combinations of their arguments and results
    have valid intrinsic counterparts.

    For example, an operation supporting different vector widths:
    ```mlir
      %res_v8 = x86vector.op %value_v8 : vector<8xf32>
      %res_v16 = x86vector.op %value_v16 : vector<16xf32>
    ```
    can be converted to the following intrinsic calls:
    ```mlir
      %res_v8 = llvm.call_intrinsic "llvm.x86.op.intr.256"(%value_v8)
      %res_v16 = llvm.call_intrinsic "llvm.x86.op.intr.512"(%value_v16)
    ```
  }];
  let cppNamespace = "::mlir::x86vector";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns mangled LLVM intrinsic function name matching the operation
        variant.
      }],
      /*retType=*/"std::string",
      /*methodName=*/"getIntrinsicName"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns operands for a corresponding LLVM intrinsic.

        Additional operations may be created to facilitate mapping
        between the source operands and the target intrinsic.
      }],
      /*retType=*/"SmallVector<Value>",
      /*methodName=*/"getIntrinsicOperands",
      /*args=*/(ins "::mlir::RewriterBase &":$rewriter, "const LLVMTypeConverter &":$typeConverter),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return SmallVector<Value>($_op->getOperands());"
    >,
  ];
}

#endif // X86VECTOR_INTERFACES
