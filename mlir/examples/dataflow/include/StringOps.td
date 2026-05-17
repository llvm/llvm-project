//===-- StringOps.td - String dialect operation definitions *- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the String dialect.
//
//===----------------------------------------------------------------------===//

#ifndef STRING_OPS
#define STRING_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def String_Dialect : Dialect {
  let name = "string";
  let cppNamespace = "mlir::string";
  let description = [{
    The `String` dialect provides string-related operations,
    which are used for string constant propagation.
  }];
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// String Type Definitions
//===----------------------------------------------------------------------===//

def StringType : TypeDef<String_Dialect, "StringType", []> {
  let mnemonic = "str";
  let summary = "string literal";
  let description = [{
    `string.str` is a type returned by ops of string dialect.
  }];
}

//===----------------------------------------------------------------------===//
// String Op Definitions
//===----------------------------------------------------------------------===//

class String_Op<string mnemonic, list<Trait> traits = [Pure]> :
  Op<String_Dialect, mnemonic, traits> {}

def ConstantOp : String_Op<"constant"> {
  let arguments = (ins StrAttr:$value);
  let results = (outs StringType:$res);
  let description = [{
    The `string.constant` op is used to create string constants.
    ```mlir
    %bar = string.constant "bar"
    ```
  }];
  let assemblyFormat = "$value attr-dict";
}

def ConcatOp : String_Op<"concat"> {
  let arguments = (ins StringType:$lhs, StringType:$rhs);
  let results = (outs StringType:$res);
  let description = [{
    The `string.concat` op is used to concatenate strings.
    ```mlir
    %bar = string.constant "bar"
    %foo = string.constant "foo"
    %concat = string.concat %bar, %foo
    ```
  }];
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

#endif // STRING_OPS
