//===-- python_test_ops.td - Python test Op definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PYTHON_TEST_OPS
#define PYTHON_TEST_OPS

include "mlir/Bindings/Python/Attributes.td"
include "mlir/IR/OpBase.td"

def Python_Test_Dialect : Dialect {
  let name = "python_test";
  let cppNamespace = "PythonTest";
}
class TestOp<string mnemonic, list<OpTrait> traits = []>
    : Op<Python_Test_Dialect, mnemonic, traits>;

def AttributedOp : TestOp<"attributed_op"> {
  let arguments = (ins I32Attr:$mandatory_i32,
                   OptionalAttr<I32Attr>:$optional_i32,
                   UnitAttr:$unit);
}

#endif // PYTHON_TEST_OPS
