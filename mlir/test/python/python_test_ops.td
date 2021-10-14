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
include "mlir/Interfaces/InferTypeOpInterface.td"

def Python_Test_Dialect : Dialect {
  let name = "python_test";
  let cppNamespace = "python_test";
}
class TestOp<string mnemonic, list<OpTrait> traits = []>
    : Op<Python_Test_Dialect, mnemonic, traits>;

def AttributedOp : TestOp<"attributed_op"> {
  let arguments = (ins I32Attr:$mandatory_i32,
                   OptionalAttr<I32Attr>:$optional_i32,
                   UnitAttr:$unit);
}

def PropertyOp : TestOp<"property_op"> {
  let arguments = (ins I32Attr:$property,
                   I32:$idx);
}

def DummyOp : TestOp<"dummy_op"> {
}

def InferResultsOp : TestOp<"infer_results_op", [InferTypeOpInterface]> {
  let arguments = (ins);
  let results = (outs AnyInteger:$single, AnyInteger:$doubled);

  let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location,
      ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
      ::mlir::Builder b(context);
      inferredReturnTypes.push_back(b.getI32Type());
      inferredReturnTypes.push_back(b.getI64Type());
      return ::mlir::success();
    }
  }];
}

#endif // PYTHON_TEST_OPS
