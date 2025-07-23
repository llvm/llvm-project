//===-- WasmSSAInterfaces.td - WasmSSA Interfaces -*- tablegen -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces for the WasmSSA dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef WasmSSA_INTERFACES
#define WasmSSA_INTERFACES

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"

def WasmSSALabelLevelOpInterface : OpInterface<"WasmSSALabelLevelOpInterface"> {
  let cppNamespace = "::mlir::wasmssa";
  let description = [{
    Operation that defines one level of nesting for wasm branching.
    These operation region can be targeted by branch instructions.
  }];
  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns the target block address",
      /*returnType=*/  "::mlir::Block*",
      /*methodName=*/  "getLabelTarget",
      /*args=*/        (ins)
    >
  ];
}

def WasmSSALabelBranchingOpInterface : OpInterface<"WasmSSALabelBranchingOpInterface"> {
  let cppNamespace = "::mlir::wasmssa";
  let description = [{
    Wasm operation that targets a label for a jump.
  }];
  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns the number of context to break from",
      /*returnType=*/  "size_t",
      /*methodName=*/  "getExitLevel",
      /*args=*/        (ins)
    >,
    InterfaceMethod<
      /*desc=*/        "Returns the destination of this operation",
      /*returnType=*/  "WasmSSALabelLevelOpInterface",
      /*methodName=*/  "getTargetOp",
      /*args=*/        (ins),
      /*methodBody=*/ [{
        return *WasmSSALabelBranchingOpInterface::getTargetOpFromBlock($_op.getOperation()->getBlock(), $_op.getExitLevel());
      }]
    >,
    InterfaceMethod<
      /*desc=*/        "Return the target control flow ops that defined the label of this operation",
      /*returnType=*/  "::mlir::Block*",
      /*methodName=*/  "getTarget",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = mlir::cast<WasmSSALabelBranchingOpInterface>(this->getOperation());
        return op.getTargetOp().getLabelTarget();
      }]
    >
  ];
  let extraClassDeclaration = [{
    static ::llvm::FailureOr<WasmSSALabelLevelOpInterface> getTargetOpFromBlock(::mlir::Block *block, uint32_t level);
  }];
  let verify = [{return verifyWasmSSALabelBranchingOpInterface($_op);}];
}

def WasmSSAImportOpInterface : OpInterface<"WasmSSAImportOpInterface"> {
  let cppNamespace = "::mlir::wasmssa";
  let description = [{
    Operation that imports a symbol from an external wasm module;
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns the module name for the import",
      /*returnType=*/  "::llvm::StringRef",
      /*methodName=*/  "getModuleName",
      /*args=*/        (ins)
      >,
    InterfaceMethod<
      /*desc=*/        "Returns the import name for the import",
      /*returnType=*/  "::llvm::StringRef",
      /*methodName=*/  "getImportName",
      /*args=*/        (ins)
      >,
    InterfaceMethod<
      /*desc=*/        "Returns the wasm index based symbol of the op",
      /*returnType=*/  "::mlir::StringAttr",
      /*methodName=*/  "getSymbolName",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = mlir::cast<ConcreteOp>(this->getOperation());
        return op.getSymNameAttr();
      }]
      >,
    InterfaceMethod<
      /*desc=*/        "Returns the qualified name of the import",
      /*returnType=*/  "std::string",
      /*methodName=*/  "getQualifiedImportName",
      /*args=*/        (ins),
      /*methodBody=*/  [{
        return ($_op.getModuleName() + ::llvm::Twine{"::"} + $_op.getImportName()).str();
      }]
      >,
  ];
}

def ConstantExpressionInitializerOpTrait : NativeOpTrait<"ConstantExpressionInitializerOpTrait"> {
  let cppNamespace = "::mlir::wasmssa";
}

def ConstantExprCheckOpInterface :
          OpInterface<"ConstantExprCheckOpInterface"> {
  let cppNamespace = "::mlir::wasmssa";
  let description = [{
    Base interface for operations that can be used in a Wasm Constant Expression.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/        [{
        Returns success if the current operation is valid in a constant expression context.
      }],
      /*returnType=*/  "::mlir::LogicalResult",
      /*methodName=*/  "isValidInConstantExpr",
      /*args=*/        (ins)
      >
  ];
}

def AlwaysValidInConstantExprOpTrait : NativeOpTrait<"AlwaysValidConstantExprOpTrait", [], [{
    ::mlir::LogicalResult isValidInConstantExpr() {
      return success();
    }
  }]> {
  let cppNamespace = "::mlir::wasmssa";
}

#endif // WEBASSEMBLY_INTERFACES
