//===-- TestOps.td - Test dialect operation definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TESTPARAMETRIC_OPS
#define TESTPARAMETRIC_OPS

include "TestParametricDialect.td"
include "mlir/Dialect/DLTI/DLTIBase.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/PatternBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/ParametricSpecializationOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


// Include the attribute definitions.
include "TestParametricAttrDefs.td"
// Include the type definitions.
include "TestParametricTypeDefs.td"


class TESTParametric_Op<string mnemonic, list<Trait> traits = []> :
    Op<TestParametric_Dialect, mnemonic, traits>;

def TESTParametric_ParametricFuncOp : TESTParametric_Op<"func", [
  AutomaticAllocationScope, FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface,
  DeclareOpInterfaceMethods<ParametricOpInterface>
  ]> {
  let summary = "Parametric function.";
  let description = [{
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictionaryAttr>:$metaParams,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the argument types of this function.
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static ::llvm::StringRef getDefaultDialect() { return "testparametric"; }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }

    //===------------------------------------------------------------------===//
    // ParametricOpInterface Methods
    //===------------------------------------------------------------------===//

    ::mlir::FailureOr<::mlir::StringAttr> getMangledName(::mlir::DictionaryAttr);
  }];
  let hasCustomAssemblyFormat = 1;
}


def ReturnOp : TESTParametric_Op<"return", [Pure, HasParent<"ParametricFuncOp">,
                                ReturnLike, Terminator]> {
  let summary = "Function return operation";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def CallOp : TESTParametric_Op<"call",
    [CallOpInterface,
     DeclareOpInterfaceMethods<SpecializingOpInterface>,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "call operation";
  let description = [{
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$operands,
    DictionaryAttr:$metaArgs
  );
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "mlir::SymbolRefAttr":$callee, "mlir::TypeRange":$results,
      CArg<"mlir::ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "mlir::StringAttr":$callee, "mlir::TypeRange":$results,
      CArg<"mlir::ValueRange", "{}">:$operands), [{
      build($_builder, $_state, mlir::SymbolRefAttr::get(callee), results, operands);
    }]>,
    OpBuilder<(ins "llvm::StringRef":$callee, "mlir::TypeRange":$results,
      CArg<"mlir::ValueRange", "{}">:$operands), [{
      build($_builder, $_state, mlir::StringAttr::get($_builder.getContext(), callee),
            results, operands);
    }]>];

  let extraClassDeclaration = [{
    mlir::FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    mlir::MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<mlir::SymbolRefAttr>());
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` `meta` `=` $metaArgs attr-dict `:` functional-type($operands, results)
  }];
}

def AddOp : TESTParametric_Op<"add", [Pure, DeclareOpInterfaceMethods<ParametricOpInterface>]> {
  let summary = "Add operation";
  let description = [{
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs
  );
  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = "$lhs `` `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def PrintAttrOp : TESTParametric_Op<"print_attr", [DeclareOpInterfaceMethods<ParametricOpInterface>]> {
  let summary = "Print operation";
  let description = [{
  }];

  let arguments = (ins
    AnyAttr:$value
  );

  let assemblyFormat = "$value attr-dict";
}



#endif // TESTPARAMETRIC_OPS
