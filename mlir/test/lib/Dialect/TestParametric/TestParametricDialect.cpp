//===- TestParametricDialect.cpp - MLIR Dialect for Testing
//----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestParametricDialect.h"
#include "TestParametricAttributes.h"
#include "TestParametricInterfaces.h"
#include "TestParametricTypes.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/ODSSupport.h"
#include "mlir/IR/OperationSupport.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <numeric>
#include <optional>

// Include this before the using namespace lines below to
// test that we don't have namespace dependencies.
#include "TestParametricOpsDialect.cpp.inc"

using namespace mlir;
using namespace testparametric;

void TestParametricDialect::initialize() {
  registerAttributes();
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "TestParametricOps.cpp.inc"
      >();
}
void testparametric::registerTestParametricDialect(DialectRegistry &registry) {
  registry.insert<TestParametricDialect>();
}

#include "TestParametricOpInterfaces.cpp.inc"
#include "TestParametricTypeInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "TestParametricOps.cpp.inc"

::mlir::ParseResult ParametricFuncOp::parse(mlir::OpAsmParser &parser,
                                            mlir::OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void ParametricFuncOp::print(mlir::OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  ParametricFuncOp fn =
      symbolTable.lookupNearestSymbolFrom<ParametricFuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  DictionaryAttr metaParams = fn.getMetaParamsAttr();
  DictionaryAttr metaArgs = getMetaArgs();
  if (metaParams && metaArgs.size() != metaParams.size())
    return emitOpError("incorrect number of meta operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i) {
    auto operandType = getOperand(i).getType();
    auto paramType = fnType.getInput(i);
    if (auto metaParamType = dyn_cast<ParamType>(paramType)) {
      auto metaArg = metaArgs.get(metaParamType.getRef());
      if (!metaArg)
        return emitOpError("Missing meta args for type operand ")
               << metaParamType.getRef();
      auto metaArgType = dyn_cast<TypeAttr>(metaArg);
      if (!metaArgType)
        return emitOpError("Expected TypeAttr for meta args ")
               << metaParamType.getRef() << ", got " << metaArg;
      if (metaArgType.getValue() != operandType)
        return emitOpError("Mismatch between operand type and meta args type: ")
               << operandType << " vs " << metaArgType;
      continue;
    }
    if (operandType != paramType) {
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;
    }
  }
  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i) {
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }
  }
  return success();
}

/// Specialization Interface Implementation

static LogicalResult replaceValueType(Value value, Type newType) {
  for (OpOperand &use : value.getUses()) {
    if (auto paramOp = dyn_cast<ParametricOpInterface>(use.getOwner())) {
      if (failed(paramOp.checkOperand(use, newType))) {
        paramOp.emitOpError() << "fails to replace operand type for operand #"
                              << use.getOperandNumber() << " with " << newType;
        return failure();
      }
    }
  }
  value.setType(newType);
  return success();
}
static LogicalResult replaceValueType(Value value, DictionaryAttr metaArgs) {
  auto paramType = dyn_cast<ParamType>(value.getType());
  if (!paramType)
    return success();
  auto metaArg =
      llvm::dyn_cast_or_null<TypeAttr>(metaArgs.get(paramType.getRef()));
  if (!metaArg) {
    if (value.getDefiningOp())
      value.getDefiningOp()->emitError()
          << "expected TypeAttr for specializing meta arg " << paramType
          << ", got " << metaArgs;
    return failure();
  }
  return replaceValueType(value, metaArg.getValue());
}

LogicalResult ParametricFuncOp::specialize(DictionaryAttr metaArgs) {
  auto mangledName = getMangledName(metaArgs);
  if (failed(mangledName))
    return failure();
  setSymNameAttr(*mangledName);
  removeMetaParamsAttr();

  auto specializeTypes = [&](auto typeRange, SmallVector<Type> &specialized) {
    for (Type ty : typeRange) {
      auto paramType = dyn_cast<ParamType>(ty);
      if (!paramType) {
        specialized.push_back(ty);
        continue;
      }
      auto metaArg =
          llvm::dyn_cast_or_null<TypeAttr>(metaArgs.get(paramType.getRef()));
      if (!metaArg) {
        emitOpError() << "expected TypeAttr for specializing meta arg "
                      << paramType << ", got " << metaArgs;
        return failure();
      }
      specialized.push_back(metaArg.getValue());
    }
    return success();
  };
  auto fnType = getFunctionType();
  SmallVector<Type> argTypes, resTypes;
  if (failed(specializeTypes(fnType.getInputs(), argTypes)))
    return failure();
  if (failed(specializeTypes(fnType.getResults(), resTypes)))
    return failure();
  for (auto argTypes : llvm::zip(argTypes, this->getArguments())) {
    auto newType = std::get<0>(argTypes);
    auto blockArg = std::get<1>(argTypes);
    if (failed(replaceValueType(blockArg, newType)))
      return failure();
  }

  setFunctionType(FunctionType::get(getContext(), argTypes, resTypes));
  if (getFunctionBody()
          .walk([&](Operation *op) {
            if (auto parametricOp = dyn_cast<ParametricOpInterface>(op)) {
              if (failed(parametricOp.specialize(metaArgs)))
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
          })
          .wasInterrupted())
    return failure();
  return success();
}

LogicalResult ParametricFuncOp::checkOperand(mlir::OpOperand &, mlir::Type) {
  return success();
}

FailureOr<StringAttr>
ParametricFuncOp::getMangledName(DictionaryAttr metaArgs) {
  auto name = getNameAttr();
  if (!name)
    return failure();
  std::string mangledName;
  llvm::raw_string_ostream os(mangledName);
  os << name.getValue() << "$__mlir_instance__";
  for (NamedAttribute name : metaArgs) {
    os << "$" << name.getName().getValue();
    Attribute value = name.getValue();
    if (auto intAttr = dyn_cast<IntegerAttr>(value))
      os << "$" << intAttr.getValue();
    else
      os << "$" << value;
  }

  return StringAttr::get(getContext(), os.str());
}

LogicalResult AddOp::specialize(DictionaryAttr metaArgs) {
  if (failed(replaceValueType(getResult(), metaArgs)))
    return failure();
  return success();
}

LogicalResult AddOp::checkOperand(mlir::OpOperand &, mlir::Type) {
  return success();
}

SymbolRefAttr CallOp::getTarget() { return getCalleeAttr(); }

LogicalResult CallOp::setSpecializedTarget(SymbolOpInterface target) {
  // TODO: check validity first.
  setCalleeAttr(SymbolRefAttr::get(target.getNameAttr()));
  setMetaArgsAttr(DictionaryAttr::get(getContext()));
  return success();
}

LogicalResult PrintAttrOp::specialize(DictionaryAttr metaArgs) {
  auto valueAttr = dyn_cast_or_null<ParamAttr>(getValueAttr());
  if (!valueAttr)
    return success();
  auto metaArg = metaArgs.get(valueAttr.getRef());
  if (!metaArg) {
    emitOpError() << "failed to specialize, missing " << valueAttr.getRef()
                  << " entry in " << metaArgs;
    return failure();
  }
  setValueAttr(metaArg);
  return success();
}

LogicalResult PrintAttrOp::checkOperand(mlir::OpOperand &, mlir::Type) {
  return success();
}
