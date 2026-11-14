//===- LLVMControlFlowOps.cpp - LLVM dialect control flow ops -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `ControlFlow` of the LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMControlFlowOps.h"

#include "IR/LLVMOps.h"
#include "mlir/Dialect/LLVMIR/LLVMCastOps.h"
#include "mlir/Dialect/LLVMIR/LLVMConstantOps.h"
#include "mlir/Dialect/LLVMIR/LLVMModuleOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

using namespace mlir;
using namespace mlir::LLVM;

static auto processFMFAttr(ArrayRef<NamedAttribute> attrs) {
  SmallVector<NamedAttribute, 8> filteredAttrs(
      llvm::make_filter_range(attrs, [&](NamedAttribute attr) {
        if (attr.getName() == "fastmathFlags") {
          auto defAttr =
              FastmathFlagsAttr::get(attr.getValue().getContext(), {});
          return defAttr != attr.getValue();
        }
        return true;
      }));
  return filteredAttrs;
}

//===----------------------------------------------------------------------===//
// Operand bundle helpers.
//===----------------------------------------------------------------------===//

static void printOneOpBundle(OpAsmPrinter &p, OperandRange operands,
                             TypeRange operandTypes, StringRef tag) {
  p.printString(tag);
  p << "(";

  if (!operands.empty()) {
    p.printOperands(operands);
    p << " : ";
    llvm::interleaveComma(operandTypes, p);
  }

  p << ")";
}

void mlir::LLVM::printOpBundles(OpAsmPrinter &p, Operation *op,
                                OperandRangeRange opBundleOperands,
                                TypeRangeRange opBundleOperandTypes,
                                std::optional<ArrayAttr> opBundleTags) {
  if (opBundleOperands.empty())
    return;
  assert(opBundleTags && "expect operand bundle tags");

  p << "[";
  llvm::interleaveComma(
      llvm::zip(opBundleOperands, opBundleOperandTypes, *opBundleTags), p,
      [&p](auto bundle) {
        auto bundleTag = cast<StringAttr>(std::get<2>(bundle)).getValue();
        printOneOpBundle(p, std::get<0>(bundle), std::get<1>(bundle),
                         bundleTag);
      });
  p << "]";
}

static ParseResult parseOneOpBundle(
    OpAsmParser &p,
    SmallVector<SmallVector<OpAsmParser::UnresolvedOperand>> &opBundleOperands,
    SmallVector<SmallVector<Type>> &opBundleOperandTypes,
    SmallVector<Attribute> &opBundleTags) {
  SMLoc currentParserLoc = p.getCurrentLocation();
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<Type> types;
  std::string tag;

  if (p.parseString(&tag))
    return p.emitError(currentParserLoc, "expect operand bundle tag");

  if (p.parseLParen())
    return failure();

  if (p.parseOptionalRParen()) {
    if (p.parseOperandList(operands) || p.parseColon() ||
        p.parseTypeList(types) || p.parseRParen())
      return failure();
  }

  opBundleOperands.push_back(std::move(operands));
  opBundleOperandTypes.push_back(std::move(types));
  opBundleTags.push_back(StringAttr::get(p.getContext(), tag));

  return success();
}

std::optional<ParseResult> mlir::LLVM::parseOpBundles(
    OpAsmParser &p,
    SmallVector<SmallVector<OpAsmParser::UnresolvedOperand>> &opBundleOperands,
    SmallVector<SmallVector<Type>> &opBundleOperandTypes,
    ArrayAttr &opBundleTags) {
  if (p.parseOptionalLSquare())
    return std::nullopt;

  if (succeeded(p.parseOptionalRSquare()))
    return success();

  SmallVector<Attribute> opBundleTagAttrs;
  auto bundleParser = [&] {
    return parseOneOpBundle(p, opBundleOperands, opBundleOperandTypes,
                            opBundleTagAttrs);
  };
  if (p.parseCommaSeparatedList(bundleParser))
    return failure();

  if (p.parseRSquare())
    return failure();

  opBundleTags = ArrayAttr::get(p.getContext(), opBundleTagAttrs);

  return success();
}

//===----------------------------------------------------------------------===//
// LLVM::BrOp
//===----------------------------------------------------------------------===//

SuccessorOperands BrOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return SuccessorOperands(getDestOperandsMutable());
}

//===----------------------------------------------------------------------===//
// LLVM::CondBrOp
//===----------------------------------------------------------------------===//

SuccessorOperands CondBrOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return SuccessorOperands(index == 0 ? getTrueDestOperandsMutable()
                                      : getFalseDestOperandsMutable());
}

void CondBrOp::build(OpBuilder &builder, OperationState &result,
                     Value condition, Block *trueDest, ValueRange trueOperands,
                     Block *falseDest, ValueRange falseOperands,
                     std::optional<std::pair<uint32_t, uint32_t>> weights) {
  DenseI32ArrayAttr weightsAttr;
  if (weights)
    weightsAttr =
        builder.getDenseI32ArrayAttr({static_cast<int32_t>(weights->first),
                                      static_cast<int32_t>(weights->second)});

  build(builder, result, condition, trueOperands, falseOperands, weightsAttr,
        /*loop_annotation=*/{}, trueDest, falseDest);
}

//===----------------------------------------------------------------------===//
// LLVM::SwitchOp
//===----------------------------------------------------------------------===//

void SwitchOp::build(OpBuilder &builder, OperationState &result, Value value,
                     Block *defaultDestination, ValueRange defaultOperands,
                     DenseIntElementsAttr caseValues,
                     BlockRange caseDestinations,
                     ArrayRef<ValueRange> caseOperands,
                     ArrayRef<int32_t> branchWeights) {
  DenseI32ArrayAttr weightsAttr;
  if (!branchWeights.empty())
    weightsAttr = builder.getDenseI32ArrayAttr(branchWeights);

  build(builder, result, value, defaultOperands, caseOperands, caseValues,
        weightsAttr, defaultDestination, caseDestinations);
}

void SwitchOp::build(OpBuilder &builder, OperationState &result, Value value,
                     Block *defaultDestination, ValueRange defaultOperands,
                     ArrayRef<APInt> caseValues, BlockRange caseDestinations,
                     ArrayRef<ValueRange> caseOperands,
                     ArrayRef<int32_t> branchWeights) {
  DenseIntElementsAttr caseValuesAttr;
  if (!caseValues.empty()) {
    ShapedType caseValueType = VectorType::get(
        static_cast<int64_t>(caseValues.size()), value.getType());
    caseValuesAttr = DenseIntElementsAttr::get(caseValueType, caseValues);
  }

  build(builder, result, value, defaultDestination, defaultOperands,
        caseValuesAttr, caseDestinations, caseOperands, branchWeights);
}

void SwitchOp::build(OpBuilder &builder, OperationState &result, Value value,
                     Block *defaultDestination, ValueRange defaultOperands,
                     ArrayRef<int32_t> caseValues, BlockRange caseDestinations,
                     ArrayRef<ValueRange> caseOperands,
                     ArrayRef<int32_t> branchWeights) {
  DenseIntElementsAttr caseValuesAttr;
  if (!caseValues.empty()) {
    ShapedType caseValueType = VectorType::get(
        static_cast<int64_t>(caseValues.size()), value.getType());
    caseValuesAttr = DenseIntElementsAttr::get(caseValueType, caseValues);
  }

  build(builder, result, value, defaultDestination, defaultOperands,
        caseValuesAttr, caseDestinations, caseOperands, branchWeights);
}

/// <cases> ::= `[` (case (`,` case )* )? `]`
/// <case>  ::= integer `:` bb-id (`(` ssa-use-and-type-list `)`)?
ParseResult mlir::LLVM::parseSwitchOpCases(
    OpAsmParser &parser, Type flagType, DenseIntElementsAttr &caseValues,
    SmallVectorImpl<Block *> &caseDestinations,
    SmallVectorImpl<SmallVector<OpAsmParser::UnresolvedOperand>> &caseOperands,
    SmallVectorImpl<SmallVector<Type>> &caseOperandTypes) {
  if (failed(parser.parseLSquare()))
    return failure();
  if (succeeded(parser.parseOptionalRSquare()))
    return success();
  SmallVector<APInt> values;
  unsigned bitWidth = flagType.getIntOrFloatBitWidth();
  auto parseCase = [&]() {
    int64_t value = 0;
    if (failed(parser.parseInteger(value)))
      return failure();
    values.push_back(APInt(bitWidth, value, /*isSigned=*/true));

    Block *destination;
    SmallVector<OpAsmParser::UnresolvedOperand> operands;
    SmallVector<Type> operandTypes;
    if (parser.parseColon() || parser.parseSuccessor(destination))
      return failure();
    if (!parser.parseOptionalLParen()) {
      if (parser.parseOperandList(operands, OpAsmParser::Delimiter::None) ||
          parser.parseColonTypeList(operandTypes) || parser.parseRParen())
        return failure();
    }
    caseDestinations.push_back(destination);
    caseOperands.emplace_back(operands);
    caseOperandTypes.emplace_back(operandTypes);
    return success();
  };
  if (failed(parser.parseCommaSeparatedList(parseCase)))
    return failure();

  ShapedType caseValueType =
      VectorType::get(static_cast<int64_t>(values.size()), flagType);
  caseValues = DenseIntElementsAttr::get(caseValueType, values);
  return parser.parseRSquare();
}

void mlir::LLVM::printSwitchOpCases(OpAsmPrinter &p, SwitchOp op, Type flagType,
                                    DenseIntElementsAttr caseValues,
                                    SuccessorRange caseDestinations,
                                    OperandRangeRange caseOperands,
                                    const TypeRangeRange &caseOperandTypes) {
  p << '[';
  p.printNewline();
  if (!caseValues) {
    p << ']';
    return;
  }

  size_t index = 0;
  llvm::interleave(
      llvm::zip(caseValues, caseDestinations),
      [&](auto i) {
        p << "  ";
        p << std::get<0>(i);
        p << ": ";
        p.printSuccessorAndUseList(std::get<1>(i), caseOperands[index++]);
      },
      [&] {
        p << ',';
        p.printNewline();
      });
  p.printNewline();
  p << ']';
}

LogicalResult SwitchOp::verify() {
  if ((!getCaseValues() && !getCaseDestinations().empty()) ||
      (getCaseValues() &&
       getCaseValues()->size() !=
           static_cast<int64_t>(getCaseDestinations().size())))
    return emitOpError("expects number of case values to match number of "
                       "case destinations");
  if (getCaseValues() &&
      getValue().getType() != getCaseValues()->getElementType())
    return emitError("expects case value type to match condition value type");
  return success();
}

SuccessorOperands SwitchOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return SuccessorOperands(index == 0 ? getDefaultOperandsMutable()
                                      : getCaseOperandsMutable(index - 1));
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

/// Gets the MLIR Op-like result types of a LLVMFunctionType.
static SmallVector<Type, 1> getCallOpResultTypes(LLVMFunctionType calleeType) {
  SmallVector<Type, 1> results;
  Type resultType = calleeType.getReturnType();
  if (!isa<LLVM::LLVMVoidType>(resultType))
    results.push_back(resultType);
  return results;
}

/// Gets the variadic callee type for a LLVMFunctionType.
static TypeAttr getCallOpVarCalleeType(LLVMFunctionType calleeType) {
  return calleeType.isVarArg() ? TypeAttr::get(calleeType) : nullptr;
}

/// Constructs a LLVMFunctionType from MLIR `results` and `args`.
static LLVMFunctionType getLLVMFuncType(MLIRContext *context, TypeRange results,
                                        ValueRange args) {
  Type resultType;
  if (results.empty())
    resultType = LLVMVoidType::get(context);
  else
    resultType = results.front();
  return LLVMFunctionType::get(resultType, llvm::to_vector(args.getTypes()),
                               /*isVarArg=*/false);
}

void CallOp::build(OpBuilder &builder, OperationState &state, TypeRange results,
                   StringRef callee, ValueRange args) {
  build(builder, state, results, builder.getStringAttr(callee), args);
}

void CallOp::build(OpBuilder &builder, OperationState &state, TypeRange results,
                   StringAttr callee, ValueRange args) {
  build(builder, state, results, SymbolRefAttr::get(callee), args);
}

void CallOp::build(OpBuilder &builder, OperationState &state, TypeRange results,
                   FlatSymbolRefAttr callee, ValueRange args) {
  assert(callee && "expected non-null callee in direct call builder");
  build(builder, state, results,
        /*var_callee_type=*/nullptr, callee, args, /*fastmathFlags=*/nullptr,
        /*CConv=*/nullptr, /*TailCallKind=*/nullptr,
        /*memory_effects=*/nullptr,
        /*convergent=*/nullptr, /*no_unwind=*/nullptr, /*will_return=*/nullptr,
        /*noreturn=*/nullptr, /*returns_twice=*/nullptr, /*hot=*/nullptr,
        /*cold=*/nullptr, /*noduplicate=*/nullptr,
        /*no_caller_saved_registers=*/nullptr, /*nocallback=*/nullptr,
        /*modular_format=*/nullptr, /*nobuiltins=*/nullptr,
        /*allocsize=*/nullptr, /*optsize=*/nullptr, /*minsize=*/nullptr,
        /*builtin=*/nullptr, /*nobuiltin=*/nullptr,
        /*save_reg_params=*/nullptr,
        /*zero_call_used_regs=*/nullptr, /*trap_func_name=*/nullptr,
        /*default_func_attrs=*/nullptr,
        /*uniform_work_group_size=*/nullptr,
        /*op_bundle_operands=*/{}, /*op_bundle_tags=*/{},
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr,
        /*access_groups=*/nullptr, /*alias_scopes=*/nullptr,
        /*noalias_scopes=*/nullptr, /*tbaa=*/nullptr,
        /*no_inline=*/nullptr, /*always_inline=*/nullptr,
        /*inline_hint=*/nullptr);
}

void CallOp::build(OpBuilder &builder, OperationState &state,
                   LLVMFunctionType calleeType, StringRef callee,
                   ValueRange args) {
  build(builder, state, calleeType, builder.getStringAttr(callee), args);
}

void CallOp::build(OpBuilder &builder, OperationState &state,
                   LLVMFunctionType calleeType, StringAttr callee,
                   ValueRange args) {
  build(builder, state, calleeType, SymbolRefAttr::get(callee), args);
}

void CallOp::build(OpBuilder &builder, OperationState &state,
                   LLVMFunctionType calleeType, FlatSymbolRefAttr callee,
                   ValueRange args) {
  build(builder, state, getCallOpResultTypes(calleeType),
        getCallOpVarCalleeType(calleeType), callee, args,
        /*fastmathFlags=*/nullptr,
        /*CConv=*/nullptr,
        /*TailCallKind=*/nullptr, /*memory_effects=*/nullptr,
        /*convergent=*/nullptr,
        /*no_unwind=*/nullptr, /*will_return=*/nullptr,
        /*noreturn=*/nullptr,
        /*returns_twice=*/nullptr, /*hot=*/nullptr,
        /*cold=*/nullptr, /*noduplicate=*/nullptr,
        /*no_caller_saved_registers=*/nullptr, /*nocallback=*/nullptr,
        /*modular_format=*/nullptr, /*nobuiltins=*/nullptr,
        /*allocsize=*/nullptr, /*optsize=*/nullptr, /*minsize=*/nullptr,
        /*builtin=*/nullptr, /*nobuiltin=*/nullptr,
        /*save_reg_params=*/nullptr,
        /*zero_call_used_regs=*/nullptr, /*trap_func_name=*/nullptr,
        /*default_func_attrs=*/nullptr,
        /*uniform_work_group_size=*/nullptr,
        /*op_bundle_operands=*/{}, /*op_bundle_tags=*/{},
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr,
        /*access_groups=*/nullptr,
        /*alias_scopes=*/nullptr, /*noalias_scopes=*/nullptr, /*tbaa=*/nullptr,
        /*no_inline=*/nullptr, /*always_inline=*/nullptr,
        /*inline_hint=*/nullptr);
}

void CallOp::build(OpBuilder &builder, OperationState &state,
                   LLVMFunctionType calleeType, ValueRange args) {
  build(builder, state, getCallOpResultTypes(calleeType),
        getCallOpVarCalleeType(calleeType),
        /*callee=*/nullptr, args,
        /*fastmathFlags=*/nullptr,
        /*CConv=*/nullptr, /*TailCallKind=*/nullptr, /*memory_effects=*/nullptr,
        /*convergent=*/nullptr, /*no_unwind=*/nullptr, /*will_return=*/nullptr,
        /*noreturn=*/nullptr,
        /*returns_twice=*/nullptr, /*hot=*/nullptr,
        /*cold=*/nullptr, /*noduplicate=*/nullptr,
        /*no_caller_saved_registers=*/nullptr, /*nocallback=*/nullptr,
        /*modular_format=*/nullptr, /*nobuiltins=*/nullptr,
        /*allocsize=*/nullptr, /*optsize=*/nullptr, /*minsize=*/nullptr,
        /*builtin=*/nullptr, /*nobuiltin=*/nullptr,
        /*save_reg_params=*/nullptr,
        /*zero_call_used_regs=*/nullptr, /*trap_func_name=*/nullptr,
        /*default_func_attrs=*/nullptr,
        /*uniform_work_group_size=*/nullptr,
        /*op_bundle_operands=*/{}, /*op_bundle_tags=*/{},
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr,
        /*access_groups=*/nullptr, /*alias_scopes=*/nullptr,
        /*noalias_scopes=*/nullptr, /*tbaa=*/nullptr,
        /*no_inline=*/nullptr, /*always_inline=*/nullptr,
        /*inline_hint=*/nullptr);
}

void CallOp::build(OpBuilder &builder, OperationState &state, LLVMFuncOp func,
                   ValueRange args) {
  auto calleeType = func.getFunctionType();
  build(builder, state, getCallOpResultTypes(calleeType),
        getCallOpVarCalleeType(calleeType), SymbolRefAttr::get(func), args,
        /*fastmathFlags=*/nullptr,
        /*CConv=*/nullptr, /*TailCallKind=*/nullptr, /*memory_effects=*/nullptr,
        /*convergent=*/nullptr, /*no_unwind=*/nullptr, /*will_return=*/nullptr,
        /*noreturn=*/nullptr,
        /*returns_twice=*/nullptr, /*hot=*/nullptr,
        /*cold=*/nullptr, /*noduplicate=*/nullptr,
        /*no_caller_saved_registers=*/nullptr, /*nocallback=*/nullptr,
        /*modular_format=*/nullptr, /*nobuiltins=*/nullptr,
        /*allocsize=*/nullptr, /*optsize=*/nullptr, /*minsize=*/nullptr,
        /*builtin=*/nullptr, /*nobuiltin=*/nullptr,
        /*save_reg_params=*/nullptr,
        /*zero_call_used_regs=*/nullptr, /*trap_func_name=*/nullptr,
        /*default_func_attrs=*/nullptr,
        /*uniform_work_group_size=*/nullptr,
        /*op_bundle_operands=*/{}, /*op_bundle_tags=*/{},
        /*access_groups=*/nullptr, /*alias_scopes=*/nullptr,
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr,
        /*noalias_scopes=*/nullptr, /*tbaa=*/nullptr,
        /*no_inline=*/nullptr, /*always_inline=*/nullptr,
        /*inline_hint=*/nullptr);
}

CallInterfaceCallable CallOp::getCallableForCallee() {
  // Direct call.
  if (FlatSymbolRefAttr calleeAttr = getCalleeAttr())
    return calleeAttr;
  // Indirect call, callee Value is the first operand.
  return getOperand(0);
}

void CallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  // Direct call.
  if (FlatSymbolRefAttr calleeAttr = getCalleeAttr()) {
    auto symRef = cast<SymbolRefAttr>(callee);
    return setCalleeAttr(cast<FlatSymbolRefAttr>(symRef));
  }
  // Indirect call, callee Value is the first operand.
  return setOperand(0, cast<Value>(callee));
}

/// Return the number of leading callee operands of `callOp` that the operation
/// consumes instead of passing them to the callee.
template <typename OpTy>
static unsigned getNumConsumedCalleeOperands(OpTy callOp) {
  // The first operand is the callee if no callee attribute is present.
  if (callOp.getCallee().has_value())
    return 0;
  return 1;
}

/// Return the operands of `callOp` that are passed to the callee, including the
/// variadic arguments in case of a call to a variadic callee.
template <typename OpTy>
static Operation::operand_range getOperandsPassedToCallee(OpTy callOp) {
  return callOp.getCalleeOperands().drop_front(
      getNumConsumedCalleeOperands(callOp));
}

/// Return the operands of `callOp` that correspond to the declared parameters
/// of the callee, i.e., its `CallOpInterface` argument operands.
///
/// The variadic arguments of a call to a variadic callee are *not* included:
/// they do not correspond to any argument of the callee. The callee does not
/// receive them as block arguments but reads them with `llvm.intr.vastart` and
/// friends, so in terms of `CallOpInterface` they are consumed operands rather
/// than forwarded ones.
template <typename OpTy>
static Operation::operand_range getArgOperandsImpl(OpTy callOp) {
  Operation::operand_range operands = getOperandsPassedToCallee(callOp);
  if (std::optional<LLVMFunctionType> varCalleeType = callOp.getVarCalleeType())
    return operands.take_front(varCalleeType->getNumParams());
  return operands;
}

Operation::operand_range CallOp::getArgOperands() {
  return getArgOperandsImpl(*this);
}

MutableOperandRange CallOp::getArgOperandsMutable() {
  // Slice the generated range to retain its segment-size metadata. A raw
  // range would not update operandSegmentSizes when arguments are erased.
  return getCalleeOperandsMutable().slice(getNumConsumedCalleeOperands(*this),
                                          getArgOperandsImpl(*this).size());
}

/// Verify that an inlinable callsite of a debug-info-bearing function in a
/// debug-info-bearing function has a debug location attached to it. This
/// mirrors an LLVM IR verifier.
template <typename OpTy>
static LogicalResult verifyCallOpDebugInfo(OpTy callOp, LLVMFuncOp callee) {
  if (callee.isExternal())
    return success();
  auto parentFunc = callOp->template getParentOfType<FunctionOpInterface>();
  if (!parentFunc)
    return success();

  auto hasSubprogram = [](Operation *op) {
    return op->getLoc()
               ->findInstanceOf<FusedLocWith<LLVM::DISubprogramAttr>>() !=
           nullptr;
  };
  if (!hasSubprogram(parentFunc) || !hasSubprogram(callee))
    return success();
  bool containsLoc = !isa<UnknownLoc>(callOp->getLoc());
  if (!containsLoc)
    return callOp.emitError()
           << "inlinable function call in a function with a DISubprogram "
              "location must have a debug location";
  return success();
}

/// Verify that the parameter and return types of the variadic callee type match
/// the `callOp` argument and result types.
template <typename OpTy>
static LogicalResult verifyCallOpVarCalleeType(OpTy callOp) {
  // An indirect call stores the callee in its first callee operand.
  if (!callOp.getCallee().has_value() && callOp.getCalleeOperands().empty())
    return callOp.emitOpError(
        "must have either a `callee` attribute or at least an operand");

  std::optional<LLVMFunctionType> varCalleeType = callOp.getVarCalleeType();
  if (!varCalleeType)
    return success();

  // Verify the variadic callee type is a variadic function type.
  if (!varCalleeType->isVarArg())
    return callOp.emitOpError(
        "expected var_callee_type to be a variadic function type");

  // Note: `getArgOperands` is derived from `var_callee_type`, so the raw callee
  // operands are used here instead.
  Operation::operand_range passedOperands = getOperandsPassedToCallee(callOp);

  // Verify the variadic callee type has at most as many parameters as the call
  // has argument operands.
  if (varCalleeType->getNumParams() > passedOperands.size())
    return callOp.emitOpError("expected var_callee_type to have at most ")
           << passedOperands.size() << " parameters";

  // Verify the variadic callee type matches the call argument types.
  for (auto [paramType, operand] :
       llvm::zip(varCalleeType->getParams(), passedOperands))
    if (paramType != operand.getType())
      return callOp.emitOpError()
             << "var_callee_type parameter type mismatch: " << paramType
             << " != " << operand.getType();

  // Verify the variadic callee type matches the call result type.
  if (!callOp.getNumResults()) {
    if (!isa<LLVMVoidType>(varCalleeType->getReturnType()))
      return callOp.emitOpError("expected var_callee_type to return void");
  } else {
    if (callOp.getResult().getType() != varCalleeType->getReturnType())
      return callOp.emitOpError("var_callee_type return type mismatch: ")
             << varCalleeType->getReturnType()
             << " != " << callOp.getResult().getType();
  }
  return success();
}

LogicalResult CallOp::verify() { return verifyOperandBundles(*this); }

template <typename OpTy>
static LogicalResult
verifyCallOpSymbolUses(OpTy callOp, SymbolTableCollection &symbolTable) {
  if (failed(verifyCallOpVarCalleeType(callOp)))
    return failure();
  // Type for the callee, we'll get it differently depending if it is a direct
  // or indirect call.
  Type fnType;

  // If this is an indirect call, the callee attribute is missing.
  FlatSymbolRefAttr calleeName = callOp.getCalleeAttr();
  if (!calleeName) {
    // Note: `verifyCallOpVarCalleeType` has already checked that there is a
    // callee operand.
    auto ptrType =
        llvm::dyn_cast<LLVMPointerType>(callOp.getOperand(0).getType());
    if (!ptrType)
      return callOp.emitOpError("indirect call expects a pointer as callee: ")
             << callOp.getOperand(0).getType();

    // Nothing else to verify: an indirect callee cannot be resolved.
    return success();
  } else {
    Operation *callee =
        symbolTable.lookupNearestSymbolFrom(callOp, calleeName.getAttr());
    if (!callee)
      return callOp.emitOpError()
             << "'" << calleeName.getValue()
             << "' does not reference a symbol in the current scope";
    if (auto fn = dyn_cast<LLVMFuncOp>(callee)) {
      if (failed(verifyCallOpDebugInfo(callOp, fn)))
        return failure();
      fnType = fn.getFunctionType();
    } else if (auto ifunc = dyn_cast<IFuncOp>(callee)) {
      fnType = ifunc.getIFuncType();
    } else if (isa<AliasOp>(callee)) {
      // Aliases can alias functions, so calling through an alias is valid.
      // The function type is determined by the call's operands and result
      // types.
      fnType = callOp.getCalleeFunctionType();
    } else {
      return callOp.emitOpError()
             << "'" << calleeName.getValue()
             << "' does not reference a valid LLVM function, IFunc, or alias";
    }
  }

  LLVMFunctionType funcType = llvm::dyn_cast<LLVMFunctionType>(fnType);
  if (!funcType)
    return callOp.emitOpError("callee does not have a functional type: ")
           << fnType;

  if (funcType.isVarArg() && !callOp.getVarCalleeType())
    return callOp.emitOpError()
           << "missing var_callee_type attribute for vararg call";

  // Verify the result types. These checks are more specific than what
  // `verifyCallOpInterface` can report, so they are run first.
  if (callOp.getNumResults() == 0 &&
      !llvm::isa<LLVM::LLVMVoidType>(funcType.getReturnType()))
    return callOp.emitOpError() << "expected function call to produce a value";

  if (callOp.getNumResults() != 0 &&
      llvm::isa<LLVM::LLVMVoidType>(funcType.getReturnType()))
    return callOp.emitOpError()
           << "calling function with void result must not produce values";

  if (callOp.getNumResults() > 1)
    return callOp.emitOpError()
           << "expected LLVM function call to produce 0 or 1 result";

  if (callOp.getNumResults() &&
      callOp.getResult().getType() != funcType.getReturnType())
    return callOp.emitOpError()
           << "result type mismatch: " << callOp.getResult().getType()
           << " != " << funcType.getReturnType();

  // Verify that the operand types match the callee. Note that this does not
  // need to special-case a variadic callee: the variadic arguments are not
  // argument operands.
  SmallVector<Type, 1> calleeResultTypes;
  if (!llvm::isa<LLVM::LLVMVoidType>(funcType.getReturnType()))
    calleeResultTypes.push_back(funcType.getReturnType());
  return call_interface_impl::verifyCallOpInterface(
      callOp, funcType.getParams(), calleeResultTypes);
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyCallOpSymbolUses(*this, symbolTable);
}

void CallOp::print(OpAsmPrinter &p) {
  auto callee = getCallee();
  bool isDirect = callee.has_value();

  p << ' ';

  // Print calling convention.
  if (getCConv() != LLVM::CConv::C)
    p << stringifyCConv(getCConv()) << ' ';

  if (getTailCallKind() != LLVM::TailCallKind::None)
    p << tailcallkind::stringifyTailCallKind(getTailCallKind()) << ' ';

  // Print the direct callee if present as a function attribute, or an indirect
  // callee (first operand) otherwise.
  if (isDirect)
    p.printSymbolName(callee.value());
  else
    p << getOperand(0);

  auto args = getCalleeOperands().drop_front(isDirect ? 0 : 1);
  p << '(' << args << ')';

  // Print the variadic callee type if the call is variadic.
  if (std::optional<LLVMFunctionType> varCalleeType = getVarCalleeType())
    p << " vararg(" << *varCalleeType << ")";

  if (!getOpBundleOperands().empty()) {
    p << " ";
    printOpBundles(p, *this, getOpBundleOperands(),
                   getOpBundleOperands().getTypes(), getOpBundleTags());
  }

  p.printOptionalAttrDict(processFMFAttr(getAttrsForPrinting(*this).getAttrs()),
                          {getCalleeAttrName(), getTailCallKindAttrName(),
                           getVarCalleeTypeAttrName(), getCConvAttrName(),
                           getOperandSegmentSizesAttrName(),
                           getOpBundleSizesAttrName(),
                           getOpBundleTagsAttrName(), getArgAttrsAttrName(),
                           getResAttrsAttrName()});

  p << " : ";
  if (!isDirect)
    p << getOperand(0).getType() << ", ";

  // Reconstruct the MLIR function type from operand and result types.
  call_interface_impl::printFunctionSignature(
      p, args.getTypes(), getArgAttrsAttr(),
      /*isVariadic=*/false, getResultTypes(), getResAttrsAttr());
}

/// Parses the type of a call operation and resolves the operands if the parsing
/// succeeds. Returns failure otherwise.
static ParseResult parseCallTypeAndResolveOperands(
    OpAsmParser &parser, OperationState &result, bool isDirect,
    ArrayRef<OpAsmParser::UnresolvedOperand> operands,
    SmallVectorImpl<DictionaryAttr> &argAttrs,
    SmallVectorImpl<DictionaryAttr> &resultAttrs) {
  SMLoc trailingTypesLoc = parser.getCurrentLocation();
  SmallVector<Type> types;
  if (parser.parseColon())
    return failure();
  if (!isDirect) {
    types.emplace_back();
    if (parser.parseType(types.back()))
      return failure();
    if (parser.parseOptionalComma())
      return parser.emitError(
          trailingTypesLoc, "expected indirect call to have 2 trailing types");
  }
  SmallVector<Type> argTypes;
  SmallVector<Type> resTypes;
  if (call_interface_impl::parseFunctionSignature(parser, argTypes, argAttrs,
                                                  resTypes, resultAttrs)) {
    if (isDirect)
      return parser.emitError(trailingTypesLoc,
                              "expected direct call to have 1 trailing types");
    return parser.emitError(trailingTypesLoc,
                            "expected trailing function type");
  }

  if (resTypes.size() > 1)
    return parser.emitError(trailingTypesLoc,
                            "expected function with 0 or 1 result");
  if (resTypes.size() == 1 && llvm::isa<LLVM::LLVMVoidType>(resTypes[0]))
    return parser.emitError(trailingTypesLoc,
                            "expected a non-void result type");

  // The head element of the types list matches the callee type for
  // indirect calls, while the types list is emtpy for direct calls.
  // Append the function input types to resolve the call operation
  // operands.
  llvm::append_range(types, argTypes);
  if (parser.resolveOperands(operands, types, parser.getNameLoc(),
                             result.operands))
    return failure();
  if (!resTypes.empty())
    result.addTypes(resTypes);

  return success();
}

/// Parses an optional function pointer operand before the call argument list
/// for indirect calls, or stops parsing at the function identifier otherwise.
static ParseResult parseOptionalCallFuncPtr(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands) {
  OpAsmParser::UnresolvedOperand funcPtrOperand;
  OptionalParseResult parseResult = parser.parseOptionalOperand(funcPtrOperand);
  if (parseResult.has_value()) {
    if (failed(*parseResult))
      return *parseResult;
    operands.push_back(funcPtrOperand);
  }
  return success();
}

static ParseResult resolveOpBundleOperands(
    OpAsmParser &parser, SMLoc loc, OperationState &state,
    ArrayRef<SmallVector<OpAsmParser::UnresolvedOperand>> opBundleOperands,
    ArrayRef<SmallVector<Type>> opBundleOperandTypes,
    StringAttr opBundleSizesAttrName) {
  unsigned opBundleIndex = 0;
  for (const auto &[operands, types] :
       llvm::zip_equal(opBundleOperands, opBundleOperandTypes)) {
    if (operands.size() != types.size())
      return parser.emitError(loc, "expected ")
             << operands.size()
             << " types for operand bundle operands for operand bundle #"
             << opBundleIndex << ", but actually got " << types.size();
    if (parser.resolveOperands(operands, types, loc, state.operands))
      return failure();
  }

  SmallVector<int32_t> opBundleSizes;
  opBundleSizes.reserve(opBundleOperands.size());
  for (const auto &operands : opBundleOperands)
    opBundleSizes.push_back(operands.size());

  state.addAttribute(
      opBundleSizesAttrName,
      DenseI32ArrayAttr::get(parser.getContext(), opBundleSizes));

  return success();
}

// <operation> ::= `llvm.call` (cconv)? (tailcallkind)? (function-id | ssa-use)
//                             `(` ssa-use-list `)`
//                             ( `vararg(` var-callee-type `)` )?
//                             ( `[` op-bundles-list `]` )?
//                             attribute-dict? `:` (type `,`)? function-type
ParseResult CallOp::parse(OpAsmParser &parser, OperationState &result) {
  SymbolRefAttr funcAttr;
  TypeAttr varCalleeType;
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<SmallVector<OpAsmParser::UnresolvedOperand>> opBundleOperands;
  SmallVector<SmallVector<Type>> opBundleOperandTypes;
  ArrayAttr opBundleTags;

  // Default to C Calling Convention if no keyword is provided.
  result.addAttribute(
      getCConvAttrName(result.name),
      CConvAttr::get(parser.getContext(),
                     parseOptionalLLVMKeyword<CConv>(parser, LLVM::CConv::C)));

  result.addAttribute(
      getTailCallKindAttrName(result.name),
      TailCallKindAttr::get(parser.getContext(),
                            parseOptionalLLVMKeyword<TailCallKind>(
                                parser, LLVM::TailCallKind::None)));

  // Parse a function pointer for indirect calls.
  if (parseOptionalCallFuncPtr(parser, operands))
    return failure();
  bool isDirect = operands.empty();

  // Parse a function identifier for direct calls.
  if (isDirect)
    if (parser.parseAttribute(funcAttr, "callee", result.attributes))
      return failure();

  // Parse the function arguments.
  if (parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren))
    return failure();

  bool isVarArg = parser.parseOptionalKeyword("vararg").succeeded();
  if (isVarArg) {
    StringAttr varCalleeTypeAttrName =
        CallOp::getVarCalleeTypeAttrName(result.name);
    if (parser.parseLParen().failed() ||
        parser
            .parseAttribute(varCalleeType, varCalleeTypeAttrName,
                            result.attributes)
            .failed() ||
        parser.parseRParen().failed())
      return failure();
  }

  SMLoc opBundlesLoc = parser.getCurrentLocation();
  if (std::optional<ParseResult> result = parseOpBundles(
          parser, opBundleOperands, opBundleOperandTypes, opBundleTags);
      result && failed(*result))
    return failure();
  if (opBundleTags && !opBundleTags.empty())
    result.addAttribute(CallOp::getOpBundleTagsAttrName(result.name).getValue(),
                        opBundleTags);

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse the trailing type list and resolve the operands.
  SmallVector<DictionaryAttr> argAttrs;
  SmallVector<DictionaryAttr> resultAttrs;
  if (parseCallTypeAndResolveOperands(parser, result, isDirect, operands,
                                      argAttrs, resultAttrs))
    return failure();
  call_interface_impl::addArgAndResultAttrs(
      parser.getBuilder(), result, argAttrs, resultAttrs,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
  if (resolveOpBundleOperands(parser, opBundlesLoc, result, opBundleOperands,
                              opBundleOperandTypes,
                              getOpBundleSizesAttrName(result.name)))
    return failure();

  int32_t numOpBundleOperands = 0;
  for (const auto &operands : opBundleOperands)
    numOpBundleOperands += operands.size();

  result.addAttribute(
      CallOp::getOperandSegmentSizeAttr(),
      parser.getBuilder().getDenseI32ArrayAttr(
          {static_cast<int32_t>(operands.size()), numOpBundleOperands}));
  return success();
}

LLVMFunctionType CallOp::getCalleeFunctionType() {
  if (std::optional<LLVMFunctionType> varCalleeType = getVarCalleeType())
    return *varCalleeType;
  return getLLVMFuncType(getContext(), getResultTypes(), getArgOperands());
}

///===---------------------------------------------------------------------===//
/// LLVM::InvokeOp
///===---------------------------------------------------------------------===//

void InvokeOp::build(OpBuilder &builder, OperationState &state, LLVMFuncOp func,
                     ValueRange ops, Block *normal, ValueRange normalOps,
                     Block *unwind, ValueRange unwindOps) {
  auto calleeType = func.getFunctionType();
  build(builder, state, getCallOpResultTypes(calleeType),
        getCallOpVarCalleeType(calleeType), SymbolRefAttr::get(func), ops,
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr, normalOps, unwindOps,
        nullptr, nullptr, /*default_func_attrs=*/nullptr,
        /*uniform_work_group_size=*/nullptr, {}, {}, normal, unwind);
}

void InvokeOp::build(OpBuilder &builder, OperationState &state, TypeRange tys,
                     FlatSymbolRefAttr callee, ValueRange ops, Block *normal,
                     ValueRange normalOps, Block *unwind,
                     ValueRange unwindOps) {
  build(builder, state, tys,
        /*var_callee_type=*/nullptr, callee, ops, /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr, normalOps, unwindOps, nullptr, nullptr,
        /*default_func_attrs=*/nullptr,
        /*uniform_work_group_size=*/nullptr, {}, {}, normal, unwind);
}

void InvokeOp::build(OpBuilder &builder, OperationState &state,
                     LLVMFunctionType calleeType, FlatSymbolRefAttr callee,
                     ValueRange ops, Block *normal, ValueRange normalOps,
                     Block *unwind, ValueRange unwindOps) {
  build(builder, state, getCallOpResultTypes(calleeType),
        getCallOpVarCalleeType(calleeType), callee, ops,
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr, normalOps, unwindOps,
        nullptr, nullptr, /*default_func_attrs=*/nullptr,
        /*uniform_work_group_size=*/nullptr, {}, {}, normal, unwind);
}

SuccessorOperands InvokeOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return SuccessorOperands(index == 0 ? getNormalDestOperandsMutable()
                                      : getUnwindDestOperandsMutable());
}

CallInterfaceCallable InvokeOp::getCallableForCallee() {
  // Direct call.
  if (FlatSymbolRefAttr calleeAttr = getCalleeAttr())
    return calleeAttr;
  // Indirect call, callee Value is the first operand.
  return getOperand(0);
}

void InvokeOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  // Direct call.
  if (FlatSymbolRefAttr calleeAttr = getCalleeAttr()) {
    auto symRef = cast<SymbolRefAttr>(callee);
    return setCalleeAttr(cast<FlatSymbolRefAttr>(symRef));
  }
  // Indirect call, callee Value is the first operand.
  return setOperand(0, cast<Value>(callee));
}

Operation::operand_range InvokeOp::getArgOperands() {
  return getArgOperandsImpl(*this);
}

MutableOperandRange InvokeOp::getArgOperandsMutable() {
  // Slice the generated range to retain its segment-size metadata. A raw
  // range would not update operandSegmentSizes when arguments are erased.
  return getCalleeOperandsMutable().slice(getNumConsumedCalleeOperands(*this),
                                          getArgOperandsImpl(*this).size());
}

LogicalResult InvokeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyCallOpSymbolUses(*this, symbolTable);
}

LogicalResult InvokeOp::verify() {
  Block *unwindDest = getUnwindDest();
  if (unwindDest->empty())
    return emitError("must have at least one operation in unwind destination");

  // In unwind destination, first operation must be LandingpadOp
  if (!isa<LandingpadOp>(unwindDest->front()))
    return emitError("first operation in unwind destination should be a "
                     "llvm.landingpad operation");

  if (failed(verifyOperandBundles(*this)))
    return failure();

  return success();
}

void InvokeOp::print(OpAsmPrinter &p) {
  auto callee = getCallee();
  bool isDirect = callee.has_value();

  p << ' ';

  // Print calling convention.
  if (getCConv() != LLVM::CConv::C)
    p << stringifyCConv(getCConv()) << ' ';

  // Either function name or pointer
  if (isDirect)
    p.printSymbolName(callee.value());
  else
    p << getOperand(0);

  p << '(' << getCalleeOperands().drop_front(isDirect ? 0 : 1) << ')';
  p << " to ";
  p.printSuccessorAndUseList(getNormalDest(), getNormalDestOperands());
  p << " unwind ";
  p.printSuccessorAndUseList(getUnwindDest(), getUnwindDestOperands());

  // Print the variadic callee type if the invoke is variadic.
  if (std::optional<LLVMFunctionType> varCalleeType = getVarCalleeType())
    p << " vararg(" << *varCalleeType << ")";

  if (!getOpBundleOperands().empty()) {
    p << " ";
    printOpBundles(p, *this, getOpBundleOperands(),
                   getOpBundleOperands().getTypes(), getOpBundleTags());
  }

  p.printOptionalAttrDict(getAttrsForPrinting(*this).getAttrs(),
                          {getCalleeAttrName(), getOperandSegmentSizeAttr(),
                           getCConvAttrName(), getVarCalleeTypeAttrName(),
                           getOpBundleSizesAttrName(),
                           getOpBundleTagsAttrName(), getArgAttrsAttrName(),
                           getResAttrsAttrName()});

  p << " : ";
  if (!isDirect)
    p << getOperand(0).getType() << ", ";
  call_interface_impl::printFunctionSignature(
      p, getCalleeOperands().drop_front(isDirect ? 0 : 1).getTypes(),
      getArgAttrsAttr(),
      /*isVariadic=*/false, getResultTypes(), getResAttrsAttr());
}

// <operation> ::= `llvm.invoke` (cconv)? (function-id | ssa-use)
//                  `(` ssa-use-list `)`
//                  `to` bb-id (`[` ssa-use-and-type-list `]`)?
//                  `unwind` bb-id (`[` ssa-use-and-type-list `]`)?
//                  ( `vararg(` var-callee-type `)` )?
//                  ( `[` op-bundles-list `]` )?
//                  attribute-dict? `:` (type `,`)?
//                  function-type-with-argument-attributes
ParseResult InvokeOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 8> operands;
  SymbolRefAttr funcAttr;
  TypeAttr varCalleeType;
  SmallVector<SmallVector<OpAsmParser::UnresolvedOperand>> opBundleOperands;
  SmallVector<SmallVector<Type>> opBundleOperandTypes;
  ArrayAttr opBundleTags;
  Block *normalDest, *unwindDest;
  SmallVector<Value, 4> normalOperands, unwindOperands;
  Builder &builder = parser.getBuilder();

  // Default to C Calling Convention if no keyword is provided.
  result.addAttribute(
      getCConvAttrName(result.name),
      CConvAttr::get(parser.getContext(),
                     parseOptionalLLVMKeyword<CConv>(parser, LLVM::CConv::C)));

  // Parse a function pointer for indirect calls.
  if (parseOptionalCallFuncPtr(parser, operands))
    return failure();
  bool isDirect = operands.empty();

  // Parse a function identifier for direct calls.
  if (isDirect && parser.parseAttribute(funcAttr, "callee", result.attributes))
    return failure();

  // Parse the function arguments.
  if (parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      parser.parseKeyword("to") ||
      parser.parseSuccessorAndUseList(normalDest, normalOperands) ||
      parser.parseKeyword("unwind") ||
      parser.parseSuccessorAndUseList(unwindDest, unwindOperands))
    return failure();

  bool isVarArg = parser.parseOptionalKeyword("vararg").succeeded();
  if (isVarArg) {
    StringAttr varCalleeTypeAttrName =
        InvokeOp::getVarCalleeTypeAttrName(result.name);
    if (parser.parseLParen().failed() ||
        parser
            .parseAttribute(varCalleeType, varCalleeTypeAttrName,
                            result.attributes)
            .failed() ||
        parser.parseRParen().failed())
      return failure();
  }

  SMLoc opBundlesLoc = parser.getCurrentLocation();
  if (std::optional<ParseResult> result = parseOpBundles(
          parser, opBundleOperands, opBundleOperandTypes, opBundleTags);
      result && failed(*result))
    return failure();
  if (opBundleTags && !opBundleTags.empty())
    result.addAttribute(
        InvokeOp::getOpBundleTagsAttrName(result.name).getValue(),
        opBundleTags);

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse the trailing type list and resolve the function operands.
  SmallVector<DictionaryAttr> argAttrs;
  SmallVector<DictionaryAttr> resultAttrs;
  if (parseCallTypeAndResolveOperands(parser, result, isDirect, operands,
                                      argAttrs, resultAttrs))
    return failure();
  call_interface_impl::addArgAndResultAttrs(
      parser.getBuilder(), result, argAttrs, resultAttrs,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));

  result.addSuccessors({normalDest, unwindDest});
  result.addOperands(normalOperands);
  result.addOperands(unwindOperands);

  if (resolveOpBundleOperands(parser, opBundlesLoc, result, opBundleOperands,
                              opBundleOperandTypes,
                              getOpBundleSizesAttrName(result.name)))
    return failure();

  int32_t numOpBundleOperands = 0;
  for (const auto &operands : opBundleOperands)
    numOpBundleOperands += operands.size();

  result.addAttribute(
      InvokeOp::getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(operands.size()),
                                    static_cast<int32_t>(normalOperands.size()),
                                    static_cast<int32_t>(unwindOperands.size()),
                                    numOpBundleOperands}));
  return success();
}

LLVMFunctionType InvokeOp::getCalleeFunctionType() {
  if (std::optional<LLVMFunctionType> varCalleeType = getVarCalleeType())
    return *varCalleeType;
  return getLLVMFuncType(getContext(), getResultTypes(), getArgOperands());
}

///===----------------------------------------------------------------------===//
/// Verifying/Printing/Parsing for LLVM::LandingpadOp.
///===----------------------------------------------------------------------===//

LogicalResult LandingpadOp::verify() {
  Value value;
  if (LLVMFuncOp func = (*this)->getParentOfType<LLVMFuncOp>()) {
    if (!func.getPersonality())
      return emitError(
          "llvm.landingpad needs to be in a function with a personality");
  }

  // Consistency of llvm.landingpad result types is checked in
  // LLVMFuncOp::verify().

  if (!getCleanup() && getOperands().empty())
    return emitError("landingpad instruction expects at least one clause or "
                     "cleanup attribute");

  for (unsigned idx = 0, ie = getNumOperands(); idx < ie; idx++) {
    value = getOperand(idx);
    bool isFilter = llvm::isa<LLVMArrayType>(value.getType());
    if (isFilter) {
      // FIXME: Verify filter clauses when arrays are appropriately handled
    } else {
      // catch - global addresses only.
      // Bitcast ops should have global addresses as their args.
      if (auto bcOp = value.getDefiningOp<BitcastOp>()) {
        if (auto addrOp = bcOp.getArg().getDefiningOp<AddressOfOp>())
          continue;
        return emitError("constant clauses expected").attachNote(bcOp.getLoc())
               << "global addresses expected as operand to "
                  "bitcast used in clauses for landingpad";
      }
      // ZeroOp and AddressOfOp allowed
      if (value.getDefiningOp<ZeroOp>())
        continue;
      if (value.getDefiningOp<AddressOfOp>())
        continue;
      return emitError("clause #")
             << idx << " is not a known constant - null, addressof, bitcast";
    }
  }
  return success();
}

void LandingpadOp::print(OpAsmPrinter &p) {
  p << (getCleanup() ? " cleanup " : " ");

  // Clauses
  for (auto value : getOperands()) {
    // Similar to llvm - if clause is an array type then it is filter
    // clause else catch clause
    bool isArrayTy = llvm::isa<LLVMArrayType>(value.getType());
    p << '(' << (isArrayTy ? "filter " : "catch ") << value << " : "
      << value.getType() << ") ";
  }

  p.printOptionalAttrDict(getAttrsForPrinting(*this).getAttrs(), {"cleanup"});

  p << ": " << getType();
}

// <operation> ::= `llvm.landingpad` `cleanup`?
//                 ((`catch` | `filter`) operand-type ssa-use)* attribute-dict?
ParseResult LandingpadOp::parse(OpAsmParser &parser, OperationState &result) {
  // Check for cleanup
  if (succeeded(parser.parseOptionalKeyword("cleanup")))
    result.addAttribute("cleanup", parser.getBuilder().getUnitAttr());

  // Parse clauses with types
  while (succeeded(parser.parseOptionalLParen()) &&
         (succeeded(parser.parseOptionalKeyword("filter")) ||
          succeeded(parser.parseOptionalKeyword("catch")))) {
    OpAsmParser::UnresolvedOperand operand;
    Type ty;
    if (parser.parseOperand(operand) || parser.parseColon() ||
        parser.parseType(ty) ||
        parser.resolveOperand(operand, ty, result.operands) ||
        parser.parseRParen())
      return failure();
  }

  Type type;
  if (parser.parseColon() || parser.parseType(type))
    return failure();

  result.addTypes(type);
  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto parent = (*this)->getParentOfType<LLVMFuncOp>();
  if (!parent)
    return success();

  Type expectedType = parent.getFunctionType().getReturnType();
  if (llvm::isa<LLVMVoidType>(expectedType)) {
    if (!getArg())
      return success();
    InFlightDiagnostic diag = emitOpError("expected no operands");
    diag.attachNote(parent->getLoc()) << "when returning from function";
    return diag;
  }
  if (!getArg()) {
    if (llvm::isa<LLVMVoidType>(expectedType))
      return success();
    InFlightDiagnostic diag = emitOpError("expected 1 operand");
    diag.attachNote(parent->getLoc()) << "when returning from function";
    return diag;
  }
  if (expectedType != getArg().getType()) {
    InFlightDiagnostic diag = emitOpError("mismatching result types");
    diag.attachNote(parent->getLoc()) << "when returning from function";
    return diag;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CallIntrinsicOp
//===----------------------------------------------------------------------===//

LogicalResult CallIntrinsicOp::verify() {
  if (!getIntrin().starts_with("llvm."))
    return emitOpError() << "intrinsic name must start with 'llvm.'";
  if (failed(verifyOperandBundles(*this)))
    return failure();
  return success();
}

void CallIntrinsicOp::build(OpBuilder &builder, OperationState &state,
                            mlir::StringAttr intrin, mlir::ValueRange args) {
  build(builder, state, /*resultTypes=*/TypeRange{}, intrin, args,
        FastmathFlagsAttr{},
        /*op_bundle_operands=*/{}, /*op_bundle_tags=*/{}, /*arg_attrs=*/{},
        /*res_attrs=*/{});
}

void CallIntrinsicOp::build(OpBuilder &builder, OperationState &state,
                            mlir::StringAttr intrin, mlir::ValueRange args,
                            mlir::LLVM::FastmathFlagsAttr fastMathFlags) {
  build(builder, state, /*resultTypes=*/TypeRange{}, intrin, args,
        fastMathFlags,
        /*op_bundle_operands=*/{}, /*op_bundle_tags=*/{}, /*arg_attrs=*/{},
        /*res_attrs=*/{});
}

void CallIntrinsicOp::build(OpBuilder &builder, OperationState &state,
                            mlir::Type resultType, mlir::StringAttr intrin,
                            mlir::ValueRange args) {
  build(builder, state, {resultType}, intrin, args, FastmathFlagsAttr{},
        /*op_bundle_operands=*/{}, /*op_bundle_tags=*/{}, /*arg_attrs=*/{},
        /*res_attrs=*/{});
}

void CallIntrinsicOp::build(OpBuilder &builder, OperationState &state,
                            mlir::TypeRange resultTypes,
                            mlir::StringAttr intrin, mlir::ValueRange args,
                            mlir::LLVM::FastmathFlagsAttr fastMathFlags) {
  build(builder, state, resultTypes, intrin, args, fastMathFlags,
        /*op_bundle_operands=*/{}, /*op_bundle_tags=*/{}, /*arg_attrs=*/{},
        /*res_attrs=*/{});
}

ParseResult CallIntrinsicOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  StringAttr intrinAttr;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  SmallVector<SmallVector<OpAsmParser::UnresolvedOperand>> opBundleOperands;
  SmallVector<SmallVector<Type>> opBundleOperandTypes;
  ArrayAttr opBundleTags;

  // Parse intrinsic name.
  if (parser.parseCustomAttributeWithFallback(
          intrinAttr, parser.getBuilder().getType<NoneType>()))
    return failure();
  result.addAttribute(CallIntrinsicOp::getIntrinAttrName(result.name),
                      intrinAttr);

  if (parser.parseLParen())
    return failure();

  // Parse the function arguments.
  if (parser.parseOperandList(operands))
    return mlir::failure();

  if (parser.parseRParen())
    return mlir::failure();

  // Handle bundles.
  SMLoc opBundlesLoc = parser.getCurrentLocation();
  if (std::optional<ParseResult> result = parseOpBundles(
          parser, opBundleOperands, opBundleOperandTypes, opBundleTags);
      result && failed(*result))
    return failure();
  if (opBundleTags && !opBundleTags.empty())
    result.addAttribute(
        CallIntrinsicOp::getOpBundleTagsAttrName(result.name).getValue(),
        opBundleTags);

  if (parser.parseOptionalAttrDict(result.attributes))
    return mlir::failure();

  SmallVector<DictionaryAttr> argAttrs;
  SmallVector<DictionaryAttr> resultAttrs;
  if (parseCallTypeAndResolveOperands(parser, result, /*isDirect=*/true,
                                      operands, argAttrs, resultAttrs))
    return failure();
  call_interface_impl::addArgAndResultAttrs(
      parser.getBuilder(), result, argAttrs, resultAttrs,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));

  if (resolveOpBundleOperands(parser, opBundlesLoc, result, opBundleOperands,
                              opBundleOperandTypes,
                              getOpBundleSizesAttrName(result.name)))
    return failure();

  int32_t numOpBundleOperands = 0;
  for (const auto &operands : opBundleOperands)
    numOpBundleOperands += operands.size();

  result.addAttribute(
      CallIntrinsicOp::getOperandSegmentSizeAttr(),
      parser.getBuilder().getDenseI32ArrayAttr(
          {static_cast<int32_t>(operands.size()), numOpBundleOperands}));

  return mlir::success();
}

void CallIntrinsicOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printAttributeWithoutType(getIntrinAttr());

  OperandRange args = getArgs();
  p << "(" << args << ")";

  // Operand bundles.
  if (!getOpBundleOperands().empty()) {
    p << ' ';
    printOpBundles(p, *this, getOpBundleOperands(),
                   getOpBundleOperands().getTypes(), getOpBundleTagsAttr());
  }

  p.printOptionalAttrDict(processFMFAttr(getAttrsForPrinting(*this).getAttrs()),
                          {getOperandSegmentSizesAttrName(),
                           getOpBundleSizesAttrName(), getIntrinAttrName(),
                           getOpBundleTagsAttrName(), getArgAttrsAttrName(),
                           getResAttrsAttrName()});

  p << " : ";

  // Reconstruct the MLIR function type from operand and result types.
  call_interface_impl::printFunctionSignature(
      p, args.getTypes(), getArgAttrsAttr(),
      /*isVariadic=*/false, getResultTypes(), getResAttrsAttr());
}

//===----------------------------------------------------------------------===//
// InlineAsmOp
//===----------------------------------------------------------------------===//

void InlineAsmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (getHasSideEffects()) {
    effects.emplace_back(MemoryEffects::Write::get());
    effects.emplace_back(MemoryEffects::Read::get());
  }
}

//===----------------------------------------------------------------------===//
// LLVM::IndirectBrOp
//===----------------------------------------------------------------------===//

SuccessorOperands IndirectBrOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return SuccessorOperands(getSuccOperandsMutable()[index]);
}

void IndirectBrOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                         Value addr, ArrayRef<ValueRange> succOperands,
                         BlockRange successors) {
  odsState.addOperands(addr);
  for (ValueRange range : succOperands)
    odsState.addOperands(range);
  SmallVector<int32_t> rangeSegments;
  for (ValueRange range : succOperands)
    rangeSegments.push_back(range.size());
  odsState.getOrAddProperties<Properties>().indbr_operand_segments =
      odsBuilder.getDenseI32ArrayAttr(rangeSegments);
  odsState.addSuccessors(successors);
}

ParseResult mlir::LLVM::parseIndirectBrOpSucessors(
    OpAsmParser &parser, Type &flagType,
    SmallVectorImpl<Block *> &succOperandBlocks,
    SmallVectorImpl<SmallVector<OpAsmParser::UnresolvedOperand>> &succOperands,
    SmallVectorImpl<SmallVector<Type>> &succOperandsTypes) {
  if (failed(parser.parseCommaSeparatedList(
          OpAsmParser::Delimiter::Square,
          [&]() {
            Block *destination = nullptr;
            SmallVector<OpAsmParser::UnresolvedOperand> operands;
            SmallVector<Type> operandTypes;

            if (parser.parseSuccessor(destination).failed())
              return failure();

            if (succeeded(parser.parseOptionalLParen())) {
              if (failed(parser.parseOperandList(
                      operands, OpAsmParser::Delimiter::None)) ||
                  failed(parser.parseColonTypeList(operandTypes)) ||
                  failed(parser.parseRParen()))
                return failure();
            }
            succOperandBlocks.push_back(destination);
            succOperands.emplace_back(operands);
            succOperandsTypes.emplace_back(operandTypes);
            return success();
          },
          "successor blocks")))
    return failure();
  return success();
}

void mlir::LLVM::printIndirectBrOpSucessors(
    OpAsmPrinter &p, IndirectBrOp op, Type flagType, SuccessorRange succs,
    OperandRangeRange succOperands, const TypeRangeRange &succOperandsTypes) {
  p << "[";
  llvm::interleave(
      llvm::zip(succs, succOperands),
      [&](auto i) {
        p.printNewline();
        p.printSuccessorAndUseList(std::get<0>(i), std::get<1>(i));
      },
      [&] { p << ','; });
  if (!succOperands.empty())
    p.printNewline();
  p << "]";
}

//===----------------------------------------------------------------------===//
// InlineAsmOp
//===----------------------------------------------------------------------===//

LogicalResult InlineAsmOp::verify() {
  if (!getTailCallKindAttr())
    return success();

  if (getTailCallKindAttr().getTailCallKind() == TailCallKind::MustTail)
    return emitOpError(
        "tail call kind 'musttail' is not supported by this operation");

  return success();
}

SmallVector<Value> mlir::LLVM::CallOp::getAccessedOperands() {
  // Note: This must not use `getArgOperands`, which excludes the variadic
  // arguments of a call to a variadic callee. Those are passed to the callee
  // and may well be accessed by it.
  Operation::operand_range operands = getCalleeOperands();
  // In an indirect call, the first callee operand is the callee itself.
  if (!getCallee().has_value() && !operands.empty())
    operands = operands.drop_front();
  return llvm::filter_to_vector(
      operands, [](Value arg) { return isa<LLVMPointerType>(arg.getType()); });
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMControlFlowOps.cpp.inc"
