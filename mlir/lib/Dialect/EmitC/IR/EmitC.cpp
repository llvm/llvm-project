//===- EmitC.cpp - EmitC Dialect ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/IR/EmitCTraits.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::emitc;

#include "mlir/Dialect/EmitC/IR/EmitCDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// EmitCDialect
//===----------------------------------------------------------------------===//

void EmitCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/EmitC/IR/EmitC.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/EmitC/IR/EmitCTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/EmitC/IR/EmitCAttributes.cpp.inc"
      >();
}

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
Operation *EmitCDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  return builder.create<emitc::ConstantOp>(loc, type, value);
}

/// Default callback for builders of ops carrying a region. Inserts a yield
/// without arguments.
void mlir::emitc::buildTerminatedBody(OpBuilder &builder, Location loc) {
  builder.create<emitc::YieldOp>(loc);
}

bool mlir::emitc::isSupportedEmitCType(Type type) {
  if (llvm::isa<emitc::OpaqueType>(type))
    return true;
  if (auto ptrType = llvm::dyn_cast<emitc::PointerType>(type))
    return isSupportedEmitCType(ptrType.getPointee());
  if (auto arrayType = llvm::dyn_cast<emitc::ArrayType>(type)) {
    auto elemType = arrayType.getElementType();
    return !llvm::isa<emitc::ArrayType>(elemType) &&
           isSupportedEmitCType(elemType);
  }
  if (type.isIndex() || emitc::isPointerWideType(type))
    return true;
  if (llvm::isa<IntegerType>(type))
    return isSupportedIntegerType(type);
  if (llvm::isa<FloatType>(type))
    return isSupportedFloatType(type);
  if (auto tensorType = llvm::dyn_cast<TensorType>(type)) {
    if (!tensorType.hasStaticShape()) {
      return false;
    }
    auto elemType = tensorType.getElementType();
    if (llvm::isa<emitc::ArrayType>(elemType)) {
      return false;
    }
    return isSupportedEmitCType(elemType);
  }
  if (auto tupleType = llvm::dyn_cast<TupleType>(type)) {
    return llvm::all_of(tupleType.getTypes(), [](Type type) {
      return !llvm::isa<emitc::ArrayType>(type) && isSupportedEmitCType(type);
    });
  }
  return false;
}

bool mlir::emitc::isSupportedIntegerType(Type type) {
  if (auto intType = llvm::dyn_cast<IntegerType>(type)) {
    switch (intType.getWidth()) {
    case 1:
    case 8:
    case 16:
    case 32:
    case 64:
      return true;
    default:
      return false;
    }
  }
  return false;
}

bool mlir::emitc::isIntegerIndexOrOpaqueType(Type type) {
  return llvm::isa<IndexType, emitc::OpaqueType>(type) ||
         isSupportedIntegerType(type) || isPointerWideType(type);
}

bool mlir::emitc::isSupportedFloatType(Type type) {
  if (auto floatType = llvm::dyn_cast<FloatType>(type)) {
    switch (floatType.getWidth()) {
    case 16: {
      if (llvm::isa<Float16Type, BFloat16Type>(type))
        return true;
      return false;
    }
    case 32:
    case 64:
      return true;
    default:
      return false;
    }
  }
  return false;
}

bool mlir::emitc::isPointerWideType(Type type) {
  return isa<emitc::SignedSizeTType, emitc::SizeTType, emitc::PtrDiffTType>(
      type);
}

/// Check that the type of the initial value is compatible with the operations
/// result type.
static LogicalResult verifyInitializationAttribute(Operation *op,
                                                   Attribute value) {
  assert(op->getNumResults() == 1 && "operation must have 1 result");

  if (llvm::isa<emitc::OpaqueAttr>(value))
    return success();

  if (llvm::isa<StringAttr>(value))
    return op->emitOpError()
           << "string attributes are not supported, use #emitc.opaque instead";

  Type resultType = op->getResult(0).getType();
  if (auto lType = dyn_cast<LValueType>(resultType))
    resultType = lType.getValueType();
  Type attrType = cast<TypedAttr>(value).getType();

  if (isPointerWideType(resultType) && attrType.isIndex())
    return success();

  if (resultType != attrType)
    return op->emitOpError()
           << "requires attribute to either be an #emitc.opaque attribute or "
              "it's type ("
           << attrType << ") to match the op's result type (" << resultType
           << ")";

  return success();
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

LogicalResult AddOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();

  if (isa<emitc::PointerType>(lhsType) && isa<emitc::PointerType>(rhsType))
    return emitOpError("requires that at most one operand is a pointer");

  if ((isa<emitc::PointerType>(lhsType) &&
       !isa<IntegerType, emitc::OpaqueType>(rhsType)) ||
      (isa<emitc::PointerType>(rhsType) &&
       !isa<IntegerType, emitc::OpaqueType>(lhsType)))
    return emitOpError("requires that one operand is an integer or of opaque "
                       "type if the other is a pointer");

  return success();
}

//===----------------------------------------------------------------------===//
// ApplyOp
//===----------------------------------------------------------------------===//

LogicalResult ApplyOp::verify() {
  StringRef applicableOperatorStr = getApplicableOperator();

  // Applicable operator must not be empty.
  if (applicableOperatorStr.empty())
    return emitOpError("applicable operator must not be empty");

  // Only `*` and `&` are supported.
  if (applicableOperatorStr != "&" && applicableOperatorStr != "*")
    return emitOpError("applicable operator is illegal");

  Type operandType = getOperand().getType();
  Type resultType = getResult().getType();
  if (applicableOperatorStr == "&") {
    if (!llvm::isa<emitc::LValueType>(operandType))
      return emitOpError("operand type must be an lvalue when applying `&`");
    if (!llvm::isa<emitc::PointerType>(resultType))
      return emitOpError("result type must be a pointer when applying `&`");
  } else {
    if (!llvm::isa<emitc::PointerType>(operandType))
      return emitOpError("operand type must be a pointer when applying `*`");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AssignOp
//===----------------------------------------------------------------------===//

/// The assign op requires that the assigned value's type matches the
/// assigned-to variable type.
LogicalResult emitc::AssignOp::verify() {
  TypedValue<emitc::LValueType> variable = getVar();

  if (!variable.getDefiningOp())
    return emitOpError() << "cannot assign to block argument";

  Type valueType = getValue().getType();
  Type variableType = variable.getType().getValueType();
  if (variableType != valueType)
    return emitOpError() << "requires value's type (" << valueType
                         << ") to match variable's type (" << variableType
                         << ")\n  variable: " << variable
                         << "\n  value: " << getValue() << "\n";
  return success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  Type input = inputs.front(), output = outputs.front();

  return ((emitc::isIntegerIndexOrOpaqueType(input) ||
           emitc::isSupportedFloatType(input) ||
           isa<emitc::PointerType>(input) || isa<emitc::ArrayType>(input)) &&
          (emitc::isIntegerIndexOrOpaqueType(output) ||
           emitc::isSupportedFloatType(output) ||
           isa<emitc::PointerType>(output)));
}

//===----------------------------------------------------------------------===//
// CallOpaqueOp
//===----------------------------------------------------------------------===//

LogicalResult emitc::CallOpaqueOp::verify() {
  // Callee must not be empty.
  if (getCallee().empty())
    return emitOpError("callee must not be empty");

  if (std::optional<ArrayAttr> argsAttr = getArgs()) {
    for (Attribute arg : *argsAttr) {
      auto intAttr = llvm::dyn_cast<IntegerAttr>(arg);
      if (intAttr && llvm::isa<IndexType>(intAttr.getType())) {
        int64_t index = intAttr.getInt();
        // Args with elements of type index must be in range
        // [0..operands.size).
        if ((index < 0) || (index >= static_cast<int64_t>(getNumOperands())))
          return emitOpError("index argument is out of range");

        // Args with elements of type ArrayAttr must have a type.
      } else if (llvm::isa<ArrayAttr>(
                     arg) /*&& llvm::isa<NoneType>(arg.getType())*/) {
        // FIXME: Array attributes never have types
        return emitOpError("array argument has no type");
      }
    }
  }

  if (std::optional<ArrayAttr> templateArgsAttr = getTemplateArgs()) {
    for (Attribute tArg : *templateArgsAttr) {
      if (!llvm::isa<TypeAttr, IntegerAttr, FloatAttr, emitc::OpaqueAttr>(tArg))
        return emitOpError("template argument has invalid type");
    }
  }

  if (llvm::any_of(getResultTypes(), llvm::IsaPred<ArrayType>)) {
    return emitOpError() << "cannot return array type";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

LogicalResult emitc::ConstantOp::verify() {
  Attribute value = getValueAttr();
  if (failed(verifyInitializationAttribute(getOperation(), value)))
    return failure();
  if (auto opaqueValue = llvm::dyn_cast<emitc::OpaqueAttr>(value)) {
    if (opaqueValue.getValue().empty())
      return emitOpError() << "value must not be empty";
  }
  return success();
}

OpFoldResult emitc::ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

//===----------------------------------------------------------------------===//
// ExpressionOp
//===----------------------------------------------------------------------===//

Operation *ExpressionOp::getRootOp() {
  auto yieldOp = cast<YieldOp>(getBody()->getTerminator());
  Value yieldedValue = yieldOp.getResult();
  Operation *rootOp = yieldedValue.getDefiningOp();
  assert(rootOp && "Yielded value not defined within expression");
  return rootOp;
}

LogicalResult ExpressionOp::verify() {
  Type resultType = getResult().getType();
  Region &region = getRegion();

  Block &body = region.front();

  if (!body.mightHaveTerminator())
    return emitOpError("must yield a value at termination");

  auto yield = cast<YieldOp>(body.getTerminator());
  Value yieldResult = yield.getResult();

  if (!yieldResult)
    return emitOpError("must yield a value at termination");

  Type yieldType = yieldResult.getType();

  if (resultType != yieldType)
    return emitOpError("requires yielded type to match return type");

  for (Operation &op : region.front().without_terminator()) {
    if (!op.hasTrait<OpTrait::emitc::CExpression>())
      return emitOpError("contains an unsupported operation");
    if (op.getNumResults() != 1)
      return emitOpError("requires exactly one result for each operation");
    if (!op.getResult(0).hasOneUse())
      return emitOpError("requires exactly one use for each operation");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

void ForOp::build(OpBuilder &builder, OperationState &result, Value lb,
                  Value ub, Value step, BodyBuilderFn bodyBuilder) {
  OpBuilder::InsertionGuard g(builder);
  result.addOperands({lb, ub, step});
  Type t = lb.getType();
  Region *bodyRegion = result.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion);
  bodyBlock->addArgument(t, result.location);

  // Create the default terminator if the builder is not provided.
  if (!bodyBuilder) {
    ForOp::ensureTerminator(*bodyRegion, builder, result.location);
  } else {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilder(builder, result.location, bodyBlock->getArgument(0));
  }
}

void ForOp::getCanonicalizationPatterns(RewritePatternSet &, MLIRContext *) {}

ParseResult ForOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder &builder = parser.getBuilder();
  Type type;

  OpAsmParser::Argument inductionVariable;
  OpAsmParser::UnresolvedOperand lb, ub, step;

  // Parse the induction variable followed by '='.
  if (parser.parseOperand(inductionVariable.ssaName) || parser.parseEqual() ||
      // Parse loop bounds.
      parser.parseOperand(lb) || parser.parseKeyword("to") ||
      parser.parseOperand(ub) || parser.parseKeyword("step") ||
      parser.parseOperand(step))
    return failure();

  // Parse the optional initial iteration arguments.
  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  regionArgs.push_back(inductionVariable);

  // Parse optional type, else assume Index.
  if (parser.parseOptionalColon())
    type = builder.getIndexType();
  else if (parser.parseType(type))
    return failure();

  // Resolve input operands.
  regionArgs.front().type = type;
  if (parser.resolveOperand(lb, type, result.operands) ||
      parser.resolveOperand(ub, type, result.operands) ||
      parser.resolveOperand(step, type, result.operands))
    return failure();

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();

  ForOp::ensureTerminator(*body, builder, result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

void ForOp::print(OpAsmPrinter &p) {
  p << " " << getInductionVar() << " = " << getLowerBound() << " to "
    << getUpperBound() << " step " << getStep();

  p << ' ';
  if (Type t = getInductionVar().getType(); !t.isIndex())
    p << " : " << t << ' ';
  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
  p.printOptionalAttrDict((*this)->getAttrs());
}

LogicalResult ForOp::verifyRegions() {
  // Check that the body defines as single block argument for the induction
  // variable.
  if (getInductionVar().getType() != getLowerBound().getType())
    return emitOpError(
        "expected induction variable to be same type as bounds and step");

  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }

  return success();
}

FunctionType CallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

//===----------------------------------------------------------------------===//
// DeclareFuncOp
//===----------------------------------------------------------------------===//

LogicalResult
DeclareFuncOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the sym_name attribute was specified.
  auto fnAttr = getSymNameAttr();
  if (!fnAttr)
    return emitOpError("requires a 'sym_name' symbol reference attribute");
  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  return success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  call_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/std::nullopt,
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

LogicalResult FuncOp::verify() {
  if (llvm::any_of(getArgumentTypes(), llvm::IsaPred<LValueType>)) {
    return emitOpError("cannot have lvalue type as argument");
  }

  if (getNumResults() > 1)
    return emitOpError("requires zero or exactly one result, but has ")
           << getNumResults();

  if (getNumResults() == 1 && isa<ArrayType>(getResultTypes()[0]))
    return emitOpError("cannot return array type");

  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto function = cast<FuncOp>((*this)->getParentOp());

  // The operand number and types must match the function signature.
  if (getNumOperands() != function.getNumResults())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function (@"
           << function.getName() << ") returns " << function.getNumResults();

  if (function.getNumResults() == 1)
    if (getOperand().getType() != function.getResultTypes()[0])
      return emitError() << "type of the return operand ("
                         << getOperand().getType()
                         << ") doesn't match function result type ("
                         << function.getResultTypes()[0] << ")"
                         << " in function @" << function.getName();
  return success();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

void IfOp::build(OpBuilder &builder, OperationState &result, Value cond,
                 bool addThenBlock, bool addElseBlock) {
  assert((!addElseBlock || addThenBlock) &&
         "must not create else block w/o then block");
  result.addOperands(cond);

  // Add regions and blocks.
  OpBuilder::InsertionGuard guard(builder);
  Region *thenRegion = result.addRegion();
  if (addThenBlock)
    builder.createBlock(thenRegion);
  Region *elseRegion = result.addRegion();
  if (addElseBlock)
    builder.createBlock(elseRegion);
}

void IfOp::build(OpBuilder &builder, OperationState &result, Value cond,
                 bool withElseRegion) {
  result.addOperands(cond);

  // Build then region.
  OpBuilder::InsertionGuard guard(builder);
  Region *thenRegion = result.addRegion();
  builder.createBlock(thenRegion);

  // Build else region.
  Region *elseRegion = result.addRegion();
  if (withElseRegion) {
    builder.createBlock(elseRegion);
  }
}

void IfOp::build(OpBuilder &builder, OperationState &result, Value cond,
                 function_ref<void(OpBuilder &, Location)> thenBuilder,
                 function_ref<void(OpBuilder &, Location)> elseBuilder) {
  assert(thenBuilder && "the builder callback for 'then' must be present");
  result.addOperands(cond);

  // Build then region.
  OpBuilder::InsertionGuard guard(builder);
  Region *thenRegion = result.addRegion();
  builder.createBlock(thenRegion);
  thenBuilder(builder, result.location);

  // Build else region.
  Region *elseRegion = result.addRegion();
  if (elseBuilder) {
    builder.createBlock(elseRegion);
    elseBuilder(builder, result.location);
  }
}

ParseResult IfOp::parse(OpAsmParser &parser, OperationState &result) {
  // Create the regions for 'then'.
  result.regions.reserve(2);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();

  Builder &builder = parser.getBuilder();
  OpAsmParser::UnresolvedOperand cond;
  Type i1Type = builder.getIntegerType(1);
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, i1Type, result.operands))
    return failure();
  // Parse the 'then' region.
  if (parser.parseRegion(*thenRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();
  IfOp::ensureTerminator(*thenRegion, parser.getBuilder(), result.location);

  // If we find an 'else' keyword then parse the 'else' region.
  if (!parser.parseOptionalKeyword("else")) {
    if (parser.parseRegion(*elseRegion, /*arguments=*/{}, /*argTypes=*/{}))
      return failure();
    IfOp::ensureTerminator(*elseRegion, parser.getBuilder(), result.location);
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void IfOp::print(OpAsmPrinter &p) {
  bool printBlockTerminators = false;

  p << " " << getCondition();
  p << ' ';
  p.printRegion(getThenRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/printBlockTerminators);

  // Print the 'else' regions if it exists and has a block.
  Region &elseRegion = getElseRegion();
  if (!elseRegion.empty()) {
    p << " else ";
    p.printRegion(elseRegion,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/printBlockTerminators);
  }

  p.printOptionalAttrDict((*this)->getAttrs());
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void IfOp::getSuccessorRegions(RegionBranchPoint point,
                               SmallVectorImpl<RegionSuccessor> &regions) {
  // The `then` and the `else` region branch back to the parent operation.
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor());
    return;
  }

  regions.push_back(RegionSuccessor(&getThenRegion()));

  // Don't consider the else region if it is empty.
  Region *elseRegion = &this->getElseRegion();
  if (elseRegion->empty())
    regions.push_back(RegionSuccessor());
  else
    regions.push_back(RegionSuccessor(elseRegion));
}

void IfOp::getEntrySuccessorRegions(ArrayRef<Attribute> operands,
                                    SmallVectorImpl<RegionSuccessor> &regions) {
  FoldAdaptor adaptor(operands, *this);
  auto boolAttr = dyn_cast_or_null<BoolAttr>(adaptor.getCondition());
  if (!boolAttr || boolAttr.getValue())
    regions.emplace_back(&getThenRegion());

  // If the else region is empty, execution continues after the parent op.
  if (!boolAttr || !boolAttr.getValue()) {
    if (!getElseRegion().empty())
      regions.emplace_back(&getElseRegion());
    else
      regions.emplace_back();
  }
}

void IfOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<InvocationBounds> &invocationBounds) {
  if (auto cond = llvm::dyn_cast_or_null<BoolAttr>(operands[0])) {
    // If the condition is known, then one region is known to be executed once
    // and the other zero times.
    invocationBounds.emplace_back(0, cond.getValue() ? 1 : 0);
    invocationBounds.emplace_back(0, cond.getValue() ? 0 : 1);
  } else {
    // Non-constant condition. Each region may be executed 0 or 1 times.
    invocationBounds.assign(2, {0, 1});
  }
}

//===----------------------------------------------------------------------===//
// IncludeOp
//===----------------------------------------------------------------------===//

void IncludeOp::print(OpAsmPrinter &p) {
  bool standardInclude = getIsStandardInclude();

  p << " ";
  if (standardInclude)
    p << "<";
  p << "\"" << getInclude() << "\"";
  if (standardInclude)
    p << ">";
}

ParseResult IncludeOp::parse(OpAsmParser &parser, OperationState &result) {
  bool standardInclude = !parser.parseOptionalLess();

  StringAttr include;
  OptionalParseResult includeParseResult =
      parser.parseOptionalAttribute(include, "include", result.attributes);
  if (!includeParseResult.has_value())
    return parser.emitError(parser.getNameLoc()) << "expected string attribute";

  if (standardInclude && parser.parseOptionalGreater())
    return parser.emitError(parser.getNameLoc())
           << "expected trailing '>' for standard include";

  if (standardInclude)
    result.addAttribute("is_standard_include",
                        UnitAttr::get(parser.getContext()));

  return success();
}

//===----------------------------------------------------------------------===//
// LiteralOp
//===----------------------------------------------------------------------===//

/// The literal op requires a non-empty value.
LogicalResult emitc::LiteralOp::verify() {
  if (getValue().empty())
    return emitOpError() << "value must not be empty";
  return success();
}
//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

LogicalResult SubOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();
  Type resultType = getResult().getType();

  if (isa<emitc::PointerType>(rhsType) && !isa<emitc::PointerType>(lhsType))
    return emitOpError("rhs can only be a pointer if lhs is a pointer");

  if (isa<emitc::PointerType>(lhsType) &&
      !isa<IntegerType, emitc::OpaqueType, emitc::PointerType>(rhsType))
    return emitOpError("requires that rhs is an integer, pointer or of opaque "
                       "type if lhs is a pointer");

  if (isa<emitc::PointerType>(lhsType) && isa<emitc::PointerType>(rhsType) &&
      !isa<IntegerType, emitc::PtrDiffTType, emitc::OpaqueType>(resultType))
    return emitOpError("requires that the result is an integer, ptrdiff_t or "
                       "of opaque type if lhs and rhs are pointers");
  return success();
}

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

LogicalResult emitc::VariableOp::verify() {
  return verifyInitializationAttribute(getOperation(), getValueAttr());
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

LogicalResult emitc::YieldOp::verify() {
  Value result = getResult();
  Operation *containingOp = getOperation()->getParentOp();

  if (result && containingOp->getNumResults() != 1)
    return emitOpError() << "yields a value not returned by parent";

  if (!result && containingOp->getNumResults() != 0)
    return emitOpError() << "does not yield a value to be returned by parent";

  return success();
}

//===----------------------------------------------------------------------===//
// SubscriptOp
//===----------------------------------------------------------------------===//

LogicalResult emitc::SubscriptOp::verify() {
  // Checks for array operand.
  if (auto arrayType = llvm::dyn_cast<emitc::ArrayType>(getValue().getType())) {
    // Check number of indices.
    if (getIndices().size() != (size_t)arrayType.getRank()) {
      return emitOpError() << "on array operand requires number of indices ("
                           << getIndices().size()
                           << ") to match the rank of the array type ("
                           << arrayType.getRank() << ")";
    }
    // Check types of index operands.
    for (unsigned i = 0, e = getIndices().size(); i != e; ++i) {
      Type type = getIndices()[i].getType();
      if (!isIntegerIndexOrOpaqueType(type)) {
        return emitOpError() << "on array operand requires index operand " << i
                             << " to be integer-like, but got " << type;
      }
    }
    // Check element type.
    Type elementType = arrayType.getElementType();
    Type resultType = getType().getValueType();
    if (elementType != resultType) {
      return emitOpError() << "on array operand requires element type ("
                           << elementType << ") and result type (" << resultType
                           << ") to match";
    }
    return success();
  }

  // Checks for pointer operand.
  if (auto pointerType =
          llvm::dyn_cast<emitc::PointerType>(getValue().getType())) {
    // Check number of indices.
    if (getIndices().size() != 1) {
      return emitOpError()
             << "on pointer operand requires one index operand, but got "
             << getIndices().size();
    }
    // Check types of index operand.
    Type type = getIndices()[0].getType();
    if (!isIntegerIndexOrOpaqueType(type)) {
      return emitOpError() << "on pointer operand requires index operand to be "
                              "integer-like, but got "
                           << type;
    }
    // Check pointee type.
    Type pointeeType = pointerType.getPointee();
    Type resultType = getType().getValueType();
    if (pointeeType != resultType) {
      return emitOpError() << "on pointer operand requires pointee type ("
                           << pointeeType << ") and result type (" << resultType
                           << ") to match";
    }
    return success();
  }

  // The operand has opaque type, so we can't assume anything about the number
  // or types of index operands.
  return success();
}

//===----------------------------------------------------------------------===//
// EmitC Enums
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/EmitC/IR/EmitCEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// EmitC Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/EmitC/IR/EmitCAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// EmitC Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/EmitC/IR/EmitCTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

Type emitc::ArrayType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  SmallVector<int64_t, 4> dimensions;
  if (parser.parseDimensionList(dimensions, /*allowDynamic=*/false,
                                /*withTrailingX=*/true))
    return Type();
  // Parse the element type.
  auto typeLoc = parser.getCurrentLocation();
  Type elementType;
  if (parser.parseType(elementType))
    return Type();

  // Check that array is formed from allowed types.
  if (!isValidElementType(elementType))
    return parser.emitError(typeLoc, "invalid array element type"), Type();
  if (parser.parseGreater())
    return Type();
  return parser.getChecked<ArrayType>(dimensions, elementType);
}

void emitc::ArrayType::print(AsmPrinter &printer) const {
  printer << "<";
  for (int64_t dim : getShape()) {
    printer << dim << 'x';
  }
  printer.printType(getElementType());
  printer << ">";
}

LogicalResult emitc::ArrayType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<int64_t> shape, Type elementType) {
  if (shape.empty())
    return emitError() << "shape must not be empty";

  for (int64_t dim : shape) {
    if (dim < 0)
      return emitError() << "dimensions must have non-negative size";
  }

  if (!elementType)
    return emitError() << "element type must not be none";

  if (!isValidElementType(elementType))
    return emitError() << "invalid array element type";

  return success();
}

emitc::ArrayType
emitc::ArrayType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                            Type elementType) const {
  if (!shape)
    return emitc::ArrayType::get(getShape(), elementType);
  return emitc::ArrayType::get(*shape, elementType);
}

//===----------------------------------------------------------------------===//
// LValueType
//===----------------------------------------------------------------------===//

LogicalResult mlir::emitc::LValueType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type value) {
  // Check that the wrapped type is valid. This especially forbids nested lvalue
  // types.
  if (!isSupportedEmitCType(value))
    return emitError()
           << "!emitc.lvalue must wrap supported emitc type, but got " << value;

  if (llvm::isa<emitc::ArrayType>(value))
    return emitError() << "!emitc.lvalue cannot wrap !emitc.array type";

  return success();
}

//===----------------------------------------------------------------------===//
// OpaqueType
//===----------------------------------------------------------------------===//

LogicalResult mlir::emitc::OpaqueType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef value) {
  if (value.empty()) {
    return emitError() << "expected non empty string in !emitc.opaque type";
  }
  if (value.back() == '*') {
    return emitError() << "pointer not allowed as outer type with "
                          "!emitc.opaque, use !emitc.ptr instead";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

LogicalResult mlir::emitc::PointerType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, Type value) {
  if (llvm::isa<emitc::LValueType>(value))
    return emitError() << "pointers to lvalues are not allowed";

  return success();
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//
static void printEmitCGlobalOpTypeAndInitialValue(OpAsmPrinter &p, GlobalOp op,
                                                  TypeAttr type,
                                                  Attribute initialValue) {
  p << type;
  if (initialValue) {
    p << " = ";
    p.printAttributeWithoutType(initialValue);
  }
}

static Type getInitializerTypeForGlobal(Type type) {
  if (auto array = llvm::dyn_cast<ArrayType>(type))
    return RankedTensorType::get(array.getShape(), array.getElementType());
  return type;
}

static ParseResult
parseEmitCGlobalOpTypeAndInitialValue(OpAsmParser &parser, TypeAttr &typeAttr,
                                      Attribute &initialValue) {
  Type type;
  if (parser.parseType(type))
    return failure();

  typeAttr = TypeAttr::get(type);

  if (parser.parseOptionalEqual())
    return success();

  if (parser.parseAttribute(initialValue, getInitializerTypeForGlobal(type)))
    return failure();

  if (!llvm::isa<ElementsAttr, IntegerAttr, FloatAttr, emitc::OpaqueAttr>(
          initialValue))
    return parser.emitError(parser.getNameLoc())
           << "initial value should be a integer, float, elements or opaque "
              "attribute";
  return success();
}

LogicalResult GlobalOp::verify() {
  if (!isSupportedEmitCType(getType())) {
    return emitOpError("expected valid emitc type");
  }
  if (getInitialValue().has_value()) {
    Attribute initValue = getInitialValue().value();
    // Check that the type of the initial value is compatible with the type of
    // the global variable.
    if (auto elementsAttr = llvm::dyn_cast<ElementsAttr>(initValue)) {
      auto arrayType = llvm::dyn_cast<ArrayType>(getType());
      if (!arrayType)
        return emitOpError("expected array type, but got ") << getType();

      Type initType = elementsAttr.getType();
      Type tensorType = getInitializerTypeForGlobal(getType());
      if (initType != tensorType) {
        return emitOpError("initial value expected to be of type ")
               << getType() << ", but was of type " << initType;
      }
    } else if (auto intAttr = dyn_cast<IntegerAttr>(initValue)) {
      if (intAttr.getType() != getType()) {
        return emitOpError("initial value expected to be of type ")
               << getType() << ", but was of type " << intAttr.getType();
      }
    } else if (auto floatAttr = dyn_cast<FloatAttr>(initValue)) {
      if (floatAttr.getType() != getType()) {
        return emitOpError("initial value expected to be of type ")
               << getType() << ", but was of type " << floatAttr.getType();
      }
    } else if (!isa<emitc::OpaqueAttr>(initValue)) {
      return emitOpError("initial value should be a integer, float, elements "
                         "or opaque attribute, but got ")
             << initValue;
    }
  }
  if (getStaticSpecifier() && getExternSpecifier()) {
    return emitOpError("cannot have both static and extern specifiers");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// GetGlobalOp
//===----------------------------------------------------------------------===//

LogicalResult
GetGlobalOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verify that the type matches the type of the global variable.
  auto global =
      symbolTable.lookupNearestSymbolFrom<GlobalOp>(*this, getNameAttr());
  if (!global)
    return emitOpError("'")
           << getName() << "' does not reference a valid emitc.global";

  Type resultType = getResult().getType();
  Type globalType = global.getType();

  // global has array type
  if (llvm::isa<ArrayType>(globalType)) {
    if (globalType != resultType)
      return emitOpError("on array type expects result type ")
             << resultType << " to match type " << globalType
             << " of the global @" << getName();
    return success();
  }

  // global has non-array type
  auto lvalueType = dyn_cast<LValueType>(resultType);
  if (!lvalueType || lvalueType.getValueType() != globalType)
    return emitOpError("on non-array type expects result inner type ")
           << lvalueType.getValueType() << " to match type " << globalType
           << " of the global @" << getName();
  return success();
}

//===----------------------------------------------------------------------===//
// SwitchOp
//===----------------------------------------------------------------------===//

/// Parse the case regions and values.
static ParseResult
parseSwitchCases(OpAsmParser &parser, DenseI64ArrayAttr &cases,
                 SmallVectorImpl<std::unique_ptr<Region>> &caseRegions) {
  SmallVector<int64_t> caseValues;
  while (succeeded(parser.parseOptionalKeyword("case"))) {
    int64_t value;
    Region &region = *caseRegions.emplace_back(std::make_unique<Region>());
    if (parser.parseInteger(value) ||
        parser.parseRegion(region, /*arguments=*/{}))
      return failure();
    caseValues.push_back(value);
  }
  cases = parser.getBuilder().getDenseI64ArrayAttr(caseValues);
  return success();
}

/// Print the case regions and values.
static void printSwitchCases(OpAsmPrinter &p, Operation *op,
                             DenseI64ArrayAttr cases, RegionRange caseRegions) {
  for (auto [value, region] : llvm::zip(cases.asArrayRef(), caseRegions)) {
    p.printNewline();
    p << "case " << value << ' ';
    p.printRegion(*region, /*printEntryBlockArgs=*/false);
  }
}

static LogicalResult verifyRegion(emitc::SwitchOp op, Region &region,
                                  const Twine &name) {
  auto yield = dyn_cast<emitc::YieldOp>(region.front().back());
  if (!yield)
    return op.emitOpError("expected region to end with emitc.yield, but got ")
           << region.front().back().getName();

  if (yield.getNumOperands() != 0) {
    return (op.emitOpError("expected each region to return ")
            << "0 values, but " << name << " returns "
            << yield.getNumOperands())
               .attachNote(yield.getLoc())
           << "see yield operation here";
  }

  return success();
}

LogicalResult emitc::SwitchOp::verify() {
  if (!isIntegerIndexOrOpaqueType(getArg().getType()))
    return emitOpError("unsupported type ") << getArg().getType();

  if (getCases().size() != getCaseRegions().size()) {
    return emitOpError("has ")
           << getCaseRegions().size() << " case regions but "
           << getCases().size() << " case values";
  }

  DenseSet<int64_t> valueSet;
  for (int64_t value : getCases())
    if (!valueSet.insert(value).second)
      return emitOpError("has duplicate case value: ") << value;

  if (failed(verifyRegion(*this, getDefaultRegion(), "default region")))
    return failure();

  for (auto [idx, caseRegion] : llvm::enumerate(getCaseRegions()))
    if (failed(verifyRegion(*this, caseRegion, "case region #" + Twine(idx))))
      return failure();

  return success();
}

unsigned emitc::SwitchOp::getNumCases() { return getCases().size(); }

Block &emitc::SwitchOp::getDefaultBlock() { return getDefaultRegion().front(); }

Block &emitc::SwitchOp::getCaseBlock(unsigned idx) {
  assert(idx < getNumCases() && "case index out-of-bounds");
  return getCaseRegions()[idx].front();
}

void SwitchOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &successors) {
  llvm::copy(getRegions(), std::back_inserter(successors));
}

void SwitchOp::getEntrySuccessorRegions(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &successors) {
  FoldAdaptor adaptor(operands, *this);

  // If a constant was not provided, all regions are possible successors.
  auto arg = dyn_cast_or_null<IntegerAttr>(adaptor.getArg());
  if (!arg) {
    llvm::copy(getRegions(), std::back_inserter(successors));
    return;
  }

  // Otherwise, try to find a case with a matching value. If not, the
  // default region is the only successor.
  for (auto [caseValue, caseRegion] : llvm::zip(getCases(), getCaseRegions())) {
    if (caseValue == arg.getInt()) {
      successors.emplace_back(&caseRegion);
      return;
    }
  }
  successors.emplace_back(&getDefaultRegion());
}

void SwitchOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands, SmallVectorImpl<InvocationBounds> &bounds) {
  auto operandValue = llvm::dyn_cast_or_null<IntegerAttr>(operands.front());
  if (!operandValue) {
    // All regions are invoked at most once.
    bounds.append(getNumRegions(), InvocationBounds(/*lb=*/0, /*ub=*/1));
    return;
  }

  unsigned liveIndex = getNumRegions() - 1;
  const auto *iteratorToInt = llvm::find(getCases(), operandValue.getInt());

  liveIndex = iteratorToInt != getCases().end()
                  ? std::distance(getCases().begin(), iteratorToInt)
                  : liveIndex;

  for (unsigned regIndex = 0, regNum = getNumRegions(); regIndex < regNum;
       ++regIndex)
    bounds.emplace_back(/*lb=*/0, /*ub=*/regIndex == liveIndex);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/EmitC/IR/EmitC.cpp.inc"
