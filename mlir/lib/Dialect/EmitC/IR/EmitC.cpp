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
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

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
  Type attrType = cast<TypedAttr>(value).getType();

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

  if (lhsType.isa<emitc::PointerType>() && rhsType.isa<emitc::PointerType>())
    return emitOpError("requires that at most one operand is a pointer");

  if ((lhsType.isa<emitc::PointerType>() &&
       !rhsType.isa<IntegerType, emitc::OpaqueType>()) ||
      (rhsType.isa<emitc::PointerType>() &&
       !lhsType.isa<IntegerType, emitc::OpaqueType>()))
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

  Operation *op = getOperand().getDefiningOp();
  if (op && dyn_cast<ConstantOp>(op))
    return emitOpError("cannot apply to constant");

  return success();
}

//===----------------------------------------------------------------------===//
// AssignOp
//===----------------------------------------------------------------------===//

/// The assign op requires that the assigned value's type matches the
/// assigned-to variable type.
LogicalResult emitc::AssignOp::verify() {
  Value variable = getVar();
  Operation *variableDef = variable.getDefiningOp();
  if (!variableDef || !llvm::isa<emitc::VariableOp>(variableDef))
    return emitOpError() << "requires first operand (" << variable
                         << ") to be a Variable";

  Value value = getValue();
  if (variable.getType() != value.getType())
    return emitOpError() << "requires value's type (" << value.getType()
                         << ") to match variable's type (" << variable.getType()
                         << ")";
  if (isa<ArrayType>(variable.getType()))
    return emitOpError() << "cannot assign to array type";
  return success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  Type input = inputs.front(), output = outputs.front();

  return ((llvm::isa<IntegerType, FloatType, IndexType, emitc::OpaqueType,
                     emitc::PointerType>(input)) &&
          (llvm::isa<IntegerType, FloatType, IndexType, emitc::OpaqueType,
                     emitc::PointerType>(output)));
}

//===----------------------------------------------------------------------===//
// CallOp
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

  if (llvm::any_of(getResultTypes(),
                   [](Type type) { return isa<ArrayType>(type); })) {
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
  function_interface_impl::addArgAndResultAttrs(
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

  if (rhsType.isa<emitc::PointerType>() && !lhsType.isa<emitc::PointerType>())
    return emitOpError("rhs can only be a pointer if lhs is a pointer");

  if (lhsType.isa<emitc::PointerType>() &&
      !rhsType.isa<IntegerType, emitc::OpaqueType, emitc::PointerType>())
    return emitOpError("requires that rhs is an integer, pointer or of opaque "
                       "type if lhs is a pointer");

  if (lhsType.isa<emitc::PointerType>() && rhsType.isa<emitc::PointerType>() &&
      !resultType.isa<IntegerType, emitc::OpaqueType>())
    return emitOpError("requires that the result is an integer or of opaque "
                       "type if lhs and rhs are pointers");
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
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/EmitC/IR/EmitC.cpp.inc"

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
    if (dim <= 0)
      return emitError() << "dimensions must have positive size";
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
