//===- EmitC.cpp - EmitC Dialect ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
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

LogicalResult emitc::CallOp::verify() {
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

  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

/// The constant op requires that the attribute's type matches the return type.
LogicalResult emitc::ConstantOp::verify() {
  if (llvm::isa<emitc::OpaqueAttr>(getValueAttr()))
    return success();

  // Value must not be empty
  StringAttr strAttr = llvm::dyn_cast<StringAttr>(getValueAttr());
  if (strAttr && strAttr.empty())
    return emitOpError() << "value must not be empty";

  auto value = cast<TypedAttr>(getValueAttr());
  Type type = getType();
  if (!llvm::isa<NoneType>(value.getType()) && type != value.getType())
    return emitOpError() << "requires attribute's type (" << value.getType()
                         << ") to match op's return type (" << type << ")";
  return success();
}

OpFoldResult emitc::ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

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

/// The variable op requires that the attribute's type matches the return type.
LogicalResult emitc::VariableOp::verify() {
  if (llvm::isa<emitc::OpaqueAttr>(getValueAttr()))
    return success();

  auto value = cast<TypedAttr>(getValueAttr());
  Type type = getType();
  if (!llvm::isa<NoneType>(value.getType()) && type != value.getType())
    return emitOpError() << "requires attribute's type (" << value.getType()
                         << ") to match op's return type (" << type << ")";
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

Attribute emitc::OpaqueAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return Attribute();
  std::string value;
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseOptionalString(&value)) {
    parser.emitError(loc) << "expected string";
    return Attribute();
  }
  if (parser.parseGreater())
    return Attribute();

  return get(parser.getContext(), value);
}

void emitc::OpaqueAttr::print(AsmPrinter &printer) const {
  printer << "<\"";
  llvm::printEscapedString(getValue(), printer.getStream());
  printer << "\">";
}

//===----------------------------------------------------------------------===//
// EmitC Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/EmitC/IR/EmitCTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// OpaqueType
//===----------------------------------------------------------------------===//

Type emitc::OpaqueType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();
  std::string value;
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseOptionalString(&value) || value.empty()) {
    parser.emitError(loc) << "expected non empty string in !emitc.opaque type";
    return Type();
  }
  if (value.back() == '*') {
    parser.emitError(loc) << "pointer not allowed as outer type with "
                             "!emitc.opaque, use !emitc.ptr instead";
    return Type();
  }
  if (parser.parseGreater())
    return Type();
  return get(parser.getContext(), value);
}

void emitc::OpaqueType::print(AsmPrinter &printer) const {
  printer << "<\"";
  llvm::printEscapedString(getValue(), printer.getStream());
  printer << "\">";
}
