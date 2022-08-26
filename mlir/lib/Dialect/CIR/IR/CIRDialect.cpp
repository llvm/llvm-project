//===- CIRDialect.cpp - MLIR CIR ops implementation -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CIR dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/CIR/IR/CIRDialect.h"
#include "mlir/Dialect/CIR/IR/CIRAttrs.h"
#include "mlir/Dialect/CIR/IR/CIRTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

using namespace mlir;
using namespace mlir::cir;

#include "mlir/Dialect/CIR/IR/CIROpsEnums.cpp.inc"
#include "mlir/Dialect/CIR/IR/CIROpsStructs.cpp.inc"

#include "mlir/Dialect/CIR/IR/CIROpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// CIR Dialect
//===----------------------------------------------------------------------===//
namespace {
struct CIROpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Type type, raw_ostream &os) const final {
    if (auto structType = type.dyn_cast<StructType>()) {
      os << structType.getTypeName();
      return AliasResult::OverridableAlias;
    }

    return AliasResult::NoAlias;
  }
};
} // namespace

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void cir::CIRDialect::initialize() {
  registerTypes();
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/CIR/IR/CIROps.cpp.inc"
      >();
  addInterfaces<CIROpAsmDialectInterface>();
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Parses one of the keywords provided in the list `keywords` and returns the
// position of the parsed keyword in the list. If none of the keywords from the
// list is parsed, returns -1.
static int parseOptionalKeywordAlternative(OpAsmParser &parser,
                                           ArrayRef<StringRef> keywords) {
  for (auto en : llvm::enumerate(keywords)) {
    if (succeeded(parser.parseOptionalKeyword(en.value())))
      return en.index();
  }
  return -1;
}

namespace {
template <typename Ty>
struct EnumTraits {};

#define REGISTER_ENUM_TYPE(Ty)                                                 \
  template <>                                                                  \
  struct EnumTraits<Ty> {                                                      \
    static StringRef stringify(Ty value) { return stringify##Ty(value); }      \
    static unsigned getMaxEnumVal() { return getMaxEnumValFor##Ty(); }         \
  }

REGISTER_ENUM_TYPE(GlobalLinkageKind);
} // namespace

/// Parse an enum from the keyword, or default to the provided default value.
/// The return type is the enum type by default, unless overriden with the
/// second template argument.
/// TODO: teach other places in this file to use this function.
template <typename EnumTy, typename RetTy = EnumTy>
static RetTy parseOptionalCIRKeyword(OpAsmParser &parser,
                                     OperationState &result,
                                     EnumTy defaultValue) {
  SmallVector<StringRef, 10> names;
  for (unsigned i = 0, e = EnumTraits<EnumTy>::getMaxEnumVal(); i <= e; ++i)
    names.push_back(EnumTraits<EnumTy>::stringify(static_cast<EnumTy>(i)));

  int index = parseOptionalKeywordAlternative(parser, names);
  if (index == -1)
    return static_cast<RetTy>(defaultValue);
  return static_cast<RetTy>(index);
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

static LogicalResult checkConstantTypes(mlir::Operation *op, mlir::Type opType,
                                        mlir::Attribute attrType) {
  if (attrType.isa<NullAttr>()) {
    if (opType.isa<::mlir::cir::PointerType>())
      return success();
    return op->emitOpError("nullptr expects pointer type");
  }

  if (attrType.isa<BoolAttr>()) {
    if (!opType.isa<mlir::cir::BoolType>())
      return op->emitOpError("result type (")
             << opType << ") must be '!cir.bool' for '" << attrType << "'";
    return success();
  }

  if (attrType.isa<IntegerAttr, FloatAttr>()) {
    auto at = attrType.cast<TypedAttr>();
    if (at.getType() != opType) {
      return op->emitOpError("result type (")
             << opType << ") does not match value type (" << at.getType()
             << ")";
    }
    return success();
  }

  if (attrType.isa<mlir::cir::CstArrayAttr>()) {
    // CstArrayAttr is already verified to bing with cir.array type.
    return success();
  }

  if (attrType.isa<SymbolRefAttr>()) {
    if (opType.isa<::mlir::cir::PointerType>())
      return success();
    return op->emitOpError("symbolref expects pointer type");
  }

  assert(attrType.isa<TypedAttr>() && "What else could we be looking at here?");
  return op->emitOpError("cannot have value of type ")
         << attrType.cast<TypedAttr>().getType();
}

LogicalResult ConstantOp::verify() {
  // ODS already generates checks to make sure the result type is valid. We just
  // need to additionally check that the value's attribute type is consistent
  // with the result type.
  return checkConstantTypes(getOperation(), getType(), getValue());
}

static ParseResult parseConstantValue(OpAsmParser &parser,
                                      mlir::Attribute &valueAttr) {
  NamedAttrList attr;
  if (parser.parseAttribute(valueAttr, "value", attr).failed()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected constant attribute to match type");
  }

  return success();
}

// FIXME: create a CIRCstAttr and hide this away for both global
// initialization and cir.cst operation.
static void printConstant(OpAsmPrinter &p, Attribute value) {
  p.printAttribute(value);
}

static void printConstantValue(OpAsmPrinter &p, cir::ConstantOp op,
                               Attribute value) {
  printConstant(p, value);
}

OpFoldResult ConstantOp::fold(FoldAdaptor /*adaptor*/) { return getValue(); }

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

LogicalResult CastOp::verify() {
  auto resType = getResult().getType();
  auto srcType = getSrc().getType();

  switch (getKind()) {
  case cir::CastKind::int_to_bool: {
    if (!resType.isa<mlir::cir::BoolType>())
      return emitOpError() << "requires !cir.bool type for result";
    if (!(srcType.isInteger(32) || srcType.isInteger(64)))
      return emitOpError() << "requires integral type for result";
    return success();
  }
  case cir::CastKind::integral: {
    if (!resType.isa<mlir::IntegerType>())
      return emitOpError() << "requires !IntegerType for result";
    if (!srcType.isa<mlir::IntegerType>())
      return emitOpError() << "requires !IntegerType for source";
    return success();
  }
  case cir::CastKind::array_to_ptrdecay: {
    auto arrayPtrTy = srcType.dyn_cast<mlir::cir::PointerType>();
    auto flatPtrTy = resType.dyn_cast<mlir::cir::PointerType>();
    if (!arrayPtrTy || !flatPtrTy)
      return emitOpError() << "requires !cir.ptr type for source and result";

    auto arrayTy = arrayPtrTy.getPointee().dyn_cast<mlir::cir::ArrayType>();
    if (!arrayTy)
      return emitOpError() << "requires !cir.array pointee";

    if (arrayTy.getEltType() != flatPtrTy.getPointee())
      return emitOpError()
             << "requires same type for array element and pointee result";
    return success();
  }
  case cir::CastKind::bitcast: {
    if (!srcType.dyn_cast<mlir::cir::PointerType>() ||
        !resType.dyn_cast<mlir::cir::PointerType>())
      return emitOpError() << "requires !cir.ptr type for source and result";
    return success();
  }
  }

  llvm_unreachable("Unknown CastOp kind?");
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

static mlir::LogicalResult checkReturnAndFunction(ReturnOp op,
                                                  cir::FuncOp function) {
  // ReturnOps currently only have a single optional operand.
  if (op.getNumOperands() > 1)
    return op.emitOpError() << "expects at most 1 return operand";

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (op.getNumOperands() != results.size())
    return op.emitOpError()
           << "does not return the same number of values ("
           << op.getNumOperands() << ") as the enclosing function ("
           << results.size() << ")";

  // If the operation does not have an input, we are done.
  if (!op.hasOperand())
    return mlir::success();

  auto inputType = *op.operand_type_begin();
  auto resultType = results.front();

  // Check that the result type of the function matches the operand type.
  if (inputType == resultType)
    return mlir::success();

  return op.emitError() << "type of return operand (" << inputType
                        << ") doesn't match function result type ("
                        << resultType << ")";
}

mlir::LogicalResult ReturnOp::verify() {
  // Returns can be present in multiple different scopes, get the
  // wrapping function and start from there.
  auto *fnOp = getOperation()->getParentOp();
  while (!isa<cir::FuncOp>(fnOp))
    fnOp = fnOp->getParentOp();

  // Make sure return types match function return type.
  if (checkReturnAndFunction(*this, cast<cir::FuncOp>(fnOp)).failed())
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

static LogicalResult checkBlockTerminator(OpAsmParser &parser,
                                          llvm::SMLoc parserLoc,
                                          std::optional<Location> l, Region *r,
                                          bool ensureTerm = true) {
  mlir::Builder &builder = parser.getBuilder();
  if (r->hasOneBlock()) {
    if (ensureTerm) {
      ::mlir::impl::ensureRegionTerminator(
          *r, builder, *l, [](OpBuilder &builder, Location loc) {
            OperationState state(loc, YieldOp::getOperationName());
            YieldOp::build(builder, state);
            return Operation::create(state);
          });
    } else {
      assert(r && "region must not be empty");
      Block &block = r->back();
      if (block.empty() || !block.back().hasTrait<OpTrait::IsTerminator>()) {
        return parser.emitError(
            parser.getCurrentLocation(),
            "blocks are expected to be explicitly terminated");
      }
    }
    return success();
  }

  // Empty regions don't need any handling.
  auto &blocks = r->getBlocks();
  if (blocks.empty())
    return success();

  // Test that at least one block has a yield/return terminator. We can
  // probably make this a bit more strict.
  for (Block &block : blocks) {
    if (block.empty())
      continue;
    auto &op = block.back();
    if (op.hasTrait<mlir::OpTrait::IsTerminator>() &&
        isa<YieldOp, ReturnOp>(op)) {
      return success();
    }
  }

  parser.emitError(parserLoc,
                   "expected at least one block with cir.yield or cir.return");
  return failure();
}

ParseResult cir::IfOp::parse(OpAsmParser &parser, OperationState &result) {
  // Create the regions for 'then'.
  result.regions.reserve(2);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  OpAsmParser::UnresolvedOperand cond;
  Type boolType = ::mlir::cir::BoolType::get(builder.getContext());

  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, boolType, result.operands))
    return failure();

  // Parse the 'then' region.
  auto parseThenLoc = parser.getCurrentLocation();
  if (parser.parseRegion(*thenRegion, /*arguments=*/{},
                         /*argTypes=*/{}))
    return failure();
  if (checkBlockTerminator(parser, parseThenLoc, result.location, thenRegion)
          .failed())
    return failure();

  // If we find an 'else' keyword, parse the 'else' region.
  if (!parser.parseOptionalKeyword("else")) {
    auto parseElseLoc = parser.getCurrentLocation();
    if (parser.parseRegion(*elseRegion, /*arguments=*/{}, /*argTypes=*/{}))
      return failure();
    if (checkBlockTerminator(parser, parseElseLoc, result.location, elseRegion)
            .failed())
      return failure();
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

bool shouldPrintTerm(mlir::Region &r) {
  if (!r.hasOneBlock())
    return true;
  auto *entryBlock = &r.front();
  if (entryBlock->empty())
    return false;
  if (isa<ReturnOp>(entryBlock->back()))
    return true;
  YieldOp y = dyn_cast<YieldOp>(entryBlock->back());
  if (y && !y.isPlain())
    return true;
  return false;
}

void cir::IfOp::print(OpAsmPrinter &p) {
  p << " " << getCondition() << " ";
  auto &thenRegion = this->getThenRegion();
  p.printRegion(thenRegion,
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/shouldPrintTerm(thenRegion));

  // Print the 'else' regions if it exists and has a block.
  auto &elseRegion = this->getElseRegion();
  if (!elseRegion.empty()) {
    p << " else ";
    p.printRegion(elseRegion,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/shouldPrintTerm(elseRegion));
  }

  p.printOptionalAttrDict(getOperation()->getAttrs());
}

/// Default callback for IfOp builders. Inserts nothing for now.
void mlir::cir::buildTerminatedBody(OpBuilder &builder, Location loc) {}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void IfOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                               SmallVectorImpl<RegionSuccessor> &regions) {
  // The `then` and the `else` region branch back to the parent operation.
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor());
    return;
  }

  // Don't consider the else region if it is empty.
  Region *elseRegion = &this->getElseRegion();
  if (elseRegion->empty())
    elseRegion = nullptr;

  // Otherwise, the successor is dependent on the condition.
  // bool condition;
  // if (auto condAttr = operands.front().dyn_cast_or_null<IntegerAttr>()) {
  //   assert(0 && "not implemented");
  // condition = condAttr.getValue().isOneValue();
  // Add the successor regions using the condition.
  // regions.push_back(RegionSuccessor(condition ? &thenRegion() :
  // elseRegion));
  // return;
  // }

  // If the condition isn't constant, both regions may be executed.
  regions.push_back(RegionSuccessor(&getThenRegion()));
  // If the else region does not exist, it is not a viable successor.
  if (elseRegion)
    regions.push_back(RegionSuccessor(elseRegion));
  return;
}

void IfOp::build(OpBuilder &builder, OperationState &result, Value cond,
                 bool withElseRegion,
                 function_ref<void(OpBuilder &, Location)> thenBuilder,
                 function_ref<void(OpBuilder &, Location)> elseBuilder) {
  assert(thenBuilder && "the builder callback for 'then' must be present");

  result.addOperands(cond);

  OpBuilder::InsertionGuard guard(builder);
  Region *thenRegion = result.addRegion();
  builder.createBlock(thenRegion);
  thenBuilder(builder, result.location);

  Region *elseRegion = result.addRegion();
  if (!withElseRegion)
    return;

  builder.createBlock(elseRegion);
  elseBuilder(builder, result.location);
}

LogicalResult IfOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// ScopeOp
//===----------------------------------------------------------------------===//

ParseResult cir::ScopeOp::parse(OpAsmParser &parser, OperationState &result) {
  // Create one region within 'scope'.
  result.regions.reserve(1);
  Region *scopeRegion = result.addRegion();
  auto loc = parser.getCurrentLocation();

  // Parse the scope region.
  if (parser.parseRegion(*scopeRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  if (checkBlockTerminator(parser, loc, result.location, scopeRegion).failed())
    return failure();

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void cir::ScopeOp::print(OpAsmPrinter &p) {
  p << ' ';
  auto &scopeRegion = this->getScopeRegion();
  p.printRegion(scopeRegion,
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/shouldPrintTerm(scopeRegion));

  p.printOptionalAttrDict(getOperation()->getAttrs());
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void ScopeOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                                  SmallVectorImpl<RegionSuccessor> &regions) {
  // The only region always branch back to the parent operation.
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor());
    return;
  }

  // If the condition isn't constant, both regions may be executed.
  regions.push_back(RegionSuccessor(&getScopeRegion()));
}

void ScopeOp::build(OpBuilder &builder, OperationState &result,
                    TypeRange resultTypes,
                    function_ref<void(OpBuilder &, Location)> scopeBuilder) {
  assert(scopeBuilder && "the builder callback for 'then' must be present");
  result.addTypes(resultTypes);

  OpBuilder::InsertionGuard guard(builder);
  Region *scopeRegion = result.addRegion();
  builder.createBlock(scopeRegion);
  scopeBuilder(builder, result.location);
}

LogicalResult ScopeOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult YieldOp::verify() {
  auto isDominatedByLoopOrSwitch = [](Operation *parentOp) {
    while (!llvm::isa<cir::FuncOp>(parentOp)) {
      if (llvm::isa<cir::SwitchOp, cir::LoopOp>(parentOp))
        return true;
      parentOp = parentOp->getParentOp();
    }
    return false;
  };

  auto isDominatedByLoop = [](Operation *parentOp) {
    while (!llvm::isa<cir::FuncOp>(parentOp)) {
      if (llvm::isa<cir::LoopOp>(parentOp))
        return true;
      parentOp = parentOp->getParentOp();
    }
    return false;
  };

  if (isBreak()) {
    if (!isDominatedByLoopOrSwitch(getOperation()->getParentOp()))
      return emitOpError()
             << "shall be dominated by 'cir.loop' or 'cir.switch'";
    return mlir::success();
  }

  if (isContinue()) {
    if (!isDominatedByLoop(getOperation()->getParentOp()))
      return emitOpError() << "shall be dominated by 'cir.loop'";
    return mlir::success();
  }

  if (isFallthrough()) {
    if (!llvm::isa<SwitchOp>(getOperation()->getParentOp()))
      return emitOpError() << "fallthrough only expected within 'cir.switch'";
    return mlir::success();
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// BrOp
//===----------------------------------------------------------------------===//

mlir::SuccessorOperands BrOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  // Current block targets do not have operands.
  return mlir::SuccessorOperands(MutableOperandRange(getOperation(), 0, 0));
}

Block *BrOp::getSuccessorForOperands(ArrayRef<Attribute>) { return getDest(); }

//===----------------------------------------------------------------------===//
// BrCondOp
//===----------------------------------------------------------------------===//

mlir::SuccessorOperands BrCondOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return SuccessorOperands(index == 0 ? getDestOperandsTrueMutable()
                                      : getDestOperandsFalseMutable());
}

Block *BrCondOp::getSuccessorForOperands(ArrayRef<Attribute> operands) {
  if (IntegerAttr condAttr = operands.front().dyn_cast_or_null<IntegerAttr>())
    return condAttr.getValue().isOne() ? getDestTrue() : getDestFalse();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// SwitchOp
//===----------------------------------------------------------------------===//

ParseResult
parseSwitchOp(OpAsmParser &parser,
              llvm::SmallVectorImpl<std::unique_ptr<::mlir::Region>> &regions,
              ::mlir::ArrayAttr &casesAttr,
              mlir::OpAsmParser::UnresolvedOperand &cond,
              mlir::Type &condType) {
  ::mlir::IntegerType intCondType;
  SmallVector<mlir::Attribute, 4> cases;

  auto parseAndCheckRegion = [&]() -> ParseResult {
    // Parse region attached to case
    regions.emplace_back(new Region);
    Region &currRegion = *regions.back().get();
    auto parserLoc = parser.getCurrentLocation();
    if (parser.parseRegion(currRegion, /*arguments=*/{}, /*argTypes=*/{})) {
      regions.clear();
      return failure();
    }

    if (currRegion.empty()) {
      return parser.emitError(parser.getCurrentLocation(),
                              "case region shall not be empty");
    }

    if (checkBlockTerminator(parser, parserLoc, std::nullopt, &currRegion,
                             /*ensureTerm=*/false)
            .failed())
      return failure();
    return success();
  };

  auto parseCase = [&]() -> ParseResult {
    auto loc = parser.getCurrentLocation();
    if (parser.parseKeyword("case").failed())
      return parser.emitError(loc, "expected 'case' keyword here");

    if (parser.parseLParen().failed())
      return parser.emitError(parser.getCurrentLocation(), "expected '('");

    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;

    //   case (equal, 20) {
    //   ...
    // 1. Get the case kind
    // 2. Get the value (next in list)

    // These needs to be in sync with CIROps.td
    if (parser.parseOptionalKeyword(&attrStr, {"default", "equal", "anyof"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult = parser.parseOptionalAttribute(
          attrVal, parser.getBuilder().getNoneType(), "kind", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      }
    }

    if (attrStr.empty()) {
      return parser.emitError(
          loc, "expected string or keyword containing one of the following "
               "enum values for attribute 'kind' [default, equal, anyof]");
    }

    auto attrOptional = ::mlir::cir::symbolizeCaseOpKind(attrStr.str());
    if (!attrOptional)
      return parser.emitError(loc, "invalid ")
             << "kind attribute specification: \"" << attrStr << '"';

    auto kindAttr = ::mlir::cir::CaseOpKindAttr::get(
        parser.getBuilder().getContext(), attrOptional.value());

    // `,` value or `,` [values,...]
    SmallVector<mlir::Attribute, 4> caseEltValueListAttr;
    mlir::ArrayAttr caseValueList;

    switch (kindAttr.getValue()) {
    case cir::CaseOpKind::Equal: {
      if (parser.parseComma().failed())
        return mlir::failure();
      int64_t val = 0;
      if (parser.parseInteger(val).failed())
        return ::mlir::failure();
      caseEltValueListAttr.push_back(mlir::IntegerAttr::get(intCondType, val));
      break;
    }
    case cir::CaseOpKind::Anyof: {
      if (parser.parseComma().failed())
        return mlir::failure();
      if (parser.parseLSquare().failed())
        return mlir::failure();
      if (parser.parseCommaSeparatedList([&]() {
            int64_t val = 0;
            if (parser.parseInteger(val).failed())
              return ::mlir::failure();
            caseEltValueListAttr.push_back(
                mlir::IntegerAttr::get(intCondType, val));
            return ::mlir::success();
          }))
        return mlir::failure();
      if (parser.parseRSquare().failed())
        return mlir::failure();
      break;
    }
    case cir::CaseOpKind::Default: {
      if (parser.parseRParen().failed())
        return parser.emitError(parser.getCurrentLocation(), "expected ')'");
      cases.push_back(cir::CaseAttr::get(
          parser.getContext(), parser.getBuilder().getArrayAttr({}), kindAttr));
      return parseAndCheckRegion();
    }
    }

    caseValueList = parser.getBuilder().getArrayAttr(caseEltValueListAttr);
    cases.push_back(
        cir::CaseAttr::get(parser.getContext(), caseValueList, kindAttr));
    if (succeeded(parser.parseOptionalColon())) {
      Type caseIntTy;
      if (parser.parseType(caseIntTy).failed())
        return parser.emitError(parser.getCurrentLocation(), "expected type");
      if (intCondType != caseIntTy)
        return parser.emitError(parser.getCurrentLocation(),
                                "expected a match with the condition type");
    }
    if (parser.parseRParen().failed())
      return parser.emitError(parser.getCurrentLocation(), "expected ')'");
    return parseAndCheckRegion();
  };

  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseOperand(cond))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseCustomTypeWithFallback(intCondType))
    return ::mlir::failure();
  condType = intCondType;
  if (parser.parseRParen())
    return ::mlir::failure();

  if (parser
          .parseCommaSeparatedList(OpAsmParser::Delimiter::Square, parseCase,
                                   " in cases list")
          .failed())
    return failure();

  casesAttr = parser.getBuilder().getArrayAttr(cases);
  return ::mlir::success();
}

void printSwitchOp(OpAsmPrinter &p, SwitchOp op,
                   mlir::MutableArrayRef<::mlir::Region> regions,
                   mlir::ArrayAttr casesAttr, mlir::Value condition,
                   mlir::Type condType) {
  int idx = 0, lastIdx = regions.size() - 1;

  p << "(";
  p << condition;
  p << " : ";
  p.printStrippedAttrOrType(condType);
  p << ") [";
  // FIXME: ideally we want some extra indentation for "cases" but too
  // cumbersome to pull it out now, since most handling is private. Perhaps
  // better improve overall mechanism.
  p.printNewline();
  for (auto &r : regions) {
    p << "case (";

    auto attr = casesAttr[idx].cast<CaseAttr>();
    auto kind = attr.getKind().getValue();
    assert((kind == CaseOpKind::Default || kind == CaseOpKind::Equal ||
            kind == CaseOpKind::Anyof) &&
           "unknown case");

    // Case kind
    p << stringifyCaseOpKind(kind);

    // Case value
    switch (kind) {
    case cir::CaseOpKind::Equal: {
      p << ", ";
      p.printStrippedAttrOrType(attr.getValue()[0]);
      break;
    }
    case cir::CaseOpKind::Anyof: {
      p << ", [";
      llvm::interleaveComma(attr.getValue(), p, [&](const Attribute &a) {
        p.printAttributeWithoutType(a);
      });
      p << "] : ";
      auto typedAttr = attr.getValue()[0].dyn_cast<TypedAttr>();
      assert(typedAttr && "this should never not have a type!");
      p.printType(typedAttr.getType());
      break;
    }
    case cir::CaseOpKind::Default:
      break;
    }

    p << ") ";
    p.printRegion(r, /*printEntryBLockArgs=*/false,
                  /*printBlockTerminators=*/true);
    if (idx < lastIdx)
      p << ",";
    p.printNewline();
    idx++;
  }
  p << "]";
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes
/// that correspond to a constant value for each operand, or null if that
/// operand is not a constant.
void SwitchOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                                   SmallVectorImpl<RegionSuccessor> &regions) {
  // If any index all the underlying regions branch back to the parent
  // operation.
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor());
    return;
  }

  // for (auto &r : this->getRegions()) {
  // If we can figure out the case stmt we are landing, this can be
  // overly simplified.
  // bool condition;
  // if (auto condAttr = operands.front().dyn_cast_or_null<IntegerAttr>()) {
  //   assert(0 && "not implemented");
  //   (void)r;
  // condition = condAttr.getValue().isOneValue();
  // Add the successor regions using the condition.
  // regions.push_back(RegionSuccessor(condition ? &thenRegion() :
  // elseRegion));
  // return;
  // }
  // }

  // If the condition isn't constant, all regions may be executed.
  for (auto &r : this->getRegions())
    regions.push_back(RegionSuccessor(&r));
}

LogicalResult SwitchOp::verify() { return success(); }

void SwitchOp::build(
    OpBuilder &builder, OperationState &result, Value cond,
    function_ref<void(OpBuilder &, Location, OperationState &)> switchBuilder) {
  assert(switchBuilder && "the builder callback for regions must be present");
  OpBuilder::InsertionGuard guardSwitch(builder);
  result.addOperands({cond});
  switchBuilder(builder, result.location, result);
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

void LoopOp::build(OpBuilder &builder, OperationState &result,
                   cir::LoopOpKind kind,
                   function_ref<void(OpBuilder &, Location)> condBuilder,
                   function_ref<void(OpBuilder &, Location)> bodyBuilder,
                   function_ref<void(OpBuilder &, Location)> stepBuilder) {
  OpBuilder::InsertionGuard guard(builder);
  ::mlir::cir::LoopOpKindAttr kindAttr =
      cir::LoopOpKindAttr::get(builder.getContext(), kind);
  result.addAttribute("kind", kindAttr);

  Region *condRegion = result.addRegion();
  builder.createBlock(condRegion);
  condBuilder(builder, result.location);

  Region *bodyRegion = result.addRegion();
  builder.createBlock(bodyRegion);
  bodyBuilder(builder, result.location);

  Region *stepRegion = result.addRegion();
  builder.createBlock(stepRegion);
  stepBuilder(builder, result.location);
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes
/// that correspond to a constant value for each operand, or null if that
/// operand is not a constant.
void LoopOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                                 SmallVectorImpl<RegionSuccessor> &regions) {
  // If any index all the underlying regions branch back to the parent
  // operation.
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor());
    return;
  }

  // FIXME: we want to look at cond region for getting more accurate results
  // if the other regions will get a chance to execute.
  regions.push_back(RegionSuccessor(&this->getCond()));
  regions.push_back(RegionSuccessor(&this->getBody()));
  regions.push_back(RegionSuccessor(&this->getStep()));
}

llvm::SmallVector<Region *> LoopOp::getLoopRegions() { return {&getBody()}; }

LogicalResult LoopOp::verify() {
  // Cond regions should only terminate with plain 'cir.yield' or
  // 'cir.yield continue'.
  auto terminateError = [&]() {
    return emitOpError() << "cond region must be terminated with "
                            "'cir.yield' or 'cir.yield continue'";
  };

  auto &blocks = getCond().getBlocks();
  for (Block &block : blocks) {
    if (block.empty())
      continue;
    auto &op = block.back();
    if (isa<BrCondOp>(op))
      continue;
    if (!isa<YieldOp>(op))
      terminateError();
    auto y = cast<YieldOp>(op);
    if (!(y.isPlain() || y.isContinue()))
      terminateError();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

static void printGlobalOpTypeAndInitialValue(OpAsmPrinter &p, GlobalOp op,
                                             TypeAttr type,
                                             Attribute initAttr) {
  auto printType = [&]() { p << ": " << type; };
  if (!op.isDeclaration()) {
    p << "= ";
    // This also prints the type...
    printConstant(p, initAttr);
    if (initAttr.isa<SymbolRefAttr>())
      printType();
  } else {
    printType();
  }
}

static ParseResult
parseGlobalOpTypeAndInitialValue(OpAsmParser &parser, TypeAttr &typeAttr,
                                 Attribute &initialValueAttr) {
  if (parser.parseOptionalEqual().failed()) {
    // Absence of equal means a declaration, so we need to parse the type.
    //  cir.global @a : i32
    Type type;
    if (parser.parseColonType(type))
      return failure();
    typeAttr = TypeAttr::get(type);
    return success();
  }

  // Parse constant with initializer, examples:
  //  cir.global @y = 3.400000e+00 : f32
  //  cir.global @rgb = #cir.cst_array<[...] : !cir.array<i8 x 3>>
  if (parseConstantValue(parser, initialValueAttr).failed())
    return failure();

  mlir::Type opTy;
  if (auto sra = initialValueAttr.dyn_cast<SymbolRefAttr>()) {
    if (parser.parseColonType(opTy))
      return failure();
  } else {
    // Handle StringAttrs
    assert(initialValueAttr.isa<mlir::TypedAttr>() &&
           "Non-typed attrs shouldn't appear here.");
    auto typedAttr = initialValueAttr.cast<mlir::TypedAttr>();
    opTy = typedAttr.getType();
  }

  typeAttr = TypeAttr::get(opTy);
  return success();
}

LogicalResult GlobalOp::verify() {
  // Verify that the initial value, if present, is either a unit attribute or
  // an attribute CIR supports.
  if (getInitialValue().has_value()) {
    if (checkConstantTypes(getOperation(), getSymType(), *getInitialValue())
            .failed())
      return failure();
  }

  if (std::optional<uint64_t> alignAttr = getAlignment()) {
    uint64_t alignment = alignAttr.value();
    if (!llvm::isPowerOf2_64(alignment))
      return emitError() << "alignment attribute value " << alignment
                         << " is not a power of 2";
  }

  switch (getLinkage()) {
  case GlobalLinkageKind::InternalLinkage:
  case GlobalLinkageKind::PrivateLinkage:
    if (isPublic())
      return emitError() << "public visibility not allowed with '"
                         << stringifyGlobalLinkageKind(getLinkage())
                         << "' linkage";
    break;
  case GlobalLinkageKind::ExternalLinkage:
  case GlobalLinkageKind::ExternalWeakLinkage:
    if (isPrivate())
      return emitError() << "private visibility not allowed with '"
                         << stringifyGlobalLinkageKind(getLinkage())
                         << "' linkage";
    break;
  default:
    assert(0 && "not implemented");
  }

  // TODO: verify visibility for declarations?
  return success();
}

void GlobalOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                     StringRef sym_name, Type sym_type, bool isConstant,
                     cir::GlobalLinkageKind linkage) {
  odsState.addAttribute(getSymNameAttrName(odsState.name),
                        odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getSymTypeAttrName(odsState.name),
                        ::mlir::TypeAttr::get(sym_type));
  if (isConstant)
    odsState.addAttribute("constant", odsBuilder.getUnitAttr());

  ::mlir::cir::GlobalLinkageKindAttr linkageAttr =
      cir::GlobalLinkageKindAttr::get(odsBuilder.getContext(), linkage);
  odsState.addAttribute("linkage", linkageAttr);
}

//===----------------------------------------------------------------------===//
// GetGlobalOp
//===----------------------------------------------------------------------===//

LogicalResult
GetGlobalOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verify that the result type underlying pointer type matches the type of the
  // referenced cir.global op.
  auto global =
      symbolTable.lookupNearestSymbolFrom<GlobalOp>(*this, getNameAttr());
  if (!global)
    return emitOpError("'")
           << getName() << "' does not reference a valid cir.global";

  auto resultType = getAddr().getType().dyn_cast<PointerType>();
  if (!resultType || global.getSymType() != resultType.getPointee())
    return emitOpError("result type pointee type '")
           << resultType.getPointee() << "' does not match type "
           << global.getSymType() << " of the global @" << getName();
  return success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

/// Returns the name used for the linkage attribute. This *must* correspond to
/// the name of the attribute in ODS.
static StringRef getLinkageAttrNameString() { return "linkage"; }

void cir::FuncOp::build(OpBuilder &builder, OperationState &result,
                        StringRef name, FunctionType type,
                        GlobalLinkageKind linkage,
                        ArrayRef<NamedAttribute> attrs,
                        ArrayRef<DictionaryAttr> argAttrs) {
  result.addRegion();
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(type));
  result.addAttribute(
      getLinkageAttrNameString(),
      GlobalLinkageKindAttr::get(builder.getContext(), linkage));
  result.attributes.append(attrs.begin(), attrs.end());
  if (argAttrs.empty())
    return;

  function_interface_impl::addArgAndResultAttrs(
      builder, result, argAttrs,
      /*resultAttrs=*/std::nullopt, getArgAttrsAttrName(result.name),
      getResAttrsAttrName(result.name));
}

ParseResult cir::FuncOp::parse(OpAsmParser &parser, OperationState &state) {
  // Default to external linkage if no keyword is provided.
  state.addAttribute(
      getLinkageAttrNameString(),
      GlobalLinkageKindAttr::get(
          parser.getContext(),
          parseOptionalCIRKeyword<GlobalLinkageKind>(
              parser, state, GlobalLinkageKind::ExternalLinkage)));

  StringAttr nameAttr;
  SmallVector<OpAsmParser::Argument, 8> arguments;
  SmallVector<DictionaryAttr, 1> argAttrs;
  SmallVector<DictionaryAttr, 1> resultAttrs;
  SmallVector<Type, 8> argTypes;
  SmallVector<Type, 4> resultTypes;
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol.
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             state.attributes))
    return failure();

  // Parse the function signature.
  bool isVariadic = false;
  if (function_interface_impl::parseFunctionSignature(
          parser, /*allowVariadic=*/false, arguments, isVariadic, resultTypes,
          resultAttrs))
    return failure();

  for (auto &arg : arguments)
    argTypes.push_back(arg.type);

  auto fnType = builder.getFunctionType(argTypes, resultTypes);
  state.addAttribute(getFunctionTypeAttrName(state.name),
                     TypeAttr::get(fnType));

  // If additional attributes are present, parse them.
  if (parser.parseOptionalAttrDictWithKeyword(state.attributes))
    return failure();

  // Add the attributes to the function arguments.
  assert(resultAttrs.size() == resultTypes.size());
  function_interface_impl::addArgAndResultAttrs(
      builder, state, arguments, resultAttrs, getArgAttrsAttrName(state.name),
      getResAttrsAttrName(state.name));

  // Parse the optional function body.
  auto *body = state.addRegion();
  llvm::SMLoc loc = parser.getCurrentLocation();
  OptionalParseResult parseResult = parser.parseOptionalRegion(
      *body, arguments, /*enableNameShadowing=*/false);
  if (parseResult.has_value()) {
    if (failed(*parseResult))
      return failure();
    // Function body was parsed, make sure its not empty.
    if (body->empty())
      return parser.emitError(loc, "expected non-empty function body");
  }
  return success();
}

void cir::FuncOp::print(OpAsmPrinter &p) {
  p << ' ';
  if (getLinkage() != GlobalLinkageKind::ExternalLinkage)
    p << stringifyGlobalLinkageKind(getLinkage()) << ' ';

  // Print function name, signature, and control.
  p.printSymbolName(getSymName());
  auto fnType = getFunctionType();
  function_interface_impl::printFunctionSignature(p, *this, fnType.getInputs(),
                                                  /*isVariadic=*/false,
                                                  fnType.getResults());
  function_interface_impl::printFunctionAttributes(
      p, *this, {getFunctionTypeAttrName(), getLinkageAttrName()});

  // Print the body if this is not an external function.
  Region &body = getOperation()->getRegion(0);
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
// attribute is present.  This can check for preconditions of the
// getNumArguments hook not failing.
LogicalResult cir::FuncOp::verifyType() {
  auto type = getFunctionType();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getFunctionTypeAttrName().str() +
                       "' attribute of function type");
  if (getFunctionType().getNumResults() > 1)
    return emitOpError("cannot have more than one result");
  return success();
}

// Verifies linkage types, similar to LLVM:
// - functions don't have 'common' linkage
// - external functions have 'external' or 'extern_weak' linkage
LogicalResult cir::FuncOp::verify() {
  if (getLinkage() == cir::GlobalLinkageKind::CommonLinkage)
    return emitOpError() << "functions cannot have '"
                         << stringifyGlobalLinkageKind(
                                cir::GlobalLinkageKind::CommonLinkage)
                         << "' linkage";

  if (isExternal()) {
    if (getLinkage() != cir::GlobalLinkageKind::ExternalLinkage &&
        getLinkage() != cir::GlobalLinkageKind::ExternalWeakLinkage)
      return emitOpError() << "external functions must have '"
                           << stringifyGlobalLinkageKind(
                                  cir::GlobalLinkageKind::ExternalLinkage)
                           << "' or '"
                           << stringifyGlobalLinkageKind(
                                  cir::GlobalLinkageKind::ExternalWeakLinkage)
                           << "' linkage";
    return success();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

/// Get the argument operands to the called function.
OperandRange cir::CallOp::getArgOperands() {
  return {arg_operand_begin(), arg_operand_end()};
}

MutableOperandRange cir::CallOp::getArgOperandsMutable() {
  return getOperandsMutable();
}

/// Return the callee of this operation
CallInterfaceCallable cir::CallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// Set the callee for this operation.
void cir::CallOp::setCalleeFromCallable(::mlir::CallInterfaceCallable callee) {
  if (auto calling =
          (*this)->getAttrOfType<mlir::SymbolRefAttr>(getCalleeAttrName()))
    (*this)->setAttr(getCalleeAttrName(), callee.get<mlir::SymbolRefAttr>());
  setOperand(0, callee.get<mlir::Value>());
}

LogicalResult
cir::CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  FuncOp fn =
      symbolTable.lookupNearestSymbolFrom<mlir::cir::FuncOp>(*this, fnAttr);
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
// CIR defined traits
//===----------------------------------------------------------------------===//

LogicalResult
mlir::OpTrait::impl::verifySameFirstOperandAndResultType(Operation *op) {
  if (failed(verifyAtLeastNOperands(op, 1)) || failed(verifyOneResult(op)))
    return failure();

  auto type = op->getResult(0).getType();
  auto opType = op->getOperand(0).getType();

  if (type != opType)
    return op->emitOpError()
           << "requires the same type for first operand and result";

  return success();
}

//===----------------------------------------------------------------------===//
// CIR attributes
//===----------------------------------------------------------------------===//

LogicalResult mlir::cir::CstArrayAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::mlir::Type type, Attribute attr) {

  if (!(attr.isa<mlir::ArrayAttr>() || attr.isa<mlir::StringAttr>()))
    return emitError() << "constant array expects ArrayAttr or StringAttr";

  if (auto strAttr = attr.dyn_cast<mlir::StringAttr>()) {
    mlir::cir::ArrayType at = type.cast<mlir::cir::ArrayType>();
    auto intTy = at.getEltType().dyn_cast<mlir::IntegerType>();

    // TODO: add CIR type for char.
    if (!intTy || intTy.getWidth() != 8) {
      emitError() << "constant array element for string literals expects i8 "
                     "array element type";
      return failure();
    }
    return success();
  }

  assert(attr.isa<mlir::ArrayAttr>());
  auto arrayAttr = attr.cast<mlir::ArrayAttr>();
  auto at = type.cast<ArrayType>();

  // Make sure both number of elements and subelement types match type.
  if (at.getSize() != arrayAttr.size())
    return emitError() << "constant array size should match type size";
  LogicalResult eltTypeCheck = success();
  arrayAttr.walkImmediateSubElements(
      [&](Attribute attr) {
        // Once we find a mismatch, stop there.
        if (eltTypeCheck.failed())
          return;
        auto typedAttr = attr.dyn_cast<TypedAttr>();
        if (!typedAttr || typedAttr.getType() != at.getEltType()) {
          eltTypeCheck = failure();
          emitError()
              << "constant array element should match array element type";
        }
      },
      [&](Type type) {});
  return eltTypeCheck;
}

::mlir::Attribute CstArrayAttr::parse(::mlir::AsmParser &parser,
                                      ::mlir::Type type) {
  ::mlir::FailureOr<::mlir::Type> resultTy;
  ::mlir::FailureOr<Attribute> resultVal;
  ::llvm::SMLoc loc = parser.getCurrentLocation();
  (void)loc;
  // Parse literal '<'
  if (parser.parseLess())
    return {};

  // Parse variable 'value'
  resultVal = ::mlir::FieldParser<Attribute>::parse(parser);
  if (failed(resultVal)) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to parse CstArrayAttr parameter 'value' which is "
                     "to be a `Attribute`");
    return {};
  }

  // ArrayAttrrs have per-element type, not the type of the array...
  if (resultVal->dyn_cast<ArrayAttr>()) {
    // Parse literal ':'
    if (parser.parseColon())
      return {};

    // Parse variable 'type'
    resultTy = ::mlir::FieldParser<::mlir::Type>::parse(parser);
    if (failed(resultTy)) {
      parser.emitError(parser.getCurrentLocation(),
                       "failed to parse CstArrayAttr parameter 'type' which is "
                       "to be a `::mlir::Type`");
      return {};
    }
  } else {
    assert(resultVal->isa<TypedAttr>() && "IDK");
    auto ta = resultVal->cast<TypedAttr>();
    resultTy = ta.getType();
    if (resultTy->isa<mlir::NoneType>()) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected type declaration for string literal");
      return {};
    }
  }

  // Parse literal '>'
  if (parser.parseGreater())
    return {};
  return parser.getChecked<CstArrayAttr>(loc, parser.getContext(),
                                         resultTy.value(), resultVal.value());
}

void CstArrayAttr::print(::mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printStrippedAttrOrType(getValue());
  if (getValue().isa<ArrayAttr>()) {
    printer << ' ' << ":";
    printer << ' ';
    printer.printStrippedAttrOrType(getType());
  }
  printer << ">";
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/CIR/IR/CIROps.cpp.inc"
