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
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

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
// ConstantOp
//===----------------------------------------------------------------------===//

LogicalResult ConstantOp::verify() {
  auto opType = getType();
  auto val = getValue();
  auto valueType = val.getType();

  if (val.isa<NullAttr>()) {
    if (opType.isa<::mlir::cir::PointerType>())
      return success();
    return emitOpError("nullptr expects pointer type");
  }

  // ODS already generates checks to make sure the result type is valid. We just
  // need to additionally check that the value's attribute type is consistent
  // with the result type.
  if (val.isa<BoolAttr>()) {
    if (!opType.isa<mlir::cir::BoolType>())
      return emitOpError("result type (")
             << opType << ") must be '!cir.bool' for '" << val << "'";
    return success();
  }

  if (opType.isa<IntegerType, FloatType>()) {
    if (valueType != opType)
      return emitOpError("result type (")
             << opType << ") does not match value type (" << valueType << ")";
    return success();
  }

  return emitOpError("cannot have value of type ") << valueType;
}

static ParseResult parseConstantValue(OpAsmParser &parser,
                                      mlir::Attribute &valueAttr) {
  NamedAttrList attr;
  if (parser.parseAttribute(valueAttr, "value", attr))
    return ::mlir::failure();

  return success();
}

static void printConstantValue(OpAsmPrinter &p, cir::ConstantOp op,
                               Attribute value) {
  p.printAttribute(value);
}

OpFoldResult ConstantOp::fold(FoldAdaptor /*adaptor*/) { return getValue(); }

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

static mlir::LogicalResult checkReturnAndFunction(ReturnOp op,
                                                  FuncOp function) {
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
  while (!isa<FuncOp>(fnOp))
    fnOp = fnOp->getParentOp();

  // Make sure return types match function return type.
  if (checkReturnAndFunction(*this, cast<FuncOp>(fnOp)).failed())
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

ParseResult IfOp::parse(OpAsmParser &parser, OperationState &result) {
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

void IfOp::print(OpAsmPrinter &p) {
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

ParseResult ScopeOp::parse(OpAsmParser &parser, OperationState &result) {
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

void ScopeOp::print(OpAsmPrinter &p) {
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
  if (llvm::isa<SwitchOp>(getOperation()->getParentOp()))
    return mlir::success();

  // FIXME: check for cir.yield continue
  if (llvm::isa<LoopOp>(getOperation()->getParentOp()))
    return mlir::success();

  assert((llvm::isa<IfOp, ScopeOp>(getOperation()->getParentOp())) &&
         "unknown parent op");
  if (isFallthrough())
    return emitOpError() << "fallthrough only expected within 'cir.switch'";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// BinOp
//===----------------------------------------------------------------------===//

ParseResult parseBinOpKind(OpAsmParser &parser, BinOpKindAttr &kindAttr) {
  ::llvm::StringRef attrStr;
  ::mlir::NamedAttrList attrStorage;
  auto loc = parser.getCurrentLocation();

  // FIXME: since a few names can't be used as enum (and, or, xor) we declared
  // them in CIROps.td capitalized, but we really wanna use lower case on
  // clang IR asm form.
  if (parser.parseOptionalKeyword(&attrStr,
                                  {"mul", "div", "rem", "add", "sub", "shl",
                                   "shr", "and", "xor", "or"})) {
    ::mlir::StringAttr attrVal;
    ::mlir::OptionalParseResult parseResult = parser.parseOptionalAttribute(
        attrVal, parser.getBuilder().getNoneType(), "kind", attrStorage);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      attrStr = attrVal.getValue();
    } else {
      return parser.emitError(
          loc, "expected string or keyword containing one of the following "
               "enum values for attribute 'kind' [mul, div, rem, add, sub, "
               "shl, shr, and, xor, or]");
    }
  }
  if (!attrStr.empty()) {
    std::string attrString = attrStr.str();
    attrString[0] = attrString[0] + 'A' - 'a';
    attrStr = attrString;
    auto attrOptional = ::mlir::cir::symbolizeBinOpKind(attrStr);
    if (!attrOptional)
      return parser.emitError(loc, "invalid ")
             << "kind attribute specification: \"" << attrStr << '"';
    ;

    kindAttr = ::mlir::cir::BinOpKindAttr::get(parser.getBuilder().getContext(),
                                               attrOptional.value());
  }

  return ::mlir::success();
}

void printBinOpKind(OpAsmPrinter &p, BinOp binOp, BinOpKindAttr kindAttr) {
  auto caseValueStr = stringifyBinOpKind(kindAttr.getValue());
  std::string attrString = caseValueStr.str();
  attrString[0] = attrString[0] + 'a' - 'A';
  caseValueStr = attrString;
  p << caseValueStr;
}

//===----------------------------------------------------------------------===//
// BrOp
//===----------------------------------------------------------------------===//

mlir::SuccessorOperands BrOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  // Current block targets do not have operands.
  // TODO(CIR): This is a hacky avoidance of actually implementing this since
  // MLIR moved it "because nobody used the std::optional::None case.........."
  return mlir::SuccessorOperands(MutableOperandRange(getOperation(), 0, 0));
}

Block *BrOp::getSuccessorForOperands(ArrayRef<Attribute>) { return getDest(); }

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

    // FIXME: since a few names can't be used as enum (default) we declared
    // them in CIROps.td capitalized, but we really wanna use lower case on
    // clang IR asm form.
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

    std::string attrString = attrStr.str();
    attrString[0] = attrString[0] + 'A' - 'a';
    attrStr = attrString;
    auto attrOptional = ::mlir::cir::symbolizeCaseOpKind(attrStr);
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
      auto result = parser.parseCommaSeparatedList([&]() {
        int64_t val = 0;
        if (parser.parseInteger(val).failed())
          return ::mlir::failure();
        caseEltValueListAttr.push_back(
            mlir::IntegerAttr::get(intCondType, val));
        return ::mlir::success();
      });
      if (result.failed())
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
    auto caseValueStr = stringifyCaseOpKind(kind);
    std::string attrString = caseValueStr.str();
    attrString[0] = attrString[0] + 'a' - 'A';
    caseValueStr = attrString;
    p << caseValueStr;

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

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/CIR/IR/CIROps.cpp.inc"
