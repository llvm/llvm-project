//===- TileReducerOps.cpp - TileReducer operations --------------*- C++ -*-===//

#include "TileReducer/TileReducerOps.h"
#include "TileReducer/TileReducerDialect.h"
#include "TileReducer/TileReducerTypes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::tr;

#define GET_OP_CLASSES
#include "TileReducer/TileReducerOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static LogicalResult verifyAxis(Operation *op, int64_t axis, int64_t rank,
                                StringRef what) {
  if (axis < 0 || axis >= rank)
    return op->emitOpError("axis ") << axis << " is out of range for " << what
                                    << " of rank " << rank;
  return success();
}

static LogicalResult verifyTileCoords(Operation *op, Type bufferType,
                                      ValueRange indices, TileType tile,
                                      StringRef kind) {
  int64_t rank = -1;
  Type elem;
  if (auto buffer = dyn_cast<BufferType>(bufferType)) {
    rank = buffer.getRank();
    elem = buffer.getElementType();
  } else if (auto memref = dyn_cast<MemRefType>(bufferType)) {
    rank = memref.getRank();
    elem = memref.getElementType();
  } else {
    return op->emitOpError("expected !tr.buffer or memref, got ") << bufferType;
  }
  if (static_cast<int64_t>(indices.size()) != rank)
    return op->emitOpError("expected ")
           << rank << " tile " << kind << " indices, got " << indices.size();
  if (tile.getRank() != rank)
    return op->emitOpError("tile rank ")
           << tile.getRank() << " does not match buffer rank " << rank;
  if (tile.getElementType() != elem)
    return op->emitOpError("tile element type ")
           << tile.getElementType() << " does not match buffer element type "
           << elem;
  return success();
}

//===----------------------------------------------------------------------===//
// ProgramIdOp
//===----------------------------------------------------------------------===//

LogicalResult ProgramIdOp::verify() {
  if (getAxis() < 0)
    return emitOpError("axis must be non-negative");
  return success();
}

//===----------------------------------------------------------------------===//
// DimOp
//===----------------------------------------------------------------------===//

LogicalResult DimOp::verify() {
  auto buffer = dyn_cast<BufferType>(getBuffer().getType());
  if (!buffer)
    return emitOpError("expected a !tr.buffer, got ") << getBuffer().getType();
  return verifyAxis(*this, getAxis(), buffer.getRank(), "buffer");
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder &b = parser.getBuilder();
  Attribute value;
  double d = 0.0;
  int64_t i = 0;
  // Parse a bare literal so `0.0 : !tr.tile<...>` does not treat the tile
  // type as the attribute's type.
  if (succeeded(parser.parseFloat(d)))
    value = b.getF64FloatAttr(d);
  else if (OptionalParseResult ir = parser.parseOptionalInteger(i);
           ir.has_value() && succeeded(*ir))
    value = b.getI64IntegerAttr(i);
  else
    return parser.emitError(parser.getNameLoc(),
                            "expected a scalar integer or float literal");

  Type resultType;
  if (parser.parseColonType(resultType))
    return failure();
  result.addAttribute("value", value);
  result.addTypes(resultType);
  return parser.parseOptionalAttrDict(result.attributes);
}

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getValue());
  p << " : " << getType();
}

LogicalResult ConstantOp::verify() {
  auto tile = dyn_cast<TileType>(getType());
  if (!tile)
    return emitOpError("result must be a !tr.tile, got ") << getType();
  Type elem = tile.getElementType();
  Attribute value = getValue();
  if (auto typed = dyn_cast<TypedAttr>(value)) {
    Type attrTy = typed.getType();
    if (attrTy == elem)
      return success();
    // Allow a wider float attribute (0.0 : f64) to splat onto an f32 tile.
    if (isa<FloatType>(attrTy) && isa<FloatType>(elem))
      return success();
    if (isa<IntegerType>(attrTy) && isa<IntegerType>(elem))
      return success();
    return emitOpError("constant attribute type ")
           << attrTy << " is not compatible with tile element type " << elem;
  }
  return emitOpError("constant value must be a typed attribute");
}

//===----------------------------------------------------------------------===//
// LoadOp / StoreOp
//===----------------------------------------------------------------------===//

LogicalResult LoadOp::verify() {
  auto tile = dyn_cast<TileType>(getType());
  if (!tile)
    return emitOpError("load result must be a !tr.tile");
  return verifyTileCoords(*this, getBuffer().getType(), getIndices(), tile,
                          "load");
}

LogicalResult StoreOp::verify() {
  auto tile = dyn_cast<TileType>(getValue().getType());
  if (!tile)
    return emitOpError("store value must be a !tr.tile");
  // Rank-0 tiles are scalars. They store into a rank-1 buffer at one
  // index (one slot per program instance, or the single full-sum slot).
  if (tile.getRank() == 0) {
    int64_t rank = -1;
    Type elem;
    if (auto buffer = dyn_cast<BufferType>(getBuffer().getType())) {
      rank = buffer.getRank();
      elem = buffer.getElementType();
    } else if (auto memref = dyn_cast<MemRefType>(getBuffer().getType())) {
      rank = memref.getRank();
      elem = memref.getElementType();
    } else {
      return emitOpError("expected !tr.buffer or memref, got ")
             << getBuffer().getType();
    }
    if (rank != 1 || getIndices().size() != 1)
      return emitOpError("scalar tile store expects one index into a rank-1 "
                         "buffer");
    if (tile.getElementType() != elem)
      return emitOpError("tile element type ")
             << tile.getElementType() << " does not match buffer element type "
             << elem;
    return success();
  }
  return verifyTileCoords(*this, getBuffer().getType(), getIndices(), tile,
                          "store");
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

LogicalResult AddOp::verify() {
  auto lhs = dyn_cast<TileType>(getLhs().getType());
  auto rhs = dyn_cast<TileType>(getRhs().getType());
  if (!lhs || !rhs)
    return emitOpError("add requires !tr.tile operands");
  if (lhs.getShape() != rhs.getShape() ||
      lhs.getElementType() != rhs.getElementType())
    return emitOpError("operand tiles must have the same shape and element "
                       "type");
  return success();
}

//===----------------------------------------------------------------------===//
// ReduceSumOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceSumOp::verify() {
  auto inTy = dyn_cast<TileType>(getInput().getType());
  auto outTy = dyn_cast<TileType>(getType());
  if (!inTy || !outTy)
    return emitOpError("reduce_sum requires !tr.tile input and result");
  int64_t axis = getAxis();

  if (failed(verifyAxis(*this, axis, inTy.getRank(), "input tile")))
    return failure();
  if (outTy.getRank() != inTy.getRank() - 1)
    return emitOpError("result rank must be ")
           << (inTy.getRank() - 1) << ", got " << outTy.getRank();
  if (outTy.getElementType() != inTy.getElementType())
    return emitOpError("element type mismatch");

  SmallVector<int64_t> expected;
  for (int64_t i = 0, e = inTy.getRank(); i < e; ++i)
    if (i != axis)
      expected.push_back(inTy.getDimSize(i));
  if (outTy.getShape() != ArrayRef<int64_t>(expected))
    return emitOpError("result extent must match the non-reduced dimensions, "
                       "expected ")
           << TileType::get(expected, outTy.getElementType()) << " got "
           << outTy;
  return success();
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

std::optional<int64_t> ForOp::getConstantLowerBound() {
  if (std::optional<uint64_t> attr = getStaticLowerBound())
    return static_cast<int64_t>(*attr);
  return std::nullopt;
}

std::optional<int64_t> ForOp::getConstantStep() {
  if (std::optional<uint64_t> attr = getStaticStep())
    return static_cast<int64_t>(*attr);
  return std::nullopt;
}

LogicalResult ForOp::verify() {
  if (getLowerBound() && getStaticLowerBound())
    return emitOpError("lower bound cannot be both an SSA value and a literal");
  if (!getLowerBound() && !getStaticLowerBound())
    return emitOpError("missing lower bound");
  if (getStep() && getStaticStep())
    return emitOpError("step cannot be both an SSA value and a literal");
  if (!getStep() && !getStaticStep())
    return emitOpError("missing step");
  if (auto step = getConstantStep(); step && *step <= 0)
    return emitOpError("step must be positive");

  if (getInitArgs().size() != getResults().size())
    return emitOpError("init_args count must match result count");

  Region &region = getRegion();
  if (region.empty())
    return emitOpError("expected a body region");
  if (region.getNumArguments() != 1 + getInitArgs().size())
    return emitOpError("body must have the induction variable plus one "
                       "argument per iter_arg");
  if (!region.getArgument(0).getType().isIndex())
    return emitOpError("induction variable must be index");

  for (auto [arg, init, res] :
       llvm::zip(getRegionIterArgs(), getInitArgs(), getResults())) {
    if (arg.getType() != init.getType() || arg.getType() != res.getType())
      return emitOpError("iter_arg / init / result types must match");
  }
  return success();
}

static ParseResult parseBound(OpAsmParser &parser, OperationState &result,
                              StringRef operandName, StringRef attrName,
                              Type indexTy, bool optionalOperand) {
  int64_t lit = 0;
  OptionalParseResult intRes = parser.parseOptionalInteger(lit);
  if (intRes.has_value()) {
    if (failed(*intRes))
      return failure();
    result.addAttribute(attrName, parser.getBuilder().getI64IntegerAttr(lit));
    return success();
  }
  OpAsmParser::UnresolvedOperand operand;
  if (parser.parseOperand(operand) ||
      parser.resolveOperand(operand, indexTy, result.operands))
    return failure();
  return success();
}

ParseResult ForOp::parse(OpAsmParser &parser, OperationState &result) {
  auto indexTy = parser.getBuilder().getIndexType();

  OpAsmParser::Argument ivArg;
  if (parser.parseArgument(ivArg) || parser.parseEqual())
    return failure();
  ivArg.type = indexTy;

  // lowerBound
  if (failed(parseBound(parser, result, "lowerBound", "staticLowerBound",
                        indexTy, /*optionalOperand=*/true)))
    return failure();
  if (parser.parseKeyword("to"))
    return failure();

  // upperBound is always an operand (canonical: %num_k_tiles) or a literal.
  int64_t ubLit = 0;
  OptionalParseResult ubInt = parser.parseOptionalInteger(ubLit);
  if (ubInt.has_value()) {
    if (failed(*ubInt))
      return failure();
    // Materialise a static upper bound as an attribute on the op; the
    // verifier/lowering treat a missing SSA upperBound as illegal, so create
    // a constant-like attribute and require an operand. For literals we store
    // them in `staticUpperBound` — but the ODS only has an SSA upperBound.
    // Accept only SSA for the upper bound to keep the operand list stable,
    // except we already parsed an integer. Re-parse as: reject literals for
    // ub except by creating an unresolved... Use a dedicated attribute.
    result.addAttribute("staticUpperBound",
                        parser.getBuilder().getIndexAttr(ubLit));
    return parser.emitError(parser.getNameLoc(),
                            "integer upper bounds are not supported; use an "
                            "SSA value (arith.constant)");
  }
  OpAsmParser::UnresolvedOperand ub;
  if (parser.parseOperand(ub) ||
      parser.resolveOperand(ub, indexTy, result.operands))
    return failure();

  if (parser.parseKeyword("step") ||
      failed(parseBound(parser, result, "step", "staticStep", indexTy, true)))
    return failure();

  // Optional iter_args(%a = %init) -> type
  SmallVector<OpAsmParser::Argument> regionArgs;
  regionArgs.push_back(ivArg);
  SmallVector<OpAsmParser::UnresolvedOperand> initOperands;
  SmallVector<Type> resultTypes;

  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    if (parser.parseAssignmentList(regionArgs, initOperands) ||
        parser.parseArrowTypeList(resultTypes))
      return failure();
    if (initOperands.size() + 1 != regionArgs.size())
      return parser.emitError(parser.getNameLoc(),
                              "iter_args assignment count mismatch");
    for (auto [arg, ty] : llvm::zip(llvm::drop_begin(regionArgs), resultTypes))
      arg.type = ty;
    if (parser.resolveOperands(initOperands, resultTypes, parser.getNameLoc(),
                               result.operands))
      return failure();
    result.addTypes(resultTypes);
  }

  int32_t hasLb = result.attributes.get("staticLowerBound") ? 0 : 1;
  int32_t hasStep = result.attributes.get("staticStep") ? 0 : 1;
  // Operands so far: [lb?] ub [step?]  -- but step SSA is parsed before
  // init_args, so count init as the remainder after lb+ub+step.
  int32_t nInit = static_cast<int32_t>(initOperands.size());
  // Reconstruct: if lb was SSA it was the first operand. ub is always
  // present. step SSA is next if not static.
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {hasLb, 1, hasStep, nInit}));

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();
  ForOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return parser.parseOptionalAttrDict(result.attributes);
}

void ForOp::print(OpAsmPrinter &p) {
  p << " " << getInductionVar() << " = ";
  if (auto lb = getConstantLowerBound())
    p << *lb;
  else
    p << getLowerBound();
  p << " to " << getUpperBound() << " step ";
  if (auto st = getConstantStep())
    p << *st;
  else
    p << getStep();

  if (!getInitArgs().empty()) {
    p << " iter_args(";
    llvm::interleaveComma(llvm::zip(getRegionIterArgs(), getInitArgs()), p,
                          [&](auto pair) {
                            p << std::get<0>(pair) << " = "
                              << std::get<1>(pair);
                          });
    p << ") -> (" << getResultTypes() << ")";
  }
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  p.printOptionalAttrDict((*this)->getAttrs(),
                          {"staticLowerBound", "staticStep",
                           "operandSegmentSizes"});
}
