//===- OpenMPDialect.cpp - MLIR Dialect for OpenMP implementation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the OpenMP dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/OpenMP/OpenMPClauseOperands.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FoldInterfaces.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/bit.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Support/InterleavedRange.h"
#include <cstddef>
#include <iterator>
#include <optional>
#include <variant>

#include "mlir/Dialect/OpenMP/OpenMPOpsDialect.cpp.inc"
#include "mlir/Dialect/OpenMP/OpenMPOpsEnums.cpp.inc"
#include "mlir/Dialect/OpenMP/OpenMPOpsInterfaces.cpp.inc"
#include "mlir/Dialect/OpenMP/OpenMPTypeInterfaces.cpp.inc"

using namespace mlir;
using namespace mlir::omp;

static ArrayAttr makeArrayAttr(MLIRContext *context,
                               llvm::ArrayRef<Attribute> attrs) {
  return attrs.empty() ? nullptr : ArrayAttr::get(context, attrs);
}

static DenseBoolArrayAttr
makeDenseBoolArrayAttr(MLIRContext *ctx, const ArrayRef<bool> boolArray) {
  return boolArray.empty() ? nullptr : DenseBoolArrayAttr::get(ctx, boolArray);
}

static DenseI64ArrayAttr
makeDenseI64ArrayAttr(MLIRContext *ctx, const ArrayRef<int64_t> intArray) {
  return intArray.empty() ? nullptr : DenseI64ArrayAttr::get(ctx, intArray);
}

namespace {
struct MemRefPointerLikeModel
    : public PointerLikeType::ExternalModel<MemRefPointerLikeModel,
                                            MemRefType> {
  Type getElementType(Type pointer) const {
    return llvm::cast<MemRefType>(pointer).getElementType();
  }
};

struct LLVMPointerPointerLikeModel
    : public PointerLikeType::ExternalModel<LLVMPointerPointerLikeModel,
                                            LLVM::LLVMPointerType> {
  Type getElementType(Type pointer) const { return Type(); }
};
} // namespace

/// Generate a name of a canonical loop nest of the format
/// `<prefix>(_r<idx>_s<idx>)*`. Hereby, `_r<idx>` identifies the region
/// argument index of an operation that has multiple regions, if the operation
/// has multiple regions.
/// `_s<idx>` identifies the position of an operation within a region, where
/// only operations that may potentially contain loops ("container operations"
/// i.e. have region arguments) are counted. Again, it is omitted if there is
/// only one such operation in a region. If there are canonical loops nested
/// inside each other, also may also use the format `_d<num>` where <num> is the
/// nesting depth of the loop.
///
/// The generated name is a best-effort to make canonical loop unique within an
/// SSA namespace. This also means that regions with IsolatedFromAbove property
/// do not consider any parents or siblings.
static std::string generateLoopNestingName(StringRef prefix,
                                           CanonicalLoopOp op) {
  struct Component {
    /// If true, this component describes a region operand of an operation (the
    /// operand's owner) If false, this component describes an operation located
    /// in a parent region
    bool isRegionArgOfOp;
    bool skip = false;
    bool isUnique = false;

    size_t idx;
    Operation *op;
    Region *parentRegion;
    size_t loopDepth;

    Operation *&getOwnerOp() {
      assert(isRegionArgOfOp && "Must describe a region operand");
      return op;
    }
    size_t &getArgIdx() {
      assert(isRegionArgOfOp && "Must describe a region operand");
      return idx;
    }

    Operation *&getContainerOp() {
      assert(!isRegionArgOfOp && "Must describe a operation of a region");
      return op;
    }
    size_t &getOpPos() {
      assert(!isRegionArgOfOp && "Must describe a operation of a region");
      return idx;
    }
    bool isLoopOp() const {
      assert(!isRegionArgOfOp && "Must describe a operation of a region");
      return isa<CanonicalLoopOp>(op);
    }
    Region *&getParentRegion() {
      assert(!isRegionArgOfOp && "Must describe a operation of a region");
      return parentRegion;
    }
    size_t &getLoopDepth() {
      assert(!isRegionArgOfOp && "Must describe a operation of a region");
      return loopDepth;
    }

    void skipIf(bool v = true) { skip = skip || v; }
  };

  // List of ancestors, from inner to outer.
  // Alternates between
  //  * region argument of an operation
  //  * operation within a region
  SmallVector<Component> components;

  // Gather a list of parent regions and operations, and the position within
  // their parent
  Operation *o = op.getOperation();
  while (o) {
    // Operation within a region
    Region *r = o->getParentRegion();
    if (!r)
      break;

    llvm::ReversePostOrderTraversal<Block *> traversal(&r->getBlocks().front());
    size_t idx = 0;
    bool found = false;
    size_t sequentialIdx = -1;
    bool isOnlyContainerOp = true;
    for (Block *b : traversal) {
      for (Operation &op : *b) {
        if (&op == o && !found) {
          sequentialIdx = idx;
          found = true;
        }
        if (op.getNumRegions()) {
          idx += 1;
          if (idx > 1)
            isOnlyContainerOp = false;
        }
        if (found && !isOnlyContainerOp)
          break;
      }
    }

    Component &containerOpInRegion = components.emplace_back();
    containerOpInRegion.isRegionArgOfOp = false;
    containerOpInRegion.isUnique = isOnlyContainerOp;
    containerOpInRegion.getContainerOp() = o;
    containerOpInRegion.getOpPos() = sequentialIdx;
    containerOpInRegion.getParentRegion() = r;

    Operation *parent = r->getParentOp();

    // Region argument of an operation
    Component &regionArgOfOperation = components.emplace_back();
    regionArgOfOperation.isRegionArgOfOp = true;
    regionArgOfOperation.isUnique = true;
    regionArgOfOperation.getArgIdx() = 0;
    regionArgOfOperation.getOwnerOp() = parent;

    // The IsolatedFromAbove trait of the parent operation implies that each
    // individual region argument has its own separate namespace, so no
    // ambiguity.
    if (!parent || parent->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>())
      break;

    // Component only needed if operation has multiple region operands. Region
    // arguments may be optional, but we currently do not consider this.
    if (parent->getRegions().size() > 1) {
      auto getRegionIndex = [](Operation *o, Region *r) {
        for (auto [idx, region] : llvm::enumerate(o->getRegions())) {
          if (&region == r)
            return idx;
        }
        llvm_unreachable("Region not child of its parent operation");
      };
      regionArgOfOperation.isUnique = false;
      regionArgOfOperation.getArgIdx() = getRegionIndex(parent, r);
    }

    // next parent
    o = parent;
  }

  // Determine whether a region-argument component is not needed
  for (Component &c : components)
    c.skipIf(c.isRegionArgOfOp && c.isUnique);

  // Find runs of nested loops and determine each loop's depth in the loop nest
  size_t numSurroundingLoops = 0;
  for (Component &c : llvm::reverse(components)) {
    if (c.skip)
      continue;

    // non-skipped multi-argument operands interrupt the loop nest
    if (c.isRegionArgOfOp) {
      numSurroundingLoops = 0;
      continue;
    }

    // Multiple loops in a region means each of them is the outermost loop of a
    // new loop nest
    if (!c.isUnique)
      numSurroundingLoops = 0;

    c.getLoopDepth() = numSurroundingLoops;

    // Next loop is surrounded by one more loop
    if (isa<CanonicalLoopOp>(c.getContainerOp()))
      numSurroundingLoops += 1;
  }

  // In loop nests, skip all but the innermost loop that contains the depth
  // number
  bool isLoopNest = false;
  for (Component &c : components) {
    if (c.skip || c.isRegionArgOfOp)
      continue;

    if (!isLoopNest && c.getLoopDepth() >= 1) {
      // Innermost loop of a loop nest of at least two loops
      isLoopNest = true;
    } else if (isLoopNest) {
      // Non-innermost loop of a loop nest
      c.skipIf(c.isUnique);

      // If there is no surrounding loop left, this must have been the outermost
      // loop; leave loop-nest mode for the next iteration
      if (c.getLoopDepth() == 0)
        isLoopNest = false;
    }
  }

  // Skip non-loop unambiguous regions (but they should interrupt loop nests, so
  // we mark them as skipped only after computing loop nests)
  for (Component &c : components)
    c.skipIf(!c.isRegionArgOfOp && c.isUnique &&
             !isa<CanonicalLoopOp>(c.getContainerOp()));

  // Components can be skipped if they are already disambiguated by their parent
  // (or does not have a parent)
  bool newRegion = true;
  for (Component &c : llvm::reverse(components)) {
    c.skipIf(newRegion && c.isUnique);

    // non-skipped components disambiguate unique children
    if (!c.skip)
      newRegion = true;

    // ...except canonical loops that need a suffix for each nest
    if (!c.isRegionArgOfOp && c.getContainerOp())
      newRegion = false;
  }

  // Compile the nesting name string
  SmallString<64> Name{prefix};
  llvm::raw_svector_ostream NameOS(Name);
  for (auto &c : llvm::reverse(components)) {
    if (c.skip)
      continue;

    if (c.isRegionArgOfOp)
      NameOS << "_r" << c.getArgIdx();
    else if (c.getLoopDepth() >= 1)
      NameOS << "_d" << c.getLoopDepth();
    else
      NameOS << "_s" << c.getOpPos();
  }

  return NameOS.str().str();
}

void OpenMPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/OpenMP/OpenMPOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/OpenMP/OpenMPOpsAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/OpenMP/OpenMPOpsTypes.cpp.inc"
      >();

  declarePromisedInterface<ConvertToLLVMPatternInterface, OpenMPDialect>();

  MemRefType::attachInterface<MemRefPointerLikeModel>(*getContext());
  LLVM::LLVMPointerType::attachInterface<LLVMPointerPointerLikeModel>(
      *getContext());

  // Attach default offload module interface to module op to access
  // offload functionality through
  mlir::ModuleOp::attachInterface<mlir::omp::OffloadModuleDefaultModel>(
      *getContext());

  // Attach default declare target interfaces to operations which can be marked
  // as declare target (Global Operations and Functions/Subroutines in dialects
  // that Fortran (or other languages that lower to MLIR) translates too
  mlir::LLVM::GlobalOp::attachInterface<
      mlir::omp::DeclareTargetDefaultModel<mlir::LLVM::GlobalOp>>(
      *getContext());
  mlir::LLVM::LLVMFuncOp::attachInterface<
      mlir::omp::DeclareTargetDefaultModel<mlir::LLVM::LLVMFuncOp>>(
      *getContext());
  mlir::func::FuncOp::attachInterface<
      mlir::omp::DeclareTargetDefaultModel<mlir::func::FuncOp>>(*getContext());
}

//===----------------------------------------------------------------------===//
// Parser and printer for Allocate Clause
//===----------------------------------------------------------------------===//

/// Parse an allocate clause with allocators and a list of operands with types.
///
/// allocate-operand-list :: = allocate-operand |
///                            allocator-operand `,` allocate-operand-list
/// allocate-operand :: = ssa-id-and-type -> ssa-id-and-type
/// ssa-id-and-type ::= ssa-id `:` type
static ParseResult parseAllocateAndAllocator(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &allocateVars,
    SmallVectorImpl<Type> &allocateTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &allocatorVars,
    SmallVectorImpl<Type> &allocatorTypes) {

  return parser.parseCommaSeparatedList([&]() {
    OpAsmParser::UnresolvedOperand operand;
    Type type;
    if (parser.parseOperand(operand) || parser.parseColonType(type))
      return failure();
    allocatorVars.push_back(operand);
    allocatorTypes.push_back(type);
    if (parser.parseArrow())
      return failure();
    if (parser.parseOperand(operand) || parser.parseColonType(type))
      return failure();

    allocateVars.push_back(operand);
    allocateTypes.push_back(type);
    return success();
  });
}

/// Print allocate clause
static void printAllocateAndAllocator(OpAsmPrinter &p, Operation *op,
                                      OperandRange allocateVars,
                                      TypeRange allocateTypes,
                                      OperandRange allocatorVars,
                                      TypeRange allocatorTypes) {
  for (unsigned i = 0; i < allocateVars.size(); ++i) {
    std::string separator = i == allocateVars.size() - 1 ? "" : ", ";
    p << allocatorVars[i] << " : " << allocatorTypes[i] << " -> ";
    p << allocateVars[i] << " : " << allocateTypes[i] << separator;
  }
}

//===----------------------------------------------------------------------===//
// Parser and printer for a clause attribute (StringEnumAttr)
//===----------------------------------------------------------------------===//

template <typename ClauseAttr>
static ParseResult parseClauseAttr(AsmParser &parser, ClauseAttr &attr) {
  using ClauseT = decltype(std::declval<ClauseAttr>().getValue());
  StringRef enumStr;
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&enumStr))
    return failure();
  if (std::optional<ClauseT> enumValue = symbolizeEnum<ClauseT>(enumStr)) {
    attr = ClauseAttr::get(parser.getContext(), *enumValue);
    return success();
  }
  return parser.emitError(loc, "invalid clause value: '") << enumStr << "'";
}

template <typename ClauseAttr>
static void printClauseAttr(OpAsmPrinter &p, Operation *op, ClauseAttr attr) {
  p << stringifyEnum(attr.getValue());
}

//===----------------------------------------------------------------------===//
// Parser and printer for Linear Clause
//===----------------------------------------------------------------------===//

/// linear ::= `linear` `(` linear-list `)`
/// linear-list := linear-val | linear-val linear-list
/// linear-val := ssa-id-and-type `=` ssa-id-and-type
static ParseResult parseLinearClause(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &linearVars,
    SmallVectorImpl<Type> &linearTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &linearStepVars) {
  return parser.parseCommaSeparatedList([&]() {
    OpAsmParser::UnresolvedOperand var;
    Type type;
    OpAsmParser::UnresolvedOperand stepVar;
    if (parser.parseOperand(var) || parser.parseEqual() ||
        parser.parseOperand(stepVar) || parser.parseColonType(type))
      return failure();

    linearVars.push_back(var);
    linearTypes.push_back(type);
    linearStepVars.push_back(stepVar);
    return success();
  });
}

/// Print Linear Clause
static void printLinearClause(OpAsmPrinter &p, Operation *op,
                              ValueRange linearVars, TypeRange linearTypes,
                              ValueRange linearStepVars) {
  size_t linearVarsSize = linearVars.size();
  for (unsigned i = 0; i < linearVarsSize; ++i) {
    std::string separator = i == linearVarsSize - 1 ? "" : ", ";
    p << linearVars[i];
    if (linearStepVars.size() > i)
      p << " = " << linearStepVars[i];
    p << " : " << linearVars[i].getType() << separator;
  }
}

//===----------------------------------------------------------------------===//
// Verifier for Nontemporal Clause
//===----------------------------------------------------------------------===//

static LogicalResult verifyNontemporalClause(Operation *op,
                                             OperandRange nontemporalVars) {

  // Check if each var is unique - OpenMP 5.0 -> 2.9.3.1 section
  DenseSet<Value> nontemporalItems;
  for (const auto &it : nontemporalVars)
    if (!nontemporalItems.insert(it).second)
      return op->emitOpError() << "nontemporal variable used more than once";

  return success();
}

//===----------------------------------------------------------------------===//
// Parser, verifier and printer for Aligned Clause
//===----------------------------------------------------------------------===//
static LogicalResult verifyAlignedClause(Operation *op,
                                         std::optional<ArrayAttr> alignments,
                                         OperandRange alignedVars) {
  // Check if number of alignment values equals to number of aligned variables
  if (!alignedVars.empty()) {
    if (!alignments || alignments->size() != alignedVars.size())
      return op->emitOpError()
             << "expected as many alignment values as aligned variables";
  } else {
    if (alignments)
      return op->emitOpError() << "unexpected alignment values attribute";
    return success();
  }

  // Check if each var is aligned only once - OpenMP 4.5 -> 2.8.1 section
  DenseSet<Value> alignedItems;
  for (auto it : alignedVars)
    if (!alignedItems.insert(it).second)
      return op->emitOpError() << "aligned variable used more than once";

  if (!alignments)
    return success();

  // Check if all alignment values are positive - OpenMP 4.5 -> 2.8.1 section
  for (unsigned i = 0; i < (*alignments).size(); ++i) {
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>((*alignments)[i])) {
      if (intAttr.getValue().sle(0))
        return op->emitOpError() << "alignment should be greater than 0";
    } else {
      return op->emitOpError() << "expected integer alignment";
    }
  }

  return success();
}

/// aligned ::= `aligned` `(` aligned-list `)`
/// aligned-list := aligned-val | aligned-val aligned-list
/// aligned-val := ssa-id-and-type `->` alignment
static ParseResult
parseAlignedClause(OpAsmParser &parser,
                   SmallVectorImpl<OpAsmParser::UnresolvedOperand> &alignedVars,
                   SmallVectorImpl<Type> &alignedTypes,
                   ArrayAttr &alignmentsAttr) {
  SmallVector<Attribute> alignmentVec;
  if (failed(parser.parseCommaSeparatedList([&]() {
        if (parser.parseOperand(alignedVars.emplace_back()) ||
            parser.parseColonType(alignedTypes.emplace_back()) ||
            parser.parseArrow() ||
            parser.parseAttribute(alignmentVec.emplace_back())) {
          return failure();
        }
        return success();
      })))
    return failure();
  SmallVector<Attribute> alignments(alignmentVec.begin(), alignmentVec.end());
  alignmentsAttr = ArrayAttr::get(parser.getContext(), alignments);
  return success();
}

/// Print Aligned Clause
static void printAlignedClause(OpAsmPrinter &p, Operation *op,
                               ValueRange alignedVars, TypeRange alignedTypes,
                               std::optional<ArrayAttr> alignments) {
  for (unsigned i = 0; i < alignedVars.size(); ++i) {
    if (i != 0)
      p << ", ";
    p << alignedVars[i] << " : " << alignedVars[i].getType();
    p << " -> " << (*alignments)[i];
  }
}

//===----------------------------------------------------------------------===//
// Parser, printer and verifier for Schedule Clause
//===----------------------------------------------------------------------===//

static ParseResult
verifyScheduleModifiers(OpAsmParser &parser,
                        SmallVectorImpl<SmallString<12>> &modifiers) {
  if (modifiers.size() > 2)
    return parser.emitError(parser.getNameLoc()) << " unexpected modifier(s)";
  for (const auto &mod : modifiers) {
    // Translate the string. If it has no value, then it was not a valid
    // modifier!
    auto symbol = symbolizeScheduleModifier(mod);
    if (!symbol)
      return parser.emitError(parser.getNameLoc())
             << " unknown modifier type: " << mod;
  }

  // If we have one modifier that is "simd", then stick a "none" modiifer in
  // index 0.
  if (modifiers.size() == 1) {
    if (symbolizeScheduleModifier(modifiers[0]) == ScheduleModifier::simd) {
      modifiers.push_back(modifiers[0]);
      modifiers[0] = stringifyScheduleModifier(ScheduleModifier::none);
    }
  } else if (modifiers.size() == 2) {
    // If there are two modifier:
    // First modifier should not be simd, second one should be simd
    if (symbolizeScheduleModifier(modifiers[0]) == ScheduleModifier::simd ||
        symbolizeScheduleModifier(modifiers[1]) != ScheduleModifier::simd)
      return parser.emitError(parser.getNameLoc())
             << " incorrect modifier order";
  }
  return success();
}

/// schedule ::= `schedule` `(` sched-list `)`
/// sched-list ::= sched-val | sched-val sched-list |
///                sched-val `,` sched-modifier
/// sched-val ::= sched-with-chunk | sched-wo-chunk
/// sched-with-chunk ::= sched-with-chunk-types (`=` ssa-id-and-type)?
/// sched-with-chunk-types ::= `static` | `dynamic` | `guided`
/// sched-wo-chunk ::=  `auto` | `runtime`
/// sched-modifier ::=  sched-mod-val | sched-mod-val `,` sched-mod-val
/// sched-mod-val ::=  `monotonic` | `nonmonotonic` | `simd` | `none`
static ParseResult
parseScheduleClause(OpAsmParser &parser, ClauseScheduleKindAttr &scheduleAttr,
                    ScheduleModifierAttr &scheduleMod, UnitAttr &scheduleSimd,
                    std::optional<OpAsmParser::UnresolvedOperand> &chunkSize,
                    Type &chunkType) {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return failure();
  std::optional<mlir::omp::ClauseScheduleKind> schedule =
      symbolizeClauseScheduleKind(keyword);
  if (!schedule)
    return parser.emitError(parser.getNameLoc()) << " expected schedule kind";

  scheduleAttr = ClauseScheduleKindAttr::get(parser.getContext(), *schedule);
  switch (*schedule) {
  case ClauseScheduleKind::Static:
  case ClauseScheduleKind::Dynamic:
  case ClauseScheduleKind::Guided:
    if (succeeded(parser.parseOptionalEqual())) {
      chunkSize = OpAsmParser::UnresolvedOperand{};
      if (parser.parseOperand(*chunkSize) || parser.parseColonType(chunkType))
        return failure();
    } else {
      chunkSize = std::nullopt;
    }
    break;
  case ClauseScheduleKind::Auto:
  case ClauseScheduleKind::Runtime:
    chunkSize = std::nullopt;
  }

  // If there is a comma, we have one or more modifiers..
  SmallVector<SmallString<12>> modifiers;
  while (succeeded(parser.parseOptionalComma())) {
    StringRef mod;
    if (parser.parseKeyword(&mod))
      return failure();
    modifiers.push_back(mod);
  }

  if (verifyScheduleModifiers(parser, modifiers))
    return failure();

  if (!modifiers.empty()) {
    SMLoc loc = parser.getCurrentLocation();
    if (std::optional<ScheduleModifier> mod =
            symbolizeScheduleModifier(modifiers[0])) {
      scheduleMod = ScheduleModifierAttr::get(parser.getContext(), *mod);
    } else {
      return parser.emitError(loc, "invalid schedule modifier");
    }
    // Only SIMD attribute is allowed here!
    if (modifiers.size() > 1) {
      assert(symbolizeScheduleModifier(modifiers[1]) == ScheduleModifier::simd);
      scheduleSimd = UnitAttr::get(parser.getBuilder().getContext());
    }
  }

  return success();
}

/// Print schedule clause
static void printScheduleClause(OpAsmPrinter &p, Operation *op,
                                ClauseScheduleKindAttr scheduleKind,
                                ScheduleModifierAttr scheduleMod,
                                UnitAttr scheduleSimd, Value scheduleChunk,
                                Type scheduleChunkType) {
  p << stringifyClauseScheduleKind(scheduleKind.getValue());
  if (scheduleChunk)
    p << " = " << scheduleChunk << " : " << scheduleChunk.getType();
  if (scheduleMod)
    p << ", " << stringifyScheduleModifier(scheduleMod.getValue());
  if (scheduleSimd)
    p << ", simd";
}

//===----------------------------------------------------------------------===//
// Parser and printer for Order Clause
//===----------------------------------------------------------------------===//

// order ::= `order` `(` [order-modiﬁer ':'] concurrent `)`
// order-modiﬁer ::= reproducible | unconstrained
static ParseResult parseOrderClause(OpAsmParser &parser,
                                    ClauseOrderKindAttr &order,
                                    OrderModifierAttr &orderMod) {
  StringRef enumStr;
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&enumStr))
    return failure();
  if (std::optional<OrderModifier> enumValue =
          symbolizeOrderModifier(enumStr)) {
    orderMod = OrderModifierAttr::get(parser.getContext(), *enumValue);
    if (parser.parseOptionalColon())
      return failure();
    loc = parser.getCurrentLocation();
    if (parser.parseKeyword(&enumStr))
      return failure();
  }
  if (std::optional<ClauseOrderKind> enumValue =
          symbolizeClauseOrderKind(enumStr)) {
    order = ClauseOrderKindAttr::get(parser.getContext(), *enumValue);
    return success();
  }
  return parser.emitError(loc, "invalid clause value: '") << enumStr << "'";
}

static void printOrderClause(OpAsmPrinter &p, Operation *op,
                             ClauseOrderKindAttr order,
                             OrderModifierAttr orderMod) {
  if (orderMod)
    p << stringifyOrderModifier(orderMod.getValue()) << ":";
  if (order)
    p << stringifyClauseOrderKind(order.getValue());
}

template <typename ClauseTypeAttr, typename ClauseType>
static ParseResult
parseGranularityClause(OpAsmParser &parser, ClauseTypeAttr &prescriptiveness,
                       std::optional<OpAsmParser::UnresolvedOperand> &operand,
                       Type &operandType,
                       std::optional<ClauseType> (*symbolizeClause)(StringRef),
                       StringRef clauseName) {
  StringRef enumStr;
  if (succeeded(parser.parseOptionalKeyword(&enumStr))) {
    if (std::optional<ClauseType> enumValue = symbolizeClause(enumStr)) {
      prescriptiveness = ClauseTypeAttr::get(parser.getContext(), *enumValue);
      if (parser.parseComma())
        return failure();
    } else {
      return parser.emitError(parser.getCurrentLocation())
             << "invalid " << clauseName << " modifier : '" << enumStr << "'";
      ;
    }
  }

  OpAsmParser::UnresolvedOperand var;
  if (succeeded(parser.parseOperand(var))) {
    operand = var;
  } else {
    return parser.emitError(parser.getCurrentLocation())
           << "expected " << clauseName << " operand";
  }

  if (operand.has_value()) {
    if (parser.parseColonType(operandType))
      return failure();
  }

  return success();
}

template <typename ClauseTypeAttr, typename ClauseType>
static void
printGranularityClause(OpAsmPrinter &p, Operation *op,
                       ClauseTypeAttr prescriptiveness, Value operand,
                       mlir::Type operandType,
                       StringRef (*stringifyClauseType)(ClauseType)) {

  if (prescriptiveness)
    p << stringifyClauseType(prescriptiveness.getValue()) << ", ";

  if (operand)
    p << operand << ": " << operandType;
}

//===----------------------------------------------------------------------===//
// Parser and printer for grainsize Clause
//===----------------------------------------------------------------------===//

// grainsize ::= `grainsize` `(` [strict ':'] grain-size `)`
static ParseResult
parseGrainsizeClause(OpAsmParser &parser, ClauseGrainsizeTypeAttr &grainsizeMod,
                     std::optional<OpAsmParser::UnresolvedOperand> &grainsize,
                     Type &grainsizeType) {
  return parseGranularityClause<ClauseGrainsizeTypeAttr, ClauseGrainsizeType>(
      parser, grainsizeMod, grainsize, grainsizeType,
      &symbolizeClauseGrainsizeType, "grainsize");
}

static void printGrainsizeClause(OpAsmPrinter &p, Operation *op,
                                 ClauseGrainsizeTypeAttr grainsizeMod,
                                 Value grainsize, mlir::Type grainsizeType) {
  printGranularityClause<ClauseGrainsizeTypeAttr, ClauseGrainsizeType>(
      p, op, grainsizeMod, grainsize, grainsizeType,
      &stringifyClauseGrainsizeType);
}

//===----------------------------------------------------------------------===//
// Parser and printer for num_tasks Clause
//===----------------------------------------------------------------------===//

// numtask ::= `num_tasks` `(` [strict ':'] num-tasks `)`
static ParseResult
parseNumTasksClause(OpAsmParser &parser, ClauseNumTasksTypeAttr &numTasksMod,
                    std::optional<OpAsmParser::UnresolvedOperand> &numTasks,
                    Type &numTasksType) {
  return parseGranularityClause<ClauseNumTasksTypeAttr, ClauseNumTasksType>(
      parser, numTasksMod, numTasks, numTasksType, &symbolizeClauseNumTasksType,
      "num_tasks");
}

static void printNumTasksClause(OpAsmPrinter &p, Operation *op,
                                ClauseNumTasksTypeAttr numTasksMod,
                                Value numTasks, mlir::Type numTasksType) {
  printGranularityClause<ClauseNumTasksTypeAttr, ClauseNumTasksType>(
      p, op, numTasksMod, numTasks, numTasksType, &stringifyClauseNumTasksType);
}

//===----------------------------------------------------------------------===//
// Parsers for operations including clauses that define entry block arguments.
//===----------------------------------------------------------------------===//

namespace {
struct MapParseArgs {
  SmallVectorImpl<OpAsmParser::UnresolvedOperand> &vars;
  SmallVectorImpl<Type> &types;
  MapParseArgs(SmallVectorImpl<OpAsmParser::UnresolvedOperand> &vars,
               SmallVectorImpl<Type> &types)
      : vars(vars), types(types) {}
};
struct PrivateParseArgs {
  llvm::SmallVectorImpl<OpAsmParser::UnresolvedOperand> &vars;
  llvm::SmallVectorImpl<Type> &types;
  ArrayAttr &syms;
  UnitAttr &needsBarrier;
  DenseI64ArrayAttr *mapIndices;
  PrivateParseArgs(SmallVectorImpl<OpAsmParser::UnresolvedOperand> &vars,
                   SmallVectorImpl<Type> &types, ArrayAttr &syms,
                   UnitAttr &needsBarrier,
                   DenseI64ArrayAttr *mapIndices = nullptr)
      : vars(vars), types(types), syms(syms), needsBarrier(needsBarrier),
        mapIndices(mapIndices) {}
};

struct ReductionParseArgs {
  SmallVectorImpl<OpAsmParser::UnresolvedOperand> &vars;
  SmallVectorImpl<Type> &types;
  DenseBoolArrayAttr &byref;
  ArrayAttr &syms;
  ReductionModifierAttr *modifier;
  ReductionParseArgs(SmallVectorImpl<OpAsmParser::UnresolvedOperand> &vars,
                     SmallVectorImpl<Type> &types, DenseBoolArrayAttr &byref,
                     ArrayAttr &syms, ReductionModifierAttr *mod = nullptr)
      : vars(vars), types(types), byref(byref), syms(syms), modifier(mod) {}
};

struct AllRegionParseArgs {
  std::optional<MapParseArgs> hasDeviceAddrArgs;
  std::optional<MapParseArgs> hostEvalArgs;
  std::optional<ReductionParseArgs> inReductionArgs;
  std::optional<MapParseArgs> mapArgs;
  std::optional<PrivateParseArgs> privateArgs;
  std::optional<ReductionParseArgs> reductionArgs;
  std::optional<ReductionParseArgs> taskReductionArgs;
  std::optional<MapParseArgs> useDeviceAddrArgs;
  std::optional<MapParseArgs> useDevicePtrArgs;
};
} // namespace

static inline constexpr StringRef getPrivateNeedsBarrierSpelling() {
  return "private_barrier";
}

static ParseResult parseClauseWithRegionArgs(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &types,
    SmallVectorImpl<OpAsmParser::Argument> &regionPrivateArgs,
    ArrayAttr *symbols = nullptr, DenseI64ArrayAttr *mapIndices = nullptr,
    DenseBoolArrayAttr *byref = nullptr,
    ReductionModifierAttr *modifier = nullptr,
    UnitAttr *needsBarrier = nullptr) {
  SmallVector<SymbolRefAttr> symbolVec;
  SmallVector<int64_t> mapIndicesVec;
  SmallVector<bool> isByRefVec;
  unsigned regionArgOffset = regionPrivateArgs.size();

  if (parser.parseLParen())
    return failure();

  if (modifier && succeeded(parser.parseOptionalKeyword("mod"))) {
    StringRef enumStr;
    if (parser.parseColon() || parser.parseKeyword(&enumStr) ||
        parser.parseComma())
      return failure();
    std::optional<ReductionModifier> enumValue =
        symbolizeReductionModifier(enumStr);
    if (!enumValue.has_value())
      return failure();
    *modifier = ReductionModifierAttr::get(parser.getContext(), *enumValue);
    if (!*modifier)
      return failure();
  }

  if (parser.parseCommaSeparatedList([&]() {
        if (byref)
          isByRefVec.push_back(
              parser.parseOptionalKeyword("byref").succeeded());

        if (symbols && parser.parseAttribute(symbolVec.emplace_back()))
          return failure();

        if (parser.parseOperand(operands.emplace_back()) ||
            parser.parseArrow() ||
            parser.parseArgument(regionPrivateArgs.emplace_back()))
          return failure();

        if (mapIndices) {
          if (parser.parseOptionalLSquare().succeeded()) {
            if (parser.parseKeyword("map_idx") || parser.parseEqual() ||
                parser.parseInteger(mapIndicesVec.emplace_back()) ||
                parser.parseRSquare())
              return failure();
          } else {
            mapIndicesVec.push_back(-1);
          }
        }

        return success();
      }))
    return failure();

  if (parser.parseColon())
    return failure();

  if (parser.parseCommaSeparatedList([&]() {
        if (parser.parseType(types.emplace_back()))
          return failure();

        return success();
      }))
    return failure();

  if (operands.size() != types.size())
    return failure();

  if (parser.parseRParen())
    return failure();

  if (needsBarrier) {
    if (parser.parseOptionalKeyword(getPrivateNeedsBarrierSpelling())
            .succeeded())
      *needsBarrier = mlir::UnitAttr::get(parser.getContext());
  }

  auto *argsBegin = regionPrivateArgs.begin();
  MutableArrayRef argsSubrange(argsBegin + regionArgOffset,
                               argsBegin + regionArgOffset + types.size());
  for (auto [prv, type] : llvm::zip_equal(argsSubrange, types)) {
    prv.type = type;
  }

  if (symbols) {
    SmallVector<Attribute> symbolAttrs(symbolVec.begin(), symbolVec.end());
    *symbols = ArrayAttr::get(parser.getContext(), symbolAttrs);
  }

  if (!mapIndicesVec.empty())
    *mapIndices =
        mlir::DenseI64ArrayAttr::get(parser.getContext(), mapIndicesVec);

  if (byref)
    *byref = makeDenseBoolArrayAttr(parser.getContext(), isByRefVec);

  return success();
}

static ParseResult parseBlockArgClause(
    OpAsmParser &parser,
    llvm::SmallVectorImpl<OpAsmParser::Argument> &entryBlockArgs,
    StringRef keyword, std::optional<MapParseArgs> mapArgs) {
  if (succeeded(parser.parseOptionalKeyword(keyword))) {
    if (!mapArgs)
      return failure();

    if (failed(parseClauseWithRegionArgs(parser, mapArgs->vars, mapArgs->types,
                                         entryBlockArgs)))
      return failure();
  }
  return success();
}

static ParseResult parseBlockArgClause(
    OpAsmParser &parser,
    llvm::SmallVectorImpl<OpAsmParser::Argument> &entryBlockArgs,
    StringRef keyword, std::optional<PrivateParseArgs> privateArgs) {
  if (succeeded(parser.parseOptionalKeyword(keyword))) {
    if (!privateArgs)
      return failure();

    if (failed(parseClauseWithRegionArgs(
            parser, privateArgs->vars, privateArgs->types, entryBlockArgs,
            &privateArgs->syms, privateArgs->mapIndices, /*byref=*/nullptr,
            /*modifier=*/nullptr, &privateArgs->needsBarrier)))
      return failure();
  }
  return success();
}

static ParseResult parseBlockArgClause(
    OpAsmParser &parser,
    llvm::SmallVectorImpl<OpAsmParser::Argument> &entryBlockArgs,
    StringRef keyword, std::optional<ReductionParseArgs> reductionArgs) {
  if (succeeded(parser.parseOptionalKeyword(keyword))) {
    if (!reductionArgs)
      return failure();
    if (failed(parseClauseWithRegionArgs(
            parser, reductionArgs->vars, reductionArgs->types, entryBlockArgs,
            &reductionArgs->syms, /*mapIndices=*/nullptr, &reductionArgs->byref,
            reductionArgs->modifier)))
      return failure();
  }
  return success();
}

static ParseResult parseBlockArgRegion(OpAsmParser &parser, Region &region,
                                       AllRegionParseArgs args) {
  llvm::SmallVector<OpAsmParser::Argument> entryBlockArgs;

  if (failed(parseBlockArgClause(parser, entryBlockArgs, "has_device_addr",
                                 args.hasDeviceAddrArgs)))
    return parser.emitError(parser.getCurrentLocation())
           << "invalid `has_device_addr` format";

  if (failed(parseBlockArgClause(parser, entryBlockArgs, "host_eval",
                                 args.hostEvalArgs)))
    return parser.emitError(parser.getCurrentLocation())
           << "invalid `host_eval` format";

  if (failed(parseBlockArgClause(parser, entryBlockArgs, "in_reduction",
                                 args.inReductionArgs)))
    return parser.emitError(parser.getCurrentLocation())
           << "invalid `in_reduction` format";

  if (failed(parseBlockArgClause(parser, entryBlockArgs, "map_entries",
                                 args.mapArgs)))
    return parser.emitError(parser.getCurrentLocation())
           << "invalid `map_entries` format";

  if (failed(parseBlockArgClause(parser, entryBlockArgs, "private",
                                 args.privateArgs)))
    return parser.emitError(parser.getCurrentLocation())
           << "invalid `private` format";

  if (failed(parseBlockArgClause(parser, entryBlockArgs, "reduction",
                                 args.reductionArgs)))
    return parser.emitError(parser.getCurrentLocation())
           << "invalid `reduction` format";

  if (failed(parseBlockArgClause(parser, entryBlockArgs, "task_reduction",
                                 args.taskReductionArgs)))
    return parser.emitError(parser.getCurrentLocation())
           << "invalid `task_reduction` format";

  if (failed(parseBlockArgClause(parser, entryBlockArgs, "use_device_addr",
                                 args.useDeviceAddrArgs)))
    return parser.emitError(parser.getCurrentLocation())
           << "invalid `use_device_addr` format";

  if (failed(parseBlockArgClause(parser, entryBlockArgs, "use_device_ptr",
                                 args.useDevicePtrArgs)))
    return parser.emitError(parser.getCurrentLocation())
           << "invalid `use_device_addr` format";

  return parser.parseRegion(region, entryBlockArgs);
}

// These parseXyz functions correspond to the custom<Xyz> definitions
// in the .td file(s).
static ParseResult parseTargetOpRegion(
    OpAsmParser &parser, Region &region,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &hasDeviceAddrVars,
    SmallVectorImpl<Type> &hasDeviceAddrTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &hostEvalVars,
    SmallVectorImpl<Type> &hostEvalTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &inReductionVars,
    SmallVectorImpl<Type> &inReductionTypes,
    DenseBoolArrayAttr &inReductionByref, ArrayAttr &inReductionSyms,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &mapVars,
    SmallVectorImpl<Type> &mapTypes,
    llvm::SmallVectorImpl<OpAsmParser::UnresolvedOperand> &privateVars,
    llvm::SmallVectorImpl<Type> &privateTypes, ArrayAttr &privateSyms,
    UnitAttr &privateNeedsBarrier, DenseI64ArrayAttr &privateMaps) {
  AllRegionParseArgs args;
  args.hasDeviceAddrArgs.emplace(hasDeviceAddrVars, hasDeviceAddrTypes);
  args.hostEvalArgs.emplace(hostEvalVars, hostEvalTypes);
  args.inReductionArgs.emplace(inReductionVars, inReductionTypes,
                               inReductionByref, inReductionSyms);
  args.mapArgs.emplace(mapVars, mapTypes);
  args.privateArgs.emplace(privateVars, privateTypes, privateSyms,
                           privateNeedsBarrier, &privateMaps);
  return parseBlockArgRegion(parser, region, args);
}

static ParseResult parseInReductionPrivateRegion(
    OpAsmParser &parser, Region &region,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &inReductionVars,
    SmallVectorImpl<Type> &inReductionTypes,
    DenseBoolArrayAttr &inReductionByref, ArrayAttr &inReductionSyms,
    llvm::SmallVectorImpl<OpAsmParser::UnresolvedOperand> &privateVars,
    llvm::SmallVectorImpl<Type> &privateTypes, ArrayAttr &privateSyms,
    UnitAttr &privateNeedsBarrier) {
  AllRegionParseArgs args;
  args.inReductionArgs.emplace(inReductionVars, inReductionTypes,
                               inReductionByref, inReductionSyms);
  args.privateArgs.emplace(privateVars, privateTypes, privateSyms,
                           privateNeedsBarrier);
  return parseBlockArgRegion(parser, region, args);
}

static ParseResult parseInReductionPrivateReductionRegion(
    OpAsmParser &parser, Region &region,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &inReductionVars,
    SmallVectorImpl<Type> &inReductionTypes,
    DenseBoolArrayAttr &inReductionByref, ArrayAttr &inReductionSyms,
    llvm::SmallVectorImpl<OpAsmParser::UnresolvedOperand> &privateVars,
    llvm::SmallVectorImpl<Type> &privateTypes, ArrayAttr &privateSyms,
    UnitAttr &privateNeedsBarrier, ReductionModifierAttr &reductionMod,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &reductionVars,
    SmallVectorImpl<Type> &reductionTypes, DenseBoolArrayAttr &reductionByref,
    ArrayAttr &reductionSyms) {
  AllRegionParseArgs args;
  args.inReductionArgs.emplace(inReductionVars, inReductionTypes,
                               inReductionByref, inReductionSyms);
  args.privateArgs.emplace(privateVars, privateTypes, privateSyms,
                           privateNeedsBarrier);
  args.reductionArgs.emplace(reductionVars, reductionTypes, reductionByref,
                             reductionSyms, &reductionMod);
  return parseBlockArgRegion(parser, region, args);
}

static ParseResult parsePrivateRegion(
    OpAsmParser &parser, Region &region,
    llvm::SmallVectorImpl<OpAsmParser::UnresolvedOperand> &privateVars,
    llvm::SmallVectorImpl<Type> &privateTypes, ArrayAttr &privateSyms,
    UnitAttr &privateNeedsBarrier) {
  AllRegionParseArgs args;
  args.privateArgs.emplace(privateVars, privateTypes, privateSyms,
                           privateNeedsBarrier);
  return parseBlockArgRegion(parser, region, args);
}

static ParseResult parsePrivateReductionRegion(
    OpAsmParser &parser, Region &region,
    llvm::SmallVectorImpl<OpAsmParser::UnresolvedOperand> &privateVars,
    llvm::SmallVectorImpl<Type> &privateTypes, ArrayAttr &privateSyms,
    UnitAttr &privateNeedsBarrier, ReductionModifierAttr &reductionMod,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &reductionVars,
    SmallVectorImpl<Type> &reductionTypes, DenseBoolArrayAttr &reductionByref,
    ArrayAttr &reductionSyms) {
  AllRegionParseArgs args;
  args.privateArgs.emplace(privateVars, privateTypes, privateSyms,
                           privateNeedsBarrier);
  args.reductionArgs.emplace(reductionVars, reductionTypes, reductionByref,
                             reductionSyms, &reductionMod);
  return parseBlockArgRegion(parser, region, args);
}

static ParseResult parseTaskReductionRegion(
    OpAsmParser &parser, Region &region,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &taskReductionVars,
    SmallVectorImpl<Type> &taskReductionTypes,
    DenseBoolArrayAttr &taskReductionByref, ArrayAttr &taskReductionSyms) {
  AllRegionParseArgs args;
  args.taskReductionArgs.emplace(taskReductionVars, taskReductionTypes,
                                 taskReductionByref, taskReductionSyms);
  return parseBlockArgRegion(parser, region, args);
}

static ParseResult parseUseDeviceAddrUseDevicePtrRegion(
    OpAsmParser &parser, Region &region,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &useDeviceAddrVars,
    SmallVectorImpl<Type> &useDeviceAddrTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &useDevicePtrVars,
    SmallVectorImpl<Type> &useDevicePtrTypes) {
  AllRegionParseArgs args;
  args.useDeviceAddrArgs.emplace(useDeviceAddrVars, useDeviceAddrTypes);
  args.useDevicePtrArgs.emplace(useDevicePtrVars, useDevicePtrTypes);
  return parseBlockArgRegion(parser, region, args);
}

//===----------------------------------------------------------------------===//
// Printers for operations including clauses that define entry block arguments.
//===----------------------------------------------------------------------===//

namespace {
struct MapPrintArgs {
  ValueRange vars;
  TypeRange types;
  MapPrintArgs(ValueRange vars, TypeRange types) : vars(vars), types(types) {}
};
struct PrivatePrintArgs {
  ValueRange vars;
  TypeRange types;
  ArrayAttr syms;
  UnitAttr needsBarrier;
  DenseI64ArrayAttr mapIndices;
  PrivatePrintArgs(ValueRange vars, TypeRange types, ArrayAttr syms,
                   UnitAttr needsBarrier, DenseI64ArrayAttr mapIndices)
      : vars(vars), types(types), syms(syms), needsBarrier(needsBarrier),
        mapIndices(mapIndices) {}
};
struct ReductionPrintArgs {
  ValueRange vars;
  TypeRange types;
  DenseBoolArrayAttr byref;
  ArrayAttr syms;
  ReductionModifierAttr modifier;
  ReductionPrintArgs(ValueRange vars, TypeRange types, DenseBoolArrayAttr byref,
                     ArrayAttr syms, ReductionModifierAttr mod = nullptr)
      : vars(vars), types(types), byref(byref), syms(syms), modifier(mod) {}
};
struct AllRegionPrintArgs {
  std::optional<MapPrintArgs> hasDeviceAddrArgs;
  std::optional<MapPrintArgs> hostEvalArgs;
  std::optional<ReductionPrintArgs> inReductionArgs;
  std::optional<MapPrintArgs> mapArgs;
  std::optional<PrivatePrintArgs> privateArgs;
  std::optional<ReductionPrintArgs> reductionArgs;
  std::optional<ReductionPrintArgs> taskReductionArgs;
  std::optional<MapPrintArgs> useDeviceAddrArgs;
  std::optional<MapPrintArgs> useDevicePtrArgs;
};
} // namespace

static void printClauseWithRegionArgs(
    OpAsmPrinter &p, MLIRContext *ctx, StringRef clauseName,
    ValueRange argsSubrange, ValueRange operands, TypeRange types,
    ArrayAttr symbols = nullptr, DenseI64ArrayAttr mapIndices = nullptr,
    DenseBoolArrayAttr byref = nullptr,
    ReductionModifierAttr modifier = nullptr, UnitAttr needsBarrier = nullptr) {
  if (argsSubrange.empty())
    return;

  p << clauseName << "(";

  if (modifier)
    p << "mod: " << stringifyReductionModifier(modifier.getValue()) << ", ";

  if (!symbols) {
    llvm::SmallVector<Attribute> values(operands.size(), nullptr);
    symbols = ArrayAttr::get(ctx, values);
  }

  if (!mapIndices) {
    llvm::SmallVector<int64_t> values(operands.size(), -1);
    mapIndices = DenseI64ArrayAttr::get(ctx, values);
  }

  if (!byref) {
    mlir::SmallVector<bool> values(operands.size(), false);
    byref = DenseBoolArrayAttr::get(ctx, values);
  }

  llvm::interleaveComma(llvm::zip_equal(operands, argsSubrange, symbols,
                                        mapIndices.asArrayRef(),
                                        byref.asArrayRef()),
                        p, [&p](auto t) {
                          auto [op, arg, sym, map, isByRef] = t;
                          if (isByRef)
                            p << "byref ";
                          if (sym)
                            p << sym << " ";

                          p << op << " -> " << arg;

                          if (map != -1)
                            p << " [map_idx=" << map << "]";
                        });
  p << " : ";
  llvm::interleaveComma(types, p);
  p << ") ";

  if (needsBarrier)
    p << getPrivateNeedsBarrierSpelling() << " ";
}

static void printBlockArgClause(OpAsmPrinter &p, MLIRContext *ctx,
                                StringRef clauseName, ValueRange argsSubrange,
                                std::optional<MapPrintArgs> mapArgs) {
  if (mapArgs)
    printClauseWithRegionArgs(p, ctx, clauseName, argsSubrange, mapArgs->vars,
                              mapArgs->types);
}

static void printBlockArgClause(OpAsmPrinter &p, MLIRContext *ctx,
                                StringRef clauseName, ValueRange argsSubrange,
                                std::optional<PrivatePrintArgs> privateArgs) {
  if (privateArgs)
    printClauseWithRegionArgs(
        p, ctx, clauseName, argsSubrange, privateArgs->vars, privateArgs->types,
        privateArgs->syms, privateArgs->mapIndices, /*byref=*/nullptr,
        /*modifier=*/nullptr, privateArgs->needsBarrier);
}

static void
printBlockArgClause(OpAsmPrinter &p, MLIRContext *ctx, StringRef clauseName,
                    ValueRange argsSubrange,
                    std::optional<ReductionPrintArgs> reductionArgs) {
  if (reductionArgs)
    printClauseWithRegionArgs(p, ctx, clauseName, argsSubrange,
                              reductionArgs->vars, reductionArgs->types,
                              reductionArgs->syms, /*mapIndices=*/nullptr,
                              reductionArgs->byref, reductionArgs->modifier);
}

static void printBlockArgRegion(OpAsmPrinter &p, Operation *op, Region &region,
                                const AllRegionPrintArgs &args) {
  auto iface = llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(op);
  MLIRContext *ctx = op->getContext();

  printBlockArgClause(p, ctx, "has_device_addr",
                      iface.getHasDeviceAddrBlockArgs(),
                      args.hasDeviceAddrArgs);
  printBlockArgClause(p, ctx, "host_eval", iface.getHostEvalBlockArgs(),
                      args.hostEvalArgs);
  printBlockArgClause(p, ctx, "in_reduction", iface.getInReductionBlockArgs(),
                      args.inReductionArgs);
  printBlockArgClause(p, ctx, "map_entries", iface.getMapBlockArgs(),
                      args.mapArgs);
  printBlockArgClause(p, ctx, "private", iface.getPrivateBlockArgs(),
                      args.privateArgs);
  printBlockArgClause(p, ctx, "reduction", iface.getReductionBlockArgs(),
                      args.reductionArgs);
  printBlockArgClause(p, ctx, "task_reduction",
                      iface.getTaskReductionBlockArgs(),
                      args.taskReductionArgs);
  printBlockArgClause(p, ctx, "use_device_addr",
                      iface.getUseDeviceAddrBlockArgs(),
                      args.useDeviceAddrArgs);
  printBlockArgClause(p, ctx, "use_device_ptr",
                      iface.getUseDevicePtrBlockArgs(), args.useDevicePtrArgs);

  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

// These parseXyz functions correspond to the custom<Xyz> definitions
// in the .td file(s).
static void printTargetOpRegion(
    OpAsmPrinter &p, Operation *op, Region &region,
    ValueRange hasDeviceAddrVars, TypeRange hasDeviceAddrTypes,
    ValueRange hostEvalVars, TypeRange hostEvalTypes,
    ValueRange inReductionVars, TypeRange inReductionTypes,
    DenseBoolArrayAttr inReductionByref, ArrayAttr inReductionSyms,
    ValueRange mapVars, TypeRange mapTypes, ValueRange privateVars,
    TypeRange privateTypes, ArrayAttr privateSyms, UnitAttr privateNeedsBarrier,
    DenseI64ArrayAttr privateMaps) {
  AllRegionPrintArgs args;
  args.hasDeviceAddrArgs.emplace(hasDeviceAddrVars, hasDeviceAddrTypes);
  args.hostEvalArgs.emplace(hostEvalVars, hostEvalTypes);
  args.inReductionArgs.emplace(inReductionVars, inReductionTypes,
                               inReductionByref, inReductionSyms);
  args.mapArgs.emplace(mapVars, mapTypes);
  args.privateArgs.emplace(privateVars, privateTypes, privateSyms,
                           privateNeedsBarrier, privateMaps);
  printBlockArgRegion(p, op, region, args);
}

static void printInReductionPrivateRegion(
    OpAsmPrinter &p, Operation *op, Region &region, ValueRange inReductionVars,
    TypeRange inReductionTypes, DenseBoolArrayAttr inReductionByref,
    ArrayAttr inReductionSyms, ValueRange privateVars, TypeRange privateTypes,
    ArrayAttr privateSyms, UnitAttr privateNeedsBarrier) {
  AllRegionPrintArgs args;
  args.inReductionArgs.emplace(inReductionVars, inReductionTypes,
                               inReductionByref, inReductionSyms);
  args.privateArgs.emplace(privateVars, privateTypes, privateSyms,
                           privateNeedsBarrier,
                           /*mapIndices=*/nullptr);
  printBlockArgRegion(p, op, region, args);
}

static void printInReductionPrivateReductionRegion(
    OpAsmPrinter &p, Operation *op, Region &region, ValueRange inReductionVars,
    TypeRange inReductionTypes, DenseBoolArrayAttr inReductionByref,
    ArrayAttr inReductionSyms, ValueRange privateVars, TypeRange privateTypes,
    ArrayAttr privateSyms, UnitAttr privateNeedsBarrier,
    ReductionModifierAttr reductionMod, ValueRange reductionVars,
    TypeRange reductionTypes, DenseBoolArrayAttr reductionByref,
    ArrayAttr reductionSyms) {
  AllRegionPrintArgs args;
  args.inReductionArgs.emplace(inReductionVars, inReductionTypes,
                               inReductionByref, inReductionSyms);
  args.privateArgs.emplace(privateVars, privateTypes, privateSyms,
                           privateNeedsBarrier,
                           /*mapIndices=*/nullptr);
  args.reductionArgs.emplace(reductionVars, reductionTypes, reductionByref,
                             reductionSyms, reductionMod);
  printBlockArgRegion(p, op, region, args);
}

static void printPrivateRegion(OpAsmPrinter &p, Operation *op, Region &region,
                               ValueRange privateVars, TypeRange privateTypes,
                               ArrayAttr privateSyms,
                               UnitAttr privateNeedsBarrier) {
  AllRegionPrintArgs args;
  args.privateArgs.emplace(privateVars, privateTypes, privateSyms,
                           privateNeedsBarrier,
                           /*mapIndices=*/nullptr);
  printBlockArgRegion(p, op, region, args);
}

static void printPrivateReductionRegion(
    OpAsmPrinter &p, Operation *op, Region &region, ValueRange privateVars,
    TypeRange privateTypes, ArrayAttr privateSyms, UnitAttr privateNeedsBarrier,
    ReductionModifierAttr reductionMod, ValueRange reductionVars,
    TypeRange reductionTypes, DenseBoolArrayAttr reductionByref,
    ArrayAttr reductionSyms) {
  AllRegionPrintArgs args;
  args.privateArgs.emplace(privateVars, privateTypes, privateSyms,
                           privateNeedsBarrier,
                           /*mapIndices=*/nullptr);
  args.reductionArgs.emplace(reductionVars, reductionTypes, reductionByref,
                             reductionSyms, reductionMod);
  printBlockArgRegion(p, op, region, args);
}

static void printTaskReductionRegion(OpAsmPrinter &p, Operation *op,
                                     Region &region,
                                     ValueRange taskReductionVars,
                                     TypeRange taskReductionTypes,
                                     DenseBoolArrayAttr taskReductionByref,
                                     ArrayAttr taskReductionSyms) {
  AllRegionPrintArgs args;
  args.taskReductionArgs.emplace(taskReductionVars, taskReductionTypes,
                                 taskReductionByref, taskReductionSyms);
  printBlockArgRegion(p, op, region, args);
}

static void printUseDeviceAddrUseDevicePtrRegion(OpAsmPrinter &p, Operation *op,
                                                 Region &region,
                                                 ValueRange useDeviceAddrVars,
                                                 TypeRange useDeviceAddrTypes,
                                                 ValueRange useDevicePtrVars,
                                                 TypeRange useDevicePtrTypes) {
  AllRegionPrintArgs args;
  args.useDeviceAddrArgs.emplace(useDeviceAddrVars, useDeviceAddrTypes);
  args.useDevicePtrArgs.emplace(useDevicePtrVars, useDevicePtrTypes);
  printBlockArgRegion(p, op, region, args);
}

/// Verifies Reduction Clause
static LogicalResult
verifyReductionVarList(Operation *op, std::optional<ArrayAttr> reductionSyms,
                       OperandRange reductionVars,
                       std::optional<ArrayRef<bool>> reductionByref) {
  if (!reductionVars.empty()) {
    if (!reductionSyms || reductionSyms->size() != reductionVars.size())
      return op->emitOpError()
             << "expected as many reduction symbol references "
                "as reduction variables";
    if (reductionByref && reductionByref->size() != reductionVars.size())
      return op->emitError() << "expected as many reduction variable by "
                                "reference attributes as reduction variables";
  } else {
    if (reductionSyms)
      return op->emitOpError() << "unexpected reduction symbol references";
    return success();
  }

  // TODO: The followings should be done in
  // SymbolUserOpInterface::verifySymbolUses.
  DenseSet<Value> accumulators;
  for (auto args : llvm::zip(reductionVars, *reductionSyms)) {
    Value accum = std::get<0>(args);

    if (!accumulators.insert(accum).second)
      return op->emitOpError() << "accumulator variable used more than once";

    Type varType = accum.getType();
    auto symbolRef = llvm::cast<SymbolRefAttr>(std::get<1>(args));
    auto decl =
        SymbolTable::lookupNearestSymbolFrom<DeclareReductionOp>(op, symbolRef);
    if (!decl)
      return op->emitOpError() << "expected symbol reference " << symbolRef
                               << " to point to a reduction declaration";

    if (decl.getAccumulatorType() && decl.getAccumulatorType() != varType)
      return op->emitOpError()
             << "expected accumulator (" << varType
             << ") to be the same type as reduction declaration ("
             << decl.getAccumulatorType() << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Parser, printer and verifier for Copyprivate
//===----------------------------------------------------------------------===//

/// copyprivate-entry-list ::= copyprivate-entry
///                          | copyprivate-entry-list `,` copyprivate-entry
/// copyprivate-entry ::= ssa-id `->` symbol-ref `:` type
static ParseResult parseCopyprivate(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &copyprivateVars,
    SmallVectorImpl<Type> &copyprivateTypes, ArrayAttr &copyprivateSyms) {
  SmallVector<SymbolRefAttr> symsVec;
  if (failed(parser.parseCommaSeparatedList([&]() {
        if (parser.parseOperand(copyprivateVars.emplace_back()) ||
            parser.parseArrow() ||
            parser.parseAttribute(symsVec.emplace_back()) ||
            parser.parseColonType(copyprivateTypes.emplace_back()))
          return failure();
        return success();
      })))
    return failure();
  SmallVector<Attribute> syms(symsVec.begin(), symsVec.end());
  copyprivateSyms = ArrayAttr::get(parser.getContext(), syms);
  return success();
}

/// Print Copyprivate clause
static void printCopyprivate(OpAsmPrinter &p, Operation *op,
                             OperandRange copyprivateVars,
                             TypeRange copyprivateTypes,
                             std::optional<ArrayAttr> copyprivateSyms) {
  if (!copyprivateSyms.has_value())
    return;
  llvm::interleaveComma(
      llvm::zip(copyprivateVars, *copyprivateSyms, copyprivateTypes), p,
      [&](const auto &args) {
        p << std::get<0>(args) << " -> " << std::get<1>(args) << " : "
          << std::get<2>(args);
      });
}

/// Verifies CopyPrivate Clause
static LogicalResult
verifyCopyprivateVarList(Operation *op, OperandRange copyprivateVars,
                         std::optional<ArrayAttr> copyprivateSyms) {
  size_t copyprivateSymsSize =
      copyprivateSyms.has_value() ? copyprivateSyms->size() : 0;
  if (copyprivateSymsSize != copyprivateVars.size())
    return op->emitOpError() << "inconsistent number of copyprivate vars (= "
                             << copyprivateVars.size()
                             << ") and functions (= " << copyprivateSymsSize
                             << "), both must be equal";
  if (!copyprivateSyms.has_value())
    return success();

  for (auto copyprivateVarAndSym :
       llvm::zip(copyprivateVars, *copyprivateSyms)) {
    auto symbolRef =
        llvm::cast<SymbolRefAttr>(std::get<1>(copyprivateVarAndSym));
    std::optional<std::variant<mlir::func::FuncOp, mlir::LLVM::LLVMFuncOp>>
        funcOp;
    if (mlir::func::FuncOp mlirFuncOp =
            SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(op,
                                                                     symbolRef))
      funcOp = mlirFuncOp;
    else if (mlir::LLVM::LLVMFuncOp llvmFuncOp =
                 SymbolTable::lookupNearestSymbolFrom<mlir::LLVM::LLVMFuncOp>(
                     op, symbolRef))
      funcOp = llvmFuncOp;

    auto getNumArguments = [&] {
      return std::visit([](auto &f) { return f.getNumArguments(); }, *funcOp);
    };

    auto getArgumentType = [&](unsigned i) {
      return std::visit([i](auto &f) { return f.getArgumentTypes()[i]; },
                        *funcOp);
    };

    if (!funcOp)
      return op->emitOpError() << "expected symbol reference " << symbolRef
                               << " to point to a copy function";

    if (getNumArguments() != 2)
      return op->emitOpError()
             << "expected copy function " << symbolRef << " to have 2 operands";

    Type argTy = getArgumentType(0);
    if (argTy != getArgumentType(1))
      return op->emitOpError() << "expected copy function " << symbolRef
                               << " arguments to have the same type";

    Type varType = std::get<0>(copyprivateVarAndSym).getType();
    if (argTy != varType)
      return op->emitOpError()
             << "expected copy function arguments' type (" << argTy
             << ") to be the same as copyprivate variable's type (" << varType
             << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Parser, printer and verifier for DependVarList
//===----------------------------------------------------------------------===//

/// depend-entry-list ::= depend-entry
///                     | depend-entry-list `,` depend-entry
/// depend-entry ::= depend-kind `->` ssa-id `:` type
static ParseResult
parseDependVarList(OpAsmParser &parser,
                   SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dependVars,
                   SmallVectorImpl<Type> &dependTypes, ArrayAttr &dependKinds) {
  SmallVector<ClauseTaskDependAttr> kindsVec;
  if (failed(parser.parseCommaSeparatedList([&]() {
        StringRef keyword;
        if (parser.parseKeyword(&keyword) || parser.parseArrow() ||
            parser.parseOperand(dependVars.emplace_back()) ||
            parser.parseColonType(dependTypes.emplace_back()))
          return failure();
        if (std::optional<ClauseTaskDepend> keywordDepend =
                (symbolizeClauseTaskDepend(keyword)))
          kindsVec.emplace_back(
              ClauseTaskDependAttr::get(parser.getContext(), *keywordDepend));
        else
          return failure();
        return success();
      })))
    return failure();
  SmallVector<Attribute> kinds(kindsVec.begin(), kindsVec.end());
  dependKinds = ArrayAttr::get(parser.getContext(), kinds);
  return success();
}

/// Print Depend clause
static void printDependVarList(OpAsmPrinter &p, Operation *op,
                               OperandRange dependVars, TypeRange dependTypes,
                               std::optional<ArrayAttr> dependKinds) {

  for (unsigned i = 0, e = dependKinds->size(); i < e; ++i) {
    if (i != 0)
      p << ", ";
    p << stringifyClauseTaskDepend(
             llvm::cast<mlir::omp::ClauseTaskDependAttr>((*dependKinds)[i])
                 .getValue())
      << " -> " << dependVars[i] << " : " << dependTypes[i];
  }
}

/// Verifies Depend clause
static LogicalResult verifyDependVarList(Operation *op,
                                         std::optional<ArrayAttr> dependKinds,
                                         OperandRange dependVars) {
  if (!dependVars.empty()) {
    if (!dependKinds || dependKinds->size() != dependVars.size())
      return op->emitOpError() << "expected as many depend values"
                                  " as depend variables";
  } else {
    if (dependKinds && !dependKinds->empty())
      return op->emitOpError() << "unexpected depend values";
    return success();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Parser, printer and verifier for Synchronization Hint (2.17.12)
//===----------------------------------------------------------------------===//

/// Parses a Synchronization Hint clause. The value of hint is an integer
/// which is a combination of different hints from `omp_sync_hint_t`.
///
/// hint-clause = `hint` `(` hint-value `)`
static ParseResult parseSynchronizationHint(OpAsmParser &parser,
                                            IntegerAttr &hintAttr) {
  StringRef hintKeyword;
  int64_t hint = 0;
  if (succeeded(parser.parseOptionalKeyword("none"))) {
    hintAttr = IntegerAttr::get(parser.getBuilder().getI64Type(), 0);
    return success();
  }
  auto parseKeyword = [&]() -> ParseResult {
    if (failed(parser.parseKeyword(&hintKeyword)))
      return failure();
    if (hintKeyword == "uncontended")
      hint |= 1;
    else if (hintKeyword == "contended")
      hint |= 2;
    else if (hintKeyword == "nonspeculative")
      hint |= 4;
    else if (hintKeyword == "speculative")
      hint |= 8;
    else
      return parser.emitError(parser.getCurrentLocation())
             << hintKeyword << " is not a valid hint";
    return success();
  };
  if (parser.parseCommaSeparatedList(parseKeyword))
    return failure();
  hintAttr = IntegerAttr::get(parser.getBuilder().getI64Type(), hint);
  return success();
}

/// Prints a Synchronization Hint clause
static void printSynchronizationHint(OpAsmPrinter &p, Operation *op,
                                     IntegerAttr hintAttr) {
  int64_t hint = hintAttr.getInt();

  if (hint == 0) {
    p << "none";
    return;
  }

  // Helper function to get n-th bit from the right end of `value`
  auto bitn = [](int value, int n) -> bool { return value & (1 << n); };

  bool uncontended = bitn(hint, 0);
  bool contended = bitn(hint, 1);
  bool nonspeculative = bitn(hint, 2);
  bool speculative = bitn(hint, 3);

  SmallVector<StringRef> hints;
  if (uncontended)
    hints.push_back("uncontended");
  if (contended)
    hints.push_back("contended");
  if (nonspeculative)
    hints.push_back("nonspeculative");
  if (speculative)
    hints.push_back("speculative");

  llvm::interleaveComma(hints, p);
}

/// Verifies a synchronization hint clause
static LogicalResult verifySynchronizationHint(Operation *op, uint64_t hint) {

  // Helper function to get n-th bit from the right end of `value`
  auto bitn = [](int value, int n) -> bool { return value & (1 << n); };

  bool uncontended = bitn(hint, 0);
  bool contended = bitn(hint, 1);
  bool nonspeculative = bitn(hint, 2);
  bool speculative = bitn(hint, 3);

  if (uncontended && contended)
    return op->emitOpError() << "the hints omp_sync_hint_uncontended and "
                                "omp_sync_hint_contended cannot be combined";
  if (nonspeculative && speculative)
    return op->emitOpError() << "the hints omp_sync_hint_nonspeculative and "
                                "omp_sync_hint_speculative cannot be combined.";
  return success();
}

//===----------------------------------------------------------------------===//
// Parser, printer and verifier for Target
//===----------------------------------------------------------------------===//

// Helper function to get bitwise AND of `value` and 'flag'
static uint64_t mapTypeToBitFlag(uint64_t value,
                                 llvm::omp::OpenMPOffloadMappingFlags flag) {
  return value & llvm::to_underlying(flag);
}

/// Parses a map_entries map type from a string format back into its numeric
/// value.
///
/// map-clause = `map_clauses (  ( `(` `always, `? `implicit, `? `ompx_hold, `?
/// `close, `? `present, `? ( `to` | `from` | `delete` `)` )+ `)` )
static ParseResult parseMapClause(OpAsmParser &parser, IntegerAttr &mapType) {
  llvm::omp::OpenMPOffloadMappingFlags mapTypeBits =
      llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_NONE;

  // This simply verifies the correct keyword is read in, the
  // keyword itself is stored inside of the operation
  auto parseTypeAndMod = [&]() -> ParseResult {
    StringRef mapTypeMod;
    if (parser.parseKeyword(&mapTypeMod))
      return failure();

    if (mapTypeMod == "always")
      mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_ALWAYS;

    if (mapTypeMod == "implicit")
      mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT;

    if (mapTypeMod == "ompx_hold")
      mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_OMPX_HOLD;

    if (mapTypeMod == "close")
      mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_CLOSE;

    if (mapTypeMod == "present")
      mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_PRESENT;

    if (mapTypeMod == "to")
      mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO;

    if (mapTypeMod == "from")
      mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;

    if (mapTypeMod == "tofrom")
      mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO |
                     llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;

    if (mapTypeMod == "delete")
      mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_DELETE;

    if (mapTypeMod == "return_param")
      mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_RETURN_PARAM;

    return success();
  };

  if (parser.parseCommaSeparatedList(parseTypeAndMod))
    return failure();

  mapType = parser.getBuilder().getIntegerAttr(
      parser.getBuilder().getIntegerType(64, /*isSigned=*/false),
      llvm::to_underlying(mapTypeBits));

  return success();
}

/// Prints a map_entries map type from its numeric value out into its string
/// format.
static void printMapClause(OpAsmPrinter &p, Operation *op,
                           IntegerAttr mapType) {
  uint64_t mapTypeBits = mapType.getUInt();

  bool emitAllocRelease = true;
  llvm::SmallVector<std::string, 4> mapTypeStrs;

  // handling of always, close, present placed at the beginning of the string
  // to aid readability
  if (mapTypeToBitFlag(mapTypeBits,
                       llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_ALWAYS))
    mapTypeStrs.push_back("always");
  if (mapTypeToBitFlag(mapTypeBits,
                       llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT))
    mapTypeStrs.push_back("implicit");
  if (mapTypeToBitFlag(mapTypeBits,
                       llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_OMPX_HOLD))
    mapTypeStrs.push_back("ompx_hold");
  if (mapTypeToBitFlag(mapTypeBits,
                       llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_CLOSE))
    mapTypeStrs.push_back("close");
  if (mapTypeToBitFlag(mapTypeBits,
                       llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_PRESENT))
    mapTypeStrs.push_back("present");

  // special handling of to/from/tofrom/delete and release/alloc, release +
  // alloc are the abscense of one of the other flags, whereas tofrom requires
  // both the to and from flag to be set.
  bool to = mapTypeToBitFlag(mapTypeBits,
                             llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO);
  bool from = mapTypeToBitFlag(
      mapTypeBits, llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM);
  if (to && from) {
    emitAllocRelease = false;
    mapTypeStrs.push_back("tofrom");
  } else if (from) {
    emitAllocRelease = false;
    mapTypeStrs.push_back("from");
  } else if (to) {
    emitAllocRelease = false;
    mapTypeStrs.push_back("to");
  }
  if (mapTypeToBitFlag(mapTypeBits,
                       llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_DELETE)) {
    emitAllocRelease = false;
    mapTypeStrs.push_back("delete");
  }
  if (mapTypeToBitFlag(
          mapTypeBits,
          llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_RETURN_PARAM)) {
    emitAllocRelease = false;
    mapTypeStrs.push_back("return_param");
  }
  if (emitAllocRelease)
    mapTypeStrs.push_back("exit_release_or_enter_alloc");

  for (unsigned int i = 0; i < mapTypeStrs.size(); ++i) {
    p << mapTypeStrs[i];
    if (i + 1 < mapTypeStrs.size()) {
      p << ", ";
    }
  }
}

static ParseResult parseMembersIndex(OpAsmParser &parser,
                                     ArrayAttr &membersIdx) {
  SmallVector<Attribute> values, memberIdxs;

  auto parseIndices = [&]() -> ParseResult {
    int64_t value;
    if (parser.parseInteger(value))
      return failure();
    values.push_back(IntegerAttr::get(parser.getBuilder().getIntegerType(64),
                                      APInt(64, value, /*isSigned=*/false)));
    return success();
  };

  do {
    if (failed(parser.parseLSquare()))
      return failure();

    if (parser.parseCommaSeparatedList(parseIndices))
      return failure();

    if (failed(parser.parseRSquare()))
      return failure();

    memberIdxs.push_back(ArrayAttr::get(parser.getContext(), values));
    values.clear();
  } while (succeeded(parser.parseOptionalComma()));

  if (!memberIdxs.empty())
    membersIdx = ArrayAttr::get(parser.getContext(), memberIdxs);

  return success();
}

static void printMembersIndex(OpAsmPrinter &p, MapInfoOp op,
                              ArrayAttr membersIdx) {
  if (!membersIdx)
    return;

  llvm::interleaveComma(membersIdx, p, [&p](Attribute v) {
    p << "[";
    auto memberIdx = cast<ArrayAttr>(v);
    llvm::interleaveComma(memberIdx.getValue(), p, [&p](Attribute v2) {
      p << cast<IntegerAttr>(v2).getInt();
    });
    p << "]";
  });
}

static void printCaptureType(OpAsmPrinter &p, Operation *op,
                             VariableCaptureKindAttr mapCaptureType) {
  std::string typeCapStr;
  llvm::raw_string_ostream typeCap(typeCapStr);
  if (mapCaptureType.getValue() == mlir::omp::VariableCaptureKind::ByRef)
    typeCap << "ByRef";
  if (mapCaptureType.getValue() == mlir::omp::VariableCaptureKind::ByCopy)
    typeCap << "ByCopy";
  if (mapCaptureType.getValue() == mlir::omp::VariableCaptureKind::VLAType)
    typeCap << "VLAType";
  if (mapCaptureType.getValue() == mlir::omp::VariableCaptureKind::This)
    typeCap << "This";
  p << typeCapStr;
}

static ParseResult parseCaptureType(OpAsmParser &parser,
                                    VariableCaptureKindAttr &mapCaptureType) {
  StringRef mapCaptureKey;
  if (parser.parseKeyword(&mapCaptureKey))
    return failure();

  if (mapCaptureKey == "This")
    mapCaptureType = mlir::omp::VariableCaptureKindAttr::get(
        parser.getContext(), mlir::omp::VariableCaptureKind::This);
  if (mapCaptureKey == "ByRef")
    mapCaptureType = mlir::omp::VariableCaptureKindAttr::get(
        parser.getContext(), mlir::omp::VariableCaptureKind::ByRef);
  if (mapCaptureKey == "ByCopy")
    mapCaptureType = mlir::omp::VariableCaptureKindAttr::get(
        parser.getContext(), mlir::omp::VariableCaptureKind::ByCopy);
  if (mapCaptureKey == "VLAType")
    mapCaptureType = mlir::omp::VariableCaptureKindAttr::get(
        parser.getContext(), mlir::omp::VariableCaptureKind::VLAType);

  return success();
}

static LogicalResult verifyMapClause(Operation *op, OperandRange mapVars) {
  llvm::DenseSet<mlir::TypedValue<mlir::omp::PointerLikeType>> updateToVars;
  llvm::DenseSet<mlir::TypedValue<mlir::omp::PointerLikeType>> updateFromVars;

  for (auto mapOp : mapVars) {
    if (!mapOp.getDefiningOp())
      return emitError(op->getLoc(), "missing map operation");

    if (auto mapInfoOp = mapOp.getDefiningOp<mlir::omp::MapInfoOp>()) {
      uint64_t mapTypeBits = mapInfoOp.getMapType();

      bool to = mapTypeToBitFlag(
          mapTypeBits, llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO);
      bool from = mapTypeToBitFlag(
          mapTypeBits, llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM);
      bool del = mapTypeToBitFlag(
          mapTypeBits, llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_DELETE);

      bool always = mapTypeToBitFlag(
          mapTypeBits, llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_ALWAYS);
      bool close = mapTypeToBitFlag(
          mapTypeBits, llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_CLOSE);
      bool implicit = mapTypeToBitFlag(
          mapTypeBits, llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT);

      if ((isa<TargetDataOp>(op) || isa<TargetOp>(op)) && del)
        return emitError(op->getLoc(),
                         "to, from, tofrom and alloc map types are permitted");

      if (isa<TargetEnterDataOp>(op) && (from || del))
        return emitError(op->getLoc(), "to and alloc map types are permitted");

      if (isa<TargetExitDataOp>(op) && to)
        return emitError(op->getLoc(),
                         "from, release and delete map types are permitted");

      if (isa<TargetUpdateOp>(op)) {
        if (del) {
          return emitError(op->getLoc(),
                           "at least one of to or from map types must be "
                           "specified, other map types are not permitted");
        }

        if (!to && !from) {
          return emitError(op->getLoc(),
                           "at least one of to or from map types must be "
                           "specified, other map types are not permitted");
        }

        auto updateVar = mapInfoOp.getVarPtr();

        if ((to && from) || (to && updateFromVars.contains(updateVar)) ||
            (from && updateToVars.contains(updateVar))) {
          return emitError(
              op->getLoc(),
              "either to or from map types can be specified, not both");
        }

        if (always || close || implicit) {
          return emitError(
              op->getLoc(),
              "present, mapper and iterator map type modifiers are permitted");
        }

        to ? updateToVars.insert(updateVar) : updateFromVars.insert(updateVar);
      }
    } else if (!isa<DeclareMapperInfoOp>(op)) {
      return emitError(op->getLoc(),
                       "map argument is not a map entry operation");
    }
  }

  return success();
}

static LogicalResult verifyPrivateVarsMapping(TargetOp targetOp) {
  std::optional<DenseI64ArrayAttr> privateMapIndices =
      targetOp.getPrivateMapsAttr();

  // None of the private operands are mapped.
  if (!privateMapIndices.has_value() || !privateMapIndices.value())
    return success();

  OperandRange privateVars = targetOp.getPrivateVars();

  if (privateMapIndices.value().size() !=
      static_cast<int64_t>(privateVars.size()))
    return emitError(targetOp.getLoc(), "sizes of `private` operand range and "
                                        "`private_maps` attribute mismatch");

  return success();
}

//===----------------------------------------------------------------------===//
// MapInfoOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyMapInfoDefinedArgs(Operation *op,
                                              StringRef clauseName,
                                              OperandRange vars) {
  for (Value var : vars)
    if (!llvm::isa_and_present<MapInfoOp>(var.getDefiningOp()))
      return op->emitOpError()
             << "'" << clauseName
             << "' arguments must be defined by 'omp.map.info' ops";
  return success();
}

LogicalResult MapInfoOp::verify() {
  if (getMapperId() &&
      !SymbolTable::lookupNearestSymbolFrom<omp::DeclareMapperOp>(
          *this, getMapperIdAttr())) {
    return emitError("invalid mapper id");
  }

  if (failed(verifyMapInfoDefinedArgs(*this, "members", getMembers())))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// TargetDataOp
//===----------------------------------------------------------------------===//

void TargetDataOp::build(OpBuilder &builder, OperationState &state,
                         const TargetDataOperands &clauses) {
  TargetDataOp::build(builder, state, clauses.device, clauses.ifExpr,
                      clauses.mapVars, clauses.useDeviceAddrVars,
                      clauses.useDevicePtrVars);
}

LogicalResult TargetDataOp::verify() {
  if (getMapVars().empty() && getUseDevicePtrVars().empty() &&
      getUseDeviceAddrVars().empty()) {
    return ::emitError(this->getLoc(),
                       "At least one of map, use_device_ptr_vars, or "
                       "use_device_addr_vars operand must be present");
  }

  if (failed(verifyMapInfoDefinedArgs(*this, "use_device_ptr",
                                      getUseDevicePtrVars())))
    return failure();

  if (failed(verifyMapInfoDefinedArgs(*this, "use_device_addr",
                                      getUseDeviceAddrVars())))
    return failure();

  return verifyMapClause(*this, getMapVars());
}

//===----------------------------------------------------------------------===//
// TargetEnterDataOp
//===----------------------------------------------------------------------===//

void TargetEnterDataOp::build(
    OpBuilder &builder, OperationState &state,
    const TargetEnterExitUpdateDataOperands &clauses) {
  MLIRContext *ctx = builder.getContext();
  TargetEnterDataOp::build(builder, state,
                           makeArrayAttr(ctx, clauses.dependKinds),
                           clauses.dependVars, clauses.device, clauses.ifExpr,
                           clauses.mapVars, clauses.nowait);
}

LogicalResult TargetEnterDataOp::verify() {
  LogicalResult verifyDependVars =
      verifyDependVarList(*this, getDependKinds(), getDependVars());
  return failed(verifyDependVars) ? verifyDependVars
                                  : verifyMapClause(*this, getMapVars());
}

//===----------------------------------------------------------------------===//
// TargetExitDataOp
//===----------------------------------------------------------------------===//

void TargetExitDataOp::build(OpBuilder &builder, OperationState &state,
                             const TargetEnterExitUpdateDataOperands &clauses) {
  MLIRContext *ctx = builder.getContext();
  TargetExitDataOp::build(builder, state,
                          makeArrayAttr(ctx, clauses.dependKinds),
                          clauses.dependVars, clauses.device, clauses.ifExpr,
                          clauses.mapVars, clauses.nowait);
}

LogicalResult TargetExitDataOp::verify() {
  LogicalResult verifyDependVars =
      verifyDependVarList(*this, getDependKinds(), getDependVars());
  return failed(verifyDependVars) ? verifyDependVars
                                  : verifyMapClause(*this, getMapVars());
}

//===----------------------------------------------------------------------===//
// TargetUpdateOp
//===----------------------------------------------------------------------===//

void TargetUpdateOp::build(OpBuilder &builder, OperationState &state,
                           const TargetEnterExitUpdateDataOperands &clauses) {
  MLIRContext *ctx = builder.getContext();
  TargetUpdateOp::build(builder, state, makeArrayAttr(ctx, clauses.dependKinds),
                        clauses.dependVars, clauses.device, clauses.ifExpr,
                        clauses.mapVars, clauses.nowait);
}

LogicalResult TargetUpdateOp::verify() {
  LogicalResult verifyDependVars =
      verifyDependVarList(*this, getDependKinds(), getDependVars());
  return failed(verifyDependVars) ? verifyDependVars
                                  : verifyMapClause(*this, getMapVars());
}

//===----------------------------------------------------------------------===//
// TargetOp
//===----------------------------------------------------------------------===//

void TargetOp::build(OpBuilder &builder, OperationState &state,
                     const TargetOperands &clauses) {
  MLIRContext *ctx = builder.getContext();
  // TODO Store clauses in op: allocateVars, allocatorVars, inReductionVars,
  // inReductionByref, inReductionSyms.
  TargetOp::build(builder, state, /*allocate_vars=*/{}, /*allocator_vars=*/{},
                  clauses.bare, makeArrayAttr(ctx, clauses.dependKinds),
                  clauses.dependVars, clauses.device, clauses.hasDeviceAddrVars,
                  clauses.hostEvalVars, clauses.ifExpr,
                  /*in_reduction_vars=*/{}, /*in_reduction_byref=*/nullptr,
                  /*in_reduction_syms=*/nullptr, clauses.isDevicePtrVars,
                  clauses.mapVars, clauses.nowait, clauses.privateVars,
                  makeArrayAttr(ctx, clauses.privateSyms),
                  clauses.privateNeedsBarrier, clauses.threadLimit,
                  /*private_maps=*/nullptr);
}

LogicalResult TargetOp::verify() {
  if (failed(verifyDependVarList(*this, getDependKinds(), getDependVars())))
    return failure();

  if (failed(verifyMapInfoDefinedArgs(*this, "has_device_addr",
                                      getHasDeviceAddrVars())))
    return failure();

  if (failed(verifyMapClause(*this, getMapVars())))
    return failure();

  return verifyPrivateVarsMapping(*this);
}

LogicalResult TargetOp::verifyRegions() {
  auto teamsOps = getOps<TeamsOp>();
  if (std::distance(teamsOps.begin(), teamsOps.end()) > 1)
    return emitError("target containing multiple 'omp.teams' nested ops");

  // Check that host_eval values are only used in legal ways.
  Operation *capturedOp = getInnermostCapturedOmpOp();
  TargetRegionFlags execFlags = getKernelExecFlags(capturedOp);
  for (Value hostEvalArg :
       cast<BlockArgOpenMPOpInterface>(getOperation()).getHostEvalBlockArgs()) {
    for (Operation *user : hostEvalArg.getUsers()) {
      if (auto teamsOp = dyn_cast<TeamsOp>(user)) {
        if (llvm::is_contained({teamsOp.getNumTeamsLower(),
                                teamsOp.getNumTeamsUpper(),
                                teamsOp.getThreadLimit()},
                               hostEvalArg))
          continue;

        return emitOpError() << "host_eval argument only legal as 'num_teams' "
                                "and 'thread_limit' in 'omp.teams'";
      }
      if (auto parallelOp = dyn_cast<ParallelOp>(user)) {
        if (bitEnumContainsAny(execFlags, TargetRegionFlags::spmd) &&
            parallelOp->isAncestor(capturedOp) &&
            hostEvalArg == parallelOp.getNumThreads())
          continue;

        return emitOpError()
               << "host_eval argument only legal as 'num_threads' in "
                  "'omp.parallel' when representing target SPMD";
      }
      if (auto loopNestOp = dyn_cast<LoopNestOp>(user)) {
        if (bitEnumContainsAny(execFlags, TargetRegionFlags::trip_count) &&
            loopNestOp.getOperation() == capturedOp &&
            (llvm::is_contained(loopNestOp.getLoopLowerBounds(), hostEvalArg) ||
             llvm::is_contained(loopNestOp.getLoopUpperBounds(), hostEvalArg) ||
             llvm::is_contained(loopNestOp.getLoopSteps(), hostEvalArg)))
          continue;

        return emitOpError() << "host_eval argument only legal as loop bounds "
                                "and steps in 'omp.loop_nest' when trip count "
                                "must be evaluated in the host";
      }

      return emitOpError() << "host_eval argument illegal use in '"
                           << user->getName() << "' operation";
    }
  }
  return success();
}

static Operation *
findCapturedOmpOp(Operation *rootOp, bool checkSingleMandatoryExec,
                  llvm::function_ref<bool(Operation *)> siblingAllowedFn) {
  assert(rootOp && "expected valid operation");

  Dialect *ompDialect = rootOp->getDialect();
  Operation *capturedOp = nullptr;
  DominanceInfo domInfo;

  // Process in pre-order to check operations from outermost to innermost,
  // ensuring we only enter the region of an operation if it meets the criteria
  // for being captured. We stop the exploration of nested operations as soon as
  // we process a region holding no operations to be captured.
  rootOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op == rootOp)
      return WalkResult::advance();

    // Ignore operations of other dialects or omp operations with no regions,
    // because these will only be checked if they are siblings of an omp
    // operation that can potentially be captured.
    bool isOmpDialect = op->getDialect() == ompDialect;
    bool hasRegions = op->getNumRegions() > 0;
    if (!isOmpDialect || !hasRegions)
      return WalkResult::skip();

    // This operation cannot be captured if it can be executed more than once
    // (i.e. its block's successors can reach it) or if it's not guaranteed to
    // be executed before all exits of the region (i.e. it doesn't dominate all
    // blocks with no successors reachable from the entry block).
    if (checkSingleMandatoryExec) {
      Region *parentRegion = op->getParentRegion();
      Block *parentBlock = op->getBlock();

      for (Block *successor : parentBlock->getSuccessors())
        if (successor->isReachable(parentBlock))
          return WalkResult::interrupt();

      for (Block &block : *parentRegion)
        if (domInfo.isReachableFromEntry(&block) && block.hasNoSuccessors() &&
            !domInfo.dominates(parentBlock, &block))
          return WalkResult::interrupt();
    }

    // Don't capture this op if it has a not-allowed sibling, and stop recursing
    // into nested operations.
    for (Operation &sibling : op->getParentRegion()->getOps())
      if (&sibling != op && !siblingAllowedFn(&sibling))
        return WalkResult::interrupt();

    // Don't continue capturing nested operations if we reach an omp.loop_nest.
    // Otherwise, process the contents of this operation.
    capturedOp = op;
    return llvm::isa<LoopNestOp>(op) ? WalkResult::interrupt()
                                     : WalkResult::advance();
  });

  return capturedOp;
}

Operation *TargetOp::getInnermostCapturedOmpOp() {
  auto *ompDialect = getContext()->getLoadedDialect<omp::OpenMPDialect>();

  // Only allow OpenMP terminators and non-OpenMP ops that have known memory
  // effects, but don't include a memory write effect.
  return findCapturedOmpOp(
      *this, /*checkSingleMandatoryExec=*/true, [&](Operation *sibling) {
        if (!sibling)
          return false;

        if (ompDialect == sibling->getDialect())
          return sibling->hasTrait<OpTrait::IsTerminator>();

        if (auto memOp = dyn_cast<MemoryEffectOpInterface>(sibling)) {
          SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4>
              effects;
          memOp.getEffects(effects);
          return !llvm::any_of(
              effects, [&](MemoryEffects::EffectInstance &effect) {
                return isa<MemoryEffects::Write>(effect.getEffect()) &&
                       isa<SideEffects::AutomaticAllocationScopeResource>(
                           effect.getResource());
              });
        }
        return true;
      });
}

/// Check if we can promote SPMD kernel to No-Loop kernel.
static bool canPromoteToNoLoop(Operation *capturedOp, TeamsOp teamsOp,
                               WsloopOp *wsLoopOp) {
  // num_teams clause can break no-loop teams/threads assumption.
  if (teamsOp.getNumTeamsUpper())
    return false;

  // Reduction kernels are slower in no-loop mode.
  if (teamsOp.getNumReductionVars())
    return false;
  if (wsLoopOp->getNumReductionVars())
    return false;

  // Check if the user allows the promotion of kernels to no-loop mode.
  OffloadModuleInterface offloadMod =
      capturedOp->getParentOfType<omp::OffloadModuleInterface>();
  if (!offloadMod)
    return false;
  auto ompFlags = offloadMod.getFlags();
  if (!ompFlags)
    return false;
  return ompFlags.getAssumeTeamsOversubscription() &&
         ompFlags.getAssumeThreadsOversubscription();
}

TargetRegionFlags TargetOp::getKernelExecFlags(Operation *capturedOp) {
  // A non-null captured op is only valid if it resides inside of a TargetOp
  // and is the result of calling getInnermostCapturedOmpOp() on it.
  TargetOp targetOp =
      capturedOp ? capturedOp->getParentOfType<TargetOp>() : nullptr;
  assert((!capturedOp ||
          (targetOp && targetOp.getInnermostCapturedOmpOp() == capturedOp)) &&
         "unexpected captured op");

  // If it's not capturing a loop, it's a default target region.
  if (!isa_and_present<LoopNestOp>(capturedOp))
    return TargetRegionFlags::generic;

  // Get the innermost non-simd loop wrapper.
  SmallVector<LoopWrapperInterface> loopWrappers;
  cast<LoopNestOp>(capturedOp).gatherWrappers(loopWrappers);
  assert(!loopWrappers.empty());

  LoopWrapperInterface *innermostWrapper = loopWrappers.begin();
  if (isa<SimdOp>(innermostWrapper))
    innermostWrapper = std::next(innermostWrapper);

  auto numWrappers = std::distance(innermostWrapper, loopWrappers.end());
  if (numWrappers != 1 && numWrappers != 2)
    return TargetRegionFlags::generic;

  // Detect target-teams-distribute-parallel-wsloop[-simd].
  if (numWrappers == 2) {
    WsloopOp *wsloopOp = dyn_cast<WsloopOp>(innermostWrapper);
    if (!wsloopOp)
      return TargetRegionFlags::generic;

    innermostWrapper = std::next(innermostWrapper);
    if (!isa<DistributeOp>(innermostWrapper))
      return TargetRegionFlags::generic;

    Operation *parallelOp = (*innermostWrapper)->getParentOp();
    if (!isa_and_present<ParallelOp>(parallelOp))
      return TargetRegionFlags::generic;

    TeamsOp teamsOp = dyn_cast<TeamsOp>(parallelOp->getParentOp());
    if (!teamsOp)
      return TargetRegionFlags::generic;

    if (teamsOp->getParentOp() == targetOp.getOperation()) {
      TargetRegionFlags result =
          TargetRegionFlags::spmd | TargetRegionFlags::trip_count;
      if (canPromoteToNoLoop(capturedOp, teamsOp, wsloopOp))
        result = result | TargetRegionFlags::no_loop;
      return result;
    }
  }
  // Detect target-teams-distribute[-simd] and target-teams-loop.
  else if (isa<DistributeOp, LoopOp>(innermostWrapper)) {
    Operation *teamsOp = (*innermostWrapper)->getParentOp();
    if (!isa_and_present<TeamsOp>(teamsOp))
      return TargetRegionFlags::generic;

    if (teamsOp->getParentOp() != targetOp.getOperation())
      return TargetRegionFlags::generic;

    if (isa<LoopOp>(innermostWrapper))
      return TargetRegionFlags::spmd | TargetRegionFlags::trip_count;

    // Find single immediately nested captured omp.parallel and add spmd flag
    // (generic-spmd case).
    //
    // TODO: This shouldn't have to be done here, as it is too easy to break.
    // The openmp-opt pass should be updated to be able to promote kernels like
    // this from "Generic" to "Generic-SPMD". However, the use of the
    // `kmpc_distribute_static_loop` family of functions produced by the
    // OMPIRBuilder for these kernels prevents that from working.
    Dialect *ompDialect = targetOp->getDialect();
    Operation *nestedCapture = findCapturedOmpOp(
        capturedOp, /*checkSingleMandatoryExec=*/false,
        [&](Operation *sibling) {
          return sibling && (ompDialect != sibling->getDialect() ||
                             sibling->hasTrait<OpTrait::IsTerminator>());
        });

    TargetRegionFlags result =
        TargetRegionFlags::generic | TargetRegionFlags::trip_count;

    if (!nestedCapture)
      return result;

    while (nestedCapture->getParentOp() != capturedOp)
      nestedCapture = nestedCapture->getParentOp();

    return isa<ParallelOp>(nestedCapture) ? result | TargetRegionFlags::spmd
                                          : result;
  }
  // Detect target-parallel-wsloop[-simd].
  else if (isa<WsloopOp>(innermostWrapper)) {
    Operation *parallelOp = (*innermostWrapper)->getParentOp();
    if (!isa_and_present<ParallelOp>(parallelOp))
      return TargetRegionFlags::generic;

    if (parallelOp->getParentOp() == targetOp.getOperation())
      return TargetRegionFlags::spmd;
  }

  return TargetRegionFlags::generic;
}

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

void ParallelOp::build(OpBuilder &builder, OperationState &state,
                       ArrayRef<NamedAttribute> attributes) {
  ParallelOp::build(builder, state, /*allocate_vars=*/ValueRange(),
                    /*allocator_vars=*/ValueRange(), /*if_expr=*/nullptr,
                    /*num_threads=*/nullptr, /*private_vars=*/ValueRange(),
                    /*private_syms=*/nullptr, /*private_needs_barrier=*/nullptr,
                    /*proc_bind_kind=*/nullptr,
                    /*reduction_mod =*/nullptr, /*reduction_vars=*/ValueRange(),
                    /*reduction_byref=*/nullptr, /*reduction_syms=*/nullptr);
  state.addAttributes(attributes);
}

void ParallelOp::build(OpBuilder &builder, OperationState &state,
                       const ParallelOperands &clauses) {
  MLIRContext *ctx = builder.getContext();
  ParallelOp::build(builder, state, clauses.allocateVars, clauses.allocatorVars,
                    clauses.ifExpr, clauses.numThreads, clauses.privateVars,
                    makeArrayAttr(ctx, clauses.privateSyms),
                    clauses.privateNeedsBarrier, clauses.procBindKind,
                    clauses.reductionMod, clauses.reductionVars,
                    makeDenseBoolArrayAttr(ctx, clauses.reductionByref),
                    makeArrayAttr(ctx, clauses.reductionSyms));
}

template <typename OpType>
static LogicalResult verifyPrivateVarList(OpType &op) {
  auto privateVars = op.getPrivateVars();
  auto privateSyms = op.getPrivateSymsAttr();

  if (privateVars.empty() && (privateSyms == nullptr || privateSyms.empty()))
    return success();

  auto numPrivateVars = privateVars.size();
  auto numPrivateSyms = (privateSyms == nullptr) ? 0 : privateSyms.size();

  if (numPrivateVars != numPrivateSyms)
    return op.emitError() << "inconsistent number of private variables and "
                             "privatizer op symbols, private vars: "
                          << numPrivateVars
                          << " vs. privatizer op symbols: " << numPrivateSyms;

  for (auto privateVarInfo : llvm::zip_equal(privateVars, privateSyms)) {
    Type varType = std::get<0>(privateVarInfo).getType();
    SymbolRefAttr privateSym = cast<SymbolRefAttr>(std::get<1>(privateVarInfo));
    PrivateClauseOp privatizerOp =
        SymbolTable::lookupNearestSymbolFrom<PrivateClauseOp>(op, privateSym);

    if (privatizerOp == nullptr)
      return op.emitError() << "failed to lookup privatizer op with symbol: '"
                            << privateSym << "'";

    Type privatizerType = privatizerOp.getArgType();

    if (privatizerType && (varType != privatizerType))
      return op.emitError()
             << "type mismatch between a "
             << (privatizerOp.getDataSharingType() ==
                         DataSharingClauseType::Private
                     ? "private"
                     : "firstprivate")
             << " variable and its privatizer op, var type: " << varType
             << " vs. privatizer op type: " << privatizerType;
  }

  return success();
}

LogicalResult ParallelOp::verify() {
  if (getAllocateVars().size() != getAllocatorVars().size())
    return emitError(
        "expected equal sizes for allocate and allocator variables");

  if (failed(verifyPrivateVarList(*this)))
    return failure();

  return verifyReductionVarList(*this, getReductionSyms(), getReductionVars(),
                                getReductionByref());
}

LogicalResult ParallelOp::verifyRegions() {
  auto distChildOps = getOps<DistributeOp>();
  int numDistChildOps = std::distance(distChildOps.begin(), distChildOps.end());
  if (numDistChildOps > 1)
    return emitError()
           << "multiple 'omp.distribute' nested inside of 'omp.parallel'";

  if (numDistChildOps == 1) {
    if (!isComposite())
      return emitError()
             << "'omp.composite' attribute missing from composite operation";

    auto *ompDialect = getContext()->getLoadedDialect<OpenMPDialect>();
    Operation &distributeOp = **distChildOps.begin();
    for (Operation &childOp : getOps()) {
      if (&childOp == &distributeOp || ompDialect != childOp.getDialect())
        continue;

      if (!childOp.hasTrait<OpTrait::IsTerminator>())
        return emitError() << "unexpected OpenMP operation inside of composite "
                              "'omp.parallel': "
                           << childOp.getName();
    }
  } else if (isComposite()) {
    return emitError()
           << "'omp.composite' attribute present in non-composite operation";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TeamsOp
//===----------------------------------------------------------------------===//

static bool opInGlobalImplicitParallelRegion(Operation *op) {
  while ((op = op->getParentOp()))
    if (isa<OpenMPDialect>(op->getDialect()))
      return false;
  return true;
}

void TeamsOp::build(OpBuilder &builder, OperationState &state,
                    const TeamsOperands &clauses) {
  MLIRContext *ctx = builder.getContext();
  // TODO Store clauses in op: privateVars, privateSyms, privateNeedsBarrier
  TeamsOp::build(builder, state, clauses.allocateVars, clauses.allocatorVars,
                 clauses.ifExpr, clauses.numTeamsLower, clauses.numTeamsUpper,
                 /*private_vars=*/{}, /*private_syms=*/nullptr,
                 /*private_needs_barrier=*/nullptr, clauses.reductionMod,
                 clauses.reductionVars,
                 makeDenseBoolArrayAttr(ctx, clauses.reductionByref),
                 makeArrayAttr(ctx, clauses.reductionSyms),
                 clauses.threadLimit);
}

LogicalResult TeamsOp::verify() {
  // Check parent region
  // TODO If nested inside of a target region, also check that it does not
  // contain any statements, declarations or directives other than this
  // omp.teams construct. The issue is how to support the initialization of
  // this operation's own arguments (allow SSA values across omp.target?).
  Operation *op = getOperation();
  if (!isa<TargetOp>(op->getParentOp()) &&
      !opInGlobalImplicitParallelRegion(op))
    return emitError("expected to be nested inside of omp.target or not nested "
                     "in any OpenMP dialect operations");

  // Check for num_teams clause restrictions
  if (auto numTeamsLowerBound = getNumTeamsLower()) {
    auto numTeamsUpperBound = getNumTeamsUpper();
    if (!numTeamsUpperBound)
      return emitError("expected num_teams upper bound to be defined if the "
                       "lower bound is defined");
    if (numTeamsLowerBound.getType() != numTeamsUpperBound.getType())
      return emitError(
          "expected num_teams upper bound and lower bound to be the same type");
  }

  // Check for allocate clause restrictions
  if (getAllocateVars().size() != getAllocatorVars().size())
    return emitError(
        "expected equal sizes for allocate and allocator variables");

  return verifyReductionVarList(*this, getReductionSyms(), getReductionVars(),
                                getReductionByref());
}

//===----------------------------------------------------------------------===//
// SectionOp
//===----------------------------------------------------------------------===//

OperandRange SectionOp::getPrivateVars() {
  return getParentOp().getPrivateVars();
}

OperandRange SectionOp::getReductionVars() {
  return getParentOp().getReductionVars();
}

//===----------------------------------------------------------------------===//
// SectionsOp
//===----------------------------------------------------------------------===//

void SectionsOp::build(OpBuilder &builder, OperationState &state,
                       const SectionsOperands &clauses) {
  MLIRContext *ctx = builder.getContext();
  // TODO Store clauses in op: privateVars, privateSyms, privateNeedsBarrier
  SectionsOp::build(builder, state, clauses.allocateVars, clauses.allocatorVars,
                    clauses.nowait, /*private_vars=*/{},
                    /*private_syms=*/nullptr, /*private_needs_barrier=*/nullptr,
                    clauses.reductionMod, clauses.reductionVars,
                    makeDenseBoolArrayAttr(ctx, clauses.reductionByref),
                    makeArrayAttr(ctx, clauses.reductionSyms));
}

LogicalResult SectionsOp::verify() {
  if (getAllocateVars().size() != getAllocatorVars().size())
    return emitError(
        "expected equal sizes for allocate and allocator variables");

  return verifyReductionVarList(*this, getReductionSyms(), getReductionVars(),
                                getReductionByref());
}

LogicalResult SectionsOp::verifyRegions() {
  for (auto &inst : *getRegion().begin()) {
    if (!(isa<SectionOp>(inst) || isa<TerminatorOp>(inst))) {
      return emitOpError()
             << "expected omp.section op or terminator op inside region";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SingleOp
//===----------------------------------------------------------------------===//

void SingleOp::build(OpBuilder &builder, OperationState &state,
                     const SingleOperands &clauses) {
  MLIRContext *ctx = builder.getContext();
  // TODO Store clauses in op: privateVars, privateSyms, privateNeedsBarrier
  SingleOp::build(builder, state, clauses.allocateVars, clauses.allocatorVars,
                  clauses.copyprivateVars,
                  makeArrayAttr(ctx, clauses.copyprivateSyms), clauses.nowait,
                  /*private_vars=*/{}, /*private_syms=*/nullptr,
                  /*private_needs_barrier=*/nullptr);
}

LogicalResult SingleOp::verify() {
  // Check for allocate clause restrictions
  if (getAllocateVars().size() != getAllocatorVars().size())
    return emitError(
        "expected equal sizes for allocate and allocator variables");

  return verifyCopyprivateVarList(*this, getCopyprivateVars(),
                                  getCopyprivateSyms());
}

//===----------------------------------------------------------------------===//
// WorkshareOp
//===----------------------------------------------------------------------===//

void WorkshareOp::build(OpBuilder &builder, OperationState &state,
                        const WorkshareOperands &clauses) {
  WorkshareOp::build(builder, state, clauses.nowait);
}

//===----------------------------------------------------------------------===//
// WorkshareLoopWrapperOp
//===----------------------------------------------------------------------===//

LogicalResult WorkshareLoopWrapperOp::verify() {
  if (!(*this)->getParentOfType<WorkshareOp>())
    return emitOpError() << "must be nested in an omp.workshare";
  return success();
}

LogicalResult WorkshareLoopWrapperOp::verifyRegions() {
  if (isa_and_nonnull<LoopWrapperInterface>((*this)->getParentOp()) ||
      getNestedWrapper())
    return emitOpError() << "expected to be a standalone loop wrapper";

  return success();
}

//===----------------------------------------------------------------------===//
// LoopWrapperInterface
//===----------------------------------------------------------------------===//

LogicalResult LoopWrapperInterface::verifyImpl() {
  Operation *op = this->getOperation();
  if (!op->hasTrait<OpTrait::NoTerminator>() ||
      !op->hasTrait<OpTrait::SingleBlock>())
    return emitOpError() << "loop wrapper must also have the `NoTerminator` "
                            "and `SingleBlock` traits";

  if (op->getNumRegions() != 1)
    return emitOpError() << "loop wrapper does not contain exactly one region";

  Region &region = op->getRegion(0);
  if (range_size(region.getOps()) != 1)
    return emitOpError()
           << "loop wrapper does not contain exactly one nested op";

  Operation &firstOp = *region.op_begin();
  if (!isa<LoopNestOp, LoopWrapperInterface>(firstOp))
    return emitOpError() << "nested in loop wrapper is not another loop "
                            "wrapper or `omp.loop_nest`";

  return success();
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

void LoopOp::build(OpBuilder &builder, OperationState &state,
                   const LoopOperands &clauses) {
  MLIRContext *ctx = builder.getContext();

  LoopOp::build(builder, state, clauses.bindKind, clauses.privateVars,
                makeArrayAttr(ctx, clauses.privateSyms),
                clauses.privateNeedsBarrier, clauses.order, clauses.orderMod,
                clauses.reductionMod, clauses.reductionVars,
                makeDenseBoolArrayAttr(ctx, clauses.reductionByref),
                makeArrayAttr(ctx, clauses.reductionSyms));
}

LogicalResult LoopOp::verify() {
  return verifyReductionVarList(*this, getReductionSyms(), getReductionVars(),
                                getReductionByref());
}

LogicalResult LoopOp::verifyRegions() {
  if (llvm::isa_and_nonnull<LoopWrapperInterface>((*this)->getParentOp()) ||
      getNestedWrapper())
    return emitOpError() << "expected to be a standalone loop wrapper";

  return success();
}

//===----------------------------------------------------------------------===//
// WsloopOp
//===----------------------------------------------------------------------===//

void WsloopOp::build(OpBuilder &builder, OperationState &state,
                     ArrayRef<NamedAttribute> attributes) {
  build(builder, state, /*allocate_vars=*/{}, /*allocator_vars=*/{},
        /*linear_vars=*/ValueRange(), /*linear_step_vars=*/ValueRange(),
        /*nowait=*/false, /*order=*/nullptr, /*order_mod=*/nullptr,
        /*ordered=*/nullptr, /*private_vars=*/{}, /*private_syms=*/nullptr,
        /*private_needs_barrier=*/false,
        /*reduction_mod=*/nullptr, /*reduction_vars=*/ValueRange(),
        /*reduction_byref=*/nullptr,
        /*reduction_syms=*/nullptr, /*schedule_kind=*/nullptr,
        /*schedule_chunk=*/nullptr, /*schedule_mod=*/nullptr,
        /*schedule_simd=*/false);
  state.addAttributes(attributes);
}

void WsloopOp::build(OpBuilder &builder, OperationState &state,
                     const WsloopOperands &clauses) {
  MLIRContext *ctx = builder.getContext();
  // TODO: Store clauses in op: allocateVars, allocatorVars
  WsloopOp::build(
      builder, state,
      /*allocate_vars=*/{}, /*allocator_vars=*/{}, clauses.linearVars,
      clauses.linearStepVars, clauses.nowait, clauses.order, clauses.orderMod,
      clauses.ordered, clauses.privateVars,
      makeArrayAttr(ctx, clauses.privateSyms), clauses.privateNeedsBarrier,
      clauses.reductionMod, clauses.reductionVars,
      makeDenseBoolArrayAttr(ctx, clauses.reductionByref),
      makeArrayAttr(ctx, clauses.reductionSyms), clauses.scheduleKind,
      clauses.scheduleChunk, clauses.scheduleMod, clauses.scheduleSimd);
}

LogicalResult WsloopOp::verify() {
  return verifyReductionVarList(*this, getReductionSyms(), getReductionVars(),
                                getReductionByref());
}

LogicalResult WsloopOp::verifyRegions() {
  bool isCompositeChildLeaf =
      llvm::dyn_cast_if_present<LoopWrapperInterface>((*this)->getParentOp());

  if (LoopWrapperInterface nested = getNestedWrapper()) {
    if (!isComposite())
      return emitError()
             << "'omp.composite' attribute missing from composite wrapper";

    // Check for the allowed leaf constructs that may appear in a composite
    // construct directly after DO/FOR.
    if (!isa<SimdOp>(nested))
      return emitError() << "only supported nested wrapper is 'omp.simd'";

  } else if (isComposite() && !isCompositeChildLeaf) {
    return emitError()
           << "'omp.composite' attribute present in non-composite wrapper";
  } else if (!isComposite() && isCompositeChildLeaf) {
    return emitError()
           << "'omp.composite' attribute missing from composite wrapper";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Simd construct [2.9.3.1]
//===----------------------------------------------------------------------===//

void SimdOp::build(OpBuilder &builder, OperationState &state,
                   const SimdOperands &clauses) {
  MLIRContext *ctx = builder.getContext();
  // TODO Store clauses in op: linearVars, linearStepVars
  SimdOp::build(builder, state, clauses.alignedVars,
                makeArrayAttr(ctx, clauses.alignments), clauses.ifExpr,
                /*linear_vars=*/{}, /*linear_step_vars=*/{},
                clauses.nontemporalVars, clauses.order, clauses.orderMod,
                clauses.privateVars, makeArrayAttr(ctx, clauses.privateSyms),
                clauses.privateNeedsBarrier, clauses.reductionMod,
                clauses.reductionVars,
                makeDenseBoolArrayAttr(ctx, clauses.reductionByref),
                makeArrayAttr(ctx, clauses.reductionSyms), clauses.safelen,
                clauses.simdlen);
}

LogicalResult SimdOp::verify() {
  if (getSimdlen().has_value() && getSafelen().has_value() &&
      getSimdlen().value() > getSafelen().value())
    return emitOpError()
           << "simdlen clause and safelen clause are both present, but the "
              "simdlen value is not less than or equal to safelen value";

  if (verifyAlignedClause(*this, getAlignments(), getAlignedVars()).failed())
    return failure();

  if (verifyNontemporalClause(*this, getNontemporalVars()).failed())
    return failure();

  bool isCompositeChildLeaf =
      llvm::dyn_cast_if_present<LoopWrapperInterface>((*this)->getParentOp());

  if (!isComposite() && isCompositeChildLeaf)
    return emitError()
           << "'omp.composite' attribute missing from composite wrapper";

  if (isComposite() && !isCompositeChildLeaf)
    return emitError()
           << "'omp.composite' attribute present in non-composite wrapper";

  // Firstprivate is not allowed for SIMD in the standard. Check that none of
  // the private decls are for firstprivate.
  std::optional<ArrayAttr> privateSyms = getPrivateSyms();
  if (privateSyms) {
    for (const Attribute &sym : *privateSyms) {
      auto symRef = cast<SymbolRefAttr>(sym);
      omp::PrivateClauseOp privatizer =
          SymbolTable::lookupNearestSymbolFrom<omp::PrivateClauseOp>(
              getOperation(), symRef);
      if (!privatizer)
        return emitError() << "Cannot find privatizer '" << symRef << "'";
      if (privatizer.getDataSharingType() ==
          DataSharingClauseType::FirstPrivate)
        return emitError() << "FIRSTPRIVATE cannot be used with SIMD";
    }
  }

  return success();
}

LogicalResult SimdOp::verifyRegions() {
  if (getNestedWrapper())
    return emitOpError() << "must wrap an 'omp.loop_nest' directly";

  return success();
}

//===----------------------------------------------------------------------===//
// Distribute construct [2.9.4.1]
//===----------------------------------------------------------------------===//

void DistributeOp::build(OpBuilder &builder, OperationState &state,
                         const DistributeOperands &clauses) {
  DistributeOp::build(builder, state, clauses.allocateVars,
                      clauses.allocatorVars, clauses.distScheduleStatic,
                      clauses.distScheduleChunkSize, clauses.order,
                      clauses.orderMod, clauses.privateVars,
                      makeArrayAttr(builder.getContext(), clauses.privateSyms),
                      clauses.privateNeedsBarrier);
}

LogicalResult DistributeOp::verify() {
  if (this->getDistScheduleChunkSize() && !this->getDistScheduleStatic())
    return emitOpError() << "chunk size set without "
                            "dist_schedule_static being present";

  if (getAllocateVars().size() != getAllocatorVars().size())
    return emitError(
        "expected equal sizes for allocate and allocator variables");

  return success();
}

LogicalResult DistributeOp::verifyRegions() {
  if (LoopWrapperInterface nested = getNestedWrapper()) {
    if (!isComposite())
      return emitError()
             << "'omp.composite' attribute missing from composite wrapper";
    // Check for the allowed leaf constructs that may appear in a composite
    // construct directly after DISTRIBUTE.
    if (isa<WsloopOp>(nested)) {
      Operation *parentOp = (*this)->getParentOp();
      if (!llvm::dyn_cast_if_present<ParallelOp>(parentOp) ||
          !cast<ComposableOpInterface>(parentOp).isComposite()) {
        return emitError() << "an 'omp.wsloop' nested wrapper is only allowed "
                              "when a composite 'omp.parallel' is the direct "
                              "parent";
      }
    } else if (!isa<SimdOp>(nested))
      return emitError() << "only supported nested wrappers are 'omp.simd' and "
                            "'omp.wsloop'";
  } else if (isComposite()) {
    return emitError()
           << "'omp.composite' attribute present in non-composite wrapper";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DeclareMapperOp / DeclareMapperInfoOp
//===----------------------------------------------------------------------===//

LogicalResult DeclareMapperInfoOp::verify() {
  return verifyMapClause(*this, getMapVars());
}

LogicalResult DeclareMapperOp::verifyRegions() {
  if (!llvm::isa_and_present<DeclareMapperInfoOp>(
          getRegion().getBlocks().front().getTerminator()))
    return emitOpError() << "expected terminator to be a DeclareMapperInfoOp";

  return success();
}

//===----------------------------------------------------------------------===//
// DeclareReductionOp
//===----------------------------------------------------------------------===//

LogicalResult DeclareReductionOp::verifyRegions() {
  if (!getAllocRegion().empty()) {
    for (YieldOp yieldOp : getAllocRegion().getOps<YieldOp>()) {
      if (yieldOp.getResults().size() != 1 ||
          yieldOp.getResults().getTypes()[0] != getType())
        return emitOpError() << "expects alloc region to yield a value "
                                "of the reduction type";
    }
  }

  if (getInitializerRegion().empty())
    return emitOpError() << "expects non-empty initializer region";
  Block &initializerEntryBlock = getInitializerRegion().front();

  if (initializerEntryBlock.getNumArguments() == 1) {
    if (!getAllocRegion().empty())
      return emitOpError() << "expects two arguments to the initializer region "
                              "when an allocation region is used";
  } else if (initializerEntryBlock.getNumArguments() == 2) {
    if (getAllocRegion().empty())
      return emitOpError() << "expects one argument to the initializer region "
                              "when no allocation region is used";
  } else {
    return emitOpError()
           << "expects one or two arguments to the initializer region";
  }

  for (mlir::Value arg : initializerEntryBlock.getArguments())
    if (arg.getType() != getType())
      return emitOpError() << "expects initializer region argument to match "
                              "the reduction type";

  for (YieldOp yieldOp : getInitializerRegion().getOps<YieldOp>()) {
    if (yieldOp.getResults().size() != 1 ||
        yieldOp.getResults().getTypes()[0] != getType())
      return emitOpError() << "expects initializer region to yield a value "
                              "of the reduction type";
  }

  if (getReductionRegion().empty())
    return emitOpError() << "expects non-empty reduction region";
  Block &reductionEntryBlock = getReductionRegion().front();
  if (reductionEntryBlock.getNumArguments() != 2 ||
      reductionEntryBlock.getArgumentTypes()[0] !=
          reductionEntryBlock.getArgumentTypes()[1] ||
      reductionEntryBlock.getArgumentTypes()[0] != getType())
    return emitOpError() << "expects reduction region with two arguments of "
                            "the reduction type";
  for (YieldOp yieldOp : getReductionRegion().getOps<YieldOp>()) {
    if (yieldOp.getResults().size() != 1 ||
        yieldOp.getResults().getTypes()[0] != getType())
      return emitOpError() << "expects reduction region to yield a value "
                              "of the reduction type";
  }

  if (!getAtomicReductionRegion().empty()) {
    Block &atomicReductionEntryBlock = getAtomicReductionRegion().front();
    if (atomicReductionEntryBlock.getNumArguments() != 2 ||
        atomicReductionEntryBlock.getArgumentTypes()[0] !=
            atomicReductionEntryBlock.getArgumentTypes()[1])
      return emitOpError() << "expects atomic reduction region with two "
                              "arguments of the same type";
    auto ptrType = llvm::dyn_cast<PointerLikeType>(
        atomicReductionEntryBlock.getArgumentTypes()[0]);
    if (!ptrType ||
        (ptrType.getElementType() && ptrType.getElementType() != getType()))
      return emitOpError() << "expects atomic reduction region arguments to "
                              "be accumulators containing the reduction type";
  }

  if (getCleanupRegion().empty())
    return success();
  Block &cleanupEntryBlock = getCleanupRegion().front();
  if (cleanupEntryBlock.getNumArguments() != 1 ||
      cleanupEntryBlock.getArgument(0).getType() != getType())
    return emitOpError() << "expects cleanup region with one argument "
                            "of the reduction type";

  return success();
}

//===----------------------------------------------------------------------===//
// TaskOp
//===----------------------------------------------------------------------===//

void TaskOp::build(OpBuilder &builder, OperationState &state,
                   const TaskOperands &clauses) {
  MLIRContext *ctx = builder.getContext();
  TaskOp::build(builder, state, clauses.allocateVars, clauses.allocatorVars,
                makeArrayAttr(ctx, clauses.dependKinds), clauses.dependVars,
                clauses.final, clauses.ifExpr, clauses.inReductionVars,
                makeDenseBoolArrayAttr(ctx, clauses.inReductionByref),
                makeArrayAttr(ctx, clauses.inReductionSyms), clauses.mergeable,
                clauses.priority, /*private_vars=*/clauses.privateVars,
                /*private_syms=*/makeArrayAttr(ctx, clauses.privateSyms),
                clauses.privateNeedsBarrier, clauses.untied,
                clauses.eventHandle);
}

LogicalResult TaskOp::verify() {
  LogicalResult verifyDependVars =
      verifyDependVarList(*this, getDependKinds(), getDependVars());
  return failed(verifyDependVars)
             ? verifyDependVars
             : verifyReductionVarList(*this, getInReductionSyms(),
                                      getInReductionVars(),
                                      getInReductionByref());
}

//===----------------------------------------------------------------------===//
// TaskgroupOp
//===----------------------------------------------------------------------===//

void TaskgroupOp::build(OpBuilder &builder, OperationState &state,
                        const TaskgroupOperands &clauses) {
  MLIRContext *ctx = builder.getContext();
  TaskgroupOp::build(builder, state, clauses.allocateVars,
                     clauses.allocatorVars, clauses.taskReductionVars,
                     makeDenseBoolArrayAttr(ctx, clauses.taskReductionByref),
                     makeArrayAttr(ctx, clauses.taskReductionSyms));
}

LogicalResult TaskgroupOp::verify() {
  return verifyReductionVarList(*this, getTaskReductionSyms(),
                                getTaskReductionVars(),
                                getTaskReductionByref());
}

//===----------------------------------------------------------------------===//
// TaskloopOp
//===----------------------------------------------------------------------===//

void TaskloopOp::build(OpBuilder &builder, OperationState &state,
                       const TaskloopOperands &clauses) {
  MLIRContext *ctx = builder.getContext();
  TaskloopOp::build(
      builder, state, clauses.allocateVars, clauses.allocatorVars,
      clauses.final, clauses.grainsizeMod, clauses.grainsize, clauses.ifExpr,
      clauses.inReductionVars,
      makeDenseBoolArrayAttr(ctx, clauses.inReductionByref),
      makeArrayAttr(ctx, clauses.inReductionSyms), clauses.mergeable,
      clauses.nogroup, clauses.numTasksMod, clauses.numTasks, clauses.priority,
      /*private_vars=*/clauses.privateVars,
      /*private_syms=*/makeArrayAttr(ctx, clauses.privateSyms),
      clauses.privateNeedsBarrier, clauses.reductionMod, clauses.reductionVars,
      makeDenseBoolArrayAttr(ctx, clauses.reductionByref),
      makeArrayAttr(ctx, clauses.reductionSyms), clauses.untied);
}

LogicalResult TaskloopOp::verify() {
  if (getAllocateVars().size() != getAllocatorVars().size())
    return emitError(
        "expected equal sizes for allocate and allocator variables");
  if (failed(verifyReductionVarList(*this, getReductionSyms(),
                                    getReductionVars(), getReductionByref())) ||
      failed(verifyReductionVarList(*this, getInReductionSyms(),
                                    getInReductionVars(),
                                    getInReductionByref())))
    return failure();

  if (!getReductionVars().empty() && getNogroup())
    return emitError("if a reduction clause is present on the taskloop "
                     "directive, the nogroup clause must not be specified");
  for (auto var : getReductionVars()) {
    if (llvm::is_contained(getInReductionVars(), var))
      return emitError("the same list item cannot appear in both a reduction "
                       "and an in_reduction clause");
  }

  if (getGrainsize() && getNumTasks()) {
    return emitError(
        "the grainsize clause and num_tasks clause are mutually exclusive and "
        "may not appear on the same taskloop directive");
  }

  return success();
}

LogicalResult TaskloopOp::verifyRegions() {
  if (LoopWrapperInterface nested = getNestedWrapper()) {
    if (!isComposite())
      return emitError()
             << "'omp.composite' attribute missing from composite wrapper";

    // Check for the allowed leaf constructs that may appear in a composite
    // construct directly after TASKLOOP.
    if (!isa<SimdOp>(nested))
      return emitError() << "only supported nested wrapper is 'omp.simd'";
  } else if (isComposite()) {
    return emitError()
           << "'omp.composite' attribute present in non-composite wrapper";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LoopNestOp
//===----------------------------------------------------------------------===//

ParseResult LoopNestOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse an opening `(` followed by induction variables followed by `)`
  SmallVector<OpAsmParser::Argument> ivs;
  SmallVector<OpAsmParser::UnresolvedOperand> lbs, ubs;
  Type loopVarType;
  if (parser.parseArgumentList(ivs, OpAsmParser::Delimiter::Paren) ||
      parser.parseColonType(loopVarType) ||
      // Parse loop bounds.
      parser.parseEqual() ||
      parser.parseOperandList(lbs, ivs.size(), OpAsmParser::Delimiter::Paren) ||
      parser.parseKeyword("to") ||
      parser.parseOperandList(ubs, ivs.size(), OpAsmParser::Delimiter::Paren))
    return failure();

  for (auto &iv : ivs)
    iv.type = loopVarType;

  auto *ctx = parser.getBuilder().getContext();
  // Parse "inclusive" flag.
  if (succeeded(parser.parseOptionalKeyword("inclusive")))
    result.addAttribute("loop_inclusive", UnitAttr::get(ctx));

  // Parse step values.
  SmallVector<OpAsmParser::UnresolvedOperand> steps;
  if (parser.parseKeyword("step") ||
      parser.parseOperandList(steps, ivs.size(), OpAsmParser::Delimiter::Paren))
    return failure();

  // Parse collapse
  int64_t value = 0;
  if (!parser.parseOptionalKeyword("collapse") &&
      (parser.parseLParen() || parser.parseInteger(value) ||
       parser.parseRParen()))
    return failure();
  if (value > 1)
    result.addAttribute(
        "collapse_num_loops",
        IntegerAttr::get(parser.getBuilder().getI64Type(), value));

  // Parse tiles
  SmallVector<int64_t> tiles;
  auto parseTiles = [&]() -> ParseResult {
    int64_t tile;
    if (parser.parseInteger(tile))
      return failure();
    tiles.push_back(tile);
    return success();
  };

  if (!parser.parseOptionalKeyword("tiles") &&
      (parser.parseLParen() || parser.parseCommaSeparatedList(parseTiles) ||
       parser.parseRParen()))
    return failure();

  if (tiles.size() > 0)
    result.addAttribute("tile_sizes", DenseI64ArrayAttr::get(ctx, tiles));

  // Parse the body.
  Region *region = result.addRegion();
  if (parser.parseRegion(*region, ivs))
    return failure();

  // Resolve operands.
  if (parser.resolveOperands(lbs, loopVarType, result.operands) ||
      parser.resolveOperands(ubs, loopVarType, result.operands) ||
      parser.resolveOperands(steps, loopVarType, result.operands))
    return failure();

  // Parse the optional attribute list.
  return parser.parseOptionalAttrDict(result.attributes);
}

void LoopNestOp::print(OpAsmPrinter &p) {
  Region &region = getRegion();
  auto args = region.getArguments();
  p << " (" << args << ") : " << args[0].getType() << " = ("
    << getLoopLowerBounds() << ") to (" << getLoopUpperBounds() << ") ";
  if (getLoopInclusive())
    p << "inclusive ";
  p << "step (" << getLoopSteps() << ") ";
  if (int64_t numCollapse = getCollapseNumLoops())
    if (numCollapse > 1)
      p << "collapse(" << numCollapse << ") ";

  if (const auto tiles = getTileSizes())
    p << "tiles(" << tiles.value() << ") ";

  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

void LoopNestOp::build(OpBuilder &builder, OperationState &state,
                       const LoopNestOperands &clauses) {
  MLIRContext *ctx = builder.getContext();
  LoopNestOp::build(builder, state, clauses.collapseNumLoops,
                    clauses.loopLowerBounds, clauses.loopUpperBounds,
                    clauses.loopSteps, clauses.loopInclusive,
                    makeDenseI64ArrayAttr(ctx, clauses.tileSizes));
}

LogicalResult LoopNestOp::verify() {
  if (getLoopLowerBounds().empty())
    return emitOpError() << "must represent at least one loop";

  if (getLoopLowerBounds().size() != getIVs().size())
    return emitOpError() << "number of range arguments and IVs do not match";

  for (auto [lb, iv] : llvm::zip_equal(getLoopLowerBounds(), getIVs())) {
    if (lb.getType() != iv.getType())
      return emitOpError()
             << "range argument type does not match corresponding IV type";
  }

  uint64_t numIVs = getIVs().size();

  if (const auto &numCollapse = getCollapseNumLoops())
    if (numCollapse > numIVs)
      return emitOpError()
             << "collapse value is larger than the number of loops";

  if (const auto &tiles = getTileSizes())
    if (tiles.value().size() > numIVs)
      return emitOpError() << "too few canonical loops for tile dimensions";

  if (!llvm::dyn_cast_if_present<LoopWrapperInterface>((*this)->getParentOp()))
    return emitOpError() << "expects parent op to be a loop wrapper";

  return success();
}

void LoopNestOp::gatherWrappers(
    SmallVectorImpl<LoopWrapperInterface> &wrappers) {
  Operation *parent = (*this)->getParentOp();
  while (auto wrapper =
             llvm::dyn_cast_if_present<LoopWrapperInterface>(parent)) {
    wrappers.push_back(wrapper);
    parent = parent->getParentOp();
  }
}

//===----------------------------------------------------------------------===//
// OpenMP canonical loop handling
//===----------------------------------------------------------------------===//

std::tuple<NewCliOp, OpOperand *, OpOperand *>
mlir::omp ::decodeCli(Value cli) {

  // Defining a CLI for a generated loop is optional; if there is none then
  // there is no followup-tranformation
  if (!cli)
    return {{}, nullptr, nullptr};

  assert(cli.getType() == CanonicalLoopInfoType::get(cli.getContext()) &&
         "Unexpected type of cli");

  NewCliOp create = cast<NewCliOp>(cli.getDefiningOp());
  OpOperand *gen = nullptr;
  OpOperand *cons = nullptr;
  for (OpOperand &use : cli.getUses()) {
    auto op = cast<LoopTransformationInterface>(use.getOwner());

    unsigned opnum = use.getOperandNumber();
    if (op.isGeneratee(opnum)) {
      assert(!gen && "Each CLI may have at most one def");
      gen = &use;
    } else if (op.isApplyee(opnum)) {
      assert(!cons && "Each CLI may have at most one consumer");
      cons = &use;
    } else {
      llvm_unreachable("Unexpected operand for a CLI");
    }
  }

  return {create, gen, cons};
}

void NewCliOp::build(::mlir::OpBuilder &odsBuilder,
                     ::mlir::OperationState &odsState) {
  odsState.addTypes(CanonicalLoopInfoType::get(odsBuilder.getContext()));
}

void NewCliOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  Value result = getResult();
  auto [newCli, gen, cons] = decodeCli(result);

  // Structured binding `gen` cannot be captured in lambdas before C++20
  OpOperand *generator = gen;

  // Derive the CLI variable name from its generator:
  //  * "canonloop" for omp.canonical_loop
  //  * custom name for loop transformation generatees
  //  * "cli" as fallback if no generator
  //  * "_r<idx>" suffix for nested loops, where <idx> is the sequential order
  //  at that level
  //  * "_s<idx>" suffix for operations with multiple regions, where <idx> is
  //  the index of that region
  std::string cliName{"cli"};
  if (gen) {
    cliName =
        TypeSwitch<Operation *, std::string>(gen->getOwner())
            .Case([&](CanonicalLoopOp op) {
              return generateLoopNestingName("canonloop", op);
            })
            .Case([&](UnrollHeuristicOp op) -> std::string {
              llvm_unreachable("heuristic unrolling does not generate a loop");
            })
            .Case([&](TileOp op) -> std::string {
              auto [generateesFirst, generateesCount] =
                  op.getGenerateesODSOperandIndexAndLength();
              unsigned firstGrid = generateesFirst;
              unsigned firstIntratile = generateesFirst + generateesCount / 2;
              unsigned end = generateesFirst + generateesCount;
              unsigned opnum = generator->getOperandNumber();
              // In the OpenMP apply and looprange clauses, indices are 1-based
              if (firstGrid <= opnum && opnum < firstIntratile) {
                unsigned gridnum = opnum - firstGrid + 1;
                return ("grid" + Twine(gridnum)).str();
              }
              if (firstIntratile <= opnum && opnum < end) {
                unsigned intratilenum = opnum - firstIntratile + 1;
                return ("intratile" + Twine(intratilenum)).str();
              }
              llvm_unreachable("Unexpected generatee argument");
            })
            .DefaultUnreachable("TODO: Custom name for this operation");
  }

  setNameFn(result, cliName);
}

LogicalResult NewCliOp::verify() {
  Value cli = getResult();

  assert(cli.getType() == CanonicalLoopInfoType::get(cli.getContext()) &&
         "Unexpected type of cli");

  // Check that the CLI is used in at most generator and one consumer
  OpOperand *gen = nullptr;
  OpOperand *cons = nullptr;
  for (mlir::OpOperand &use : cli.getUses()) {
    auto op = cast<mlir::omp::LoopTransformationInterface>(use.getOwner());

    unsigned opnum = use.getOperandNumber();
    if (op.isGeneratee(opnum)) {
      if (gen) {
        InFlightDiagnostic error =
            emitOpError("CLI must have at most one generator");
        error.attachNote(gen->getOwner()->getLoc())
            .append("first generator here:");
        error.attachNote(use.getOwner()->getLoc())
            .append("second generator here:");
        return error;
      }

      gen = &use;
    } else if (op.isApplyee(opnum)) {
      if (cons) {
        InFlightDiagnostic error =
            emitOpError("CLI must have at most one consumer");
        error.attachNote(cons->getOwner()->getLoc())
            .append("first consumer here:")
            .appendOp(*cons->getOwner(),
                      OpPrintingFlags().printGenericOpForm());
        error.attachNote(use.getOwner()->getLoc())
            .append("second consumer here:")
            .appendOp(*use.getOwner(), OpPrintingFlags().printGenericOpForm());
        return error;
      }

      cons = &use;
    } else {
      llvm_unreachable("Unexpected operand for a CLI");
    }
  }

  // If the CLI is source of a transformation, it must have a generator
  if (cons && !gen) {
    InFlightDiagnostic error = emitOpError("CLI has no generator");
    error.attachNote(cons->getOwner()->getLoc())
        .append("see consumer here: ")
        .appendOp(*cons->getOwner(), OpPrintingFlags().printGenericOpForm());
    return error;
  }

  return success();
}

void CanonicalLoopOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                            Value tripCount) {
  odsState.addOperands(tripCount);
  odsState.addOperands(Value());
  (void)odsState.addRegion();
}

void CanonicalLoopOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                            Value tripCount, ::mlir::Value cli) {
  odsState.addOperands(tripCount);
  odsState.addOperands(cli);
  (void)odsState.addRegion();
}

void CanonicalLoopOp::getAsmBlockNames(OpAsmSetBlockNameFn setNameFn) {
  setNameFn(&getRegion().front(), "body_entry");
}

void CanonicalLoopOp::getAsmBlockArgumentNames(Region &region,
                                               OpAsmSetValueNameFn setNameFn) {
  std::string ivName = generateLoopNestingName("iv", *this);
  setNameFn(region.getArgument(0), ivName);
}

void CanonicalLoopOp::print(OpAsmPrinter &p) {
  if (getCli())
    p << '(' << getCli() << ')';
  p << ' ' << getInductionVar() << " : " << getInductionVar().getType()
    << " in range(" << getTripCount() << ") ";

  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);

  p.printOptionalAttrDict((*this)->getAttrs());
}

mlir::ParseResult CanonicalLoopOp::parse(::mlir::OpAsmParser &parser,
                                         ::mlir::OperationState &result) {
  CanonicalLoopInfoType cliType =
      CanonicalLoopInfoType::get(parser.getContext());

  // Parse (optional) omp.cli identifier
  OpAsmParser::UnresolvedOperand cli;
  SmallVector<mlir::Value, 1> cliOperand;
  if (!parser.parseOptionalLParen()) {
    if (parser.parseOperand(cli) ||
        parser.resolveOperand(cli, cliType, cliOperand) || parser.parseRParen())
      return failure();
  }

  // We derive the type of tripCount from inductionVariable. MLIR requires the
  // type of tripCount to be known when calling resolveOperand so we have parse
  // the type before processing the inductionVariable.
  OpAsmParser::Argument inductionVariable;
  OpAsmParser::UnresolvedOperand tripcount;
  if (parser.parseArgument(inductionVariable, /*allowType*/ true) ||
      parser.parseKeyword("in") || parser.parseKeyword("range") ||
      parser.parseLParen() || parser.parseOperand(tripcount) ||
      parser.parseRParen() ||
      parser.resolveOperand(tripcount, inductionVariable.type, result.operands))
    return failure();

  // Parse the loop body.
  Region *region = result.addRegion();
  if (parser.parseRegion(*region, {inductionVariable}))
    return failure();

  // We parsed the cli operand forst, but because it is optional, it must be
  // last in the operand list.
  result.operands.append(cliOperand);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return mlir::success();
}

LogicalResult CanonicalLoopOp::verify() {
  // The region's entry must accept the induction variable
  // It can also be empty if just created
  if (!getRegion().empty()) {
    Region &region = getRegion();
    if (region.getNumArguments() != 1)
      return emitOpError(
          "Canonical loop region must have exactly one argument");

    if (getInductionVar().getType() != getTripCount().getType())
      return emitOpError(
          "Region argument must be the same type as the trip count");
  }

  return success();
}

Value CanonicalLoopOp::getInductionVar() { return getRegion().getArgument(0); }

std::pair<unsigned, unsigned>
CanonicalLoopOp::getApplyeesODSOperandIndexAndLength() {
  // No applyees
  return {0, 0};
}

std::pair<unsigned, unsigned>
CanonicalLoopOp::getGenerateesODSOperandIndexAndLength() {
  return getODSOperandIndexAndLength(odsIndex_cli);
}

//===----------------------------------------------------------------------===//
// UnrollHeuristicOp
//===----------------------------------------------------------------------===//

void UnrollHeuristicOp::build(::mlir::OpBuilder &odsBuilder,
                              ::mlir::OperationState &odsState,
                              ::mlir::Value cli) {
  odsState.addOperands(cli);
}

void UnrollHeuristicOp::print(OpAsmPrinter &p) {
  p << '(' << getApplyee() << ')';

  p.printOptionalAttrDict((*this)->getAttrs());
}

mlir::ParseResult UnrollHeuristicOp::parse(::mlir::OpAsmParser &parser,
                                           ::mlir::OperationState &result) {
  auto cliType = CanonicalLoopInfoType::get(parser.getContext());

  if (parser.parseLParen())
    return failure();

  OpAsmParser::UnresolvedOperand applyee;
  if (parser.parseOperand(applyee) ||
      parser.resolveOperand(applyee, cliType, result.operands))
    return failure();

  if (parser.parseRParen())
    return failure();

  // Optional output loop (full unrolling has none)
  if (!parser.parseOptionalArrow()) {
    if (parser.parseLParen() || parser.parseRParen())
      return failure();
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return mlir::success();
}

std::pair<unsigned, unsigned>
UnrollHeuristicOp ::getApplyeesODSOperandIndexAndLength() {
  return getODSOperandIndexAndLength(odsIndex_applyee);
}

std::pair<unsigned, unsigned>
UnrollHeuristicOp::getGenerateesODSOperandIndexAndLength() {
  return {0, 0};
}

//===----------------------------------------------------------------------===//
// TileOp
//===----------------------------------------------------------------------===//

static void printLoopTransformClis(OpAsmPrinter &p, TileOp op,
                                   OperandRange generatees,
                                   OperandRange applyees) {
  if (!generatees.empty())
    p << '(' << llvm::interleaved(generatees) << ')';

  if (!applyees.empty())
    p << " <- (" << llvm::interleaved(applyees) << ')';
}

static ParseResult parseLoopTransformClis(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &generateesOperands,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &applyeesOperands) {
  if (parser.parseOptionalLess()) {
    // Syntax 1: generatees present

    if (parser.parseOperandList(generateesOperands,
                                mlir::OpAsmParser::Delimiter::Paren))
      return failure();

    if (parser.parseLess())
      return failure();
  } else {
    // Syntax 2: generatees omitted
  }

  // Parse `<-` (`<` has already been parsed)
  if (parser.parseMinus())
    return failure();

  if (parser.parseOperandList(applyeesOperands,
                              mlir::OpAsmParser::Delimiter::Paren))
    return failure();

  return success();
}

LogicalResult TileOp::verify() {
  if (getApplyees().empty())
    return emitOpError() << "must apply to at least one loop";

  if (getSizes().size() != getApplyees().size())
    return emitOpError() << "there must be one tile size for each applyee";

  if (!getGeneratees().empty() &&
      2 * getSizes().size() != getGeneratees().size())
    return emitOpError()
           << "expecting two times the number of generatees than applyees";

  DenseSet<Value> parentIVs;

  Value parent = getApplyees().front();
  for (auto &&applyee : llvm::drop_begin(getApplyees())) {
    auto [parentCreate, parentGen, parentCons] = decodeCli(parent);
    auto [create, gen, cons] = decodeCli(applyee);

    if (!parentGen)
      return emitOpError() << "applyee CLI has no generator";

    auto parentLoop = dyn_cast_or_null<CanonicalLoopOp>(parentGen->getOwner());
    if (!parentGen)
      return emitOpError()
             << "currently only supports omp.canonical_loop as applyee";

    parentIVs.insert(parentLoop.getInductionVar());

    if (!gen)
      return emitOpError() << "applyee CLI has no generator";
    auto loop = dyn_cast_or_null<CanonicalLoopOp>(gen->getOwner());
    if (!loop)
      return emitOpError()
             << "currently only supports omp.canonical_loop as applyee";

    // Canonical loop must be perfectly nested, i.e. the body of the parent must
    // only contain the omp.canonical_loop of the nested loops, and
    // omp.terminator
    bool isPerfectlyNested = [&]() {
      auto &parentBody = parentLoop.getRegion();
      if (!parentBody.hasOneBlock())
        return false;
      auto &parentBlock = parentBody.getBlocks().front();

      auto nestedLoopIt = parentBlock.begin();
      if (nestedLoopIt == parentBlock.end() ||
          (&*nestedLoopIt != loop.getOperation()))
        return false;

      auto termIt = std::next(nestedLoopIt);
      if (termIt == parentBlock.end() || !isa<TerminatorOp>(termIt))
        return false;

      if (std::next(termIt) != parentBlock.end())
        return false;

      return true;
    }();
    if (!isPerfectlyNested)
      return emitOpError() << "tiled loop nest must be perfectly nested";

    if (parentIVs.contains(loop.getTripCount()))
      return emitOpError() << "tiled loop nest must be rectangular";

    parent = applyee;
  }

  // TODO: The tile sizes must be computed before the loop, but checking this
  // requires dominance analysis. For instance:
  //
  //      %canonloop = omp.new_cli
  //      omp.canonical_loop(%canonloop) %iv : i32 in range(%tc) {
  //        // write to %x
  //        omp.terminator
  //      }
  //      %ts = llvm.load %x
  //      omp.tile <- (%canonloop) sizes(%ts : i32)

  return success();
}

std::pair<unsigned, unsigned> TileOp ::getApplyeesODSOperandIndexAndLength() {
  return getODSOperandIndexAndLength(odsIndex_applyees);
}

std::pair<unsigned, unsigned> TileOp::getGenerateesODSOperandIndexAndLength() {
  return getODSOperandIndexAndLength(odsIndex_generatees);
}

//===----------------------------------------------------------------------===//
// Critical construct (2.17.1)
//===----------------------------------------------------------------------===//

void CriticalDeclareOp::build(OpBuilder &builder, OperationState &state,
                              const CriticalDeclareOperands &clauses) {
  CriticalDeclareOp::build(builder, state, clauses.symName, clauses.hint);
}

LogicalResult CriticalDeclareOp::verify() {
  return verifySynchronizationHint(*this, getHint());
}

LogicalResult CriticalOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (getNameAttr()) {
    SymbolRefAttr symbolRef = getNameAttr();
    auto decl = symbolTable.lookupNearestSymbolFrom<CriticalDeclareOp>(
        *this, symbolRef);
    if (!decl) {
      return emitOpError() << "expected symbol reference " << symbolRef
                           << " to point to a critical declaration";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Ordered construct
//===----------------------------------------------------------------------===//

static LogicalResult verifyOrderedParent(Operation &op) {
  bool hasRegion = op.getNumRegions() > 0;
  auto loopOp = op.getParentOfType<LoopNestOp>();
  if (!loopOp) {
    if (hasRegion)
      return success();

    // TODO: Consider if this needs to be the case only for the standalone
    // variant of the ordered construct.
    return op.emitOpError() << "must be nested inside of a loop";
  }

  Operation *wrapper = loopOp->getParentOp();
  if (auto wsloopOp = dyn_cast<WsloopOp>(wrapper)) {
    IntegerAttr orderedAttr = wsloopOp.getOrderedAttr();
    if (!orderedAttr)
      return op.emitOpError() << "the enclosing worksharing-loop region must "
                                 "have an ordered clause";

    if (hasRegion && orderedAttr.getInt() != 0)
      return op.emitOpError() << "the enclosing loop's ordered clause must not "
                                 "have a parameter present";

    if (!hasRegion && orderedAttr.getInt() == 0)
      return op.emitOpError() << "the enclosing loop's ordered clause must "
                                 "have a parameter present";
  } else if (!isa<SimdOp>(wrapper)) {
    return op.emitOpError() << "must be nested inside of a worksharing, simd "
                               "or worksharing simd loop";
  }
  return success();
}

void OrderedOp::build(OpBuilder &builder, OperationState &state,
                      const OrderedOperands &clauses) {
  OrderedOp::build(builder, state, clauses.doacrossDependType,
                   clauses.doacrossNumLoops, clauses.doacrossDependVars);
}

LogicalResult OrderedOp::verify() {
  if (failed(verifyOrderedParent(**this)))
    return failure();

  auto wrapper = (*this)->getParentOfType<WsloopOp>();
  if (!wrapper || *wrapper.getOrdered() != *getDoacrossNumLoops())
    return emitOpError() << "number of variables in depend clause does not "
                         << "match number of iteration variables in the "
                         << "doacross loop";

  return success();
}

void OrderedRegionOp::build(OpBuilder &builder, OperationState &state,
                            const OrderedRegionOperands &clauses) {
  OrderedRegionOp::build(builder, state, clauses.parLevelSimd);
}

LogicalResult OrderedRegionOp::verify() { return verifyOrderedParent(**this); }

//===----------------------------------------------------------------------===//
// TaskwaitOp
//===----------------------------------------------------------------------===//

void TaskwaitOp::build(OpBuilder &builder, OperationState &state,
                       const TaskwaitOperands &clauses) {
  // TODO Store clauses in op: dependKinds, dependVars, nowait.
  TaskwaitOp::build(builder, state, /*depend_kinds=*/nullptr,
                    /*depend_vars=*/{}, /*nowait=*/nullptr);
}

//===----------------------------------------------------------------------===//
// Verifier for AtomicReadOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicReadOp::verify() {
  if (verifyCommon().failed())
    return mlir::failure();

  if (auto mo = getMemoryOrder()) {
    if (*mo == ClauseMemoryOrderKind::Acq_rel ||
        *mo == ClauseMemoryOrderKind::Release) {
      return emitError(
          "memory-order must not be acq_rel or release for atomic reads");
    }
  }
  return verifySynchronizationHint(*this, getHint());
}

//===----------------------------------------------------------------------===//
// Verifier for AtomicWriteOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicWriteOp::verify() {
  if (verifyCommon().failed())
    return mlir::failure();

  if (auto mo = getMemoryOrder()) {
    if (*mo == ClauseMemoryOrderKind::Acq_rel ||
        *mo == ClauseMemoryOrderKind::Acquire) {
      return emitError(
          "memory-order must not be acq_rel or acquire for atomic writes");
    }
  }
  return verifySynchronizationHint(*this, getHint());
}

//===----------------------------------------------------------------------===//
// Verifier for AtomicUpdateOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicUpdateOp::canonicalize(AtomicUpdateOp op,
                                           PatternRewriter &rewriter) {
  if (op.isNoOp()) {
    rewriter.eraseOp(op);
    return success();
  }
  if (Value writeVal = op.getWriteOpVal()) {
    rewriter.replaceOpWithNewOp<AtomicWriteOp>(
        op, op.getX(), writeVal, op.getHintAttr(), op.getMemoryOrderAttr());
    return success();
  }
  return failure();
}

LogicalResult AtomicUpdateOp::verify() {
  if (verifyCommon().failed())
    return mlir::failure();

  if (auto mo = getMemoryOrder()) {
    if (*mo == ClauseMemoryOrderKind::Acq_rel ||
        *mo == ClauseMemoryOrderKind::Acquire) {
      return emitError(
          "memory-order must not be acq_rel or acquire for atomic updates");
    }
  }

  return verifySynchronizationHint(*this, getHint());
}

LogicalResult AtomicUpdateOp::verifyRegions() { return verifyRegionsCommon(); }

//===----------------------------------------------------------------------===//
// Verifier for AtomicCaptureOp
//===----------------------------------------------------------------------===//

AtomicReadOp AtomicCaptureOp::getAtomicReadOp() {
  if (auto op = dyn_cast<AtomicReadOp>(getFirstOp()))
    return op;
  return dyn_cast<AtomicReadOp>(getSecondOp());
}

AtomicWriteOp AtomicCaptureOp::getAtomicWriteOp() {
  if (auto op = dyn_cast<AtomicWriteOp>(getFirstOp()))
    return op;
  return dyn_cast<AtomicWriteOp>(getSecondOp());
}

AtomicUpdateOp AtomicCaptureOp::getAtomicUpdateOp() {
  if (auto op = dyn_cast<AtomicUpdateOp>(getFirstOp()))
    return op;
  return dyn_cast<AtomicUpdateOp>(getSecondOp());
}

LogicalResult AtomicCaptureOp::verify() {
  return verifySynchronizationHint(*this, getHint());
}

LogicalResult AtomicCaptureOp::verifyRegions() {
  if (verifyRegionsCommon().failed())
    return mlir::failure();

  if (getFirstOp()->getAttr("hint") || getSecondOp()->getAttr("hint"))
    return emitOpError(
        "operations inside capture region must not have hint clause");

  if (getFirstOp()->getAttr("memory_order") ||
      getSecondOp()->getAttr("memory_order"))
    return emitOpError(
        "operations inside capture region must not have memory_order clause");
  return success();
}

//===----------------------------------------------------------------------===//
// CancelOp
//===----------------------------------------------------------------------===//

void CancelOp::build(OpBuilder &builder, OperationState &state,
                     const CancelOperands &clauses) {
  CancelOp::build(builder, state, clauses.cancelDirective, clauses.ifExpr);
}

static Operation *getParentInSameDialect(Operation *thisOp) {
  Operation *parent = thisOp->getParentOp();
  while (parent) {
    if (parent->getDialect() == thisOp->getDialect())
      return parent;
    parent = parent->getParentOp();
  }
  return nullptr;
}

LogicalResult CancelOp::verify() {
  ClauseCancellationConstructType cct = getCancelDirective();
  // The next OpenMP operation in the chain of parents
  Operation *structuralParent = getParentInSameDialect((*this).getOperation());
  if (!structuralParent)
    return emitOpError() << "Orphaned cancel construct";

  if ((cct == ClauseCancellationConstructType::Parallel) &&
      !mlir::isa<ParallelOp>(structuralParent)) {
    return emitOpError() << "cancel parallel must appear "
                         << "inside a parallel region";
  }
  if (cct == ClauseCancellationConstructType::Loop) {
    // structural parent will be omp.loop_nest, directly nested inside
    // omp.wsloop
    auto wsloopOp = mlir::dyn_cast<WsloopOp>(structuralParent->getParentOp());

    if (!wsloopOp) {
      return emitOpError()
             << "cancel loop must appear inside a worksharing-loop region";
    }
    if (wsloopOp.getNowaitAttr()) {
      return emitError() << "A worksharing construct that is canceled "
                         << "must not have a nowait clause";
    }
    if (wsloopOp.getOrderedAttr()) {
      return emitError() << "A worksharing construct that is canceled "
                         << "must not have an ordered clause";
    }

  } else if (cct == ClauseCancellationConstructType::Sections) {
    // structural parent will be an omp.section, directly nested inside
    // omp.sections
    auto sectionsOp =
        mlir::dyn_cast<SectionsOp>(structuralParent->getParentOp());
    if (!sectionsOp) {
      return emitOpError() << "cancel sections must appear "
                           << "inside a sections region";
    }
    if (sectionsOp.getNowait()) {
      return emitError() << "A sections construct that is canceled "
                         << "must not have a nowait clause";
    }
  }
  if ((cct == ClauseCancellationConstructType::Taskgroup) &&
      (!mlir::isa<omp::TaskOp>(structuralParent) &&
       !mlir::isa<omp::TaskloopOp>(structuralParent->getParentOp()))) {
    return emitOpError() << "cancel taskgroup must appear "
                         << "inside a task region";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CancellationPointOp
//===----------------------------------------------------------------------===//

void CancellationPointOp::build(OpBuilder &builder, OperationState &state,
                                const CancellationPointOperands &clauses) {
  CancellationPointOp::build(builder, state, clauses.cancelDirective);
}

LogicalResult CancellationPointOp::verify() {
  ClauseCancellationConstructType cct = getCancelDirective();
  // The next OpenMP operation in the chain of parents
  Operation *structuralParent = getParentInSameDialect((*this).getOperation());
  if (!structuralParent)
    return emitOpError() << "Orphaned cancellation point";

  if ((cct == ClauseCancellationConstructType::Parallel) &&
      !mlir::isa<ParallelOp>(structuralParent)) {
    return emitOpError() << "cancellation point parallel must appear "
                         << "inside a parallel region";
  }
  // Strucutal parent here will be an omp.loop_nest. Get the parent of that to
  // find the wsloop
  if ((cct == ClauseCancellationConstructType::Loop) &&
      !mlir::isa<WsloopOp>(structuralParent->getParentOp())) {
    return emitOpError() << "cancellation point loop must appear "
                         << "inside a worksharing-loop region";
  }
  if ((cct == ClauseCancellationConstructType::Sections) &&
      !mlir::isa<omp::SectionOp>(structuralParent)) {
    return emitOpError() << "cancellation point sections must appear "
                         << "inside a sections region";
  }
  if ((cct == ClauseCancellationConstructType::Taskgroup) &&
      !mlir::isa<omp::TaskOp>(structuralParent)) {
    return emitOpError() << "cancellation point taskgroup must appear "
                         << "inside a task region";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MapBoundsOp
//===----------------------------------------------------------------------===//

LogicalResult MapBoundsOp::verify() {
  auto extent = getExtent();
  auto upperbound = getUpperBound();
  if (!extent && !upperbound)
    return emitError("expected extent or upperbound.");
  return success();
}

void PrivateClauseOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                            TypeRange /*result_types*/, StringAttr symName,
                            TypeAttr type) {
  PrivateClauseOp::build(
      odsBuilder, odsState, symName, type,
      DataSharingClauseTypeAttr::get(odsBuilder.getContext(),
                                     DataSharingClauseType::Private));
}

LogicalResult PrivateClauseOp::verifyRegions() {
  Type argType = getArgType();
  auto verifyTerminator = [&](Operation *terminator,
                              bool yieldsValue) -> LogicalResult {
    if (!terminator->getBlock()->getSuccessors().empty())
      return success();

    if (!llvm::isa<YieldOp>(terminator))
      return mlir::emitError(terminator->getLoc())
             << "expected exit block terminator to be an `omp.yield` op.";

    YieldOp yieldOp = llvm::cast<YieldOp>(terminator);
    TypeRange yieldedTypes = yieldOp.getResults().getTypes();

    if (!yieldsValue) {
      if (yieldedTypes.empty())
        return success();

      return mlir::emitError(terminator->getLoc())
             << "Did not expect any values to be yielded.";
    }

    if (yieldedTypes.size() == 1 && yieldedTypes.front() == argType)
      return success();

    auto error = mlir::emitError(yieldOp.getLoc())
                 << "Invalid yielded value. Expected type: " << argType
                 << ", got: ";

    if (yieldedTypes.empty())
      error << "None";
    else
      error << yieldedTypes;

    return error;
  };

  auto verifyRegion = [&](Region &region, unsigned expectedNumArgs,
                          StringRef regionName,
                          bool yieldsValue) -> LogicalResult {
    assert(!region.empty());

    if (region.getNumArguments() != expectedNumArgs)
      return mlir::emitError(region.getLoc())
             << "`" << regionName << "`: "
             << "expected " << expectedNumArgs
             << " region arguments, got: " << region.getNumArguments();

    for (Block &block : region) {
      // MLIR will verify the absence of the terminator for us.
      if (!block.mightHaveTerminator())
        continue;

      if (failed(verifyTerminator(block.getTerminator(), yieldsValue)))
        return failure();
    }

    return success();
  };

  // Ensure all of the region arguments have the same type
  for (Region *region : getRegions())
    for (Type ty : region->getArgumentTypes())
      if (ty != argType)
        return emitError() << "Region argument type mismatch: got " << ty
                           << " expected " << argType << ".";

  mlir::Region &initRegion = getInitRegion();
  if (!initRegion.empty() &&
      failed(verifyRegion(getInitRegion(), /*expectedNumArgs=*/2, "init",
                          /*yieldsValue=*/true)))
    return failure();

  DataSharingClauseType dsType = getDataSharingType();

  if (dsType == DataSharingClauseType::Private && !getCopyRegion().empty())
    return emitError("`private` clauses do not require a `copy` region.");

  if (dsType == DataSharingClauseType::FirstPrivate && getCopyRegion().empty())
    return emitError(
        "`firstprivate` clauses require at least a `copy` region.");

  if (dsType == DataSharingClauseType::FirstPrivate &&
      failed(verifyRegion(getCopyRegion(), /*expectedNumArgs=*/2, "copy",
                          /*yieldsValue=*/true)))
    return failure();

  if (!getDeallocRegion().empty() &&
      failed(verifyRegion(getDeallocRegion(), /*expectedNumArgs=*/1, "dealloc",
                          /*yieldsValue=*/false)))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// Spec 5.2: Masked construct (10.5)
//===----------------------------------------------------------------------===//

void MaskedOp::build(OpBuilder &builder, OperationState &state,
                     const MaskedOperands &clauses) {
  MaskedOp::build(builder, state, clauses.filteredThreadId);
}

//===----------------------------------------------------------------------===//
// Spec 5.2: Scan construct (5.6)
//===----------------------------------------------------------------------===//

void ScanOp::build(OpBuilder &builder, OperationState &state,
                   const ScanOperands &clauses) {
  ScanOp::build(builder, state, clauses.inclusiveVars, clauses.exclusiveVars);
}

LogicalResult ScanOp::verify() {
  if (hasExclusiveVars() == hasInclusiveVars())
    return emitError(
        "Exactly one of EXCLUSIVE or INCLUSIVE clause is expected");
  if (WsloopOp parentWsLoopOp = (*this)->getParentOfType<WsloopOp>()) {
    if (parentWsLoopOp.getReductionModAttr() &&
        parentWsLoopOp.getReductionModAttr().getValue() ==
            ReductionModifier::inscan)
      return success();
  }
  if (SimdOp parentSimdOp = (*this)->getParentOfType<SimdOp>()) {
    if (parentSimdOp.getReductionModAttr() &&
        parentSimdOp.getReductionModAttr().getValue() ==
            ReductionModifier::inscan)
      return success();
  }
  return emitError("SCAN directive needs to be enclosed within a parent "
                   "worksharing loop construct or SIMD construct with INSCAN "
                   "reduction modifier");
}

/// Verifies align clause in allocate directive

LogicalResult AllocateDirOp::verify() {
  std::optional<uint64_t> align = this->getAlign();

  if (align.has_value()) {
    if ((align.value() > 0) && !llvm::has_single_bit(align.value()))
      return emitError() << "ALIGN value : " << align.value()
                         << " must be power of 2";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TargetAllocMemOp
//===----------------------------------------------------------------------===//

mlir::Type omp::TargetAllocMemOp::getAllocatedType() {
  return getInTypeAttr().getValue();
}

/// operation ::= %res = (`omp.target_alloc_mem`) $device : devicetype,
///                      $in_type ( `(` $typeparams `)` )? ( `,` $shape )?
///                      attr-dict-without-keyword
static mlir::ParseResult parseTargetAllocMemOp(mlir::OpAsmParser &parser,
                                               mlir::OperationState &result) {
  auto &builder = parser.getBuilder();
  bool hasOperands = false;
  std::int32_t typeparamsSize = 0;

  // Parse device number as a new operand
  mlir::OpAsmParser::UnresolvedOperand deviceOperand;
  mlir::Type deviceType;
  if (parser.parseOperand(deviceOperand) || parser.parseColonType(deviceType))
    return mlir::failure();
  if (parser.resolveOperand(deviceOperand, deviceType, result.operands))
    return mlir::failure();
  if (parser.parseComma())
    return mlir::failure();

  mlir::Type intype;
  if (parser.parseType(intype))
    return mlir::failure();
  result.addAttribute("in_type", mlir::TypeAttr::get(intype));
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> operands;
  llvm::SmallVector<mlir::Type> typeVec;
  if (!parser.parseOptionalLParen()) {
    // parse the LEN params of the derived type. (<params> : <types>)
    if (parser.parseOperandList(operands, mlir::OpAsmParser::Delimiter::None) ||
        parser.parseColonTypeList(typeVec) || parser.parseRParen())
      return mlir::failure();
    typeparamsSize = operands.size();
    hasOperands = true;
  }
  std::int32_t shapeSize = 0;
  if (!parser.parseOptionalComma()) {
    // parse size to scale by, vector of n dimensions of type index
    if (parser.parseOperandList(operands, mlir::OpAsmParser::Delimiter::None))
      return mlir::failure();
    shapeSize = operands.size() - typeparamsSize;
    auto idxTy = builder.getIndexType();
    for (std::int32_t i = typeparamsSize, end = operands.size(); i != end; ++i)
      typeVec.push_back(idxTy);
    hasOperands = true;
  }
  if (hasOperands &&
      parser.resolveOperands(operands, typeVec, parser.getNameLoc(),
                             result.operands))
    return mlir::failure();

  mlir::Type restype = builder.getIntegerType(64);
  if (!restype) {
    parser.emitError(parser.getNameLoc(), "invalid allocate type: ") << intype;
    return mlir::failure();
  }
  llvm::SmallVector<std::int32_t> segmentSizes{1, typeparamsSize, shapeSize};
  result.addAttribute("operandSegmentSizes",
                      builder.getDenseI32ArrayAttr(segmentSizes));
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.addTypeToList(restype, result.types))
    return mlir::failure();
  return mlir::success();
}

mlir::ParseResult omp::TargetAllocMemOp::parse(mlir::OpAsmParser &parser,
                                               mlir::OperationState &result) {
  return parseTargetAllocMemOp(parser, result);
}

void omp::TargetAllocMemOp::print(mlir::OpAsmPrinter &p) {
  p << " ";
  p.printOperand(getDevice());
  p << " : ";
  p << getDevice().getType();
  p << ", ";
  p << getInType();
  if (!getTypeparams().empty()) {
    p << '(' << getTypeparams() << " : " << getTypeparams().getTypes() << ')';
  }
  for (auto sh : getShape()) {
    p << ", ";
    p.printOperand(sh);
  }
  p.printOptionalAttrDict((*this)->getAttrs(),
                          {"in_type", "operandSegmentSizes"});
}

llvm::LogicalResult omp::TargetAllocMemOp::verify() {
  mlir::Type outType = getType();
  if (!mlir::dyn_cast<IntegerType>(outType))
    return emitOpError("must be a integer type");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// WorkdistributeOp
//===----------------------------------------------------------------------===//

LogicalResult WorkdistributeOp::verify() {
  // Check that region exists and is not empty
  Region &region = getRegion();
  if (region.empty())
    return emitOpError("region cannot be empty");
  // Verify single entry point.
  Block &entryBlock = region.front();
  if (entryBlock.empty())
    return emitOpError("region must contain a structured block");
  // Verify single exit point.
  bool hasTerminator = false;
  for (Block &block : region) {
    if (isa<TerminatorOp>(block.back())) {
      if (hasTerminator) {
        return emitOpError("region must have exactly one terminator");
      }
      hasTerminator = true;
    }
  }
  if (!hasTerminator) {
    return emitOpError("region must be terminated with omp.terminator");
  }
  auto walkResult = region.walk([&](Operation *op) -> WalkResult {
    // No implicit barrier at end
    if (isa<BarrierOp>(op)) {
      return emitOpError(
          "explicit barriers are not allowed in workdistribute region");
    }
    // Check for invalid nested constructs
    if (isa<ParallelOp>(op)) {
      return emitOpError(
          "nested parallel constructs not allowed in workdistribute");
    }
    if (isa<TeamsOp>(op)) {
      return emitOpError(
          "nested teams constructs not allowed in workdistribute");
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();

  Operation *parentOp = (*this)->getParentOp();
  if (!llvm::dyn_cast<TeamsOp>(parentOp))
    return emitOpError("workdistribute must be nested under teams");
  return success();
}

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOpsAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOpsTypes.cpp.inc"
