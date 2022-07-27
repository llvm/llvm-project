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
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cstddef>

#include "mlir/Dialect/OpenMP/OpenMPOpsDialect.cpp.inc"
#include "mlir/Dialect/OpenMP/OpenMPOpsEnums.cpp.inc"
#include "mlir/Dialect/OpenMP/OpenMPOpsInterfaces.cpp.inc"
#include "mlir/Dialect/OpenMP/OpenMPTypeInterfaces.cpp.inc"

using namespace mlir;
using namespace mlir::omp;

namespace {
/// Model for pointer-like types that already provide a `getElementType` method.
template <typename T>
struct PointerLikeModel
    : public PointerLikeType::ExternalModel<PointerLikeModel<T>, T> {
  Type getElementType(Type pointer) const {
    return pointer.cast<T>().getElementType();
  }
};
} // namespace

void OpenMPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/OpenMP/OpenMPOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/OpenMP/OpenMPOpsAttributes.cpp.inc"
      >();

  LLVM::LLVMPointerType::attachInterface<
      PointerLikeModel<LLVM::LLVMPointerType>>(*getContext());
  MemRefType::attachInterface<PointerLikeModel<MemRefType>>(*getContext());
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
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operandsAllocate,
    SmallVectorImpl<Type> &typesAllocate,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operandsAllocator,
    SmallVectorImpl<Type> &typesAllocator) {

  return parser.parseCommaSeparatedList([&]() {
    OpAsmParser::UnresolvedOperand operand;
    Type type;
    if (parser.parseOperand(operand) || parser.parseColonType(type))
      return failure();
    operandsAllocator.push_back(operand);
    typesAllocator.push_back(type);
    if (parser.parseArrow())
      return failure();
    if (parser.parseOperand(operand) || parser.parseColonType(type))
      return failure();

    operandsAllocate.push_back(operand);
    typesAllocate.push_back(type);
    return success();
  });
}

/// Print allocate clause
static void printAllocateAndAllocator(OpAsmPrinter &p, Operation *op,
                                      OperandRange varsAllocate,
                                      TypeRange typesAllocate,
                                      OperandRange varsAllocator,
                                      TypeRange typesAllocator) {
  for (unsigned i = 0; i < varsAllocate.size(); ++i) {
    std::string separator = i == varsAllocate.size() - 1 ? "" : ", ";
    p << varsAllocator[i] << " : " << typesAllocator[i] << " -> ";
    p << varsAllocate[i] << " : " << typesAllocate[i] << separator;
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
  if (Optional<ClauseT> enumValue = symbolizeEnum<ClauseT>(enumStr)) {
    attr = ClauseAttr::get(parser.getContext(), *enumValue);
    return success();
  }
  return parser.emitError(loc, "invalid clause value: '") << enumStr << "'";
}

template <typename ClauseAttr>
void printClauseAttr(OpAsmPrinter &p, Operation *op, ClauseAttr attr) {
  p << stringifyEnum(attr.getValue());
}

//===----------------------------------------------------------------------===//
// Parser and printer for Linear Clause
//===----------------------------------------------------------------------===//

/// linear ::= `linear` `(` linear-list `)`
/// linear-list := linear-val | linear-val linear-list
/// linear-val := ssa-id-and-type `=` ssa-id-and-type
static ParseResult
parseLinearClause(OpAsmParser &parser,
                  SmallVectorImpl<OpAsmParser::UnresolvedOperand> &vars,
                  SmallVectorImpl<Type> &types,
                  SmallVectorImpl<OpAsmParser::UnresolvedOperand> &stepVars) {
  return parser.parseCommaSeparatedList([&]() {
    OpAsmParser::UnresolvedOperand var;
    Type type;
    OpAsmParser::UnresolvedOperand stepVar;
    if (parser.parseOperand(var) || parser.parseEqual() ||
        parser.parseOperand(stepVar) || parser.parseColonType(type))
      return failure();

    vars.push_back(var);
    types.push_back(type);
    stepVars.push_back(stepVar);
    return success();
  });
}

/// Print Linear Clause
static void printLinearClause(OpAsmPrinter &p, Operation *op,
                              ValueRange linearVars, TypeRange linearVarTypes,
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
static ParseResult parseScheduleClause(
    OpAsmParser &parser, ClauseScheduleKindAttr &scheduleAttr,
    ScheduleModifierAttr &scheduleModifier, UnitAttr &simdModifier,
    Optional<OpAsmParser::UnresolvedOperand> &chunkSize, Type &chunkType) {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return failure();
  llvm::Optional<mlir::omp::ClauseScheduleKind> schedule =
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
      chunkSize = llvm::NoneType::None;
    }
    break;
  case ClauseScheduleKind::Auto:
  case ClauseScheduleKind::Runtime:
    chunkSize = llvm::NoneType::None;
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
    if (Optional<ScheduleModifier> mod =
            symbolizeScheduleModifier(modifiers[0])) {
      scheduleModifier = ScheduleModifierAttr::get(parser.getContext(), *mod);
    } else {
      return parser.emitError(loc, "invalid schedule modifier");
    }
    // Only SIMD attribute is allowed here!
    if (modifiers.size() > 1) {
      assert(symbolizeScheduleModifier(modifiers[1]) == ScheduleModifier::simd);
      simdModifier = UnitAttr::get(parser.getBuilder().getContext());
    }
  }

  return success();
}

/// Print schedule clause
static void printScheduleClause(OpAsmPrinter &p, Operation *op,
                                ClauseScheduleKindAttr schedAttr,
                                ScheduleModifierAttr modifier, UnitAttr simd,
                                Value scheduleChunkVar,
                                Type scheduleChunkType) {
  p << stringifyClauseScheduleKind(schedAttr.getValue());
  if (scheduleChunkVar)
    p << " = " << scheduleChunkVar << " : " << scheduleChunkVar.getType();
  if (modifier)
    p << ", " << stringifyScheduleModifier(modifier.getValue());
  if (simd)
    p << ", simd";
}

//===----------------------------------------------------------------------===//
// Parser, printer and verifier for ReductionVarList
//===----------------------------------------------------------------------===//

/// reduction-entry-list ::= reduction-entry
///                        | reduction-entry-list `,` reduction-entry
/// reduction-entry ::= symbol-ref `->` ssa-id `:` type
static ParseResult
parseReductionVarList(OpAsmParser &parser,
                      SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
                      SmallVectorImpl<Type> &types,
                      ArrayAttr &redcuctionSymbols) {
  SmallVector<SymbolRefAttr> reductionVec;
  if (failed(parser.parseCommaSeparatedList([&]() {
        if (parser.parseAttribute(reductionVec.emplace_back()) ||
            parser.parseArrow() ||
            parser.parseOperand(operands.emplace_back()) ||
            parser.parseColonType(types.emplace_back()))
          return failure();
        return success();
      })))
    return failure();
  SmallVector<Attribute> reductions(reductionVec.begin(), reductionVec.end());
  redcuctionSymbols = ArrayAttr::get(parser.getContext(), reductions);
  return success();
}

/// Print Reduction clause
static void printReductionVarList(OpAsmPrinter &p, Operation *op,
                                  OperandRange reductionVars,
                                  TypeRange reductionTypes,
                                  Optional<ArrayAttr> reductions) {
  for (unsigned i = 0, e = reductions->size(); i < e; ++i) {
    if (i != 0)
      p << ", ";
    p << (*reductions)[i] << " -> " << reductionVars[i] << " : "
      << reductionVars[i].getType();
  }
}

/// Verifies Reduction Clause
static LogicalResult verifyReductionVarList(Operation *op,
                                            Optional<ArrayAttr> reductions,
                                            OperandRange reductionVars) {
  if (!reductionVars.empty()) {
    if (!reductions || reductions->size() != reductionVars.size())
      return op->emitOpError()
             << "expected as many reduction symbol references "
                "as reduction variables";
  } else {
    if (reductions)
      return op->emitOpError() << "unexpected reduction symbol references";
    return success();
  }

  // TODO: The followings should be done in
  // SymbolUserOpInterface::verifySymbolUses.
  DenseSet<Value> accumulators;
  for (auto args : llvm::zip(reductionVars, *reductions)) {
    Value accum = std::get<0>(args);

    if (!accumulators.insert(accum).second)
      return op->emitOpError() << "accumulator variable used more than once";

    Type varType = accum.getType().cast<PointerLikeType>();
    auto symbolRef = std::get<1>(args).cast<SymbolRefAttr>();
    auto decl =
        SymbolTable::lookupNearestSymbolFrom<ReductionDeclareOp>(op, symbolRef);
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
// ParallelOp
//===----------------------------------------------------------------------===//

void ParallelOp::build(OpBuilder &builder, OperationState &state,
                       ArrayRef<NamedAttribute> attributes) {
  ParallelOp::build(
      builder, state, /*if_expr_var=*/nullptr, /*num_threads_var=*/nullptr,
      /*allocate_vars=*/ValueRange(), /*allocators_vars=*/ValueRange(),
      /*reduction_vars=*/ValueRange(), /*reductions=*/nullptr,
      /*proc_bind_val=*/nullptr);
  state.addAttributes(attributes);
}

LogicalResult ParallelOp::verify() {
  if (allocate_vars().size() != allocators_vars().size())
    return emitError(
        "expected equal sizes for allocate and allocator variables");
  return verifyReductionVarList(*this, reductions(), reduction_vars());
}

//===----------------------------------------------------------------------===//
// Verifier for SectionsOp
//===----------------------------------------------------------------------===//

LogicalResult SectionsOp::verify() {
  if (allocate_vars().size() != allocators_vars().size())
    return emitError(
        "expected equal sizes for allocate and allocator variables");

  return verifyReductionVarList(*this, reductions(), reduction_vars());
}

LogicalResult SectionsOp::verifyRegions() {
  for (auto &inst : *region().begin()) {
    if (!(isa<SectionOp>(inst) || isa<TerminatorOp>(inst))) {
      return emitOpError()
             << "expected omp.section op or terminator op inside region";
    }
  }

  return success();
}

LogicalResult SingleOp::verify() {
  // Check for allocate clause restrictions
  if (allocate_vars().size() != allocators_vars().size())
    return emitError(
        "expected equal sizes for allocate and allocator variables");

  return success();
}

//===----------------------------------------------------------------------===//
// WsLoopOp
//===----------------------------------------------------------------------===//

/// loop-control ::= `(` ssa-id-list `)` `:` type `=`  loop-bounds
/// loop-bounds := `(` ssa-id-list `)` to `(` ssa-id-list `)` inclusive? steps
/// steps := `step` `(`ssa-id-list`)`
ParseResult
parseLoopControl(OpAsmParser &parser, Region &region,
                 SmallVectorImpl<OpAsmParser::UnresolvedOperand> &lowerBound,
                 SmallVectorImpl<OpAsmParser::UnresolvedOperand> &upperBound,
                 SmallVectorImpl<OpAsmParser::UnresolvedOperand> &steps,
                 SmallVectorImpl<Type> &loopVarTypes, UnitAttr &inclusive) {
  // Parse an opening `(` followed by induction variables followed by `)`
  SmallVector<OpAsmParser::Argument> ivs;
  Type loopVarType;
  if (parser.parseArgumentList(ivs, OpAsmParser::Delimiter::Paren) ||
      parser.parseColonType(loopVarType) ||
      // Parse loop bounds.
      parser.parseEqual() ||
      parser.parseOperandList(lowerBound, ivs.size(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.parseKeyword("to") ||
      parser.parseOperandList(upperBound, ivs.size(),
                              OpAsmParser::Delimiter::Paren))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("inclusive")))
    inclusive = UnitAttr::get(parser.getBuilder().getContext());

  // Parse step values.
  if (parser.parseKeyword("step") ||
      parser.parseOperandList(steps, ivs.size(), OpAsmParser::Delimiter::Paren))
    return failure();

  // Now parse the body.
  loopVarTypes = SmallVector<Type>(ivs.size(), loopVarType);
  for (auto &iv : ivs)
    iv.type = loopVarType;
  return parser.parseRegion(region, ivs);
}

void printLoopControl(OpAsmPrinter &p, Operation *op, Region &region,
                      ValueRange lowerBound, ValueRange upperBound,
                      ValueRange steps, TypeRange loopVarTypes,
                      UnitAttr inclusive) {
  auto args = region.front().getArguments();
  p << " (" << args << ") : " << args[0].getType() << " = (" << lowerBound
    << ") to (" << upperBound << ") ";
  if (inclusive)
    p << "inclusive ";
  p << "step (" << steps << ") ";
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// Verifier for Simd construct [2.9.3.1]
//===----------------------------------------------------------------------===//

LogicalResult SimdLoopOp::verify() {
  if (this->lowerBound().empty()) {
    return emitOpError() << "empty lowerbound for simd loop operation";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ReductionOp
//===----------------------------------------------------------------------===//

static ParseResult parseAtomicReductionRegion(OpAsmParser &parser,
                                              Region &region) {
  if (parser.parseOptionalKeyword("atomic"))
    return success();
  return parser.parseRegion(region);
}

static void printAtomicReductionRegion(OpAsmPrinter &printer,
                                       ReductionDeclareOp op, Region &region) {
  if (region.empty())
    return;
  printer << "atomic ";
  printer.printRegion(region);
}

LogicalResult ReductionDeclareOp::verifyRegions() {
  if (initializerRegion().empty())
    return emitOpError() << "expects non-empty initializer region";
  Block &initializerEntryBlock = initializerRegion().front();
  if (initializerEntryBlock.getNumArguments() != 1 ||
      initializerEntryBlock.getArgument(0).getType() != type()) {
    return emitOpError() << "expects initializer region with one argument "
                            "of the reduction type";
  }

  for (YieldOp yieldOp : initializerRegion().getOps<YieldOp>()) {
    if (yieldOp.results().size() != 1 ||
        yieldOp.results().getTypes()[0] != type())
      return emitOpError() << "expects initializer region to yield a value "
                              "of the reduction type";
  }

  if (reductionRegion().empty())
    return emitOpError() << "expects non-empty reduction region";
  Block &reductionEntryBlock = reductionRegion().front();
  if (reductionEntryBlock.getNumArguments() != 2 ||
      reductionEntryBlock.getArgumentTypes()[0] !=
          reductionEntryBlock.getArgumentTypes()[1] ||
      reductionEntryBlock.getArgumentTypes()[0] != type())
    return emitOpError() << "expects reduction region with two arguments of "
                            "the reduction type";
  for (YieldOp yieldOp : reductionRegion().getOps<YieldOp>()) {
    if (yieldOp.results().size() != 1 ||
        yieldOp.results().getTypes()[0] != type())
      return emitOpError() << "expects reduction region to yield a value "
                              "of the reduction type";
  }

  if (atomicReductionRegion().empty())
    return success();

  Block &atomicReductionEntryBlock = atomicReductionRegion().front();
  if (atomicReductionEntryBlock.getNumArguments() != 2 ||
      atomicReductionEntryBlock.getArgumentTypes()[0] !=
          atomicReductionEntryBlock.getArgumentTypes()[1])
    return emitOpError() << "expects atomic reduction region with two "
                            "arguments of the same type";
  auto ptrType = atomicReductionEntryBlock.getArgumentTypes()[0]
                     .dyn_cast<PointerLikeType>();
  if (!ptrType || ptrType.getElementType() != type())
    return emitOpError() << "expects atomic reduction region arguments to "
                            "be accumulators containing the reduction type";
  return success();
}

LogicalResult ReductionOp::verify() {
  auto *op = (*this)->getParentWithTrait<ReductionClauseInterface::Trait>();
  if (!op)
    return emitOpError() << "must be used within an operation supporting "
                            "reduction clause interface";
  while (op) {
    for (const auto &var :
         cast<ReductionClauseInterface>(op).getReductionVars())
      if (var == accumulator())
        return success();
    op = op->getParentWithTrait<ReductionClauseInterface::Trait>();
  }
  return emitOpError() << "the accumulator is not used by the parent";
}

//===----------------------------------------------------------------------===//
// TaskOp
//===----------------------------------------------------------------------===//
LogicalResult TaskOp::verify() {
  return verifyReductionVarList(*this, in_reductions(), in_reduction_vars());
}

//===----------------------------------------------------------------------===//
// TaskGroupOp
//===----------------------------------------------------------------------===//
LogicalResult TaskGroupOp::verify() {
  return verifyReductionVarList(*this, task_reductions(),
                                task_reduction_vars());
}

//===----------------------------------------------------------------------===//
// TaskLoopOp
//===----------------------------------------------------------------------===//
SmallVector<Value> TaskLoopOp::getReductionVars() {
  SmallVector<Value> all_reduction_nvars(in_reduction_vars().begin(),
                                         in_reduction_vars().end());
  all_reduction_nvars.insert(all_reduction_nvars.end(),
                             reduction_vars().begin(), reduction_vars().end());
  return all_reduction_nvars;
}

LogicalResult TaskLoopOp::verify() {
  if (allocate_vars().size() != allocators_vars().size())
    return emitError(
        "expected equal sizes for allocate and allocator variables");
  if (failed(verifyReductionVarList(*this, reductions(), reduction_vars())) ||
      failed(
          verifyReductionVarList(*this, in_reductions(), in_reduction_vars())))
    return failure();

  if (reduction_vars().size() > 0 && nogroup())
    return emitError("if a reduction clause is present on the taskloop "
                     "directive, the nogroup clause must not be specified");
  for (auto var : reduction_vars()) {
    if (llvm::is_contained(in_reduction_vars(), var))
      return emitError("the same list item cannot appear in both a reduction "
                       "and an in_reduction clause");
  }

  if (grain_size() && num_tasks()) {
    return emitError(
        "the grainsize clause and num_tasks clause are mutually exclusive and "
        "may not appear on the same taskloop directive");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// WsLoopOp
//===----------------------------------------------------------------------===//

void WsLoopOp::build(OpBuilder &builder, OperationState &state,
                     ValueRange lowerBound, ValueRange upperBound,
                     ValueRange step, ArrayRef<NamedAttribute> attributes) {
  build(builder, state, lowerBound, upperBound, step,
        /*linear_vars=*/ValueRange(),
        /*linear_step_vars=*/ValueRange(), /*reduction_vars=*/ValueRange(),
        /*reductions=*/nullptr, /*schedule_val=*/nullptr,
        /*schedule_chunk_var=*/nullptr, /*schedule_modifier=*/nullptr,
        /*simd_modifier=*/false, /*nowait=*/false, /*ordered_val=*/nullptr,
        /*order_val=*/nullptr, /*inclusive=*/false);
  state.addAttributes(attributes);
}

LogicalResult WsLoopOp::verify() {
  return verifyReductionVarList(*this, reductions(), reduction_vars());
}

//===----------------------------------------------------------------------===//
// Verifier for critical construct (2.17.1)
//===----------------------------------------------------------------------===//

LogicalResult CriticalDeclareOp::verify() {
  return verifySynchronizationHint(*this, hint_val());
}

LogicalResult CriticalOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (nameAttr()) {
    SymbolRefAttr symbolRef = nameAttr();
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
// Verifier for ordered construct
//===----------------------------------------------------------------------===//

LogicalResult OrderedOp::verify() {
  auto container = (*this)->getParentOfType<WsLoopOp>();
  if (!container || !container.ordered_valAttr() ||
      container.ordered_valAttr().getInt() == 0)
    return emitOpError() << "ordered depend directive must be closely "
                         << "nested inside a worksharing-loop with ordered "
                         << "clause with parameter present";

  if (container.ordered_valAttr().getInt() != (int64_t)*num_loops_val())
    return emitOpError() << "number of variables in depend clause does not "
                         << "match number of iteration variables in the "
                         << "doacross loop";

  return success();
}

LogicalResult OrderedRegionOp::verify() {
  // TODO: The code generation for ordered simd directive is not supported yet.
  if (simd())
    return failure();

  if (auto container = (*this)->getParentOfType<WsLoopOp>()) {
    if (!container.ordered_valAttr() ||
        container.ordered_valAttr().getInt() != 0)
      return emitOpError() << "ordered region must be closely nested inside "
                           << "a worksharing-loop region with an ordered "
                           << "clause without parameter present";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Verifier for AtomicReadOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicReadOp::verify() {
  if (auto mo = memory_order_val()) {
    if (*mo == ClauseMemoryOrderKind::Acq_rel ||
        *mo == ClauseMemoryOrderKind::Release) {
      return emitError(
          "memory-order must not be acq_rel or release for atomic reads");
    }
  }
  if (x() == v())
    return emitError(
        "read and write must not be to the same location for atomic reads");
  return verifySynchronizationHint(*this, hint_val());
}

//===----------------------------------------------------------------------===//
// Verifier for AtomicWriteOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicWriteOp::verify() {
  if (auto mo = memory_order_val()) {
    if (*mo == ClauseMemoryOrderKind::Acq_rel ||
        *mo == ClauseMemoryOrderKind::Acquire) {
      return emitError(
          "memory-order must not be acq_rel or acquire for atomic writes");
    }
  }
  if (address().getType().cast<PointerLikeType>().getElementType() !=
      value().getType())
    return emitError("address must dereference to value type");
  return verifySynchronizationHint(*this, hint_val());
}

//===----------------------------------------------------------------------===//
// Verifier for AtomicUpdateOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicUpdateOp::verify() {
  if (auto mo = memory_order_val()) {
    if (*mo == ClauseMemoryOrderKind::Acq_rel ||
        *mo == ClauseMemoryOrderKind::Acquire) {
      return emitError(
          "memory-order must not be acq_rel or acquire for atomic updates");
    }
  }

  if (x().getType().cast<PointerLikeType>().getElementType() !=
      region().getArgument(0).getType()) {
    return emitError("the type of the operand must be a pointer type whose "
                     "element type is the same as that of the region argument");
  }

  return verifySynchronizationHint(*this, hint_val());
}

LogicalResult AtomicUpdateOp::verifyRegions() {
  if (region().getNumArguments() != 1)
    return emitError("the region must accept exactly one argument");

  if (region().front().getOperations().size() < 2)
    return emitError() << "the update region must have at least two operations "
                          "(binop and terminator)";

  YieldOp yieldOp = *region().getOps<YieldOp>().begin();

  if (yieldOp.results().size() != 1)
    return emitError("only updated value must be returned");
  if (yieldOp.results().front().getType() != region().getArgument(0).getType())
    return emitError("input and yielded value must have the same type");
  return success();
}

//===----------------------------------------------------------------------===//
// Verifier for AtomicCaptureOp
//===----------------------------------------------------------------------===//

Operation *AtomicCaptureOp::getFirstOp() {
  return &getRegion().front().getOperations().front();
}

Operation *AtomicCaptureOp::getSecondOp() {
  auto &ops = getRegion().front().getOperations();
  return ops.getNextNode(ops.front());
}

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
  return verifySynchronizationHint(*this, hint_val());
}

LogicalResult AtomicCaptureOp::verifyRegions() {
  Block::OpListType &ops = region().front().getOperations();
  if (ops.size() != 3)
    return emitError()
           << "expected three operations in omp.atomic.capture region (one "
              "terminator, and two atomic ops)";
  auto &firstOp = ops.front();
  auto &secondOp = *ops.getNextNode(firstOp);
  auto firstReadStmt = dyn_cast<AtomicReadOp>(firstOp);
  auto firstUpdateStmt = dyn_cast<AtomicUpdateOp>(firstOp);
  auto secondReadStmt = dyn_cast<AtomicReadOp>(secondOp);
  auto secondUpdateStmt = dyn_cast<AtomicUpdateOp>(secondOp);
  auto secondWriteStmt = dyn_cast<AtomicWriteOp>(secondOp);

  if (!((firstUpdateStmt && secondReadStmt) ||
        (firstReadStmt && secondUpdateStmt) ||
        (firstReadStmt && secondWriteStmt)))
    return ops.front().emitError()
           << "invalid sequence of operations in the capture region";
  if (firstUpdateStmt && secondReadStmt &&
      firstUpdateStmt.x() != secondReadStmt.x())
    return firstUpdateStmt.emitError()
           << "updated variable in omp.atomic.update must be captured in "
              "second operation";
  if (firstReadStmt && secondUpdateStmt &&
      firstReadStmt.x() != secondUpdateStmt.x())
    return firstReadStmt.emitError()
           << "captured variable in omp.atomic.read must be updated in second "
              "operation";
  if (firstReadStmt && secondWriteStmt &&
      firstReadStmt.x() != secondWriteStmt.address())
    return firstReadStmt.emitError()
           << "captured variable in omp.atomic.read must be updated in "
              "second operation";

  if (getFirstOp()->getAttr("hint_val") || getSecondOp()->getAttr("hint_val"))
    return emitOpError(
        "operations inside capture region must not have hint clause");

  if (getFirstOp()->getAttr("memory_order_val") ||
      getSecondOp()->getAttr("memory_order_val"))
    return emitOpError(
        "operations inside capture region must not have memory_order clause");
  return success();
}

//===----------------------------------------------------------------------===//
// Verifier for CancelOp
//===----------------------------------------------------------------------===//

LogicalResult CancelOp::verify() {
  ClauseCancellationConstructType cct = cancellation_construct_type_val();
  Operation *parentOp = (*this)->getParentOp();

  if (!parentOp) {
    return emitOpError() << "must be used within a region supporting "
                            "cancel directive";
  }

  if ((cct == ClauseCancellationConstructType::Parallel) &&
      !isa<ParallelOp>(parentOp)) {
    return emitOpError() << "cancel parallel must appear "
                         << "inside a parallel region";
  }
  if (cct == ClauseCancellationConstructType::Loop) {
    if (!isa<WsLoopOp>(parentOp)) {
      return emitOpError() << "cancel loop must appear "
                           << "inside a worksharing-loop region";
    }
    if (cast<WsLoopOp>(parentOp).nowaitAttr()) {
      return emitError() << "A worksharing construct that is canceled "
                         << "must not have a nowait clause";
    }
    if (cast<WsLoopOp>(parentOp).ordered_valAttr()) {
      return emitError() << "A worksharing construct that is canceled "
                         << "must not have an ordered clause";
    }

  } else if (cct == ClauseCancellationConstructType::Sections) {
    if (!(isa<SectionsOp>(parentOp) || isa<SectionOp>(parentOp))) {
      return emitOpError() << "cancel sections must appear "
                           << "inside a sections region";
    }
    if (isa_and_nonnull<SectionsOp>(parentOp->getParentOp()) &&
        cast<SectionsOp>(parentOp->getParentOp()).nowaitAttr()) {
      return emitError() << "A sections construct that is canceled "
                         << "must not have a nowait clause";
    }
  }
  // TODO : Add more when we support taskgroup.
  return success();
}
//===----------------------------------------------------------------------===//
// Verifier for CancelOp
//===----------------------------------------------------------------------===//

LogicalResult CancellationPointOp::verify() {
  ClauseCancellationConstructType cct = cancellation_construct_type_val();
  Operation *parentOp = (*this)->getParentOp();

  if (!parentOp) {
    return emitOpError() << "must be used within a region supporting "
                            "cancellation point directive";
  }

  if ((cct == ClauseCancellationConstructType::Parallel) &&
      !(isa<ParallelOp>(parentOp))) {
    return emitOpError() << "cancellation point parallel must appear "
                         << "inside a parallel region";
  }
  if ((cct == ClauseCancellationConstructType::Loop) &&
      !isa<WsLoopOp>(parentOp)) {
    return emitOpError() << "cancellation point loop must appear "
                         << "inside a worksharing-loop region";
  }
  if ((cct == ClauseCancellationConstructType::Sections) &&
      !(isa<SectionsOp>(parentOp) || isa<SectionOp>(parentOp))) {
    return emitOpError() << "cancellation point sections must appear "
                         << "inside a sections region";
  }
  // TODO : Add more when we support taskgroup.
  return success();
}

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOpsAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOps.cpp.inc"
