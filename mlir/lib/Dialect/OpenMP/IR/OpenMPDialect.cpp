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
// ParallelOp
//===----------------------------------------------------------------------===//

void ParallelOp::build(OpBuilder &builder, OperationState &state,
                       ArrayRef<NamedAttribute> attributes) {
  ParallelOp::build(
      builder, state, /*if_expr_var=*/nullptr, /*num_threads_var=*/nullptr,
      /*allocate_vars=*/ValueRange(), /*allocators_vars=*/ValueRange(),
      /*proc_bind_val=*/nullptr);
  state.addAttributes(attributes);
}

//===----------------------------------------------------------------------===//
// Parser and printer for Allocate Clause
//===----------------------------------------------------------------------===//

/// Parse an allocate clause with allocators and a list of operands with types.
///
/// allocate ::= `allocate` `(` allocate-operand-list `)`
/// allocate-operand-list :: = allocate-operand |
///                            allocator-operand `,` allocate-operand-list
/// allocate-operand :: = ssa-id-and-type -> ssa-id-and-type
/// ssa-id-and-type ::= ssa-id `:` type
static ParseResult parseAllocateAndAllocator(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::OperandType> &operandsAllocate,
    SmallVectorImpl<Type> &typesAllocate,
    SmallVectorImpl<OpAsmParser::OperandType> &operandsAllocator,
    SmallVectorImpl<Type> &typesAllocator) {

  return parser.parseCommaSeparatedList([&]() -> ParseResult {
    OpAsmParser::OperandType operand;
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

ParseResult parseProcBindKind(OpAsmParser &parser,
                              omp::ClauseProcBindKindAttr &procBindAttr) {
  StringRef procBindStr;
  if (parser.parseKeyword(&procBindStr))
    return failure();
  if (auto procBindVal = symbolizeClauseProcBindKind(procBindStr)) {
    procBindAttr =
        ClauseProcBindKindAttr::get(parser.getContext(), *procBindVal);
    return success();
  }
  return failure();
}

void printProcBindKind(OpAsmPrinter &p, Operation *op,
                       omp::ClauseProcBindKindAttr procBindAttr) {
  p << stringifyClauseProcBindKind(procBindAttr.getValue());
}

LogicalResult ParallelOp::verify() {
  if (allocate_vars().size() != allocators_vars().size())
    return emitError(
        "expected equal sizes for allocate and allocator variables");
  return success();
}

//===----------------------------------------------------------------------===//
// Parser and printer for Linear Clause
//===----------------------------------------------------------------------===//

/// linear ::= `linear` `(` linear-list `)`
/// linear-list := linear-val | linear-val linear-list
/// linear-val := ssa-id-and-type `=` ssa-id-and-type
static ParseResult
parseLinearClause(OpAsmParser &parser,
                  SmallVectorImpl<OpAsmParser::OperandType> &vars,
                  SmallVectorImpl<Type> &types,
                  SmallVectorImpl<OpAsmParser::OperandType> &stepVars) {
  if (parser.parseLParen())
    return failure();

  do {
    OpAsmParser::OperandType var;
    Type type;
    OpAsmParser::OperandType stepVar;
    if (parser.parseOperand(var) || parser.parseEqual() ||
        parser.parseOperand(stepVar) || parser.parseColonType(type))
      return failure();

    vars.push_back(var);
    types.push_back(type);
    stepVars.push_back(stepVar);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRParen())
    return failure();

  return success();
}

/// Print Linear Clause
static void printLinearClause(OpAsmPrinter &p, OperandRange linearVars,
                              OperandRange linearStepVars) {
  size_t linearVarsSize = linearVars.size();
  p << "linear(";
  for (unsigned i = 0; i < linearVarsSize; ++i) {
    std::string separator = i == linearVarsSize - 1 ? ") " : ", ";
    p << linearVars[i];
    if (linearStepVars.size() > i)
      p << " = " << linearStepVars[i];
    p << " : " << linearVars[i].getType() << separator;
  }
}

//===----------------------------------------------------------------------===//
// Parser and printer for Schedule Clause
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
    if (!symbol.hasValue())
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
parseScheduleClause(OpAsmParser &parser, SmallString<8> &schedule,
                    SmallVectorImpl<SmallString<12>> &modifiers,
                    Optional<OpAsmParser::OperandType> &chunkSize,
                    Type &chunkType) {
  if (parser.parseLParen())
    return failure();

  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return failure();

  schedule = keyword;
  if (keyword == "static" || keyword == "dynamic" || keyword == "guided") {
    if (succeeded(parser.parseOptionalEqual())) {
      chunkSize = OpAsmParser::OperandType{};
      if (parser.parseOperand(*chunkSize) || parser.parseColonType(chunkType))
        return failure();
    } else {
      chunkSize = llvm::NoneType::None;
    }
  } else if (keyword == "auto" || keyword == "runtime") {
    chunkSize = llvm::NoneType::None;
  } else {
    return parser.emitError(parser.getNameLoc()) << " expected schedule kind";
  }

  // If there is a comma, we have one or more modifiers..
  while (succeeded(parser.parseOptionalComma())) {
    StringRef mod;
    if (parser.parseKeyword(&mod))
      return failure();
    modifiers.push_back(mod);
  }

  if (parser.parseRParen())
    return failure();

  if (verifyScheduleModifiers(parser, modifiers))
    return failure();

  return success();
}

/// Print schedule clause
static void printScheduleClause(OpAsmPrinter &p, ClauseScheduleKind sched,
                                Optional<ScheduleModifier> modifier, bool simd,
                                Value scheduleChunkVar) {
  p << "schedule(" << stringifyClauseScheduleKind(sched).lower();
  if (scheduleChunkVar)
    p << " = " << scheduleChunkVar << " : " << scheduleChunkVar.getType();
  if (modifier)
    p << ", " << stringifyScheduleModifier(*modifier);
  if (simd)
    p << ", simd";
  p << ") ";
}

//===----------------------------------------------------------------------===//
// Parser, printer and verifier for ReductionVarList
//===----------------------------------------------------------------------===//

/// reduction ::= `reduction` `(` reduction-entry-list `)`
/// reduction-entry-list ::= reduction-entry
///                        | reduction-entry-list `,` reduction-entry
/// reduction-entry ::= symbol-ref `->` ssa-id `:` type
static ParseResult
parseReductionVarList(OpAsmParser &parser,
                      SmallVectorImpl<SymbolRefAttr> &symbols,
                      SmallVectorImpl<OpAsmParser::OperandType> &operands,
                      SmallVectorImpl<Type> &types) {
  if (failed(parser.parseLParen()))
    return failure();

  do {
    if (parser.parseAttribute(symbols.emplace_back()) || parser.parseArrow() ||
        parser.parseOperand(operands.emplace_back()) ||
        parser.parseColonType(types.emplace_back()))
      return failure();
  } while (succeeded(parser.parseOptionalComma()));
  return parser.parseRParen();
}

/// Print Reduction clause
static void printReductionVarList(OpAsmPrinter &p,
                                  Optional<ArrayAttr> reductions,
                                  OperandRange reductionVars) {
  p << "reduction(";
  for (unsigned i = 0, e = reductions->size(); i < e; ++i) {
    if (i != 0)
      p << ", ";
    p << (*reductions)[i] << " -> " << reductionVars[i] << " : "
      << reductionVars[i].getType();
  }
  p << ") ";
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
                                            IntegerAttr &hintAttr,
                                            bool parseKeyword = true) {
  if (parseKeyword && failed(parser.parseOptionalKeyword("hint"))) {
    hintAttr = IntegerAttr::get(parser.getBuilder().getI64Type(), 0);
    return success();
  }

  if (failed(parser.parseLParen()))
    return failure();
  StringRef hintKeyword;
  int64_t hint = 0;
  do {
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
  } while (succeeded(parser.parseOptionalComma()));
  if (failed(parser.parseRParen()))
    return failure();
  hintAttr = IntegerAttr::get(parser.getBuilder().getI64Type(), hint);
  return success();
}

/// Prints a Synchronization Hint clause
static void printSynchronizationHint(OpAsmPrinter &p, Operation *op,
                                     IntegerAttr hintAttr) {
  int64_t hint = hintAttr.getInt();

  if (hint == 0)
    return;

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

  p << "hint(";
  llvm::interleaveComma(hints, p);
  p << ") ";
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

enum ClauseType {
  ifClause,
  numThreadsClause,
  deviceClause,
  threadLimitClause,
  allocateClause,
  procBindClause,
  reductionClause,
  nowaitClause,
  linearClause,
  scheduleClause,
  collapseClause,
  orderClause,
  orderedClause,
  memoryOrderClause,
  hintClause,
  COUNT
};

//===----------------------------------------------------------------------===//
// Parser for Clause List
//===----------------------------------------------------------------------===//

/// Parse a clause attribute `(` $value `)`.
template <typename ClauseAttr>
static ParseResult parseClauseAttr(AsmParser &parser, OperationState &state,
                                   StringRef attrName, StringRef name) {
  using ClauseT = decltype(std::declval<ClauseAttr>().getValue());
  StringRef enumStr;
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseLParen() || parser.parseKeyword(&enumStr) ||
      parser.parseRParen())
    return failure();
  if (Optional<ClauseT> enumValue = symbolizeEnum<ClauseT>(enumStr)) {
    auto attr = ClauseAttr::get(parser.getContext(), *enumValue);
    state.addAttribute(attrName, attr);
    return success();
  }
  return parser.emitError(loc, "invalid ") << name << " kind";
}

/// Parse a list of clauses. The clauses can appear in any order, but their
/// operand segment indices are in the same order that they are passed in the
/// `clauses` list. The operand segments are added over the prevSegments

/// clause-list ::= clause clause-list | empty
/// clause ::= if | num-threads | allocate | proc-bind | reduction | nowait
///          | linear | schedule | collapse | order | ordered | inclusive
/// if ::= `if` `(` ssa-id-and-type `)`
/// num-threads ::= `num_threads` `(` ssa-id-and-type `)`
/// allocate ::= `allocate` `(` allocate-operand-list `)`
/// proc-bind ::= `proc_bind` `(` (`master` | `close` | `spread`) `)`
/// reduction ::= `reduction` `(` reduction-entry-list `)`
/// nowait ::= `nowait`
/// linear ::= `linear` `(` linear-list `)`
/// schedule ::= `schedule` `(` sched-list `)`
/// collapse ::= `collapse` `(` ssa-id-and-type `)`
/// order ::= `order` `(` `concurrent` `)`
/// ordered ::= `ordered` `(` ssa-id-and-type `)`
/// inclusive ::= `inclusive`
///
/// Note that each clause can only appear once in the clase-list.
static ParseResult parseClauses(OpAsmParser &parser, OperationState &result,
                                SmallVectorImpl<ClauseType> &clauses,
                                SmallVectorImpl<int> &segments) {

  // Check done[clause] to see if it has been parsed already
  BitVector done(ClauseType::COUNT, false);

  // See pos[clause] to get position of clause in operand segments
  SmallVector<int> pos(ClauseType::COUNT, -1);

  // Stores the last parsed clause keyword
  StringRef clauseKeyword;
  StringRef opName = result.name.getStringRef();

  // Containers for storing operands, types and attributes for various clauses
  std::pair<OpAsmParser::OperandType, Type> ifCond;
  std::pair<OpAsmParser::OperandType, Type> numThreads;
  std::pair<OpAsmParser::OperandType, Type> device;
  std::pair<OpAsmParser::OperandType, Type> threadLimit;

  SmallVector<OpAsmParser::OperandType> allocates, allocators;
  SmallVector<Type> allocateTypes, allocatorTypes;

  SmallVector<SymbolRefAttr> reductionSymbols;
  SmallVector<OpAsmParser::OperandType> reductionVars;
  SmallVector<Type> reductionVarTypes;

  SmallVector<OpAsmParser::OperandType> linears;
  SmallVector<Type> linearTypes;
  SmallVector<OpAsmParser::OperandType> linearSteps;

  SmallString<8> schedule;
  SmallVector<SmallString<12>> modifiers;
  Optional<OpAsmParser::OperandType> scheduleChunkSize;
  Type scheduleChunkType;

  // Compute the position of clauses in operand segments
  int currPos = 0;
  for (ClauseType clause : clauses) {

    // Skip the following clauses - they do not take any position in operand
    // segments
    if (clause == procBindClause || clause == nowaitClause ||
        clause == collapseClause || clause == orderClause ||
        clause == orderedClause)
      continue;

    pos[clause] = currPos++;

    // For the following clauses, two positions are reserved in the operand
    // segments
    if (clause == allocateClause || clause == linearClause)
      currPos++;
  }

  SmallVector<int> clauseSegments(currPos);

  // Helper function to check if a clause is allowed/repeated or not
  auto checkAllowed = [&](ClauseType clause) -> ParseResult {
    if (!llvm::is_contained(clauses, clause))
      return parser.emitError(parser.getCurrentLocation())
             << clauseKeyword << " is not a valid clause for the " << opName
             << " operation";
    if (done[clause])
      return parser.emitError(parser.getCurrentLocation())
             << "at most one " << clauseKeyword << " clause can appear on the "
             << opName << " operation";
    done[clause] = true;
    return success();
  };

  while (succeeded(parser.parseOptionalKeyword(&clauseKeyword))) {
    if (clauseKeyword == "if") {
      if (checkAllowed(ifClause) || parser.parseLParen() ||
          parser.parseOperand(ifCond.first) ||
          parser.parseColonType(ifCond.second) || parser.parseRParen())
        return failure();
      clauseSegments[pos[ifClause]] = 1;
    } else if (clauseKeyword == "num_threads") {
      if (checkAllowed(numThreadsClause) || parser.parseLParen() ||
          parser.parseOperand(numThreads.first) ||
          parser.parseColonType(numThreads.second) || parser.parseRParen())
        return failure();
      clauseSegments[pos[numThreadsClause]] = 1;
    } else if (clauseKeyword == "device") {
      if (checkAllowed(deviceClause) || parser.parseLParen() ||
          parser.parseOperand(device.first) ||
          parser.parseColonType(device.second) || parser.parseRParen())
        return failure();
      clauseSegments[pos[deviceClause]] = 1;
    } else if (clauseKeyword == "thread_limit") {
      if (checkAllowed(threadLimitClause) || parser.parseLParen() ||
          parser.parseOperand(threadLimit.first) ||
          parser.parseColonType(threadLimit.second) || parser.parseRParen())
        return failure();
      clauseSegments[pos[threadLimitClause]] = 1;
    } else if (clauseKeyword == "allocate") {
      if (checkAllowed(allocateClause) || parser.parseLParen() ||
          parseAllocateAndAllocator(parser, allocates, allocateTypes,
                                    allocators, allocatorTypes) ||
          parser.parseRParen())
        return failure();
      clauseSegments[pos[allocateClause]] = allocates.size();
      clauseSegments[pos[allocateClause] + 1] = allocators.size();
    } else if (clauseKeyword == "proc_bind") {
      if (checkAllowed(procBindClause) ||
          parseClauseAttr<ClauseProcBindKindAttr>(parser, result,
                                                  "proc_bind_val", "proc bind"))
        return failure();
    } else if (clauseKeyword == "reduction") {
      if (checkAllowed(reductionClause) ||
          parseReductionVarList(parser, reductionSymbols, reductionVars,
                                reductionVarTypes))
        return failure();
      clauseSegments[pos[reductionClause]] = reductionVars.size();
    } else if (clauseKeyword == "nowait") {
      if (checkAllowed(nowaitClause))
        return failure();
      auto attr = UnitAttr::get(parser.getBuilder().getContext());
      result.addAttribute("nowait", attr);
    } else if (clauseKeyword == "linear") {
      if (checkAllowed(linearClause) ||
          parseLinearClause(parser, linears, linearTypes, linearSteps))
        return failure();
      clauseSegments[pos[linearClause]] = linears.size();
      clauseSegments[pos[linearClause] + 1] = linearSteps.size();
    } else if (clauseKeyword == "schedule") {
      if (checkAllowed(scheduleClause) ||
          parseScheduleClause(parser, schedule, modifiers, scheduleChunkSize,
                              scheduleChunkType))
        return failure();
      if (scheduleChunkSize) {
        clauseSegments[pos[scheduleClause]] = 1;
      }
    } else if (clauseKeyword == "collapse") {
      auto type = parser.getBuilder().getI64Type();
      mlir::IntegerAttr attr;
      if (checkAllowed(collapseClause) || parser.parseLParen() ||
          parser.parseAttribute(attr, type) || parser.parseRParen())
        return failure();
      result.addAttribute("collapse_val", attr);
    } else if (clauseKeyword == "ordered") {
      mlir::IntegerAttr attr;
      if (checkAllowed(orderedClause))
        return failure();
      if (succeeded(parser.parseOptionalLParen())) {
        auto type = parser.getBuilder().getI64Type();
        if (parser.parseAttribute(attr, type) || parser.parseRParen())
          return failure();
      } else {
        // Use 0 to represent no ordered parameter was specified
        attr = parser.getBuilder().getI64IntegerAttr(0);
      }
      result.addAttribute("ordered_val", attr);
    } else if (clauseKeyword == "order") {
      if (checkAllowed(orderClause) ||
          parseClauseAttr<ClauseOrderKindAttr>(parser, result, "order_val",
                                               "order"))
        return failure();
    } else if (clauseKeyword == "memory_order") {
      if (checkAllowed(memoryOrderClause) ||
          parseClauseAttr<ClauseMemoryOrderKindAttr>(
              parser, result, "memory_order", "memory order"))
        return failure();
    } else if (clauseKeyword == "hint") {
      IntegerAttr hint;
      if (checkAllowed(hintClause) ||
          parseSynchronizationHint(parser, hint, false))
        return failure();
      result.addAttribute("hint", hint);
    } else {
      return parser.emitError(parser.getNameLoc())
             << clauseKeyword << " is not a valid clause";
    }
  }

  // Add if parameter.
  if (done[ifClause] && clauseSegments[pos[ifClause]] &&
      failed(
          parser.resolveOperand(ifCond.first, ifCond.second, result.operands)))
    return failure();

  // Add num_threads parameter.
  if (done[numThreadsClause] && clauseSegments[pos[numThreadsClause]] &&
      failed(parser.resolveOperand(numThreads.first, numThreads.second,
                                   result.operands)))
    return failure();

  // Add device parameter.
  if (done[deviceClause] && clauseSegments[pos[deviceClause]] &&
      failed(
          parser.resolveOperand(device.first, device.second, result.operands)))
    return failure();

  // Add thread_limit parameter.
  if (done[threadLimitClause] && clauseSegments[pos[threadLimitClause]] &&
      failed(parser.resolveOperand(threadLimit.first, threadLimit.second,
                                   result.operands)))
    return failure();

  // Add allocate parameters.
  if (done[allocateClause] && clauseSegments[pos[allocateClause]] &&
      failed(parser.resolveOperands(allocates, allocateTypes,
                                    allocates[0].location, result.operands)))
    return failure();

  // Add allocator parameters.
  if (done[allocateClause] && clauseSegments[pos[allocateClause] + 1] &&
      failed(parser.resolveOperands(allocators, allocatorTypes,
                                    allocators[0].location, result.operands)))
    return failure();

  // Add reduction parameters and symbols
  if (done[reductionClause] && clauseSegments[pos[reductionClause]]) {
    if (failed(parser.resolveOperands(reductionVars, reductionVarTypes,
                                      parser.getNameLoc(), result.operands)))
      return failure();

    SmallVector<Attribute> reductions(reductionSymbols.begin(),
                                      reductionSymbols.end());
    result.addAttribute("reductions",
                        parser.getBuilder().getArrayAttr(reductions));
  }

  // Add linear parameters
  if (done[linearClause] && clauseSegments[pos[linearClause]]) {
    auto linearStepType = parser.getBuilder().getI32Type();
    SmallVector<Type> linearStepTypes(linearSteps.size(), linearStepType);
    if (failed(parser.resolveOperands(linears, linearTypes, linears[0].location,
                                      result.operands)) ||
        failed(parser.resolveOperands(linearSteps, linearStepTypes,
                                      linearSteps[0].location,
                                      result.operands)))
      return failure();
  }

  // Add schedule parameters
  if (done[scheduleClause] && !schedule.empty()) {
    schedule[0] = llvm::toUpper(schedule[0]);
    if (Optional<ClauseScheduleKind> sched =
            symbolizeClauseScheduleKind(schedule)) {
      auto attr = ClauseScheduleKindAttr::get(parser.getContext(), *sched);
      result.addAttribute("schedule_val", attr);
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "invalid schedule kind");
    }
    if (!modifiers.empty()) {
      SMLoc loc = parser.getCurrentLocation();
      if (Optional<ScheduleModifier> mod =
              symbolizeScheduleModifier(modifiers[0])) {
        result.addAttribute(
            "schedule_modifier",
            ScheduleModifierAttr::get(parser.getContext(), *mod));
      } else {
        return parser.emitError(loc, "invalid schedule modifier");
      }
      // Only SIMD attribute is allowed here!
      if (modifiers.size() > 1) {
        assert(symbolizeScheduleModifier(modifiers[1]) ==
               ScheduleModifier::simd);
        auto attr = UnitAttr::get(parser.getBuilder().getContext());
        result.addAttribute("simd_modifier", attr);
      }
    }
    if (scheduleChunkSize)
      parser.resolveOperand(*scheduleChunkSize, scheduleChunkType,
                            result.operands);
  }

  segments.insert(segments.end(), clauseSegments.begin(), clauseSegments.end());

  return success();
}

//===----------------------------------------------------------------------===//
// Parser, printer and verifier for SectionsOp
//===----------------------------------------------------------------------===//

/// Parses an OpenMP Sections operation
///
/// sections ::= `omp.sections` clause-list
/// clause-list ::= clause clause-list | empty
/// clause ::= reduction | allocate | nowait
ParseResult SectionsOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<ClauseType> clauses = {reductionClause, allocateClause,
                                     nowaitClause};

  SmallVector<int> segments;

  if (failed(parseClauses(parser, result, clauses, segments)))
    return failure();

  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getI32VectorAttr(segments));

  // Now parse the body.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body))
    return failure();
  return success();
}

void SectionsOp::print(OpAsmPrinter &p) {
  p << " ";

  if (!reduction_vars().empty())
    printReductionVarList(p, reductions(), reduction_vars());

  if (!allocate_vars().empty()) {
    printAllocateAndAllocator(p << "allocate(", *this, allocate_vars(),
                              allocate_vars().getTypes(), allocators_vars(),
                              allocators_vars().getTypes());
    p << ")";
  }

  if (nowait())
    p << "nowait";

  p << ' ';
  p.printRegion(region());
}

LogicalResult SectionsOp::verify() {
  if (allocate_vars().size() != allocators_vars().size())
    return emitError(
        "expected equal sizes for allocate and allocator variables");

  for (auto &inst : *region().begin()) {
    if (!(isa<SectionOp>(inst) || isa<TerminatorOp>(inst))) {
      return emitOpError()
             << "expected omp.section op or terminator op inside region";
    }
  }

  return verifyReductionVarList(*this, reductions(), reduction_vars());
}

/// Parses an OpenMP Workshare Loop operation
///
/// wsloop ::= `omp.wsloop` loop-control clause-list
/// loop-control ::= `(` ssa-id-list `)` `:` type `=`  loop-bounds
/// loop-bounds := `(` ssa-id-list `)` to `(` ssa-id-list `)` inclusive? steps
/// steps := `step` `(`ssa-id-list`)`
/// clause-list ::= clause clause-list | empty
/// clause ::= linear | schedule | collapse | nowait | ordered | order
///          | reduction
ParseResult WsLoopOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse an opening `(` followed by induction variables followed by `)`
  SmallVector<OpAsmParser::OperandType> ivs;
  if (parser.parseRegionArgumentList(ivs, /*requiredOperandCount=*/-1,
                                     OpAsmParser::Delimiter::Paren))
    return failure();

  int numIVs = static_cast<int>(ivs.size());
  Type loopVarType;
  if (parser.parseColonType(loopVarType))
    return failure();

  // Parse loop bounds.
  SmallVector<OpAsmParser::OperandType> lower;
  if (parser.parseEqual() ||
      parser.parseOperandList(lower, numIVs, OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(lower, loopVarType, result.operands))
    return failure();

  SmallVector<OpAsmParser::OperandType> upper;
  if (parser.parseKeyword("to") ||
      parser.parseOperandList(upper, numIVs, OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(upper, loopVarType, result.operands))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("inclusive"))) {
    auto attr = UnitAttr::get(parser.getBuilder().getContext());
    result.addAttribute("inclusive", attr);
  }

  // Parse step values.
  SmallVector<OpAsmParser::OperandType> steps;
  if (parser.parseKeyword("step") ||
      parser.parseOperandList(steps, numIVs, OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(steps, loopVarType, result.operands))
    return failure();

  SmallVector<ClauseType> clauses = {
      linearClause,  reductionClause, collapseClause, orderClause,
      orderedClause, nowaitClause,    scheduleClause};
  SmallVector<int> segments{numIVs, numIVs, numIVs};
  if (failed(parseClauses(parser, result, clauses, segments)))
    return failure();

  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getI32VectorAttr(segments));

  // Now parse the body.
  Region *body = result.addRegion();
  SmallVector<Type> ivTypes(numIVs, loopVarType);
  SmallVector<OpAsmParser::OperandType> blockArgs(ivs);
  if (parser.parseRegion(*body, blockArgs, ivTypes))
    return failure();
  return success();
}

void WsLoopOp::print(OpAsmPrinter &p) {
  auto args = getRegion().front().getArguments();
  p << " (" << args << ") : " << args[0].getType() << " = (" << lowerBound()
    << ") to (" << upperBound() << ") ";
  if (inclusive()) {
    p << "inclusive ";
  }
  p << "step (" << step() << ") ";

  if (!linear_vars().empty())
    printLinearClause(p, linear_vars(), linear_step_vars());

  if (auto sched = schedule_val())
    printScheduleClause(p, sched.getValue(), schedule_modifier(),
                        simd_modifier(), schedule_chunk_var());

  if (auto collapse = collapse_val())
    p << "collapse(" << collapse << ") ";

  if (nowait())
    p << "nowait ";

  if (auto ordered = ordered_val())
    p << "ordered(" << ordered << ") ";

  if (auto order = order_val())
    p << "order(" << stringifyClauseOrderKind(*order) << ") ";

  if (!reduction_vars().empty())
    printReductionVarList(p, reductions(), reduction_vars());

  p << ' ';
  p.printRegion(region(), /*printEntryBlockArgs=*/false);
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

LogicalResult ReductionDeclareOp::verify() {
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
  // TODO: generalize this to an op interface when there is more than one op
  // that supports reductions.
  auto container = (*this)->getParentOfType<WsLoopOp>();
  for (unsigned i = 0, e = container.getNumReductionVars(); i < e; ++i)
    if (container.reduction_vars()[i] == accumulator())
      return success();

  return emitOpError() << "the accumulator is not used by the parent";
}

//===----------------------------------------------------------------------===//
// WsLoopOp
//===----------------------------------------------------------------------===//

void WsLoopOp::build(OpBuilder &builder, OperationState &state,
                     ValueRange lowerBound, ValueRange upperBound,
                     ValueRange step, ArrayRef<NamedAttribute> attributes) {
  build(builder, state, TypeRange(), lowerBound, upperBound, step,
        /*linear_vars=*/ValueRange(), /*linear_step_vars=*/ValueRange(),
        /*reduction_vars=*/ValueRange(), /*schedule_val=*/nullptr,
        /*schedule_chunk_var=*/nullptr, /*collapse_val=*/nullptr,
        /*nowait=*/nullptr, /*ordered_val=*/nullptr, /*order_val=*/nullptr,
        /*inclusive=*/nullptr, /*buildBody=*/false);
  state.addAttributes(attributes);
}

void WsLoopOp::build(OpBuilder &, OperationState &state, TypeRange resultTypes,
                     ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  state.addOperands(operands);
  state.addAttributes(attributes);
  (void)state.addRegion();
  assert(resultTypes.empty() && "mismatched number of return types");
  state.addTypes(resultTypes);
}

void WsLoopOp::build(OpBuilder &builder, OperationState &result,
                     TypeRange typeRange, ValueRange lowerBounds,
                     ValueRange upperBounds, ValueRange steps,
                     ValueRange linearVars, ValueRange linearStepVars,
                     ValueRange reductionVars, StringAttr scheduleVal,
                     Value scheduleChunkVar, IntegerAttr collapseVal,
                     UnitAttr nowait, IntegerAttr orderedVal,
                     StringAttr orderVal, UnitAttr inclusive, bool buildBody) {
  result.addOperands(lowerBounds);
  result.addOperands(upperBounds);
  result.addOperands(steps);
  result.addOperands(linearVars);
  result.addOperands(linearStepVars);
  if (scheduleChunkVar)
    result.addOperands(scheduleChunkVar);

  if (scheduleVal)
    result.addAttribute("schedule_val", scheduleVal);
  if (collapseVal)
    result.addAttribute("collapse_val", collapseVal);
  if (nowait)
    result.addAttribute("nowait", nowait);
  if (orderedVal)
    result.addAttribute("ordered_val", orderedVal);
  if (orderVal)
    result.addAttribute("order", orderVal);
  if (inclusive)
    result.addAttribute("inclusive", inclusive);
  result.addAttribute(
      WsLoopOp::getOperandSegmentSizeAttr(),
      builder.getI32VectorAttr(
          {static_cast<int32_t>(lowerBounds.size()),
           static_cast<int32_t>(upperBounds.size()),
           static_cast<int32_t>(steps.size()),
           static_cast<int32_t>(linearVars.size()),
           static_cast<int32_t>(linearStepVars.size()),
           static_cast<int32_t>(reductionVars.size()),
           static_cast<int32_t>(scheduleChunkVar != nullptr ? 1 : 0)}));

  Region *bodyRegion = result.addRegion();
  if (buildBody) {
    OpBuilder::InsertionGuard guard(builder);
    unsigned numIVs = steps.size();
    SmallVector<Type, 8> argTypes(numIVs, steps.getType().front());
    SmallVector<Location, 8> argLocs(numIVs, result.location);
    builder.createBlock(bodyRegion, {}, argTypes, argLocs);
  }
}

LogicalResult WsLoopOp::verify() {
  return verifyReductionVarList(*this, reductions(), reduction_vars());
}

//===----------------------------------------------------------------------===//
// Verifier for critical construct (2.17.1)
//===----------------------------------------------------------------------===//

LogicalResult CriticalDeclareOp::verify() {
  return verifySynchronizationHint(*this, hint());
}

LogicalResult CriticalOp::verify() {
  if (nameAttr()) {
    SymbolRefAttr symbolRef = nameAttr();
    auto decl = SymbolTable::lookupNearestSymbolFrom<CriticalDeclareOp>(
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

  if (container.ordered_valAttr().getInt() !=
      (int64_t)num_loops_val().getValue())
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
// AtomicReadOp
//===----------------------------------------------------------------------===//

/// Parser for AtomicReadOp
///
/// operation ::= `omp.atomic.read` atomic-clause-list address `->` result-type
/// address ::= operand `:` type
ParseResult AtomicReadOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType x, v;
  Type addressType;
  SmallVector<ClauseType> clauses = {memoryOrderClause, hintClause};
  SmallVector<int> segments;

  if (parser.parseOperand(v) || parser.parseEqual() || parser.parseOperand(x) ||
      parseClauses(parser, result, clauses, segments) ||
      parser.parseColonType(addressType) ||
      parser.resolveOperand(x, addressType, result.operands) ||
      parser.resolveOperand(v, addressType, result.operands))
    return failure();

  return success();
}

void AtomicReadOp::print(OpAsmPrinter &p) {
  p << " " << v() << " = " << x() << " ";
  if (auto mo = memory_order())
    p << "memory_order(" << stringifyClauseMemoryOrderKind(*mo) << ") ";
  if (hintAttr())
    printSynchronizationHint(p << " ", *this, hintAttr());
  p << ": " << x().getType();
}

/// Verifier for AtomicReadOp
LogicalResult AtomicReadOp::verify() {
  if (auto mo = memory_order()) {
    if (*mo == ClauseMemoryOrderKind::acq_rel ||
        *mo == ClauseMemoryOrderKind::release) {
      return emitError(
          "memory-order must not be acq_rel or release for atomic reads");
    }
  }
  if (x() == v())
    return emitError(
        "read and write must not be to the same location for atomic reads");
  return verifySynchronizationHint(*this, hint());
}

//===----------------------------------------------------------------------===//
// AtomicWriteOp
//===----------------------------------------------------------------------===//

/// Parser for AtomicWriteOp
///
/// operation ::= `omp.atomic.write` atomic-clause-list operands
/// operands ::= address `,` value
/// address ::= operand `:` type
/// value ::= operand `:` type
ParseResult AtomicWriteOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType address, value;
  Type addrType, valueType;
  SmallVector<ClauseType> clauses = {memoryOrderClause, hintClause};
  SmallVector<int> segments;

  if (parser.parseOperand(address) || parser.parseEqual() ||
      parser.parseOperand(value) ||
      parseClauses(parser, result, clauses, segments) ||
      parser.parseColonType(addrType) || parser.parseComma() ||
      parser.parseType(valueType) ||
      parser.resolveOperand(address, addrType, result.operands) ||
      parser.resolveOperand(value, valueType, result.operands))
    return failure();
  return success();
}

void AtomicWriteOp::print(OpAsmPrinter &p) {
  p << " " << address() << " = " << value() << " ";
  if (auto mo = memory_order())
    p << "memory_order(" << stringifyClauseMemoryOrderKind(*mo) << ") ";
  if (hintAttr())
    printSynchronizationHint(p, *this, hintAttr());
  p << ": " << address().getType() << ", " << value().getType();
}

/// Verifier for AtomicWriteOp
LogicalResult AtomicWriteOp::verify() {
  if (auto mo = memory_order()) {
    if (*mo == ClauseMemoryOrderKind::acq_rel ||
        *mo == ClauseMemoryOrderKind::acquire) {
      return emitError(
          "memory-order must not be acq_rel or acquire for atomic writes");
    }
  }
  return verifySynchronizationHint(*this, hint());
}

//===----------------------------------------------------------------------===//
// AtomicUpdateOp
//===----------------------------------------------------------------------===//

/// Parser for AtomicUpdateOp
///
/// operation ::= `omp.atomic.update` atomic-clause-list ssa-id-and-type region
ParseResult AtomicUpdateOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<ClauseType> clauses = {memoryOrderClause, hintClause};
  SmallVector<int> segments;
  OpAsmParser::OperandType x, expr;
  Type xType;

  if (parseClauses(parser, result, clauses, segments) ||
      parser.parseOperand(x) || parser.parseColon() ||
      parser.parseType(xType) ||
      parser.resolveOperand(x, xType, result.operands) ||
      parser.parseRegion(*result.addRegion()))
    return failure();
  return success();
}

void AtomicUpdateOp::print(OpAsmPrinter &p) {
  p << " ";
  if (auto mo = memory_order())
    p << "memory_order(" << stringifyClauseMemoryOrderKind(*mo) << ") ";
  if (hintAttr())
    printSynchronizationHint(p, *this, hintAttr());
  p << x() << " : " << x().getType();
  p.printRegion(region());
}

/// Verifier for AtomicUpdateOp
LogicalResult AtomicUpdateOp::verify() {
  if (auto mo = memory_order()) {
    if (*mo == ClauseMemoryOrderKind::acq_rel ||
        *mo == ClauseMemoryOrderKind::acquire) {
      return emitError(
          "memory-order must not be acq_rel or acquire for atomic updates");
    }
  }

  if (region().getNumArguments() != 1)
    return emitError("the region must accept exactly one argument");

  if (x().getType().cast<PointerLikeType>().getElementType() !=
      region().getArgument(0).getType()) {
    return emitError("the type of the operand must be a pointer type whose "
                     "element type is the same as that of the region argument");
  }

  YieldOp yieldOp = *region().getOps<YieldOp>().begin();

  if (yieldOp.results().size() != 1)
    return emitError("only updated value must be returned");
  if (yieldOp.results().front().getType() != region().getArgument(0).getType())
    return emitError("input and yielded value must have the same type");
  return success();
}

//===----------------------------------------------------------------------===//
// AtomicCaptureOp
//===----------------------------------------------------------------------===//

ParseResult AtomicCaptureOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  SmallVector<ClauseType> clauses = {memoryOrderClause, hintClause};
  SmallVector<int> segments;
  if (parseClauses(parser, result, clauses, segments) ||
      parser.parseRegion(*result.addRegion()))
    return failure();
  return success();
}

void AtomicCaptureOp::print(OpAsmPrinter &p) {
  if (memory_order())
    p << "memory_order(" << memory_order() << ") ";
  if (hintAttr())
    printSynchronizationHint(p, *this, hintAttr());
  p.printRegion(region());
}

/// Verifier for AtomicCaptureOp
LogicalResult AtomicCaptureOp::verify() {
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
  return success();
}

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOpsAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOps.cpp.inc"
