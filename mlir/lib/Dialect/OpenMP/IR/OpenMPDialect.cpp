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
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
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
} // end namespace

void OpenMPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/OpenMP/OpenMPOps.cpp.inc"
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
      /*default_val=*/nullptr, /*private_vars=*/ValueRange(),
      /*firstprivate_vars=*/ValueRange(), /*shared_vars=*/ValueRange(),
      /*copyin_vars=*/ValueRange(), /*allocate_vars=*/ValueRange(),
      /*allocators_vars=*/ValueRange(), /*proc_bind_val=*/nullptr);
  state.addAttributes(attributes);
}

/// Parse a list of operands with types.
///
/// operand-and-type-list ::= `(` ssa-id-and-type-list `)`
/// ssa-id-and-type-list ::= ssa-id-and-type |
///                          ssa-id-and-type `,` ssa-id-and-type-list
/// ssa-id-and-type ::= ssa-id `:` type
static ParseResult
parseOperandAndTypeList(OpAsmParser &parser,
                        SmallVectorImpl<OpAsmParser::OperandType> &operands,
                        SmallVectorImpl<Type> &types) {
  return parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
        OpAsmParser::OperandType operand;
        Type type;
        if (parser.parseOperand(operand) || parser.parseColonType(type))
          return failure();
        operands.push_back(operand);
        types.push_back(type);
        return success();
      });
}

/// Parse an allocate clause with allocators and a list of operands with types.
///
/// operand-and-type-list ::= `(` allocate-operand-list `)`
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

  return parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
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

static LogicalResult verifyParallelOp(ParallelOp op) {
  if (op.allocate_vars().size() != op.allocators_vars().size())
    return op.emitError(
        "expected equal sizes for allocate and allocator variables");
  return success();
}

static void printParallelOp(OpAsmPrinter &p, ParallelOp op) {
  if (auto ifCond = op.if_expr_var())
    p << " if(" << ifCond << " : " << ifCond.getType() << ")";

  if (auto threads = op.num_threads_var())
    p << " num_threads(" << threads << " : " << threads.getType() << ")";

  // Print private, firstprivate, shared and copyin parameters
  auto printDataVars = [&p](StringRef name, OperandRange vars) {
    if (vars.size()) {
      p << " " << name << "(";
      for (unsigned i = 0; i < vars.size(); ++i) {
        std::string separator = i == vars.size() - 1 ? ")" : ", ";
        p << vars[i] << " : " << vars[i].getType() << separator;
      }
    }
  };

  // Print allocator and allocate parameters
  auto printAllocateAndAllocator = [&p](OperandRange varsAllocate,
                                        OperandRange varsAllocator) {
    if (varsAllocate.empty())
      return;

    p << " allocate(";
    for (unsigned i = 0; i < varsAllocate.size(); ++i) {
      std::string separator = i == varsAllocate.size() - 1 ? ")" : ", ";
      p << varsAllocator[i] << " : " << varsAllocator[i].getType() << " -> ";
      p << varsAllocate[i] << " : " << varsAllocate[i].getType() << separator;
    }
  };

  printDataVars("private", op.private_vars());
  printDataVars("firstprivate", op.firstprivate_vars());
  printDataVars("shared", op.shared_vars());
  printDataVars("copyin", op.copyin_vars());
  printAllocateAndAllocator(op.allocate_vars(), op.allocators_vars());

  if (auto def = op.default_val())
    p << " default(" << def->drop_front(3) << ")";

  if (auto bind = op.proc_bind_val())
    p << " proc_bind(" << bind << ")";

  p.printRegion(op.getRegion());
}

/// Emit an error if the same clause is present more than once on an operation.
static ParseResult allowedOnce(OpAsmParser &parser, StringRef clause,
                               StringRef operation) {
  return parser.emitError(parser.getNameLoc())
         << " at most one " << clause << " clause can appear on the "
         << operation << " operation";
}

/// Parses a parallel operation.
///
/// operation ::= `omp.parallel` clause-list
/// clause-list ::= clause | clause clause-list
/// clause ::= if | numThreads | private | firstprivate | shared | copyin |
///            default | procBind
/// if ::= `if` `(` ssa-id `)`
/// numThreads ::= `num_threads` `(` ssa-id-and-type `)`
/// private ::= `private` operand-and-type-list
/// firstprivate ::= `firstprivate` operand-and-type-list
/// shared ::= `shared` operand-and-type-list
/// copyin ::= `copyin` operand-and-type-list
/// allocate ::= `allocate` operand-and-type `->` operand-and-type-list
/// default ::= `default` `(` (`private` | `firstprivate` | `shared` | `none`)
/// procBind ::= `proc_bind` `(` (`master` | `close` | `spread`) `)`
///
/// Note that each clause can only appear once in the clase-list.
static ParseResult parseParallelOp(OpAsmParser &parser,
                                   OperationState &result) {
  std::pair<OpAsmParser::OperandType, Type> ifCond;
  std::pair<OpAsmParser::OperandType, Type> numThreads;
  SmallVector<OpAsmParser::OperandType, 4> privates;
  SmallVector<Type, 4> privateTypes;
  SmallVector<OpAsmParser::OperandType, 4> firstprivates;
  SmallVector<Type, 4> firstprivateTypes;
  SmallVector<OpAsmParser::OperandType, 4> shareds;
  SmallVector<Type, 4> sharedTypes;
  SmallVector<OpAsmParser::OperandType, 4> copyins;
  SmallVector<Type, 4> copyinTypes;
  SmallVector<OpAsmParser::OperandType, 4> allocates;
  SmallVector<Type, 4> allocateTypes;
  SmallVector<OpAsmParser::OperandType, 4> allocators;
  SmallVector<Type, 4> allocatorTypes;
  std::array<int, 8> segments{0, 0, 0, 0, 0, 0, 0, 0};
  StringRef keyword;
  bool defaultVal = false;
  bool procBind = false;

  const int ifClausePos = 0;
  const int numThreadsClausePos = 1;
  const int privateClausePos = 2;
  const int firstprivateClausePos = 3;
  const int sharedClausePos = 4;
  const int copyinClausePos = 5;
  const int allocateClausePos = 6;
  const int allocatorPos = 7;
  const StringRef opName = result.name.getStringRef();

  while (succeeded(parser.parseOptionalKeyword(&keyword))) {
    if (keyword == "if") {
      // Fail if there was already another if condition.
      if (segments[ifClausePos])
        return allowedOnce(parser, "if", opName);
      if (parser.parseLParen() || parser.parseOperand(ifCond.first) ||
          parser.parseColonType(ifCond.second) || parser.parseRParen())
        return failure();
      segments[ifClausePos] = 1;
    } else if (keyword == "num_threads") {
      // Fail if there was already another num_threads clause.
      if (segments[numThreadsClausePos])
        return allowedOnce(parser, "num_threads", opName);
      if (parser.parseLParen() || parser.parseOperand(numThreads.first) ||
          parser.parseColonType(numThreads.second) || parser.parseRParen())
        return failure();
      segments[numThreadsClausePos] = 1;
    } else if (keyword == "private") {
      // Fail if there was already another private clause.
      if (segments[privateClausePos])
        return allowedOnce(parser, "private", opName);
      if (parseOperandAndTypeList(parser, privates, privateTypes))
        return failure();
      segments[privateClausePos] = privates.size();
    } else if (keyword == "firstprivate") {
      // Fail if there was already another firstprivate clause.
      if (segments[firstprivateClausePos])
        return allowedOnce(parser, "firstprivate", opName);
      if (parseOperandAndTypeList(parser, firstprivates, firstprivateTypes))
        return failure();
      segments[firstprivateClausePos] = firstprivates.size();
    } else if (keyword == "shared") {
      // Fail if there was already another shared clause.
      if (segments[sharedClausePos])
        return allowedOnce(parser, "shared", opName);
      if (parseOperandAndTypeList(parser, shareds, sharedTypes))
        return failure();
      segments[sharedClausePos] = shareds.size();
    } else if (keyword == "copyin") {
      // Fail if there was already another copyin clause.
      if (segments[copyinClausePos])
        return allowedOnce(parser, "copyin", opName);
      if (parseOperandAndTypeList(parser, copyins, copyinTypes))
        return failure();
      segments[copyinClausePos] = copyins.size();
    } else if (keyword == "allocate") {
      // Fail if there was already another allocate clause.
      if (segments[allocateClausePos])
        return allowedOnce(parser, "allocate", opName);
      if (parseAllocateAndAllocator(parser, allocates, allocateTypes,
                                    allocators, allocatorTypes))
        return failure();
      segments[allocateClausePos] = allocates.size();
      segments[allocatorPos] = allocators.size();
    } else if (keyword == "default") {
      // Fail if there was already another default clause.
      if (defaultVal)
        return allowedOnce(parser, "default", opName);
      defaultVal = true;
      StringRef defval;
      if (parser.parseLParen() || parser.parseKeyword(&defval) ||
          parser.parseRParen())
        return failure();
      // The def prefix is required for the attribute as "private" is a keyword
      // in C++.
      auto attr = parser.getBuilder().getStringAttr("def" + defval);
      result.addAttribute("default_val", attr);
    } else if (keyword == "proc_bind") {
      // Fail if there was already another proc_bind clause.
      if (procBind)
        return allowedOnce(parser, "proc_bind", opName);
      procBind = true;
      StringRef bind;
      if (parser.parseLParen() || parser.parseKeyword(&bind) ||
          parser.parseRParen())
        return failure();
      auto attr = parser.getBuilder().getStringAttr(bind);
      result.addAttribute("proc_bind_val", attr);
    } else {
      return parser.emitError(parser.getNameLoc())
             << keyword << " is not a valid clause for the " << opName
             << " operation";
    }
  }

  // Add if parameter.
  if (segments[ifClausePos] &&
      parser.resolveOperand(ifCond.first, ifCond.second, result.operands))
    return failure();

  // Add num_threads parameter.
  if (segments[numThreadsClausePos] &&
      parser.resolveOperand(numThreads.first, numThreads.second,
                            result.operands))
    return failure();

  // Add private parameters.
  if (segments[privateClausePos] &&
      parser.resolveOperands(privates, privateTypes, privates[0].location,
                             result.operands))
    return failure();

  // Add firstprivate parameters.
  if (segments[firstprivateClausePos] &&
      parser.resolveOperands(firstprivates, firstprivateTypes,
                             firstprivates[0].location, result.operands))
    return failure();

  // Add shared parameters.
  if (segments[sharedClausePos] &&
      parser.resolveOperands(shareds, sharedTypes, shareds[0].location,
                             result.operands))
    return failure();

  // Add copyin parameters.
  if (segments[copyinClausePos] &&
      parser.resolveOperands(copyins, copyinTypes, copyins[0].location,
                             result.operands))
    return failure();

  // Add allocate parameters.
  if (segments[allocateClausePos] &&
      parser.resolveOperands(allocates, allocateTypes, allocates[0].location,
                             result.operands))
    return failure();

  // Add allocator parameters.
  if (segments[allocatorPos] &&
      parser.resolveOperands(allocators, allocatorTypes, allocators[0].location,
                             result.operands))
    return failure();

  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getI32VectorAttr(segments));

  Region *body = result.addRegion();
  SmallVector<OpAsmParser::OperandType, 4> regionArgs;
  SmallVector<Type, 4> regionArgTypes;
  if (parser.parseRegion(*body, regionArgs, regionArgTypes))
    return failure();
  return success();
}

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

/// schedule ::= `schedule` `(` sched-list `)`
/// sched-list ::= sched-val | sched-val sched-list
/// sched-val ::= sched-with-chunk | sched-wo-chunk
/// sched-with-chunk ::= sched-with-chunk-types (`=` ssa-id-and-type)?
/// sched-with-chunk-types ::= `static` | `dynamic` | `guided`
/// sched-wo-chunk ::=  `auto` | `runtime`
static ParseResult
parseScheduleClause(OpAsmParser &parser, SmallString<8> &schedule,
                    Optional<OpAsmParser::OperandType> &chunkSize) {
  if (parser.parseLParen())
    return failure();

  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return failure();

  schedule = keyword;
  if (keyword == "static" || keyword == "dynamic" || keyword == "guided") {
    if (succeeded(parser.parseOptionalEqual())) {
      chunkSize = OpAsmParser::OperandType{};
      if (parser.parseOperand(*chunkSize))
        return failure();
    } else {
      chunkSize = llvm::NoneType::None;
    }
  } else if (keyword == "auto" || keyword == "runtime") {
    chunkSize = llvm::NoneType::None;
  } else {
    return parser.emitError(parser.getNameLoc()) << " expected schedule kind";
  }

  if (parser.parseRParen())
    return failure();

  return success();
}

/// reduction-init ::= `reduction` `(` reduction-entry-list `)`
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

/// Parses an OpenMP Workshare Loop operation
///
/// operation ::= `omp.wsloop` loop-control clause-list
/// loop-control ::= `(` ssa-id-list `)` `:` type `=`  loop-bounds
/// loop-bounds := `(` ssa-id-list `)` to `(` ssa-id-list `)` steps
/// steps := `step` `(`ssa-id-list`)`
/// clause-list ::= clause | empty | clause-list
/// clause ::= private | firstprivate | lastprivate | linear | schedule |
//             collapse | nowait | ordered | order | inclusive
/// private ::= `private` `(` ssa-id-and-type-list `)`
/// firstprivate ::= `firstprivate` `(` ssa-id-and-type-list `)`
/// lastprivate ::= `lastprivate` `(` ssa-id-and-type-list `)`
/// linear ::= `linear` `(` linear-list `)`
/// schedule ::= `schedule` `(` sched-list `)`
/// collapse ::= `collapse` `(` ssa-id-and-type `)`
/// nowait ::= `nowait`
/// ordered ::= `ordered` `(` ssa-id-and-type `)`
/// order ::= `order` `(` `concurrent` `)`
/// inclusive ::= `inclusive`
///
static ParseResult parseWsLoopOp(OpAsmParser &parser, OperationState &result) {
  Type loopVarType;
  int numIVs;

  // Parse an opening `(` followed by induction variables followed by `)`
  SmallVector<OpAsmParser::OperandType> ivs;
  if (parser.parseRegionArgumentList(ivs, /*requiredOperandCount=*/-1,
                                     OpAsmParser::Delimiter::Paren))
    return failure();

  numIVs = static_cast<int>(ivs.size());

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

  // Parse step values.
  SmallVector<OpAsmParser::OperandType> steps;
  if (parser.parseKeyword("step") ||
      parser.parseOperandList(steps, numIVs, OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(steps, loopVarType, result.operands))
    return failure();

  SmallVector<OpAsmParser::OperandType> privates;
  SmallVector<Type> privateTypes;
  SmallVector<OpAsmParser::OperandType> firstprivates;
  SmallVector<Type> firstprivateTypes;
  SmallVector<OpAsmParser::OperandType> lastprivates;
  SmallVector<Type> lastprivateTypes;
  SmallVector<OpAsmParser::OperandType> linears;
  SmallVector<Type> linearTypes;
  SmallVector<OpAsmParser::OperandType> linearSteps;
  SmallVector<SymbolRefAttr> reductionSymbols;
  SmallVector<OpAsmParser::OperandType> reductionVars;
  SmallVector<Type> reductionVarTypes;
  SmallString<8> schedule;
  Optional<OpAsmParser::OperandType> scheduleChunkSize;

  const StringRef opName = result.name.getStringRef();
  StringRef keyword;

  enum SegmentPos {
    lbPos = 0,
    ubPos,
    stepPos,
    privateClausePos,
    firstprivateClausePos,
    lastprivateClausePos,
    linearClausePos,
    linearStepPos,
    reductionVarPos,
    scheduleClausePos,
  };
  std::array<int, 10> segments{numIVs, numIVs, numIVs, 0, 0, 0, 0, 0, 0, 0};

  while (succeeded(parser.parseOptionalKeyword(&keyword))) {
    if (keyword == "private") {
      if (segments[privateClausePos])
        return allowedOnce(parser, "private", opName);
      if (parseOperandAndTypeList(parser, privates, privateTypes))
        return failure();
      segments[privateClausePos] = privates.size();
    } else if (keyword == "firstprivate") {
      // fail if there was already another firstprivate clause
      if (segments[firstprivateClausePos])
        return allowedOnce(parser, "firstprivate", opName);
      if (parseOperandAndTypeList(parser, firstprivates, firstprivateTypes))
        return failure();
      segments[firstprivateClausePos] = firstprivates.size();
    } else if (keyword == "lastprivate") {
      // fail if there was already another shared clause
      if (segments[lastprivateClausePos])
        return allowedOnce(parser, "lastprivate", opName);
      if (parseOperandAndTypeList(parser, lastprivates, lastprivateTypes))
        return failure();
      segments[lastprivateClausePos] = lastprivates.size();
    } else if (keyword == "linear") {
      // fail if there was already another linear clause
      if (segments[linearClausePos])
        return allowedOnce(parser, "linear", opName);
      if (parseLinearClause(parser, linears, linearTypes, linearSteps))
        return failure();
      segments[linearClausePos] = linears.size();
      segments[linearStepPos] = linearSteps.size();
    } else if (keyword == "schedule") {
      if (!schedule.empty())
        return allowedOnce(parser, "schedule", opName);
      if (parseScheduleClause(parser, schedule, scheduleChunkSize))
        return failure();
      if (scheduleChunkSize) {
        segments[scheduleClausePos] = 1;
      }
    } else if (keyword == "collapse") {
      auto type = parser.getBuilder().getI64Type();
      mlir::IntegerAttr attr;
      if (parser.parseLParen() || parser.parseAttribute(attr, type) ||
          parser.parseRParen())
        return failure();
      result.addAttribute("collapse_val", attr);
    } else if (keyword == "nowait") {
      auto attr = UnitAttr::get(parser.getContext());
      result.addAttribute("nowait", attr);
    } else if (keyword == "ordered") {
      mlir::IntegerAttr attr;
      if (succeeded(parser.parseOptionalLParen())) {
        auto type = parser.getBuilder().getI64Type();
        if (parser.parseAttribute(attr, type))
          return failure();
        if (parser.parseRParen())
          return failure();
      } else {
        // Use 0 to represent no ordered parameter was specified
        attr = parser.getBuilder().getI64IntegerAttr(0);
      }
      result.addAttribute("ordered_val", attr);
    } else if (keyword == "order") {
      StringRef order;
      if (parser.parseLParen() || parser.parseKeyword(&order) ||
          parser.parseRParen())
        return failure();
      auto attr = parser.getBuilder().getStringAttr(order);
      result.addAttribute("order", attr);
    } else if (keyword == "inclusive") {
      auto attr = UnitAttr::get(parser.getContext());
      result.addAttribute("inclusive", attr);
    } else if (keyword == "reduction") {
      if (segments[reductionVarPos])
        return allowedOnce(parser, "reduction", opName);
      if (failed(parseReductionVarList(parser, reductionSymbols, reductionVars,
                                       reductionVarTypes)))
        return failure();
      segments[reductionVarPos] = reductionVars.size();
    }
  }

  if (segments[privateClausePos]) {
    parser.resolveOperands(privates, privateTypes, privates[0].location,
                           result.operands);
  }

  if (segments[firstprivateClausePos]) {
    parser.resolveOperands(firstprivates, firstprivateTypes,
                           firstprivates[0].location, result.operands);
  }

  if (segments[lastprivateClausePos]) {
    parser.resolveOperands(lastprivates, lastprivateTypes,
                           lastprivates[0].location, result.operands);
  }

  if (segments[linearClausePos]) {
    parser.resolveOperands(linears, linearTypes, linears[0].location,
                           result.operands);
    auto linearStepType = parser.getBuilder().getI32Type();
    SmallVector<Type> linearStepTypes(linearSteps.size(), linearStepType);
    parser.resolveOperands(linearSteps, linearStepTypes,
                           linearSteps[0].location, result.operands);
  }

  if (segments[reductionVarPos]) {
    if (failed(parser.resolveOperands(reductionVars, reductionVarTypes,
                                      parser.getNameLoc(), result.operands))) {
      return failure();
    }
    SmallVector<Attribute> reductions(reductionSymbols.begin(),
                                      reductionSymbols.end());
    result.addAttribute("reductions",
                        parser.getBuilder().getArrayAttr(reductions));
  }

  if (!schedule.empty()) {
    schedule[0] = llvm::toUpper(schedule[0]);
    auto attr = parser.getBuilder().getStringAttr(schedule);
    result.addAttribute("schedule_val", attr);
    if (scheduleChunkSize) {
      auto chunkSizeType = parser.getBuilder().getI32Type();
      parser.resolveOperand(*scheduleChunkSize, chunkSizeType, result.operands);
    }
  }

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

static void printWsLoopOp(OpAsmPrinter &p, WsLoopOp op) {
  auto args = op.getRegion().front().getArguments();
  p << " (" << args << ") : " << args[0].getType() << " = (" << op.lowerBound()
    << ") to (" << op.upperBound() << ") step (" << op.step() << ")";

  // Print private, firstprivate, shared and copyin parameters
  auto printDataVars = [&p](StringRef name, OperandRange vars) {
    if (vars.empty())
      return;

    p << " " << name << "(";
    llvm::interleaveComma(
        vars, p, [&](const Value &v) { p << v << " : " << v.getType(); });
    p << ")";
  };
  printDataVars("private", op.private_vars());
  printDataVars("firstprivate", op.firstprivate_vars());
  printDataVars("lastprivate", op.lastprivate_vars());

  auto linearVars = op.linear_vars();
  auto linearVarsSize = linearVars.size();
  if (linearVarsSize) {
    p << " "
      << "linear"
      << "(";
    for (unsigned i = 0; i < linearVarsSize; ++i) {
      std::string separator = i == linearVarsSize - 1 ? ")" : ", ";
      p << linearVars[i];
      if (op.linear_step_vars().size() > i)
        p << " = " << op.linear_step_vars()[i];
      p << " : " << linearVars[i].getType() << separator;
    }
  }

  if (auto sched = op.schedule_val()) {
    auto schedLower = sched->lower();
    p << " schedule(" << schedLower;
    if (auto chunk = op.schedule_chunk_var()) {
      p << " = " << chunk;
    }
    p << ")";
  }

  if (auto collapse = op.collapse_val())
    p << " collapse(" << collapse << ")";

  if (op.nowait())
    p << " nowait";

  if (auto ordered = op.ordered_val()) {
    p << " ordered(" << ordered << ")";
  }

  if (!op.reduction_vars().empty()) {
    p << " reduction(";
    for (unsigned i = 0, e = op.getNumReductionVars(); i < e; ++i) {
      if (i != 0)
        p << ", ";
      p << (*op.reductions())[i] << " -> " << op.reduction_vars()[i] << " : "
        << op.reduction_vars()[i].getType();
    }
    p << ")";
  }

  if (op.inclusive()) {
    p << " inclusive";
  }

  p.printRegion(op.region(), /*printEntryBlockArgs=*/false);
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

static LogicalResult verifyReductionDeclareOp(ReductionDeclareOp op) {
  if (op.initializerRegion().empty())
    return op.emitOpError() << "expects non-empty initializer region";
  Block &initializerEntryBlock = op.initializerRegion().front();
  if (initializerEntryBlock.getNumArguments() != 1 ||
      initializerEntryBlock.getArgument(0).getType() != op.type()) {
    return op.emitOpError() << "expects initializer region with one argument "
                               "of the reduction type";
  }

  for (YieldOp yieldOp : op.initializerRegion().getOps<YieldOp>()) {
    if (yieldOp.results().size() != 1 ||
        yieldOp.results().getTypes()[0] != op.type())
      return op.emitOpError() << "expects initializer region to yield a value "
                                 "of the reduction type";
  }

  if (op.reductionRegion().empty())
    return op.emitOpError() << "expects non-empty reduction region";
  Block &reductionEntryBlock = op.reductionRegion().front();
  if (reductionEntryBlock.getNumArguments() != 2 ||
      reductionEntryBlock.getArgumentTypes()[0] !=
          reductionEntryBlock.getArgumentTypes()[1] ||
      reductionEntryBlock.getArgumentTypes()[0] != op.type())
    return op.emitOpError() << "expects reduction region with two arguments of "
                               "the reduction type";
  for (YieldOp yieldOp : op.reductionRegion().getOps<YieldOp>()) {
    if (yieldOp.results().size() != 1 ||
        yieldOp.results().getTypes()[0] != op.type())
      return op.emitOpError() << "expects reduction region to yield a value "
                                 "of the reduction type";
  }

  if (op.atomicReductionRegion().empty())
    return success();

  Block &atomicReductionEntryBlock = op.atomicReductionRegion().front();
  if (atomicReductionEntryBlock.getNumArguments() != 2 ||
      atomicReductionEntryBlock.getArgumentTypes()[0] !=
          atomicReductionEntryBlock.getArgumentTypes()[1])
    return op.emitOpError() << "expects atomic reduction region with two "
                               "arguments of the same type";
  auto ptrType = atomicReductionEntryBlock.getArgumentTypes()[0]
                     .dyn_cast<PointerLikeType>();
  if (!ptrType || ptrType.getElementType() != op.type())
    return op.emitOpError() << "expects atomic reduction region arguments to "
                               "be accumulators containing the reduction type";
  return success();
}

static LogicalResult verifyReductionOp(ReductionOp op) {
  // TODO: generalize this to an op interface when there is more than one op
  // that supports reductions.
  auto container = op->getParentOfType<WsLoopOp>();
  for (unsigned i = 0, e = container.getNumReductionVars(); i < e; ++i)
    if (container.reduction_vars()[i] == op.accumulator())
      return success();

  return op.emitOpError() << "the accumulator is not used by the parent";
}

//===----------------------------------------------------------------------===//
// WsLoopOp
//===----------------------------------------------------------------------===//

void WsLoopOp::build(OpBuilder &builder, OperationState &state,
                     ValueRange lowerBound, ValueRange upperBound,
                     ValueRange step, ArrayRef<NamedAttribute> attributes) {
  build(builder, state, TypeRange(), lowerBound, upperBound, step,
        /*private_vars=*/ValueRange(),
        /*firstprivate_vars=*/ValueRange(), /*lastprivate_vars=*/ValueRange(),
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
                     ValueRange privateVars, ValueRange firstprivateVars,
                     ValueRange lastprivateVars, ValueRange linearVars,
                     ValueRange linearStepVars, ValueRange reductionVars,
                     StringAttr scheduleVal, Value scheduleChunkVar,
                     IntegerAttr collapseVal, UnitAttr nowait,
                     IntegerAttr orderedVal, StringAttr orderVal,
                     UnitAttr inclusive, bool buildBody) {
  result.addOperands(lowerBounds);
  result.addOperands(upperBounds);
  result.addOperands(steps);
  result.addOperands(privateVars);
  result.addOperands(firstprivateVars);
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
           static_cast<int32_t>(privateVars.size()),
           static_cast<int32_t>(firstprivateVars.size()),
           static_cast<int32_t>(lastprivateVars.size()),
           static_cast<int32_t>(linearVars.size()),
           static_cast<int32_t>(linearStepVars.size()),
           static_cast<int32_t>(reductionVars.size()),
           static_cast<int32_t>(scheduleChunkVar != nullptr ? 1 : 0)}));

  Region *bodyRegion = result.addRegion();
  if (buildBody) {
    OpBuilder::InsertionGuard guard(builder);
    unsigned numIVs = steps.size();
    SmallVector<Type, 8> argTypes(numIVs, steps.getType().front());
    builder.createBlock(bodyRegion, {}, argTypes);
  }
}

static LogicalResult verifyWsLoopOp(WsLoopOp op) {
  if (op.getNumReductionVars() != 0) {
    if (!op.reductions() ||
        op.reductions()->size() != op.getNumReductionVars()) {
      return op.emitOpError() << "expected as many reduction symbol references "
                                 "as reduction variables";
    }
  } else {
    if (op.reductions())
      return op.emitOpError() << "unexpected reduction symbol references";
    return success();
  }

  DenseSet<Value> accumulators;
  for (auto args : llvm::zip(op.reduction_vars(), *op.reductions())) {
    Value accum = std::get<0>(args);
    if (!accumulators.insert(accum).second) {
      return op.emitOpError() << "accumulator variable used more than once";
    }
    Type varType = accum.getType().cast<PointerLikeType>();
    auto symbolRef = std::get<1>(args).cast<SymbolRefAttr>();
    auto decl =
        SymbolTable::lookupNearestSymbolFrom<ReductionDeclareOp>(op, symbolRef);
    if (!decl) {
      return op.emitOpError() << "expected symbol reference " << symbolRef
                              << " to point to a reduction declaration";
    }

    if (decl.getAccumulatorType() && decl.getAccumulatorType() != varType) {
      return op.emitOpError()
             << "expected accumulator (" << varType
             << ") to be the same type as reduction declaration ("
             << decl.getAccumulatorType() << ")";
    }
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
  if (failed(parser.parseOptionalKeyword("hint"))) {
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
  p << ")";
}

/// Verifies a synchronization hint clause
static LogicalResult verifySynchronizationHint(Operation *op, int32_t hint) {

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
// Verifier for critical construct (2.17.1)
//===----------------------------------------------------------------------===//

static LogicalResult verifyCriticalOp(CriticalOp op) {

  if (failed(verifySynchronizationHint(op, op.hint()))) {
    return failure();
  }
  if (!op.name().hasValue() && (op.hint() != 0))
    return op.emitOpError() << "must specify a name unless the effect is as if "
                               "no hint is specified";

  if (op.nameAttr()) {
    auto symbolRef = op.nameAttr().cast<SymbolRefAttr>();
    auto decl =
        SymbolTable::lookupNearestSymbolFrom<CriticalDeclareOp>(op, symbolRef);
    if (!decl) {
      return op.emitOpError() << "expected symbol reference " << symbolRef
                              << " to point to a critical declaration";
    }
  }

  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOps.cpp.inc"
