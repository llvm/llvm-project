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
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include <cstddef>

#include "mlir/Dialect/OpenMP/OpenMPOpsEnums.cpp.inc"

using namespace mlir;
using namespace mlir::omp;

void OpenMPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/OpenMP/OpenMPOps.cpp.inc"
      >();
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
  if (parser.parseLParen())
    return failure();

  do {
    OpAsmParser::OperandType operand;
    Type type;
    if (parser.parseOperand(operand) || parser.parseColonType(type))
      return failure();
    operands.push_back(operand);
    types.push_back(type);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRParen())
    return failure();

  return success();
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
  if (parser.parseLParen())
    return failure();

  do {
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
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRParen())
    return failure();

  return success();
}

static LogicalResult verifyParallelOp(ParallelOp op) {
  if (op.allocate_vars().size() != op.allocators_vars().size())
    return op.emitError(
        "expected equal sizes for allocate and allocator variables");
  return success();
}

static void printParallelOp(OpAsmPrinter &p, ParallelOp op) {
  p << "omp.parallel";

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
  SmallString<8> schedule;
  Optional<OpAsmParser::OperandType> scheduleChunkSize;
  std::array<int, 9> segments{numIVs, numIVs, numIVs, 0, 0, 0, 0, 0, 0};

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
    scheduleClausePos,
  };

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
      auto attr = UnitAttr::get(parser.getBuilder().getContext());
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
      auto attr = UnitAttr::get(parser.getBuilder().getContext());
      result.addAttribute("inclusive", attr);
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
  if (parser.parseRegion(*body, ivs, ivTypes))
    return failure();
  return success();
}

static void printWsLoopOp(OpAsmPrinter &p, WsLoopOp op) {
  auto args = op.getRegion().front().getArguments();
  p << op.getOperationName() << " (" << args << ") : " << args[0].getType()
    << " = (" << op.lowerBound() << ") to (" << op.upperBound() << ") step ("
    << op.step() << ")";

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

  if (op.inclusive()) {
    p << " inclusive";
  }

  p.printRegion(op.region(), /*printEntryBlockArgs=*/false);
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
        /*schedule_val=*/nullptr, /*schedule_chunk_var=*/nullptr,
        /*collapse_val=*/nullptr,
        /*nowait=*/nullptr, /*ordered_val=*/nullptr, /*order_val=*/nullptr,
        /*inclusive=*/nullptr, /*buildBody=*/false);
  state.addAttributes(attributes);
}

void WsLoopOp::build(OpBuilder &, OperationState &state, TypeRange resultTypes,
                     ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  state.addOperands(operands);
  state.addAttributes(attributes);
  (void)state.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  state.addTypes(resultTypes);
}

void WsLoopOp::build(OpBuilder &builder, OperationState &result,
                     TypeRange typeRange, ValueRange lowerBounds,
                     ValueRange upperBounds, ValueRange steps,
                     ValueRange privateVars, ValueRange firstprivateVars,
                     ValueRange lastprivateVars, ValueRange linearVars,
                     ValueRange linearStepVars, StringAttr scheduleVal,
                     Value scheduleChunkVar, IntegerAttr collapseVal,
                     UnitAttr nowait, IntegerAttr orderedVal,
                     StringAttr orderVal, UnitAttr inclusive, bool buildBody) {
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
           static_cast<int32_t>(scheduleChunkVar != nullptr ? 1 : 0)}));

  Region *bodyRegion = result.addRegion();
  if (buildBody) {
    OpBuilder::InsertionGuard guard(builder);
    unsigned numIVs = steps.size();
    SmallVector<Type, 8> argTypes(numIVs, steps.getType().front());
    builder.createBlock(bodyRegion, {}, argTypes);
  }
}

#define GET_OP_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOps.cpp.inc"
