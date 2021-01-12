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
static ParseResult allowedOnce(OpAsmParser &parser, llvm::StringRef clause,
                               llvm::StringRef operation) {
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
  llvm::StringRef keyword;
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
  const llvm::StringRef opName = result.name.getStringRef();

  while (succeeded(parser.parseOptionalKeyword(&keyword))) {
    if (keyword == "if") {
      // Fail if there was already another if condition
      if (segments[ifClausePos])
        return allowedOnce(parser, "if", opName);
      if (parser.parseLParen() || parser.parseOperand(ifCond.first) ||
          parser.parseColonType(ifCond.second) || parser.parseRParen())
        return failure();
      segments[ifClausePos] = 1;
    } else if (keyword == "num_threads") {
      // fail if there was already another num_threads clause
      if (segments[numThreadsClausePos])
        return allowedOnce(parser, "num_threads", opName);
      if (parser.parseLParen() || parser.parseOperand(numThreads.first) ||
          parser.parseColonType(numThreads.second) || parser.parseRParen())
        return failure();
      segments[numThreadsClausePos] = 1;
    } else if (keyword == "private") {
      // fail if there was already another private clause
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
    } else if (keyword == "shared") {
      // fail if there was already another shared clause
      if (segments[sharedClausePos])
        return allowedOnce(parser, "shared", opName);
      if (parseOperandAndTypeList(parser, shareds, sharedTypes))
        return failure();
      segments[sharedClausePos] = shareds.size();
    } else if (keyword == "copyin") {
      // fail if there was already another copyin clause
      if (segments[copyinClausePos])
        return allowedOnce(parser, "copyin", opName);
      if (parseOperandAndTypeList(parser, copyins, copyinTypes))
        return failure();
      segments[copyinClausePos] = copyins.size();
    } else if (keyword == "allocate") {
      // fail if there was already another allocate clause
      if (segments[allocateClausePos])
        return allowedOnce(parser, "allocate", opName);
      if (parseAllocateAndAllocator(parser, allocates, allocateTypes,
                                    allocators, allocatorTypes))
        return failure();
      segments[allocateClausePos] = allocates.size();
      segments[allocatorPos] = allocators.size();
    } else if (keyword == "default") {
      // fail if there was already another default clause
      if (defaultVal)
        return allowedOnce(parser, "default", opName);
      defaultVal = true;
      llvm::StringRef defval;
      if (parser.parseLParen() || parser.parseKeyword(&defval) ||
          parser.parseRParen())
        return failure();
      llvm::SmallString<16> attrval;
      // The def prefix is required for the attribute as "private" is a keyword
      // in C++
      attrval += "def";
      attrval += defval;
      auto attr = parser.getBuilder().getStringAttr(attrval);
      result.addAttribute("default_val", attr);
    } else if (keyword == "proc_bind") {
      // fail if there was already another proc_bind clause
      if (procBind)
        return allowedOnce(parser, "proc_bind", opName);
      procBind = true;
      llvm::StringRef bind;
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

  // Add if parameter
  if (segments[ifClausePos] &&
      parser.resolveOperand(ifCond.first, ifCond.second, result.operands))
    return failure();

  // Add num_threads parameter
  if (segments[numThreadsClausePos] &&
      parser.resolveOperand(numThreads.first, numThreads.second,
                            result.operands))
    return failure();

  // Add private parameters
  if (segments[privateClausePos] &&
      parser.resolveOperands(privates, privateTypes, privates[0].location,
                             result.operands))
    return failure();

  // Add firstprivate parameters
  if (segments[firstprivateClausePos] &&
      parser.resolveOperands(firstprivates, firstprivateTypes,
                             firstprivates[0].location, result.operands))
    return failure();

  // Add shared parameters
  if (segments[sharedClausePos] &&
      parser.resolveOperands(shareds, sharedTypes, shareds[0].location,
                             result.operands))
    return failure();

  // Add copyin parameters
  if (segments[copyinClausePos] &&
      parser.resolveOperands(copyins, copyinTypes, copyins[0].location,
                             result.operands))
    return failure();

  // Add allocate parameters
  if (segments[allocateClausePos] &&
      parser.resolveOperands(allocates, allocateTypes, allocates[0].location,
                             result.operands))
    return failure();

  // Add allocator parameters
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
        /*nowait=*/false, /*ordered_val=*/nullptr, /*order_val=*/nullptr,
        /*inclusive=*/false);
  state.addAttributes(attributes);
}

#define GET_OP_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOps.cpp.inc"
