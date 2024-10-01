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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/OpenACCMPCommon/Interfaces/AtomicInterfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/FoldInterfaces.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
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
void printClauseAttr(OpAsmPrinter &p, Operation *op, ClauseAttr attr) {
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
  PrivateParseArgs(SmallVectorImpl<OpAsmParser::UnresolvedOperand> &vars,
                   SmallVectorImpl<Type> &types, ArrayAttr &syms)
      : vars(vars), types(types), syms(syms) {}
};
struct ReductionParseArgs {
  SmallVectorImpl<OpAsmParser::UnresolvedOperand> &vars;
  SmallVectorImpl<Type> &types;
  DenseBoolArrayAttr &byref;
  ArrayAttr &syms;
  ReductionParseArgs(SmallVectorImpl<OpAsmParser::UnresolvedOperand> &vars,
                     SmallVectorImpl<Type> &types, DenseBoolArrayAttr &byref,
                     ArrayAttr &syms)
      : vars(vars), types(types), byref(byref), syms(syms) {}
};
struct AllRegionParseArgs {
  std::optional<ReductionParseArgs> inReductionArgs;
  std::optional<MapParseArgs> mapArgs;
  std::optional<PrivateParseArgs> privateArgs;
  std::optional<ReductionParseArgs> reductionArgs;
  std::optional<ReductionParseArgs> taskReductionArgs;
  std::optional<MapParseArgs> useDeviceAddrArgs;
  std::optional<MapParseArgs> useDevicePtrArgs;
};
} // namespace

static ParseResult parseClauseWithRegionArgs(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &types,
    SmallVectorImpl<OpAsmParser::Argument> &regionPrivateArgs,
    ArrayAttr *symbols = nullptr, DenseBoolArrayAttr *byref = nullptr) {
  SmallVector<SymbolRefAttr> symbolVec;
  SmallVector<bool> isByRefVec;
  unsigned regionArgOffset = regionPrivateArgs.size();

  if (parser.parseLParen())
    return failure();

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
    StringRef keyword, std::optional<PrivateParseArgs> reductionArgs) {
  if (succeeded(parser.parseOptionalKeyword(keyword))) {
    if (!reductionArgs)
      return failure();

    if (failed(parseClauseWithRegionArgs(parser, reductionArgs->vars,
                                         reductionArgs->types, entryBlockArgs,
                                         &reductionArgs->syms)))
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
            &reductionArgs->syms, &reductionArgs->byref)))
      return failure();
  }
  return success();
}

static ParseResult parseBlockArgRegion(OpAsmParser &parser, Region &region,
                                       AllRegionParseArgs args) {
  llvm::SmallVector<OpAsmParser::Argument> entryBlockArgs;

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

static ParseResult parseInReductionMapPrivateRegion(
    OpAsmParser &parser, Region &region,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &inReductionVars,
    SmallVectorImpl<Type> &inReductionTypes,
    DenseBoolArrayAttr &inReductionByref, ArrayAttr &inReductionSyms,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &mapVars,
    SmallVectorImpl<Type> &mapTypes,
    llvm::SmallVectorImpl<OpAsmParser::UnresolvedOperand> &privateVars,
    llvm::SmallVectorImpl<Type> &privateTypes, ArrayAttr &privateSyms) {
  AllRegionParseArgs args;
  args.inReductionArgs.emplace(inReductionVars, inReductionTypes,
                               inReductionByref, inReductionSyms);
  args.mapArgs.emplace(mapVars, mapTypes);
  args.privateArgs.emplace(privateVars, privateTypes, privateSyms);
  return parseBlockArgRegion(parser, region, args);
}

static ParseResult parseInReductionPrivateRegion(
    OpAsmParser &parser, Region &region,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &inReductionVars,
    SmallVectorImpl<Type> &inReductionTypes,
    DenseBoolArrayAttr &inReductionByref, ArrayAttr &inReductionSyms,
    llvm::SmallVectorImpl<OpAsmParser::UnresolvedOperand> &privateVars,
    llvm::SmallVectorImpl<Type> &privateTypes, ArrayAttr &privateSyms) {
  AllRegionParseArgs args;
  args.inReductionArgs.emplace(inReductionVars, inReductionTypes,
                               inReductionByref, inReductionSyms);
  args.privateArgs.emplace(privateVars, privateTypes, privateSyms);
  return parseBlockArgRegion(parser, region, args);
}

static ParseResult parseInReductionPrivateReductionRegion(
    OpAsmParser &parser, Region &region,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &inReductionVars,
    SmallVectorImpl<Type> &inReductionTypes,
    DenseBoolArrayAttr &inReductionByref, ArrayAttr &inReductionSyms,
    llvm::SmallVectorImpl<OpAsmParser::UnresolvedOperand> &privateVars,
    llvm::SmallVectorImpl<Type> &privateTypes, ArrayAttr &privateSyms,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &reductionVars,
    SmallVectorImpl<Type> &reductionTypes, DenseBoolArrayAttr &reductionByref,
    ArrayAttr &reductionSyms) {
  AllRegionParseArgs args;
  args.inReductionArgs.emplace(inReductionVars, inReductionTypes,
                               inReductionByref, inReductionSyms);
  args.privateArgs.emplace(privateVars, privateTypes, privateSyms);
  args.reductionArgs.emplace(reductionVars, reductionTypes, reductionByref,
                             reductionSyms);
  return parseBlockArgRegion(parser, region, args);
}

static ParseResult parsePrivateRegion(
    OpAsmParser &parser, Region &region,
    llvm::SmallVectorImpl<OpAsmParser::UnresolvedOperand> &privateVars,
    llvm::SmallVectorImpl<Type> &privateTypes, ArrayAttr &privateSyms) {
  AllRegionParseArgs args;
  args.privateArgs.emplace(privateVars, privateTypes, privateSyms);
  return parseBlockArgRegion(parser, region, args);
}

static ParseResult parsePrivateReductionRegion(
    OpAsmParser &parser, Region &region,
    llvm::SmallVectorImpl<OpAsmParser::UnresolvedOperand> &privateVars,
    llvm::SmallVectorImpl<Type> &privateTypes, ArrayAttr &privateSyms,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &reductionVars,
    SmallVectorImpl<Type> &reductionTypes, DenseBoolArrayAttr &reductionByref,
    ArrayAttr &reductionSyms) {
  AllRegionParseArgs args;
  args.privateArgs.emplace(privateVars, privateTypes, privateSyms);
  args.reductionArgs.emplace(reductionVars, reductionTypes, reductionByref,
                             reductionSyms);
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
  PrivatePrintArgs(ValueRange vars, TypeRange types, ArrayAttr syms)
      : vars(vars), types(types), syms(syms) {}
};
struct ReductionPrintArgs {
  ValueRange vars;
  TypeRange types;
  DenseBoolArrayAttr byref;
  ArrayAttr syms;
  ReductionPrintArgs(ValueRange vars, TypeRange types, DenseBoolArrayAttr byref,
                     ArrayAttr syms)
      : vars(vars), types(types), byref(byref), syms(syms) {}
};
struct AllRegionPrintArgs {
  std::optional<ReductionPrintArgs> inReductionArgs;
  std::optional<MapPrintArgs> mapArgs;
  std::optional<PrivatePrintArgs> privateArgs;
  std::optional<ReductionPrintArgs> reductionArgs;
  std::optional<ReductionPrintArgs> taskReductionArgs;
  std::optional<MapPrintArgs> useDeviceAddrArgs;
  std::optional<MapPrintArgs> useDevicePtrArgs;
};
} // namespace

static void printClauseWithRegionArgs(OpAsmPrinter &p, MLIRContext *ctx,
                                      StringRef clauseName,
                                      ValueRange argsSubrange,
                                      ValueRange operands, TypeRange types,
                                      ArrayAttr symbols = nullptr,
                                      DenseBoolArrayAttr byref = nullptr) {
  if (argsSubrange.empty())
    return;

  p << clauseName << "(";

  if (!symbols) {
    llvm::SmallVector<Attribute> values(operands.size(), nullptr);
    symbols = ArrayAttr::get(ctx, values);
  }

  if (!byref) {
    mlir::SmallVector<bool> values(operands.size(), false);
    byref = DenseBoolArrayAttr::get(ctx, values);
  }

  llvm::interleaveComma(
      llvm::zip_equal(operands, argsSubrange, symbols, byref.asArrayRef()), p,
      [&p](auto t) {
        auto [op, arg, sym, isByRef] = t;
        if (isByRef)
          p << "byref ";
        if (sym)
          p << sym << " ";
        p << op << " -> " << arg;
      });
  p << " : ";
  llvm::interleaveComma(types, p);
  p << ") ";
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
    printClauseWithRegionArgs(p, ctx, clauseName, argsSubrange,
                              privateArgs->vars, privateArgs->types,
                              privateArgs->syms);
}

static void
printBlockArgClause(OpAsmPrinter &p, MLIRContext *ctx, StringRef clauseName,
                    ValueRange argsSubrange,
                    std::optional<ReductionPrintArgs> reductionArgs) {
  if (reductionArgs)
    printClauseWithRegionArgs(p, ctx, clauseName, argsSubrange,
                              reductionArgs->vars, reductionArgs->types,
                              reductionArgs->syms, reductionArgs->byref);
}

static void printBlockArgRegion(OpAsmPrinter &p, Operation *op, Region &region,
                                const AllRegionPrintArgs &args) {
  auto iface = llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(op);
  MLIRContext *ctx = op->getContext();

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

static void printInReductionMapPrivateRegion(
    OpAsmPrinter &p, Operation *op, Region &region, ValueRange inReductionVars,
    TypeRange inReductionTypes, DenseBoolArrayAttr inReductionByref,
    ArrayAttr inReductionSyms, ValueRange mapVars, TypeRange mapTypes,
    ValueRange privateVars, TypeRange privateTypes, ArrayAttr privateSyms) {
  AllRegionPrintArgs args;
  args.inReductionArgs.emplace(inReductionVars, inReductionTypes,
                               inReductionByref, inReductionSyms);
  args.mapArgs.emplace(mapVars, mapTypes);
  args.privateArgs.emplace(privateVars, privateTypes, privateSyms);
  printBlockArgRegion(p, op, region, args);
}

static void printInReductionPrivateRegion(
    OpAsmPrinter &p, Operation *op, Region &region, ValueRange inReductionVars,
    TypeRange inReductionTypes, DenseBoolArrayAttr inReductionByref,
    ArrayAttr inReductionSyms, ValueRange privateVars, TypeRange privateTypes,
    ArrayAttr privateSyms) {
  AllRegionPrintArgs args;
  args.inReductionArgs.emplace(inReductionVars, inReductionTypes,
                               inReductionByref, inReductionSyms);
  args.privateArgs.emplace(privateVars, privateTypes, privateSyms);
  printBlockArgRegion(p, op, region, args);
}

static void printInReductionPrivateReductionRegion(
    OpAsmPrinter &p, Operation *op, Region &region, ValueRange inReductionVars,
    TypeRange inReductionTypes, DenseBoolArrayAttr inReductionByref,
    ArrayAttr inReductionSyms, ValueRange privateVars, TypeRange privateTypes,
    ArrayAttr privateSyms, ValueRange reductionVars, TypeRange reductionTypes,
    DenseBoolArrayAttr reductionByref, ArrayAttr reductionSyms) {
  AllRegionPrintArgs args;
  args.inReductionArgs.emplace(inReductionVars, inReductionTypes,
                               inReductionByref, inReductionSyms);
  args.privateArgs.emplace(privateVars, privateTypes, privateSyms);
  args.reductionArgs.emplace(reductionVars, reductionTypes, reductionByref,
                             reductionSyms);
  printBlockArgRegion(p, op, region, args);
}

static void printPrivateRegion(OpAsmPrinter &p, Operation *op, Region &region,
                               ValueRange privateVars, TypeRange privateTypes,
                               ArrayAttr privateSyms) {
  AllRegionPrintArgs args;
  args.privateArgs.emplace(privateVars, privateTypes, privateSyms);
  printBlockArgRegion(p, op, region, args);
}

static void printPrivateReductionRegion(
    OpAsmPrinter &p, Operation *op, Region &region, ValueRange privateVars,
    TypeRange privateTypes, ArrayAttr privateSyms, ValueRange reductionVars,
    TypeRange reductionTypes, DenseBoolArrayAttr reductionByref,
    ArrayAttr reductionSyms) {
  AllRegionPrintArgs args;
  args.privateArgs.emplace(privateVars, privateTypes, privateSyms);
  args.reductionArgs.emplace(reductionVars, reductionTypes, reductionByref,
                             reductionSyms);
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
uint64_t mapTypeToBitFlag(uint64_t value,
                          llvm::omp::OpenMPOffloadMappingFlags flag) {
  return value & llvm::to_underlying(flag);
}

/// Parses a map_entries map type from a string format back into its numeric
/// value.
///
/// map-clause = `map_clauses (  ( `(` `always, `? `close, `? `present, `? (
/// `to` | `from` | `delete` `)` )+ `)` )
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
                                     DenseIntElementsAttr &membersIdx) {
  SmallVector<APInt> values;
  int64_t value;
  int64_t shape[2] = {0, 0};
  unsigned shapeTmp = 0;
  auto parseIndices = [&]() -> ParseResult {
    if (parser.parseInteger(value))
      return failure();
    shapeTmp++;
    values.push_back(APInt(32, value));
    return success();
  };

  do {
    if (failed(parser.parseLSquare()))
      return failure();

    if (parser.parseCommaSeparatedList(parseIndices))
      return failure();

    if (failed(parser.parseRSquare()))
      return failure();

    // Only set once, if any indices are not the same size
    // we error out in the next check as that's unsupported
    if (shape[1] == 0)
      shape[1] = shapeTmp;

    // Verify that the recently parsed list is equal to the
    // first one we parsed, they must be equal lengths to
    // keep the rectangular shape DenseIntElementsAttr
    // requires
    if (shapeTmp != shape[1])
      return failure();

    shapeTmp = 0;
    shape[0]++;
  } while (succeeded(parser.parseOptionalComma()));

  if (!values.empty()) {
    ShapedType valueType =
        VectorType::get(shape, IntegerType::get(parser.getContext(), 32));
    membersIdx = DenseIntElementsAttr::get(valueType, values);
  }

  return success();
}

static void printMembersIndex(OpAsmPrinter &p, MapInfoOp op,
                              DenseIntElementsAttr membersIdx) {
  llvm::ArrayRef<int64_t> shape = membersIdx.getShapedType().getShape();
  assert(shape.size() <= 2);

  if (!membersIdx)
    return;

  for (int i = 0; i < shape[0]; ++i) {
    p << "[";
    int rowOffset = i * shape[1];
    for (int j = 0; j < shape[1]; ++j) {
      p << membersIdx.getValues<int32_t>()[rowOffset + j];
      if ((j + 1) < shape[1])
        p << ",";
    }
    p << "]";

    if ((i + 1) < shape[0])
      p << ", ";
  }
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
      emitError(op->getLoc(), "missing map operation");

    if (auto mapInfoOp =
            mlir::dyn_cast<mlir::omp::MapInfoOp>(mapOp.getDefiningOp())) {
      if (!mapInfoOp.getMapType().has_value())
        emitError(op->getLoc(), "missing map type for map operand");

      if (!mapInfoOp.getMapCaptureType().has_value())
        emitError(op->getLoc(), "missing map capture type for map operand");

      uint64_t mapTypeBits = mapInfoOp.getMapType().value();

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
    } else {
      emitError(op->getLoc(), "map argument is not a map entry operation");
    }
  }

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
                  makeArrayAttr(ctx, clauses.dependKinds), clauses.dependVars,
                  clauses.device, clauses.hasDeviceAddrVars, clauses.ifExpr,
                  /*in_reduction_vars=*/{}, /*in_reduction_byref=*/nullptr,
                  /*in_reduction_syms=*/nullptr, clauses.isDevicePtrVars,
                  clauses.mapVars, clauses.nowait, clauses.privateVars,
                  makeArrayAttr(ctx, clauses.privateSyms), clauses.threadLimit);
}

LogicalResult TargetOp::verify() {
  LogicalResult verifyDependVars =
      verifyDependVarList(*this, getDependKinds(), getDependVars());
  return failed(verifyDependVars) ? verifyDependVars
                                  : verifyMapClause(*this, getMapVars());
}

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

void ParallelOp::build(OpBuilder &builder, OperationState &state,
                       ArrayRef<NamedAttribute> attributes) {
  ParallelOp::build(builder, state, /*allocate_vars=*/ValueRange(),
                    /*allocator_vars=*/ValueRange(), /*if_expr=*/nullptr,
                    /*num_threads=*/nullptr, /*private_vars=*/ValueRange(),
                    /*private_syms=*/nullptr, /*proc_bind_kind=*/nullptr,
                    /*reduction_vars=*/ValueRange(),
                    /*reduction_byref=*/nullptr, /*reduction_syms=*/nullptr);
  state.addAttributes(attributes);
}

void ParallelOp::build(OpBuilder &builder, OperationState &state,
                       const ParallelOperands &clauses) {
  MLIRContext *ctx = builder.getContext();
  ParallelOp::build(builder, state, clauses.allocateVars, clauses.allocatorVars,
                    clauses.ifExpr, clauses.numThreads, clauses.privateVars,
                    makeArrayAttr(ctx, clauses.privateSyms),
                    clauses.procBindKind, clauses.reductionVars,
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

    Type privatizerType = privatizerOp.getType();

    if (varType != privatizerType)
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
  auto distributeChildOps = getOps<DistributeOp>();
  if (!distributeChildOps.empty()) {
    if (!isComposite())
      return emitError()
             << "'omp.composite' attribute missing from composite operation";

    auto *ompDialect = getContext()->getLoadedDialect<OpenMPDialect>();
    Operation &distributeOp = **distributeChildOps.begin();
    for (Operation &childOp : getOps()) {
      if (&childOp == &distributeOp || ompDialect != childOp.getDialect())
        continue;

      if (!childOp.hasTrait<OpTrait::IsTerminator>())
        return emitError() << "unexpected OpenMP operation inside of composite "
                              "'omp.parallel'";
    }
  } else if (isComposite()) {
    return emitError()
           << "'omp.composite' attribute present in non-composite operation";
  }

  if (getAllocateVars().size() != getAllocatorVars().size())
    return emitError(
        "expected equal sizes for allocate and allocator variables");

  if (failed(verifyPrivateVarList(*this)))
    return failure();

  return verifyReductionVarList(*this, getReductionSyms(), getReductionVars(),
                                getReductionByref());
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
  // TODO Store clauses in op: privateVars, privateSyms.
  TeamsOp::build(
      builder, state, clauses.allocateVars, clauses.allocatorVars,
      clauses.ifExpr, clauses.numTeamsLower, clauses.numTeamsUpper,
      /*private_vars=*/{}, /*private_syms=*/nullptr, clauses.reductionVars,
      makeDenseBoolArrayAttr(ctx, clauses.reductionByref),
      makeArrayAttr(ctx, clauses.reductionSyms), clauses.threadLimit);
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

unsigned SectionOp::numPrivateBlockArgs() {
  return getParentOp().numPrivateBlockArgs();
}

unsigned SectionOp::numReductionBlockArgs() {
  return getParentOp().numReductionBlockArgs();
}

//===----------------------------------------------------------------------===//
// SectionsOp
//===----------------------------------------------------------------------===//

void SectionsOp::build(OpBuilder &builder, OperationState &state,
                       const SectionsOperands &clauses) {
  MLIRContext *ctx = builder.getContext();
  // TODO Store clauses in op: privateVars, privateSyms.
  SectionsOp::build(builder, state, clauses.allocateVars, clauses.allocatorVars,
                    clauses.nowait, /*private_vars=*/{},
                    /*private_syms=*/nullptr, clauses.reductionVars,
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
  // TODO Store clauses in op: privateVars, privateSyms.
  SingleOp::build(builder, state, clauses.allocateVars, clauses.allocatorVars,
                  clauses.copyprivateVars,
                  makeArrayAttr(ctx, clauses.copyprivateSyms), clauses.nowait,
                  /*private_vars=*/{}, /*private_syms=*/nullptr);
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
// LoopWrapperInterface
//===----------------------------------------------------------------------===//

LogicalResult LoopWrapperInterface::verifyImpl() {
  Operation *op = this->getOperation();
  if (op->getNumRegions() != 1)
    return emitOpError() << "loop wrapper contains multiple regions";

  Region &region = op->getRegion(0);
  if (!region.hasOneBlock())
    return emitOpError() << "loop wrapper contains multiple blocks";

  if (::llvm::range_size(region.getOps()) != 2)
    return emitOpError()
           << "loop wrapper does not contain exactly two nested ops";

  Operation &firstOp = *region.op_begin();
  Operation &secondOp = *(std::next(region.op_begin()));

  if (!secondOp.hasTrait<OpTrait::IsTerminator>())
    return emitOpError()
           << "second nested op in loop wrapper is not a terminator";

  if (!::llvm::isa<LoopNestOp, LoopWrapperInterface>(firstOp))
    return emitOpError() << "first nested op in loop wrapper is not "
                            "another loop wrapper or `omp.loop_nest`";

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
        /*reduction_vars=*/ValueRange(), /*reduction_byref=*/nullptr,
        /*reduction_syms=*/nullptr, /*schedule_kind=*/nullptr,
        /*schedule_chunk=*/nullptr, /*schedule_mod=*/nullptr,
        /*schedule_simd=*/false);
  state.addAttributes(attributes);
}

void WsloopOp::build(OpBuilder &builder, OperationState &state,
                     const WsloopOperands &clauses) {
  MLIRContext *ctx = builder.getContext();
  // TODO: Store clauses in op: allocateVars, allocatorVars, privateVars,
  // privateSyms.
  WsloopOp::build(
      builder, state,
      /*allocate_vars=*/{}, /*allocator_vars=*/{}, clauses.linearVars,
      clauses.linearStepVars, clauses.nowait, clauses.order, clauses.orderMod,
      clauses.ordered, /*private_vars=*/{}, /*private_syms=*/nullptr,
      clauses.reductionVars,
      makeDenseBoolArrayAttr(ctx, clauses.reductionByref),
      makeArrayAttr(ctx, clauses.reductionSyms), clauses.scheduleKind,
      clauses.scheduleChunk, clauses.scheduleMod, clauses.scheduleSimd);
}

LogicalResult WsloopOp::verify() {
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

  return verifyReductionVarList(*this, getReductionSyms(), getReductionVars(),
                                getReductionByref());
}

//===----------------------------------------------------------------------===//
// Simd construct [2.9.3.1]
//===----------------------------------------------------------------------===//

void SimdOp::build(OpBuilder &builder, OperationState &state,
                   const SimdOperands &clauses) {
  MLIRContext *ctx = builder.getContext();
  // TODO Store clauses in op: linearVars, linearStepVars, privateVars,
  // privateSyms, reductionVars, reductionByref, reductionSyms.
  SimdOp::build(builder, state, clauses.alignedVars,
                makeArrayAttr(ctx, clauses.alignments), clauses.ifExpr,
                /*linear_vars=*/{}, /*linear_step_vars=*/{},
                clauses.nontemporalVars, clauses.order, clauses.orderMod,
                /*private_vars=*/{}, /*private_syms=*/nullptr,
                /*reduction_vars=*/{}, /*reduction_byref=*/nullptr,
                /*reduction_syms=*/nullptr, clauses.safelen, clauses.simdlen);
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

  if (getNestedWrapper())
    return emitOpError() << "must wrap an 'omp.loop_nest' directly";

  bool isCompositeChildLeaf =
      llvm::dyn_cast_if_present<LoopWrapperInterface>((*this)->getParentOp());

  if (!isComposite() && isCompositeChildLeaf)
    return emitError()
           << "'omp.composite' attribute missing from composite wrapper";

  if (isComposite() && !isCompositeChildLeaf)
    return emitError()
           << "'omp.composite' attribute present in non-composite wrapper";

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
                      makeArrayAttr(builder.getContext(), clauses.privateSyms));
}

LogicalResult DistributeOp::verify() {
  if (this->getDistScheduleChunkSize() && !this->getDistScheduleStatic())
    return emitOpError() << "chunk size set without "
                            "dist_schedule_static being present";

  if (getAllocateVars().size() != getAllocatorVars().size())
    return emitError(
        "expected equal sizes for allocate and allocator variables");

  if (LoopWrapperInterface nested = getNestedWrapper()) {
    if (!isComposite())
      return emitError()
             << "'omp.composite' attribute missing from composite wrapper";
    // Check for the allowed leaf constructs that may appear in a composite
    // construct directly after DISTRIBUTE.
    if (isa<WsloopOp>(nested)) {
      if (!llvm::dyn_cast_if_present<ParallelOp>((*this)->getParentOp()))
        return emitError() << "an 'omp.wsloop' nested wrapper is only allowed "
                              "when 'omp.parallel' is the direct parent";
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
  // TODO Store clauses in op: privateVars, privateSyms.
  TaskOp::build(builder, state, clauses.allocateVars, clauses.allocatorVars,
                makeArrayAttr(ctx, clauses.dependKinds), clauses.dependVars,
                clauses.final, clauses.ifExpr, clauses.inReductionVars,
                makeDenseBoolArrayAttr(ctx, clauses.inReductionByref),
                makeArrayAttr(ctx, clauses.inReductionSyms), clauses.mergeable,
                clauses.priority, /*private_vars=*/{}, /*private_syms=*/nullptr,
                clauses.untied);
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
  // TODO Store clauses in op: privateVars, privateSyms.
  TaskloopOp::build(
      builder, state, clauses.allocateVars, clauses.allocatorVars,
      clauses.final, clauses.grainsize, clauses.ifExpr, clauses.inReductionVars,
      makeDenseBoolArrayAttr(ctx, clauses.inReductionByref),
      makeArrayAttr(ctx, clauses.inReductionSyms), clauses.mergeable,
      clauses.nogroup, clauses.numTasks, clauses.priority, /*private_vars=*/{},
      /*private_syms=*/nullptr, clauses.reductionVars,
      makeDenseBoolArrayAttr(ctx, clauses.reductionByref),
      makeArrayAttr(ctx, clauses.reductionSyms), clauses.untied);
}

SmallVector<Value> TaskloopOp::getAllReductionVars() {
  SmallVector<Value> allReductionNvars(getInReductionVars().begin(),
                                       getInReductionVars().end());
  allReductionNvars.insert(allReductionNvars.end(), getReductionVars().begin(),
                           getReductionVars().end());
  return allReductionNvars;
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

  // Parse "inclusive" flag.
  if (succeeded(parser.parseOptionalKeyword("inclusive")))
    result.addAttribute("loop_inclusive",
                        UnitAttr::get(parser.getBuilder().getContext()));

  // Parse step values.
  SmallVector<OpAsmParser::UnresolvedOperand> steps;
  if (parser.parseKeyword("step") ||
      parser.parseOperandList(steps, ivs.size(), OpAsmParser::Delimiter::Paren))
    return failure();

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
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

void LoopNestOp::build(OpBuilder &builder, OperationState &state,
                       const LoopNestOperands &clauses) {
  LoopNestOp::build(builder, state, clauses.loopLowerBounds,
                    clauses.loopUpperBounds, clauses.loopSteps,
                    clauses.loopInclusive);
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

LogicalResult OrderedRegionOp::verify() {
  // TODO: The code generation for ordered simd directive is not supported yet.
  if (getParLevelSimd())
    return failure();

  return verifyOrderedParent(**this);
}

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

LogicalResult CancelOp::verify() {
  ClauseCancellationConstructType cct = getCancelDirective();
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
    auto loopOp = dyn_cast<LoopNestOp>(parentOp);
    auto wsloopOp = llvm::dyn_cast_if_present<WsloopOp>(
        loopOp ? loopOp->getParentOp() : nullptr);

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
    if (!(isa<SectionsOp>(parentOp) || isa<SectionOp>(parentOp))) {
      return emitOpError() << "cancel sections must appear "
                           << "inside a sections region";
    }
    if (isa_and_nonnull<SectionsOp>(parentOp->getParentOp()) &&
        cast<SectionsOp>(parentOp->getParentOp()).getNowaitAttr()) {
      return emitError() << "A sections construct that is canceled "
                         << "must not have a nowait clause";
    }
  }
  // TODO : Add more when we support taskgroup.
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
      (!isa<LoopNestOp>(parentOp) || !isa<WsloopOp>(parentOp->getParentOp()))) {
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

LogicalResult PrivateClauseOp::verify() {
  Type symType = getType();

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

    if (yieldedTypes.size() == 1 && yieldedTypes.front() == symType)
      return success();

    auto error = mlir::emitError(yieldOp.getLoc())
                 << "Invalid yielded value. Expected type: " << symType
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

  if (failed(verifyRegion(getAllocRegion(), /*expectedNumArgs=*/1, "alloc",
                          /*yieldsValue=*/true)))
    return failure();

  DataSharingClauseType dsType = getDataSharingType();

  if (dsType == DataSharingClauseType::Private && !getCopyRegion().empty())
    return emitError("`private` clauses require only an `alloc` region.");

  if (dsType == DataSharingClauseType::FirstPrivate && getCopyRegion().empty())
    return emitError(
        "`firstprivate` clauses require both `alloc` and `copy` regions.");

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

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOpsAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOpsTypes.cpp.inc"
