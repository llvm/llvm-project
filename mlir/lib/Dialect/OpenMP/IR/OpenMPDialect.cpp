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

#include "mlir/Dialect/OpenMP/OpenMPOpsDialect.cpp.inc"
#include "mlir/Dialect/OpenMP/OpenMPOpsEnums.cpp.inc"
#include "mlir/Dialect/OpenMP/OpenMPOpsInterfaces.cpp.inc"
#include "mlir/Dialect/OpenMP/OpenMPTypeInterfaces.cpp.inc"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::omp;

static ArrayAttr makeArrayAttr(MLIRContext *context,
                               llvm::ArrayRef<Attribute> attrs) {
  return attrs.empty() ? nullptr : ArrayAttr::get(context, attrs);
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

struct OpenMPDialectFoldInterface : public DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  bool shouldMaterializeInto(Region *region) const final {
    // Avoid folding constants across target regions
    return isa<TargetOp>(region->getParentOp());
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
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/OpenMP/OpenMPOpsTypes.cpp.inc"
      >();

  addInterface<OpenMPDialectFoldInterface>();
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
// Verifier for Nontemporal Clause
//===----------------------------------------------------------------------===//

static LogicalResult
verifyNontemporalClause(Operation *op, OperandRange nontemporalVariables) {

  // Check if each var is unique - OpenMP 5.0 -> 2.9.3.1 section
  DenseSet<Value> nontemporalItems;
  for (const auto &it : nontemporalVariables)
    if (!nontemporalItems.insert(it).second)
      return op->emitOpError() << "nontemporal variable used more than once";

  return success();
}

//===----------------------------------------------------------------------===//
// Parser, verifier and printer for Aligned Clause
//===----------------------------------------------------------------------===//
static LogicalResult
verifyAlignedClause(Operation *op, std::optional<ArrayAttr> alignmentValues,
                    OperandRange alignedVariables) {
  // Check if number of alignment values equals to number of aligned variables
  if (!alignedVariables.empty()) {
    if (!alignmentValues || alignmentValues->size() != alignedVariables.size())
      return op->emitOpError()
             << "expected as many alignment values as aligned variables";
  } else {
    if (alignmentValues)
      return op->emitOpError() << "unexpected alignment values attribute";
    return success();
  }

  // Check if each var is aligned only once - OpenMP 4.5 -> 2.8.1 section
  DenseSet<Value> alignedItems;
  for (auto it : alignedVariables)
    if (!alignedItems.insert(it).second)
      return op->emitOpError() << "aligned variable used more than once";

  if (!alignmentValues)
    return success();

  // Check if all alignment values are positive - OpenMP 4.5 -> 2.8.1 section
  for (unsigned i = 0; i < (*alignmentValues).size(); ++i) {
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>((*alignmentValues)[i])) {
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
static ParseResult parseAlignedClause(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &alignedItems,
    SmallVectorImpl<Type> &types, ArrayAttr &alignmentValues) {
  SmallVector<Attribute> alignmentVec;
  if (failed(parser.parseCommaSeparatedList([&]() {
        if (parser.parseOperand(alignedItems.emplace_back()) ||
            parser.parseColonType(types.emplace_back()) ||
            parser.parseArrow() ||
            parser.parseAttribute(alignmentVec.emplace_back())) {
          return failure();
        }
        return success();
      })))
    return failure();
  SmallVector<Attribute> alignments(alignmentVec.begin(), alignmentVec.end());
  alignmentValues = ArrayAttr::get(parser.getContext(), alignments);
  return success();
}

/// Print Aligned Clause
static void printAlignedClause(OpAsmPrinter &p, Operation *op,
                               ValueRange alignedVars,
                               TypeRange alignedVarTypes,
                               std::optional<ArrayAttr> alignmentValues) {
  for (unsigned i = 0; i < alignedVars.size(); ++i) {
    if (i != 0)
      p << ", ";
    p << alignedVars[i] << " : " << alignedVars[i].getType();
    p << " -> " << (*alignmentValues)[i];
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
    std::optional<OpAsmParser::UnresolvedOperand> &chunkSize, Type &chunkType) {
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
// Parser and printer for Order Clause
//===----------------------------------------------------------------------===//

// order ::= `order` `(` [order-modiﬁer ':'] concurrent `)`
// order-modiﬁer ::= reproducible | unconstrained
static ParseResult parseOrderClause(OpAsmParser &parser,
                                    ClauseOrderKindAttr &kindAttr,
                                    OrderModifierAttr &modifierAttr) {
  StringRef enumStr;
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&enumStr))
    return failure();
  if (std::optional<OrderModifier> enumValue =
          symbolizeOrderModifier(enumStr)) {
    modifierAttr = OrderModifierAttr::get(parser.getContext(), *enumValue);
    if (parser.parseOptionalColon())
      return failure();
    loc = parser.getCurrentLocation();
    if (parser.parseKeyword(&enumStr))
      return failure();
  }
  if (std::optional<ClauseOrderKind> enumValue =
          symbolizeClauseOrderKind(enumStr)) {
    kindAttr = ClauseOrderKindAttr::get(parser.getContext(), *enumValue);
    return success();
  }
  return parser.emitError(loc, "invalid clause value: '") << enumStr << "'";
}

static void printOrderClause(OpAsmPrinter &p, Operation *op,
                             ClauseOrderKindAttr kindAttr,
                             OrderModifierAttr modifierAttr) {
  if (modifierAttr)
    p << stringifyOrderModifier(modifierAttr.getValue()) << ":";
  if (kindAttr)
    p << stringifyClauseOrderKind(kindAttr.getValue());
}

//===----------------------------------------------------------------------===//
// Parser, printer and verifier for ReductionVarList
//===----------------------------------------------------------------------===//

static ParseResult parseClauseWithRegionArgs(
    OpAsmParser &parser, Region &region,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &types, DenseBoolArrayAttr &isByRef,
    ArrayAttr &symbols,
    SmallVectorImpl<OpAsmParser::Argument> &regionPrivateArgs) {
  SmallVector<SymbolRefAttr> reductionVec;
  SmallVector<bool> isByRefVec;
  unsigned regionArgOffset = regionPrivateArgs.size();

  if (failed(
          parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, [&]() {
            ParseResult optionalByref = parser.parseOptionalKeyword("byref");
            if (parser.parseAttribute(reductionVec.emplace_back()) ||
                parser.parseOperand(operands.emplace_back()) ||
                parser.parseArrow() ||
                parser.parseArgument(regionPrivateArgs.emplace_back()) ||
                parser.parseColonType(types.emplace_back()))
              return failure();
            isByRefVec.push_back(optionalByref.succeeded());
            return success();
          })))
    return failure();
  isByRef = DenseBoolArrayAttr::get(parser.getContext(), isByRefVec);

  auto *argsBegin = regionPrivateArgs.begin();
  MutableArrayRef argsSubrange(argsBegin + regionArgOffset,
                               argsBegin + regionArgOffset + types.size());
  for (auto [prv, type] : llvm::zip_equal(argsSubrange, types)) {
    prv.type = type;
  }
  SmallVector<Attribute> reductions(reductionVec.begin(), reductionVec.end());
  symbols = ArrayAttr::get(parser.getContext(), reductions);
  return success();
}

static void printClauseWithRegionArgs(OpAsmPrinter &p, Operation *op,
                                      ValueRange argsSubrange,
                                      StringRef clauseName, ValueRange operands,
                                      TypeRange types, DenseBoolArrayAttr byRef,
                                      ArrayAttr symbols) {
  if (!clauseName.empty())
    p << clauseName << "(";

  llvm::interleaveComma(llvm::zip_equal(symbols, operands, argsSubrange, types,
                                        byRef.asArrayRef()),
                        p, [&p](auto t) {
                          auto [sym, op, arg, type, isByRef] = t;
                          p << (isByRef ? "byref " : "") << sym << " " << op
                            << " -> " << arg << " : " << type;
                        });

  if (!clauseName.empty())
    p << ") ";
}

static ParseResult parseParallelRegion(
    OpAsmParser &parser, Region &region,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &reductionVarOperands,
    SmallVectorImpl<Type> &reductionVarTypes,
    DenseBoolArrayAttr &reductionByRef, ArrayAttr &reductionSymbols,
    llvm::SmallVectorImpl<OpAsmParser::UnresolvedOperand> &privateVarOperands,
    llvm::SmallVectorImpl<Type> &privateVarsTypes,
    ArrayAttr &privatizerSymbols) {
  llvm::SmallVector<OpAsmParser::Argument> regionPrivateArgs;

  if (succeeded(parser.parseOptionalKeyword("reduction"))) {
    if (failed(parseClauseWithRegionArgs(parser, region, reductionVarOperands,
                                         reductionVarTypes, reductionByRef,
                                         reductionSymbols, regionPrivateArgs)))
      return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("private"))) {
    auto privateByRef = DenseBoolArrayAttr::get(parser.getContext(), {});
    if (failed(parseClauseWithRegionArgs(parser, region, privateVarOperands,
                                         privateVarsTypes, privateByRef,
                                         privatizerSymbols, regionPrivateArgs)))
      return failure();
    if (llvm::any_of(privateByRef.asArrayRef(),
                     [](bool byref) { return byref; })) {
      parser.emitError(parser.getCurrentLocation(),
                       "private clause cannot have byref attributes");
      return failure();
    }
  }

  return parser.parseRegion(region, regionPrivateArgs);
}

static void printParallelRegion(OpAsmPrinter &p, Operation *op, Region &region,
                                ValueRange reductionVarOperands,
                                TypeRange reductionVarTypes,
                                DenseBoolArrayAttr reductionVarIsByRef,
                                ArrayAttr reductionSymbols,
                                ValueRange privateVarOperands,
                                TypeRange privateVarTypes,
                                ArrayAttr privatizerSymbols) {
  if (reductionSymbols) {
    auto *argsBegin = region.front().getArguments().begin();
    MutableArrayRef argsSubrange(argsBegin,
                                 argsBegin + reductionVarTypes.size());
    printClauseWithRegionArgs(p, op, argsSubrange, "reduction",
                              reductionVarOperands, reductionVarTypes,
                              reductionVarIsByRef, reductionSymbols);
  }

  if (privatizerSymbols) {
    auto *argsBegin = region.front().getArguments().begin();
    MutableArrayRef argsSubrange(argsBegin + reductionVarOperands.size(),
                                 argsBegin + reductionVarOperands.size() +
                                     privateVarTypes.size());
    mlir::SmallVector<bool> isByRefVec;
    isByRefVec.resize(privateVarTypes.size(), false);
    DenseBoolArrayAttr isByRef =
        DenseBoolArrayAttr::get(op->getContext(), isByRefVec);

    printClauseWithRegionArgs(p, op, argsSubrange, "private",
                              privateVarOperands, privateVarTypes, isByRef,
                              privatizerSymbols);
  }

  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

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
                                  std::optional<ArrayAttr> reductions) {
  for (unsigned i = 0, e = reductions->size(); i < e; ++i) {
    if (i != 0)
      p << ", ";
    p << (*reductions)[i] << " -> " << reductionVars[i] << " : "
      << reductionVars[i].getType();
  }
}

/// Verifies Reduction Clause
static LogicalResult
verifyReductionVarList(Operation *op, std::optional<ArrayAttr> reductions,
                       OperandRange reductionVars,
                       std::optional<ArrayRef<bool>> byRef = std::nullopt) {
  if (!reductionVars.empty()) {
    if (!reductions || reductions->size() != reductionVars.size())
      return op->emitOpError()
             << "expected as many reduction symbol references "
                "as reduction variables";
    if (mlir::isa<omp::WsloopOp, omp::ParallelOp>(op))
      assert(byRef);
    else
      assert(!byRef); // TODO: support byref reductions on other operations
    if (byRef && byRef->size() != reductionVars.size())
      return op->emitError() << "expected as many reduction variable by "
                                "reference attributes as reduction variables";
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
// Parser, printer and verifier for CopyPrivateVarList
//===----------------------------------------------------------------------===//

/// copyprivate-entry-list ::= copyprivate-entry
///                          | copyprivate-entry-list `,` copyprivate-entry
/// copyprivate-entry ::= ssa-id `->` symbol-ref `:` type
static ParseResult parseCopyPrivateVarList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &types, ArrayAttr &copyPrivateSymbols) {
  SmallVector<SymbolRefAttr> copyPrivateFuncsVec;
  if (failed(parser.parseCommaSeparatedList([&]() {
        if (parser.parseOperand(operands.emplace_back()) ||
            parser.parseArrow() ||
            parser.parseAttribute(copyPrivateFuncsVec.emplace_back()) ||
            parser.parseColonType(types.emplace_back()))
          return failure();
        return success();
      })))
    return failure();
  SmallVector<Attribute> copyPrivateFuncs(copyPrivateFuncsVec.begin(),
                                          copyPrivateFuncsVec.end());
  copyPrivateSymbols = ArrayAttr::get(parser.getContext(), copyPrivateFuncs);
  return success();
}

/// Print CopyPrivate clause
static void printCopyPrivateVarList(OpAsmPrinter &p, Operation *op,
                                    OperandRange copyPrivateVars,
                                    TypeRange copyPrivateTypes,
                                    std::optional<ArrayAttr> copyPrivateFuncs) {
  if (!copyPrivateFuncs.has_value())
    return;
  llvm::interleaveComma(
      llvm::zip(copyPrivateVars, *copyPrivateFuncs, copyPrivateTypes), p,
      [&](const auto &args) {
        p << std::get<0>(args) << " -> " << std::get<1>(args) << " : "
          << std::get<2>(args);
      });
}

/// Verifies CopyPrivate Clause
static LogicalResult
verifyCopyPrivateVarList(Operation *op, OperandRange copyPrivateVars,
                         std::optional<ArrayAttr> copyPrivateFuncs) {
  size_t copyPrivateFuncsSize =
      copyPrivateFuncs.has_value() ? copyPrivateFuncs->size() : 0;
  if (copyPrivateFuncsSize != copyPrivateVars.size())
    return op->emitOpError() << "inconsistent number of copyPrivate vars (= "
                             << copyPrivateVars.size()
                             << ") and functions (= " << copyPrivateFuncsSize
                             << "), both must be equal";
  if (!copyPrivateFuncs.has_value())
    return success();

  for (auto copyPrivateVarAndFunc :
       llvm::zip(copyPrivateVars, *copyPrivateFuncs)) {
    auto symbolRef =
        llvm::cast<SymbolRefAttr>(std::get<1>(copyPrivateVarAndFunc));
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

    Type varType = std::get<0>(copyPrivateVarAndFunc).getType();
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
                   SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
                   SmallVectorImpl<Type> &types, ArrayAttr &dependsArray) {
  SmallVector<ClauseTaskDependAttr> dependVec;
  if (failed(parser.parseCommaSeparatedList([&]() {
        StringRef keyword;
        if (parser.parseKeyword(&keyword) || parser.parseArrow() ||
            parser.parseOperand(operands.emplace_back()) ||
            parser.parseColonType(types.emplace_back()))
          return failure();
        if (std::optional<ClauseTaskDepend> keywordDepend =
                (symbolizeClauseTaskDepend(keyword)))
          dependVec.emplace_back(
              ClauseTaskDependAttr::get(parser.getContext(), *keywordDepend));
        else
          return failure();
        return success();
      })))
    return failure();
  SmallVector<Attribute> depends(dependVec.begin(), dependVec.end());
  dependsArray = ArrayAttr::get(parser.getContext(), depends);
  return success();
}

/// Print Depend clause
static void printDependVarList(OpAsmPrinter &p, Operation *op,
                               OperandRange dependVars, TypeRange dependTypes,
                               std::optional<ArrayAttr> depends) {

  for (unsigned i = 0, e = depends->size(); i < e; ++i) {
    if (i != 0)
      p << ", ";
    p << stringifyClauseTaskDepend(
             llvm::cast<mlir::omp::ClauseTaskDependAttr>((*depends)[i])
                 .getValue())
      << " -> " << dependVars[i] << " : " << dependTypes[i];
  }
}

/// Verifies Depend clause
static LogicalResult verifyDependVarList(Operation *op,
                                         std::optional<ArrayAttr> depends,
                                         OperandRange dependVars) {
  if (!dependVars.empty()) {
    if (!depends || depends->size() != dependVars.size())
      return op->emitOpError() << "expected as many depend values"
                                  " as depend variables";
  } else {
    if (depends && !depends->empty())
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
      p << membersIdx.getValues<
          int32_t>()[rowOffset + j];
      if ((j + 1) < shape[1])
        p << ",";
    }
    p << "]";

    if ((i + 1) < shape[0])
      p << ", ";
  }
}

static ParseResult
parseMapEntries(OpAsmParser &parser,
                SmallVectorImpl<OpAsmParser::UnresolvedOperand> &mapOperands,
                SmallVectorImpl<Type> &mapOperandTypes) {
  OpAsmParser::UnresolvedOperand arg;
  OpAsmParser::UnresolvedOperand blockArg;
  Type argType;
  auto parseEntries = [&]() -> ParseResult {
    if (parser.parseOperand(arg) || parser.parseArrow() ||
        parser.parseOperand(blockArg))
      return failure();
    mapOperands.push_back(arg);
    return success();
  };

  auto parseTypes = [&]() -> ParseResult {
    if (parser.parseType(argType))
      return failure();
    mapOperandTypes.push_back(argType);
    return success();
  };

  if (parser.parseCommaSeparatedList(parseEntries))
    return failure();

  if (parser.parseColon())
    return failure();

  if (parser.parseCommaSeparatedList(parseTypes))
    return failure();

  return success();
}

static void printMapEntries(OpAsmPrinter &p, Operation *op,
                            OperandRange mapOperands,
                            TypeRange mapOperandTypes) {
  auto &region = op->getRegion(0);
  unsigned argIndex = 0;

  for (const auto &mapOp : mapOperands) {
    const auto &blockArg = region.front().getArgument(argIndex);
    p << mapOp << " -> " << blockArg;
    argIndex++;
    if (argIndex < mapOperands.size())
      p << ", ";
  }
  p << " : ";

  argIndex = 0;
  for (const auto &mapType : mapOperandTypes) {
    p << mapType;
    argIndex++;
    if (argIndex < mapOperands.size())
      p << ", ";
  }
}

static ParseResult parsePrivateList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &privateOperands,
    SmallVectorImpl<Type> &privateOperandTypes, ArrayAttr &privatizerSymbols) {
  SmallVector<SymbolRefAttr> privateSymRefs;
  SmallVector<OpAsmParser::Argument> regionPrivateArgs;

  if (failed(parser.parseCommaSeparatedList([&]() {
        if (parser.parseAttribute(privateSymRefs.emplace_back()) ||
            parser.parseOperand(privateOperands.emplace_back()) ||
            parser.parseArrow() ||
            parser.parseArgument(regionPrivateArgs.emplace_back()) ||
            parser.parseColonType(privateOperandTypes.emplace_back()))
          return failure();
        return success();
      })))
    return failure();

  SmallVector<Attribute> privateSymAttrs(privateSymRefs.begin(),
                                         privateSymRefs.end());
  privatizerSymbols = ArrayAttr::get(parser.getContext(), privateSymAttrs);

  return success();
}

static void printPrivateList(OpAsmPrinter &p, Operation *op,
                             ValueRange privateVarOperands,
                             TypeRange privateVarTypes,
                             ArrayAttr privatizerSymbols) {
  // TODO: Remove target-specific logic from this function.
  auto targetOp = mlir::dyn_cast<mlir::omp::TargetOp>(op);
  assert(targetOp);

  auto &region = op->getRegion(0);
  auto *argsBegin = region.front().getArguments().begin();
  MutableArrayRef argsSubrange(argsBegin + targetOp.getMapOperands().size(),
                               argsBegin + targetOp.getMapOperands().size() +
                                   privateVarTypes.size());
  mlir::SmallVector<bool> isByRefVec;
  isByRefVec.resize(privateVarTypes.size(), false);
  DenseBoolArrayAttr isByRef =
      DenseBoolArrayAttr::get(op->getContext(), isByRefVec);

  printClauseWithRegionArgs(
      p, op, argsSubrange, /*clauseName=*/llvm::StringRef{}, privateVarOperands,
      privateVarTypes, isByRef, privatizerSymbols);
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
  p << typeCap.str();
}

static ParseResult parseCaptureType(OpAsmParser &parser,
                                    VariableCaptureKindAttr &mapCapture) {
  StringRef mapCaptureKey;
  if (parser.parseKeyword(&mapCaptureKey))
    return failure();

  if (mapCaptureKey == "This")
    mapCapture = mlir::omp::VariableCaptureKindAttr::get(
        parser.getContext(), mlir::omp::VariableCaptureKind::This);
  if (mapCaptureKey == "ByRef")
    mapCapture = mlir::omp::VariableCaptureKindAttr::get(
        parser.getContext(), mlir::omp::VariableCaptureKind::ByRef);
  if (mapCaptureKey == "ByCopy")
    mapCapture = mlir::omp::VariableCaptureKindAttr::get(
        parser.getContext(), mlir::omp::VariableCaptureKind::ByCopy);
  if (mapCaptureKey == "VLAType")
    mapCapture = mlir::omp::VariableCaptureKindAttr::get(
        parser.getContext(), mlir::omp::VariableCaptureKind::VLAType);

  return success();
}

static LogicalResult verifyMapClause(Operation *op, OperandRange mapOperands) {
  llvm::DenseSet<mlir::TypedValue<mlir::omp::PointerLikeType>> updateToVars;
  llvm::DenseSet<mlir::TypedValue<mlir::omp::PointerLikeType>> updateFromVars;

  for (auto mapOp : mapOperands) {
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
                         const TargetDataClauseOps &clauses) {
  TargetDataOp::build(builder, state, clauses.ifVar, clauses.deviceVar,
                      clauses.useDevicePtrVars, clauses.useDeviceAddrVars,
                      clauses.mapVars);
}

LogicalResult TargetDataOp::verify() {
  if (getMapOperands().empty() && getUseDevicePtr().empty() &&
      getUseDeviceAddr().empty()) {
    return ::emitError(this->getLoc(), "At least one of map, useDevicePtr, or "
                                       "useDeviceAddr operand must be present");
  }
  return verifyMapClause(*this, getMapOperands());
}

//===----------------------------------------------------------------------===//
// TargetEnterDataOp
//===----------------------------------------------------------------------===//

void TargetEnterDataOp::build(
    OpBuilder &builder, OperationState &state,
    const TargetEnterExitUpdateDataClauseOps &clauses) {
  MLIRContext *ctx = builder.getContext();
  TargetEnterDataOp::build(builder, state, clauses.ifVar, clauses.deviceVar,
                           makeArrayAttr(ctx, clauses.dependTypeAttrs),
                           clauses.dependVars, clauses.nowaitAttr,
                           clauses.mapVars);
}

LogicalResult TargetEnterDataOp::verify() {
  LogicalResult verifyDependVars =
      verifyDependVarList(*this, getDepends(), getDependVars());
  return failed(verifyDependVars) ? verifyDependVars
                                  : verifyMapClause(*this, getMapOperands());
}

//===----------------------------------------------------------------------===//
// TargetExitDataOp
//===----------------------------------------------------------------------===//

void TargetExitDataOp::build(
    OpBuilder &builder, OperationState &state,
    const TargetEnterExitUpdateDataClauseOps &clauses) {
  MLIRContext *ctx = builder.getContext();
  TargetExitDataOp::build(builder, state, clauses.ifVar, clauses.deviceVar,
                          makeArrayAttr(ctx, clauses.dependTypeAttrs),
                          clauses.dependVars, clauses.nowaitAttr,
                          clauses.mapVars);
}

LogicalResult TargetExitDataOp::verify() {
  LogicalResult verifyDependVars =
      verifyDependVarList(*this, getDepends(), getDependVars());
  return failed(verifyDependVars) ? verifyDependVars
                                  : verifyMapClause(*this, getMapOperands());
}

//===----------------------------------------------------------------------===//
// TargetUpdateOp
//===----------------------------------------------------------------------===//

void TargetUpdateOp::build(OpBuilder &builder, OperationState &state,
                           const TargetEnterExitUpdateDataClauseOps &clauses) {
  MLIRContext *ctx = builder.getContext();
  TargetUpdateOp::build(builder, state, clauses.ifVar, clauses.deviceVar,
                        makeArrayAttr(ctx, clauses.dependTypeAttrs),
                        clauses.dependVars, clauses.nowaitAttr,
                        clauses.mapVars);
}

LogicalResult TargetUpdateOp::verify() {
  LogicalResult verifyDependVars =
      verifyDependVarList(*this, getDepends(), getDependVars());
  return failed(verifyDependVars) ? verifyDependVars
                                  : verifyMapClause(*this, getMapOperands());
}

//===----------------------------------------------------------------------===//
// TargetOp
//===----------------------------------------------------------------------===//

void TargetOp::build(OpBuilder &builder, OperationState &state,
                     const TargetClauseOps &clauses) {
  MLIRContext *ctx = builder.getContext();
  // TODO Store clauses in op: allocateVars, allocatorVars, inReductionVars,
  // inReductionDeclSymbols, reductionVars, reductionByRefAttr,
  // reductionDeclSymbols.
  TargetOp::build(
      builder, state, clauses.ifVar, clauses.deviceVar, clauses.threadLimitVar,
      makeArrayAttr(ctx, clauses.dependTypeAttrs), clauses.dependVars,
      clauses.nowaitAttr, clauses.isDevicePtrVars, clauses.hasDeviceAddrVars,
      clauses.mapVars, clauses.privateVars,
      makeArrayAttr(ctx, clauses.privatizers));
}

LogicalResult TargetOp::verify() {
  LogicalResult verifyDependVars =
      verifyDependVarList(*this, getDepends(), getDependVars());
  return failed(verifyDependVars) ? verifyDependVars
                                  : verifyMapClause(*this, getMapOperands());
}

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

void ParallelOp::build(OpBuilder &builder, OperationState &state,
                       ArrayRef<NamedAttribute> attributes) {
  ParallelOp::build(
      builder, state, /*if_expr_var=*/nullptr, /*num_threads_var=*/nullptr,
      /*allocate_vars=*/ValueRange(), /*allocators_vars=*/ValueRange(),
      /*reduction_vars=*/ValueRange(), /*reduction_vars_byref=*/nullptr,
      /*reductions=*/nullptr,
      /*proc_bind_val=*/nullptr, /*private_vars=*/ValueRange(),
      /*privatizers=*/nullptr);
  state.addAttributes(attributes);
}

void ParallelOp::build(OpBuilder &builder, OperationState &state,
                       const ParallelClauseOps &clauses) {
  MLIRContext *ctx = builder.getContext();

  ParallelOp::build(builder, state, clauses.ifVar, clauses.numThreadsVar,
                    clauses.allocateVars, clauses.allocatorVars,
                    clauses.reductionVars,
                    DenseBoolArrayAttr::get(ctx, clauses.reduceVarByRef),
                    makeArrayAttr(ctx, clauses.reductionDeclSymbols),
                    clauses.procBindKindAttr, clauses.privateVars,
                    makeArrayAttr(ctx, clauses.privatizers));
}

template <typename OpType>
static LogicalResult verifyPrivateVarList(OpType &op) {
  auto privateVars = op.getPrivateVars();
  auto privatizers = op.getPrivatizersAttr();

  if (privateVars.empty() && (privatizers == nullptr || privatizers.empty()))
    return success();

  auto numPrivateVars = privateVars.size();
  auto numPrivatizers = (privatizers == nullptr) ? 0 : privatizers.size();

  if (numPrivateVars != numPrivatizers)
    return op.emitError() << "inconsistent number of private variables and "
                             "privatizer op symbols, private vars: "
                          << numPrivateVars
                          << " vs. privatizer op symbols: " << numPrivatizers;

  for (auto privateVarInfo : llvm::zip_equal(privateVars, privatizers)) {
    Type varType = std::get<0>(privateVarInfo).getType();
    SymbolRefAttr privatizerSym =
        cast<SymbolRefAttr>(std::get<1>(privateVarInfo));
    PrivateClauseOp privatizerOp =
        SymbolTable::lookupNearestSymbolFrom<PrivateClauseOp>(op,
                                                              privatizerSym);

    if (privatizerOp == nullptr)
      return op.emitError() << "failed to lookup privatizer op with symbol: '"
                            << privatizerSym << "'";

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
  // Check that it is a valid loop wrapper if it's taking that role.
  if (isa<DistributeOp>((*this)->getParentOp())) {
    if (!isWrapper())
      return emitOpError() << "must take a loop wrapper role if nested inside "
                              "of 'omp.distribute'";

    if (LoopWrapperInterface nested = getNestedWrapper()) {
      // Check for the allowed leaf constructs that may appear in a composite
      // construct directly after PARALLEL.
      if (!isa<WsloopOp>(nested))
        return emitError() << "only supported nested wrapper is 'omp.wsloop'";
    } else {
      return emitOpError() << "must not wrap an 'omp.loop_nest' directly";
    }
  }

  if (getAllocateVars().size() != getAllocatorsVars().size())
    return emitError(
        "expected equal sizes for allocate and allocator variables");

  if (failed(verifyPrivateVarList(*this)))
    return failure();

  return verifyReductionVarList(*this, getReductions(), getReductionVars(),
                                getReductionVarsByref());
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
                    const TeamsClauseOps &clauses) {
  MLIRContext *ctx = builder.getContext();
  // TODO Store clauses in op: reductionByRefAttr, privateVars, privatizers.
  TeamsOp::build(builder, state, clauses.numTeamsLowerVar,
                 clauses.numTeamsUpperVar, clauses.ifVar,
                 clauses.threadLimitVar, clauses.allocateVars,
                 clauses.allocatorVars, clauses.reductionVars,
                 makeArrayAttr(ctx, clauses.reductionDeclSymbols));
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
  if (getAllocateVars().size() != getAllocatorsVars().size())
    return emitError(
        "expected equal sizes for allocate and allocator variables");

  return verifyReductionVarList(*this, getReductions(), getReductionVars());
}

//===----------------------------------------------------------------------===//
// SectionsOp
//===----------------------------------------------------------------------===//

void SectionsOp::build(OpBuilder &builder, OperationState &state,
                       const SectionsClauseOps &clauses) {
  MLIRContext *ctx = builder.getContext();
  // TODO Store clauses in op: reductionByRefAttr, privateVars, privatizers.
  SectionsOp::build(builder, state, clauses.reductionVars,
                    makeArrayAttr(ctx, clauses.reductionDeclSymbols),
                    clauses.allocateVars, clauses.allocatorVars,
                    clauses.nowaitAttr);
}

LogicalResult SectionsOp::verify() {
  if (getAllocateVars().size() != getAllocatorsVars().size())
    return emitError(
        "expected equal sizes for allocate and allocator variables");

  return verifyReductionVarList(*this, getReductions(), getReductionVars());
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
                     const SingleClauseOps &clauses) {
  MLIRContext *ctx = builder.getContext();
  // TODO Store clauses in op: privateVars, privatizers.
  SingleOp::build(builder, state, clauses.allocateVars, clauses.allocatorVars,
                  clauses.copyprivateVars,
                  makeArrayAttr(ctx, clauses.copyprivateFuncs),
                  clauses.nowaitAttr);
}

LogicalResult SingleOp::verify() {
  // Check for allocate clause restrictions
  if (getAllocateVars().size() != getAllocatorsVars().size())
    return emitError(
        "expected equal sizes for allocate and allocator variables");

  return verifyCopyPrivateVarList(*this, getCopyprivateVars(),
                                  getCopyprivateFuncs());
}

//===----------------------------------------------------------------------===//
// WsloopOp
//===----------------------------------------------------------------------===//

ParseResult
parseWsloop(OpAsmParser &parser, Region &region,
            SmallVectorImpl<OpAsmParser::UnresolvedOperand> &reductionOperands,
            SmallVectorImpl<Type> &reductionTypes,
            DenseBoolArrayAttr &reductionByRef, ArrayAttr &reductionSymbols) {
  // Parse an optional reduction clause
  llvm::SmallVector<OpAsmParser::Argument> privates;
  if (succeeded(parser.parseOptionalKeyword("reduction"))) {
    if (failed(parseClauseWithRegionArgs(parser, region, reductionOperands,
                                         reductionTypes, reductionByRef,
                                         reductionSymbols, privates)))
      return failure();
  }
  return parser.parseRegion(region, privates);
}

void printWsloop(OpAsmPrinter &p, Operation *op, Region &region,
                 ValueRange reductionOperands, TypeRange reductionTypes,
                 DenseBoolArrayAttr isByRef, ArrayAttr reductionSymbols) {
  if (reductionSymbols) {
    auto reductionArgs = region.front().getArguments();
    printClauseWithRegionArgs(p, op, reductionArgs, "reduction",
                              reductionOperands, reductionTypes, isByRef,
                              reductionSymbols);
  }
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

void WsloopOp::build(OpBuilder &builder, OperationState &state,
                     ArrayRef<NamedAttribute> attributes) {
  build(builder, state, /*linear_vars=*/ValueRange(),
        /*linear_step_vars=*/ValueRange(), /*reduction_vars=*/ValueRange(),
        /*reduction_vars_byref=*/nullptr,
        /*reductions=*/nullptr, /*schedule_val=*/nullptr,
        /*schedule_chunk_var=*/nullptr, /*schedule_modifier=*/nullptr,
        /*simd_modifier=*/false, /*nowait=*/false,
        /*ordered_val=*/nullptr, /*order_val=*/nullptr,
        /*order_modifier=*/nullptr);
  state.addAttributes(attributes);
}

void WsloopOp::build(OpBuilder &builder, OperationState &state,
                     const WsloopClauseOps &clauses) {
  MLIRContext *ctx = builder.getContext();
  // TODO: Store clauses in op: allocateVars, allocatorVars, privateVars,
  // privatizers.
  WsloopOp::build(builder, state, clauses.linearVars, clauses.linearStepVars,
                  clauses.reductionVars,
                  DenseBoolArrayAttr::get(ctx, clauses.reduceVarByRef),
                  makeArrayAttr(ctx, clauses.reductionDeclSymbols),
                  clauses.scheduleValAttr, clauses.scheduleChunkVar,
                  clauses.scheduleModAttr, clauses.scheduleSimdAttr,
                  clauses.nowaitAttr, clauses.orderedAttr, clauses.orderAttr,
                  clauses.orderModAttr);
}

LogicalResult WsloopOp::verify() {
  if (!isWrapper())
    return emitOpError() << "must be a loop wrapper";

  if (LoopWrapperInterface nested = getNestedWrapper()) {
    // Check for the allowed leaf constructs that may appear in a composite
    // construct directly after DO/FOR.
    if (!isa<SimdOp>(nested))
      return emitError() << "only supported nested wrapper is 'omp.simd'";
  }

  return verifyReductionVarList(*this, getReductions(), getReductionVars(),
                                getReductionVarsByref());
}

//===----------------------------------------------------------------------===//
// Simd construct [2.9.3.1]
//===----------------------------------------------------------------------===//

void SimdOp::build(OpBuilder &builder, OperationState &state,
                   const SimdClauseOps &clauses) {
  MLIRContext *ctx = builder.getContext();
  // TODO Store clauses in op: privateVars, reductionByRefAttr, reductionVars,
  // privatizers, reductionDeclSymbols.
  SimdOp::build(builder, state, clauses.alignedVars,
                makeArrayAttr(ctx, clauses.alignmentAttrs), clauses.ifVar,
                clauses.nontemporalVars, clauses.orderAttr,
                clauses.orderModAttr, clauses.simdlenAttr, clauses.safelenAttr);
}

LogicalResult SimdOp::verify() {
  if (getSimdlen().has_value() && getSafelen().has_value() &&
      getSimdlen().value() > getSafelen().value())
    return emitOpError()
           << "simdlen clause and safelen clause are both present, but the "
              "simdlen value is not less than or equal to safelen value";

  if (verifyAlignedClause(*this, getAlignmentValues(), getAlignedVars())
          .failed())
    return failure();

  if (verifyNontemporalClause(*this, getNontemporalVars()).failed())
    return failure();

  if (!isWrapper())
    return emitOpError() << "must be a loop wrapper";

  if (getNestedWrapper())
    return emitOpError() << "must wrap an 'omp.loop_nest' directly";

  return success();
}

//===----------------------------------------------------------------------===//
// Distribute construct [2.9.4.1]
//===----------------------------------------------------------------------===//

void DistributeOp::build(OpBuilder &builder, OperationState &state,
                         const DistributeClauseOps &clauses) {
  // TODO Store clauses in op: privateVars, privatizers.
  DistributeOp::build(builder, state, clauses.distScheduleStaticAttr,
                      clauses.distScheduleChunkSizeVar, clauses.allocateVars,
                      clauses.allocatorVars, clauses.orderAttr,
                      clauses.orderModAttr);
}

LogicalResult DistributeOp::verify() {
  if (this->getChunkSize() && !this->getDistScheduleStatic())
    return emitOpError() << "chunk size set without "
                            "dist_schedule_static being present";

  if (getAllocateVars().size() != getAllocatorsVars().size())
    return emitError(
        "expected equal sizes for allocate and allocator variables");

  if (!isWrapper())
    return emitOpError() << "must be a loop wrapper";

  if (LoopWrapperInterface nested = getNestedWrapper()) {
    // Check for the allowed leaf constructs that may appear in a composite
    // construct directly after DISTRIBUTE.
    if (!isa<ParallelOp, SimdOp>(nested))
      return emitError() << "only supported nested wrappers are 'omp.parallel' "
                            "and 'omp.simd'";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DeclareReductionOp
//===----------------------------------------------------------------------===//

static ParseResult parseAtomicReductionRegion(OpAsmParser &parser,
                                              Region &region) {
  if (parser.parseOptionalKeyword("atomic"))
    return success();
  return parser.parseRegion(region);
}

static void printAtomicReductionRegion(OpAsmPrinter &printer,
                                       DeclareReductionOp op, Region &region) {
  if (region.empty())
    return;
  printer << "atomic ";
  printer.printRegion(region);
}

static ParseResult parseCleanupReductionRegion(OpAsmParser &parser,
                                               Region &region) {
  if (parser.parseOptionalKeyword("cleanup"))
    return success();
  return parser.parseRegion(region);
}

static void printCleanupReductionRegion(OpAsmPrinter &printer,
                                        DeclareReductionOp op, Region &region) {
  if (region.empty())
    return;
  printer << "cleanup ";
  printer.printRegion(region);
}

LogicalResult DeclareReductionOp::verifyRegions() {
  if (getInitializerRegion().empty())
    return emitOpError() << "expects non-empty initializer region";
  Block &initializerEntryBlock = getInitializerRegion().front();
  if (initializerEntryBlock.getNumArguments() != 1 ||
      initializerEntryBlock.getArgument(0).getType() != getType()) {
    return emitOpError() << "expects initializer region with one argument "
                            "of the reduction type";
  }

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
                   const TaskClauseOps &clauses) {
  MLIRContext *ctx = builder.getContext();
  // TODO Store clauses in op: privateVars, privatizers.
  TaskOp::build(
      builder, state, clauses.ifVar, clauses.finalVar, clauses.untiedAttr,
      clauses.mergeableAttr, clauses.inReductionVars,
      makeArrayAttr(ctx, clauses.inReductionDeclSymbols), clauses.priorityVar,
      makeArrayAttr(ctx, clauses.dependTypeAttrs), clauses.dependVars,
      clauses.allocateVars, clauses.allocatorVars);
}

LogicalResult TaskOp::verify() {
  LogicalResult verifyDependVars =
      verifyDependVarList(*this, getDepends(), getDependVars());
  return failed(verifyDependVars)
             ? verifyDependVars
             : verifyReductionVarList(*this, getInReductions(),
                                      getInReductionVars());
}

//===----------------------------------------------------------------------===//
// TaskgroupOp
//===----------------------------------------------------------------------===//

void TaskgroupOp::build(OpBuilder &builder, OperationState &state,
                        const TaskgroupClauseOps &clauses) {
  MLIRContext *ctx = builder.getContext();
  TaskgroupOp::build(builder, state, clauses.taskReductionVars,
                     makeArrayAttr(ctx, clauses.taskReductionDeclSymbols),
                     clauses.allocateVars, clauses.allocatorVars);
}

LogicalResult TaskgroupOp::verify() {
  return verifyReductionVarList(*this, getTaskReductions(),
                                getTaskReductionVars());
}

//===----------------------------------------------------------------------===//
// TaskloopOp
//===----------------------------------------------------------------------===//

void TaskloopOp::build(OpBuilder &builder, OperationState &state,
                       const TaskloopClauseOps &clauses) {
  MLIRContext *ctx = builder.getContext();
  // TODO Store clauses in op: reductionByRefAttr, privateVars, privatizers.
  TaskloopOp::build(
      builder, state, clauses.ifVar, clauses.finalVar, clauses.untiedAttr,
      clauses.mergeableAttr, clauses.inReductionVars,
      makeArrayAttr(ctx, clauses.inReductionDeclSymbols), clauses.reductionVars,
      makeArrayAttr(ctx, clauses.reductionDeclSymbols), clauses.priorityVar,
      clauses.allocateVars, clauses.allocatorVars, clauses.grainsizeVar,
      clauses.numTasksVar, clauses.nogroupAttr);
}

SmallVector<Value> TaskloopOp::getAllReductionVars() {
  SmallVector<Value> allReductionNvars(getInReductionVars().begin(),
                                       getInReductionVars().end());
  allReductionNvars.insert(allReductionNvars.end(), getReductionVars().begin(),
                           getReductionVars().end());
  return allReductionNvars;
}

LogicalResult TaskloopOp::verify() {
  if (getAllocateVars().size() != getAllocatorsVars().size())
    return emitError(
        "expected equal sizes for allocate and allocator variables");
  if (failed(
          verifyReductionVarList(*this, getReductions(), getReductionVars())) ||
      failed(verifyReductionVarList(*this, getInReductions(),
                                    getInReductionVars())))
    return failure();

  if (!getReductionVars().empty() && getNogroup())
    return emitError("if a reduction clause is present on the taskloop "
                     "directive, the nogroup clause must not be specified");
  for (auto var : getReductionVars()) {
    if (llvm::is_contained(getInReductionVars(), var))
      return emitError("the same list item cannot appear in both a reduction "
                       "and an in_reduction clause");
  }

  if (getGrainSize() && getNumTasks()) {
    return emitError(
        "the grainsize clause and num_tasks clause are mutually exclusive and "
        "may not appear on the same taskloop directive");
  }

  if (!isWrapper())
    return emitOpError() << "must be a loop wrapper";

  if (LoopWrapperInterface nested = getNestedWrapper()) {
    // Check for the allowed leaf constructs that may appear in a composite
    // construct directly after TASKLOOP.
    if (!isa<SimdOp>(nested))
      return emitError() << "only supported nested wrapper is 'omp.simd'";
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
    result.addAttribute("inclusive",
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
  p << " (" << args << ") : " << args[0].getType() << " = (" << getLowerBound()
    << ") to (" << getUpperBound() << ") ";
  if (getInclusive())
    p << "inclusive ";
  p << "step (" << getStep() << ") ";
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

void LoopNestOp::build(OpBuilder &builder, OperationState &state,
                       const LoopNestClauseOps &clauses) {
  LoopNestOp::build(builder, state, clauses.loopLBVar, clauses.loopUBVar,
                    clauses.loopStepVar, clauses.loopInclusiveAttr);
}

LogicalResult LoopNestOp::verify() {
  if (getLowerBound().empty())
    return emitOpError() << "must represent at least one loop";

  if (getLowerBound().size() != getIVs().size())
    return emitOpError() << "number of range arguments and IVs do not match";

  for (auto [lb, iv] : llvm::zip_equal(getLowerBound(), getIVs())) {
    if (lb.getType() != iv.getType())
      return emitOpError()
             << "range argument type does not match corresponding IV type";
  }

  auto wrapper =
      llvm::dyn_cast_if_present<LoopWrapperInterface>((*this)->getParentOp());

  if (!wrapper || !wrapper.isWrapper())
    return emitOpError() << "expects parent op to be a valid loop wrapper";

  return success();
}

void LoopNestOp::gatherWrappers(
    SmallVectorImpl<LoopWrapperInterface> &wrappers) {
  Operation *parent = (*this)->getParentOp();
  while (auto wrapper =
             llvm::dyn_cast_if_present<LoopWrapperInterface>(parent)) {
    if (!wrapper.isWrapper())
      break;
    wrappers.push_back(wrapper);
    parent = parent->getParentOp();
  }
}

//===----------------------------------------------------------------------===//
// Critical construct (2.17.1)
//===----------------------------------------------------------------------===//

void CriticalDeclareOp::build(OpBuilder &builder, OperationState &state,
                              const CriticalClauseOps &clauses) {
  CriticalDeclareOp::build(builder, state, clauses.nameAttr, clauses.hintAttr);
}

LogicalResult CriticalDeclareOp::verify() {
  return verifySynchronizationHint(*this, getHintVal());
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
    IntegerAttr orderedAttr = wsloopOp.getOrderedValAttr();
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
                      const OrderedOpClauseOps &clauses) {
  OrderedOp::build(builder, state, clauses.doacrossDependTypeAttr,
                   clauses.doacrossNumLoopsAttr, clauses.doacrossVectorVars);
}

LogicalResult OrderedOp::verify() {
  if (failed(verifyOrderedParent(**this)))
    return failure();

  auto wrapper = (*this)->getParentOfType<WsloopOp>();
  if (!wrapper || *wrapper.getOrderedVal() != *getNumLoopsVal())
    return emitOpError() << "number of variables in depend clause does not "
                         << "match number of iteration variables in the "
                         << "doacross loop";

  return success();
}

void OrderedRegionOp::build(OpBuilder &builder, OperationState &state,
                            const OrderedRegionClauseOps &clauses) {
  OrderedRegionOp::build(builder, state, clauses.parLevelSimdAttr);
}

LogicalResult OrderedRegionOp::verify() {
  // TODO: The code generation for ordered simd directive is not supported yet.
  if (getSimd())
    return failure();

  return verifyOrderedParent(**this);
}

//===----------------------------------------------------------------------===//
// TaskwaitOp
//===----------------------------------------------------------------------===//

void TaskwaitOp::build(OpBuilder &builder, OperationState &state,
                       const TaskwaitClauseOps &clauses) {
  // TODO Store clauses in op: dependTypeAttrs, dependVars, nowaitAttr.
  TaskwaitOp::build(builder, state);
}

//===----------------------------------------------------------------------===//
// Verifier for AtomicReadOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicReadOp::verify() {
  if (verifyCommon().failed())
    return mlir::failure();

  if (auto mo = getMemoryOrderVal()) {
    if (*mo == ClauseMemoryOrderKind::Acq_rel ||
        *mo == ClauseMemoryOrderKind::Release) {
      return emitError(
          "memory-order must not be acq_rel or release for atomic reads");
    }
  }
  return verifySynchronizationHint(*this, getHintVal());
}

//===----------------------------------------------------------------------===//
// Verifier for AtomicWriteOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicWriteOp::verify() {
  if (verifyCommon().failed())
    return mlir::failure();

  if (auto mo = getMemoryOrderVal()) {
    if (*mo == ClauseMemoryOrderKind::Acq_rel ||
        *mo == ClauseMemoryOrderKind::Acquire) {
      return emitError(
          "memory-order must not be acq_rel or acquire for atomic writes");
    }
  }
  return verifySynchronizationHint(*this, getHintVal());
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
    rewriter.replaceOpWithNewOp<AtomicWriteOp>(op, op.getX(), writeVal,
                                               op.getHintValAttr(),
                                               op.getMemoryOrderValAttr());
    return success();
  }
  return failure();
}

LogicalResult AtomicUpdateOp::verify() {
  if (verifyCommon().failed())
    return mlir::failure();

  if (auto mo = getMemoryOrderVal()) {
    if (*mo == ClauseMemoryOrderKind::Acq_rel ||
        *mo == ClauseMemoryOrderKind::Acquire) {
      return emitError(
          "memory-order must not be acq_rel or acquire for atomic updates");
    }
  }

  return verifySynchronizationHint(*this, getHintVal());
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
  return verifySynchronizationHint(*this, getHintVal());
}

LogicalResult AtomicCaptureOp::verifyRegions() {
  if (verifyRegionsCommon().failed())
    return mlir::failure();

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
  ClauseCancellationConstructType cct = getCancellationConstructTypeVal();
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
    if (wsloopOp.getOrderedValAttr()) {
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
// Verifier for CancelOp
//===----------------------------------------------------------------------===//

LogicalResult CancellationPointOp::verify() {
  ClauseCancellationConstructType cct = getCancellationConstructTypeVal();
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

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOpsAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOpsTypes.cpp.inc"
