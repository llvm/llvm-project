//===- ByteCode.cpp - Pattern ByteCode Interpreter ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MLIR to byte-code generation and the interpreter.
//
//===----------------------------------------------------------------------===//

#include "ByteCode.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "pdl-bytecode"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// PDLByteCodePattern
//===----------------------------------------------------------------------===//

PDLByteCodePattern PDLByteCodePattern::create(pdl_interp::RecordMatchOp matchOp,
                                              ByteCodeAddr rewriterAddr) {
  SmallVector<StringRef, 8> generatedOps;
  if (ArrayAttr generatedOpsAttr = matchOp.generatedOpsAttr())
    generatedOps =
        llvm::to_vector<8>(generatedOpsAttr.getAsValueRange<StringAttr>());

  PatternBenefit benefit = matchOp.benefit();
  MLIRContext *ctx = matchOp.getContext();

  // Check to see if this is pattern matches a specific operation type.
  if (Optional<StringRef> rootKind = matchOp.rootKind())
    return PDLByteCodePattern(rewriterAddr, *rootKind, generatedOps, benefit,
                              ctx);
  return PDLByteCodePattern(rewriterAddr, generatedOps, benefit, ctx,
                            MatchAnyOpTypeTag());
}

//===----------------------------------------------------------------------===//
// PDLByteCodeMutableState
//===----------------------------------------------------------------------===//

/// Set the new benefit for a bytecode pattern. The `patternIndex` corresponds
/// to the position of the pattern within the range returned by
/// `PDLByteCode::getPatterns`.
void PDLByteCodeMutableState::updatePatternBenefit(unsigned patternIndex,
                                                   PatternBenefit benefit) {
  currentPatternBenefits[patternIndex] = benefit;
}

//===----------------------------------------------------------------------===//
// Bytecode OpCodes
//===----------------------------------------------------------------------===//

namespace {
enum OpCode : ByteCodeField {
  /// Apply an externally registered constraint.
  ApplyConstraint,
  /// Apply an externally registered rewrite.
  ApplyRewrite,
  /// Check if two generic values are equal.
  AreEqual,
  /// Unconditional branch.
  Branch,
  /// Compare the operand count of an operation with a constant.
  CheckOperandCount,
  /// Compare the name of an operation with a constant.
  CheckOperationName,
  /// Compare the result count of an operation with a constant.
  CheckResultCount,
  /// Create an operation.
  CreateOperation,
  /// Erase an operation.
  EraseOp,
  /// Terminate a matcher or rewrite sequence.
  Finalize,
  /// Get a specific attribute of an operation.
  GetAttribute,
  /// Get the type of an attribute.
  GetAttributeType,
  /// Get the defining operation of a value.
  GetDefiningOp,
  /// Get a specific operand of an operation.
  GetOperand0,
  GetOperand1,
  GetOperand2,
  GetOperand3,
  GetOperandN,
  /// Get a specific result of an operation.
  GetResult0,
  GetResult1,
  GetResult2,
  GetResult3,
  GetResultN,
  /// Get the type of a value.
  GetValueType,
  /// Check if a generic value is not null.
  IsNotNull,
  /// Record a successful pattern match.
  RecordMatch,
  /// Replace an operation.
  ReplaceOp,
  /// Compare an attribute with a set of constants.
  SwitchAttribute,
  /// Compare the operand count of an operation with a set of constants.
  SwitchOperandCount,
  /// Compare the name of an operation with a set of constants.
  SwitchOperationName,
  /// Compare the result count of an operation with a set of constants.
  SwitchResultCount,
  /// Compare a type with a set of constants.
  SwitchType,
};

enum class PDLValueKind { Attribute, Operation, Type, Value };
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ByteCode Generation
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Generator

namespace {
struct ByteCodeWriter;

/// This class represents the main generator for the pattern bytecode.
class Generator {
public:
  Generator(MLIRContext *ctx, std::vector<const void *> &uniquedData,
            SmallVectorImpl<ByteCodeField> &matcherByteCode,
            SmallVectorImpl<ByteCodeField> &rewriterByteCode,
            SmallVectorImpl<PDLByteCodePattern> &patterns,
            ByteCodeField &maxValueMemoryIndex,
            llvm::StringMap<PDLConstraintFunction> &constraintFns,
            llvm::StringMap<PDLRewriteFunction> &rewriteFns)
      : ctx(ctx), uniquedData(uniquedData), matcherByteCode(matcherByteCode),
        rewriterByteCode(rewriterByteCode), patterns(patterns),
        maxValueMemoryIndex(maxValueMemoryIndex) {
    for (auto it : llvm::enumerate(constraintFns))
      constraintToMemIndex.try_emplace(it.value().first(), it.index());
    for (auto it : llvm::enumerate(rewriteFns))
      externalRewriterToMemIndex.try_emplace(it.value().first(), it.index());
  }

  /// Generate the bytecode for the given PDL interpreter module.
  void generate(ModuleOp module);

  /// Return the memory index to use for the given value.
  ByteCodeField &getMemIndex(Value value) {
    assert(valueToMemIndex.count(value) &&
           "expected memory index to be assigned");
    return valueToMemIndex[value];
  }

  /// Return an index to use when referring to the given data that is uniqued in
  /// the MLIR context.
  template <typename T>
  std::enable_if_t<!std::is_convertible<T, Value>::value, ByteCodeField &>
  getMemIndex(T val) {
    const void *opaqueVal = val.getAsOpaquePointer();

    // Get or insert a reference to this value.
    auto it = uniquedDataToMemIndex.try_emplace(
        opaqueVal, maxValueMemoryIndex + uniquedData.size());
    if (it.second)
      uniquedData.push_back(opaqueVal);
    return it.first->second;
  }

private:
  /// Allocate memory indices for the results of operations within the matcher
  /// and rewriters.
  void allocateMemoryIndices(FuncOp matcherFunc, ModuleOp rewriterModule);

  /// Generate the bytecode for the given operation.
  void generate(Operation *op, ByteCodeWriter &writer);
  void generate(pdl_interp::ApplyConstraintOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::ApplyRewriteOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::AreEqualOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::BranchOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::CheckAttributeOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::CheckOperandCountOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::CheckOperationNameOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::CheckResultCountOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::CheckTypeOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::CreateAttributeOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::CreateOperationOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::CreateTypeOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::EraseOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::FinalizeOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::GetAttributeOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::GetAttributeTypeOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::GetDefiningOpOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::GetOperandOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::GetResultOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::GetValueTypeOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::InferredTypesOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::IsNotNullOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::RecordMatchOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::ReplaceOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::SwitchAttributeOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::SwitchTypeOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::SwitchOperandCountOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::SwitchOperationNameOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::SwitchResultCountOp op, ByteCodeWriter &writer);

  /// Mapping from value to its corresponding memory index.
  DenseMap<Value, ByteCodeField> valueToMemIndex;

  /// Mapping from the name of an externally registered rewrite to its index in
  /// the bytecode registry.
  llvm::StringMap<ByteCodeField> externalRewriterToMemIndex;

  /// Mapping from the name of an externally registered constraint to its index
  /// in the bytecode registry.
  llvm::StringMap<ByteCodeField> constraintToMemIndex;

  /// Mapping from rewriter function name to the bytecode address of the
  /// rewriter function in byte.
  llvm::StringMap<ByteCodeAddr> rewriterToAddr;

  /// Mapping from a uniqued storage object to its memory index within
  /// `uniquedData`.
  DenseMap<const void *, ByteCodeField> uniquedDataToMemIndex;

  /// The current MLIR context.
  MLIRContext *ctx;

  /// Data of the ByteCode class to be populated.
  std::vector<const void *> &uniquedData;
  SmallVectorImpl<ByteCodeField> &matcherByteCode;
  SmallVectorImpl<ByteCodeField> &rewriterByteCode;
  SmallVectorImpl<PDLByteCodePattern> &patterns;
  ByteCodeField &maxValueMemoryIndex;
};

/// This class provides utilities for writing a bytecode stream.
struct ByteCodeWriter {
  ByteCodeWriter(SmallVectorImpl<ByteCodeField> &bytecode, Generator &generator)
      : bytecode(bytecode), generator(generator) {}

  /// Append a field to the bytecode.
  void append(ByteCodeField field) { bytecode.push_back(field); }
  void append(OpCode opCode) { bytecode.push_back(opCode); }

  /// Append an address to the bytecode.
  void append(ByteCodeAddr field) {
    static_assert((sizeof(ByteCodeAddr) / sizeof(ByteCodeField)) == 2,
                  "unexpected ByteCode address size");

    ByteCodeField fieldParts[2];
    std::memcpy(fieldParts, &field, sizeof(ByteCodeAddr));
    bytecode.append({fieldParts[0], fieldParts[1]});
  }

  /// Append a successor range to the bytecode, the exact address will need to
  /// be resolved later.
  void append(SuccessorRange successors) {
    // Add back references to the any successors so that the address can be
    // resolved later.
    for (Block *successor : successors) {
      unresolvedSuccessorRefs[successor].push_back(bytecode.size());
      append(ByteCodeAddr(0));
    }
  }

  /// Append a range of values that will be read as generic PDLValues.
  void appendPDLValueList(OperandRange values) {
    bytecode.push_back(values.size());
    for (Value value : values) {
      // Append the type of the value in addition to the value itself.
      PDLValueKind kind =
          TypeSwitch<Type, PDLValueKind>(value.getType())
              .Case<pdl::AttributeType>(
                  [](Type) { return PDLValueKind::Attribute; })
              .Case<pdl::OperationType>(
                  [](Type) { return PDLValueKind::Operation; })
              .Case<pdl::TypeType>([](Type) { return PDLValueKind::Type; })
              .Case<pdl::ValueType>([](Type) { return PDLValueKind::Value; });
      bytecode.push_back(static_cast<ByteCodeField>(kind));
      append(value);
    }
  }

  /// Check if the given class `T` has an iterator type.
  template <typename T, typename... Args>
  using has_pointer_traits = decltype(std::declval<T>().getAsOpaquePointer());

  /// Append a value that will be stored in a memory slot and not inline within
  /// the bytecode.
  template <typename T>
  std::enable_if_t<llvm::is_detected<has_pointer_traits, T>::value ||
                   std::is_pointer<T>::value>
  append(T value) {
    bytecode.push_back(generator.getMemIndex(value));
  }

  /// Append a range of values.
  template <typename T, typename IteratorT = llvm::detail::IterOfRange<T>>
  std::enable_if_t<!llvm::is_detected<has_pointer_traits, T>::value>
  append(T range) {
    bytecode.push_back(llvm::size(range));
    for (auto it : range)
      append(it);
  }

  /// Append a variadic number of fields to the bytecode.
  template <typename FieldTy, typename Field2Ty, typename... FieldTys>
  void append(FieldTy field, Field2Ty field2, FieldTys... fields) {
    append(field);
    append(field2, fields...);
  }

  /// Successor references in the bytecode that have yet to be resolved.
  DenseMap<Block *, SmallVector<unsigned, 4>> unresolvedSuccessorRefs;

  /// The underlying bytecode buffer.
  SmallVectorImpl<ByteCodeField> &bytecode;

  /// The main generator producing PDL.
  Generator &generator;
};
} // end anonymous namespace

void Generator::generate(ModuleOp module) {
  FuncOp matcherFunc = module.lookupSymbol<FuncOp>(
      pdl_interp::PDLInterpDialect::getMatcherFunctionName());
  ModuleOp rewriterModule = module.lookupSymbol<ModuleOp>(
      pdl_interp::PDLInterpDialect::getRewriterModuleName());
  assert(matcherFunc && rewriterModule && "invalid PDL Interpreter module");

  // Allocate memory indices for the results of operations within the matcher
  // and rewriters.
  allocateMemoryIndices(matcherFunc, rewriterModule);

  // Generate code for the rewriter functions.
  ByteCodeWriter rewriterByteCodeWriter(rewriterByteCode, *this);
  for (FuncOp rewriterFunc : rewriterModule.getOps<FuncOp>()) {
    rewriterToAddr.try_emplace(rewriterFunc.getName(), rewriterByteCode.size());
    for (Operation &op : rewriterFunc.getOps())
      generate(&op, rewriterByteCodeWriter);
  }
  assert(rewriterByteCodeWriter.unresolvedSuccessorRefs.empty() &&
         "unexpected branches in rewriter function");

  // Generate code for the matcher function.
  DenseMap<Block *, ByteCodeAddr> blockToAddr;
  llvm::ReversePostOrderTraversal<Region *> rpot(&matcherFunc.getBody());
  ByteCodeWriter matcherByteCodeWriter(matcherByteCode, *this);
  for (Block *block : rpot) {
    // Keep track of where this block begins within the matcher function.
    blockToAddr.try_emplace(block, matcherByteCode.size());
    for (Operation &op : *block)
      generate(&op, matcherByteCodeWriter);
  }

  // Resolve successor references in the matcher.
  for (auto &it : matcherByteCodeWriter.unresolvedSuccessorRefs) {
    ByteCodeAddr addr = blockToAddr[it.first];
    for (unsigned offsetToFix : it.second)
      std::memcpy(&matcherByteCode[offsetToFix], &addr, sizeof(ByteCodeAddr));
  }
}

void Generator::allocateMemoryIndices(FuncOp matcherFunc,
                                      ModuleOp rewriterModule) {
  // Rewriters use simplistic allocation scheme that simply assigns an index to
  // each result.
  for (FuncOp rewriterFunc : rewriterModule.getOps<FuncOp>()) {
    ByteCodeField index = 0;
    for (BlockArgument arg : rewriterFunc.getArguments())
      valueToMemIndex.try_emplace(arg, index++);
    rewriterFunc.getBody().walk([&](Operation *op) {
      for (Value result : op->getResults())
        valueToMemIndex.try_emplace(result, index++);
    });
    if (index > maxValueMemoryIndex)
      maxValueMemoryIndex = index;
  }

  // The matcher function uses a more sophisticated numbering that tries to
  // minimize the number of memory indices assigned. This is done by determining
  // a live range of the values within the matcher, then the allocation is just
  // finding the minimal number of overlapping live ranges. This is essentially
  // a simplified form of register allocation where we don't necessarily have a
  // limited number of registers, but we still want to minimize the number used.
  DenseMap<Operation *, ByteCodeField> opToIndex;
  matcherFunc.getBody().walk([&](Operation *op) {
    opToIndex.insert(std::make_pair(op, opToIndex.size()));
  });

  // Liveness info for each of the defs within the matcher.
  using LivenessSet = llvm::IntervalMap<ByteCodeField, char, 16>;
  LivenessSet::Allocator allocator;
  DenseMap<Value, LivenessSet> valueDefRanges;

  // Assign the root operation being matched to slot 0.
  BlockArgument rootOpArg = matcherFunc.getArgument(0);
  valueToMemIndex[rootOpArg] = 0;

  // Walk each of the blocks, computing the def interval that the value is used.
  Liveness matcherLiveness(matcherFunc);
  for (Block &block : matcherFunc.getBody()) {
    const LivenessBlockInfo *info = matcherLiveness.getLiveness(&block);
    assert(info && "expected liveness info for block");
    auto processValue = [&](Value value, Operation *firstUseOrDef) {
      // We don't need to process the root op argument, this value is always
      // assigned to the first memory slot.
      if (value == rootOpArg)
        return;

      // Set indices for the range of this block that the value is used.
      auto defRangeIt = valueDefRanges.try_emplace(value, allocator).first;
      defRangeIt->second.insert(
          opToIndex[firstUseOrDef],
          opToIndex[info->getEndOperation(value, firstUseOrDef)],
          /*dummyValue*/ 0);
    };

    // Process the live-ins of this block.
    for (Value liveIn : info->in())
      processValue(liveIn, &block.front());

    // Process any new defs within this block.
    for (Operation &op : block)
      for (Value result : op.getResults())
        processValue(result, &op);
  }

  // Greedily allocate memory slots using the computed def live ranges.
  std::vector<LivenessSet> allocatedIndices;
  for (auto &defIt : valueDefRanges) {
    ByteCodeField &memIndex = valueToMemIndex[defIt.first];
    LivenessSet &defSet = defIt.second;

    // Try to allocate to an existing index.
    for (auto existingIndexIt : llvm::enumerate(allocatedIndices)) {
      LivenessSet &existingIndex = existingIndexIt.value();
      llvm::IntervalMapOverlaps<LivenessSet, LivenessSet> overlaps(
          defIt.second, existingIndex);
      if (overlaps.valid())
        continue;
      // Union the range of the def within the existing index.
      for (auto it = defSet.begin(), e = defSet.end(); it != e; ++it)
        existingIndex.insert(it.start(), it.stop(), /*dummyValue*/ 0);
      memIndex = existingIndexIt.index() + 1;
    }

    // If no existing index could be used, add a new one.
    if (memIndex == 0) {
      allocatedIndices.emplace_back(allocator);
      for (auto it = defSet.begin(), e = defSet.end(); it != e; ++it)
        allocatedIndices.back().insert(it.start(), it.stop(), /*dummyValue*/ 0);
      memIndex = allocatedIndices.size();
    }
  }

  // Update the max number of indices.
  ByteCodeField numMatcherIndices = allocatedIndices.size() + 1;
  if (numMatcherIndices > maxValueMemoryIndex)
    maxValueMemoryIndex = numMatcherIndices;
}

void Generator::generate(Operation *op, ByteCodeWriter &writer) {
  TypeSwitch<Operation *>(op)
      .Case<pdl_interp::ApplyConstraintOp, pdl_interp::ApplyRewriteOp,
            pdl_interp::AreEqualOp, pdl_interp::BranchOp,
            pdl_interp::CheckAttributeOp, pdl_interp::CheckOperandCountOp,
            pdl_interp::CheckOperationNameOp, pdl_interp::CheckResultCountOp,
            pdl_interp::CheckTypeOp, pdl_interp::CreateAttributeOp,
            pdl_interp::CreateOperationOp, pdl_interp::CreateTypeOp,
            pdl_interp::EraseOp, pdl_interp::FinalizeOp,
            pdl_interp::GetAttributeOp, pdl_interp::GetAttributeTypeOp,
            pdl_interp::GetDefiningOpOp, pdl_interp::GetOperandOp,
            pdl_interp::GetResultOp, pdl_interp::GetValueTypeOp,
            pdl_interp::InferredTypesOp, pdl_interp::IsNotNullOp,
            pdl_interp::RecordMatchOp, pdl_interp::ReplaceOp,
            pdl_interp::SwitchAttributeOp, pdl_interp::SwitchTypeOp,
            pdl_interp::SwitchOperandCountOp, pdl_interp::SwitchOperationNameOp,
            pdl_interp::SwitchResultCountOp>(
          [&](auto interpOp) { this->generate(interpOp, writer); })
      .Default([](Operation *) {
        llvm_unreachable("unknown `pdl_interp` operation");
      });
}

void Generator::generate(pdl_interp::ApplyConstraintOp op,
                         ByteCodeWriter &writer) {
  assert(constraintToMemIndex.count(op.name()) &&
         "expected index for constraint function");
  writer.append(OpCode::ApplyConstraint, constraintToMemIndex[op.name()],
                op.constParamsAttr());
  writer.appendPDLValueList(op.args());
  writer.append(op.getSuccessors());
}
void Generator::generate(pdl_interp::ApplyRewriteOp op,
                         ByteCodeWriter &writer) {
  assert(externalRewriterToMemIndex.count(op.name()) &&
         "expected index for rewrite function");
  writer.append(OpCode::ApplyRewrite, externalRewriterToMemIndex[op.name()],
                op.constParamsAttr());
  writer.appendPDLValueList(op.args());

#ifndef NDEBUG
  // In debug mode we also append the number of results so that we can assert
  // that the native creation function gave us the correct number of results.
  writer.append(ByteCodeField(op.results().size()));
#endif
  for (Value result : op.results())
    writer.append(result);
}
void Generator::generate(pdl_interp::AreEqualOp op, ByteCodeWriter &writer) {
  writer.append(OpCode::AreEqual, op.lhs(), op.rhs(), op.getSuccessors());
}
void Generator::generate(pdl_interp::BranchOp op, ByteCodeWriter &writer) {
  writer.append(OpCode::Branch, SuccessorRange(op.getOperation()));
}
void Generator::generate(pdl_interp::CheckAttributeOp op,
                         ByteCodeWriter &writer) {
  writer.append(OpCode::AreEqual, op.attribute(), op.constantValue(),
                op.getSuccessors());
}
void Generator::generate(pdl_interp::CheckOperandCountOp op,
                         ByteCodeWriter &writer) {
  writer.append(OpCode::CheckOperandCount, op.operation(), op.count(),
                op.getSuccessors());
}
void Generator::generate(pdl_interp::CheckOperationNameOp op,
                         ByteCodeWriter &writer) {
  writer.append(OpCode::CheckOperationName, op.operation(),
                OperationName(op.name(), ctx), op.getSuccessors());
}
void Generator::generate(pdl_interp::CheckResultCountOp op,
                         ByteCodeWriter &writer) {
  writer.append(OpCode::CheckResultCount, op.operation(), op.count(),
                op.getSuccessors());
}
void Generator::generate(pdl_interp::CheckTypeOp op, ByteCodeWriter &writer) {
  writer.append(OpCode::AreEqual, op.value(), op.type(), op.getSuccessors());
}
void Generator::generate(pdl_interp::CreateAttributeOp op,
                         ByteCodeWriter &writer) {
  // Simply repoint the memory index of the result to the constant.
  getMemIndex(op.attribute()) = getMemIndex(op.value());
}
void Generator::generate(pdl_interp::CreateOperationOp op,
                         ByteCodeWriter &writer) {
  writer.append(OpCode::CreateOperation, op.operation(),
                OperationName(op.name(), ctx), op.operands());

  // Add the attributes.
  OperandRange attributes = op.attributes();
  writer.append(static_cast<ByteCodeField>(attributes.size()));
  for (auto it : llvm::zip(op.attributeNames(), op.attributes())) {
    writer.append(
        Identifier::get(std::get<0>(it).cast<StringAttr>().getValue(), ctx),
        std::get<1>(it));
  }
  writer.append(op.types());
}
void Generator::generate(pdl_interp::CreateTypeOp op, ByteCodeWriter &writer) {
  // Simply repoint the memory index of the result to the constant.
  getMemIndex(op.result()) = getMemIndex(op.value());
}
void Generator::generate(pdl_interp::EraseOp op, ByteCodeWriter &writer) {
  writer.append(OpCode::EraseOp, op.operation());
}
void Generator::generate(pdl_interp::FinalizeOp op, ByteCodeWriter &writer) {
  writer.append(OpCode::Finalize);
}
void Generator::generate(pdl_interp::GetAttributeOp op,
                         ByteCodeWriter &writer) {
  writer.append(OpCode::GetAttribute, op.attribute(), op.operation(),
                Identifier::get(op.name(), ctx));
}
void Generator::generate(pdl_interp::GetAttributeTypeOp op,
                         ByteCodeWriter &writer) {
  writer.append(OpCode::GetAttributeType, op.result(), op.value());
}
void Generator::generate(pdl_interp::GetDefiningOpOp op,
                         ByteCodeWriter &writer) {
  writer.append(OpCode::GetDefiningOp, op.operation(), op.value());
}
void Generator::generate(pdl_interp::GetOperandOp op, ByteCodeWriter &writer) {
  uint32_t index = op.index();
  if (index < 4)
    writer.append(static_cast<OpCode>(OpCode::GetOperand0 + index));
  else
    writer.append(OpCode::GetOperandN, index);
  writer.append(op.operation(), op.value());
}
void Generator::generate(pdl_interp::GetResultOp op, ByteCodeWriter &writer) {
  uint32_t index = op.index();
  if (index < 4)
    writer.append(static_cast<OpCode>(OpCode::GetResult0 + index));
  else
    writer.append(OpCode::GetResultN, index);
  writer.append(op.operation(), op.value());
}
void Generator::generate(pdl_interp::GetValueTypeOp op,
                         ByteCodeWriter &writer) {
  writer.append(OpCode::GetValueType, op.result(), op.value());
}
void Generator::generate(pdl_interp::InferredTypesOp op,
                         ByteCodeWriter &writer) {
  // InferType maps to a null type as a marker for inferring result types.
  getMemIndex(op.type()) = getMemIndex(Type());
}
void Generator::generate(pdl_interp::IsNotNullOp op, ByteCodeWriter &writer) {
  writer.append(OpCode::IsNotNull, op.value(), op.getSuccessors());
}
void Generator::generate(pdl_interp::RecordMatchOp op, ByteCodeWriter &writer) {
  ByteCodeField patternIndex = patterns.size();
  patterns.emplace_back(PDLByteCodePattern::create(
      op, rewriterToAddr[op.rewriter().getLeafReference()]));
  writer.append(OpCode::RecordMatch, patternIndex,
                SuccessorRange(op.getOperation()), op.matchedOps(),
                op.inputs());
}
void Generator::generate(pdl_interp::ReplaceOp op, ByteCodeWriter &writer) {
  writer.append(OpCode::ReplaceOp, op.operation(), op.replValues());
}
void Generator::generate(pdl_interp::SwitchAttributeOp op,
                         ByteCodeWriter &writer) {
  writer.append(OpCode::SwitchAttribute, op.attribute(), op.caseValuesAttr(),
                op.getSuccessors());
}
void Generator::generate(pdl_interp::SwitchOperandCountOp op,
                         ByteCodeWriter &writer) {
  writer.append(OpCode::SwitchOperandCount, op.operation(), op.caseValuesAttr(),
                op.getSuccessors());
}
void Generator::generate(pdl_interp::SwitchOperationNameOp op,
                         ByteCodeWriter &writer) {
  auto cases = llvm::map_range(op.caseValuesAttr(), [&](Attribute attr) {
    return OperationName(attr.cast<StringAttr>().getValue(), ctx);
  });
  writer.append(OpCode::SwitchOperationName, op.operation(), cases,
                op.getSuccessors());
}
void Generator::generate(pdl_interp::SwitchResultCountOp op,
                         ByteCodeWriter &writer) {
  writer.append(OpCode::SwitchResultCount, op.operation(), op.caseValuesAttr(),
                op.getSuccessors());
}
void Generator::generate(pdl_interp::SwitchTypeOp op, ByteCodeWriter &writer) {
  writer.append(OpCode::SwitchType, op.value(), op.caseValuesAttr(),
                op.getSuccessors());
}

//===----------------------------------------------------------------------===//
// PDLByteCode
//===----------------------------------------------------------------------===//

PDLByteCode::PDLByteCode(ModuleOp module,
                         llvm::StringMap<PDLConstraintFunction> constraintFns,
                         llvm::StringMap<PDLRewriteFunction> rewriteFns) {
  Generator generator(module.getContext(), uniquedData, matcherByteCode,
                      rewriterByteCode, patterns, maxValueMemoryIndex,
                      constraintFns, rewriteFns);
  generator.generate(module);

  // Initialize the external functions.
  for (auto &it : constraintFns)
    constraintFunctions.push_back(std::move(it.second));
  for (auto &it : rewriteFns)
    rewriteFunctions.push_back(std::move(it.second));
}

/// Initialize the given state such that it can be used to execute the current
/// bytecode.
void PDLByteCode::initializeMutableState(PDLByteCodeMutableState &state) const {
  state.memory.resize(maxValueMemoryIndex, nullptr);
  state.currentPatternBenefits.reserve(patterns.size());
  for (const PDLByteCodePattern &pattern : patterns)
    state.currentPatternBenefits.push_back(pattern.getBenefit());
}

//===----------------------------------------------------------------------===//
// ByteCode Execution

namespace {
/// This class provides support for executing a bytecode stream.
class ByteCodeExecutor {
public:
  ByteCodeExecutor(const ByteCodeField *curCodeIt,
                   MutableArrayRef<const void *> memory,
                   ArrayRef<const void *> uniquedMemory,
                   ArrayRef<ByteCodeField> code,
                   ArrayRef<PatternBenefit> currentPatternBenefits,
                   ArrayRef<PDLByteCodePattern> patterns,
                   ArrayRef<PDLConstraintFunction> constraintFunctions,
                   ArrayRef<PDLRewriteFunction> rewriteFunctions)
      : curCodeIt(curCodeIt), memory(memory), uniquedMemory(uniquedMemory),
        code(code), currentPatternBenefits(currentPatternBenefits),
        patterns(patterns), constraintFunctions(constraintFunctions),
        rewriteFunctions(rewriteFunctions) {}

  /// Start executing the code at the current bytecode index. `matches` is an
  /// optional field provided when this function is executed in a matching
  /// context.
  void execute(PatternRewriter &rewriter,
               SmallVectorImpl<PDLByteCode::MatchResult> *matches = nullptr,
               Optional<Location> mainRewriteLoc = {});

private:
  /// Internal implementation of executing each of the bytecode commands.
  void executeApplyConstraint(PatternRewriter &rewriter);
  void executeApplyRewrite(PatternRewriter &rewriter);
  void executeAreEqual();
  void executeBranch();
  void executeCheckOperandCount();
  void executeCheckOperationName();
  void executeCheckResultCount();
  void executeCreateOperation(PatternRewriter &rewriter,
                              Location mainRewriteLoc);
  void executeEraseOp(PatternRewriter &rewriter);
  void executeGetAttribute();
  void executeGetAttributeType();
  void executeGetDefiningOp();
  void executeGetOperand(unsigned index);
  void executeGetResult(unsigned index);
  void executeGetValueType();
  void executeIsNotNull();
  void executeRecordMatch(PatternRewriter &rewriter,
                          SmallVectorImpl<PDLByteCode::MatchResult> &matches);
  void executeReplaceOp(PatternRewriter &rewriter);
  void executeSwitchAttribute();
  void executeSwitchOperandCount();
  void executeSwitchOperationName();
  void executeSwitchResultCount();
  void executeSwitchType();

  /// Read a value from the bytecode buffer, optionally skipping a certain
  /// number of prefix values. These methods always update the buffer to point
  /// to the next field after the read data.
  template <typename T = ByteCodeField>
  T read(size_t skipN = 0) {
    curCodeIt += skipN;
    return readImpl<T>();
  }
  ByteCodeField read(size_t skipN = 0) { return read<ByteCodeField>(skipN); }

  /// Read a list of values from the bytecode buffer.
  template <typename ValueT, typename T>
  void readList(SmallVectorImpl<T> &list) {
    list.clear();
    for (unsigned i = 0, e = read(); i != e; ++i)
      list.push_back(read<ValueT>());
  }

  /// Jump to a specific successor based on a predicate value.
  void selectJump(bool isTrue) { selectJump(size_t(isTrue ? 0 : 1)); }
  /// Jump to a specific successor based on a destination index.
  void selectJump(size_t destIndex) {
    curCodeIt = &code[read<ByteCodeAddr>(destIndex * 2)];
  }

  /// Handle a switch operation with the provided value and cases.
  template <typename T, typename RangeT>
  void handleSwitch(const T &value, RangeT &&cases) {
    LLVM_DEBUG({
      llvm::dbgs() << "  * Value: " << value << "\n"
                   << "  * Cases: ";
      llvm::interleaveComma(cases, llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    // Check to see if the attribute value is within the case list. Jump to
    // the correct successor index based on the result.
    for (auto it = cases.begin(), e = cases.end(); it != e; ++it)
      if (*it == value)
        return selectJump(size_t((it - cases.begin()) + 1));
    selectJump(size_t(0));
  }

  /// Internal implementation of reading various data types from the bytecode
  /// stream.
  template <typename T>
  const void *readFromMemory() {
    size_t index = *curCodeIt++;

    // If this type is an SSA value, it can only be stored in non-const memory.
    if (llvm::is_one_of<T, Operation *, Value>::value || index < memory.size())
      return memory[index];

    // Otherwise, if this index is not inbounds it is uniqued.
    return uniquedMemory[index - memory.size()];
  }
  template <typename T>
  std::enable_if_t<std::is_pointer<T>::value, T> readImpl() {
    return reinterpret_cast<T>(const_cast<void *>(readFromMemory<T>()));
  }
  template <typename T>
  std::enable_if_t<std::is_class<T>::value && !std::is_same<PDLValue, T>::value,
                   T>
  readImpl() {
    return T(T::getFromOpaquePointer(readFromMemory<T>()));
  }
  template <typename T>
  std::enable_if_t<std::is_same<PDLValue, T>::value, T> readImpl() {
    switch (static_cast<PDLValueKind>(read())) {
    case PDLValueKind::Attribute:
      return read<Attribute>();
    case PDLValueKind::Operation:
      return read<Operation *>();
    case PDLValueKind::Type:
      return read<Type>();
    case PDLValueKind::Value:
      return read<Value>();
    }
    llvm_unreachable("unhandled PDLValueKind");
  }
  template <typename T>
  std::enable_if_t<std::is_same<T, ByteCodeAddr>::value, T> readImpl() {
    static_assert((sizeof(ByteCodeAddr) / sizeof(ByteCodeField)) == 2,
                  "unexpected ByteCode address size");
    ByteCodeAddr result;
    std::memcpy(&result, curCodeIt, sizeof(ByteCodeAddr));
    curCodeIt += 2;
    return result;
  }
  template <typename T>
  std::enable_if_t<std::is_same<T, ByteCodeField>::value, T> readImpl() {
    return *curCodeIt++;
  }

  /// The underlying bytecode buffer.
  const ByteCodeField *curCodeIt;

  /// The current execution memory.
  MutableArrayRef<const void *> memory;

  /// References to ByteCode data necessary for execution.
  ArrayRef<const void *> uniquedMemory;
  ArrayRef<ByteCodeField> code;
  ArrayRef<PatternBenefit> currentPatternBenefits;
  ArrayRef<PDLByteCodePattern> patterns;
  ArrayRef<PDLConstraintFunction> constraintFunctions;
  ArrayRef<PDLRewriteFunction> rewriteFunctions;
};

/// This class is an instantiation of the PDLResultList that provides access to
/// the returned results. This API is not on `PDLResultList` to avoid
/// overexposing access to information specific solely to the ByteCode.
class ByteCodeRewriteResultList : public PDLResultList {
public:
  /// Return the list of PDL results.
  MutableArrayRef<PDLValue> getResults() { return results; }
};
} // end anonymous namespace

void ByteCodeExecutor::executeApplyConstraint(PatternRewriter &rewriter) {
  LLVM_DEBUG(llvm::dbgs() << "Executing ApplyConstraint:\n");
  const PDLConstraintFunction &constraintFn = constraintFunctions[read()];
  ArrayAttr constParams = read<ArrayAttr>();
  SmallVector<PDLValue, 16> args;
  readList<PDLValue>(args);

  LLVM_DEBUG({
    llvm::dbgs() << "  * Arguments: ";
    llvm::interleaveComma(args, llvm::dbgs());
    llvm::dbgs() << "\n  * Parameters: " << constParams << "\n";
  });

  // Invoke the constraint and jump to the proper destination.
  selectJump(succeeded(constraintFn(args, constParams, rewriter)));
}

void ByteCodeExecutor::executeApplyRewrite(PatternRewriter &rewriter) {
  LLVM_DEBUG(llvm::dbgs() << "Executing ApplyRewrite:\n");
  const PDLRewriteFunction &rewriteFn = rewriteFunctions[read()];
  ArrayAttr constParams = read<ArrayAttr>();
  SmallVector<PDLValue, 16> args;
  readList<PDLValue>(args);

  LLVM_DEBUG({
    llvm::dbgs() << "  * Arguments: ";
    llvm::interleaveComma(args, llvm::dbgs());
    llvm::dbgs() << "\n  * Parameters: " << constParams << "\n";
  });
  ByteCodeRewriteResultList results;
  rewriteFn(args, constParams, rewriter, results);

  // Store the results in the bytecode memory.
#ifndef NDEBUG
  ByteCodeField expectedNumberOfResults = read();
  assert(results.getResults().size() == expectedNumberOfResults &&
         "native PDL rewrite function returned unexpected number of results");
#endif

  // Store the results in the bytecode memory.
  for (PDLValue &result : results.getResults()) {
    LLVM_DEBUG(llvm::dbgs() << "  * Result: " << result << "\n");
    memory[read()] = result.getAsOpaquePointer();
  }
}

void ByteCodeExecutor::executeAreEqual() {
  LLVM_DEBUG(llvm::dbgs() << "Executing AreEqual:\n");
  const void *lhs = read<const void *>();
  const void *rhs = read<const void *>();

  LLVM_DEBUG(llvm::dbgs() << "  * " << lhs << " == " << rhs << "\n");
  selectJump(lhs == rhs);
}

void ByteCodeExecutor::executeBranch() {
  LLVM_DEBUG(llvm::dbgs() << "Executing Branch\n");
  curCodeIt = &code[read<ByteCodeAddr>()];
}

void ByteCodeExecutor::executeCheckOperandCount() {
  LLVM_DEBUG(llvm::dbgs() << "Executing CheckOperandCount:\n");
  Operation *op = read<Operation *>();
  uint32_t expectedCount = read<uint32_t>();

  LLVM_DEBUG(llvm::dbgs() << "  * Found: " << op->getNumOperands() << "\n"
                          << "  * Expected: " << expectedCount << "\n");
  selectJump(op->getNumOperands() == expectedCount);
}

void ByteCodeExecutor::executeCheckOperationName() {
  LLVM_DEBUG(llvm::dbgs() << "Executing CheckOperationName:\n");
  Operation *op = read<Operation *>();
  OperationName expectedName = read<OperationName>();

  LLVM_DEBUG(llvm::dbgs() << "  * Found: \"" << op->getName() << "\"\n"
                          << "  * Expected: \"" << expectedName << "\"\n");
  selectJump(op->getName() == expectedName);
}

void ByteCodeExecutor::executeCheckResultCount() {
  LLVM_DEBUG(llvm::dbgs() << "Executing CheckResultCount:\n");
  Operation *op = read<Operation *>();
  uint32_t expectedCount = read<uint32_t>();

  LLVM_DEBUG(llvm::dbgs() << "  * Found: " << op->getNumResults() << "\n"
                          << "  * Expected: " << expectedCount << "\n");
  selectJump(op->getNumResults() == expectedCount);
}

void ByteCodeExecutor::executeCreateOperation(PatternRewriter &rewriter,
                                              Location mainRewriteLoc) {
  LLVM_DEBUG(llvm::dbgs() << "Executing CreateOperation:\n");

  unsigned memIndex = read();
  OperationState state(mainRewriteLoc, read<OperationName>());
  readList<Value>(state.operands);
  for (unsigned i = 0, e = read(); i != e; ++i) {
    Identifier name = read<Identifier>();
    if (Attribute attr = read<Attribute>())
      state.addAttribute(name, attr);
  }

  bool hasInferredTypes = false;
  for (unsigned i = 0, e = read(); i != e; ++i) {
    Type resultType = read<Type>();
    hasInferredTypes |= !resultType;
    state.types.push_back(resultType);
  }

  // Handle the case where the operation has inferred types.
  if (hasInferredTypes) {
    InferTypeOpInterface::Concept *concept =
        state.name.getAbstractOperation()->getInterface<InferTypeOpInterface>();

    // TODO: Handle failure.
    state.types.clear();
    if (failed(concept->inferReturnTypes(
            state.getContext(), state.location, state.operands,
            state.attributes.getDictionary(state.getContext()), state.regions,
            state.types)))
      return;
  }
  Operation *resultOp = rewriter.createOperation(state);
  memory[memIndex] = resultOp;

  LLVM_DEBUG({
    llvm::dbgs() << "  * Attributes: "
                 << state.attributes.getDictionary(state.getContext())
                 << "\n  * Operands: ";
    llvm::interleaveComma(state.operands, llvm::dbgs());
    llvm::dbgs() << "\n  * Result Types: ";
    llvm::interleaveComma(state.types, llvm::dbgs());
    llvm::dbgs() << "\n  * Result: " << *resultOp << "\n";
  });
}

void ByteCodeExecutor::executeEraseOp(PatternRewriter &rewriter) {
  LLVM_DEBUG(llvm::dbgs() << "Executing EraseOp:\n");
  Operation *op = read<Operation *>();

  LLVM_DEBUG(llvm::dbgs() << "  * Operation: " << *op << "\n");
  rewriter.eraseOp(op);
}

void ByteCodeExecutor::executeGetAttribute() {
  LLVM_DEBUG(llvm::dbgs() << "Executing GetAttribute:\n");
  unsigned memIndex = read();
  Operation *op = read<Operation *>();
  Identifier attrName = read<Identifier>();
  Attribute attr = op->getAttr(attrName);

  LLVM_DEBUG(llvm::dbgs() << "  * Operation: " << *op << "\n"
                          << "  * Attribute: " << attrName << "\n"
                          << "  * Result: " << attr << "\n");
  memory[memIndex] = attr.getAsOpaquePointer();
}

void ByteCodeExecutor::executeGetAttributeType() {
  LLVM_DEBUG(llvm::dbgs() << "Executing GetAttributeType:\n");
  unsigned memIndex = read();
  Attribute attr = read<Attribute>();
  Type type = attr ? attr.getType() : Type();

  LLVM_DEBUG(llvm::dbgs() << "  * Attribute: " << attr << "\n"
                          << "  * Result: " << type << "\n");
  memory[memIndex] = type.getAsOpaquePointer();
}

void ByteCodeExecutor::executeGetDefiningOp() {
  LLVM_DEBUG(llvm::dbgs() << "Executing GetDefiningOp:\n");
  unsigned memIndex = read();
  Value value = read<Value>();
  Operation *op = value ? value.getDefiningOp() : nullptr;

  LLVM_DEBUG(llvm::dbgs() << "  * Value: " << value << "\n"
                          << "  * Result: " << *op << "\n");
  memory[memIndex] = op;
}

void ByteCodeExecutor::executeGetOperand(unsigned index) {
  Operation *op = read<Operation *>();
  unsigned memIndex = read();
  Value operand =
      index < op->getNumOperands() ? op->getOperand(index) : Value();

  LLVM_DEBUG(llvm::dbgs() << "  * Operation: " << *op << "\n"
                          << "  * Index: " << index << "\n"
                          << "  * Result: " << operand << "\n");
  memory[memIndex] = operand.getAsOpaquePointer();
}

void ByteCodeExecutor::executeGetResult(unsigned index) {
  Operation *op = read<Operation *>();
  unsigned memIndex = read();
  OpResult result =
      index < op->getNumResults() ? op->getResult(index) : OpResult();

  LLVM_DEBUG(llvm::dbgs() << "  * Operation: " << *op << "\n"
                          << "  * Index: " << index << "\n"
                          << "  * Result: " << result << "\n");
  memory[memIndex] = result.getAsOpaquePointer();
}

void ByteCodeExecutor::executeGetValueType() {
  LLVM_DEBUG(llvm::dbgs() << "Executing GetValueType:\n");
  unsigned memIndex = read();
  Value value = read<Value>();
  Type type = value ? value.getType() : Type();

  LLVM_DEBUG(llvm::dbgs() << "  * Value: " << value << "\n"
                          << "  * Result: " << type << "\n");
  memory[memIndex] = type.getAsOpaquePointer();
}

void ByteCodeExecutor::executeIsNotNull() {
  LLVM_DEBUG(llvm::dbgs() << "Executing IsNotNull:\n");
  const void *value = read<const void *>();

  LLVM_DEBUG(llvm::dbgs() << "  * Value: " << value << "\n");
  selectJump(value != nullptr);
}

void ByteCodeExecutor::executeRecordMatch(
    PatternRewriter &rewriter,
    SmallVectorImpl<PDLByteCode::MatchResult> &matches) {
  LLVM_DEBUG(llvm::dbgs() << "Executing RecordMatch:\n");
  unsigned patternIndex = read();
  PatternBenefit benefit = currentPatternBenefits[patternIndex];
  const ByteCodeField *dest = &code[read<ByteCodeAddr>()];

  // If the benefit of the pattern is impossible, skip the processing of the
  // rest of the pattern.
  if (benefit.isImpossibleToMatch()) {
    LLVM_DEBUG(llvm::dbgs() << "  * Benefit: Impossible To Match\n");
    curCodeIt = dest;
    return;
  }

  // Create a fused location containing the locations of each of the
  // operations used in the match. This will be used as the location for
  // created operations during the rewrite that don't already have an
  // explicit location set.
  unsigned numMatchLocs = read();
  SmallVector<Location, 4> matchLocs;
  matchLocs.reserve(numMatchLocs);
  for (unsigned i = 0; i != numMatchLocs; ++i)
    matchLocs.push_back(read<Operation *>()->getLoc());
  Location matchLoc = rewriter.getFusedLoc(matchLocs);

  LLVM_DEBUG(llvm::dbgs() << "  * Benefit: " << benefit.getBenefit() << "\n"
                          << "  * Location: " << matchLoc << "\n");
  matches.emplace_back(matchLoc, patterns[patternIndex], benefit);
  readList<const void *>(matches.back().values);
  curCodeIt = dest;
}

void ByteCodeExecutor::executeReplaceOp(PatternRewriter &rewriter) {
  LLVM_DEBUG(llvm::dbgs() << "Executing ReplaceOp:\n");
  Operation *op = read<Operation *>();
  SmallVector<Value, 16> args;
  readList<Value>(args);

  LLVM_DEBUG({
    llvm::dbgs() << "  * Operation: " << *op << "\n"
                 << "  * Values: ";
    llvm::interleaveComma(args, llvm::dbgs());
    llvm::dbgs() << "\n";
  });
  rewriter.replaceOp(op, args);
}

void ByteCodeExecutor::executeSwitchAttribute() {
  LLVM_DEBUG(llvm::dbgs() << "Executing SwitchAttribute:\n");
  Attribute value = read<Attribute>();
  ArrayAttr cases = read<ArrayAttr>();
  handleSwitch(value, cases);
}

void ByteCodeExecutor::executeSwitchOperandCount() {
  LLVM_DEBUG(llvm::dbgs() << "Executing SwitchOperandCount:\n");
  Operation *op = read<Operation *>();
  auto cases = read<DenseIntOrFPElementsAttr>().getValues<uint32_t>();

  LLVM_DEBUG(llvm::dbgs() << "  * Operation: " << *op << "\n");
  handleSwitch(op->getNumOperands(), cases);
}

void ByteCodeExecutor::executeSwitchOperationName() {
  LLVM_DEBUG(llvm::dbgs() << "Executing SwitchOperationName:\n");
  OperationName value = read<Operation *>()->getName();
  size_t caseCount = read();

  // The operation names are stored in-line, so to print them out for
  // debugging purposes we need to read the array before executing the
  // switch so that we can display all of the possible values.
  LLVM_DEBUG({
    const ByteCodeField *prevCodeIt = curCodeIt;
    llvm::dbgs() << "  * Value: " << value << "\n"
                 << "  * Cases: ";
    llvm::interleaveComma(
        llvm::map_range(llvm::seq<size_t>(0, caseCount),
                        [&](size_t) { return read<OperationName>(); }),
        llvm::dbgs());
    llvm::dbgs() << "\n";
    curCodeIt = prevCodeIt;
  });

  // Try to find the switch value within any of the cases.
  for (size_t i = 0; i != caseCount; ++i) {
    if (read<OperationName>() == value) {
      curCodeIt += (caseCount - i - 1);
      return selectJump(i + 1);
    }
  }
  selectJump(size_t(0));
}

void ByteCodeExecutor::executeSwitchResultCount() {
  LLVM_DEBUG(llvm::dbgs() << "Executing SwitchResultCount:\n");
  Operation *op = read<Operation *>();
  auto cases = read<DenseIntOrFPElementsAttr>().getValues<uint32_t>();

  LLVM_DEBUG(llvm::dbgs() << "  * Operation: " << *op << "\n");
  handleSwitch(op->getNumResults(), cases);
}

void ByteCodeExecutor::executeSwitchType() {
  LLVM_DEBUG(llvm::dbgs() << "Executing SwitchType:\n");
  Type value = read<Type>();
  auto cases = read<ArrayAttr>().getAsValueRange<TypeAttr>();
  handleSwitch(value, cases);
}

void ByteCodeExecutor::execute(
    PatternRewriter &rewriter,
    SmallVectorImpl<PDLByteCode::MatchResult> *matches,
    Optional<Location> mainRewriteLoc) {
  while (true) {
    OpCode opCode = static_cast<OpCode>(read());
    switch (opCode) {
    case ApplyConstraint:
      executeApplyConstraint(rewriter);
      break;
    case ApplyRewrite:
      executeApplyRewrite(rewriter);
      break;
    case AreEqual:
      executeAreEqual();
      break;
    case Branch:
      executeBranch();
      break;
    case CheckOperandCount:
      executeCheckOperandCount();
      break;
    case CheckOperationName:
      executeCheckOperationName();
      break;
    case CheckResultCount:
      executeCheckResultCount();
      break;
    case CreateOperation:
      executeCreateOperation(rewriter, *mainRewriteLoc);
      break;
    case EraseOp:
      executeEraseOp(rewriter);
      break;
    case Finalize:
      LLVM_DEBUG(llvm::dbgs() << "Executing Finalize\n\n");
      return;
    case GetAttribute:
      executeGetAttribute();
      break;
    case GetAttributeType:
      executeGetAttributeType();
      break;
    case GetDefiningOp:
      executeGetDefiningOp();
      break;
    case GetOperand0:
    case GetOperand1:
    case GetOperand2:
    case GetOperand3: {
      unsigned index = opCode - GetOperand0;
      LLVM_DEBUG(llvm::dbgs() << "Executing GetOperand" << index << ":\n");
      executeGetOperand(index);
      break;
    }
    case GetOperandN:
      LLVM_DEBUG(llvm::dbgs() << "Executing GetOperandN:\n");
      executeGetOperand(read<uint32_t>());
      break;
    case GetResult0:
    case GetResult1:
    case GetResult2:
    case GetResult3: {
      unsigned index = opCode - GetResult0;
      LLVM_DEBUG(llvm::dbgs() << "Executing GetResult" << index << ":\n");
      executeGetResult(index);
      break;
    }
    case GetResultN:
      LLVM_DEBUG(llvm::dbgs() << "Executing GetResultN:\n");
      executeGetResult(read<uint32_t>());
      break;
    case GetValueType:
      executeGetValueType();
      break;
    case IsNotNull:
      executeIsNotNull();
      break;
    case RecordMatch:
      assert(matches &&
             "expected matches to be provided when executing the matcher");
      executeRecordMatch(rewriter, *matches);
      break;
    case ReplaceOp:
      executeReplaceOp(rewriter);
      break;
    case SwitchAttribute:
      executeSwitchAttribute();
      break;
    case SwitchOperandCount:
      executeSwitchOperandCount();
      break;
    case SwitchOperationName:
      executeSwitchOperationName();
      break;
    case SwitchResultCount:
      executeSwitchResultCount();
      break;
    case SwitchType:
      executeSwitchType();
      break;
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }
}

/// Run the pattern matcher on the given root operation, collecting the matched
/// patterns in `matches`.
void PDLByteCode::match(Operation *op, PatternRewriter &rewriter,
                        SmallVectorImpl<MatchResult> &matches,
                        PDLByteCodeMutableState &state) const {
  // The first memory slot is always the root operation.
  state.memory[0] = op;

  // The matcher function always starts at code address 0.
  ByteCodeExecutor executor(matcherByteCode.data(), state.memory, uniquedData,
                            matcherByteCode, state.currentPatternBenefits,
                            patterns, constraintFunctions, rewriteFunctions);
  executor.execute(rewriter, &matches);

  // Order the found matches by benefit.
  std::stable_sort(matches.begin(), matches.end(),
                   [](const MatchResult &lhs, const MatchResult &rhs) {
                     return lhs.benefit > rhs.benefit;
                   });
}

/// Run the rewriter of the given pattern on the root operation `op`.
void PDLByteCode::rewrite(PatternRewriter &rewriter, const MatchResult &match,
                          PDLByteCodeMutableState &state) const {
  // The arguments of the rewrite function are stored at the start of the
  // memory buffer.
  llvm::copy(match.values, state.memory.begin());

  ByteCodeExecutor executor(&rewriterByteCode[match.pattern->getRewriterAddr()],
                            state.memory, uniquedData, rewriterByteCode,
                            state.currentPatternBenefits, patterns,
                            constraintFunctions, rewriteFunctions);
  executor.execute(rewriter, /*matches=*/nullptr, match.location);
}
