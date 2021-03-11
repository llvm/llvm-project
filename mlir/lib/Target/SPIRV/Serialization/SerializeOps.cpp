//===- SerializeOps.cpp - MLIR SPIR-V Serialization (Ops) -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the serialization methods for MLIR SPIR-V module ops.
//
//===----------------------------------------------------------------------===//

#include "Serializer.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/SPIRV/SPIRVBinaryUtils.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "spirv-serialization"

using namespace mlir;

/// A pre-order depth-first visitor function for processing basic blocks.
///
/// Visits the basic blocks starting from the given `headerBlock` in pre-order
/// depth-first manner and calls `blockHandler` on each block. Skips handling
/// blocks in the `skipBlocks` list. If `skipHeader` is true, `blockHandler`
/// will not be invoked in `headerBlock` but still handles all `headerBlock`'s
/// successors.
///
/// SPIR-V spec "2.16.1. Universal Validation Rules" requires that "the order
/// of blocks in a function must satisfy the rule that blocks appear before
/// all blocks they dominate." This can be achieved by a pre-order CFG
/// traversal algorithm. To make the serialization output more logical and
/// readable to human, we perform depth-first CFG traversal and delay the
/// serialization of the merge block and the continue block, if exists, until
/// after all other blocks have been processed.
static LogicalResult
visitInPrettyBlockOrder(Block *headerBlock,
                        function_ref<LogicalResult(Block *)> blockHandler,
                        bool skipHeader = false, BlockRange skipBlocks = {}) {
  llvm::df_iterator_default_set<Block *, 4> doneBlocks;
  doneBlocks.insert(skipBlocks.begin(), skipBlocks.end());

  for (Block *block : llvm::depth_first_ext(headerBlock, doneBlocks)) {
    if (skipHeader && block == headerBlock)
      continue;
    if (failed(blockHandler(block)))
      return failure();
  }
  return success();
}

namespace mlir {
namespace spirv {
LogicalResult Serializer::processConstantOp(spirv::ConstantOp op) {
  if (auto resultID = prepareConstant(op.getLoc(), op.getType(), op.value())) {
    valueIDMap[op.getResult()] = resultID;
    return success();
  }
  return failure();
}

LogicalResult Serializer::processSpecConstantOp(spirv::SpecConstantOp op) {
  if (auto resultID = prepareConstantScalar(op.getLoc(), op.default_value(),
                                            /*isSpec=*/true)) {
    // Emit the OpDecorate instruction for SpecId.
    if (auto specID = op->getAttrOfType<IntegerAttr>("spec_id")) {
      auto val = static_cast<uint32_t>(specID.getInt());
      (void)emitDecoration(resultID, spirv::Decoration::SpecId, {val});
    }

    specConstIDMap[op.sym_name()] = resultID;
    return processName(resultID, op.sym_name());
  }
  return failure();
}

LogicalResult
Serializer::processSpecConstantCompositeOp(spirv::SpecConstantCompositeOp op) {
  uint32_t typeID = 0;
  if (failed(processType(op.getLoc(), op.type(), typeID))) {
    return failure();
  }

  auto resultID = getNextID();

  SmallVector<uint32_t, 8> operands;
  operands.push_back(typeID);
  operands.push_back(resultID);

  auto constituents = op.constituents();

  for (auto index : llvm::seq<uint32_t>(0, constituents.size())) {
    auto constituent = constituents[index].dyn_cast<FlatSymbolRefAttr>();

    auto constituentName = constituent.getValue();
    auto constituentID = getSpecConstID(constituentName);

    if (!constituentID) {
      return op.emitError("unknown result <id> for specialization constant ")
             << constituentName;
    }

    operands.push_back(constituentID);
  }

  (void)encodeInstructionInto(typesGlobalValues,
                              spirv::Opcode::OpSpecConstantComposite, operands);
  specConstIDMap[op.sym_name()] = resultID;

  return processName(resultID, op.sym_name());
}

LogicalResult
Serializer::processSpecConstantOperationOp(spirv::SpecConstantOperationOp op) {
  uint32_t typeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), typeID))) {
    return failure();
  }

  auto resultID = getNextID();

  SmallVector<uint32_t, 8> operands;
  operands.push_back(typeID);
  operands.push_back(resultID);

  Block &block = op.getRegion().getBlocks().front();
  Operation &enclosedOp = block.getOperations().front();

  std::string enclosedOpName;
  llvm::raw_string_ostream rss(enclosedOpName);
  rss << "Op" << enclosedOp.getName().stripDialect();
  auto enclosedOpcode = spirv::symbolizeOpcode(rss.str());

  if (!enclosedOpcode) {
    op.emitError("Couldn't find op code for op ")
        << enclosedOp.getName().getStringRef();
    return failure();
  }

  operands.push_back(static_cast<uint32_t>(enclosedOpcode.getValue()));

  // Append operands to the enclosed op to the list of operands.
  for (Value operand : enclosedOp.getOperands()) {
    uint32_t id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }

  (void)encodeInstructionInto(typesGlobalValues,
                              spirv::Opcode::OpSpecConstantOp, operands);
  valueIDMap[op.getResult()] = resultID;

  return success();
}

LogicalResult Serializer::processUndefOp(spirv::UndefOp op) {
  auto undefType = op.getType();
  auto &id = undefValIDMap[undefType];
  if (!id) {
    id = getNextID();
    uint32_t typeID = 0;
    if (failed(processType(op.getLoc(), undefType, typeID)) ||
        failed(encodeInstructionInto(typesGlobalValues, spirv::Opcode::OpUndef,
                                     {typeID, id}))) {
      return failure();
    }
  }
  valueIDMap[op.getResult()] = id;
  return success();
}

LogicalResult Serializer::processFuncOp(spirv::FuncOp op) {
  LLVM_DEBUG(llvm::dbgs() << "-- start function '" << op.getName() << "' --\n");
  assert(functionHeader.empty() && functionBody.empty());

  uint32_t fnTypeID = 0;
  // Generate type of the function.
  (void)processType(op.getLoc(), op.getType(), fnTypeID);

  // Add the function definition.
  SmallVector<uint32_t, 4> operands;
  uint32_t resTypeID = 0;
  auto resultTypes = op.getType().getResults();
  if (resultTypes.size() > 1) {
    return op.emitError("cannot serialize function with multiple return types");
  }
  if (failed(processType(op.getLoc(),
                         (resultTypes.empty() ? getVoidType() : resultTypes[0]),
                         resTypeID))) {
    return failure();
  }
  operands.push_back(resTypeID);
  auto funcID = getOrCreateFunctionID(op.getName());
  operands.push_back(funcID);
  operands.push_back(static_cast<uint32_t>(op.function_control()));
  operands.push_back(fnTypeID);
  (void)encodeInstructionInto(functionHeader, spirv::Opcode::OpFunction,
                              operands);

  // Add function name.
  if (failed(processName(funcID, op.getName()))) {
    return failure();
  }

  // Declare the parameters.
  for (auto arg : op.getArguments()) {
    uint32_t argTypeID = 0;
    if (failed(processType(op.getLoc(), arg.getType(), argTypeID))) {
      return failure();
    }
    auto argValueID = getNextID();
    valueIDMap[arg] = argValueID;
    (void)encodeInstructionInto(functionHeader,
                                spirv::Opcode::OpFunctionParameter,
                                {argTypeID, argValueID});
  }

  // Process the body.
  if (op.isExternal()) {
    return op.emitError("external function is unhandled");
  }

  // Some instructions (e.g., OpVariable) in a function must be in the first
  // block in the function. These instructions will be put in functionHeader.
  // Thus, we put the label in functionHeader first, and omit it from the first
  // block.
  (void)encodeInstructionInto(functionHeader, spirv::Opcode::OpLabel,
                              {getOrCreateBlockID(&op.front())});
  (void)processBlock(&op.front(), /*omitLabel=*/true);
  if (failed(visitInPrettyBlockOrder(
          &op.front(), [&](Block *block) { return processBlock(block); },
          /*skipHeader=*/true))) {
    return failure();
  }

  // There might be OpPhi instructions who have value references needing to fix.
  for (auto deferredValue : deferredPhiValues) {
    Value value = deferredValue.first;
    uint32_t id = getValueID(value);
    LLVM_DEBUG(llvm::dbgs() << "[phi] fix reference of value " << value
                            << " to id = " << id << '\n');
    assert(id && "OpPhi references undefined value!");
    for (size_t offset : deferredValue.second)
      functionBody[offset] = id;
  }
  deferredPhiValues.clear();

  LLVM_DEBUG(llvm::dbgs() << "-- completed function '" << op.getName()
                          << "' --\n");
  // Insert OpFunctionEnd.
  if (failed(encodeInstructionInto(functionBody, spirv::Opcode::OpFunctionEnd,
                                   {}))) {
    return failure();
  }

  functions.append(functionHeader.begin(), functionHeader.end());
  functions.append(functionBody.begin(), functionBody.end());
  functionHeader.clear();
  functionBody.clear();

  return success();
}

LogicalResult Serializer::processVariableOp(spirv::VariableOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  auto attr = op->getAttr(spirv::attributeName<spirv::StorageClass>());
  if (attr) {
    operands.push_back(static_cast<uint32_t>(
        attr.cast<IntegerAttr>().getValue().getZExtValue()));
  }
  elidedAttrs.push_back(spirv::attributeName<spirv::StorageClass>());
  for (auto arg : op.getODSOperands(0)) {
    auto argID = getValueID(arg);
    if (!argID) {
      return emitError(op.getLoc(), "operand 0 has a use before def");
    }
    operands.push_back(argID);
  }
  (void)emitDebugLine(functionHeader, op.getLoc());
  (void)encodeInstructionInto(functionHeader, spirv::Opcode::OpVariable,
                              operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::any_of(elidedAttrs,
                     [&](StringRef elided) { return attr.first == elided; })) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

LogicalResult
Serializer::processGlobalVariableOp(spirv::GlobalVariableOp varOp) {
  // Get TypeID.
  uint32_t resultTypeID = 0;
  SmallVector<StringRef, 4> elidedAttrs;
  if (failed(processType(varOp.getLoc(), varOp.type(), resultTypeID))) {
    return failure();
  }

  if (isInterfaceStructPtrType(varOp.type())) {
    auto structType = varOp.type()
                          .cast<spirv::PointerType>()
                          .getPointeeType()
                          .cast<spirv::StructType>();
    if (failed(
            emitDecoration(getTypeID(structType), spirv::Decoration::Block))) {
      return varOp.emitError("cannot decorate ")
             << structType << " with Block decoration";
    }
  }

  elidedAttrs.push_back("type");
  SmallVector<uint32_t, 4> operands;
  operands.push_back(resultTypeID);
  auto resultID = getNextID();

  // Encode the name.
  auto varName = varOp.sym_name();
  elidedAttrs.push_back(SymbolTable::getSymbolAttrName());
  if (failed(processName(resultID, varName))) {
    return failure();
  }
  globalVarIDMap[varName] = resultID;
  operands.push_back(resultID);

  // Encode StorageClass.
  operands.push_back(static_cast<uint32_t>(varOp.storageClass()));

  // Encode initialization.
  if (auto initializer = varOp.initializer()) {
    auto initializerID = getVariableID(initializer.getValue());
    if (!initializerID) {
      return emitError(varOp.getLoc(),
                       "invalid usage of undefined variable as initializer");
    }
    operands.push_back(initializerID);
    elidedAttrs.push_back("initializer");
  }

  (void)emitDebugLine(typesGlobalValues, varOp.getLoc());
  if (failed(encodeInstructionInto(typesGlobalValues, spirv::Opcode::OpVariable,
                                   operands))) {
    elidedAttrs.push_back("initializer");
    return failure();
  }

  // Encode decorations.
  for (auto attr : varOp->getAttrs()) {
    if (llvm::any_of(elidedAttrs,
                     [&](StringRef elided) { return attr.first == elided; })) {
      continue;
    }
    if (failed(processDecoration(varOp.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

LogicalResult Serializer::processSelectionOp(spirv::SelectionOp selectionOp) {
  // Assign <id>s to all blocks so that branches inside the SelectionOp can
  // resolve properly.
  auto &body = selectionOp.body();
  for (Block &block : body)
    getOrCreateBlockID(&block);

  auto *headerBlock = selectionOp.getHeaderBlock();
  auto *mergeBlock = selectionOp.getMergeBlock();
  auto mergeID = getBlockID(mergeBlock);
  auto loc = selectionOp.getLoc();

  // Emit the selection header block, which dominates all other blocks, first.
  // We need to emit an OpSelectionMerge instruction before the selection header
  // block's terminator.
  auto emitSelectionMerge = [&]() {
    (void)emitDebugLine(functionBody, loc);
    lastProcessedWasMergeInst = true;
    (void)encodeInstructionInto(
        functionBody, spirv::Opcode::OpSelectionMerge,
        {mergeID, static_cast<uint32_t>(selectionOp.selection_control())});
  };
  // For structured selection, we cannot have blocks in the selection construct
  // branching to the selection header block. Entering the selection (and
  // reaching the selection header) must be from the block containing the
  // spv.mlir.selection op. If there are ops ahead of the spv.mlir.selection op
  // in the block, we can "merge" them into the selection header. So here we
  // don't need to emit a separate block; just continue with the existing block.
  if (failed(processBlock(headerBlock, /*omitLabel=*/true, emitSelectionMerge)))
    return failure();

  // Process all blocks with a depth-first visitor starting from the header
  // block. The selection header block and merge block are skipped by this
  // visitor.
  if (failed(visitInPrettyBlockOrder(
          headerBlock, [&](Block *block) { return processBlock(block); },
          /*skipHeader=*/true, /*skipBlocks=*/{mergeBlock})))
    return failure();

  // There is nothing to do for the merge block in the selection, which just
  // contains a spv.mlir.merge op, itself. But we need to have an OpLabel
  // instruction to start a new SPIR-V block for ops following this SelectionOp.
  // The block should use the <id> for the merge block.
  return encodeInstructionInto(functionBody, spirv::Opcode::OpLabel, {mergeID});
}

LogicalResult Serializer::processLoopOp(spirv::LoopOp loopOp) {
  // Assign <id>s to all blocks so that branches inside the LoopOp can resolve
  // properly. We don't need to assign for the entry block, which is just for
  // satisfying MLIR region's structural requirement.
  auto &body = loopOp.body();
  for (Block &block :
       llvm::make_range(std::next(body.begin(), 1), body.end())) {
    getOrCreateBlockID(&block);
  }
  auto *headerBlock = loopOp.getHeaderBlock();
  auto *continueBlock = loopOp.getContinueBlock();
  auto *mergeBlock = loopOp.getMergeBlock();
  auto headerID = getBlockID(headerBlock);
  auto continueID = getBlockID(continueBlock);
  auto mergeID = getBlockID(mergeBlock);
  auto loc = loopOp.getLoc();

  // This LoopOp is in some MLIR block with preceding and following ops. In the
  // binary format, it should reside in separate SPIR-V blocks from its
  // preceding and following ops. So we need to emit unconditional branches to
  // jump to this LoopOp's SPIR-V blocks and jumping back to the normal flow
  // afterwards.
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpBranch,
                              {headerID});

  // LoopOp's entry block is just there for satisfying MLIR's structural
  // requirements so we omit it and start serialization from the loop header
  // block.

  // Emit the loop header block, which dominates all other blocks, first. We
  // need to emit an OpLoopMerge instruction before the loop header block's
  // terminator.
  auto emitLoopMerge = [&]() {
    (void)emitDebugLine(functionBody, loc);
    lastProcessedWasMergeInst = true;
    (void)encodeInstructionInto(
        functionBody, spirv::Opcode::OpLoopMerge,
        {mergeID, continueID, static_cast<uint32_t>(loopOp.loop_control())});
  };
  if (failed(processBlock(headerBlock, /*omitLabel=*/false, emitLoopMerge)))
    return failure();

  // Process all blocks with a depth-first visitor starting from the header
  // block. The loop header block, loop continue block, and loop merge block are
  // skipped by this visitor and handled later in this function.
  if (failed(visitInPrettyBlockOrder(
          headerBlock, [&](Block *block) { return processBlock(block); },
          /*skipHeader=*/true, /*skipBlocks=*/{continueBlock, mergeBlock})))
    return failure();

  // We have handled all other blocks. Now get to the loop continue block.
  if (failed(processBlock(continueBlock)))
    return failure();

  // There is nothing to do for the merge block in the loop, which just contains
  // a spv.mlir.merge op, itself. But we need to have an OpLabel instruction to
  // start a new SPIR-V block for ops following this LoopOp. The block should
  // use the <id> for the merge block.
  return encodeInstructionInto(functionBody, spirv::Opcode::OpLabel, {mergeID});
}

LogicalResult Serializer::processBranchConditionalOp(
    spirv::BranchConditionalOp condBranchOp) {
  auto conditionID = getValueID(condBranchOp.condition());
  auto trueLabelID = getOrCreateBlockID(condBranchOp.getTrueBlock());
  auto falseLabelID = getOrCreateBlockID(condBranchOp.getFalseBlock());
  SmallVector<uint32_t, 5> arguments{conditionID, trueLabelID, falseLabelID};

  if (auto weights = condBranchOp.branch_weights()) {
    for (auto val : weights->getValue())
      arguments.push_back(val.cast<IntegerAttr>().getInt());
  }

  (void)emitDebugLine(functionBody, condBranchOp.getLoc());
  return encodeInstructionInto(functionBody, spirv::Opcode::OpBranchConditional,
                               arguments);
}

LogicalResult Serializer::processBranchOp(spirv::BranchOp branchOp) {
  (void)emitDebugLine(functionBody, branchOp.getLoc());
  return encodeInstructionInto(functionBody, spirv::Opcode::OpBranch,
                               {getOrCreateBlockID(branchOp.getTarget())});
}

LogicalResult Serializer::processAddressOfOp(spirv::AddressOfOp addressOfOp) {
  auto varName = addressOfOp.variable();
  auto variableID = getVariableID(varName);
  if (!variableID) {
    return addressOfOp.emitError("unknown result <id> for variable ")
           << varName;
  }
  valueIDMap[addressOfOp.pointer()] = variableID;
  return success();
}

LogicalResult
Serializer::processReferenceOfOp(spirv::ReferenceOfOp referenceOfOp) {
  auto constName = referenceOfOp.spec_const();
  auto constID = getSpecConstID(constName);
  if (!constID) {
    return referenceOfOp.emitError(
               "unknown result <id> for specialization constant ")
           << constName;
  }
  valueIDMap[referenceOfOp.reference()] = constID;
  return success();
}

template <>
LogicalResult
Serializer::processOp<spirv::EntryPointOp>(spirv::EntryPointOp op) {
  SmallVector<uint32_t, 4> operands;
  // Add the ExecutionModel.
  operands.push_back(static_cast<uint32_t>(op.execution_model()));
  // Add the function <id>.
  auto funcID = getFunctionID(op.fn());
  if (!funcID) {
    return op.emitError("missing <id> for function ")
           << op.fn()
           << "; function needs to be defined before spv.EntryPoint is "
              "serialized";
  }
  operands.push_back(funcID);
  // Add the name of the function.
  (void)spirv::encodeStringLiteralInto(operands, op.fn());

  // Add the interface values.
  if (auto interface = op.interface()) {
    for (auto var : interface.getValue()) {
      auto id = getVariableID(var.cast<FlatSymbolRefAttr>().getValue());
      if (!id) {
        return op.emitError("referencing undefined global variable."
                            "spv.EntryPoint is at the end of spv.module. All "
                            "referenced variables should already be defined");
      }
      operands.push_back(id);
    }
  }
  return encodeInstructionInto(entryPoints, spirv::Opcode::OpEntryPoint,
                               operands);
}

template <>
LogicalResult
Serializer::processOp<spirv::ControlBarrierOp>(spirv::ControlBarrierOp op) {
  StringRef argNames[] = {"execution_scope", "memory_scope",
                          "memory_semantics"};
  SmallVector<uint32_t, 3> operands;

  for (auto argName : argNames) {
    auto argIntAttr = op->getAttrOfType<IntegerAttr>(argName);
    auto operand = prepareConstantInt(op.getLoc(), argIntAttr);
    if (!operand) {
      return failure();
    }
    operands.push_back(operand);
  }

  return encodeInstructionInto(functionBody, spirv::Opcode::OpControlBarrier,
                               operands);
}

template <>
LogicalResult
Serializer::processOp<spirv::ExecutionModeOp>(spirv::ExecutionModeOp op) {
  SmallVector<uint32_t, 4> operands;
  // Add the function <id>.
  auto funcID = getFunctionID(op.fn());
  if (!funcID) {
    return op.emitError("missing <id> for function ")
           << op.fn()
           << "; function needs to be serialized before ExecutionModeOp is "
              "serialized";
  }
  operands.push_back(funcID);
  // Add the ExecutionMode.
  operands.push_back(static_cast<uint32_t>(op.execution_mode()));

  // Serialize values if any.
  auto values = op.values();
  if (values) {
    for (auto &intVal : values.getValue()) {
      operands.push_back(static_cast<uint32_t>(
          intVal.cast<IntegerAttr>().getValue().getZExtValue()));
    }
  }
  return encodeInstructionInto(executionModes, spirv::Opcode::OpExecutionMode,
                               operands);
}

template <>
LogicalResult
Serializer::processOp<spirv::MemoryBarrierOp>(spirv::MemoryBarrierOp op) {
  StringRef argNames[] = {"memory_scope", "memory_semantics"};
  SmallVector<uint32_t, 2> operands;

  for (auto argName : argNames) {
    auto argIntAttr = op->getAttrOfType<IntegerAttr>(argName);
    auto operand = prepareConstantInt(op.getLoc(), argIntAttr);
    if (!operand) {
      return failure();
    }
    operands.push_back(operand);
  }

  return encodeInstructionInto(functionBody, spirv::Opcode::OpMemoryBarrier,
                               operands);
}

template <>
LogicalResult
Serializer::processOp<spirv::FunctionCallOp>(spirv::FunctionCallOp op) {
  auto funcName = op.callee();
  uint32_t resTypeID = 0;

  Type resultTy = op.getNumResults() ? *op.result_type_begin() : getVoidType();
  if (failed(processType(op.getLoc(), resultTy, resTypeID)))
    return failure();

  auto funcID = getOrCreateFunctionID(funcName);
  auto funcCallID = getNextID();
  SmallVector<uint32_t, 8> operands{resTypeID, funcCallID, funcID};

  for (auto value : op.arguments()) {
    auto valueID = getValueID(value);
    assert(valueID && "cannot find a value for spv.FunctionCall");
    operands.push_back(valueID);
  }

  if (!resultTy.isa<NoneType>())
    valueIDMap[op.getResult(0)] = funcCallID;

  return encodeInstructionInto(functionBody, spirv::Opcode::OpFunctionCall,
                               operands);
}

template <>
LogicalResult
Serializer::processOp<spirv::CopyMemoryOp>(spirv::CopyMemoryOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;

  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }

  if (auto attr = op->getAttr("memory_access")) {
    operands.push_back(static_cast<uint32_t>(
        attr.cast<IntegerAttr>().getValue().getZExtValue()));
  }

  elidedAttrs.push_back("memory_access");

  if (auto attr = op->getAttr("alignment")) {
    operands.push_back(static_cast<uint32_t>(
        attr.cast<IntegerAttr>().getValue().getZExtValue()));
  }

  elidedAttrs.push_back("alignment");

  if (auto attr = op->getAttr("source_memory_access")) {
    operands.push_back(static_cast<uint32_t>(
        attr.cast<IntegerAttr>().getValue().getZExtValue()));
  }

  elidedAttrs.push_back("source_memory_access");

  if (auto attr = op->getAttr("source_alignment")) {
    operands.push_back(static_cast<uint32_t>(
        attr.cast<IntegerAttr>().getValue().getZExtValue()));
  }

  elidedAttrs.push_back("source_alignment");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpCopyMemory,
                              operands);

  return success();
}

// Pull in auto-generated Serializer::dispatchToAutogenSerialization() and
// various Serializer::processOp<...>() specializations.
#define GET_SERIALIZATION_FNS
#include "mlir/Dialect/SPIRV/IR/SPIRVSerialization.inc"

} // namespace spirv
} // namespace mlir
