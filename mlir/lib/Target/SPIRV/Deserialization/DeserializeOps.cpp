//===- DeserializeOps.cpp - MLIR SPIR-V Deserialization (Ops) -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Deserializer methods for SPIR-V binary instructions.
//
//===----------------------------------------------------------------------===//

#include "Deserializer.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define DEBUG_TYPE "spirv-deserialization"

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Extracts the opcode from the given first word of a SPIR-V instruction.
static inline spirv::Opcode extractOpcode(uint32_t word) {
  return static_cast<spirv::Opcode>(word & 0xffff);
}

//===----------------------------------------------------------------------===//
// Instruction
//===----------------------------------------------------------------------===//

Value spirv::Deserializer::getValue(uint32_t id) {
  if (auto constInfo = getConstant(id)) {
    // Materialize a `spv.Constant` op at every use site.
    return opBuilder.create<spirv::ConstantOp>(unknownLoc, constInfo->second,
                                               constInfo->first);
  }
  if (auto varOp = getGlobalVariable(id)) {
    auto addressOfOp = opBuilder.create<spirv::AddressOfOp>(
        unknownLoc, varOp.type(),
        opBuilder.getSymbolRefAttr(varOp.getOperation()));
    return addressOfOp.pointer();
  }
  if (auto constOp = getSpecConstant(id)) {
    auto referenceOfOp = opBuilder.create<spirv::ReferenceOfOp>(
        unknownLoc, constOp.default_value().getType(),
        opBuilder.getSymbolRefAttr(constOp.getOperation()));
    return referenceOfOp.reference();
  }
  if (auto constCompositeOp = getSpecConstantComposite(id)) {
    auto referenceOfOp = opBuilder.create<spirv::ReferenceOfOp>(
        unknownLoc, constCompositeOp.type(),
        opBuilder.getSymbolRefAttr(constCompositeOp.getOperation()));
    return referenceOfOp.reference();
  }
  if (auto specConstOperationInfo = getSpecConstantOperation(id)) {
    return materializeSpecConstantOperation(
        id, specConstOperationInfo->enclodesOpcode,
        specConstOperationInfo->resultTypeID,
        specConstOperationInfo->enclosedOpOperands);
  }
  if (auto undef = getUndefType(id)) {
    return opBuilder.create<spirv::UndefOp>(unknownLoc, undef);
  }
  return valueMap.lookup(id);
}

LogicalResult
spirv::Deserializer::sliceInstruction(spirv::Opcode &opcode,
                                      ArrayRef<uint32_t> &operands,
                                      Optional<spirv::Opcode> expectedOpcode) {
  auto binarySize = binary.size();
  if (curOffset >= binarySize) {
    return emitError(unknownLoc, "expected ")
           << (expectedOpcode ? spirv::stringifyOpcode(*expectedOpcode)
                              : "more")
           << " instruction";
  }

  // For each instruction, get its word count from the first word to slice it
  // from the stream properly, and then dispatch to the instruction handler.

  uint32_t wordCount = binary[curOffset] >> 16;

  if (wordCount == 0)
    return emitError(unknownLoc, "word count cannot be zero");

  uint32_t nextOffset = curOffset + wordCount;
  if (nextOffset > binarySize)
    return emitError(unknownLoc, "insufficient words for the last instruction");

  opcode = extractOpcode(binary[curOffset]);
  operands = binary.slice(curOffset + 1, wordCount - 1);
  curOffset = nextOffset;
  return success();
}

LogicalResult spirv::Deserializer::processInstruction(
    spirv::Opcode opcode, ArrayRef<uint32_t> operands, bool deferInstructions) {
  LLVM_DEBUG(llvm::dbgs() << "[inst] processing instruction "
                          << spirv::stringifyOpcode(opcode) << "\n");

  // First dispatch all the instructions whose opcode does not correspond to
  // those that have a direct mirror in the SPIR-V dialect
  switch (opcode) {
  case spirv::Opcode::OpCapability:
    return processCapability(operands);
  case spirv::Opcode::OpExtension:
    return processExtension(operands);
  case spirv::Opcode::OpExtInst:
    return processExtInst(operands);
  case spirv::Opcode::OpExtInstImport:
    return processExtInstImport(operands);
  case spirv::Opcode::OpMemberName:
    return processMemberName(operands);
  case spirv::Opcode::OpMemoryModel:
    return processMemoryModel(operands);
  case spirv::Opcode::OpEntryPoint:
  case spirv::Opcode::OpExecutionMode:
    if (deferInstructions) {
      deferredInstructions.emplace_back(opcode, operands);
      return success();
    }
    break;
  case spirv::Opcode::OpVariable:
    if (isa<spirv::ModuleOp>(opBuilder.getBlock()->getParentOp())) {
      return processGlobalVariable(operands);
    }
    break;
  case spirv::Opcode::OpLine:
    return processDebugLine(operands);
  case spirv::Opcode::OpNoLine:
    return clearDebugLine();
  case spirv::Opcode::OpName:
    return processName(operands);
  case spirv::Opcode::OpString:
    return processDebugString(operands);
  case spirv::Opcode::OpModuleProcessed:
  case spirv::Opcode::OpSource:
  case spirv::Opcode::OpSourceContinued:
  case spirv::Opcode::OpSourceExtension:
    // TODO: This is debug information embedded in the binary which should be
    // translated into the spv.module.
    return success();
  case spirv::Opcode::OpTypeVoid:
  case spirv::Opcode::OpTypeBool:
  case spirv::Opcode::OpTypeInt:
  case spirv::Opcode::OpTypeFloat:
  case spirv::Opcode::OpTypeVector:
  case spirv::Opcode::OpTypeMatrix:
  case spirv::Opcode::OpTypeArray:
  case spirv::Opcode::OpTypeFunction:
  case spirv::Opcode::OpTypeImage:
  case spirv::Opcode::OpTypeSampledImage:
  case spirv::Opcode::OpTypeRuntimeArray:
  case spirv::Opcode::OpTypeStruct:
  case spirv::Opcode::OpTypePointer:
  case spirv::Opcode::OpTypeCooperativeMatrixNV:
    return processType(opcode, operands);
  case spirv::Opcode::OpTypeForwardPointer:
    return processTypeForwardPointer(operands);
  case spirv::Opcode::OpConstant:
    return processConstant(operands, /*isSpec=*/false);
  case spirv::Opcode::OpSpecConstant:
    return processConstant(operands, /*isSpec=*/true);
  case spirv::Opcode::OpConstantComposite:
    return processConstantComposite(operands);
  case spirv::Opcode::OpSpecConstantComposite:
    return processSpecConstantComposite(operands);
  case spirv::Opcode::OpSpecConstantOp:
    return processSpecConstantOperation(operands);
  case spirv::Opcode::OpConstantTrue:
    return processConstantBool(/*isTrue=*/true, operands, /*isSpec=*/false);
  case spirv::Opcode::OpSpecConstantTrue:
    return processConstantBool(/*isTrue=*/true, operands, /*isSpec=*/true);
  case spirv::Opcode::OpConstantFalse:
    return processConstantBool(/*isTrue=*/false, operands, /*isSpec=*/false);
  case spirv::Opcode::OpSpecConstantFalse:
    return processConstantBool(/*isTrue=*/false, operands, /*isSpec=*/true);
  case spirv::Opcode::OpConstantNull:
    return processConstantNull(operands);
  case spirv::Opcode::OpDecorate:
    return processDecoration(operands);
  case spirv::Opcode::OpMemberDecorate:
    return processMemberDecoration(operands);
  case spirv::Opcode::OpFunction:
    return processFunction(operands);
  case spirv::Opcode::OpLabel:
    return processLabel(operands);
  case spirv::Opcode::OpBranch:
    return processBranch(operands);
  case spirv::Opcode::OpBranchConditional:
    return processBranchConditional(operands);
  case spirv::Opcode::OpSelectionMerge:
    return processSelectionMerge(operands);
  case spirv::Opcode::OpLoopMerge:
    return processLoopMerge(operands);
  case spirv::Opcode::OpPhi:
    return processPhi(operands);
  case spirv::Opcode::OpUndef:
    return processUndef(operands);
  default:
    break;
  }
  return dispatchToAutogenDeserialization(opcode, operands);
}

LogicalResult spirv::Deserializer::processOpWithoutGrammarAttr(
    ArrayRef<uint32_t> words, StringRef opName, bool hasResult,
    unsigned numOperands) {
  SmallVector<Type, 1> resultTypes;
  uint32_t valueID = 0;

  size_t wordIndex = 0;
  if (hasResult) {
    if (wordIndex >= words.size())
      return emitError(unknownLoc,
                       "expected result type <id> while deserializing for ")
             << opName;

    // Decode the type <id>
    auto type = getType(words[wordIndex]);
    if (!type)
      return emitError(unknownLoc, "unknown type result <id>: ")
             << words[wordIndex];
    resultTypes.push_back(type);
    ++wordIndex;

    // Decode the result <id>
    if (wordIndex >= words.size())
      return emitError(unknownLoc,
                       "expected result <id> while deserializing for ")
             << opName;
    valueID = words[wordIndex];
    ++wordIndex;
  }

  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;

  // Decode operands
  size_t operandIndex = 0;
  for (; operandIndex < numOperands && wordIndex < words.size();
       ++operandIndex, ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg)
      return emitError(unknownLoc, "unknown result <id>: ") << words[wordIndex];
    operands.push_back(arg);
  }
  if (operandIndex != numOperands) {
    return emitError(
               unknownLoc,
               "found less operands than expected when deserializing for ")
           << opName << "; only " << operandIndex << " of " << numOperands
           << " processed";
  }
  if (wordIndex != words.size()) {
    return emitError(
               unknownLoc,
               "found more operands than expected when deserializing for ")
           << opName << "; only " << wordIndex << " of " << words.size()
           << " processed";
  }

  // Attach attributes from decorations
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }

  // Create the op and update bookkeeping maps
  Location loc = createFileLineColLoc(opBuilder);
  OperationState opState(loc, opName);
  opState.addOperands(operands);
  if (hasResult)
    opState.addTypes(resultTypes);
  opState.addAttributes(attributes);
  Operation *op = opBuilder.createOperation(opState);
  if (hasResult)
    valueMap[valueID] = op->getResult(0);

  if (op->hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();

  return success();
}

LogicalResult spirv::Deserializer::processUndef(ArrayRef<uint32_t> operands) {
  if (operands.size() != 2) {
    return emitError(unknownLoc, "OpUndef instruction must have two operands");
  }
  auto type = getType(operands[0]);
  if (!type) {
    return emitError(unknownLoc, "unknown type <id> with OpUndef instruction");
  }
  undefMap[operands[1]] = type;
  return success();
}

LogicalResult spirv::Deserializer::processExtInst(ArrayRef<uint32_t> operands) {
  if (operands.size() < 4) {
    return emitError(unknownLoc,
                     "OpExtInst must have at least 4 operands, result type "
                     "<id>, result <id>, set <id> and instruction opcode");
  }
  if (!extendedInstSets.count(operands[2])) {
    return emitError(unknownLoc, "undefined set <id> in OpExtInst");
  }
  SmallVector<uint32_t, 4> slicedOperands;
  slicedOperands.append(operands.begin(), std::next(operands.begin(), 2));
  slicedOperands.append(std::next(operands.begin(), 4), operands.end());
  return dispatchToExtensionSetAutogenDeserialization(
      extendedInstSets[operands[2]], operands[3], slicedOperands);
}

namespace mlir {
namespace spirv {

template <>
LogicalResult
Deserializer::processOp<spirv::EntryPointOp>(ArrayRef<uint32_t> words) {
  unsigned wordIndex = 0;
  if (wordIndex >= words.size()) {
    return emitError(unknownLoc,
                     "missing Execution Model specification in OpEntryPoint");
  }
  auto execModel = opBuilder.getI32IntegerAttr(words[wordIndex++]);
  if (wordIndex >= words.size()) {
    return emitError(unknownLoc, "missing <id> in OpEntryPoint");
  }
  // Get the function <id>
  auto fnID = words[wordIndex++];
  // Get the function name
  auto fnName = decodeStringLiteral(words, wordIndex);
  // Verify that the function <id> matches the fnName
  auto parsedFunc = getFunction(fnID);
  if (!parsedFunc) {
    return emitError(unknownLoc, "no function matching <id> ") << fnID;
  }
  if (parsedFunc.getName() != fnName) {
    return emitError(unknownLoc, "function name mismatch between OpEntryPoint "
                                 "and OpFunction with <id> ")
           << fnID << ": " << fnName << " vs. " << parsedFunc.getName();
  }
  SmallVector<Attribute, 4> interface;
  while (wordIndex < words.size()) {
    auto arg = getGlobalVariable(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "undefined result <id> ")
             << words[wordIndex] << " while decoding OpEntryPoint";
    }
    interface.push_back(opBuilder.getSymbolRefAttr(arg.getOperation()));
    wordIndex++;
  }
  opBuilder.create<spirv::EntryPointOp>(unknownLoc, execModel,
                                        opBuilder.getSymbolRefAttr(fnName),
                                        opBuilder.getArrayAttr(interface));
  return success();
}

template <>
LogicalResult
Deserializer::processOp<spirv::ExecutionModeOp>(ArrayRef<uint32_t> words) {
  unsigned wordIndex = 0;
  if (wordIndex >= words.size()) {
    return emitError(unknownLoc,
                     "missing function result <id> in OpExecutionMode");
  }
  // Get the function <id> to get the name of the function
  auto fnID = words[wordIndex++];
  auto fn = getFunction(fnID);
  if (!fn) {
    return emitError(unknownLoc, "no function matching <id> ") << fnID;
  }
  // Get the Execution mode
  if (wordIndex >= words.size()) {
    return emitError(unknownLoc, "missing Execution Mode in OpExecutionMode");
  }
  auto execMode = opBuilder.getI32IntegerAttr(words[wordIndex++]);

  // Get the values
  SmallVector<Attribute, 4> attrListElems;
  while (wordIndex < words.size()) {
    attrListElems.push_back(opBuilder.getI32IntegerAttr(words[wordIndex++]));
  }
  auto values = opBuilder.getArrayAttr(attrListElems);
  opBuilder.create<spirv::ExecutionModeOp>(
      unknownLoc, opBuilder.getSymbolRefAttr(fn.getName()), execMode, values);
  return success();
}

template <>
LogicalResult
Deserializer::processOp<spirv::ControlBarrierOp>(ArrayRef<uint32_t> operands) {
  if (operands.size() != 3) {
    return emitError(
        unknownLoc,
        "OpControlBarrier must have execution scope <id>, memory scope <id> "
        "and memory semantics <id>");
  }

  SmallVector<IntegerAttr, 3> argAttrs;
  for (auto operand : operands) {
    auto argAttr = getConstantInt(operand);
    if (!argAttr) {
      return emitError(unknownLoc,
                       "expected 32-bit integer constant from <id> ")
             << operand << " for OpControlBarrier";
    }
    argAttrs.push_back(argAttr);
  }

  opBuilder.create<spirv::ControlBarrierOp>(unknownLoc, argAttrs[0],
                                            argAttrs[1], argAttrs[2]);
  return success();
}

template <>
LogicalResult
Deserializer::processOp<spirv::FunctionCallOp>(ArrayRef<uint32_t> operands) {
  if (operands.size() < 3) {
    return emitError(unknownLoc,
                     "OpFunctionCall must have at least 3 operands");
  }

  Type resultType = getType(operands[0]);
  if (!resultType) {
    return emitError(unknownLoc, "undefined result type from <id> ")
           << operands[0];
  }

  // Use null type to mean no result type.
  if (isVoidType(resultType))
    resultType = nullptr;

  auto resultID = operands[1];
  auto functionID = operands[2];

  auto functionName = getFunctionSymbol(functionID);

  SmallVector<Value, 4> arguments;
  for (auto operand : llvm::drop_begin(operands, 3)) {
    auto value = getValue(operand);
    if (!value) {
      return emitError(unknownLoc, "unknown <id> ")
             << operand << " used by OpFunctionCall";
    }
    arguments.push_back(value);
  }

  auto opFunctionCall = opBuilder.create<spirv::FunctionCallOp>(
      unknownLoc, resultType, opBuilder.getSymbolRefAttr(functionName),
      arguments);

  if (resultType)
    valueMap[resultID] = opFunctionCall.getResult(0);
  return success();
}

template <>
LogicalResult
Deserializer::processOp<spirv::MemoryBarrierOp>(ArrayRef<uint32_t> operands) {
  if (operands.size() != 2) {
    return emitError(unknownLoc, "OpMemoryBarrier must have memory scope <id> "
                                 "and memory semantics <id>");
  }

  SmallVector<IntegerAttr, 2> argAttrs;
  for (auto operand : operands) {
    auto argAttr = getConstantInt(operand);
    if (!argAttr) {
      return emitError(unknownLoc,
                       "expected 32-bit integer constant from <id> ")
             << operand << " for OpMemoryBarrier";
    }
    argAttrs.push_back(argAttr);
  }

  opBuilder.create<spirv::MemoryBarrierOp>(unknownLoc, argAttrs[0],
                                           argAttrs[1]);
  return success();
}

template <>
LogicalResult
Deserializer::processOp<spirv::CopyMemoryOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0;
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;

  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);

    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ")
             << words[wordIndex];
    }

    operands.push_back(arg);
    wordIndex++;
  }

  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);

    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ")
             << words[wordIndex];
    }

    operands.push_back(arg);
    wordIndex++;
  }

  bool isAlignedAttr = false;

  if (wordIndex < words.size()) {
    auto attrValue = words[wordIndex++];
    attributes.push_back(opBuilder.getNamedAttr(
        "memory_access", opBuilder.getI32IntegerAttr(attrValue)));
    isAlignedAttr = (attrValue == 2);
  }

  if (isAlignedAttr && wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr(
        "alignment", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }

  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr(
        "source_memory_access",
        opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }

  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr(
        "source_alignment", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }

  if (wordIndex != words.size()) {
    return emitError(unknownLoc,
                     "found more operands than expected when deserializing "
                     "spirv::CopyMemoryOp, only ")
           << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  opBuilder.create<spirv::CopyMemoryOp>(loc, resultTypes, operands, attributes);

  return success();
}

// Pull in auto-generated Deserializer::dispatchToAutogenDeserialization() and
// various Deserializer::processOp<...>() specializations.
#define GET_DESERIALIZATION_FNS
#include "mlir/Dialect/SPIRV/IR/SPIRVSerialization.inc"

} // namespace spirv
} // namespace mlir
