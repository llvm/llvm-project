//===- Deserializer.cpp - MLIR SPIR-V Deserializer ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SPIR-V binary to MLIR SPIR-V module deserializer.
//
//===----------------------------------------------------------------------===//

#include "Deserializer.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVModule.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/SPIRV/SPIRVBinaryUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

#define DEBUG_TYPE "spirv-deserialization"

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Returns true if the given `block` is a function entry block.
static inline bool isFnEntryBlock(Block *block) {
  return block->isEntryBlock() &&
         isa_and_nonnull<spirv::FuncOp>(block->getParentOp());
}

//===----------------------------------------------------------------------===//
// Deserializer Method Definitions
//===----------------------------------------------------------------------===//

spirv::Deserializer::Deserializer(ArrayRef<uint32_t> binary,
                                  MLIRContext *context)
    : binary(binary), context(context), unknownLoc(UnknownLoc::get(context)),
      module(createModuleOp()), opBuilder(module->body()) {}

LogicalResult spirv::Deserializer::deserialize() {
  LLVM_DEBUG(llvm::dbgs() << "+++ starting deserialization +++\n");

  if (failed(processHeader()))
    return failure();

  spirv::Opcode opcode = spirv::Opcode::OpNop;
  ArrayRef<uint32_t> operands;
  auto binarySize = binary.size();
  while (curOffset < binarySize) {
    // Slice the next instruction out and populate `opcode` and `operands`.
    // Internally this also updates `curOffset`.
    if (failed(sliceInstruction(opcode, operands)))
      return failure();

    if (failed(processInstruction(opcode, operands)))
      return failure();
  }

  assert(curOffset == binarySize &&
         "deserializer should never index beyond the binary end");

  for (auto &deferred : deferredInstructions) {
    if (failed(processInstruction(deferred.first, deferred.second, false))) {
      return failure();
    }
  }

  attachVCETriple();

  LLVM_DEBUG(llvm::dbgs() << "+++ completed deserialization +++\n");
  return success();
}

spirv::OwningSPIRVModuleRef spirv::Deserializer::collect() {
  return std::move(module);
}

//===----------------------------------------------------------------------===//
// Module structure
//===----------------------------------------------------------------------===//

spirv::OwningSPIRVModuleRef spirv::Deserializer::createModuleOp() {
  OpBuilder builder(context);
  OperationState state(unknownLoc, spirv::ModuleOp::getOperationName());
  spirv::ModuleOp::build(builder, state);
  return cast<spirv::ModuleOp>(Operation::create(state));
}

LogicalResult spirv::Deserializer::processHeader() {
  if (binary.size() < spirv::kHeaderWordCount)
    return emitError(unknownLoc,
                     "SPIR-V binary module must have a 5-word header");

  if (binary[0] != spirv::kMagicNumber)
    return emitError(unknownLoc, "incorrect magic number");

  // Version number bytes: 0 | major number | minor number | 0
  uint32_t majorVersion = (binary[1] << 8) >> 24;
  uint32_t minorVersion = (binary[1] << 16) >> 24;
  if (majorVersion == 1) {
    switch (minorVersion) {
#define MIN_VERSION_CASE(v)                                                    \
  case v:                                                                      \
    version = spirv::Version::V_1_##v;                                         \
    break

      MIN_VERSION_CASE(0);
      MIN_VERSION_CASE(1);
      MIN_VERSION_CASE(2);
      MIN_VERSION_CASE(3);
      MIN_VERSION_CASE(4);
      MIN_VERSION_CASE(5);
#undef MIN_VERSION_CASE
    default:
      return emitError(unknownLoc, "unsupported SPIR-V minor version: ")
             << minorVersion;
    }
  } else {
    return emitError(unknownLoc, "unsupported SPIR-V major version: ")
           << majorVersion;
  }

  // TODO: generator number, bound, schema
  curOffset = spirv::kHeaderWordCount;
  return success();
}

LogicalResult
spirv::Deserializer::processCapability(ArrayRef<uint32_t> operands) {
  if (operands.size() != 1)
    return emitError(unknownLoc, "OpMemoryModel must have one parameter");

  auto cap = spirv::symbolizeCapability(operands[0]);
  if (!cap)
    return emitError(unknownLoc, "unknown capability: ") << operands[0];

  capabilities.insert(*cap);
  return success();
}

LogicalResult spirv::Deserializer::processExtension(ArrayRef<uint32_t> words) {
  if (words.empty()) {
    return emitError(
        unknownLoc,
        "OpExtension must have a literal string for the extension name");
  }

  unsigned wordIndex = 0;
  StringRef extName = decodeStringLiteral(words, wordIndex);
  if (wordIndex != words.size())
    return emitError(unknownLoc,
                     "unexpected trailing words in OpExtension instruction");
  auto ext = spirv::symbolizeExtension(extName);
  if (!ext)
    return emitError(unknownLoc, "unknown extension: ") << extName;

  extensions.insert(*ext);
  return success();
}

LogicalResult
spirv::Deserializer::processExtInstImport(ArrayRef<uint32_t> words) {
  if (words.size() < 2) {
    return emitError(unknownLoc,
                     "OpExtInstImport must have a result <id> and a literal "
                     "string for the extended instruction set name");
  }

  unsigned wordIndex = 1;
  extendedInstSets[words[0]] = decodeStringLiteral(words, wordIndex);
  if (wordIndex != words.size()) {
    return emitError(unknownLoc,
                     "unexpected trailing words in OpExtInstImport");
  }
  return success();
}

void spirv::Deserializer::attachVCETriple() {
  (*module)->setAttr(
      spirv::ModuleOp::getVCETripleAttrName(),
      spirv::VerCapExtAttr::get(version, capabilities.getArrayRef(),
                                extensions.getArrayRef(), context));
}

LogicalResult
spirv::Deserializer::processMemoryModel(ArrayRef<uint32_t> operands) {
  if (operands.size() != 2)
    return emitError(unknownLoc, "OpMemoryModel must have two operands");

  (*module)->setAttr(
      "addressing_model",
      opBuilder.getI32IntegerAttr(llvm::bit_cast<int32_t>(operands.front())));
  (*module)->setAttr(
      "memory_model",
      opBuilder.getI32IntegerAttr(llvm::bit_cast<int32_t>(operands.back())));

  return success();
}

LogicalResult spirv::Deserializer::processDecoration(ArrayRef<uint32_t> words) {
  // TODO: This function should also be auto-generated. For now, since only a
  // few decorations are processed/handled in a meaningful manner, going with a
  // manual implementation.
  if (words.size() < 2) {
    return emitError(
        unknownLoc, "OpDecorate must have at least result <id> and Decoration");
  }
  auto decorationName =
      stringifyDecoration(static_cast<spirv::Decoration>(words[1]));
  if (decorationName.empty()) {
    return emitError(unknownLoc, "invalid Decoration code : ") << words[1];
  }
  auto attrName = llvm::convertToSnakeFromCamelCase(decorationName);
  auto symbol = opBuilder.getIdentifier(attrName);
  switch (static_cast<spirv::Decoration>(words[1])) {
  case spirv::Decoration::DescriptorSet:
  case spirv::Decoration::Binding:
    if (words.size() != 3) {
      return emitError(unknownLoc, "OpDecorate with ")
             << decorationName << " needs a single integer literal";
    }
    decorations[words[0]].set(
        symbol, opBuilder.getI32IntegerAttr(static_cast<int32_t>(words[2])));
    break;
  case spirv::Decoration::BuiltIn:
    if (words.size() != 3) {
      return emitError(unknownLoc, "OpDecorate with ")
             << decorationName << " needs a single integer literal";
    }
    decorations[words[0]].set(
        symbol, opBuilder.getStringAttr(
                    stringifyBuiltIn(static_cast<spirv::BuiltIn>(words[2]))));
    break;
  case spirv::Decoration::ArrayStride:
    if (words.size() != 3) {
      return emitError(unknownLoc, "OpDecorate with ")
             << decorationName << " needs a single integer literal";
    }
    typeDecorations[words[0]] = words[2];
    break;
  case spirv::Decoration::Aliased:
  case spirv::Decoration::Block:
  case spirv::Decoration::BufferBlock:
  case spirv::Decoration::Flat:
  case spirv::Decoration::NonReadable:
  case spirv::Decoration::NonWritable:
  case spirv::Decoration::NoPerspective:
  case spirv::Decoration::Restrict:
    if (words.size() != 2) {
      return emitError(unknownLoc, "OpDecoration with ")
             << decorationName << "needs a single target <id>";
    }
    // Block decoration does not affect spv.struct type, but is still stored for
    // verification.
    // TODO: Update StructType to contain this information since
    // it is needed for many validation rules.
    decorations[words[0]].set(symbol, opBuilder.getUnitAttr());
    break;
  case spirv::Decoration::Location:
  case spirv::Decoration::SpecId:
    if (words.size() != 3) {
      return emitError(unknownLoc, "OpDecoration with ")
             << decorationName << "needs a single integer literal";
    }
    decorations[words[0]].set(
        symbol, opBuilder.getI32IntegerAttr(static_cast<int32_t>(words[2])));
    break;
  default:
    return emitError(unknownLoc, "unhandled Decoration : '") << decorationName;
  }
  return success();
}

LogicalResult
spirv::Deserializer::processMemberDecoration(ArrayRef<uint32_t> words) {
  // The binary layout of OpMemberDecorate is different comparing to OpDecorate
  if (words.size() < 3) {
    return emitError(unknownLoc,
                     "OpMemberDecorate must have at least 3 operands");
  }

  auto decoration = static_cast<spirv::Decoration>(words[2]);
  if (decoration == spirv::Decoration::Offset && words.size() != 4) {
    return emitError(unknownLoc,
                     " missing offset specification in OpMemberDecorate with "
                     "Offset decoration");
  }
  ArrayRef<uint32_t> decorationOperands;
  if (words.size() > 3) {
    decorationOperands = words.slice(3);
  }
  memberDecorationMap[words[0]][words[1]][decoration] = decorationOperands;
  return success();
}

LogicalResult spirv::Deserializer::processMemberName(ArrayRef<uint32_t> words) {
  if (words.size() < 3) {
    return emitError(unknownLoc, "OpMemberName must have at least 3 operands");
  }
  unsigned wordIndex = 2;
  auto name = decodeStringLiteral(words, wordIndex);
  if (wordIndex != words.size()) {
    return emitError(unknownLoc,
                     "unexpected trailing words in OpMemberName instruction");
  }
  memberNameMap[words[0]][words[1]] = name;
  return success();
}

LogicalResult
spirv::Deserializer::processFunction(ArrayRef<uint32_t> operands) {
  if (curFunction) {
    return emitError(unknownLoc, "found function inside function");
  }

  // Get the result type
  if (operands.size() != 4) {
    return emitError(unknownLoc, "OpFunction must have 4 parameters");
  }
  Type resultType = getType(operands[0]);
  if (!resultType) {
    return emitError(unknownLoc, "undefined result type from <id> ")
           << operands[0];
  }

  if (funcMap.count(operands[1])) {
    return emitError(unknownLoc, "duplicate function definition/declaration");
  }

  auto fnControl = spirv::symbolizeFunctionControl(operands[2]);
  if (!fnControl) {
    return emitError(unknownLoc, "unknown Function Control: ") << operands[2];
  }

  Type fnType = getType(operands[3]);
  if (!fnType || !fnType.isa<FunctionType>()) {
    return emitError(unknownLoc, "unknown function type from <id> ")
           << operands[3];
  }
  auto functionType = fnType.cast<FunctionType>();

  if ((isVoidType(resultType) && functionType.getNumResults() != 0) ||
      (functionType.getNumResults() == 1 &&
       functionType.getResult(0) != resultType)) {
    return emitError(unknownLoc, "mismatch in function type ")
           << functionType << " and return type " << resultType << " specified";
  }

  std::string fnName = getFunctionSymbol(operands[1]);
  auto funcOp = opBuilder.create<spirv::FuncOp>(
      unknownLoc, fnName, functionType, fnControl.getValue());
  curFunction = funcMap[operands[1]] = funcOp;
  LLVM_DEBUG(llvm::dbgs() << "-- start function " << fnName << " (type = "
                          << fnType << ", id = " << operands[1] << ") --\n");
  auto *entryBlock = funcOp.addEntryBlock();
  LLVM_DEBUG(llvm::dbgs() << "[block] created entry block " << entryBlock
                          << "\n");

  // Parse the op argument instructions
  if (functionType.getNumInputs()) {
    for (size_t i = 0, e = functionType.getNumInputs(); i != e; ++i) {
      auto argType = functionType.getInput(i);
      spirv::Opcode opcode = spirv::Opcode::OpNop;
      ArrayRef<uint32_t> operands;
      if (failed(sliceInstruction(opcode, operands,
                                  spirv::Opcode::OpFunctionParameter))) {
        return failure();
      }
      if (opcode != spirv::Opcode::OpFunctionParameter) {
        return emitError(
                   unknownLoc,
                   "missing OpFunctionParameter instruction for argument ")
               << i;
      }
      if (operands.size() != 2) {
        return emitError(
            unknownLoc,
            "expected result type and result <id> for OpFunctionParameter");
      }
      auto argDefinedType = getType(operands[0]);
      if (!argDefinedType || argDefinedType != argType) {
        return emitError(unknownLoc,
                         "mismatch in argument type between function type "
                         "definition ")
               << functionType << " and argument type definition "
               << argDefinedType << " at argument " << i;
      }
      if (getValue(operands[1])) {
        return emitError(unknownLoc, "duplicate definition of result <id> '")
               << operands[1];
      }
      auto argValue = funcOp.getArgument(i);
      valueMap[operands[1]] = argValue;
    }
  }

  // RAII guard to reset the insertion point to the module's region after
  // deserializing the body of this function.
  OpBuilder::InsertionGuard moduleInsertionGuard(opBuilder);

  spirv::Opcode opcode = spirv::Opcode::OpNop;
  ArrayRef<uint32_t> instOperands;

  // Special handling for the entry block. We need to make sure it starts with
  // an OpLabel instruction. The entry block takes the same parameters as the
  // function. All other blocks do not take any parameter. We have already
  // created the entry block, here we need to register it to the correct label
  // <id>.
  if (failed(sliceInstruction(opcode, instOperands,
                              spirv::Opcode::OpFunctionEnd))) {
    return failure();
  }
  if (opcode == spirv::Opcode::OpFunctionEnd) {
    LLVM_DEBUG(llvm::dbgs()
               << "-- completed function '" << fnName << "' (type = " << fnType
               << ", id = " << operands[1] << ") --\n");
    return processFunctionEnd(instOperands);
  }
  if (opcode != spirv::Opcode::OpLabel) {
    return emitError(unknownLoc, "a basic block must start with OpLabel");
  }
  if (instOperands.size() != 1) {
    return emitError(unknownLoc, "OpLabel should only have result <id>");
  }
  blockMap[instOperands[0]] = entryBlock;
  if (failed(processLabel(instOperands))) {
    return failure();
  }

  // Then process all the other instructions in the function until we hit
  // OpFunctionEnd.
  while (succeeded(sliceInstruction(opcode, instOperands,
                                    spirv::Opcode::OpFunctionEnd)) &&
         opcode != spirv::Opcode::OpFunctionEnd) {
    if (failed(processInstruction(opcode, instOperands))) {
      return failure();
    }
  }
  if (opcode != spirv::Opcode::OpFunctionEnd) {
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "-- completed function '" << fnName << "' (type = "
                          << fnType << ", id = " << operands[1] << ") --\n");
  return processFunctionEnd(instOperands);
}

LogicalResult
spirv::Deserializer::processFunctionEnd(ArrayRef<uint32_t> operands) {
  // Process OpFunctionEnd.
  if (!operands.empty()) {
    return emitError(unknownLoc, "unexpected operands for OpFunctionEnd");
  }

  // Wire up block arguments from OpPhi instructions.
  // Put all structured control flow in spv.selection/spv.loop ops.
  if (failed(wireUpBlockArgument()) || failed(structurizeControlFlow())) {
    return failure();
  }

  curBlock = nullptr;
  curFunction = llvm::None;

  return success();
}

Optional<std::pair<Attribute, Type>>
spirv::Deserializer::getConstant(uint32_t id) {
  auto constIt = constantMap.find(id);
  if (constIt == constantMap.end())
    return llvm::None;
  return constIt->getSecond();
}

Optional<spirv::SpecConstOperationMaterializationInfo>
spirv::Deserializer::getSpecConstantOperation(uint32_t id) {
  auto constIt = specConstOperationMap.find(id);
  if (constIt == specConstOperationMap.end())
    return llvm::None;
  return constIt->getSecond();
}

std::string spirv::Deserializer::getFunctionSymbol(uint32_t id) {
  auto funcName = nameMap.lookup(id).str();
  if (funcName.empty()) {
    funcName = "spirv_fn_" + std::to_string(id);
  }
  return funcName;
}

std::string spirv::Deserializer::getSpecConstantSymbol(uint32_t id) {
  auto constName = nameMap.lookup(id).str();
  if (constName.empty()) {
    constName = "spirv_spec_const_" + std::to_string(id);
  }
  return constName;
}

spirv::SpecConstantOp
spirv::Deserializer::createSpecConstant(Location loc, uint32_t resultID,
                                        Attribute defaultValue) {
  auto symName = opBuilder.getStringAttr(getSpecConstantSymbol(resultID));
  auto op = opBuilder.create<spirv::SpecConstantOp>(unknownLoc, symName,
                                                    defaultValue);
  if (decorations.count(resultID)) {
    for (auto attr : decorations[resultID].getAttrs())
      op->setAttr(attr.first, attr.second);
  }
  specConstMap[resultID] = op;
  return op;
}

LogicalResult
spirv::Deserializer::processGlobalVariable(ArrayRef<uint32_t> operands) {
  unsigned wordIndex = 0;
  if (operands.size() < 3) {
    return emitError(
        unknownLoc,
        "OpVariable needs at least 3 operands, type, <id> and storage class");
  }

  // Result Type.
  auto type = getType(operands[wordIndex]);
  if (!type) {
    return emitError(unknownLoc, "unknown result type <id> : ")
           << operands[wordIndex];
  }
  auto ptrType = type.dyn_cast<spirv::PointerType>();
  if (!ptrType) {
    return emitError(unknownLoc,
                     "expected a result type <id> to be a spv.ptr, found : ")
           << type;
  }
  wordIndex++;

  // Result <id>.
  auto variableID = operands[wordIndex];
  auto variableName = nameMap.lookup(variableID).str();
  if (variableName.empty()) {
    variableName = "spirv_var_" + std::to_string(variableID);
  }
  wordIndex++;

  // Storage class.
  auto storageClass = static_cast<spirv::StorageClass>(operands[wordIndex]);
  if (ptrType.getStorageClass() != storageClass) {
    return emitError(unknownLoc, "mismatch in storage class of pointer type ")
           << type << " and that specified in OpVariable instruction  : "
           << stringifyStorageClass(storageClass);
  }
  wordIndex++;

  // Initializer.
  FlatSymbolRefAttr initializer = nullptr;
  if (wordIndex < operands.size()) {
    auto initializerOp = getGlobalVariable(operands[wordIndex]);
    if (!initializerOp) {
      return emitError(unknownLoc, "unknown <id> ")
             << operands[wordIndex] << "used as initializer";
    }
    wordIndex++;
    initializer = opBuilder.getSymbolRefAttr(initializerOp.getOperation());
  }
  if (wordIndex != operands.size()) {
    return emitError(unknownLoc,
                     "found more operands than expected when deserializing "
                     "OpVariable instruction, only ")
           << wordIndex << " of " << operands.size() << " processed";
  }
  auto loc = createFileLineColLoc(opBuilder);
  auto varOp = opBuilder.create<spirv::GlobalVariableOp>(
      loc, TypeAttr::get(type), opBuilder.getStringAttr(variableName),
      initializer);

  // Decorations.
  if (decorations.count(variableID)) {
    for (auto attr : decorations[variableID].getAttrs()) {
      varOp->setAttr(attr.first, attr.second);
    }
  }
  globalVariableMap[variableID] = varOp;
  return success();
}

IntegerAttr spirv::Deserializer::getConstantInt(uint32_t id) {
  auto constInfo = getConstant(id);
  if (!constInfo) {
    return nullptr;
  }
  return constInfo->first.dyn_cast<IntegerAttr>();
}

LogicalResult spirv::Deserializer::processName(ArrayRef<uint32_t> operands) {
  if (operands.size() < 2) {
    return emitError(unknownLoc, "OpName needs at least 2 operands");
  }
  if (!nameMap.lookup(operands[0]).empty()) {
    return emitError(unknownLoc, "duplicate name found for result <id> ")
           << operands[0];
  }
  unsigned wordIndex = 1;
  StringRef name = decodeStringLiteral(operands, wordIndex);
  if (wordIndex != operands.size()) {
    return emitError(unknownLoc,
                     "unexpected trailing words in OpName instruction");
  }
  nameMap[operands[0]] = name;
  return success();
}

//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

LogicalResult spirv::Deserializer::processType(spirv::Opcode opcode,
                                               ArrayRef<uint32_t> operands) {
  if (operands.empty()) {
    return emitError(unknownLoc, "type instruction with opcode ")
           << spirv::stringifyOpcode(opcode) << " needs at least one <id>";
  }

  /// TODO: Types might be forward declared in some instructions and need to be
  /// handled appropriately.
  if (typeMap.count(operands[0])) {
    return emitError(unknownLoc, "duplicate definition for result <id> ")
           << operands[0];
  }

  switch (opcode) {
  case spirv::Opcode::OpTypeVoid:
    if (operands.size() != 1)
      return emitError(unknownLoc, "OpTypeVoid must have no parameters");
    typeMap[operands[0]] = opBuilder.getNoneType();
    break;
  case spirv::Opcode::OpTypeBool:
    if (operands.size() != 1)
      return emitError(unknownLoc, "OpTypeBool must have no parameters");
    typeMap[operands[0]] = opBuilder.getI1Type();
    break;
  case spirv::Opcode::OpTypeInt: {
    if (operands.size() != 3)
      return emitError(
          unknownLoc, "OpTypeInt must have bitwidth and signedness parameters");

    // SPIR-V OpTypeInt "Signedness specifies whether there are signed semantics
    // to preserve or validate.
    // 0 indicates unsigned, or no signedness semantics
    // 1 indicates signed semantics."
    //
    // So we cannot differentiate signless and unsigned integers; always use
    // signless semantics for such cases.
    auto sign = operands[2] == 1 ? IntegerType::SignednessSemantics::Signed
                                 : IntegerType::SignednessSemantics::Signless;
    typeMap[operands[0]] = IntegerType::get(context, operands[1], sign);
  } break;
  case spirv::Opcode::OpTypeFloat: {
    if (operands.size() != 2)
      return emitError(unknownLoc, "OpTypeFloat must have bitwidth parameter");

    Type floatTy;
    switch (operands[1]) {
    case 16:
      floatTy = opBuilder.getF16Type();
      break;
    case 32:
      floatTy = opBuilder.getF32Type();
      break;
    case 64:
      floatTy = opBuilder.getF64Type();
      break;
    default:
      return emitError(unknownLoc, "unsupported OpTypeFloat bitwidth: ")
             << operands[1];
    }
    typeMap[operands[0]] = floatTy;
  } break;
  case spirv::Opcode::OpTypeVector: {
    if (operands.size() != 3) {
      return emitError(
          unknownLoc,
          "OpTypeVector must have element type and count parameters");
    }
    Type elementTy = getType(operands[1]);
    if (!elementTy) {
      return emitError(unknownLoc, "OpTypeVector references undefined <id> ")
             << operands[1];
    }
    typeMap[operands[0]] = VectorType::get({operands[2]}, elementTy);
  } break;
  case spirv::Opcode::OpTypePointer: {
    return processOpTypePointer(operands);
  } break;
  case spirv::Opcode::OpTypeArray:
    return processArrayType(operands);
  case spirv::Opcode::OpTypeCooperativeMatrixNV:
    return processCooperativeMatrixType(operands);
  case spirv::Opcode::OpTypeFunction:
    return processFunctionType(operands);
  case spirv::Opcode::OpTypeRuntimeArray:
    return processRuntimeArrayType(operands);
  case spirv::Opcode::OpTypeStruct:
    return processStructType(operands);
  case spirv::Opcode::OpTypeMatrix:
    return processMatrixType(operands);
  default:
    return emitError(unknownLoc, "unhandled type instruction");
  }
  return success();
}

LogicalResult
spirv::Deserializer::processOpTypePointer(ArrayRef<uint32_t> operands) {
  if (operands.size() != 3)
    return emitError(unknownLoc, "OpTypePointer must have two parameters");

  auto pointeeType = getType(operands[2]);
  if (!pointeeType)
    return emitError(unknownLoc, "unknown OpTypePointer pointee type <id> ")
           << operands[2];

  uint32_t typePointerID = operands[0];
  auto storageClass = static_cast<spirv::StorageClass>(operands[1]);
  typeMap[typePointerID] = spirv::PointerType::get(pointeeType, storageClass);

  for (auto *deferredStructIt = std::begin(deferredStructTypesInfos);
       deferredStructIt != std::end(deferredStructTypesInfos);) {
    for (auto *unresolvedMemberIt =
             std::begin(deferredStructIt->unresolvedMemberTypes);
         unresolvedMemberIt !=
         std::end(deferredStructIt->unresolvedMemberTypes);) {
      if (unresolvedMemberIt->first == typePointerID) {
        // The newly constructed pointer type can resolve one of the
        // deferred struct type members; update the memberTypes list and
        // clean the unresolvedMemberTypes list accordingly.
        deferredStructIt->memberTypes[unresolvedMemberIt->second] =
            typeMap[typePointerID];
        unresolvedMemberIt =
            deferredStructIt->unresolvedMemberTypes.erase(unresolvedMemberIt);
      } else {
        ++unresolvedMemberIt;
      }
    }

    if (deferredStructIt->unresolvedMemberTypes.empty()) {
      // All deferred struct type members are now resolved, set the struct body.
      auto structType = deferredStructIt->deferredStructType;

      assert(structType && "expected a spirv::StructType");
      assert(structType.isIdentified() && "expected an indentified struct");

      if (failed(structType.trySetBody(
              deferredStructIt->memberTypes, deferredStructIt->offsetInfo,
              deferredStructIt->memberDecorationsInfo)))
        return failure();

      deferredStructIt = deferredStructTypesInfos.erase(deferredStructIt);
    } else {
      ++deferredStructIt;
    }
  }

  return success();
}

LogicalResult
spirv::Deserializer::processArrayType(ArrayRef<uint32_t> operands) {
  if (operands.size() != 3) {
    return emitError(unknownLoc,
                     "OpTypeArray must have element type and count parameters");
  }

  Type elementTy = getType(operands[1]);
  if (!elementTy) {
    return emitError(unknownLoc, "OpTypeArray references undefined <id> ")
           << operands[1];
  }

  unsigned count = 0;
  // TODO: The count can also come frome a specialization constant.
  auto countInfo = getConstant(operands[2]);
  if (!countInfo) {
    return emitError(unknownLoc, "OpTypeArray count <id> ")
           << operands[2] << "can only come from normal constant right now";
  }

  if (auto intVal = countInfo->first.dyn_cast<IntegerAttr>()) {
    count = intVal.getValue().getZExtValue();
  } else {
    return emitError(unknownLoc, "OpTypeArray count must come from a "
                                 "scalar integer constant instruction");
  }

  typeMap[operands[0]] = spirv::ArrayType::get(
      elementTy, count, typeDecorations.lookup(operands[0]));
  return success();
}

LogicalResult
spirv::Deserializer::processFunctionType(ArrayRef<uint32_t> operands) {
  assert(!operands.empty() && "No operands for processing function type");
  if (operands.size() == 1) {
    return emitError(unknownLoc, "missing return type for OpTypeFunction");
  }
  auto returnType = getType(operands[1]);
  if (!returnType) {
    return emitError(unknownLoc, "unknown return type in OpTypeFunction");
  }
  SmallVector<Type, 1> argTypes;
  for (size_t i = 2, e = operands.size(); i < e; ++i) {
    auto ty = getType(operands[i]);
    if (!ty) {
      return emitError(unknownLoc, "unknown argument type in OpTypeFunction");
    }
    argTypes.push_back(ty);
  }
  ArrayRef<Type> returnTypes;
  if (!isVoidType(returnType)) {
    returnTypes = llvm::makeArrayRef(returnType);
  }
  typeMap[operands[0]] = FunctionType::get(context, argTypes, returnTypes);
  return success();
}

LogicalResult
spirv::Deserializer::processCooperativeMatrixType(ArrayRef<uint32_t> operands) {
  if (operands.size() != 5) {
    return emitError(unknownLoc, "OpTypeCooperativeMatrix must have element "
                                 "type and row x column parameters");
  }

  Type elementTy = getType(operands[1]);
  if (!elementTy) {
    return emitError(unknownLoc,
                     "OpTypeCooperativeMatrix references undefined <id> ")
           << operands[1];
  }

  auto scope = spirv::symbolizeScope(getConstantInt(operands[2]).getInt());
  if (!scope) {
    return emitError(unknownLoc,
                     "OpTypeCooperativeMatrix references undefined scope <id> ")
           << operands[2];
  }

  unsigned rows = getConstantInt(operands[3]).getInt();
  unsigned columns = getConstantInt(operands[4]).getInt();

  typeMap[operands[0]] = spirv::CooperativeMatrixNVType::get(
      elementTy, scope.getValue(), rows, columns);
  return success();
}

LogicalResult
spirv::Deserializer::processRuntimeArrayType(ArrayRef<uint32_t> operands) {
  if (operands.size() != 2) {
    return emitError(unknownLoc, "OpTypeRuntimeArray must have two operands");
  }
  Type memberType = getType(operands[1]);
  if (!memberType) {
    return emitError(unknownLoc,
                     "OpTypeRuntimeArray references undefined <id> ")
           << operands[1];
  }
  typeMap[operands[0]] = spirv::RuntimeArrayType::get(
      memberType, typeDecorations.lookup(operands[0]));
  return success();
}

LogicalResult
spirv::Deserializer::processStructType(ArrayRef<uint32_t> operands) {
  // TODO: Find a way to handle identified structs when debug info is stripped.

  if (operands.empty()) {
    return emitError(unknownLoc, "OpTypeStruct must have at least result <id>");
  }

  if (operands.size() == 1) {
    // Handle empty struct.
    typeMap[operands[0]] =
        spirv::StructType::getEmpty(context, nameMap.lookup(operands[0]).str());
    return success();
  }

  // First element is operand ID, second element is member index in the struct.
  SmallVector<std::pair<uint32_t, unsigned>, 0> unresolvedMemberTypes;
  SmallVector<Type, 4> memberTypes;

  for (auto op : llvm::drop_begin(operands, 1)) {
    Type memberType = getType(op);
    bool typeForwardPtr = (typeForwardPointerIDs.count(op) != 0);

    if (!memberType && !typeForwardPtr)
      return emitError(unknownLoc, "OpTypeStruct references undefined <id> ")
             << op;

    if (!memberType)
      unresolvedMemberTypes.emplace_back(op, memberTypes.size());

    memberTypes.push_back(memberType);
  }

  SmallVector<spirv::StructType::OffsetInfo, 0> offsetInfo;
  SmallVector<spirv::StructType::MemberDecorationInfo, 0> memberDecorationsInfo;
  if (memberDecorationMap.count(operands[0])) {
    auto &allMemberDecorations = memberDecorationMap[operands[0]];
    for (auto memberIndex : llvm::seq<uint32_t>(0, memberTypes.size())) {
      if (allMemberDecorations.count(memberIndex)) {
        for (auto &memberDecoration : allMemberDecorations[memberIndex]) {
          // Check for offset.
          if (memberDecoration.first == spirv::Decoration::Offset) {
            // If offset info is empty, resize to the number of members;
            if (offsetInfo.empty()) {
              offsetInfo.resize(memberTypes.size());
            }
            offsetInfo[memberIndex] = memberDecoration.second[0];
          } else {
            if (!memberDecoration.second.empty()) {
              memberDecorationsInfo.emplace_back(memberIndex, /*hasValue=*/1,
                                                 memberDecoration.first,
                                                 memberDecoration.second[0]);
            } else {
              memberDecorationsInfo.emplace_back(memberIndex, /*hasValue=*/0,
                                                 memberDecoration.first, 0);
            }
          }
        }
      }
    }
  }

  uint32_t structID = operands[0];
  std::string structIdentifier = nameMap.lookup(structID).str();

  if (structIdentifier.empty()) {
    assert(unresolvedMemberTypes.empty() &&
           "didn't expect unresolved member types");
    typeMap[structID] =
        spirv::StructType::get(memberTypes, offsetInfo, memberDecorationsInfo);
  } else {
    auto structTy = spirv::StructType::getIdentified(context, structIdentifier);
    typeMap[structID] = structTy;

    if (!unresolvedMemberTypes.empty())
      deferredStructTypesInfos.push_back({structTy, unresolvedMemberTypes,
                                          memberTypes, offsetInfo,
                                          memberDecorationsInfo});
    else if (failed(structTy.trySetBody(memberTypes, offsetInfo,
                                        memberDecorationsInfo)))
      return failure();
  }

  // TODO: Update StructType to have member name as attribute as
  // well.
  return success();
}

LogicalResult
spirv::Deserializer::processMatrixType(ArrayRef<uint32_t> operands) {
  if (operands.size() != 3) {
    // Three operands are needed: result_id, column_type, and column_count
    return emitError(unknownLoc, "OpTypeMatrix must have 3 operands"
                                 " (result_id, column_type, and column_count)");
  }
  // Matrix columns must be of vector type
  Type elementTy = getType(operands[1]);
  if (!elementTy) {
    return emitError(unknownLoc,
                     "OpTypeMatrix references undefined column type.")
           << operands[1];
  }

  uint32_t colsCount = operands[2];
  typeMap[operands[0]] = spirv::MatrixType::get(elementTy, colsCount);
  return success();
}

LogicalResult
spirv::Deserializer::processTypeForwardPointer(ArrayRef<uint32_t> operands) {
  if (operands.size() != 2)
    return emitError(unknownLoc,
                     "OpTypeForwardPointer instruction must have two operands");

  typeForwardPointerIDs.insert(operands[0]);
  // TODO: Use the 2nd operand (Storage Class) to validate the OpTypePointer
  // instruction that defines the actual type.

  return success();
}

//===----------------------------------------------------------------------===//
// Constant
//===----------------------------------------------------------------------===//

LogicalResult spirv::Deserializer::processConstant(ArrayRef<uint32_t> operands,
                                                   bool isSpec) {
  StringRef opname = isSpec ? "OpSpecConstant" : "OpConstant";

  if (operands.size() < 2) {
    return emitError(unknownLoc)
           << opname << " must have type <id> and result <id>";
  }
  if (operands.size() < 3) {
    return emitError(unknownLoc)
           << opname << " must have at least 1 more parameter";
  }

  Type resultType = getType(operands[0]);
  if (!resultType) {
    return emitError(unknownLoc, "undefined result type from <id> ")
           << operands[0];
  }

  auto checkOperandSizeForBitwidth = [&](unsigned bitwidth) -> LogicalResult {
    if (bitwidth == 64) {
      if (operands.size() == 4) {
        return success();
      }
      return emitError(unknownLoc)
             << opname << " should have 2 parameters for 64-bit values";
    }
    if (bitwidth <= 32) {
      if (operands.size() == 3) {
        return success();
      }

      return emitError(unknownLoc)
             << opname
             << " should have 1 parameter for values with no more than 32 bits";
    }
    return emitError(unknownLoc, "unsupported OpConstant bitwidth: ")
           << bitwidth;
  };

  auto resultID = operands[1];

  if (auto intType = resultType.dyn_cast<IntegerType>()) {
    auto bitwidth = intType.getWidth();
    if (failed(checkOperandSizeForBitwidth(bitwidth))) {
      return failure();
    }

    APInt value;
    if (bitwidth == 64) {
      // 64-bit integers are represented with two SPIR-V words. According to
      // SPIR-V spec: "When the type’s bit width is larger than one word, the
      // literal’s low-order words appear first."
      struct DoubleWord {
        uint32_t word1;
        uint32_t word2;
      } words = {operands[2], operands[3]};
      value = APInt(64, llvm::bit_cast<uint64_t>(words), /*isSigned=*/true);
    } else if (bitwidth <= 32) {
      value = APInt(bitwidth, operands[2], /*isSigned=*/true);
    }

    auto attr = opBuilder.getIntegerAttr(intType, value);

    if (isSpec) {
      createSpecConstant(unknownLoc, resultID, attr);
    } else {
      // For normal constants, we just record the attribute (and its type) for
      // later materialization at use sites.
      constantMap.try_emplace(resultID, attr, intType);
    }

    return success();
  }

  if (auto floatType = resultType.dyn_cast<FloatType>()) {
    auto bitwidth = floatType.getWidth();
    if (failed(checkOperandSizeForBitwidth(bitwidth))) {
      return failure();
    }

    APFloat value(0.f);
    if (floatType.isF64()) {
      // Double values are represented with two SPIR-V words. According to
      // SPIR-V spec: "When the type’s bit width is larger than one word, the
      // literal’s low-order words appear first."
      struct DoubleWord {
        uint32_t word1;
        uint32_t word2;
      } words = {operands[2], operands[3]};
      value = APFloat(llvm::bit_cast<double>(words));
    } else if (floatType.isF32()) {
      value = APFloat(llvm::bit_cast<float>(operands[2]));
    } else if (floatType.isF16()) {
      APInt data(16, operands[2]);
      value = APFloat(APFloat::IEEEhalf(), data);
    }

    auto attr = opBuilder.getFloatAttr(floatType, value);
    if (isSpec) {
      createSpecConstant(unknownLoc, resultID, attr);
    } else {
      // For normal constants, we just record the attribute (and its type) for
      // later materialization at use sites.
      constantMap.try_emplace(resultID, attr, floatType);
    }

    return success();
  }

  return emitError(unknownLoc, "OpConstant can only generate values of "
                               "scalar integer or floating-point type");
}

LogicalResult spirv::Deserializer::processConstantBool(
    bool isTrue, ArrayRef<uint32_t> operands, bool isSpec) {
  if (operands.size() != 2) {
    return emitError(unknownLoc, "Op")
           << (isSpec ? "Spec" : "") << "Constant"
           << (isTrue ? "True" : "False")
           << " must have type <id> and result <id>";
  }

  auto attr = opBuilder.getBoolAttr(isTrue);
  auto resultID = operands[1];
  if (isSpec) {
    createSpecConstant(unknownLoc, resultID, attr);
  } else {
    // For normal constants, we just record the attribute (and its type) for
    // later materialization at use sites.
    constantMap.try_emplace(resultID, attr, opBuilder.getI1Type());
  }

  return success();
}

LogicalResult
spirv::Deserializer::processConstantComposite(ArrayRef<uint32_t> operands) {
  if (operands.size() < 2) {
    return emitError(unknownLoc,
                     "OpConstantComposite must have type <id> and result <id>");
  }
  if (operands.size() < 3) {
    return emitError(unknownLoc,
                     "OpConstantComposite must have at least 1 parameter");
  }

  Type resultType = getType(operands[0]);
  if (!resultType) {
    return emitError(unknownLoc, "undefined result type from <id> ")
           << operands[0];
  }

  SmallVector<Attribute, 4> elements;
  elements.reserve(operands.size() - 2);
  for (unsigned i = 2, e = operands.size(); i < e; ++i) {
    auto elementInfo = getConstant(operands[i]);
    if (!elementInfo) {
      return emitError(unknownLoc, "OpConstantComposite component <id> ")
             << operands[i] << " must come from a normal constant";
    }
    elements.push_back(elementInfo->first);
  }

  auto resultID = operands[1];
  if (auto vectorType = resultType.dyn_cast<VectorType>()) {
    auto attr = DenseElementsAttr::get(vectorType, elements);
    // For normal constants, we just record the attribute (and its type) for
    // later materialization at use sites.
    constantMap.try_emplace(resultID, attr, resultType);
  } else if (auto arrayType = resultType.dyn_cast<spirv::ArrayType>()) {
    auto attr = opBuilder.getArrayAttr(elements);
    constantMap.try_emplace(resultID, attr, resultType);
  } else {
    return emitError(unknownLoc, "unsupported OpConstantComposite type: ")
           << resultType;
  }

  return success();
}

LogicalResult
spirv::Deserializer::processSpecConstantComposite(ArrayRef<uint32_t> operands) {
  if (operands.size() < 2) {
    return emitError(unknownLoc,
                     "OpConstantComposite must have type <id> and result <id>");
  }
  if (operands.size() < 3) {
    return emitError(unknownLoc,
                     "OpConstantComposite must have at least 1 parameter");
  }

  Type resultType = getType(operands[0]);
  if (!resultType) {
    return emitError(unknownLoc, "undefined result type from <id> ")
           << operands[0];
  }

  auto resultID = operands[1];
  auto symName = opBuilder.getStringAttr(getSpecConstantSymbol(resultID));

  SmallVector<Attribute, 4> elements;
  elements.reserve(operands.size() - 2);
  for (unsigned i = 2, e = operands.size(); i < e; ++i) {
    auto elementInfo = getSpecConstant(operands[i]);
    elements.push_back(opBuilder.getSymbolRefAttr(elementInfo));
  }

  auto op = opBuilder.create<spirv::SpecConstantCompositeOp>(
      unknownLoc, TypeAttr::get(resultType), symName,
      opBuilder.getArrayAttr(elements));
  specConstCompositeMap[resultID] = op;

  return success();
}

LogicalResult
spirv::Deserializer::processSpecConstantOperation(ArrayRef<uint32_t> operands) {
  if (operands.size() < 3)
    return emitError(unknownLoc, "OpConstantOperation must have type <id>, "
                                 "result <id>, and operand opcode");

  uint32_t resultTypeID = operands[0];

  if (!getType(resultTypeID))
    return emitError(unknownLoc, "undefined result type from <id> ")
           << resultTypeID;

  uint32_t resultID = operands[1];
  spirv::Opcode enclosedOpcode = static_cast<spirv::Opcode>(operands[2]);
  auto emplaceResult = specConstOperationMap.try_emplace(
      resultID,
      SpecConstOperationMaterializationInfo{
          enclosedOpcode, resultTypeID,
          SmallVector<uint32_t>{operands.begin() + 3, operands.end()}});

  if (!emplaceResult.second)
    return emitError(unknownLoc, "value with <id>: ")
           << resultID << " is probably defined before.";

  return success();
}

Value spirv::Deserializer::materializeSpecConstantOperation(
    uint32_t resultID, spirv::Opcode enclosedOpcode, uint32_t resultTypeID,
    ArrayRef<uint32_t> enclosedOpOperands) {

  Type resultType = getType(resultTypeID);

  // Instructions wrapped by OpSpecConstantOp need an ID for their
  // Deserializer::processOp<op_name>(...) to emit the corresponding SPIR-V
  // dialect wrapped op. For that purpose, a new value map is created and "fake"
  // ID in that map is assigned to the result of the enclosed instruction. Note
  // that there is no need to update this fake ID since we only need to
  // reference the created Value for the enclosed op from the spv::YieldOp
  // created later in this method (both of which are the only values in their
  // region: the SpecConstantOperation's region). If we encounter another
  // SpecConstantOperation in the module, we simply re-use the fake ID since the
  // previous Value assigned to it isn't visible in the current scope anyway.
  DenseMap<uint32_t, Value> newValueMap;
  llvm::SaveAndRestore<DenseMap<uint32_t, Value>> valueMapGuard(valueMap,
                                                                newValueMap);
  constexpr uint32_t fakeID = static_cast<uint32_t>(-3);

  SmallVector<uint32_t, 4> enclosedOpResultTypeAndOperands;
  enclosedOpResultTypeAndOperands.push_back(resultTypeID);
  enclosedOpResultTypeAndOperands.push_back(fakeID);
  enclosedOpResultTypeAndOperands.append(enclosedOpOperands.begin(),
                                         enclosedOpOperands.end());

  // Process enclosed instruction before creating the enclosing
  // specConstantOperation (and its region). This way, references to constants,
  // global variables, and spec constants will be materialized outside the new
  // op's region. For more info, see Deserializer::getValue's implementation.
  if (failed(
          processInstruction(enclosedOpcode, enclosedOpResultTypeAndOperands)))
    return Value();

  // Since the enclosed op is emitted in the current block, split it in a
  // separate new block.
  Block *enclosedBlock = curBlock->splitBlock(&curBlock->back());

  auto loc = createFileLineColLoc(opBuilder);
  auto specConstOperationOp =
      opBuilder.create<spirv::SpecConstantOperationOp>(loc, resultType);

  Region &body = specConstOperationOp.body();
  // Move the new block into SpecConstantOperation's body.
  body.getBlocks().splice(body.end(), curBlock->getParent()->getBlocks(),
                          Region::iterator(enclosedBlock));
  Block &block = body.back();

  // RAII guard to reset the insertion point to the module's region after
  // deserializing the body of the specConstantOperation.
  OpBuilder::InsertionGuard moduleInsertionGuard(opBuilder);
  opBuilder.setInsertionPointToEnd(&block);

  opBuilder.create<spirv::YieldOp>(loc, block.front().getResult(0));
  return specConstOperationOp.getResult();
}

LogicalResult
spirv::Deserializer::processConstantNull(ArrayRef<uint32_t> operands) {
  if (operands.size() != 2) {
    return emitError(unknownLoc,
                     "OpConstantNull must have type <id> and result <id>");
  }

  Type resultType = getType(operands[0]);
  if (!resultType) {
    return emitError(unknownLoc, "undefined result type from <id> ")
           << operands[0];
  }

  auto resultID = operands[1];
  if (resultType.isIntOrFloat() || resultType.isa<VectorType>()) {
    auto attr = opBuilder.getZeroAttr(resultType);
    // For normal constants, we just record the attribute (and its type) for
    // later materialization at use sites.
    constantMap.try_emplace(resultID, attr, resultType);
    return success();
  }

  return emitError(unknownLoc, "unsupported OpConstantNull type: ")
         << resultType;
}

//===----------------------------------------------------------------------===//
// Control flow
//===----------------------------------------------------------------------===//

Block *spirv::Deserializer::getOrCreateBlock(uint32_t id) {
  if (auto *block = getBlock(id)) {
    LLVM_DEBUG(llvm::dbgs() << "[block] got exiting block for id = " << id
                            << " @ " << block << "\n");
    return block;
  }

  // We don't know where this block will be placed finally (in a spv.selection
  // or spv.loop or function). Create it into the function for now and sort
  // out the proper place later.
  auto *block = curFunction->addBlock();
  LLVM_DEBUG(llvm::dbgs() << "[block] created block for id = " << id << " @ "
                          << block << "\n");
  return blockMap[id] = block;
}

LogicalResult spirv::Deserializer::processBranch(ArrayRef<uint32_t> operands) {
  if (!curBlock) {
    return emitError(unknownLoc, "OpBranch must appear inside a block");
  }

  if (operands.size() != 1) {
    return emitError(unknownLoc, "OpBranch must take exactly one target label");
  }

  auto *target = getOrCreateBlock(operands[0]);
  auto loc = createFileLineColLoc(opBuilder);
  // The preceding instruction for the OpBranch instruction could be an
  // OpLoopMerge or an OpSelectionMerge instruction, in this case they will have
  // the same OpLine information.
  opBuilder.create<spirv::BranchOp>(loc, target);

  clearDebugLine();
  return success();
}

LogicalResult
spirv::Deserializer::processBranchConditional(ArrayRef<uint32_t> operands) {
  if (!curBlock) {
    return emitError(unknownLoc,
                     "OpBranchConditional must appear inside a block");
  }

  if (operands.size() != 3 && operands.size() != 5) {
    return emitError(unknownLoc,
                     "OpBranchConditional must have condition, true label, "
                     "false label, and optionally two branch weights");
  }

  auto condition = getValue(operands[0]);
  auto *trueBlock = getOrCreateBlock(operands[1]);
  auto *falseBlock = getOrCreateBlock(operands[2]);

  Optional<std::pair<uint32_t, uint32_t>> weights;
  if (operands.size() == 5) {
    weights = std::make_pair(operands[3], operands[4]);
  }
  // The preceding instruction for the OpBranchConditional instruction could be
  // an OpSelectionMerge instruction, in this case they will have the same
  // OpLine information.
  auto loc = createFileLineColLoc(opBuilder);
  opBuilder.create<spirv::BranchConditionalOp>(
      loc, condition, trueBlock,
      /*trueArguments=*/ArrayRef<Value>(), falseBlock,
      /*falseArguments=*/ArrayRef<Value>(), weights);

  clearDebugLine();
  return success();
}

LogicalResult spirv::Deserializer::processLabel(ArrayRef<uint32_t> operands) {
  if (!curFunction) {
    return emitError(unknownLoc, "OpLabel must appear inside a function");
  }

  if (operands.size() != 1) {
    return emitError(unknownLoc, "OpLabel should only have result <id>");
  }

  auto labelID = operands[0];
  // We may have forward declared this block.
  auto *block = getOrCreateBlock(labelID);
  LLVM_DEBUG(llvm::dbgs() << "[block] populating block " << block << "\n");
  // If we have seen this block, make sure it was just a forward declaration.
  assert(block->empty() && "re-deserialize the same block!");

  opBuilder.setInsertionPointToStart(block);
  blockMap[labelID] = curBlock = block;

  return success();
}

LogicalResult
spirv::Deserializer::processSelectionMerge(ArrayRef<uint32_t> operands) {
  if (!curBlock) {
    return emitError(unknownLoc, "OpSelectionMerge must appear in a block");
  }

  if (operands.size() < 2) {
    return emitError(
        unknownLoc,
        "OpSelectionMerge must specify merge target and selection control");
  }

  auto *mergeBlock = getOrCreateBlock(operands[0]);
  auto loc = createFileLineColLoc(opBuilder);
  auto selectionControl = operands[1];

  if (!blockMergeInfo.try_emplace(curBlock, loc, selectionControl, mergeBlock)
           .second) {
    return emitError(
        unknownLoc,
        "a block cannot have more than one OpSelectionMerge instruction");
  }

  return success();
}

LogicalResult
spirv::Deserializer::processLoopMerge(ArrayRef<uint32_t> operands) {
  if (!curBlock) {
    return emitError(unknownLoc, "OpLoopMerge must appear in a block");
  }

  if (operands.size() < 3) {
    return emitError(unknownLoc, "OpLoopMerge must specify merge target, "
                                 "continue target and loop control");
  }

  auto *mergeBlock = getOrCreateBlock(operands[0]);
  auto *continueBlock = getOrCreateBlock(operands[1]);
  auto loc = createFileLineColLoc(opBuilder);
  uint32_t loopControl = operands[2];

  if (!blockMergeInfo
           .try_emplace(curBlock, loc, loopControl, mergeBlock, continueBlock)
           .second) {
    return emitError(
        unknownLoc,
        "a block cannot have more than one OpLoopMerge instruction");
  }

  return success();
}

LogicalResult spirv::Deserializer::processPhi(ArrayRef<uint32_t> operands) {
  if (!curBlock) {
    return emitError(unknownLoc, "OpPhi must appear in a block");
  }

  if (operands.size() < 4) {
    return emitError(unknownLoc, "OpPhi must specify result type, result <id>, "
                                 "and variable-parent pairs");
  }

  // Create a block argument for this OpPhi instruction.
  Type blockArgType = getType(operands[0]);
  BlockArgument blockArg = curBlock->addArgument(blockArgType);
  valueMap[operands[1]] = blockArg;
  LLVM_DEBUG(llvm::dbgs() << "[phi] created block argument " << blockArg
                          << " id = " << operands[1] << " of type "
                          << blockArgType << '\n');

  // For each (value, predecessor) pair, insert the value to the predecessor's
  // blockPhiInfo entry so later we can fix the block argument there.
  for (unsigned i = 2, e = operands.size(); i < e; i += 2) {
    uint32_t value = operands[i];
    Block *predecessor = getOrCreateBlock(operands[i + 1]);
    blockPhiInfo[predecessor].push_back(value);
    LLVM_DEBUG(llvm::dbgs() << "[phi] predecessor @ " << predecessor
                            << " with arg id = " << value << '\n');
  }

  return success();
}

namespace {
/// A class for putting all blocks in a structured selection/loop in a
/// spv.selection/spv.loop op.
class ControlFlowStructurizer {
public:
  /// Structurizes the loop at the given `headerBlock`.
  ///
  /// This method will create an spv.loop op in the `mergeBlock` and move all
  /// blocks in the structured loop into the spv.loop's region. All branches to
  /// the `headerBlock` will be redirected to the `mergeBlock`.
  /// This method will also update `mergeInfo` by remapping all blocks inside to
  /// the newly cloned ones inside structured control flow op's regions.
  static LogicalResult structurize(Location loc, uint32_t control,
                                   spirv::BlockMergeInfoMap &mergeInfo,
                                   Block *headerBlock, Block *mergeBlock,
                                   Block *continueBlock) {
    return ControlFlowStructurizer(loc, control, mergeInfo, headerBlock,
                                   mergeBlock, continueBlock)
        .structurizeImpl();
  }

private:
  ControlFlowStructurizer(Location loc, uint32_t control,
                          spirv::BlockMergeInfoMap &mergeInfo, Block *header,
                          Block *merge, Block *cont)
      : location(loc), control(control), blockMergeInfo(mergeInfo),
        headerBlock(header), mergeBlock(merge), continueBlock(cont) {}

  /// Creates a new spv.selection op at the beginning of the `mergeBlock`.
  spirv::SelectionOp createSelectionOp(uint32_t selectionControl);

  /// Creates a new spv.loop op at the beginning of the `mergeBlock`.
  spirv::LoopOp createLoopOp(uint32_t loopControl);

  /// Collects all blocks reachable from `headerBlock` except `mergeBlock`.
  void collectBlocksInConstruct();

  LogicalResult structurizeImpl();

  Location location;
  uint32_t control;

  spirv::BlockMergeInfoMap &blockMergeInfo;

  Block *headerBlock;
  Block *mergeBlock;
  Block *continueBlock; // nullptr for spv.selection

  llvm::SetVector<Block *> constructBlocks;
};
} // namespace

spirv::SelectionOp
ControlFlowStructurizer::createSelectionOp(uint32_t selectionControl) {
  // Create a builder and set the insertion point to the beginning of the
  // merge block so that the newly created SelectionOp will be inserted there.
  OpBuilder builder(&mergeBlock->front());

  auto control = builder.getI32IntegerAttr(selectionControl);
  auto selectionOp = builder.create<spirv::SelectionOp>(location, control);
  selectionOp.addMergeBlock();

  return selectionOp;
}

spirv::LoopOp ControlFlowStructurizer::createLoopOp(uint32_t loopControl) {
  // Create a builder and set the insertion point to the beginning of the
  // merge block so that the newly created LoopOp will be inserted there.
  OpBuilder builder(&mergeBlock->front());

  auto control = builder.getI32IntegerAttr(loopControl);
  auto loopOp = builder.create<spirv::LoopOp>(location, control);
  loopOp.addEntryAndMergeBlock();

  return loopOp;
}

void ControlFlowStructurizer::collectBlocksInConstruct() {
  assert(constructBlocks.empty() && "expected empty constructBlocks");

  // Put the header block in the work list first.
  constructBlocks.insert(headerBlock);

  // For each item in the work list, add its successors excluding the merge
  // block.
  for (unsigned i = 0; i < constructBlocks.size(); ++i) {
    for (auto *successor : constructBlocks[i]->getSuccessors())
      if (successor != mergeBlock)
        constructBlocks.insert(successor);
  }
}

LogicalResult ControlFlowStructurizer::structurizeImpl() {
  Operation *op = nullptr;
  bool isLoop = continueBlock != nullptr;
  if (isLoop) {
    if (auto loopOp = createLoopOp(control))
      op = loopOp.getOperation();
  } else {
    if (auto selectionOp = createSelectionOp(control))
      op = selectionOp.getOperation();
  }
  if (!op)
    return failure();
  Region &body = op->getRegion(0);

  BlockAndValueMapping mapper;
  // All references to the old merge block should be directed to the
  // selection/loop merge block in the SelectionOp/LoopOp's region.
  mapper.map(mergeBlock, &body.back());

  collectBlocksInConstruct();

  // We've identified all blocks belonging to the selection/loop's region. Now
  // need to "move" them into the selection/loop. Instead of really moving the
  // blocks, in the following we copy them and remap all values and branches.
  // This is because:
  // * Inserting a block into a region requires the block not in any region
  //   before. But selections/loops can nest so we can create selection/loop ops
  //   in a nested manner, which means some blocks may already be in a
  //   selection/loop region when to be moved again.
  // * It's much trickier to fix up the branches into and out of the loop's
  //   region: we need to treat not-moved blocks and moved blocks differently:
  //   Not-moved blocks jumping to the loop header block need to jump to the
  //   merge point containing the new loop op but not the loop continue block's
  //   back edge. Moved blocks jumping out of the loop need to jump to the
  //   merge block inside the loop region but not other not-moved blocks.
  //   We cannot use replaceAllUsesWith clearly and it's harder to follow the
  //   logic.

  // Create a corresponding block in the SelectionOp/LoopOp's region for each
  // block in this loop construct.
  OpBuilder builder(body);
  for (auto *block : constructBlocks) {
    // Create a block and insert it before the selection/loop merge block in the
    // SelectionOp/LoopOp's region.
    auto *newBlock = builder.createBlock(&body.back());
    mapper.map(block, newBlock);
    LLVM_DEBUG(llvm::dbgs() << "[cf] cloned block " << newBlock
                            << " from block " << block << "\n");
    if (!isFnEntryBlock(block)) {
      for (BlockArgument blockArg : block->getArguments()) {
        auto newArg = newBlock->addArgument(blockArg.getType());
        mapper.map(blockArg, newArg);
        LLVM_DEBUG(llvm::dbgs() << "[cf] remapped block argument " << blockArg
                                << " to " << newArg << '\n');
      }
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "[cf] block " << block << " is a function entry block\n");
    }
    for (auto &op : *block)
      newBlock->push_back(op.clone(mapper));
  }

  // Go through all ops and remap the operands.
  auto remapOperands = [&](Operation *op) {
    for (auto &operand : op->getOpOperands())
      if (Value mappedOp = mapper.lookupOrNull(operand.get()))
        operand.set(mappedOp);
    for (auto &succOp : op->getBlockOperands())
      if (Block *mappedOp = mapper.lookupOrNull(succOp.get()))
        succOp.set(mappedOp);
  };
  for (auto &block : body) {
    block.walk(remapOperands);
  }

  // We have created the SelectionOp/LoopOp and "moved" all blocks belonging to
  // the selection/loop construct into its region. Next we need to fix the
  // connections between this new SelectionOp/LoopOp with existing blocks.

  // All existing incoming branches should go to the merge block, where the
  // SelectionOp/LoopOp resides right now.
  headerBlock->replaceAllUsesWith(mergeBlock);

  if (isLoop) {
    // The loop selection/loop header block may have block arguments. Since now
    // we place the selection/loop op inside the old merge block, we need to
    // make sure the old merge block has the same block argument list.
    assert(mergeBlock->args_empty() && "OpPhi in loop merge block unsupported");
    for (BlockArgument blockArg : headerBlock->getArguments()) {
      mergeBlock->addArgument(blockArg.getType());
    }

    // If the loop header block has block arguments, make sure the spv.branch op
    // matches.
    SmallVector<Value, 4> blockArgs;
    if (!headerBlock->args_empty())
      blockArgs = {mergeBlock->args_begin(), mergeBlock->args_end()};

    // The loop entry block should have a unconditional branch jumping to the
    // loop header block.
    builder.setInsertionPointToEnd(&body.front());
    builder.create<spirv::BranchOp>(location, mapper.lookupOrNull(headerBlock),
                                    ArrayRef<Value>(blockArgs));
  }

  // All the blocks cloned into the SelectionOp/LoopOp's region can now be
  // cleaned up.
  LLVM_DEBUG(llvm::dbgs() << "[cf] cleaning up blocks after clone\n");
  // First we need to drop all operands' references inside all blocks. This is
  // needed because we can have blocks referencing SSA values from one another.
  for (auto *block : constructBlocks)
    block->dropAllReferences();

  // Then erase all old blocks.
  for (auto *block : constructBlocks) {
    // We've cloned all blocks belonging to this construct into the structured
    // control flow op's region. Among these blocks, some may compose another
    // selection/loop. If so, they will be recorded within blockMergeInfo.
    // We need to update the pointers there to the newly remapped ones so we can
    // continue structurizing them later.
    // TODO: The asserts in the following assumes input SPIR-V blob
    // forms correctly nested selection/loop constructs. We should relax this
    // and support error cases better.
    auto it = blockMergeInfo.find(block);
    if (it != blockMergeInfo.end()) {
      Block *newHeader = mapper.lookupOrNull(block);
      assert(newHeader && "nested loop header block should be remapped!");

      Block *newContinue = it->second.continueBlock;
      if (newContinue) {
        newContinue = mapper.lookupOrNull(newContinue);
        assert(newContinue && "nested loop continue block should be remapped!");
      }

      Block *newMerge = it->second.mergeBlock;
      if (Block *mappedTo = mapper.lookupOrNull(newMerge))
        newMerge = mappedTo;

      // Keep original location for nested selection/loop ops.
      Location loc = it->second.loc;
      // The iterator should be erased before adding a new entry into
      // blockMergeInfo to avoid iterator invalidation.
      blockMergeInfo.erase(it);
      blockMergeInfo.try_emplace(newHeader, loc, it->second.control, newMerge,
                                 newContinue);
    }

    // The structured selection/loop's entry block does not have arguments.
    // If the function's header block is also part of the structured control
    // flow, we cannot just simply erase it because it may contain arguments
    // matching the function signature and used by the cloned blocks.
    if (isFnEntryBlock(block)) {
      LLVM_DEBUG(llvm::dbgs() << "[cf] changing entry block " << block
                              << " to only contain a spv.Branch op\n");
      // Still keep the function entry block for the potential block arguments,
      // but replace all ops inside with a branch to the merge block.
      block->clear();
      builder.setInsertionPointToEnd(block);
      builder.create<spirv::BranchOp>(location, mergeBlock);
    } else {
      LLVM_DEBUG(llvm::dbgs() << "[cf] erasing block " << block << "\n");
      block->erase();
    }
  }

  LLVM_DEBUG(
      llvm::dbgs() << "[cf] after structurizing construct with header block "
                   << headerBlock << ":\n"
                   << *op << '\n');

  return success();
}

LogicalResult spirv::Deserializer::wireUpBlockArgument() {
  LLVM_DEBUG(llvm::dbgs() << "[phi] start wiring up block arguments\n");

  OpBuilder::InsertionGuard guard(opBuilder);

  for (const auto &info : blockPhiInfo) {
    Block *block = info.first;
    const BlockPhiInfo &phiInfo = info.second;
    LLVM_DEBUG(llvm::dbgs() << "[phi] block " << block << "\n");
    LLVM_DEBUG(llvm::dbgs() << "[phi] before creating block argument:\n");
    LLVM_DEBUG(block->getParentOp()->print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << '\n');

    // Set insertion point to before this block's terminator early because we
    // may materialize ops via getValue() call.
    auto *op = block->getTerminator();
    opBuilder.setInsertionPoint(op);

    SmallVector<Value, 4> blockArgs;
    blockArgs.reserve(phiInfo.size());
    for (uint32_t valueId : phiInfo) {
      if (Value value = getValue(valueId)) {
        blockArgs.push_back(value);
        LLVM_DEBUG(llvm::dbgs() << "[phi] block argument " << value
                                << " id = " << valueId << '\n');
      } else {
        return emitError(unknownLoc, "OpPhi references undefined value!");
      }
    }

    if (auto branchOp = dyn_cast<spirv::BranchOp>(op)) {
      // Replace the previous branch op with a new one with block arguments.
      opBuilder.create<spirv::BranchOp>(branchOp.getLoc(), branchOp.getTarget(),
                                        blockArgs);
      branchOp.erase();
    } else {
      return emitError(unknownLoc, "unimplemented terminator for Phi creation");
    }

    LLVM_DEBUG(llvm::dbgs() << "[phi] after creating block argument:\n");
    LLVM_DEBUG(block->getParentOp()->print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << '\n');
  }
  blockPhiInfo.clear();

  LLVM_DEBUG(llvm::dbgs() << "[phi] completed wiring up block arguments\n");
  return success();
}

LogicalResult spirv::Deserializer::structurizeControlFlow() {
  LLVM_DEBUG(llvm::dbgs() << "[cf] start structurizing control flow\n");

  while (!blockMergeInfo.empty()) {
    Block *headerBlock = blockMergeInfo.begin()->first;
    BlockMergeInfo mergeInfo = blockMergeInfo.begin()->second;

    LLVM_DEBUG(llvm::dbgs() << "[cf] header block " << headerBlock << ":\n");
    LLVM_DEBUG(headerBlock->print(llvm::dbgs()));

    auto *mergeBlock = mergeInfo.mergeBlock;
    assert(mergeBlock && "merge block cannot be nullptr");
    if (!mergeBlock->args_empty())
      return emitError(unknownLoc, "OpPhi in loop merge block unimplemented");
    LLVM_DEBUG(llvm::dbgs() << "[cf] merge block " << mergeBlock << ":\n");
    LLVM_DEBUG(mergeBlock->print(llvm::dbgs()));

    auto *continueBlock = mergeInfo.continueBlock;
    if (continueBlock) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[cf] continue block " << continueBlock << ":\n");
      LLVM_DEBUG(continueBlock->print(llvm::dbgs()));
    }
    // Erase this case before calling into structurizer, who will update
    // blockMergeInfo.
    blockMergeInfo.erase(blockMergeInfo.begin());
    if (failed(ControlFlowStructurizer::structurize(
            mergeInfo.loc, mergeInfo.control, blockMergeInfo, headerBlock,
            mergeBlock, continueBlock)))
      return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "[cf] completed structurizing control flow\n");
  return success();
}

//===----------------------------------------------------------------------===//
// Debug
//===----------------------------------------------------------------------===//

Location spirv::Deserializer::createFileLineColLoc(OpBuilder opBuilder) {
  if (!debugLine)
    return unknownLoc;

  auto fileName = debugInfoMap.lookup(debugLine->fileID).str();
  if (fileName.empty())
    fileName = "<unknown>";
  return opBuilder.getFileLineColLoc(opBuilder.getIdentifier(fileName),
                                     debugLine->line, debugLine->col);
}

LogicalResult
spirv::Deserializer::processDebugLine(ArrayRef<uint32_t> operands) {
  // According to SPIR-V spec:
  // "This location information applies to the instructions physically
  // following this instruction, up to the first occurrence of any of the
  // following: the next end of block, the next OpLine instruction, or the next
  // OpNoLine instruction."
  if (operands.size() != 3)
    return emitError(unknownLoc, "OpLine must have 3 operands");
  debugLine = DebugLine(operands[0], operands[1], operands[2]);
  return success();
}

LogicalResult spirv::Deserializer::clearDebugLine() {
  debugLine = llvm::None;
  return success();
}

LogicalResult
spirv::Deserializer::processDebugString(ArrayRef<uint32_t> operands) {
  if (operands.size() < 2)
    return emitError(unknownLoc, "OpString needs at least 2 operands");

  if (!debugInfoMap.lookup(operands[0]).empty())
    return emitError(unknownLoc,
                     "duplicate debug string found for result <id> ")
           << operands[0];

  unsigned wordIndex = 1;
  StringRef debugString = decodeStringLiteral(operands, wordIndex);
  if (wordIndex != operands.size())
    return emitError(unknownLoc,
                     "unexpected trailing words in OpString instruction");

  debugInfoMap[operands[0]] = debugString;
  return success();
}
