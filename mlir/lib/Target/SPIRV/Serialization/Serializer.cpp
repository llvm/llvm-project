//===- Serializer.cpp - MLIR SPIR-V Serializer ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MLIR SPIR-V module to SPIR-V binary serializer.
//
//===----------------------------------------------------------------------===//

#include "Serializer.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/SPIRV/SPIRVBinaryUtils.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "spirv-serialization"

using namespace mlir;

/// Returns the merge block if the given `op` is a structured control flow op.
/// Otherwise returns nullptr.
static Block *getStructuredControlFlowOpMergeBlock(Operation *op) {
  if (auto selectionOp = dyn_cast<spirv::SelectionOp>(op))
    return selectionOp.getMergeBlock();
  if (auto loopOp = dyn_cast<spirv::LoopOp>(op))
    return loopOp.getMergeBlock();
  return nullptr;
}

/// Given a predecessor `block` for a block with arguments, returns the block
/// that should be used as the parent block for SPIR-V OpPhi instructions
/// corresponding to the block arguments.
static Block *getPhiIncomingBlock(Block *block) {
  // If the predecessor block in question is the entry block for a
  // spv.mlir.loop, we jump to this spv.mlir.loop from its enclosing block.
  if (block->isEntryBlock()) {
    if (auto loopOp = dyn_cast<spirv::LoopOp>(block->getParentOp())) {
      // Then the incoming parent block for OpPhi should be the merge block of
      // the structured control flow op before this loop.
      Operation *op = loopOp.getOperation();
      while ((op = op->getPrevNode()) != nullptr)
        if (Block *incomingBlock = getStructuredControlFlowOpMergeBlock(op))
          return incomingBlock;
      // Or the enclosing block itself if no structured control flow ops
      // exists before this loop.
      return loopOp->getBlock();
    }
  }

  // Otherwise, we jump from the given predecessor block. Try to see if there is
  // a structured control flow op inside it.
  for (Operation &op : llvm::reverse(block->getOperations())) {
    if (Block *incomingBlock = getStructuredControlFlowOpMergeBlock(&op))
      return incomingBlock;
  }
  return block;
}

namespace mlir {
namespace spirv {

/// Encodes an SPIR-V instruction with the given `opcode` and `operands` into
/// the given `binary` vector.
LogicalResult encodeInstructionInto(SmallVectorImpl<uint32_t> &binary,
                                    spirv::Opcode op,
                                    ArrayRef<uint32_t> operands) {
  uint32_t wordCount = 1 + operands.size();
  binary.push_back(spirv::getPrefixedOpcode(wordCount, op));
  binary.append(operands.begin(), operands.end());
  return success();
}

Serializer::Serializer(spirv::ModuleOp module, bool emitDebugInfo)
    : module(module), mlirBuilder(module.getContext()),
      emitDebugInfo(emitDebugInfo) {}

LogicalResult Serializer::serialize() {
  LLVM_DEBUG(llvm::dbgs() << "+++ starting serialization +++\n");

  if (failed(module.verify()))
    return failure();

  // TODO: handle the other sections
  processCapability();
  processExtension();
  processMemoryModel();
  processDebugInfo();

  // Iterate over the module body to serialize it. Assumptions are that there is
  // only one basic block in the moduleOp
  for (auto &op : *module.getBody()) {
    if (failed(processOperation(&op))) {
      return failure();
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "+++ completed serialization +++\n");
  return success();
}

void Serializer::collect(SmallVectorImpl<uint32_t> &binary) {
  auto moduleSize = spirv::kHeaderWordCount + capabilities.size() +
                    extensions.size() + extendedSets.size() +
                    memoryModel.size() + entryPoints.size() +
                    executionModes.size() + decorations.size() +
                    typesGlobalValues.size() + functions.size();

  binary.clear();
  binary.reserve(moduleSize);

  spirv::appendModuleHeader(binary, module.vce_triple()->getVersion(), nextID);
  binary.append(capabilities.begin(), capabilities.end());
  binary.append(extensions.begin(), extensions.end());
  binary.append(extendedSets.begin(), extendedSets.end());
  binary.append(memoryModel.begin(), memoryModel.end());
  binary.append(entryPoints.begin(), entryPoints.end());
  binary.append(executionModes.begin(), executionModes.end());
  binary.append(debug.begin(), debug.end());
  binary.append(names.begin(), names.end());
  binary.append(decorations.begin(), decorations.end());
  binary.append(typesGlobalValues.begin(), typesGlobalValues.end());
  binary.append(functions.begin(), functions.end());
}

#ifndef NDEBUG
void Serializer::printValueIDMap(raw_ostream &os) {
  os << "\n= Value <id> Map =\n\n";
  for (auto valueIDPair : valueIDMap) {
    Value val = valueIDPair.first;
    os << "  " << val << " "
       << "id = " << valueIDPair.second << ' ';
    if (auto *op = val.getDefiningOp()) {
      os << "from op '" << op->getName() << "'";
    } else if (auto arg = val.dyn_cast<BlockArgument>()) {
      Block *block = arg.getOwner();
      os << "from argument of block " << block << ' ';
      os << " in op '" << block->getParentOp()->getName() << "'";
    }
    os << '\n';
  }
}
#endif

//===----------------------------------------------------------------------===//
// Module structure
//===----------------------------------------------------------------------===//

uint32_t Serializer::getOrCreateFunctionID(StringRef fnName) {
  auto funcID = funcIDMap.lookup(fnName);
  if (!funcID) {
    funcID = getNextID();
    funcIDMap[fnName] = funcID;
  }
  return funcID;
}

void Serializer::processCapability() {
  for (auto cap : module.vce_triple()->getCapabilities())
    (void)encodeInstructionInto(capabilities, spirv::Opcode::OpCapability,
                                {static_cast<uint32_t>(cap)});
}

void Serializer::processDebugInfo() {
  if (!emitDebugInfo)
    return;
  auto fileLoc = module.getLoc().dyn_cast<FileLineColLoc>();
  auto fileName = fileLoc ? fileLoc.getFilename().strref() : "<unknown>";
  fileID = getNextID();
  SmallVector<uint32_t, 16> operands;
  operands.push_back(fileID);
  (void)spirv::encodeStringLiteralInto(operands, fileName);
  (void)encodeInstructionInto(debug, spirv::Opcode::OpString, operands);
  // TODO: Encode more debug instructions.
}

void Serializer::processExtension() {
  llvm::SmallVector<uint32_t, 16> extName;
  for (spirv::Extension ext : module.vce_triple()->getExtensions()) {
    extName.clear();
    (void)spirv::encodeStringLiteralInto(extName,
                                         spirv::stringifyExtension(ext));
    (void)encodeInstructionInto(extensions, spirv::Opcode::OpExtension,
                                extName);
  }
}

void Serializer::processMemoryModel() {
  uint32_t mm = module->getAttrOfType<IntegerAttr>("memory_model").getInt();
  uint32_t am = module->getAttrOfType<IntegerAttr>("addressing_model").getInt();

  (void)encodeInstructionInto(memoryModel, spirv::Opcode::OpMemoryModel,
                              {am, mm});
}

LogicalResult Serializer::processDecoration(Location loc, uint32_t resultID,
                                            NamedAttribute attr) {
  auto attrName = attr.first.strref();
  auto decorationName = llvm::convertToCamelFromSnakeCase(attrName, true);
  auto decoration = spirv::symbolizeDecoration(decorationName);
  if (!decoration) {
    return emitError(
               loc, "non-argument attributes expected to have snake-case-ified "
                    "decoration name, unhandled attribute with name : ")
           << attrName;
  }
  SmallVector<uint32_t, 1> args;
  switch (decoration.getValue()) {
  case spirv::Decoration::Binding:
  case spirv::Decoration::DescriptorSet:
  case spirv::Decoration::Location:
    if (auto intAttr = attr.second.dyn_cast<IntegerAttr>()) {
      args.push_back(intAttr.getValue().getZExtValue());
      break;
    }
    return emitError(loc, "expected integer attribute for ") << attrName;
  case spirv::Decoration::BuiltIn:
    if (auto strAttr = attr.second.dyn_cast<StringAttr>()) {
      auto enumVal = spirv::symbolizeBuiltIn(strAttr.getValue());
      if (enumVal) {
        args.push_back(static_cast<uint32_t>(enumVal.getValue()));
        break;
      }
      return emitError(loc, "invalid ")
             << attrName << " attribute " << strAttr.getValue();
    }
    return emitError(loc, "expected string attribute for ") << attrName;
  case spirv::Decoration::Aliased:
  case spirv::Decoration::Flat:
  case spirv::Decoration::NonReadable:
  case spirv::Decoration::NonWritable:
  case spirv::Decoration::NoPerspective:
  case spirv::Decoration::Restrict:
    // For unit attributes, the args list has no values so we do nothing
    if (auto unitAttr = attr.second.dyn_cast<UnitAttr>())
      break;
    return emitError(loc, "expected unit attribute for ") << attrName;
  default:
    return emitError(loc, "unhandled decoration ") << decorationName;
  }
  return emitDecoration(resultID, decoration.getValue(), args);
}

LogicalResult Serializer::processName(uint32_t resultID, StringRef name) {
  assert(!name.empty() && "unexpected empty string for OpName");

  SmallVector<uint32_t, 4> nameOperands;
  nameOperands.push_back(resultID);
  if (failed(spirv::encodeStringLiteralInto(nameOperands, name))) {
    return failure();
  }
  return encodeInstructionInto(names, spirv::Opcode::OpName, nameOperands);
}

template <>
LogicalResult Serializer::processTypeDecoration<spirv::ArrayType>(
    Location loc, spirv::ArrayType type, uint32_t resultID) {
  if (unsigned stride = type.getArrayStride()) {
    // OpDecorate %arrayTypeSSA ArrayStride strideLiteral
    return emitDecoration(resultID, spirv::Decoration::ArrayStride, {stride});
  }
  return success();
}

template <>
LogicalResult Serializer::processTypeDecoration<spirv::RuntimeArrayType>(
    Location loc, spirv::RuntimeArrayType type, uint32_t resultID) {
  if (unsigned stride = type.getArrayStride()) {
    // OpDecorate %arrayTypeSSA ArrayStride strideLiteral
    return emitDecoration(resultID, spirv::Decoration::ArrayStride, {stride});
  }
  return success();
}

LogicalResult Serializer::processMemberDecoration(
    uint32_t structID,
    const spirv::StructType::MemberDecorationInfo &memberDecoration) {
  SmallVector<uint32_t, 4> args(
      {structID, memberDecoration.memberIndex,
       static_cast<uint32_t>(memberDecoration.decoration)});
  if (memberDecoration.hasValue) {
    args.push_back(memberDecoration.decorationValue);
  }
  return encodeInstructionInto(decorations, spirv::Opcode::OpMemberDecorate,
                               args);
}

//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

// According to the SPIR-V spec "Validation Rules for Shader Capabilities":
// "Composite objects in the StorageBuffer, PhysicalStorageBuffer, Uniform, and
// PushConstant Storage Classes must be explicitly laid out."
bool Serializer::isInterfaceStructPtrType(Type type) const {
  if (auto ptrType = type.dyn_cast<spirv::PointerType>()) {
    switch (ptrType.getStorageClass()) {
    case spirv::StorageClass::PhysicalStorageBuffer:
    case spirv::StorageClass::PushConstant:
    case spirv::StorageClass::StorageBuffer:
    case spirv::StorageClass::Uniform:
      return ptrType.getPointeeType().isa<spirv::StructType>();
    default:
      break;
    }
  }
  return false;
}

LogicalResult Serializer::processType(Location loc, Type type,
                                      uint32_t &typeID) {
  // Maintains a set of names for nested identified struct types. This is used
  // to properly serialize recursive references.
  SetVector<StringRef> serializationCtx;
  return processTypeImpl(loc, type, typeID, serializationCtx);
}

LogicalResult
Serializer::processTypeImpl(Location loc, Type type, uint32_t &typeID,
                            SetVector<StringRef> &serializationCtx) {
  typeID = getTypeID(type);
  if (typeID) {
    return success();
  }
  typeID = getNextID();
  SmallVector<uint32_t, 4> operands;

  operands.push_back(typeID);
  auto typeEnum = spirv::Opcode::OpTypeVoid;
  bool deferSerialization = false;

  if ((type.isa<FunctionType>() &&
       succeeded(prepareFunctionType(loc, type.cast<FunctionType>(), typeEnum,
                                     operands))) ||
      succeeded(prepareBasicType(loc, type, typeID, typeEnum, operands,
                                 deferSerialization, serializationCtx))) {
    if (deferSerialization)
      return success();

    typeIDMap[type] = typeID;

    if (failed(encodeInstructionInto(typesGlobalValues, typeEnum, operands)))
      return failure();

    if (recursiveStructInfos.count(type) != 0) {
      // This recursive struct type is emitted already, now the OpTypePointer
      // instructions referring to recursive references are emitted as well.
      for (auto &ptrInfo : recursiveStructInfos[type]) {
        // TODO: This might not work if more than 1 recursive reference is
        // present in the struct.
        SmallVector<uint32_t, 4> ptrOperands;
        ptrOperands.push_back(ptrInfo.pointerTypeID);
        ptrOperands.push_back(static_cast<uint32_t>(ptrInfo.storageClass));
        ptrOperands.push_back(typeIDMap[type]);

        if (failed(encodeInstructionInto(
                typesGlobalValues, spirv::Opcode::OpTypePointer, ptrOperands)))
          return failure();
      }

      recursiveStructInfos[type].clear();
    }

    return success();
  }

  return failure();
}

LogicalResult Serializer::prepareBasicType(
    Location loc, Type type, uint32_t resultID, spirv::Opcode &typeEnum,
    SmallVectorImpl<uint32_t> &operands, bool &deferSerialization,
    SetVector<StringRef> &serializationCtx) {
  deferSerialization = false;

  if (isVoidType(type)) {
    typeEnum = spirv::Opcode::OpTypeVoid;
    return success();
  }

  if (auto intType = type.dyn_cast<IntegerType>()) {
    if (intType.getWidth() == 1) {
      typeEnum = spirv::Opcode::OpTypeBool;
      return success();
    }

    typeEnum = spirv::Opcode::OpTypeInt;
    operands.push_back(intType.getWidth());
    // SPIR-V OpTypeInt "Signedness specifies whether there are signed semantics
    // to preserve or validate.
    // 0 indicates unsigned, or no signedness semantics
    // 1 indicates signed semantics."
    operands.push_back(intType.isSigned() ? 1 : 0);
    return success();
  }

  if (auto floatType = type.dyn_cast<FloatType>()) {
    typeEnum = spirv::Opcode::OpTypeFloat;
    operands.push_back(floatType.getWidth());
    return success();
  }

  if (auto vectorType = type.dyn_cast<VectorType>()) {
    uint32_t elementTypeID = 0;
    if (failed(processTypeImpl(loc, vectorType.getElementType(), elementTypeID,
                               serializationCtx))) {
      return failure();
    }
    typeEnum = spirv::Opcode::OpTypeVector;
    operands.push_back(elementTypeID);
    operands.push_back(vectorType.getNumElements());
    return success();
  }

  if (auto imageType = type.dyn_cast<spirv::ImageType>()) {
    typeEnum = spirv::Opcode::OpTypeImage;
    uint32_t sampledTypeID = 0;
    if (failed(processType(loc, imageType.getElementType(), sampledTypeID)))
      return failure();

    operands.push_back(sampledTypeID);
    operands.push_back(static_cast<uint32_t>(imageType.getDim()));
    operands.push_back(static_cast<uint32_t>(imageType.getDepthInfo()));
    operands.push_back(static_cast<uint32_t>(imageType.getArrayedInfo()));
    operands.push_back(static_cast<uint32_t>(imageType.getSamplingInfo()));
    operands.push_back(static_cast<uint32_t>(imageType.getSamplerUseInfo()));
    operands.push_back(static_cast<uint32_t>(imageType.getImageFormat()));
    return success();
  }

  if (auto arrayType = type.dyn_cast<spirv::ArrayType>()) {
    typeEnum = spirv::Opcode::OpTypeArray;
    uint32_t elementTypeID = 0;
    if (failed(processTypeImpl(loc, arrayType.getElementType(), elementTypeID,
                               serializationCtx))) {
      return failure();
    }
    operands.push_back(elementTypeID);
    if (auto elementCountID = prepareConstantInt(
            loc, mlirBuilder.getI32IntegerAttr(arrayType.getNumElements()))) {
      operands.push_back(elementCountID);
    }
    return processTypeDecoration(loc, arrayType, resultID);
  }

  if (auto ptrType = type.dyn_cast<spirv::PointerType>()) {
    uint32_t pointeeTypeID = 0;
    spirv::StructType pointeeStruct =
        ptrType.getPointeeType().dyn_cast<spirv::StructType>();

    if (pointeeStruct && pointeeStruct.isIdentified() &&
        serializationCtx.count(pointeeStruct.getIdentifier()) != 0) {
      // A recursive reference to an enclosing struct is found.
      //
      // 1. Prepare an OpTypeForwardPointer with resultID and the ptr storage
      // class as operands.
      SmallVector<uint32_t, 2> forwardPtrOperands;
      forwardPtrOperands.push_back(resultID);
      forwardPtrOperands.push_back(
          static_cast<uint32_t>(ptrType.getStorageClass()));

      (void)encodeInstructionInto(typesGlobalValues,
                                  spirv::Opcode::OpTypeForwardPointer,
                                  forwardPtrOperands);

      // 2. Find the pointee (enclosing) struct.
      auto structType = spirv::StructType::getIdentified(
          module.getContext(), pointeeStruct.getIdentifier());

      if (!structType)
        return failure();

      // 3. Mark the OpTypePointer that is supposed to be emitted by this call
      // as deferred.
      deferSerialization = true;

      // 4. Record the info needed to emit the deferred OpTypePointer
      // instruction when the enclosing struct is completely serialized.
      recursiveStructInfos[structType].push_back(
          {resultID, ptrType.getStorageClass()});
    } else {
      if (failed(processTypeImpl(loc, ptrType.getPointeeType(), pointeeTypeID,
                                 serializationCtx)))
        return failure();
    }

    typeEnum = spirv::Opcode::OpTypePointer;
    operands.push_back(static_cast<uint32_t>(ptrType.getStorageClass()));
    operands.push_back(pointeeTypeID);
    return success();
  }

  if (auto runtimeArrayType = type.dyn_cast<spirv::RuntimeArrayType>()) {
    uint32_t elementTypeID = 0;
    if (failed(processTypeImpl(loc, runtimeArrayType.getElementType(),
                               elementTypeID, serializationCtx))) {
      return failure();
    }
    typeEnum = spirv::Opcode::OpTypeRuntimeArray;
    operands.push_back(elementTypeID);
    return processTypeDecoration(loc, runtimeArrayType, resultID);
  }

  if (auto sampledImageType = type.dyn_cast<spirv::SampledImageType>()) {
    typeEnum = spirv::Opcode::OpTypeSampledImage;
    uint32_t imageTypeID = 0;
    if (failed(
            processType(loc, sampledImageType.getImageType(), imageTypeID))) {
      return failure();
    }
    operands.push_back(imageTypeID);
    return success();
  }

  if (auto structType = type.dyn_cast<spirv::StructType>()) {
    if (structType.isIdentified()) {
      (void)processName(resultID, structType.getIdentifier());
      serializationCtx.insert(structType.getIdentifier());
    }

    bool hasOffset = structType.hasOffset();
    for (auto elementIndex :
         llvm::seq<uint32_t>(0, structType.getNumElements())) {
      uint32_t elementTypeID = 0;
      if (failed(processTypeImpl(loc, structType.getElementType(elementIndex),
                                 elementTypeID, serializationCtx))) {
        return failure();
      }
      operands.push_back(elementTypeID);
      if (hasOffset) {
        // Decorate each struct member with an offset
        spirv::StructType::MemberDecorationInfo offsetDecoration{
            elementIndex, /*hasValue=*/1, spirv::Decoration::Offset,
            static_cast<uint32_t>(structType.getMemberOffset(elementIndex))};
        if (failed(processMemberDecoration(resultID, offsetDecoration))) {
          return emitError(loc, "cannot decorate ")
                 << elementIndex << "-th member of " << structType
                 << " with its offset";
        }
      }
    }
    SmallVector<spirv::StructType::MemberDecorationInfo, 4> memberDecorations;
    structType.getMemberDecorations(memberDecorations);

    for (auto &memberDecoration : memberDecorations) {
      if (failed(processMemberDecoration(resultID, memberDecoration))) {
        return emitError(loc, "cannot decorate ")
               << static_cast<uint32_t>(memberDecoration.memberIndex)
               << "-th member of " << structType << " with "
               << stringifyDecoration(memberDecoration.decoration);
      }
    }

    typeEnum = spirv::Opcode::OpTypeStruct;

    if (structType.isIdentified())
      serializationCtx.remove(structType.getIdentifier());

    return success();
  }

  if (auto cooperativeMatrixType =
          type.dyn_cast<spirv::CooperativeMatrixNVType>()) {
    uint32_t elementTypeID = 0;
    if (failed(processTypeImpl(loc, cooperativeMatrixType.getElementType(),
                               elementTypeID, serializationCtx))) {
      return failure();
    }
    typeEnum = spirv::Opcode::OpTypeCooperativeMatrixNV;
    auto getConstantOp = [&](uint32_t id) {
      auto attr = IntegerAttr::get(IntegerType::get(type.getContext(), 32), id);
      return prepareConstantInt(loc, attr);
    };
    operands.push_back(elementTypeID);
    operands.push_back(
        getConstantOp(static_cast<uint32_t>(cooperativeMatrixType.getScope())));
    operands.push_back(getConstantOp(cooperativeMatrixType.getRows()));
    operands.push_back(getConstantOp(cooperativeMatrixType.getColumns()));
    return success();
  }

  if (auto matrixType = type.dyn_cast<spirv::MatrixType>()) {
    uint32_t elementTypeID = 0;
    if (failed(processTypeImpl(loc, matrixType.getColumnType(), elementTypeID,
                               serializationCtx))) {
      return failure();
    }
    typeEnum = spirv::Opcode::OpTypeMatrix;
    operands.push_back(elementTypeID);
    operands.push_back(matrixType.getNumColumns());
    return success();
  }

  // TODO: Handle other types.
  return emitError(loc, "unhandled type in serialization: ") << type;
}

LogicalResult
Serializer::prepareFunctionType(Location loc, FunctionType type,
                                spirv::Opcode &typeEnum,
                                SmallVectorImpl<uint32_t> &operands) {
  typeEnum = spirv::Opcode::OpTypeFunction;
  assert(type.getNumResults() <= 1 &&
         "serialization supports only a single return value");
  uint32_t resultID = 0;
  if (failed(processType(
          loc, type.getNumResults() == 1 ? type.getResult(0) : getVoidType(),
          resultID))) {
    return failure();
  }
  operands.push_back(resultID);
  for (auto &res : type.getInputs()) {
    uint32_t argTypeID = 0;
    if (failed(processType(loc, res, argTypeID))) {
      return failure();
    }
    operands.push_back(argTypeID);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Constant
//===----------------------------------------------------------------------===//

uint32_t Serializer::prepareConstant(Location loc, Type constType,
                                     Attribute valueAttr) {
  if (auto id = prepareConstantScalar(loc, valueAttr)) {
    return id;
  }

  // This is a composite literal. We need to handle each component separately
  // and then emit an OpConstantComposite for the whole.

  if (auto id = getConstantID(valueAttr)) {
    return id;
  }

  uint32_t typeID = 0;
  if (failed(processType(loc, constType, typeID))) {
    return 0;
  }

  uint32_t resultID = 0;
  if (auto attr = valueAttr.dyn_cast<DenseElementsAttr>()) {
    int rank = attr.getType().dyn_cast<ShapedType>().getRank();
    SmallVector<uint64_t, 4> index(rank);
    resultID = prepareDenseElementsConstant(loc, constType, attr,
                                            /*dim=*/0, index);
  } else if (auto arrayAttr = valueAttr.dyn_cast<ArrayAttr>()) {
    resultID = prepareArrayConstant(loc, constType, arrayAttr);
  }

  if (resultID == 0) {
    emitError(loc, "cannot serialize attribute: ") << valueAttr;
    return 0;
  }

  constIDMap[valueAttr] = resultID;
  return resultID;
}

uint32_t Serializer::prepareArrayConstant(Location loc, Type constType,
                                          ArrayAttr attr) {
  uint32_t typeID = 0;
  if (failed(processType(loc, constType, typeID))) {
    return 0;
  }

  uint32_t resultID = getNextID();
  SmallVector<uint32_t, 4> operands = {typeID, resultID};
  operands.reserve(attr.size() + 2);
  auto elementType = constType.cast<spirv::ArrayType>().getElementType();
  for (Attribute elementAttr : attr) {
    if (auto elementID = prepareConstant(loc, elementType, elementAttr)) {
      operands.push_back(elementID);
    } else {
      return 0;
    }
  }
  spirv::Opcode opcode = spirv::Opcode::OpConstantComposite;
  (void)encodeInstructionInto(typesGlobalValues, opcode, operands);

  return resultID;
}

// TODO: Turn the below function into iterative function, instead of
// recursive function.
uint32_t
Serializer::prepareDenseElementsConstant(Location loc, Type constType,
                                         DenseElementsAttr valueAttr, int dim,
                                         MutableArrayRef<uint64_t> index) {
  auto shapedType = valueAttr.getType().dyn_cast<ShapedType>();
  assert(dim <= shapedType.getRank());
  if (shapedType.getRank() == dim) {
    if (auto attr = valueAttr.dyn_cast<DenseIntElementsAttr>()) {
      return attr.getType().getElementType().isInteger(1)
                 ? prepareConstantBool(loc, attr.getValue<BoolAttr>(index))
                 : prepareConstantInt(loc, attr.getValue<IntegerAttr>(index));
    }
    if (auto attr = valueAttr.dyn_cast<DenseFPElementsAttr>()) {
      return prepareConstantFp(loc, attr.getValue<FloatAttr>(index));
    }
    return 0;
  }

  uint32_t typeID = 0;
  if (failed(processType(loc, constType, typeID))) {
    return 0;
  }

  uint32_t resultID = getNextID();
  SmallVector<uint32_t, 4> operands = {typeID, resultID};
  operands.reserve(shapedType.getDimSize(dim) + 2);
  auto elementType = constType.cast<spirv::CompositeType>().getElementType(0);
  for (int i = 0; i < shapedType.getDimSize(dim); ++i) {
    index[dim] = i;
    if (auto elementID = prepareDenseElementsConstant(
            loc, elementType, valueAttr, dim + 1, index)) {
      operands.push_back(elementID);
    } else {
      return 0;
    }
  }
  spirv::Opcode opcode = spirv::Opcode::OpConstantComposite;
  (void)encodeInstructionInto(typesGlobalValues, opcode, operands);

  return resultID;
}

uint32_t Serializer::prepareConstantScalar(Location loc, Attribute valueAttr,
                                           bool isSpec) {
  if (auto floatAttr = valueAttr.dyn_cast<FloatAttr>()) {
    return prepareConstantFp(loc, floatAttr, isSpec);
  }
  if (auto boolAttr = valueAttr.dyn_cast<BoolAttr>()) {
    return prepareConstantBool(loc, boolAttr, isSpec);
  }
  if (auto intAttr = valueAttr.dyn_cast<IntegerAttr>()) {
    return prepareConstantInt(loc, intAttr, isSpec);
  }

  return 0;
}

uint32_t Serializer::prepareConstantBool(Location loc, BoolAttr boolAttr,
                                         bool isSpec) {
  if (!isSpec) {
    // We can de-duplicate normal constants, but not specialization constants.
    if (auto id = getConstantID(boolAttr)) {
      return id;
    }
  }

  // Process the type for this bool literal
  uint32_t typeID = 0;
  if (failed(processType(loc, boolAttr.getType(), typeID))) {
    return 0;
  }

  auto resultID = getNextID();
  auto opcode = boolAttr.getValue()
                    ? (isSpec ? spirv::Opcode::OpSpecConstantTrue
                              : spirv::Opcode::OpConstantTrue)
                    : (isSpec ? spirv::Opcode::OpSpecConstantFalse
                              : spirv::Opcode::OpConstantFalse);
  (void)encodeInstructionInto(typesGlobalValues, opcode, {typeID, resultID});

  if (!isSpec) {
    constIDMap[boolAttr] = resultID;
  }
  return resultID;
}

uint32_t Serializer::prepareConstantInt(Location loc, IntegerAttr intAttr,
                                        bool isSpec) {
  if (!isSpec) {
    // We can de-duplicate normal constants, but not specialization constants.
    if (auto id = getConstantID(intAttr)) {
      return id;
    }
  }

  // Process the type for this integer literal
  uint32_t typeID = 0;
  if (failed(processType(loc, intAttr.getType(), typeID))) {
    return 0;
  }

  auto resultID = getNextID();
  APInt value = intAttr.getValue();
  unsigned bitwidth = value.getBitWidth();
  bool isSigned = value.isSignedIntN(bitwidth);

  auto opcode =
      isSpec ? spirv::Opcode::OpSpecConstant : spirv::Opcode::OpConstant;

  // According to SPIR-V spec, "When the type's bit width is less than 32-bits,
  // the literal's value appears in the low-order bits of the word, and the
  // high-order bits must be 0 for a floating-point type, or 0 for an integer
  // type with Signedness of 0, or sign extended when Signedness is 1."
  if (bitwidth == 32 || bitwidth == 16) {
    uint32_t word = 0;
    if (isSigned) {
      word = static_cast<int32_t>(value.getSExtValue());
    } else {
      word = static_cast<uint32_t>(value.getZExtValue());
    }
    (void)encodeInstructionInto(typesGlobalValues, opcode,
                                {typeID, resultID, word});
  }
  // According to SPIR-V spec: "When the type's bit width is larger than one
  // word, the literal’s low-order words appear first."
  else if (bitwidth == 64) {
    struct DoubleWord {
      uint32_t word1;
      uint32_t word2;
    } words;
    if (isSigned) {
      words = llvm::bit_cast<DoubleWord>(value.getSExtValue());
    } else {
      words = llvm::bit_cast<DoubleWord>(value.getZExtValue());
    }
    (void)encodeInstructionInto(typesGlobalValues, opcode,
                                {typeID, resultID, words.word1, words.word2});
  } else {
    std::string valueStr;
    llvm::raw_string_ostream rss(valueStr);
    value.print(rss, /*isSigned=*/false);

    emitError(loc, "cannot serialize ")
        << bitwidth << "-bit integer literal: " << rss.str();
    return 0;
  }

  if (!isSpec) {
    constIDMap[intAttr] = resultID;
  }
  return resultID;
}

uint32_t Serializer::prepareConstantFp(Location loc, FloatAttr floatAttr,
                                       bool isSpec) {
  if (!isSpec) {
    // We can de-duplicate normal constants, but not specialization constants.
    if (auto id = getConstantID(floatAttr)) {
      return id;
    }
  }

  // Process the type for this float literal
  uint32_t typeID = 0;
  if (failed(processType(loc, floatAttr.getType(), typeID))) {
    return 0;
  }

  auto resultID = getNextID();
  APFloat value = floatAttr.getValue();
  APInt intValue = value.bitcastToAPInt();

  auto opcode =
      isSpec ? spirv::Opcode::OpSpecConstant : spirv::Opcode::OpConstant;

  if (&value.getSemantics() == &APFloat::IEEEsingle()) {
    uint32_t word = llvm::bit_cast<uint32_t>(value.convertToFloat());
    (void)encodeInstructionInto(typesGlobalValues, opcode,
                                {typeID, resultID, word});
  } else if (&value.getSemantics() == &APFloat::IEEEdouble()) {
    struct DoubleWord {
      uint32_t word1;
      uint32_t word2;
    } words = llvm::bit_cast<DoubleWord>(value.convertToDouble());
    (void)encodeInstructionInto(typesGlobalValues, opcode,
                                {typeID, resultID, words.word1, words.word2});
  } else if (&value.getSemantics() == &APFloat::IEEEhalf()) {
    uint32_t word =
        static_cast<uint32_t>(value.bitcastToAPInt().getZExtValue());
    (void)encodeInstructionInto(typesGlobalValues, opcode,
                                {typeID, resultID, word});
  } else {
    std::string valueStr;
    llvm::raw_string_ostream rss(valueStr);
    value.print(rss);

    emitError(loc, "cannot serialize ")
        << floatAttr.getType() << "-typed float literal: " << rss.str();
    return 0;
  }

  if (!isSpec) {
    constIDMap[floatAttr] = resultID;
  }
  return resultID;
}

//===----------------------------------------------------------------------===//
// Control flow
//===----------------------------------------------------------------------===//

uint32_t Serializer::getOrCreateBlockID(Block *block) {
  if (uint32_t id = getBlockID(block))
    return id;
  return blockIDMap[block] = getNextID();
}

LogicalResult
Serializer::processBlock(Block *block, bool omitLabel,
                         function_ref<void()> actionBeforeTerminator) {
  LLVM_DEBUG(llvm::dbgs() << "processing block " << block << ":\n");
  LLVM_DEBUG(block->print(llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << '\n');
  if (!omitLabel) {
    uint32_t blockID = getOrCreateBlockID(block);
    LLVM_DEBUG(llvm::dbgs()
               << "[block] " << block << " (id = " << blockID << ")\n");

    // Emit OpLabel for this block.
    (void)encodeInstructionInto(functionBody, spirv::Opcode::OpLabel,
                                {blockID});
  }

  // Emit OpPhi instructions for block arguments, if any.
  if (failed(emitPhiForBlockArguments(block)))
    return failure();

  // Process each op in this block except the terminator.
  for (auto &op : llvm::make_range(block->begin(), std::prev(block->end()))) {
    if (failed(processOperation(&op)))
      return failure();
  }

  // Process the terminator.
  if (actionBeforeTerminator)
    actionBeforeTerminator();
  if (failed(processOperation(&block->back())))
    return failure();

  return success();
}

LogicalResult Serializer::emitPhiForBlockArguments(Block *block) {
  // Nothing to do if this block has no arguments or it's the entry block, which
  // always has the same arguments as the function signature.
  if (block->args_empty() || block->isEntryBlock())
    return success();

  // If the block has arguments, we need to create SPIR-V OpPhi instructions.
  // A SPIR-V OpPhi instruction is of the syntax:
  //   OpPhi | result type | result <id> | (value <id>, parent block <id>) pair
  // So we need to collect all predecessor blocks and the arguments they send
  // to this block.
  SmallVector<std::pair<Block *, OperandRange>, 4> predecessors;
  for (Block *predecessor : block->getPredecessors()) {
    auto *terminator = predecessor->getTerminator();
    // The predecessor here is the immediate one according to MLIR's IR
    // structure. It does not directly map to the incoming parent block for the
    // OpPhi instructions at SPIR-V binary level. This is because structured
    // control flow ops are serialized to multiple SPIR-V blocks. If there is a
    // spv.mlir.selection/spv.mlir.loop op in the MLIR predecessor block, the
    // branch op jumping to the OpPhi's block then resides in the previous
    // structured control flow op's merge block.
    predecessor = getPhiIncomingBlock(predecessor);
    if (auto branchOp = dyn_cast<spirv::BranchOp>(terminator)) {
      predecessors.emplace_back(predecessor, branchOp.getOperands());
    } else if (auto branchCondOp =
                   dyn_cast<spirv::BranchConditionalOp>(terminator)) {
      Optional<OperandRange> blockOperands;

      for (auto successorIdx :
           llvm::seq<unsigned>(0, predecessor->getNumSuccessors()))
        if (predecessor->getSuccessors()[successorIdx] == block) {
          blockOperands = branchCondOp.getSuccessorOperands(successorIdx);
          break;
        }

      assert(blockOperands && !blockOperands->empty() &&
             "expected non-empty block operand range");
      predecessors.emplace_back(predecessor, *blockOperands);
    } else {
      return terminator->emitError("unimplemented terminator for Phi creation");
    }
  }

  // Then create OpPhi instruction for each of the block argument.
  for (auto argIndex : llvm::seq<unsigned>(0, block->getNumArguments())) {
    BlockArgument arg = block->getArgument(argIndex);

    // Get the type <id> and result <id> for this OpPhi instruction.
    uint32_t phiTypeID = 0;
    if (failed(processType(arg.getLoc(), arg.getType(), phiTypeID)))
      return failure();
    uint32_t phiID = getNextID();

    LLVM_DEBUG(llvm::dbgs() << "[phi] for block argument #" << argIndex << ' '
                            << arg << " (id = " << phiID << ")\n");

    // Prepare the (value <id>, parent block <id>) pairs.
    SmallVector<uint32_t, 8> phiArgs;
    phiArgs.push_back(phiTypeID);
    phiArgs.push_back(phiID);

    for (auto predIndex : llvm::seq<unsigned>(0, predecessors.size())) {
      Value value = predecessors[predIndex].second[argIndex];
      uint32_t predBlockId = getOrCreateBlockID(predecessors[predIndex].first);
      LLVM_DEBUG(llvm::dbgs() << "[phi] use predecessor (id = " << predBlockId
                              << ") value " << value << ' ');
      // Each pair is a value <id> ...
      uint32_t valueId = getValueID(value);
      if (valueId == 0) {
        // The op generating this value hasn't been visited yet so we don't have
        // an <id> assigned yet. Record this to fix up later.
        LLVM_DEBUG(llvm::dbgs() << "(need to fix)\n");
        deferredPhiValues[value].push_back(functionBody.size() + 1 +
                                           phiArgs.size());
      } else {
        LLVM_DEBUG(llvm::dbgs() << "(id = " << valueId << ")\n");
      }
      phiArgs.push_back(valueId);
      // ... and a parent block <id>.
      phiArgs.push_back(predBlockId);
    }

    (void)encodeInstructionInto(functionBody, spirv::Opcode::OpPhi, phiArgs);
    valueIDMap[arg] = phiID;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Operation
//===----------------------------------------------------------------------===//

LogicalResult Serializer::encodeExtensionInstruction(
    Operation *op, StringRef extensionSetName, uint32_t extensionOpcode,
    ArrayRef<uint32_t> operands) {
  // Check if the extension has been imported.
  auto &setID = extendedInstSetIDMap[extensionSetName];
  if (!setID) {
    setID = getNextID();
    SmallVector<uint32_t, 16> importOperands;
    importOperands.push_back(setID);
    if (failed(
            spirv::encodeStringLiteralInto(importOperands, extensionSetName)) ||
        failed(encodeInstructionInto(
            extendedSets, spirv::Opcode::OpExtInstImport, importOperands))) {
      return failure();
    }
  }

  // The first two operands are the result type <id> and result <id>. The set
  // <id> and the opcode need to be insert after this.
  if (operands.size() < 2) {
    return op->emitError("extended instructions must have a result encoding");
  }
  SmallVector<uint32_t, 8> extInstOperands;
  extInstOperands.reserve(operands.size() + 2);
  extInstOperands.append(operands.begin(), std::next(operands.begin(), 2));
  extInstOperands.push_back(setID);
  extInstOperands.push_back(extensionOpcode);
  extInstOperands.append(std::next(operands.begin(), 2), operands.end());
  return encodeInstructionInto(functionBody, spirv::Opcode::OpExtInst,
                               extInstOperands);
}

LogicalResult Serializer::processOperation(Operation *opInst) {
  LLVM_DEBUG(llvm::dbgs() << "[op] '" << opInst->getName() << "'\n");

  // First dispatch the ops that do not directly mirror an instruction from
  // the SPIR-V spec.
  return TypeSwitch<Operation *, LogicalResult>(opInst)
      .Case([&](spirv::AddressOfOp op) { return processAddressOfOp(op); })
      .Case([&](spirv::BranchOp op) { return processBranchOp(op); })
      .Case([&](spirv::BranchConditionalOp op) {
        return processBranchConditionalOp(op);
      })
      .Case([&](spirv::ConstantOp op) { return processConstantOp(op); })
      .Case([&](spirv::FuncOp op) { return processFuncOp(op); })
      .Case([&](spirv::GlobalVariableOp op) {
        return processGlobalVariableOp(op);
      })
      .Case([&](spirv::LoopOp op) { return processLoopOp(op); })
      .Case([&](spirv::ReferenceOfOp op) { return processReferenceOfOp(op); })
      .Case([&](spirv::SelectionOp op) { return processSelectionOp(op); })
      .Case([&](spirv::SpecConstantOp op) { return processSpecConstantOp(op); })
      .Case([&](spirv::SpecConstantCompositeOp op) {
        return processSpecConstantCompositeOp(op);
      })
      .Case([&](spirv::SpecConstantOperationOp op) {
        return processSpecConstantOperationOp(op);
      })
      .Case([&](spirv::UndefOp op) { return processUndefOp(op); })
      .Case([&](spirv::VariableOp op) { return processVariableOp(op); })

      // Then handle all the ops that directly mirror SPIR-V instructions with
      // auto-generated methods.
      .Default(
          [&](Operation *op) { return dispatchToAutogenSerialization(op); });
}

LogicalResult Serializer::processOpWithoutGrammarAttr(Operation *op,
                                                      StringRef extInstSet,
                                                      uint32_t opcode) {
  SmallVector<uint32_t, 4> operands;
  Location loc = op->getLoc();

  uint32_t resultID = 0;
  if (op->getNumResults() != 0) {
    uint32_t resultTypeID = 0;
    if (failed(processType(loc, op->getResult(0).getType(), resultTypeID)))
      return failure();
    operands.push_back(resultTypeID);

    resultID = getNextID();
    operands.push_back(resultID);
    valueIDMap[op->getResult(0)] = resultID;
  };

  for (Value operand : op->getOperands())
    operands.push_back(getValueID(operand));

  (void)emitDebugLine(functionBody, loc);

  if (extInstSet.empty()) {
    (void)encodeInstructionInto(functionBody,
                                static_cast<spirv::Opcode>(opcode), operands);
  } else {
    (void)encodeExtensionInstruction(op, extInstSet, opcode, operands);
  }

  if (op->getNumResults() != 0) {
    for (auto attr : op->getAttrs()) {
      if (failed(processDecoration(loc, resultID, attr)))
        return failure();
    }
  }

  return success();
}

LogicalResult Serializer::emitDecoration(uint32_t target,
                                         spirv::Decoration decoration,
                                         ArrayRef<uint32_t> params) {
  uint32_t wordCount = 3 + params.size();
  decorations.push_back(
      spirv::getPrefixedOpcode(wordCount, spirv::Opcode::OpDecorate));
  decorations.push_back(target);
  decorations.push_back(static_cast<uint32_t>(decoration));
  decorations.append(params.begin(), params.end());
  return success();
}

LogicalResult Serializer::emitDebugLine(SmallVectorImpl<uint32_t> &binary,
                                        Location loc) {
  if (!emitDebugInfo)
    return success();

  if (lastProcessedWasMergeInst) {
    lastProcessedWasMergeInst = false;
    return success();
  }

  auto fileLoc = loc.dyn_cast<FileLineColLoc>();
  if (fileLoc)
    (void)encodeInstructionInto(
        binary, spirv::Opcode::OpLine,
        {fileID, fileLoc.getLine(), fileLoc.getColumn()});
  return success();
}
} // namespace spirv
} // namespace mlir
