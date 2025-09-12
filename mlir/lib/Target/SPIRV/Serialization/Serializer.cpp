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
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Target/SPIRV/SPIRVBinaryUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Debug.h"
#include <cstdint>
#include <optional>

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
  // spirv.mlir.loop, we jump to this spirv.mlir.loop from its enclosing block.
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

static bool isZeroValue(Attribute attr) {
  if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    return floatAttr.getValue().isZero();
  }
  if (auto boolAttr = dyn_cast<BoolAttr>(attr)) {
    return !boolAttr.getValue();
  }
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    return intAttr.getValue().isZero();
  }
  if (auto splatElemAttr = dyn_cast<SplatElementsAttr>(attr)) {
    return isZeroValue(splatElemAttr.getSplatValue<Attribute>());
  }
  if (auto denseElemAttr = dyn_cast<DenseElementsAttr>(attr)) {
    return all_of(denseElemAttr.getValues<Attribute>(), isZeroValue);
  }
  return false;
}

namespace mlir {
namespace spirv {

/// Encodes an SPIR-V instruction with the given `opcode` and `operands` into
/// the given `binary` vector.
void encodeInstructionInto(SmallVectorImpl<uint32_t> &binary, spirv::Opcode op,
                           ArrayRef<uint32_t> operands) {
  uint32_t wordCount = 1 + operands.size();
  binary.push_back(spirv::getPrefixedOpcode(wordCount, op));
  binary.append(operands.begin(), operands.end());
}

Serializer::Serializer(spirv::ModuleOp module,
                       const SerializationOptions &options)
    : module(module), mlirBuilder(module.getContext()), options(options) {}

LogicalResult Serializer::serialize() {
  LLVM_DEBUG(llvm::dbgs() << "+++ starting serialization +++\n");

  if (failed(module.verifyInvariants()))
    return failure();

  // TODO: handle the other sections
  processCapability();
  if (failed(processExtension())) {
    return failure();
  }
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
                    typesGlobalValues.size() + functions.size() + graphs.size();

  binary.clear();
  binary.reserve(moduleSize);

  spirv::appendModuleHeader(binary, module.getVceTriple()->getVersion(),
                            nextID);
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
  binary.append(graphs.begin(), graphs.end());
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
    } else if (auto arg = dyn_cast<BlockArgument>(val)) {
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
  for (auto cap : module.getVceTriple()->getCapabilities())
    encodeInstructionInto(capabilities, spirv::Opcode::OpCapability,
                          {static_cast<uint32_t>(cap)});
}

void Serializer::processDebugInfo() {
  if (!options.emitDebugInfo)
    return;
  auto fileLoc = dyn_cast<FileLineColLoc>(module.getLoc());
  auto fileName = fileLoc ? fileLoc.getFilename().strref() : "<unknown>";
  fileID = getNextID();
  SmallVector<uint32_t, 16> operands;
  operands.push_back(fileID);
  spirv::encodeStringLiteralInto(operands, fileName);
  encodeInstructionInto(debug, spirv::Opcode::OpString, operands);
  // TODO: Encode more debug instructions.
}

LogicalResult Serializer::processExtension() {
  llvm::SmallVector<uint32_t, 16> extName;
  llvm::SmallSet<Extension, 4> deducedExts(
      llvm::from_range, module.getVceTriple()->getExtensions());
  auto nonSemanticInfoExt = spirv::Extension::SPV_KHR_non_semantic_info;
  if (options.emitDebugInfo && !deducedExts.contains(nonSemanticInfoExt)) {
    TargetEnvAttr targetEnvAttr = lookupTargetEnvOrDefault(module);
    if (!is_contained(targetEnvAttr.getExtensions(), nonSemanticInfoExt))
      return module.emitError(
          "SPV_KHR_non_semantic_info extension not available");
    deducedExts.insert(nonSemanticInfoExt);
  }
  for (spirv::Extension ext : deducedExts) {
    extName.clear();
    spirv::encodeStringLiteralInto(extName, spirv::stringifyExtension(ext));
    encodeInstructionInto(extensions, spirv::Opcode::OpExtension, extName);
  }
  return success();
}

void Serializer::processMemoryModel() {
  StringAttr memoryModelName = module.getMemoryModelAttrName();
  auto mm = static_cast<uint32_t>(
      module->getAttrOfType<spirv::MemoryModelAttr>(memoryModelName)
          .getValue());

  StringAttr addressingModelName = module.getAddressingModelAttrName();
  auto am = static_cast<uint32_t>(
      module->getAttrOfType<spirv::AddressingModelAttr>(addressingModelName)
          .getValue());

  encodeInstructionInto(memoryModel, spirv::Opcode::OpMemoryModel, {am, mm});
}

static std::string getDecorationName(StringRef attrName) {
  // convertToCamelFromSnakeCase will convert this to FpFastMathMode instead of
  // expected FPFastMathMode.
  if (attrName == "fp_fast_math_mode")
    return "FPFastMathMode";
  // similar here
  if (attrName == "fp_rounding_mode")
    return "FPRoundingMode";
  // convertToCamelFromSnakeCase will not capitalize "INTEL".
  if (attrName == "cache_control_load_intel")
    return "CacheControlLoadINTEL";
  if (attrName == "cache_control_store_intel")
    return "CacheControlStoreINTEL";

  return llvm::convertToCamelFromSnakeCase(attrName, /*capitalizeFirst=*/true);
}

template <typename AttrTy, typename EmitF>
LogicalResult processDecorationList(Location loc, Decoration decoration,
                                    Attribute attrList, StringRef attrName,
                                    EmitF emitter) {
  auto arrayAttr = dyn_cast<ArrayAttr>(attrList);
  if (!arrayAttr) {
    return emitError(loc, "expecting array attribute of ")
           << attrName << " for " << stringifyDecoration(decoration);
  }
  if (arrayAttr.empty()) {
    return emitError(loc, "expecting non-empty array attribute of ")
           << attrName << " for " << stringifyDecoration(decoration);
  }
  for (Attribute attr : arrayAttr.getValue()) {
    auto cacheControlAttr = dyn_cast<AttrTy>(attr);
    if (!cacheControlAttr) {
      return emitError(loc, "expecting array attribute of ")
             << attrName << " for " << stringifyDecoration(decoration);
    }
    // This named attribute encodes several decorations. Emit one per
    // element in the array.
    if (failed(emitter(cacheControlAttr)))
      return failure();
  }
  return success();
}

LogicalResult Serializer::processDecorationAttr(Location loc, uint32_t resultID,
                                                Decoration decoration,
                                                Attribute attr) {
  SmallVector<uint32_t, 1> args;
  switch (decoration) {
  case spirv::Decoration::LinkageAttributes: {
    // Get the value of the Linkage Attributes
    // e.g., LinkageAttributes=["linkageName", linkageType].
    auto linkageAttr = llvm::dyn_cast<spirv::LinkageAttributesAttr>(attr);
    auto linkageName = linkageAttr.getLinkageName();
    auto linkageType = linkageAttr.getLinkageType().getValue();
    // Encode the Linkage Name (string literal to uint32_t).
    spirv::encodeStringLiteralInto(args, linkageName);
    // Encode LinkageType & Add the Linkagetype to the args.
    args.push_back(static_cast<uint32_t>(linkageType));
    break;
  }
  case spirv::Decoration::FPFastMathMode:
    if (auto intAttr = dyn_cast<FPFastMathModeAttr>(attr)) {
      args.push_back(static_cast<uint32_t>(intAttr.getValue()));
      break;
    }
    return emitError(loc, "expected FPFastMathModeAttr attribute for ")
           << stringifyDecoration(decoration);
  case spirv::Decoration::FPRoundingMode:
    if (auto intAttr = dyn_cast<FPRoundingModeAttr>(attr)) {
      args.push_back(static_cast<uint32_t>(intAttr.getValue()));
      break;
    }
    return emitError(loc, "expected FPRoundingModeAttr attribute for ")
           << stringifyDecoration(decoration);
  case spirv::Decoration::Binding:
  case spirv::Decoration::DescriptorSet:
  case spirv::Decoration::Location:
    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      args.push_back(intAttr.getValue().getZExtValue());
      break;
    }
    return emitError(loc, "expected integer attribute for ")
           << stringifyDecoration(decoration);
  case spirv::Decoration::BuiltIn:
    if (auto strAttr = dyn_cast<StringAttr>(attr)) {
      auto enumVal = spirv::symbolizeBuiltIn(strAttr.getValue());
      if (enumVal) {
        args.push_back(static_cast<uint32_t>(*enumVal));
        break;
      }
      return emitError(loc, "invalid ")
             << stringifyDecoration(decoration) << " decoration attribute "
             << strAttr.getValue();
    }
    return emitError(loc, "expected string attribute for ")
           << stringifyDecoration(decoration);
  case spirv::Decoration::Aliased:
  case spirv::Decoration::AliasedPointer:
  case spirv::Decoration::Flat:
  case spirv::Decoration::NonReadable:
  case spirv::Decoration::NonWritable:
  case spirv::Decoration::NoPerspective:
  case spirv::Decoration::NoSignedWrap:
  case spirv::Decoration::NoUnsignedWrap:
  case spirv::Decoration::RelaxedPrecision:
  case spirv::Decoration::Restrict:
  case spirv::Decoration::RestrictPointer:
  case spirv::Decoration::NoContraction:
  case spirv::Decoration::Constant:
  case spirv::Decoration::Block:
  case spirv::Decoration::Invariant:
  case spirv::Decoration::Patch:
    // For unit attributes and decoration attributes, the args list
    // has no values so we do nothing.
    if (isa<UnitAttr, DecorationAttr>(attr))
      break;
    return emitError(loc,
                     "expected unit attribute or decoration attribute for ")
           << stringifyDecoration(decoration);
  case spirv::Decoration::CacheControlLoadINTEL:
    return processDecorationList<CacheControlLoadINTELAttr>(
        loc, decoration, attr, "CacheControlLoadINTEL",
        [&](CacheControlLoadINTELAttr attr) {
          unsigned cacheLevel = attr.getCacheLevel();
          LoadCacheControl loadCacheControl = attr.getLoadCacheControl();
          return emitDecoration(
              resultID, decoration,
              {cacheLevel, static_cast<uint32_t>(loadCacheControl)});
        });
  case spirv::Decoration::CacheControlStoreINTEL:
    return processDecorationList<CacheControlStoreINTELAttr>(
        loc, decoration, attr, "CacheControlStoreINTEL",
        [&](CacheControlStoreINTELAttr attr) {
          unsigned cacheLevel = attr.getCacheLevel();
          StoreCacheControl storeCacheControl = attr.getStoreCacheControl();
          return emitDecoration(
              resultID, decoration,
              {cacheLevel, static_cast<uint32_t>(storeCacheControl)});
        });
  default:
    return emitError(loc, "unhandled decoration ")
           << stringifyDecoration(decoration);
  }
  return emitDecoration(resultID, decoration, args);
}

LogicalResult Serializer::processDecoration(Location loc, uint32_t resultID,
                                            NamedAttribute attr) {
  StringRef attrName = attr.getName().strref();
  std::string decorationName = getDecorationName(attrName);
  std::optional<Decoration> decoration =
      spirv::symbolizeDecoration(decorationName);
  if (!decoration) {
    return emitError(
               loc, "non-argument attributes expected to have snake-case-ified "
                    "decoration name, unhandled attribute with name : ")
           << attrName;
  }
  return processDecorationAttr(loc, resultID, *decoration, attr.getValue());
}

LogicalResult Serializer::processName(uint32_t resultID, StringRef name) {
  assert(!name.empty() && "unexpected empty string for OpName");
  if (!options.emitSymbolName)
    return success();

  SmallVector<uint32_t, 4> nameOperands;
  nameOperands.push_back(resultID);
  spirv::encodeStringLiteralInto(nameOperands, name);
  encodeInstructionInto(names, spirv::Opcode::OpName, nameOperands);
  return success();
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
  if (memberDecoration.hasValue()) {
    args.push_back(
        cast<IntegerAttr>(memberDecoration.decorationValue).getInt());
  }
  encodeInstructionInto(decorations, spirv::Opcode::OpMemberDecorate, args);
  return success();
}

//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

// According to the SPIR-V spec "Validation Rules for Shader Capabilities":
// "Composite objects in the StorageBuffer, PhysicalStorageBuffer, Uniform, and
// PushConstant Storage Classes must be explicitly laid out."
bool Serializer::isInterfaceStructPtrType(Type type) const {
  if (auto ptrType = dyn_cast<spirv::PointerType>(type)) {
    switch (ptrType.getStorageClass()) {
    case spirv::StorageClass::PhysicalStorageBuffer:
    case spirv::StorageClass::PushConstant:
    case spirv::StorageClass::StorageBuffer:
    case spirv::StorageClass::Uniform:
      return isa<spirv::StructType>(ptrType.getPointeeType());
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

  // Map unsigned integer types to singless integer types.
  // This is needed otherwise the generated spirv assembly will contain
  // twice a type declaration (like OpTypeInt 32 0) which is no permitted and
  // such module fails validation. Indeed at MLIR level the two types are
  // different and lookup in the cache below misses.
  // Note: This conversion needs to happen here before the type is looked up in
  // the cache.
  if (type.isUnsignedInteger()) {
    type = IntegerType::get(loc->getContext(), type.getIntOrFloatBitWidth(),
                            IntegerType::SignednessSemantics::Signless);
  }

  typeID = getTypeID(type);
  if (typeID)
    return success();

  typeID = getNextID();
  SmallVector<uint32_t, 4> operands;

  operands.push_back(typeID);
  auto typeEnum = spirv::Opcode::OpTypeVoid;
  bool deferSerialization = false;

  if ((isa<FunctionType>(type) &&
       succeeded(prepareFunctionType(loc, cast<FunctionType>(type), typeEnum,
                                     operands))) ||
      (isa<GraphType>(type) &&
       succeeded(
           prepareGraphType(loc, cast<GraphType>(type), typeEnum, operands))) ||
      succeeded(prepareBasicType(loc, type, typeID, typeEnum, operands,
                                 deferSerialization, serializationCtx))) {
    if (deferSerialization)
      return success();

    typeIDMap[type] = typeID;

    encodeInstructionInto(typesGlobalValues, typeEnum, operands);

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

        encodeInstructionInto(typesGlobalValues, spirv::Opcode::OpTypePointer,
                              ptrOperands);
      }

      recursiveStructInfos[type].clear();
    }

    return success();
  }

  return emitError(loc, "failed to process type: ") << type;
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

  if (auto intType = dyn_cast<IntegerType>(type)) {
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

  if (auto floatType = dyn_cast<FloatType>(type)) {
    typeEnum = spirv::Opcode::OpTypeFloat;
    operands.push_back(floatType.getWidth());
    if (floatType.isBF16()) {
      operands.push_back(static_cast<uint32_t>(spirv::FPEncoding::BFloat16KHR));
    }
    return success();
  }

  if (auto vectorType = dyn_cast<VectorType>(type)) {
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

  if (auto imageType = dyn_cast<spirv::ImageType>(type)) {
    typeEnum = spirv::Opcode::OpTypeImage;
    uint32_t sampledTypeID = 0;
    if (failed(processType(loc, imageType.getElementType(), sampledTypeID)))
      return failure();

    llvm::append_values(operands, sampledTypeID,
                        static_cast<uint32_t>(imageType.getDim()),
                        static_cast<uint32_t>(imageType.getDepthInfo()),
                        static_cast<uint32_t>(imageType.getArrayedInfo()),
                        static_cast<uint32_t>(imageType.getSamplingInfo()),
                        static_cast<uint32_t>(imageType.getSamplerUseInfo()),
                        static_cast<uint32_t>(imageType.getImageFormat()));
    return success();
  }

  if (auto arrayType = dyn_cast<spirv::ArrayType>(type)) {
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

  if (auto ptrType = dyn_cast<spirv::PointerType>(type)) {
    uint32_t pointeeTypeID = 0;
    spirv::StructType pointeeStruct =
        dyn_cast<spirv::StructType>(ptrType.getPointeeType());

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

      encodeInstructionInto(typesGlobalValues,
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

    // TODO: Now struct decorations are supported this code may not be
    // necessary. However, it is left to support backwards compatibility.
    // Ideally, Block decorations should be inserted when converting to SPIR-V.
    if (isInterfaceStructPtrType(ptrType)) {
      auto structType = cast<spirv::StructType>(ptrType.getPointeeType());
      if (!structType.hasDecoration(spirv::Decoration::Block))
        if (failed(emitDecoration(getTypeID(pointeeStruct),
                                  spirv::Decoration::Block)))
          return emitError(loc, "cannot decorate ")
                 << pointeeStruct << " with Block decoration";
    }

    return success();
  }

  if (auto runtimeArrayType = dyn_cast<spirv::RuntimeArrayType>(type)) {
    uint32_t elementTypeID = 0;
    if (failed(processTypeImpl(loc, runtimeArrayType.getElementType(),
                               elementTypeID, serializationCtx))) {
      return failure();
    }
    typeEnum = spirv::Opcode::OpTypeRuntimeArray;
    operands.push_back(elementTypeID);
    return processTypeDecoration(loc, runtimeArrayType, resultID);
  }

  if (auto sampledImageType = dyn_cast<spirv::SampledImageType>(type)) {
    typeEnum = spirv::Opcode::OpTypeSampledImage;
    uint32_t imageTypeID = 0;
    if (failed(
            processType(loc, sampledImageType.getImageType(), imageTypeID))) {
      return failure();
    }
    operands.push_back(imageTypeID);
    return success();
  }

  if (auto structType = dyn_cast<spirv::StructType>(type)) {
    if (structType.isIdentified()) {
      if (failed(processName(resultID, structType.getIdentifier())))
        return failure();
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
        auto intType = IntegerType::get(structType.getContext(), 32);
        // Decorate each struct member with an offset
        spirv::StructType::MemberDecorationInfo offsetDecoration{
            elementIndex, spirv::Decoration::Offset,
            IntegerAttr::get(intType,
                             structType.getMemberOffset(elementIndex))};
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

    SmallVector<spirv::StructType::StructDecorationInfo, 1> structDecorations;
    structType.getStructDecorations(structDecorations);

    for (spirv::StructType::StructDecorationInfo &structDecoration :
         structDecorations) {
      if (failed(processDecorationAttr(loc, resultID,
                                       structDecoration.decoration,
                                       structDecoration.decorationValue))) {
        return emitError(loc, "cannot decorate struct ")
               << structType << " with "
               << stringifyDecoration(structDecoration.decoration);
      }
    }

    typeEnum = spirv::Opcode::OpTypeStruct;

    if (structType.isIdentified())
      serializationCtx.remove(structType.getIdentifier());

    return success();
  }

  if (auto cooperativeMatrixType =
          dyn_cast<spirv::CooperativeMatrixType>(type)) {
    uint32_t elementTypeID = 0;
    if (failed(processTypeImpl(loc, cooperativeMatrixType.getElementType(),
                               elementTypeID, serializationCtx))) {
      return failure();
    }
    typeEnum = spirv::Opcode::OpTypeCooperativeMatrixKHR;
    auto getConstantOp = [&](uint32_t id) {
      auto attr = IntegerAttr::get(IntegerType::get(type.getContext(), 32), id);
      return prepareConstantInt(loc, attr);
    };
    llvm::append_values(
        operands, elementTypeID,
        getConstantOp(static_cast<uint32_t>(cooperativeMatrixType.getScope())),
        getConstantOp(cooperativeMatrixType.getRows()),
        getConstantOp(cooperativeMatrixType.getColumns()),
        getConstantOp(static_cast<uint32_t>(cooperativeMatrixType.getUse())));
    return success();
  }

  if (auto matrixType = dyn_cast<spirv::MatrixType>(type)) {
    uint32_t elementTypeID = 0;
    if (failed(processTypeImpl(loc, matrixType.getColumnType(), elementTypeID,
                               serializationCtx))) {
      return failure();
    }
    typeEnum = spirv::Opcode::OpTypeMatrix;
    llvm::append_values(operands, elementTypeID, matrixType.getNumColumns());
    return success();
  }

  if (auto tensorArmType = llvm::dyn_cast<TensorArmType>(type)) {
    uint32_t elementTypeID = 0;
    uint32_t rank = 0;
    uint32_t shapeID = 0;
    uint32_t rankID = 0;
    if (failed(processTypeImpl(loc, tensorArmType.getElementType(),
                               elementTypeID, serializationCtx))) {
      return failure();
    }
    if (tensorArmType.hasRank()) {
      ArrayRef<int64_t> dims = tensorArmType.getShape();
      rank = dims.size();
      rankID = prepareConstantInt(loc, mlirBuilder.getI32IntegerAttr(rank));
      if (rankID == 0) {
        return failure();
      }

      bool shaped = llvm::all_of(dims, [](const auto &dim) { return dim > 0; });
      if (rank > 0 && shaped) {
        auto I32Type = IntegerType::get(type.getContext(), 32);
        auto shapeType = ArrayType::get(I32Type, rank);
        if (rank == 1) {
          SmallVector<uint64_t, 1> index(rank);
          shapeID = prepareDenseElementsConstant(
              loc, shapeType,
              mlirBuilder.getI32TensorAttr(SmallVector<int32_t>(dims)), 0,
              index);
        } else {
          shapeID = prepareArrayConstant(
              loc, shapeType,
              mlirBuilder.getI32ArrayAttr(SmallVector<int32_t>(dims)));
        }
        if (shapeID == 0) {
          return failure();
        }
      }
    }
    typeEnum = spirv::Opcode::OpTypeTensorARM;
    operands.push_back(elementTypeID);
    if (rankID == 0)
      return success();
    operands.push_back(rankID);
    if (shapeID == 0)
      return success();
    operands.push_back(shapeID);
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

LogicalResult
Serializer::prepareGraphType(Location loc, GraphType type,
                             spirv::Opcode &typeEnum,
                             SmallVectorImpl<uint32_t> &operands) {
  typeEnum = spirv::Opcode::OpTypeGraphARM;
  assert(type.getNumResults() >= 1 &&
         "serialization requires at least a return value");

  operands.push_back(type.getNumInputs());

  for (Type argType : type.getInputs()) {
    uint32_t argTypeID = 0;
    if (failed(processType(loc, argType, argTypeID)))
      return failure();
    operands.push_back(argTypeID);
  }

  for (Type resType : type.getResults()) {
    uint32_t resTypeID = 0;
    if (failed(processType(loc, resType, resTypeID)))
      return failure();
    operands.push_back(resTypeID);
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
  if (auto attr = dyn_cast<DenseElementsAttr>(valueAttr)) {
    int rank = dyn_cast<ShapedType>(attr.getType()).getRank();
    SmallVector<uint64_t, 4> index(rank);
    resultID = prepareDenseElementsConstant(loc, constType, attr,
                                            /*dim=*/0, index);
  } else if (auto arrayAttr = dyn_cast<ArrayAttr>(valueAttr)) {
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
  auto elementType = cast<spirv::ArrayType>(constType).getElementType();
  for (Attribute elementAttr : attr) {
    if (auto elementID = prepareConstant(loc, elementType, elementAttr)) {
      operands.push_back(elementID);
    } else {
      return 0;
    }
  }
  spirv::Opcode opcode = spirv::Opcode::OpConstantComposite;
  encodeInstructionInto(typesGlobalValues, opcode, operands);

  return resultID;
}

// TODO: Turn the below function into iterative function, instead of
// recursive function.
uint32_t
Serializer::prepareDenseElementsConstant(Location loc, Type constType,
                                         DenseElementsAttr valueAttr, int dim,
                                         MutableArrayRef<uint64_t> index) {
  auto shapedType = dyn_cast<ShapedType>(valueAttr.getType());
  assert(dim <= shapedType.getRank());
  if (shapedType.getRank() == dim) {
    if (auto attr = dyn_cast<DenseIntElementsAttr>(valueAttr)) {
      return attr.getType().getElementType().isInteger(1)
                 ? prepareConstantBool(loc, attr.getValues<BoolAttr>()[index])
                 : prepareConstantInt(loc,
                                      attr.getValues<IntegerAttr>()[index]);
    }
    if (auto attr = dyn_cast<DenseFPElementsAttr>(valueAttr)) {
      return prepareConstantFp(loc, attr.getValues<FloatAttr>()[index]);
    }
    return 0;
  }

  uint32_t typeID = 0;
  if (failed(processType(loc, constType, typeID))) {
    return 0;
  }

  int64_t numberOfConstituents = shapedType.getDimSize(dim);
  uint32_t resultID = getNextID();
  SmallVector<uint32_t, 4> operands = {typeID, resultID};
  auto elementType = cast<spirv::CompositeType>(constType).getElementType(0);
  if (auto tensorArmType = dyn_cast<spirv::TensorArmType>(constType)) {
    ArrayRef<int64_t> innerShape = tensorArmType.getShape().drop_front();
    if (!innerShape.empty())
      elementType = spirv::TensorArmType::get(innerShape, elementType);
  }

  // "If the Result Type is a cooperative matrix type, then there must be only
  // one Constituent, with scalar type matching the cooperative matrix Component
  // Type, and all components of the matrix are initialized to that value."
  // (https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_cooperative_matrix.html)
  if (isa<spirv::CooperativeMatrixType>(constType)) {
    if (!valueAttr.isSplat()) {
      emitError(
          loc,
          "cannot serialize a non-splat value for a cooperative matrix type");
      return 0;
    }
    // numberOfConstituents is 1, so we only need one more elements in the
    // SmallVector, so the total is 3 (1 + 2).
    operands.reserve(3);
    // We set dim directly to `shapedType.getRank()` so the recursive call
    // directly returns the scalar type.
    if (auto elementID = prepareDenseElementsConstant(
            loc, elementType, valueAttr, /*dim=*/shapedType.getRank(), index)) {
      operands.push_back(elementID);
    } else {
      return 0;
    }
  } else if (isa<spirv::TensorArmType>(constType) && isZeroValue(valueAttr)) {
    encodeInstructionInto(typesGlobalValues, spirv::Opcode::OpConstantNull,
                          {typeID, resultID});
    return resultID;
  } else {
    operands.reserve(numberOfConstituents + 2);
    for (int i = 0; i < numberOfConstituents; ++i) {
      index[dim] = i;
      if (auto elementID = prepareDenseElementsConstant(
              loc, elementType, valueAttr, dim + 1, index)) {
        operands.push_back(elementID);
      } else {
        return 0;
      }
    }
  }
  spirv::Opcode opcode = spirv::Opcode::OpConstantComposite;
  encodeInstructionInto(typesGlobalValues, opcode, operands);

  return resultID;
}

uint32_t Serializer::prepareConstantScalar(Location loc, Attribute valueAttr,
                                           bool isSpec) {
  if (auto floatAttr = dyn_cast<FloatAttr>(valueAttr)) {
    return prepareConstantFp(loc, floatAttr, isSpec);
  }
  if (auto boolAttr = dyn_cast<BoolAttr>(valueAttr)) {
    return prepareConstantBool(loc, boolAttr, isSpec);
  }
  if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
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
  if (failed(processType(loc, cast<IntegerAttr>(boolAttr).getType(), typeID))) {
    return 0;
  }

  auto resultID = getNextID();
  auto opcode = boolAttr.getValue()
                    ? (isSpec ? spirv::Opcode::OpSpecConstantTrue
                              : spirv::Opcode::OpConstantTrue)
                    : (isSpec ? spirv::Opcode::OpSpecConstantFalse
                              : spirv::Opcode::OpConstantFalse);
  encodeInstructionInto(typesGlobalValues, opcode, {typeID, resultID});

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
  bool isSigned = intAttr.getType().isSignedInteger();
  auto opcode =
      isSpec ? spirv::Opcode::OpSpecConstant : spirv::Opcode::OpConstant;

  switch (bitwidth) {
    // According to SPIR-V spec, "When the type's bit width is less than
    // 32-bits, the literal's value appears in the low-order bits of the word,
    // and the high-order bits must be 0 for a floating-point type, or 0 for an
    // integer type with Signedness of 0, or sign extended when Signedness
    // is 1."
  case 32:
  case 16:
  case 8: {
    uint32_t word = 0;
    if (isSigned) {
      word = static_cast<int32_t>(value.getSExtValue());
    } else {
      word = static_cast<uint32_t>(value.getZExtValue());
    }
    encodeInstructionInto(typesGlobalValues, opcode, {typeID, resultID, word});
  } break;
    // According to SPIR-V spec: "When the type's bit width is larger than one
    // word, the literal’s low-order words appear first."
  case 64: {
    struct DoubleWord {
      uint32_t word1;
      uint32_t word2;
    } words;
    if (isSigned) {
      words = llvm::bit_cast<DoubleWord>(value.getSExtValue());
    } else {
      words = llvm::bit_cast<DoubleWord>(value.getZExtValue());
    }
    encodeInstructionInto(typesGlobalValues, opcode,
                          {typeID, resultID, words.word1, words.word2});
  } break;
  default: {
    std::string valueStr;
    llvm::raw_string_ostream rss(valueStr);
    value.print(rss, /*isSigned=*/false);

    emitError(loc, "cannot serialize ")
        << bitwidth << "-bit integer literal: " << valueStr;
    return 0;
  }
  }

  if (!isSpec) {
    constIDMap[intAttr] = resultID;
  }
  return resultID;
}

uint32_t Serializer::prepareGraphConstantId(Location loc, Type graphConstType,
                                            IntegerAttr intAttr) {
  // De-duplicate graph constants.
  if (uint32_t id = getGraphConstantARMId(intAttr)) {
    return id;
  }

  // Process the type for this graph constant.
  uint32_t typeID = 0;
  if (failed(processType(loc, graphConstType, typeID))) {
    return 0;
  }

  uint32_t resultID = getNextID();
  APInt value = intAttr.getValue();
  unsigned bitwidth = value.getBitWidth();
  if (bitwidth > 32) {
    emitError(loc, "Too wide attribute for OpGraphConstantARM: ")
        << bitwidth << " bits";
    return 0;
  }
  bool isSigned = value.isSignedIntN(bitwidth);

  uint32_t word = 0;
  if (isSigned) {
    word = static_cast<int32_t>(value.getSExtValue());
  } else {
    word = static_cast<uint32_t>(value.getZExtValue());
  }
  encodeInstructionInto(typesGlobalValues, spirv::Opcode::OpGraphConstantARM,
                        {typeID, resultID, word});
  graphConstIDMap[intAttr] = resultID;
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
  const llvm::fltSemantics *semantics = &value.getSemantics();

  auto opcode =
      isSpec ? spirv::Opcode::OpSpecConstant : spirv::Opcode::OpConstant;

  if (semantics == &APFloat::IEEEsingle()) {
    uint32_t word = llvm::bit_cast<uint32_t>(value.convertToFloat());
    encodeInstructionInto(typesGlobalValues, opcode, {typeID, resultID, word});
  } else if (semantics == &APFloat::IEEEdouble()) {
    struct DoubleWord {
      uint32_t word1;
      uint32_t word2;
    } words = llvm::bit_cast<DoubleWord>(value.convertToDouble());
    encodeInstructionInto(typesGlobalValues, opcode,
                          {typeID, resultID, words.word1, words.word2});
  } else if (semantics == &APFloat::IEEEhalf() ||
             semantics == &APFloat::BFloat()) {
    uint32_t word =
        static_cast<uint32_t>(value.bitcastToAPInt().getZExtValue());
    encodeInstructionInto(typesGlobalValues, opcode, {typeID, resultID, word});
  } else {
    std::string valueStr;
    llvm::raw_string_ostream rss(valueStr);
    value.print(rss);

    emitError(loc, "cannot serialize ")
        << floatAttr.getType() << "-typed float literal: " << valueStr;
    return 0;
  }

  if (!isSpec) {
    constIDMap[floatAttr] = resultID;
  }
  return resultID;
}

// Returns type of attribute. In case of a TypedAttr this will simply return
// the type. But for an ArrayAttr which is untyped and can be multidimensional
// it creates the ArrayType recursively.
static Type getValueType(Attribute attr) {
  if (auto typedAttr = dyn_cast<TypedAttr>(attr)) {
    return typedAttr.getType();
  }

  if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
    return spirv::ArrayType::get(getValueType(arrayAttr[0]), arrayAttr.size());
  }

  return nullptr;
}

uint32_t Serializer::prepareConstantCompositeReplicate(Location loc,
                                                       Type resultType,
                                                       Attribute valueAttr) {
  std::pair<Attribute, Type> valueTypePair{valueAttr, resultType};
  if (uint32_t id = getConstantCompositeReplicateID(valueTypePair)) {
    return id;
  }

  uint32_t typeID = 0;
  if (failed(processType(loc, resultType, typeID))) {
    return 0;
  }

  Type valueType = getValueType(valueAttr);
  if (!valueAttr)
    return 0;

  auto compositeType = dyn_cast<CompositeType>(resultType);
  if (!compositeType)
    return 0;
  Type elementType = compositeType.getElementType(0);

  uint32_t constandID;
  if (elementType == valueType) {
    constandID = prepareConstant(loc, elementType, valueAttr);
  } else {
    constandID = prepareConstantCompositeReplicate(loc, elementType, valueAttr);
  }

  uint32_t resultID = getNextID();
  if (dyn_cast<spirv::TensorArmType>(resultType) && isZeroValue(valueAttr)) {
    encodeInstructionInto(typesGlobalValues, spirv::Opcode::OpConstantNull,
                          {typeID, resultID});
  } else {
    encodeInstructionInto(typesGlobalValues,
                          spirv::Opcode::OpConstantCompositeReplicateEXT,
                          {typeID, resultID, constandID});
  }

  constCompositeReplicateIDMap[valueTypePair] = resultID;
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

#ifndef NDEBUG
void Serializer::printBlock(Block *block, raw_ostream &os) {
  os << "block " << block << " (id = ";
  if (uint32_t id = getBlockID(block))
    os << id;
  else
    os << "unknown";
  os << ")\n";
}
#endif

LogicalResult
Serializer::processBlock(Block *block, bool omitLabel,
                         function_ref<LogicalResult()> emitMerge) {
  LLVM_DEBUG(llvm::dbgs() << "processing block " << block << ":\n");
  LLVM_DEBUG(block->print(llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << '\n');
  if (!omitLabel) {
    uint32_t blockID = getOrCreateBlockID(block);
    LLVM_DEBUG(printBlock(block, llvm::dbgs()));

    // Emit OpLabel for this block.
    encodeInstructionInto(functionBody, spirv::Opcode::OpLabel, {blockID});
  }

  // Emit OpPhi instructions for block arguments, if any.
  if (failed(emitPhiForBlockArguments(block)))
    return failure();

  // If we need to emit merge instructions, it must happen in this block. Check
  // whether we have other structured control flow ops, which will be expanded
  // into multiple basic blocks. If that's the case, we need to emit the merge
  // right now and then create new blocks for further serialization of the ops
  // in this block.
  if (emitMerge &&
      llvm::any_of(block->getOperations(),
                   llvm::IsaPred<spirv::LoopOp, spirv::SelectionOp>)) {
    if (failed(emitMerge()))
      return failure();
    emitMerge = nullptr;

    // Start a new block for further serialization.
    uint32_t blockID = getNextID();
    encodeInstructionInto(functionBody, spirv::Opcode::OpBranch, {blockID});
    encodeInstructionInto(functionBody, spirv::Opcode::OpLabel, {blockID});
  }

  // Process each op in this block except the terminator.
  for (Operation &op : llvm::drop_end(*block)) {
    if (failed(processOperation(&op)))
      return failure();
  }

  // Process the terminator.
  if (emitMerge)
    if (failed(emitMerge()))
      return failure();
  if (failed(processOperation(&block->back())))
    return failure();

  return success();
}

LogicalResult Serializer::emitPhiForBlockArguments(Block *block) {
  // Nothing to do if this block has no arguments or it's the entry block, which
  // always has the same arguments as the function signature.
  if (block->args_empty() || block->isEntryBlock())
    return success();

  LLVM_DEBUG(llvm::dbgs() << "emitting phi instructions..\n");

  // If the block has arguments, we need to create SPIR-V OpPhi instructions.
  // A SPIR-V OpPhi instruction is of the syntax:
  //   OpPhi | result type | result <id> | (value <id>, parent block <id>) pair
  // So we need to collect all predecessor blocks and the arguments they send
  // to this block.
  SmallVector<std::pair<Block *, OperandRange>, 4> predecessors;
  for (Block *mlirPredecessor : block->getPredecessors()) {
    auto *terminator = mlirPredecessor->getTerminator();
    LLVM_DEBUG(llvm::dbgs() << "  mlir predecessor ");
    LLVM_DEBUG(printBlock(mlirPredecessor, llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "    terminator: " << *terminator << "\n");
    // The predecessor here is the immediate one according to MLIR's IR
    // structure. It does not directly map to the incoming parent block for the
    // OpPhi instructions at SPIR-V binary level. This is because structured
    // control flow ops are serialized to multiple SPIR-V blocks. If there is a
    // spirv.mlir.selection/spirv.mlir.loop op in the MLIR predecessor block,
    // the branch op jumping to the OpPhi's block then resides in the previous
    // structured control flow op's merge block.
    Block *spirvPredecessor = getPhiIncomingBlock(mlirPredecessor);
    LLVM_DEBUG(llvm::dbgs() << "  spirv predecessor ");
    LLVM_DEBUG(printBlock(spirvPredecessor, llvm::dbgs()));
    if (auto branchOp = dyn_cast<spirv::BranchOp>(terminator)) {
      predecessors.emplace_back(spirvPredecessor, branchOp.getOperands());
    } else if (auto branchCondOp =
                   dyn_cast<spirv::BranchConditionalOp>(terminator)) {
      std::optional<OperandRange> blockOperands;
      if (branchCondOp.getTrueTarget() == block) {
        blockOperands = branchCondOp.getTrueTargetOperands();
      } else {
        assert(branchCondOp.getFalseTarget() == block);
        blockOperands = branchCondOp.getFalseTargetOperands();
      }

      assert(!blockOperands->empty() &&
             "expected non-empty block operand range");
      predecessors.emplace_back(spirvPredecessor, *blockOperands);
    } else {
      return terminator->emitError("unimplemented terminator for Phi creation");
    }
    LLVM_DEBUG({
      llvm::dbgs() << "    block arguments:\n";
      for (Value v : predecessors.back().second)
        llvm::dbgs() << "      " << v << "\n";
    });
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

    encodeInstructionInto(functionBody, spirv::Opcode::OpPhi, phiArgs);
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
    spirv::encodeStringLiteralInto(importOperands, extensionSetName);
    encodeInstructionInto(extendedSets, spirv::Opcode::OpExtInstImport,
                          importOperands);
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
  encodeInstructionInto(functionBody, spirv::Opcode::OpExtInst,
                        extInstOperands);
  return success();
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
      .Case([&](spirv::EXTConstantCompositeReplicateOp op) {
        return processConstantCompositeReplicateOp(op);
      })
      .Case([&](spirv::FuncOp op) { return processFuncOp(op); })
      .Case([&](spirv::GraphARMOp op) { return processGraphARMOp(op); })
      .Case([&](spirv::GraphEntryPointARMOp op) {
        return processGraphEntryPointARMOp(op);
      })
      .Case([&](spirv::GraphOutputsARMOp op) {
        return processGraphOutputsARMOp(op);
      })
      .Case([&](spirv::GlobalVariableOp op) {
        return processGlobalVariableOp(op);
      })
      .Case([&](spirv::GraphConstantARMOp op) {
        return processGraphConstantARMOp(op);
      })
      .Case([&](spirv::LoopOp op) { return processLoopOp(op); })
      .Case([&](spirv::ReferenceOfOp op) { return processReferenceOfOp(op); })
      .Case([&](spirv::SelectionOp op) { return processSelectionOp(op); })
      .Case([&](spirv::SpecConstantOp op) { return processSpecConstantOp(op); })
      .Case([&](spirv::SpecConstantCompositeOp op) {
        return processSpecConstantCompositeOp(op);
      })
      .Case([&](spirv::EXTSpecConstantCompositeReplicateOp op) {
        return processSpecConstantCompositeReplicateOp(op);
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

  if (failed(emitDebugLine(functionBody, loc)))
    return failure();

  if (extInstSet.empty()) {
    encodeInstructionInto(functionBody, static_cast<spirv::Opcode>(opcode),
                          operands);
  } else {
    if (failed(encodeExtensionInstruction(op, extInstSet, opcode, operands)))
      return failure();
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
  llvm::append_values(
      decorations,
      spirv::getPrefixedOpcode(wordCount, spirv::Opcode::OpDecorate), target,
      static_cast<uint32_t>(decoration));
  llvm::append_range(decorations, params);
  return success();
}

LogicalResult Serializer::emitDebugLine(SmallVectorImpl<uint32_t> &binary,
                                        Location loc) {
  if (!options.emitDebugInfo)
    return success();

  if (lastProcessedWasMergeInst) {
    lastProcessedWasMergeInst = false;
    return success();
  }

  auto fileLoc = dyn_cast<FileLineColLoc>(loc);
  if (fileLoc)
    encodeInstructionInto(binary, spirv::Opcode::OpLine,
                          {fileID, fileLoc.getLine(), fileLoc.getColumn()});
  return success();
}
} // namespace spirv
} // namespace mlir
