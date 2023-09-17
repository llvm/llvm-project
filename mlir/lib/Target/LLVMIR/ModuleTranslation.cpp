//===- ModuleTranslation.cpp - MLIR to LLVM conversion --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the translation between an MLIR LLVM dialect module and
// the corresponding LLVMIR module. It only handles core LLVM IR operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "AttrKindDetail.h"
#include "DebugTranslation.h"
#include "LoopAnnotationTranslation.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMInterfaces.h"
#include "mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <optional>

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::LLVM::detail;

#include "mlir/Dialect/LLVMIR/LLVMConversionEnumsToLLVM.inc"

/// Translates the given data layout spec attribute to the LLVM IR data layout.
/// Only integer, float, pointer and endianness entries are currently supported.
static FailureOr<llvm::DataLayout>
translateDataLayout(DataLayoutSpecInterface attribute,
                    const DataLayout &dataLayout,
                    std::optional<Location> loc = std::nullopt) {
  if (!loc)
    loc = UnknownLoc::get(attribute.getContext());

  // Translate the endianness attribute.
  std::string llvmDataLayout;
  llvm::raw_string_ostream layoutStream(llvmDataLayout);
  for (DataLayoutEntryInterface entry : attribute.getEntries()) {
    auto key = llvm::dyn_cast_if_present<StringAttr>(entry.getKey());
    if (!key)
      continue;
    if (key.getValue() == DLTIDialect::kDataLayoutEndiannessKey) {
      auto value = cast<StringAttr>(entry.getValue());
      bool isLittleEndian =
          value.getValue() == DLTIDialect::kDataLayoutEndiannessLittle;
      layoutStream << "-" << (isLittleEndian ? "e" : "E");
      layoutStream.flush();
      continue;
    }
    if (key.getValue() == DLTIDialect::kDataLayoutAllocaMemorySpaceKey) {
      auto value = cast<IntegerAttr>(entry.getValue());
      uint64_t space = value.getValue().getZExtValue();
      // Skip the default address space.
      if (space == 0)
        continue;
      layoutStream << "-A" << space;
      layoutStream.flush();
      continue;
    }
    if (key.getValue() == DLTIDialect::kDataLayoutStackAlignmentKey) {
      auto value = cast<IntegerAttr>(entry.getValue());
      uint64_t alignment = value.getValue().getZExtValue();
      // Skip the default stack alignment.
      if (alignment == 0)
        continue;
      layoutStream << "-S" << alignment;
      layoutStream.flush();
      continue;
    }
    emitError(*loc) << "unsupported data layout key " << key;
    return failure();
  }

  // Go through the list of entries to check which types are explicitly
  // specified in entries. Where possible, data layout queries are used instead
  // of directly inspecting the entries.
  for (DataLayoutEntryInterface entry : attribute.getEntries()) {
    auto type = llvm::dyn_cast_if_present<Type>(entry.getKey());
    if (!type)
      continue;
    // Data layout for the index type is irrelevant at this point.
    if (isa<IndexType>(type))
      continue;
    layoutStream << "-";
    LogicalResult result =
        llvm::TypeSwitch<Type, LogicalResult>(type)
            .Case<IntegerType, Float16Type, Float32Type, Float64Type,
                  Float80Type, Float128Type>([&](Type type) -> LogicalResult {
              if (auto intType = dyn_cast<IntegerType>(type)) {
                if (intType.getSignedness() != IntegerType::Signless)
                  return emitError(*loc)
                         << "unsupported data layout for non-signless integer "
                         << intType;
                layoutStream << "i";
              } else {
                layoutStream << "f";
              }
              unsigned size = dataLayout.getTypeSizeInBits(type);
              unsigned abi = dataLayout.getTypeABIAlignment(type) * 8u;
              unsigned preferred =
                  dataLayout.getTypePreferredAlignment(type) * 8u;
              layoutStream << size << ":" << abi;
              if (abi != preferred)
                layoutStream << ":" << preferred;
              return success();
            })
            .Case([&](LLVMPointerType ptrType) {
              layoutStream << "p" << ptrType.getAddressSpace() << ":";
              unsigned size = dataLayout.getTypeSizeInBits(type);
              unsigned abi = dataLayout.getTypeABIAlignment(type) * 8u;
              unsigned preferred =
                  dataLayout.getTypePreferredAlignment(type) * 8u;
              layoutStream << size << ":" << abi << ":" << preferred;
              if (std::optional<unsigned> index = extractPointerSpecValue(
                      entry.getValue(), PtrDLEntryPos::Index))
                layoutStream << ":" << *index;
              return success();
            })
            .Default([loc](Type type) {
              return emitError(*loc)
                     << "unsupported type in data layout: " << type;
            });
    if (failed(result))
      return failure();
  }
  layoutStream.flush();
  StringRef layoutSpec(llvmDataLayout);
  if (layoutSpec.startswith("-"))
    layoutSpec = layoutSpec.drop_front();

  return llvm::DataLayout(layoutSpec);
}

/// Builds a constant of a sequential LLVM type `type`, potentially containing
/// other sequential types recursively, from the individual constant values
/// provided in `constants`. `shape` contains the number of elements in nested
/// sequential types. Reports errors at `loc` and returns nullptr on error.
static llvm::Constant *
buildSequentialConstant(ArrayRef<llvm::Constant *> &constants,
                        ArrayRef<int64_t> shape, llvm::Type *type,
                        Location loc) {
  if (shape.empty()) {
    llvm::Constant *result = constants.front();
    constants = constants.drop_front();
    return result;
  }

  llvm::Type *elementType;
  if (auto *arrayTy = dyn_cast<llvm::ArrayType>(type)) {
    elementType = arrayTy->getElementType();
  } else if (auto *vectorTy = dyn_cast<llvm::VectorType>(type)) {
    elementType = vectorTy->getElementType();
  } else {
    emitError(loc) << "expected sequential LLVM types wrapping a scalar";
    return nullptr;
  }

  SmallVector<llvm::Constant *, 8> nested;
  nested.reserve(shape.front());
  for (int64_t i = 0; i < shape.front(); ++i) {
    nested.push_back(buildSequentialConstant(constants, shape.drop_front(),
                                             elementType, loc));
    if (!nested.back())
      return nullptr;
  }

  if (shape.size() == 1 && type->isVectorTy())
    return llvm::ConstantVector::get(nested);
  return llvm::ConstantArray::get(
      llvm::ArrayType::get(elementType, shape.front()), nested);
}

/// Returns the first non-sequential type nested in sequential types.
static llvm::Type *getInnermostElementType(llvm::Type *type) {
  do {
    if (auto *arrayTy = dyn_cast<llvm::ArrayType>(type)) {
      type = arrayTy->getElementType();
    } else if (auto *vectorTy = dyn_cast<llvm::VectorType>(type)) {
      type = vectorTy->getElementType();
    } else {
      return type;
    }
  } while (true);
}

/// Convert a dense elements attribute to an LLVM IR constant using its raw data
/// storage if possible. This supports elements attributes of tensor or vector
/// type and avoids constructing separate objects for individual values of the
/// innermost dimension. Constants for other dimensions are still constructed
/// recursively. Returns null if constructing from raw data is not supported for
/// this type, e.g., element type is not a power-of-two-sized primitive. Reports
/// other errors at `loc`.
static llvm::Constant *
convertDenseElementsAttr(Location loc, DenseElementsAttr denseElementsAttr,
                         llvm::Type *llvmType,
                         const ModuleTranslation &moduleTranslation) {
  if (!denseElementsAttr)
    return nullptr;

  llvm::Type *innermostLLVMType = getInnermostElementType(llvmType);
  if (!llvm::ConstantDataSequential::isElementTypeCompatible(innermostLLVMType))
    return nullptr;

  ShapedType type = denseElementsAttr.getType();
  if (type.getNumElements() == 0)
    return nullptr;

  // Check that the raw data size matches what is expected for the scalar size.
  // TODO: in theory, we could repack the data here to keep constructing from
  // raw data.
  // TODO: we may also need to consider endianness when cross-compiling to an
  // architecture where it is different.
  unsigned elementByteSize = denseElementsAttr.getRawData().size() /
                             denseElementsAttr.getNumElements();
  if (8 * elementByteSize != innermostLLVMType->getScalarSizeInBits())
    return nullptr;

  // Compute the shape of all dimensions but the innermost. Note that the
  // innermost dimension may be that of the vector element type.
  bool hasVectorElementType = isa<VectorType>(type.getElementType());
  unsigned numAggregates =
      denseElementsAttr.getNumElements() /
      (hasVectorElementType ? 1
                            : denseElementsAttr.getType().getShape().back());
  ArrayRef<int64_t> outerShape = type.getShape();
  if (!hasVectorElementType)
    outerShape = outerShape.drop_back();

  // Handle the case of vector splat, LLVM has special support for it.
  if (denseElementsAttr.isSplat() &&
      (isa<VectorType>(type) || hasVectorElementType)) {
    llvm::Constant *splatValue = LLVM::detail::getLLVMConstant(
        innermostLLVMType, denseElementsAttr.getSplatValue<Attribute>(), loc,
        moduleTranslation);
    llvm::Constant *splatVector =
        llvm::ConstantDataVector::getSplat(0, splatValue);
    SmallVector<llvm::Constant *> constants(numAggregates, splatVector);
    ArrayRef<llvm::Constant *> constantsRef = constants;
    return buildSequentialConstant(constantsRef, outerShape, llvmType, loc);
  }
  if (denseElementsAttr.isSplat())
    return nullptr;

  // In case of non-splat, create a constructor for the innermost constant from
  // a piece of raw data.
  std::function<llvm::Constant *(StringRef)> buildCstData;
  if (isa<TensorType>(type)) {
    auto vectorElementType = dyn_cast<VectorType>(type.getElementType());
    if (vectorElementType && vectorElementType.getRank() == 1) {
      buildCstData = [&](StringRef data) {
        return llvm::ConstantDataVector::getRaw(
            data, vectorElementType.getShape().back(), innermostLLVMType);
      };
    } else if (!vectorElementType) {
      buildCstData = [&](StringRef data) {
        return llvm::ConstantDataArray::getRaw(data, type.getShape().back(),
                                               innermostLLVMType);
      };
    }
  } else if (isa<VectorType>(type)) {
    buildCstData = [&](StringRef data) {
      return llvm::ConstantDataVector::getRaw(data, type.getShape().back(),
                                              innermostLLVMType);
    };
  }
  if (!buildCstData)
    return nullptr;

  // Create innermost constants and defer to the default constant creation
  // mechanism for other dimensions.
  SmallVector<llvm::Constant *> constants;
  unsigned aggregateSize = denseElementsAttr.getType().getShape().back() *
                           (innermostLLVMType->getScalarSizeInBits() / 8);
  constants.reserve(numAggregates);
  for (unsigned i = 0; i < numAggregates; ++i) {
    StringRef data(denseElementsAttr.getRawData().data() + i * aggregateSize,
                   aggregateSize);
    constants.push_back(buildCstData(data));
  }

  ArrayRef<llvm::Constant *> constantsRef = constants;
  return buildSequentialConstant(constantsRef, outerShape, llvmType, loc);
}

/// Create an LLVM IR constant of `llvmType` from the MLIR attribute `attr`.
/// This currently supports integer, floating point, splat and dense element
/// attributes and combinations thereof. Also, an array attribute with two
/// elements is supported to represent a complex constant.  In case of error,
/// report it to `loc` and return nullptr.
llvm::Constant *mlir::LLVM::detail::getLLVMConstant(
    llvm::Type *llvmType, Attribute attr, Location loc,
    const ModuleTranslation &moduleTranslation) {
  if (!attr)
    return llvm::UndefValue::get(llvmType);
  if (auto *structType = dyn_cast<::llvm::StructType>(llvmType)) {
    auto arrayAttr = dyn_cast<ArrayAttr>(attr);
    if (!arrayAttr || arrayAttr.size() != 2) {
      emitError(loc, "expected struct type to be a complex number");
      return nullptr;
    }
    llvm::Type *elementType = structType->getElementType(0);
    llvm::Constant *real =
        getLLVMConstant(elementType, arrayAttr[0], loc, moduleTranslation);
    if (!real)
      return nullptr;
    llvm::Constant *imag =
        getLLVMConstant(elementType, arrayAttr[1], loc, moduleTranslation);
    if (!imag)
      return nullptr;
    return llvm::ConstantStruct::get(structType, {real, imag});
  }
  if (auto *targetExtType = dyn_cast<::llvm::TargetExtType>(llvmType)) {
    // TODO: Replace with 'zeroinitializer' once there is a dedicated
    // zeroinitializer operation in the LLVM dialect.
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    if (!intAttr || intAttr.getInt() != 0)
      emitError(loc,
                "Only zero-initialization allowed for target extension type");

    return llvm::ConstantTargetNone::get(targetExtType);
  }
  // For integer types, we allow a mismatch in sizes as the index type in
  // MLIR might have a different size than the index type in the LLVM module.
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return llvm::ConstantInt::get(
        llvmType,
        intAttr.getValue().sextOrTrunc(llvmType->getIntegerBitWidth()));
  if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    const llvm::fltSemantics &sem = floatAttr.getValue().getSemantics();
    // Special case for 8-bit floats, which are represented by integers due to
    // the lack of native fp8 types in LLVM at the moment. Additionally, handle
    // targets (like AMDGPU) that don't implement bfloat and convert all bfloats
    // to i16.
    unsigned floatWidth = APFloat::getSizeInBits(sem);
    if (llvmType->isIntegerTy(floatWidth))
      return llvm::ConstantInt::get(llvmType,
                                    floatAttr.getValue().bitcastToAPInt());
    if (llvmType !=
        llvm::Type::getFloatingPointTy(llvmType->getContext(),
                                       floatAttr.getValue().getSemantics())) {
      emitError(loc, "FloatAttr does not match expected type of the constant");
      return nullptr;
    }
    return llvm::ConstantFP::get(llvmType, floatAttr.getValue());
  }
  if (auto funcAttr = dyn_cast<FlatSymbolRefAttr>(attr))
    return llvm::ConstantExpr::getBitCast(
        moduleTranslation.lookupFunction(funcAttr.getValue()), llvmType);
  if (auto splatAttr = dyn_cast<SplatElementsAttr>(attr)) {
    llvm::Type *elementType;
    uint64_t numElements;
    bool isScalable = false;
    if (auto *arrayTy = dyn_cast<llvm::ArrayType>(llvmType)) {
      elementType = arrayTy->getElementType();
      numElements = arrayTy->getNumElements();
    } else if (auto *fVectorTy = dyn_cast<llvm::FixedVectorType>(llvmType)) {
      elementType = fVectorTy->getElementType();
      numElements = fVectorTy->getNumElements();
    } else if (auto *sVectorTy = dyn_cast<llvm::ScalableVectorType>(llvmType)) {
      elementType = sVectorTy->getElementType();
      numElements = sVectorTy->getMinNumElements();
      isScalable = true;
    } else {
      llvm_unreachable("unrecognized constant vector type");
    }
    // Splat value is a scalar. Extract it only if the element type is not
    // another sequence type. The recursion terminates because each step removes
    // one outer sequential type.
    bool elementTypeSequential =
        isa<llvm::ArrayType, llvm::VectorType>(elementType);
    llvm::Constant *child = getLLVMConstant(
        elementType,
        elementTypeSequential ? splatAttr
                              : splatAttr.getSplatValue<Attribute>(),
        loc, moduleTranslation);
    if (!child)
      return nullptr;
    if (llvmType->isVectorTy())
      return llvm::ConstantVector::getSplat(
          llvm::ElementCount::get(numElements, /*Scalable=*/isScalable), child);
    if (llvmType->isArrayTy()) {
      auto *arrayType = llvm::ArrayType::get(elementType, numElements);
      SmallVector<llvm::Constant *, 8> constants(numElements, child);
      return llvm::ConstantArray::get(arrayType, constants);
    }
  }

  // Try using raw elements data if possible.
  if (llvm::Constant *result =
          convertDenseElementsAttr(loc, dyn_cast<DenseElementsAttr>(attr),
                                   llvmType, moduleTranslation)) {
    return result;
  }

  // Fall back to element-by-element construction otherwise.
  if (auto elementsAttr = dyn_cast<ElementsAttr>(attr)) {
    assert(elementsAttr.getShapedType().hasStaticShape());
    assert(!elementsAttr.getShapedType().getShape().empty() &&
           "unexpected empty elements attribute shape");

    SmallVector<llvm::Constant *, 8> constants;
    constants.reserve(elementsAttr.getNumElements());
    llvm::Type *innermostType = getInnermostElementType(llvmType);
    for (auto n : elementsAttr.getValues<Attribute>()) {
      constants.push_back(
          getLLVMConstant(innermostType, n, loc, moduleTranslation));
      if (!constants.back())
        return nullptr;
    }
    ArrayRef<llvm::Constant *> constantsRef = constants;
    llvm::Constant *result = buildSequentialConstant(
        constantsRef, elementsAttr.getShapedType().getShape(), llvmType, loc);
    assert(constantsRef.empty() && "did not consume all elemental constants");
    return result;
  }

  if (auto stringAttr = dyn_cast<StringAttr>(attr)) {
    return llvm::ConstantDataArray::get(
        moduleTranslation.getLLVMContext(),
        ArrayRef<char>{stringAttr.getValue().data(),
                       stringAttr.getValue().size()});
  }
  emitError(loc, "unsupported constant value");
  return nullptr;
}

ModuleTranslation::ModuleTranslation(Operation *module,
                                     std::unique_ptr<llvm::Module> llvmModule)
    : mlirModule(module), llvmModule(std::move(llvmModule)),
      debugTranslation(
          std::make_unique<DebugTranslation>(module, *this->llvmModule)),
      loopAnnotationTranslation(std::make_unique<LoopAnnotationTranslation>(
          *this, *this->llvmModule)),
      typeTranslator(this->llvmModule->getContext()),
      iface(module->getContext()) {
  assert(satisfiesLLVMModule(mlirModule) &&
         "mlirModule should honor LLVM's module semantics.");
}

ModuleTranslation::~ModuleTranslation() {
  if (ompBuilder)
    ompBuilder->finalize();
}

void ModuleTranslation::forgetMapping(Region &region) {
  SmallVector<Region *> toProcess;
  toProcess.push_back(&region);
  while (!toProcess.empty()) {
    Region *current = toProcess.pop_back_val();
    for (Block &block : *current) {
      blockMapping.erase(&block);
      for (Value arg : block.getArguments())
        valueMapping.erase(arg);
      for (Operation &op : block) {
        for (Value value : op.getResults())
          valueMapping.erase(value);
        if (op.hasSuccessors())
          branchMapping.erase(&op);
        if (isa<LLVM::GlobalOp>(op))
          globalsMapping.erase(&op);
        llvm::append_range(
            toProcess,
            llvm::map_range(op.getRegions(), [](Region &r) { return &r; }));
      }
    }
  }
}

/// Get the SSA value passed to the current block from the terminator operation
/// of its predecessor.
static Value getPHISourceValue(Block *current, Block *pred,
                               unsigned numArguments, unsigned index) {
  Operation &terminator = *pred->getTerminator();
  if (isa<LLVM::BrOp>(terminator))
    return terminator.getOperand(index);

#ifndef NDEBUG
  llvm::SmallPtrSet<Block *, 4> seenSuccessors;
  for (unsigned i = 0, e = terminator.getNumSuccessors(); i < e; ++i) {
    Block *successor = terminator.getSuccessor(i);
    auto branch = cast<BranchOpInterface>(terminator);
    SuccessorOperands successorOperands = branch.getSuccessorOperands(i);
    assert(
        (!seenSuccessors.contains(successor) || successorOperands.empty()) &&
        "successors with arguments in LLVM branches must be different blocks");
    seenSuccessors.insert(successor);
  }
#endif

  // For instructions that branch based on a condition value, we need to take
  // the operands for the branch that was taken.
  if (auto condBranchOp = dyn_cast<LLVM::CondBrOp>(terminator)) {
    // For conditional branches, we take the operands from either the "true" or
    // the "false" branch.
    return condBranchOp.getSuccessor(0) == current
               ? condBranchOp.getTrueDestOperands()[index]
               : condBranchOp.getFalseDestOperands()[index];
  }

  if (auto switchOp = dyn_cast<LLVM::SwitchOp>(terminator)) {
    // For switches, we take the operands from either the default case, or from
    // the case branch that was taken.
    if (switchOp.getDefaultDestination() == current)
      return switchOp.getDefaultOperands()[index];
    for (const auto &i : llvm::enumerate(switchOp.getCaseDestinations()))
      if (i.value() == current)
        return switchOp.getCaseOperands(i.index())[index];
  }

  if (auto invokeOp = dyn_cast<LLVM::InvokeOp>(terminator)) {
    return invokeOp.getNormalDest() == current
               ? invokeOp.getNormalDestOperands()[index]
               : invokeOp.getUnwindDestOperands()[index];
  }

  llvm_unreachable(
      "only branch, switch or invoke operations can be terminators "
      "of a block that has successors");
}

/// Connect the PHI nodes to the results of preceding blocks.
void mlir::LLVM::detail::connectPHINodes(Region &region,
                                         const ModuleTranslation &state) {
  // Skip the first block, it cannot be branched to and its arguments correspond
  // to the arguments of the LLVM function.
  for (Block &bb : llvm::drop_begin(region)) {
    llvm::BasicBlock *llvmBB = state.lookupBlock(&bb);
    auto phis = llvmBB->phis();
    auto numArguments = bb.getNumArguments();
    assert(numArguments == std::distance(phis.begin(), phis.end()));
    for (auto [index, phiNode] : llvm::enumerate(phis)) {
      for (auto *pred : bb.getPredecessors()) {
        // Find the LLVM IR block that contains the converted terminator
        // instruction and use it in the PHI node. Note that this block is not
        // necessarily the same as state.lookupBlock(pred), some operations
        // (in particular, OpenMP operations using OpenMPIRBuilder) may have
        // split the blocks.
        llvm::Instruction *terminator =
            state.lookupBranch(pred->getTerminator());
        assert(terminator && "missing the mapping for a terminator");
        phiNode.addIncoming(state.lookupValue(getPHISourceValue(
                                &bb, pred, numArguments, index)),
                            terminator->getParent());
      }
    }
  }
}

/// Sort function blocks topologically.
SetVector<Block *>
mlir::LLVM::detail::getTopologicallySortedBlocks(Region &region) {
  // For each block that has not been visited yet (i.e. that has no
  // predecessors), add it to the list as well as its successors.
  SetVector<Block *> blocks;
  for (Block &b : region) {
    if (blocks.count(&b) == 0) {
      llvm::ReversePostOrderTraversal<Block *> traversal(&b);
      blocks.insert(traversal.begin(), traversal.end());
    }
  }
  assert(blocks.size() == region.getBlocks().size() &&
         "some blocks are not sorted");

  return blocks;
}

llvm::CallInst *mlir::LLVM::detail::createIntrinsicCall(
    llvm::IRBuilderBase &builder, llvm::Intrinsic::ID intrinsic,
    ArrayRef<llvm::Value *> args, ArrayRef<llvm::Type *> tys) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::Function *fn = llvm::Intrinsic::getDeclaration(module, intrinsic, tys);
  return builder.CreateCall(fn, args);
}

/// Given a single MLIR operation, create the corresponding LLVM IR operation
/// using the `builder`.
LogicalResult
ModuleTranslation::convertOperation(Operation &op,
                                    llvm::IRBuilderBase &builder) {
  const LLVMTranslationDialectInterface *opIface = iface.getInterfaceFor(&op);
  if (!opIface)
    return op.emitError("cannot be converted to LLVM IR: missing "
                        "`LLVMTranslationDialectInterface` registration for "
                        "dialect for op: ")
           << op.getName();

  if (failed(opIface->convertOperation(&op, builder, *this)))
    return op.emitError("LLVM Translation failed for operation: ")
           << op.getName();

  return convertDialectAttributes(&op);
}

/// Convert block to LLVM IR.  Unless `ignoreArguments` is set, emit PHI nodes
/// to define values corresponding to the MLIR block arguments.  These nodes
/// are not connected to the source basic blocks, which may not exist yet.  Uses
/// `builder` to construct the LLVM IR. Expects the LLVM IR basic block to have
/// been created for `bb` and included in the block mapping.  Inserts new
/// instructions at the end of the block and leaves `builder` in a state
/// suitable for further insertion into the end of the block.
LogicalResult ModuleTranslation::convertBlock(Block &bb, bool ignoreArguments,
                                              llvm::IRBuilderBase &builder) {
  builder.SetInsertPoint(lookupBlock(&bb));
  auto *subprogram = builder.GetInsertBlock()->getParent()->getSubprogram();

  // Before traversing operations, make block arguments available through
  // value remapping and PHI nodes, but do not add incoming edges for the PHI
  // nodes just yet: those values may be defined by this or following blocks.
  // This step is omitted if "ignoreArguments" is set.  The arguments of the
  // first block have been already made available through the remapping of
  // LLVM function arguments.
  if (!ignoreArguments) {
    auto predecessors = bb.getPredecessors();
    unsigned numPredecessors =
        std::distance(predecessors.begin(), predecessors.end());
    for (auto arg : bb.getArguments()) {
      auto wrappedType = arg.getType();
      if (!isCompatibleType(wrappedType))
        return emitError(bb.front().getLoc(),
                         "block argument does not have an LLVM type");
      llvm::Type *type = convertType(wrappedType);
      llvm::PHINode *phi = builder.CreatePHI(type, numPredecessors);
      mapValue(arg, phi);
    }
  }

  // Traverse operations.
  for (auto &op : bb) {
    // Set the current debug location within the builder.
    builder.SetCurrentDebugLocation(
        debugTranslation->translateLoc(op.getLoc(), subprogram));

    if (failed(convertOperation(op, builder)))
      return failure();

    // Set the branch weight metadata on the translated instruction.
    if (auto iface = dyn_cast<BranchWeightOpInterface>(op))
      setBranchWeightsMetadata(iface);
  }

  return success();
}

/// A helper method to get the single Block in an operation honoring LLVM's
/// module requirements.
static Block &getModuleBody(Operation *module) {
  return module->getRegion(0).front();
}

/// A helper method to decide if a constant must not be set as a global variable
/// initializer. For an external linkage variable, the variable with an
/// initializer is considered externally visible and defined in this module, the
/// variable without an initializer is externally available and is defined
/// elsewhere.
static bool shouldDropGlobalInitializer(llvm::GlobalValue::LinkageTypes linkage,
                                        llvm::Constant *cst) {
  return (linkage == llvm::GlobalVariable::ExternalLinkage && !cst) ||
         linkage == llvm::GlobalVariable::ExternalWeakLinkage;
}

/// Sets the runtime preemption specifier of `gv` to dso_local if
/// `dsoLocalRequested` is true, otherwise it is left unchanged.
static void addRuntimePreemptionSpecifier(bool dsoLocalRequested,
                                          llvm::GlobalValue *gv) {
  if (dsoLocalRequested)
    gv->setDSOLocal(true);
}

/// Create named global variables that correspond to llvm.mlir.global
/// definitions. Convert llvm.global_ctors and global_dtors ops.
LogicalResult ModuleTranslation::convertGlobals() {
  for (auto op : getModuleBody(mlirModule).getOps<LLVM::GlobalOp>()) {
    llvm::Type *type = convertType(op.getType());
    llvm::Constant *cst = nullptr;
    if (op.getValueOrNull()) {
      // String attributes are treated separately because they cannot appear as
      // in-function constants and are thus not supported by getLLVMConstant.
      if (auto strAttr = dyn_cast_or_null<StringAttr>(op.getValueOrNull())) {
        cst = llvm::ConstantDataArray::getString(
            llvmModule->getContext(), strAttr.getValue(), /*AddNull=*/false);
        type = cst->getType();
      } else if (!(cst = getLLVMConstant(type, op.getValueOrNull(), op.getLoc(),
                                         *this))) {
        return failure();
      }
    }

    auto linkage = convertLinkageToLLVM(op.getLinkage());
    auto addrSpace = op.getAddrSpace();

    // LLVM IR requires constant with linkage other than external or weak
    // external to have initializers. If MLIR does not provide an initializer,
    // default to undef.
    bool dropInitializer = shouldDropGlobalInitializer(linkage, cst);
    if (!dropInitializer && !cst)
      cst = llvm::UndefValue::get(type);
    else if (dropInitializer && cst)
      cst = nullptr;

    auto *var = new llvm::GlobalVariable(
        *llvmModule, type, op.getConstant(), linkage, cst, op.getSymName(),
        /*InsertBefore=*/nullptr,
        op.getThreadLocal_() ? llvm::GlobalValue::GeneralDynamicTLSModel
                             : llvm::GlobalValue::NotThreadLocal,
        addrSpace);

    if (std::optional<mlir::SymbolRefAttr> comdat = op.getComdat()) {
      auto selectorOp = cast<ComdatSelectorOp>(
          SymbolTable::lookupNearestSymbolFrom(op, *comdat));
      var->setComdat(comdatMapping.lookup(selectorOp));
    }

    if (op.getUnnamedAddr().has_value())
      var->setUnnamedAddr(convertUnnamedAddrToLLVM(*op.getUnnamedAddr()));

    if (op.getSection().has_value())
      var->setSection(*op.getSection());

    addRuntimePreemptionSpecifier(op.getDsoLocal(), var);

    std::optional<uint64_t> alignment = op.getAlignment();
    if (alignment.has_value())
      var->setAlignment(llvm::MaybeAlign(alignment.value()));

    var->setVisibility(convertVisibilityToLLVM(op.getVisibility_()));

    globalsMapping.try_emplace(op, var);
  }

  // Convert global variable bodies. This is done after all global variables
  // have been created in LLVM IR because a global body may refer to another
  // global or itself. So all global variables need to be mapped first.
  for (auto op : getModuleBody(mlirModule).getOps<LLVM::GlobalOp>()) {
    if (Block *initializer = op.getInitializerBlock()) {
      llvm::IRBuilder<> builder(llvmModule->getContext());
      for (auto &op : initializer->without_terminator()) {
        if (failed(convertOperation(op, builder)) ||
            !isa<llvm::Constant>(lookupValue(op.getResult(0))))
          return emitError(op.getLoc(), "unemittable constant value");
      }
      ReturnOp ret = cast<ReturnOp>(initializer->getTerminator());
      llvm::Constant *cst =
          cast<llvm::Constant>(lookupValue(ret.getOperand(0)));
      auto *global = cast<llvm::GlobalVariable>(lookupGlobal(op));
      if (!shouldDropGlobalInitializer(global->getLinkage(), cst))
        global->setInitializer(cst);
    }
  }

  // Convert llvm.mlir.global_ctors and dtors.
  for (Operation &op : getModuleBody(mlirModule)) {
    auto ctorOp = dyn_cast<GlobalCtorsOp>(op);
    auto dtorOp = dyn_cast<GlobalDtorsOp>(op);
    if (!ctorOp && !dtorOp)
      continue;
    auto range = ctorOp ? llvm::zip(ctorOp.getCtors(), ctorOp.getPriorities())
                        : llvm::zip(dtorOp.getDtors(), dtorOp.getPriorities());
    auto appendGlobalFn =
        ctorOp ? llvm::appendToGlobalCtors : llvm::appendToGlobalDtors;
    for (auto symbolAndPriority : range) {
      llvm::Function *f = lookupFunction(
          cast<FlatSymbolRefAttr>(std::get<0>(symbolAndPriority)).getValue());
      appendGlobalFn(*llvmModule, f,
                     cast<IntegerAttr>(std::get<1>(symbolAndPriority)).getInt(),
                     /*Data=*/nullptr);
    }
  }

  for (auto op : getModuleBody(mlirModule).getOps<LLVM::GlobalOp>())
    if (failed(convertDialectAttributes(op)))
      return failure();

  return success();
}

/// Attempts to add an attribute identified by `key`, optionally with the given
/// `value` to LLVM function `llvmFunc`. Reports errors at `loc` if any. If the
/// attribute has a kind known to LLVM IR, create the attribute of this kind,
/// otherwise keep it as a string attribute. Performs additional checks for
/// attributes known to have or not have a value in order to avoid assertions
/// inside LLVM upon construction.
static LogicalResult checkedAddLLVMFnAttribute(Location loc,
                                               llvm::Function *llvmFunc,
                                               StringRef key,
                                               StringRef value = StringRef()) {
  auto kind = llvm::Attribute::getAttrKindFromName(key);
  if (kind == llvm::Attribute::None) {
    llvmFunc->addFnAttr(key, value);
    return success();
  }

  if (llvm::Attribute::isIntAttrKind(kind)) {
    if (value.empty())
      return emitError(loc) << "LLVM attribute '" << key << "' expects a value";

    int64_t result;
    if (!value.getAsInteger(/*Radix=*/0, result))
      llvmFunc->addFnAttr(
          llvm::Attribute::get(llvmFunc->getContext(), kind, result));
    else
      llvmFunc->addFnAttr(key, value);
    return success();
  }

  if (!value.empty())
    return emitError(loc) << "LLVM attribute '" << key
                          << "' does not expect a value, found '" << value
                          << "'";

  llvmFunc->addFnAttr(kind);
  return success();
}

/// Attaches the attributes listed in the given array attribute to `llvmFunc`.
/// Reports error to `loc` if any and returns immediately. Expects `attributes`
/// to be an array attribute containing either string attributes, treated as
/// value-less LLVM attributes, or array attributes containing two string
/// attributes, with the first string being the name of the corresponding LLVM
/// attribute and the second string beings its value. Note that even integer
/// attributes are expected to have their values expressed as strings.
static LogicalResult
forwardPassthroughAttributes(Location loc, std::optional<ArrayAttr> attributes,
                             llvm::Function *llvmFunc) {
  if (!attributes)
    return success();

  for (Attribute attr : *attributes) {
    if (auto stringAttr = dyn_cast<StringAttr>(attr)) {
      if (failed(
              checkedAddLLVMFnAttribute(loc, llvmFunc, stringAttr.getValue())))
        return failure();
      continue;
    }

    auto arrayAttr = dyn_cast<ArrayAttr>(attr);
    if (!arrayAttr || arrayAttr.size() != 2)
      return emitError(loc)
             << "expected 'passthrough' to contain string or array attributes";

    auto keyAttr = dyn_cast<StringAttr>(arrayAttr[0]);
    auto valueAttr = dyn_cast<StringAttr>(arrayAttr[1]);
    if (!keyAttr || !valueAttr)
      return emitError(loc)
             << "expected arrays within 'passthrough' to contain two strings";

    if (failed(checkedAddLLVMFnAttribute(loc, llvmFunc, keyAttr.getValue(),
                                         valueAttr.getValue())))
      return failure();
  }
  return success();
}

LogicalResult ModuleTranslation::convertOneFunction(LLVMFuncOp func) {
  // Clear the block, branch value mappings, they are only relevant within one
  // function.
  blockMapping.clear();
  valueMapping.clear();
  branchMapping.clear();
  llvm::Function *llvmFunc = lookupFunction(func.getName());

  // Translate the debug information for this function.
  debugTranslation->translate(func, *llvmFunc);

  // Add function arguments to the value remapping table.
  for (auto [mlirArg, llvmArg] :
       llvm::zip(func.getArguments(), llvmFunc->args()))
    mapValue(mlirArg, &llvmArg);

  // Check the personality and set it.
  if (func.getPersonality()) {
    llvm::Type *ty = llvm::Type::getInt8PtrTy(llvmFunc->getContext());
    if (llvm::Constant *pfunc = getLLVMConstant(ty, func.getPersonalityAttr(),
                                                func.getLoc(), *this))
      llvmFunc->setPersonalityFn(pfunc);
  }

  if (std::optional<StringRef> section = func.getSection())
    llvmFunc->setSection(*section);

  if (func.getArmStreaming())
    llvmFunc->addFnAttr("aarch64_pstate_sm_enabled");
  else if (func.getArmLocallyStreaming())
    llvmFunc->addFnAttr("aarch64_pstate_sm_body");

  // First, create all blocks so we can jump to them.
  llvm::LLVMContext &llvmContext = llvmFunc->getContext();
  for (auto &bb : func) {
    auto *llvmBB = llvm::BasicBlock::Create(llvmContext);
    llvmBB->insertInto(llvmFunc);
    mapBlock(&bb, llvmBB);
  }

  // Then, convert blocks one by one in topological order to ensure defs are
  // converted before uses.
  auto blocks = detail::getTopologicallySortedBlocks(func.getBody());
  for (Block *bb : blocks) {
    llvm::IRBuilder<> builder(llvmContext);
    if (failed(convertBlock(*bb, bb->isEntryBlock(), builder)))
      return failure();
  }

  // After all blocks have been traversed and values mapped, connect the PHI
  // nodes to the results of preceding blocks.
  detail::connectPHINodes(func.getBody(), *this);

  // Finally, convert dialect attributes attached to the function.
  return convertDialectAttributes(func);
}

LogicalResult ModuleTranslation::convertDialectAttributes(Operation *op) {
  for (NamedAttribute attribute : op->getDialectAttrs())
    if (failed(iface.amendOperation(op, attribute, *this)))
      return failure();
  return success();
}

/// Converts the function attributes from LLVMFuncOp and attaches them to the
/// llvm::Function.
static void convertFunctionAttributes(LLVMFuncOp func,
                                      llvm::Function *llvmFunc) {
  if (!func.getMemory())
    return;

  MemoryEffectsAttr memEffects = func.getMemoryAttr();

  // Add memory effects incrementally.
  llvm::MemoryEffects newMemEffects =
      llvm::MemoryEffects(llvm::MemoryEffects::Location::ArgMem,
                          convertModRefInfoToLLVM(memEffects.getArgMem()));
  newMemEffects |= llvm::MemoryEffects(
      llvm::MemoryEffects::Location::InaccessibleMem,
      convertModRefInfoToLLVM(memEffects.getInaccessibleMem()));
  newMemEffects |=
      llvm::MemoryEffects(llvm::MemoryEffects::Location::Other,
                          convertModRefInfoToLLVM(memEffects.getOther()));
  llvmFunc->setMemoryEffects(newMemEffects);
}

llvm::AttrBuilder
ModuleTranslation::convertParameterAttrs(DictionaryAttr paramAttrs) {
  llvm::AttrBuilder attrBuilder(llvmModule->getContext());

  for (auto [llvmKind, mlirName] : getAttrKindToNameMapping()) {
    Attribute attr = paramAttrs.get(mlirName);
    // Skip attributes that are not present.
    if (!attr)
      continue;

    // NOTE: C++17 does not support capturing structured bindings.
    llvm::Attribute::AttrKind llvmKindCap = llvmKind;

    llvm::TypeSwitch<Attribute>(attr)
        .Case<TypeAttr>([&](auto typeAttr) {
          attrBuilder.addTypeAttr(llvmKindCap,
                                  convertType(typeAttr.getValue()));
        })
        .Case<IntegerAttr>([&](auto intAttr) {
          attrBuilder.addRawIntAttr(llvmKindCap, intAttr.getInt());
        })
        .Case<UnitAttr>([&](auto) { attrBuilder.addAttribute(llvmKindCap); });
  }

  return attrBuilder;
}

LogicalResult ModuleTranslation::convertFunctionSignatures() {
  // Declare all functions first because there may be function calls that form a
  // call graph with cycles, or global initializers that reference functions.
  for (auto function : getModuleBody(mlirModule).getOps<LLVMFuncOp>()) {
    llvm::FunctionCallee llvmFuncCst = llvmModule->getOrInsertFunction(
        function.getName(),
        cast<llvm::FunctionType>(convertType(function.getFunctionType())));
    llvm::Function *llvmFunc = cast<llvm::Function>(llvmFuncCst.getCallee());
    llvmFunc->setLinkage(convertLinkageToLLVM(function.getLinkage()));
    llvmFunc->setCallingConv(convertCConvToLLVM(function.getCConv()));
    mapFunction(function.getName(), llvmFunc);
    addRuntimePreemptionSpecifier(function.getDsoLocal(), llvmFunc);

    // Convert function attributes.
    convertFunctionAttributes(function, llvmFunc);

    // Convert function_entry_count attribute to metadata.
    if (std::optional<uint64_t> entryCount = function.getFunctionEntryCount())
      llvmFunc->setEntryCount(entryCount.value());

    // Convert result attributes.
    if (ArrayAttr allResultAttrs = function.getAllResultAttrs()) {
      DictionaryAttr resultAttrs = cast<DictionaryAttr>(allResultAttrs[0]);
      llvmFunc->addRetAttrs(convertParameterAttrs(resultAttrs));
    }

    // Convert argument attributes.
    for (auto [argIdx, llvmArg] : llvm::enumerate(llvmFunc->args())) {
      if (DictionaryAttr argAttrs = function.getArgAttrDict(argIdx)) {
        llvm::AttrBuilder attrBuilder = convertParameterAttrs(argAttrs);
        llvmArg.addAttrs(attrBuilder);
      }
    }

    // Forward the pass-through attributes to LLVM.
    if (failed(forwardPassthroughAttributes(
            function.getLoc(), function.getPassthrough(), llvmFunc)))
      return failure();

    // Convert visibility attribute.
    llvmFunc->setVisibility(convertVisibilityToLLVM(function.getVisibility_()));

    // Convert the comdat attribute.
    if (std::optional<mlir::SymbolRefAttr> comdat = function.getComdat()) {
      auto selectorOp = cast<ComdatSelectorOp>(
          SymbolTable::lookupNearestSymbolFrom(function, *comdat));
      llvmFunc->setComdat(comdatMapping.lookup(selectorOp));
    }

    if (auto gc = function.getGarbageCollector())
      llvmFunc->setGC(gc->str());

    if (auto unnamedAddr = function.getUnnamedAddr())
      llvmFunc->setUnnamedAddr(convertUnnamedAddrToLLVM(*unnamedAddr));

    if (auto alignment = function.getAlignment())
      llvmFunc->setAlignment(llvm::MaybeAlign(*alignment));
  }

  return success();
}

LogicalResult ModuleTranslation::convertFunctions() {
  // Convert functions.
  for (auto function : getModuleBody(mlirModule).getOps<LLVMFuncOp>()) {
    // Do not convert external functions, but do process dialect attributes
    // attached to them.
    if (function.isExternal()) {
      if (failed(convertDialectAttributes(function)))
        return failure();
      continue;
    }

    if (failed(convertOneFunction(function)))
      return failure();
  }

  return success();
}

LogicalResult ModuleTranslation::convertComdats() {
  for (auto comdatOp : getModuleBody(mlirModule).getOps<ComdatOp>()) {
    for (auto selectorOp : comdatOp.getOps<ComdatSelectorOp>()) {
      llvm::Module *module = getLLVMModule();
      if (module->getComdatSymbolTable().contains(selectorOp.getSymName()))
        return emitError(selectorOp.getLoc())
               << "comdat selection symbols must be unique even in different "
                  "comdat regions";
      llvm::Comdat *comdat = module->getOrInsertComdat(selectorOp.getSymName());
      comdat->setSelectionKind(convertComdatToLLVM(selectorOp.getComdat()));
      comdatMapping.try_emplace(selectorOp, comdat);
    }
  }
  return success();
}

void ModuleTranslation::setAccessGroupsMetadata(AccessGroupOpInterface op,
                                                llvm::Instruction *inst) {
  if (llvm::MDNode *node = loopAnnotationTranslation->getAccessGroups(op))
    inst->setMetadata(llvm::LLVMContext::MD_access_group, node);
}

llvm::MDNode *
ModuleTranslation::getOrCreateAliasScope(AliasScopeAttr aliasScopeAttr) {
  auto [scopeIt, scopeInserted] =
      aliasScopeMetadataMapping.try_emplace(aliasScopeAttr, nullptr);
  if (!scopeInserted)
    return scopeIt->second;
  llvm::LLVMContext &ctx = llvmModule->getContext();
  // Convert the domain metadata node if necessary.
  auto [domainIt, insertedDomain] = aliasDomainMetadataMapping.try_emplace(
      aliasScopeAttr.getDomain(), nullptr);
  if (insertedDomain) {
    llvm::SmallVector<llvm::Metadata *, 2> operands;
    // Placeholder for self-reference.
    operands.push_back({});
    if (StringAttr description = aliasScopeAttr.getDomain().getDescription())
      operands.push_back(llvm::MDString::get(ctx, description));
    domainIt->second = llvm::MDNode::get(ctx, operands);
    // Self-reference for uniqueness.
    domainIt->second->replaceOperandWith(0, domainIt->second);
  }
  // Convert the scope metadata node.
  assert(domainIt->second && "Scope's domain should already be valid");
  llvm::SmallVector<llvm::Metadata *, 3> operands;
  // Placeholder for self-reference.
  operands.push_back({});
  operands.push_back(domainIt->second);
  if (StringAttr description = aliasScopeAttr.getDescription())
    operands.push_back(llvm::MDString::get(ctx, description));
  scopeIt->second = llvm::MDNode::get(ctx, operands);
  // Self-reference for uniqueness.
  scopeIt->second->replaceOperandWith(0, scopeIt->second);
  return scopeIt->second;
}

llvm::MDNode *ModuleTranslation::getOrCreateAliasScopes(
    ArrayRef<AliasScopeAttr> aliasScopeAttrs) {
  SmallVector<llvm::Metadata *> nodes;
  nodes.reserve(aliasScopeAttrs.size());
  for (AliasScopeAttr aliasScopeAttr : aliasScopeAttrs)
    nodes.push_back(getOrCreateAliasScope(aliasScopeAttr));
  return llvm::MDNode::get(getLLVMContext(), nodes);
}

void ModuleTranslation::setAliasScopeMetadata(AliasAnalysisOpInterface op,
                                              llvm::Instruction *inst) {
  auto populateScopeMetadata = [&](ArrayAttr aliasScopeAttrs, unsigned kind) {
    if (!aliasScopeAttrs || aliasScopeAttrs.empty())
      return;
    llvm::MDNode *node = getOrCreateAliasScopes(
        llvm::to_vector(aliasScopeAttrs.getAsRange<AliasScopeAttr>()));
    inst->setMetadata(kind, node);
  };

  populateScopeMetadata(op.getAliasScopesOrNull(),
                        llvm::LLVMContext::MD_alias_scope);
  populateScopeMetadata(op.getNoAliasScopesOrNull(),
                        llvm::LLVMContext::MD_noalias);
}

llvm::MDNode *ModuleTranslation::getTBAANode(TBAATagAttr tbaaAttr) const {
  return tbaaMetadataMapping.lookup(tbaaAttr);
}

void ModuleTranslation::setTBAAMetadata(AliasAnalysisOpInterface op,
                                        llvm::Instruction *inst) {
  ArrayAttr tagRefs = op.getTBAATagsOrNull();
  if (!tagRefs || tagRefs.empty())
    return;

  // LLVM IR currently does not support attaching more than one TBAA access tag
  // to a memory accessing instruction. It may be useful to support this in
  // future, but for the time being just ignore the metadata if MLIR operation
  // has multiple access tags.
  if (tagRefs.size() > 1) {
    op.emitWarning() << "TBAA access tags were not translated, because LLVM "
                        "IR only supports a single tag per instruction";
    return;
  }

  llvm::MDNode *node = getTBAANode(cast<TBAATagAttr>(tagRefs[0]));
  inst->setMetadata(llvm::LLVMContext::MD_tbaa, node);
}

void ModuleTranslation::setBranchWeightsMetadata(BranchWeightOpInterface op) {
  DenseI32ArrayAttr weightsAttr = op.getBranchWeightsOrNull();
  if (!weightsAttr)
    return;

  llvm::Instruction *inst = isa<CallOp>(op) ? lookupCall(op) : lookupBranch(op);
  assert(inst && "expected the operation to have a mapping to an instruction");
  SmallVector<uint32_t> weights(weightsAttr.asArrayRef());
  inst->setMetadata(
      llvm::LLVMContext::MD_prof,
      llvm::MDBuilder(getLLVMContext()).createBranchWeights(weights));
}

LogicalResult ModuleTranslation::createTBAAMetadata() {
  llvm::LLVMContext &ctx = llvmModule->getContext();
  llvm::IntegerType *offsetTy = llvm::IntegerType::get(ctx, 64);

  // Walk the entire module and create all metadata nodes for the TBAA
  // attributes. The code below relies on two invariants of the
  // `AttrTypeWalker`:
  // 1. Attributes are visited in post-order: Since the attributes create a DAG,
  //    this ensures that any lookups into `tbaaMetadataMapping` for child
  //    attributes succeed.
  // 2. Attributes are only ever visited once: This way we don't leak any
  //    LLVM metadata instances.
  AttrTypeWalker walker;
  walker.addWalk([&](TBAARootAttr root) {
    tbaaMetadataMapping.insert(
        {root, llvm::MDNode::get(ctx, llvm::MDString::get(ctx, root.getId()))});
  });

  walker.addWalk([&](TBAATypeDescriptorAttr descriptor) {
    SmallVector<llvm::Metadata *> operands;
    operands.push_back(llvm::MDString::get(ctx, descriptor.getId()));
    for (TBAAMemberAttr member : descriptor.getMembers()) {
      operands.push_back(tbaaMetadataMapping.lookup(member.getTypeDesc()));
      operands.push_back(llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(offsetTy, member.getOffset())));
    }

    tbaaMetadataMapping.insert({descriptor, llvm::MDNode::get(ctx, operands)});
  });

  walker.addWalk([&](TBAATagAttr tag) {
    SmallVector<llvm::Metadata *> operands;

    operands.push_back(tbaaMetadataMapping.lookup(tag.getBaseType()));
    operands.push_back(tbaaMetadataMapping.lookup(tag.getAccessType()));

    operands.push_back(llvm::ConstantAsMetadata::get(
        llvm::ConstantInt::get(offsetTy, tag.getOffset())));
    if (tag.getConstant())
      operands.push_back(
          llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(offsetTy, 1)));

    tbaaMetadataMapping.insert({tag, llvm::MDNode::get(ctx, operands)});
  });

  mlirModule->walk([&](AliasAnalysisOpInterface analysisOpInterface) {
    if (auto attr = analysisOpInterface.getTBAATagsOrNull())
      walker.walk(attr);
  });

  return success();
}

void ModuleTranslation::setLoopMetadata(Operation *op,
                                        llvm::Instruction *inst) {
  LoopAnnotationAttr attr =
      TypeSwitch<Operation *, LoopAnnotationAttr>(op)
          .Case<LLVM::BrOp, LLVM::CondBrOp>(
              [](auto branchOp) { return branchOp.getLoopAnnotationAttr(); });
  if (!attr)
    return;
  llvm::MDNode *loopMD =
      loopAnnotationTranslation->translateLoopAnnotation(attr, op);
  inst->setMetadata(llvm::LLVMContext::MD_loop, loopMD);
}

llvm::Type *ModuleTranslation::convertType(Type type) {
  return typeTranslator.translateType(type);
}

/// A helper to look up remapped operands in the value remapping table.
SmallVector<llvm::Value *> ModuleTranslation::lookupValues(ValueRange values) {
  SmallVector<llvm::Value *> remapped;
  remapped.reserve(values.size());
  for (Value v : values)
    remapped.push_back(lookupValue(v));
  return remapped;
}

llvm::OpenMPIRBuilder *ModuleTranslation::getOpenMPBuilder() {
  if (!ompBuilder) {
    ompBuilder = std::make_unique<llvm::OpenMPIRBuilder>(*llvmModule);
    ompBuilder->initialize();

    // Flags represented as top-level OpenMP dialect attributes are set in
    // `OpenMPDialectLLVMIRTranslationInterface::amendOperation()`. Here we set
    // the default configuration.
    ompBuilder->setConfig(llvm::OpenMPIRBuilderConfig(
        /* IsTargetDevice = */ false, /* IsGPU = */ false,
        /* OpenMPOffloadMandatory = */ false,
        /* HasRequiresReverseOffload = */ false,
        /* HasRequiresUnifiedAddress = */ false,
        /* HasRequiresUnifiedSharedMemory = */ false,
        /* HasRequiresDynamicAllocators = */ false));
  }
  return ompBuilder.get();
}

llvm::DILocation *ModuleTranslation::translateLoc(Location loc,
                                                  llvm::DILocalScope *scope) {
  return debugTranslation->translateLoc(loc, scope);
}

llvm::Metadata *ModuleTranslation::translateDebugInfo(LLVM::DINodeAttr attr) {
  return debugTranslation->translate(attr);
}

llvm::NamedMDNode *
ModuleTranslation::getOrInsertNamedModuleMetadata(StringRef name) {
  return llvmModule->getOrInsertNamedMetadata(name);
}

void ModuleTranslation::StackFrame::anchor() {}

static std::unique_ptr<llvm::Module>
prepareLLVMModule(Operation *m, llvm::LLVMContext &llvmContext,
                  StringRef name) {
  m->getContext()->getOrLoadDialect<LLVM::LLVMDialect>();
  auto llvmModule = std::make_unique<llvm::Module>(name, llvmContext);
  if (auto dataLayoutAttr =
          m->getDiscardableAttr(LLVM::LLVMDialect::getDataLayoutAttrName())) {
    llvmModule->setDataLayout(cast<StringAttr>(dataLayoutAttr).getValue());
  } else {
    FailureOr<llvm::DataLayout> llvmDataLayout(llvm::DataLayout(""));
    if (auto iface = dyn_cast<DataLayoutOpInterface>(m)) {
      if (DataLayoutSpecInterface spec = iface.getDataLayoutSpec()) {
        llvmDataLayout =
            translateDataLayout(spec, DataLayout(iface), m->getLoc());
      }
    } else if (auto mod = dyn_cast<ModuleOp>(m)) {
      if (DataLayoutSpecInterface spec = mod.getDataLayoutSpec()) {
        llvmDataLayout =
            translateDataLayout(spec, DataLayout(mod), m->getLoc());
      }
    }
    if (failed(llvmDataLayout))
      return nullptr;
    llvmModule->setDataLayout(*llvmDataLayout);
  }
  if (auto targetTripleAttr =
          m->getDiscardableAttr(LLVM::LLVMDialect::getTargetTripleAttrName()))
    llvmModule->setTargetTriple(cast<StringAttr>(targetTripleAttr).getValue());

  // Inject declarations for `malloc` and `free` functions that can be used in
  // memref allocation/deallocation coming from standard ops lowering.
  llvm::IRBuilder<> builder(llvmContext);
  llvmModule->getOrInsertFunction("malloc", builder.getInt8PtrTy(),
                                  builder.getInt64Ty());
  llvmModule->getOrInsertFunction("free", builder.getVoidTy(),
                                  builder.getInt8PtrTy());

  return llvmModule;
}

std::unique_ptr<llvm::Module>
mlir::translateModuleToLLVMIR(Operation *module, llvm::LLVMContext &llvmContext,
                              StringRef name) {
  if (!satisfiesLLVMModule(module)) {
    module->emitOpError("can not be translated to an LLVMIR module");
    return nullptr;
  }

  std::unique_ptr<llvm::Module> llvmModule =
      prepareLLVMModule(module, llvmContext, name);
  if (!llvmModule)
    return nullptr;

  LLVM::ensureDistinctSuccessors(module);

  ModuleTranslation translator(module, std::move(llvmModule));
  llvm::IRBuilder<> llvmBuilder(llvmContext);

  // Convert module before functions and operations inside, so dialect
  // attributes can be used to change dialect-specific global configurations via
  // `amendOperation()`. These configurations can then influence the translation
  // of operations afterwards.
  if (failed(translator.convertOperation(*module, llvmBuilder)))
    return nullptr;

  if (failed(translator.convertComdats()))
    return nullptr;
  if (failed(translator.convertFunctionSignatures()))
    return nullptr;
  if (failed(translator.convertGlobals()))
    return nullptr;
  if (failed(translator.createTBAAMetadata()))
    return nullptr;

  // Convert other top-level operations if possible.
  for (Operation &o : getModuleBody(module).getOperations()) {
    if (!isa<LLVM::LLVMFuncOp, LLVM::GlobalOp, LLVM::GlobalCtorsOp,
             LLVM::GlobalDtorsOp, LLVM::ComdatOp>(&o) &&
        !o.hasTrait<OpTrait::IsTerminator>() &&
        failed(translator.convertOperation(o, llvmBuilder))) {
      return nullptr;
    }
  }

  // Operations in function bodies with symbolic references must be converted
  // after the top-level operations they refer to are declared, so we do it
  // last.
  if (failed(translator.convertFunctions()))
    return nullptr;

  if (llvm::verifyModule(*translator.llvmModule, &llvm::errs()))
    return nullptr;

  return std::move(translator.llvmModule);
}
