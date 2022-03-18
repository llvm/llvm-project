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

#include "DebugTranslation.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "llvm/ADT/TypeSwitch.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
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

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::LLVM::detail;

#include "mlir/Dialect/LLVMIR/LLVMConversionEnumsToLLVM.inc"

/// Translates the given data layout spec attribute to the LLVM IR data layout.
/// Only integer, float and endianness entries are currently supported.
FailureOr<llvm::DataLayout>
translateDataLayout(DataLayoutSpecInterface attribute,
                    const DataLayout &dataLayout,
                    Optional<Location> loc = llvm::None) {
  if (!loc)
    loc = UnknownLoc::get(attribute.getContext());

  // Translate the endianness attribute.
  std::string llvmDataLayout;
  llvm::raw_string_ostream layoutStream(llvmDataLayout);
  for (DataLayoutEntryInterface entry : attribute.getEntries()) {
    auto key = entry.getKey().dyn_cast<StringAttr>();
    if (!key)
      continue;
    if (key.getValue() == DLTIDialect::kDataLayoutEndiannessKey) {
      auto value = entry.getValue().cast<StringAttr>();
      bool isLittleEndian =
          value.getValue() == DLTIDialect::kDataLayoutEndiannessLittle;
      layoutStream << (isLittleEndian ? "e" : "E");
      layoutStream.flush();
      continue;
    }
    emitError(*loc) << "unsupported data layout key " << key;
    return failure();
  }

  // Go through the list of entries to check which types are explicitly
  // specified in entries. Don't use the entries directly though but query the
  // data from the layout.
  for (DataLayoutEntryInterface entry : attribute.getEntries()) {
    auto type = entry.getKey().dyn_cast<Type>();
    if (!type)
      continue;
    // Data layout for the index type is irrelevant at this point.
    if (type.isa<IndexType>())
      continue;
    FailureOr<std::string> prefix =
        llvm::TypeSwitch<Type, FailureOr<std::string>>(type)
            .Case<IntegerType>(
                [loc](IntegerType integerType) -> FailureOr<std::string> {
                  if (integerType.getSignedness() == IntegerType::Signless)
                    return std::string("i");
                  emitError(*loc)
                      << "unsupported data layout for non-signless integer "
                      << integerType;
                  return failure();
                })
            .Case<Float16Type, Float32Type, Float64Type, Float80Type,
                  Float128Type>([](Type) { return std::string("f"); })
            .Default([loc](Type type) -> FailureOr<std::string> {
              emitError(*loc) << "unsupported type in data layout: " << type;
              return failure();
            });
    if (failed(prefix))
      return failure();

    unsigned size = dataLayout.getTypeSizeInBits(type);
    unsigned abi = dataLayout.getTypeABIAlignment(type) * 8u;
    unsigned preferred = dataLayout.getTypePreferredAlignment(type) * 8u;
    layoutStream << "-" << *prefix << size << ":" << abi;
    if (abi != preferred)
      layoutStream << ":" << preferred;
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

  // Compute the shape of all dimensions but the innermost. Note that the
  // innermost dimension may be that of the vector element type.
  bool hasVectorElementType = type.getElementType().isa<VectorType>();
  unsigned numAggregates =
      denseElementsAttr.getNumElements() /
      (hasVectorElementType ? 1
                            : denseElementsAttr.getType().getShape().back());
  ArrayRef<int64_t> outerShape = type.getShape();
  if (!hasVectorElementType)
    outerShape = outerShape.drop_back();

  // Handle the case of vector splat, LLVM has special support for it.
  if (denseElementsAttr.isSplat() &&
      (type.isa<VectorType>() || hasVectorElementType)) {
    llvm::Constant *splatValue = LLVM::detail::getLLVMConstant(
        innermostLLVMType, denseElementsAttr.getSplatValue<Attribute>(), loc,
        moduleTranslation, /*isTopLevel=*/false);
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
  if (type.isa<TensorType>()) {
    auto vectorElementType = type.getElementType().dyn_cast<VectorType>();
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
  } else if (type.isa<VectorType>()) {
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
    const ModuleTranslation &moduleTranslation, bool isTopLevel) {
  if (!attr)
    return llvm::UndefValue::get(llvmType);
  if (auto *structType = dyn_cast<::llvm::StructType>(llvmType)) {
    if (!isTopLevel) {
      emitError(loc, "nested struct types are not supported in constants");
      return nullptr;
    }
    auto arrayAttr = attr.cast<ArrayAttr>();
    llvm::Type *elementType = structType->getElementType(0);
    llvm::Constant *real = getLLVMConstant(elementType, arrayAttr[0], loc,
                                           moduleTranslation, false);
    if (!real)
      return nullptr;
    llvm::Constant *imag = getLLVMConstant(elementType, arrayAttr[1], loc,
                                           moduleTranslation, false);
    if (!imag)
      return nullptr;
    return llvm::ConstantStruct::get(structType, {real, imag});
  }
  // For integer types, we allow a mismatch in sizes as the index type in
  // MLIR might have a different size than the index type in the LLVM module.
  if (auto intAttr = attr.dyn_cast<IntegerAttr>())
    return llvm::ConstantInt::get(
        llvmType,
        intAttr.getValue().sextOrTrunc(llvmType->getIntegerBitWidth()));
  if (auto floatAttr = attr.dyn_cast<FloatAttr>()) {
    if (llvmType !=
        llvm::Type::getFloatingPointTy(llvmType->getContext(),
                                       floatAttr.getValue().getSemantics())) {
      emitError(loc, "FloatAttr does not match expected type of the constant");
      return nullptr;
    }
    return llvm::ConstantFP::get(llvmType, floatAttr.getValue());
  }
  if (auto funcAttr = attr.dyn_cast<FlatSymbolRefAttr>())
    return llvm::ConstantExpr::getBitCast(
        moduleTranslation.lookupFunction(funcAttr.getValue()), llvmType);
  if (auto splatAttr = attr.dyn_cast<SplatElementsAttr>()) {
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
        loc, moduleTranslation, false);
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
          convertDenseElementsAttr(loc, attr.dyn_cast<DenseElementsAttr>(),
                                   llvmType, moduleTranslation)) {
    return result;
  }

  // Fall back to element-by-element construction otherwise.
  if (auto elementsAttr = attr.dyn_cast<ElementsAttr>()) {
    assert(elementsAttr.getType().hasStaticShape());
    assert(!elementsAttr.getType().getShape().empty() &&
           "unexpected empty elements attribute shape");

    SmallVector<llvm::Constant *, 8> constants;
    constants.reserve(elementsAttr.getNumElements());
    llvm::Type *innermostType = getInnermostElementType(llvmType);
    for (auto n : elementsAttr.getValues<Attribute>()) {
      constants.push_back(
          getLLVMConstant(innermostType, n, loc, moduleTranslation, false));
      if (!constants.back())
        return nullptr;
    }
    ArrayRef<llvm::Constant *> constantsRef = constants;
    llvm::Constant *result = buildSequentialConstant(
        constantsRef, elementsAttr.getType().getShape(), llvmType, loc);
    assert(constantsRef.empty() && "did not consume all elemental constants");
    return result;
  }

  if (auto stringAttr = attr.dyn_cast<StringAttr>()) {
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
        accessGroupMetadataMapping.erase(&op);
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
    Optional<OperandRange> successorOperands = branch.getSuccessorOperands(i);
    assert(
        (!seenSuccessors.contains(successor) ||
         (successorOperands && successorOperands->empty())) &&
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
  for (auto it = std::next(region.begin()), eit = region.end(); it != eit;
       ++it) {
    Block *bb = &*it;
    llvm::BasicBlock *llvmBB = state.lookupBlock(bb);
    auto phis = llvmBB->phis();
    auto numArguments = bb->getNumArguments();
    assert(numArguments == std::distance(phis.begin(), phis.end()));
    for (auto &numberedPhiNode : llvm::enumerate(phis)) {
      auto &phiNode = numberedPhiNode.value();
      unsigned index = numberedPhiNode.index();
      for (auto *pred : bb->getPredecessors()) {
        // Find the LLVM IR block that contains the converted terminator
        // instruction and use it in the PHI node. Note that this block is not
        // necessarily the same as state.lookupBlock(pred), some operations
        // (in particular, OpenMP operations using OpenMPIRBuilder) may have
        // split the blocks.
        llvm::Instruction *terminator =
            state.lookupBranch(pred->getTerminator());
        assert(terminator && "missing the mapping for a terminator");
        phiNode.addIncoming(
            state.lookupValue(getPHISourceValue(bb, pred, numArguments, index)),
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

llvm::Value *mlir::LLVM::detail::createIntrinsicCall(
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
      if (auto strAttr = op.getValueOrNull().dyn_cast_or_null<StringAttr>()) {
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
        /*InsertBefore=*/nullptr, llvm::GlobalValue::NotThreadLocal, addrSpace);

    if (op.getUnnamedAddr().hasValue())
      var->setUnnamedAddr(convertUnnamedAddrToLLVM(*op.getUnnamedAddr()));

    if (op.getSection().hasValue())
      var->setSection(*op.getSection());

    addRuntimePreemptionSpecifier(op.getDsoLocal(), var);

    Optional<uint64_t> alignment = op.getAlignment();
    if (alignment.hasValue())
      var->setAlignment(llvm::MaybeAlign(alignment.getValue()));

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
          std::get<0>(symbolAndPriority).cast<FlatSymbolRefAttr>().getValue());
      appendGlobalFn(
          *llvmModule, f,
          std::get<1>(symbolAndPriority).cast<IntegerAttr>().getInt(),
          /*Data=*/nullptr);
    }
  }

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

    int result;
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
forwardPassthroughAttributes(Location loc, Optional<ArrayAttr> attributes,
                             llvm::Function *llvmFunc) {
  if (!attributes)
    return success();

  for (Attribute attr : *attributes) {
    if (auto stringAttr = attr.dyn_cast<StringAttr>()) {
      if (failed(
              checkedAddLLVMFnAttribute(loc, llvmFunc, stringAttr.getValue())))
        return failure();
      continue;
    }

    auto arrayAttr = attr.dyn_cast<ArrayAttr>();
    if (!arrayAttr || arrayAttr.size() != 2)
      return emitError(loc)
             << "expected 'passthrough' to contain string or array attributes";

    auto keyAttr = arrayAttr[0].dyn_cast<StringAttr>();
    auto valueAttr = arrayAttr[1].dyn_cast<StringAttr>();
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
  // If there was noalias info then we decorate each argument accordingly.
  unsigned int argIdx = 0;
  for (auto kvp : llvm::zip(func.getArguments(), llvmFunc->args())) {
    llvm::Argument &llvmArg = std::get<1>(kvp);
    BlockArgument mlirArg = std::get<0>(kvp);

    if (auto attr = func.getArgAttrOfType<UnitAttr>(
            argIdx, LLVMDialect::getNoAliasAttrName())) {
      // NB: Attribute already verified to be boolean, so check if we can indeed
      // attach the attribute to this argument, based on its type.
      auto argTy = mlirArg.getType();
      if (!argTy.isa<LLVM::LLVMPointerType>())
        return func.emitError(
            "llvm.noalias attribute attached to LLVM non-pointer argument");
      llvmArg.addAttr(llvm::Attribute::AttrKind::NoAlias);
    }

    if (auto attr = func.getArgAttrOfType<IntegerAttr>(
            argIdx, LLVMDialect::getAlignAttrName())) {
      // NB: Attribute already verified to be int, so check if we can indeed
      // attach the attribute to this argument, based on its type.
      auto argTy = mlirArg.getType();
      if (!argTy.isa<LLVM::LLVMPointerType>())
        return func.emitError(
            "llvm.align attribute attached to LLVM non-pointer argument");
      llvmArg.addAttrs(llvm::AttrBuilder(llvmArg.getContext())
                           .addAlignmentAttr(llvm::Align(attr.getInt())));
    }

    if (auto attr = func.getArgAttrOfType<UnitAttr>(argIdx, "llvm.sret")) {
      auto argTy = mlirArg.getType().dyn_cast<LLVM::LLVMPointerType>();
      if (!argTy)
        return func.emitError(
            "llvm.sret attribute attached to LLVM non-pointer argument");
      llvmArg.addAttrs(
          llvm::AttrBuilder(llvmArg.getContext())
              .addStructRetAttr(convertType(argTy.getElementType())));
    }

    if (auto attr = func.getArgAttrOfType<UnitAttr>(argIdx, "llvm.byval")) {
      auto argTy = mlirArg.getType().dyn_cast<LLVM::LLVMPointerType>();
      if (!argTy)
        return func.emitError(
            "llvm.byval attribute attached to LLVM non-pointer argument");
      llvmArg.addAttrs(llvm::AttrBuilder(llvmArg.getContext())
                           .addByValAttr(convertType(argTy.getElementType())));
    }

    if (auto attr = func.getArgAttrOfType<UnitAttr>(argIdx, "llvm.nest")) {
      auto argTy = mlirArg.getType();
      if (!argTy.isa<LLVM::LLVMPointerType>())
        return func.emitError(
            "llvm.nest attribute attached to LLVM non-pointer argument");
      llvmArg.addAttrs(llvm::AttrBuilder(llvmArg.getContext())
                           .addAttribute(llvm::Attribute::Nest));
    }

    mapValue(mlirArg, &llvmArg);
    argIdx++;
  }

  // Check the personality and set it.
  if (func.getPersonality().hasValue()) {
    llvm::Type *ty = llvm::Type::getInt8PtrTy(llvmFunc->getContext());
    if (llvm::Constant *pfunc = getLLVMConstant(ty, func.getPersonalityAttr(),
                                                func.getLoc(), *this))
      llvmFunc->setPersonalityFn(pfunc);
  }

  if (auto gc = func.getGarbageCollector())
    llvmFunc->setGC(gc->str());

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

LogicalResult ModuleTranslation::convertFunctionSignatures() {
  // Declare all functions first because there may be function calls that form a
  // call graph with cycles, or global initializers that reference functions.
  for (auto function : getModuleBody(mlirModule).getOps<LLVMFuncOp>()) {
    llvm::FunctionCallee llvmFuncCst = llvmModule->getOrInsertFunction(
        function.getName(),
        cast<llvm::FunctionType>(convertType(function.getFunctionType())));
    llvm::Function *llvmFunc = cast<llvm::Function>(llvmFuncCst.getCallee());
    llvmFunc->setLinkage(convertLinkageToLLVM(function.getLinkage()));
    mapFunction(function.getName(), llvmFunc);
    addRuntimePreemptionSpecifier(function.getDsoLocal(), llvmFunc);

    // Forward the pass-through attributes to LLVM.
    if (failed(forwardPassthroughAttributes(
            function.getLoc(), function.getPassthrough(), llvmFunc)))
      return failure();
  }

  return success();
}

LogicalResult ModuleTranslation::convertFunctions() {
  // Convert functions.
  for (auto function : getModuleBody(mlirModule).getOps<LLVMFuncOp>()) {
    // Ignore external functions.
    if (function.isExternal())
      continue;

    if (failed(convertOneFunction(function)))
      return failure();
  }

  return success();
}

llvm::MDNode *
ModuleTranslation::getAccessGroup(Operation &opInst,
                                  SymbolRefAttr accessGroupRef) const {
  auto metadataName = accessGroupRef.getRootReference();
  auto accessGroupName = accessGroupRef.getLeafReference();
  auto metadataOp = SymbolTable::lookupNearestSymbolFrom<LLVM::MetadataOp>(
      opInst.getParentOp(), metadataName);
  auto *accessGroupOp =
      SymbolTable::lookupNearestSymbolFrom(metadataOp, accessGroupName);
  return accessGroupMetadataMapping.lookup(accessGroupOp);
}

LogicalResult ModuleTranslation::createAccessGroupMetadata() {
  mlirModule->walk([&](LLVM::MetadataOp metadatas) {
    metadatas.walk([&](LLVM::AccessGroupMetadataOp op) {
      llvm::LLVMContext &ctx = llvmModule->getContext();
      llvm::MDNode *accessGroup = llvm::MDNode::getDistinct(ctx, {});
      accessGroupMetadataMapping.insert({op, accessGroup});
    });
  });
  return success();
}

void ModuleTranslation::setAccessGroupsMetadata(Operation *op,
                                                llvm::Instruction *inst) {
  auto accessGroups =
      op->getAttrOfType<ArrayAttr>(LLVMDialect::getAccessGroupsAttrName());
  if (accessGroups && !accessGroups.empty()) {
    llvm::Module *module = inst->getModule();
    SmallVector<llvm::Metadata *> metadatas;
    for (SymbolRefAttr accessGroupRef :
         accessGroups.getAsRange<SymbolRefAttr>())
      metadatas.push_back(getAccessGroup(*op, accessGroupRef));

    llvm::MDNode *unionMD = nullptr;
    if (metadatas.size() == 1)
      unionMD = llvm::cast<llvm::MDNode>(metadatas.front());
    else if (metadatas.size() >= 2)
      unionMD = llvm::MDNode::get(module->getContext(), metadatas);

    inst->setMetadata(module->getMDKindID("llvm.access.group"), unionMD);
  }
}

LogicalResult ModuleTranslation::createAliasScopeMetadata() {
  mlirModule->walk([&](LLVM::MetadataOp metadatas) {
    // Create the domains first, so they can be reference below in the scopes.
    DenseMap<Operation *, llvm::MDNode *> aliasScopeDomainMetadataMapping;
    metadatas.walk([&](LLVM::AliasScopeDomainMetadataOp op) {
      llvm::LLVMContext &ctx = llvmModule->getContext();
      llvm::SmallVector<llvm::Metadata *, 2> operands;
      operands.push_back({}); // Placeholder for self-reference
      if (Optional<StringRef> description = op.getDescription())
        operands.push_back(llvm::MDString::get(ctx, description.getValue()));
      llvm::MDNode *domain = llvm::MDNode::get(ctx, operands);
      domain->replaceOperandWith(0, domain); // Self-reference for uniqueness
      aliasScopeDomainMetadataMapping.insert({op, domain});
    });

    // Now create the scopes, referencing the domains created above.
    metadatas.walk([&](LLVM::AliasScopeMetadataOp op) {
      llvm::LLVMContext &ctx = llvmModule->getContext();
      assert(isa<LLVM::MetadataOp>(op->getParentOp()));
      auto metadataOp = dyn_cast<LLVM::MetadataOp>(op->getParentOp());
      Operation *domainOp =
          SymbolTable::lookupNearestSymbolFrom(metadataOp, op.getDomainAttr());
      llvm::MDNode *domain = aliasScopeDomainMetadataMapping.lookup(domainOp);
      assert(domain && "Scope's domain should already be valid");
      llvm::SmallVector<llvm::Metadata *, 3> operands;
      operands.push_back({}); // Placeholder for self-reference
      operands.push_back(domain);
      if (Optional<StringRef> description = op.getDescription())
        operands.push_back(llvm::MDString::get(ctx, description.getValue()));
      llvm::MDNode *scope = llvm::MDNode::get(ctx, operands);
      scope->replaceOperandWith(0, scope); // Self-reference for uniqueness
      aliasScopeMetadataMapping.insert({op, scope});
    });
  });
  return success();
}

llvm::MDNode *
ModuleTranslation::getAliasScope(Operation &opInst,
                                 SymbolRefAttr aliasScopeRef) const {
  StringAttr metadataName = aliasScopeRef.getRootReference();
  StringAttr scopeName = aliasScopeRef.getLeafReference();
  auto metadataOp = SymbolTable::lookupNearestSymbolFrom<LLVM::MetadataOp>(
      opInst.getParentOp(), metadataName);
  Operation *aliasScopeOp =
      SymbolTable::lookupNearestSymbolFrom(metadataOp, scopeName);
  return aliasScopeMetadataMapping.lookup(aliasScopeOp);
}

void ModuleTranslation::setAliasScopeMetadata(Operation *op,
                                              llvm::Instruction *inst) {
  auto populateScopeMetadata = [this, op, inst](StringRef attrName,
                                                StringRef llvmMetadataName) {
    auto scopes = op->getAttrOfType<ArrayAttr>(attrName);
    if (!scopes || scopes.empty())
      return;
    llvm::Module *module = inst->getModule();
    SmallVector<llvm::Metadata *> scopeMDs;
    for (SymbolRefAttr scopeRef : scopes.getAsRange<SymbolRefAttr>())
      scopeMDs.push_back(getAliasScope(*op, scopeRef));
    llvm::MDNode *unionMD = llvm::MDNode::get(module->getContext(), scopeMDs);
    inst->setMetadata(module->getMDKindID(llvmMetadataName), unionMD);
  };

  populateScopeMetadata(LLVMDialect::getAliasScopesAttrName(), "alias.scope");
  populateScopeMetadata(LLVMDialect::getNoAliasScopesAttrName(), "noalias");
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

const llvm::DILocation *
ModuleTranslation::translateLoc(Location loc, llvm::DILocalScope *scope) {
  return debugTranslation->translateLoc(loc, scope);
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
          m->getAttr(LLVM::LLVMDialect::getDataLayoutAttrName())) {
    llvmModule->setDataLayout(dataLayoutAttr.cast<StringAttr>().getValue());
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
          m->getAttr(LLVM::LLVMDialect::getTargetTripleAttrName()))
    llvmModule->setTargetTriple(targetTripleAttr.cast<StringAttr>().getValue());

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
  if (!satisfiesLLVMModule(module))
    return nullptr;

  std::unique_ptr<llvm::Module> llvmModule =
      prepareLLVMModule(module, llvmContext, name);
  if (!llvmModule)
    return nullptr;

  LLVM::ensureDistinctSuccessors(module);

  ModuleTranslation translator(module, std::move(llvmModule));
  if (failed(translator.convertFunctionSignatures()))
    return nullptr;
  if (failed(translator.convertGlobals()))
    return nullptr;
  if (failed(translator.createAccessGroupMetadata()))
    return nullptr;
  if (failed(translator.createAliasScopeMetadata()))
    return nullptr;
  if (failed(translator.convertFunctions()))
    return nullptr;

  // Convert other top-level operations if possible.
  llvm::IRBuilder<> llvmBuilder(llvmContext);
  for (Operation &o : getModuleBody(module).getOperations()) {
    if (!isa<LLVM::LLVMFuncOp, LLVM::GlobalOp, LLVM::GlobalCtorsOp,
             LLVM::GlobalDtorsOp, LLVM::MetadataOp>(&o) &&
        !o.hasTrait<OpTrait::IsTerminator>() &&
        failed(translator.convertOperation(o, llvmBuilder))) {
      return nullptr;
    }
  }

  if (llvm::verifyModule(*translator.llvmModule, &llvm::errs()))
    return nullptr;

  return std::move(translator.llvmModule);
}
