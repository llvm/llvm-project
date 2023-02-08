//===- LLVMIRToLLVMTranslation.cpp - Translate LLVM IR to LLVM dialect ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between LLVM IR and the MLIR LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/ModuleImport.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/ModRef.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::LLVM::detail;

#include "mlir/Dialect/LLVMIR/LLVMConversionEnumsFromLLVM.inc"

/// Returns true if the LLVM IR intrinsic is convertible to an MLIR LLVM dialect
/// intrinsic. Returns false otherwise.
static bool isConvertibleIntrinsic(llvm::Intrinsic::ID id) {
  static const DenseSet<unsigned> convertibleIntrinsics = {
#include "mlir/Dialect/LLVMIR/LLVMConvertibleLLVMIRIntrinsics.inc"
  };
  return convertibleIntrinsics.contains(id);
}

/// Returns the list of LLVM IR intrinsic identifiers that are convertible to
/// MLIR LLVM dialect intrinsics.
static ArrayRef<unsigned> getSupportedIntrinsicsImpl() {
  static const SmallVector<unsigned> convertibleIntrinsics = {
#include "mlir/Dialect/LLVMIR/LLVMConvertibleLLVMIRIntrinsics.inc"
  };
  return convertibleIntrinsics;
}

/// Converts the LLVM intrinsic to an MLIR LLVM dialect operation if a
/// conversion exits. Returns failure otherwise.
static LogicalResult convertIntrinsicImpl(OpBuilder &odsBuilder,
                                          llvm::CallInst *inst,
                                          LLVM::ModuleImport &moduleImport) {
  llvm::Intrinsic::ID intrinsicID = inst->getIntrinsicID();

  // Check if the intrinsic is convertible to an MLIR dialect counterpart and
  // copy the arguments to an an LLVM operands array reference for conversion.
  if (isConvertibleIntrinsic(intrinsicID)) {
    SmallVector<llvm::Value *> args(inst->args());
    ArrayRef<llvm::Value *> llvmOperands(args);
#include "mlir/Dialect/LLVMIR/LLVMIntrinsicFromLLVMIRConversions.inc"
  }

  return failure();
}

/// Returns the list of LLVM IR metadata kinds that are convertible to MLIR LLVM
/// dialect attributes.
static ArrayRef<unsigned> getSupportedMetadataImpl() {
  static const SmallVector<unsigned> convertibleMetadata = {
      llvm::LLVMContext::MD_prof, llvm::LLVMContext::MD_tbaa,
      llvm::LLVMContext::MD_access_group, llvm::LLVMContext::MD_loop};
  return convertibleMetadata;
}

/// Converts the given profiling metadata `node` to an MLIR profiling attribute
/// and attaches it to the imported operation if the translation succeeds.
/// Returns failure otherwise.
static LogicalResult setProfilingAttr(OpBuilder &builder, llvm::MDNode *node,
                                      Operation *op,
                                      LLVM::ModuleImport &moduleImport) {
  // Return success for empty metadata nodes since there is nothing to import.
  if (!node->getNumOperands())
    return op->emitWarning() << "expected non-empty profiling metadata node";

  auto *name = dyn_cast<llvm::MDString>(node->getOperand(0));
  if (!name)
    return op->emitWarning()
           << "expected profiling metadata node to have a string identifier";

  // Handle function entry count metadata.
  if (name->getString().equals("function_entry_count")) {
    auto emitNodeWarning = [&]() {
      return op->emitWarning()
             << "expected function_entry_count to hold a single i64 value";
    };

    // TODO support function entry count metadata with GUID fields.
    if (node->getNumOperands() != 2)
      return emitNodeWarning();

    llvm::ConstantInt *entryCount =
        llvm::mdconst::dyn_extract<llvm::ConstantInt>(node->getOperand(1));
    if (!entryCount)
      return emitNodeWarning();
    if (auto funcOp = dyn_cast<LLVMFuncOp>(op)) {
      funcOp.setFunctionEntryCount(entryCount->getZExtValue());
      return success();
    }
    return op->emitWarning()
           << "expected function_entry_count to be attached to a function";
  }

  if (!name->getString().equals("branch_weights"))
    return op->emitWarning()
           << "unknown profiling metadata node " << name->getString();

  // Handle branch weights metadata.
  SmallVector<int32_t> branchWeights;
  branchWeights.reserve(node->getNumOperands() - 1);
  for (unsigned i = 1, e = node->getNumOperands(); i != e; ++i) {
    llvm::ConstantInt *branchWeight =
        llvm::mdconst::dyn_extract<llvm::ConstantInt>(node->getOperand(i));
    if (!branchWeight)
      return op->emitWarning() << "expected branch weights to be integers";
    branchWeights.push_back(branchWeight->getZExtValue());
  }

  // Attach the branch weights to the operations that support it.
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<CondBrOp, SwitchOp, CallOp, InvokeOp>([&](auto branchWeightOp) {
        branchWeightOp.setBranchWeightsAttr(
            builder.getI32VectorAttr(branchWeights));
        return success();
      })
      .Default([op](auto) {
        return op->emitWarning()
               << op->getName() << " does not support branch weights";
      });
}

/// Searches the symbol reference pointing to the metadata operation that
/// maps to the given TBAA metadata `node` and attaches it to the imported
/// operation if the lookup succeeds. Returns failure otherwise.
static LogicalResult setTBAAAttr(const llvm::MDNode *node, Operation *op,
                                 LLVM::ModuleImport &moduleImport) {
  SymbolRefAttr tbaaTagSym = moduleImport.lookupTBAAAttr(node);
  if (!tbaaTagSym)
    return failure();

  op->setAttr(LLVMDialect::getTBAAAttrName(),
              ArrayAttr::get(op->getContext(), tbaaTagSym));
  return success();
}

/// Looks up all the symbol references pointing to the access group operations
/// that map to the access group nodes starting from the access group metadata
/// `node`, and attaches all of them to the imported operation if the lookups
/// succeed. Returns failure otherwise.
static LogicalResult setAccessGroupAttr(const llvm::MDNode *node, Operation *op,
                                        LLVM::ModuleImport &moduleImport) {
  FailureOr<SmallVector<SymbolRefAttr>> accessGroups =
      moduleImport.lookupAccessGroupAttrs(node);
  if (failed(accessGroups))
    return failure();

  SmallVector<Attribute> accessGroupAttrs(accessGroups->begin(),
                                          accessGroups->end());
  op->setAttr(LLVMDialect::getAccessGroupsAttrName(),
              ArrayAttr::get(op->getContext(), accessGroupAttrs));
  return success();
}

/// Converts the given loop metadata node to an MLIR loop annotation attribute
/// and attaches it to the imported operation if the translation succeeds.
/// Returns failure otherwise.
static LogicalResult setLoopAttr(const llvm::MDNode *node, Operation *op,
                                 LLVM::ModuleImport &moduleImport) {
  LoopAnnotationAttr attr =
      moduleImport.translateLoopAnnotationAttr(node, op->getLoc());
  if (!attr)
    return failure();

  op->setAttr(LLVMDialect::getLoopAttrName(), attr);
  return success();
}

namespace {

/// Implementation of the dialect interface that converts operations belonging
/// to the LLVM dialect to LLVM IR.
class LLVMDialectLLVMIRImportInterface : public LLVMImportDialectInterface {
public:
  using LLVMImportDialectInterface::LLVMImportDialectInterface;

  /// Converts the LLVM intrinsic to an MLIR LLVM dialect operation if a
  /// conversion exits. Returns failure otherwise.
  LogicalResult convertIntrinsic(OpBuilder &builder, llvm::CallInst *inst,
                                 LLVM::ModuleImport &moduleImport) const final {
    return convertIntrinsicImpl(builder, inst, moduleImport);
  }

  /// Attaches the given LLVM metadata to the imported operation if a conversion
  /// to an LLVM dialect attribute exists and succeeds. Returns failure
  /// otherwise.
  LogicalResult setMetadataAttrs(OpBuilder &builder, unsigned kind,
                                 llvm::MDNode *node, Operation *op,
                                 LLVM::ModuleImport &moduleImport) const final {
    // Call metadata specific handlers.
    if (kind == llvm::LLVMContext::MD_prof)
      return setProfilingAttr(builder, node, op, moduleImport);
    if (kind == llvm::LLVMContext::MD_tbaa)
      return setTBAAAttr(node, op, moduleImport);
    if (kind == llvm::LLVMContext::MD_access_group)
      return setAccessGroupAttr(node, op, moduleImport);
    if (kind == llvm::LLVMContext::MD_loop)
      return setLoopAttr(node, op, moduleImport);

    // A handler for a supported metadata kind is missing.
    llvm_unreachable("unknown metadata type");
  }

  /// Returns the list of LLVM IR intrinsic identifiers that are convertible to
  /// MLIR LLVM dialect intrinsics.
  ArrayRef<unsigned> getSupportedIntrinsics() const final {
    return getSupportedIntrinsicsImpl();
  }

  /// Returns the list of LLVM IR metadata kinds that are convertible to MLIR
  /// LLVM dialect attributes.
  ArrayRef<unsigned> getSupportedMetadata() const final {
    return getSupportedMetadataImpl();
  }
};
} // namespace

void mlir::registerLLVMDialectImport(DialectRegistry &registry) {
  registry.insert<LLVM::LLVMDialect>();
  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    dialect->addInterfaces<LLVMDialectLLVMIRImportInterface>();
  });
}

void mlir::registerLLVMDialectImport(MLIRContext &context) {
  DialectRegistry registry;
  registerLLVMDialectImport(registry);
  context.appendDialectRegistry(registry);
}
