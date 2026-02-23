//===-- XeVMToLLVMIRTranslation.cpp - Translate XeVM to LLVM IR -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR XeVM dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/XeVM/XeVMToLLVMIRTranslation.h"
#include "mlir/Dialect/LLVMIR/XeVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {
//===----------------------------------------------------------------------===//
// Utility functions for the translation
//===----------------------------------------------------------------------===//
// Extract the source filename from the debug location of \p inst, if available.
static std::string getSourceFilename(const llvm::Instruction *inst) {
  if (const llvm::DebugLoc &dbgLoc = inst->getDebugLoc()) {
    if (auto *loc = dbgLoc.get()) {
      if (llvm::DIFile *file = loc->getFile()) {
        if (!file->getDirectory().empty())
          return (file->getDirectory() + "/" + file->getFilename()).str();
        return file->getFilename().str();
      }
    }
  }
  return "";
}

// Build one cache-control payload string per attribute.
//
// Each mlir::Attribute is expected to be an ArrayAttr of (at least) 3
// IntegerAttr values: [SPIR-V token number of that attribute, value for L1
// cache, value for L3 cache].
//
// A single entry produces a string that appears in LLVM IR as:
//   {6442:\220,1\22}\00
static llvm::SmallVector<std::string>
buildCacheControlPayloads(llvm::ArrayRef<mlir::Attribute> attrs) {
  llvm::SmallVector<std::string> payloads;
  llvm::StringMap<bool> seen;

  for (mlir::Attribute a : attrs) {
    auto arr = mlir::dyn_cast<mlir::ArrayAttr>(a);
    if (!arr)
      continue;

    auto vals = arr.getValue();
    // Assert that the attribute has at most 4 integer values: [SPIR-V token, L1
    // value, L3 value, optional extra value].
    assert(vals.size() <= 4 &&
           "Expected at most 4 integer values in cache control attribute.");

    //  Although the caching value is allowed for 3 levels (L1, L2, L3), current
    //  Intel GPUs only have L1, and L3. So we only use L1 and L3 values. The L2
    //  value is ignored.
    auto firstAttr = mlir::dyn_cast<mlir::IntegerAttr>(vals[0]); // Token number
    auto secondAttr = mlir::dyn_cast<mlir::IntegerAttr>(vals[1]); // L1 value
    // L2 value is ignored: vals[2]
    auto thirdAttr = mlir::dyn_cast<mlir::IntegerAttr>(vals[3]); // L3 value

    if (!firstAttr || !secondAttr || !thirdAttr)
      continue;

    uint64_t first = firstAttr.getValue().getZExtValue();
    uint64_t second = secondAttr.getValue().getZExtValue();
    uint64_t third = thirdAttr.getValue().getZExtValue();

    // Produce: {FIRST:\22SECOND,THIRD\22}
    // where \22 is the 0x22 byte ("), which LLVM IR prints as \22.
    // The null terminator (\00) is added by ConstantDataArray::getString.
    std::string entry;
    entry.push_back('{');
    entry += std::to_string(first);
    entry.push_back(':');
    entry.push_back('"'); // 0x22 → prints as \22 in LLVM IR
    entry += std::to_string(second);
    entry.push_back(',');
    entry += std::to_string(third);
    entry.push_back('"'); // 0x22 → prints as \22 in LLVM IR
    entry.push_back('}');

    // Skip duplicates — identical annotations on the same pointer are
    // redundant.
    if (!seen.insert({entry, true}).second)
      continue;

    payloads.push_back(std::move(entry));
  }

  return payloads;
}

// Create (or reuse) an addrspace(1) global string in section "llvm.metadata"
// and return an i8 addrspace(1)* pointer to its first element.
//
// A module-level StringMap caches previously created globals so that
// identical strings are not duplicated.
static llvm::Value *getOrCreateAS1MetadataStringPtr(llvm::Module &module,
                                                    llvm::StringRef str,
                                                    llvm::StringRef nameHint) {
  // Per-module cache keyed on the string content.
  static llvm::DenseMap<llvm::Module *, llvm::StringMap<llvm::Value *>> cache;
  auto &moduleCache = cache[&module];

  auto it = moduleCache.find(str);
  if (it != moduleCache.end())
    return it->second;

  llvm::LLVMContext &ctx = module.getContext();

  llvm::Constant *arr =
      llvm::ConstantDataArray::getString(ctx, str, /*AddNull=*/true);
  auto *arrTy = llvm::cast<llvm::ArrayType>(arr->getType());

  auto *gv = new llvm::GlobalVariable(
      module, arrTy,
      /*isConstant=*/true, llvm::GlobalValue::PrivateLinkage, arr, nameHint,
      /*InsertBefore=*/nullptr, llvm::GlobalValue::NotThreadLocal,
      /*AddressSpace=*/1);

  gv->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
  gv->setSection("llvm.metadata");
  gv->setAlignment(llvm::Align(1));

  llvm::Constant *zero = llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 0);
  llvm::Constant *idxs[] = {zero, zero};

  llvm::Constant *gep =
      llvm::ConstantExpr::getInBoundsGetElementPtr(arrTy, gv, idxs);

  auto *as1PtrTy = llvm::PointerType::get(ctx, /*AddrSpace=*/1);
  llvm::Value *result = llvm::ConstantExpr::getBitCast(gep, as1PtrTy);

  moduleCache[str] = result;
  return result;
}
/// Implementation of the dialect interface that converts operations belonging
/// to the XeVM dialect to LLVM IR.
class XeVMDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Attaches module-level metadata for functions marked as kernels.
  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    StringRef attrName = attribute.getName().getValue();
    if (attrName == mlir::xevm::XeVMDialect::getCacheControlsAttrName()) {
      auto cacheControlsArray = dyn_cast<ArrayAttr>(attribute.getValue());
      if (cacheControlsArray.size() != 2) {
        return op->emitOpError(
            "Expected both L1 and L3 cache control attributes!");
      }
      if (instructions.size() != 1) {
        return op->emitOpError("Expecting a single instruction");
      }
      return handleDecorationCacheControl(instructions.front(),
                                          cacheControlsArray.getValue());
    }
    return success();
  }

private:
  // Attach cache-control metadata to the pointer operand of \p inst.
  //
  // Each attribute in \p attrs becomes a separate
  // llvm.ptr.annotation.p<AS>.p1 call, where AS is the address space of the
  // input pointer.  When there are multiple attributes the calls are chained:
  // the result of the first annotation is fed as the pointer into the second,
  // and so on, so that every annotation is associated with the same logical
  // pointer.
  //
  // Example:
  //   ; For a ptr addrspace(1) operand -> llvm.ptr.annotation.p1.p1
  //   %a1 = call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(
  //              ptr addrspace(1) %ptr, ... "{6442:\220,1\22}" ...)
  //   %a2 = call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(
  //              ptr addrspace(1) %a1,  ... "{6442:\221,1\22}" ...)
  //   ; instruction now uses %a2
  //
  //   ; For a ptr addrspace(0) operand -> llvm.ptr.annotation.p0.p1
  //   %a1 = call ptr @llvm.ptr.annotation.p0.p1(
  //              ptr %ptr, ... "{6442:\220,1\22}" ...)
  static mlir::LogicalResult
  handleDecorationCacheControl(llvm::Instruction *inst,
                               llvm::ArrayRef<mlir::Attribute> attrs) {
    llvm::Module *module = inst->getModule();
    if (!module)
      return mlir::failure();

    // @TODO: Expect the pointer to be the first operand, but this is not a
    // strict requirement of the protocol; it can be adjusted if needed.
    constexpr unsigned ptrIdx = 0;
    llvm::Value *ptr = inst->getOperand(ptrIdx);

    if (!ptr || !ptr->getType()->isPointerTy())
      return mlir::success();

    auto *ptrTy = llvm::cast<llvm::PointerType>(ptr->getType());

    llvm::SmallVector<std::string> payloads = buildCacheControlPayloads(attrs);
    if (payloads.empty())
      return mlir::success();

    llvm::LLVMContext &ctx = module->getContext();
    llvm::IRBuilder<> builder(inst);
    auto *as1PtrTy = llvm::PointerType::get(ctx, 1);

    // Shared across all annotations for this instruction.
    std::string fileName = getSourceFilename(inst);
    llvm::Value *fileStrAS1 = getOrCreateAS1MetadataStringPtr(
        *module,
        fileName.empty() ? llvm::StringRef("") : llvm::StringRef(fileName),
        ".str.file");

    unsigned line = 0;
    if (llvm::DILocation *loc = inst->getDebugLoc().get())
      line = loc->getLine();
    llvm::Value *lineVal =
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), line);

    llvm::Value *nullAS1 =
        llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(as1PtrTy));

    unsigned ptrAS = ptrTy->getAddressSpace();

    llvm::FunctionType *fTy = llvm::FunctionType::get(
        ptrTy,
        {ptrTy, as1PtrTy, as1PtrTy, llvm::Type::getInt32Ty(ctx), as1PtrTy},
        /*isVarArg=*/false);

    // llvm.ptr.annotation.p<ptrAS>.p1
    std::string intrinsicName =
        "llvm.ptr.annotation.p" + std::to_string(ptrAS) + ".p1";
    llvm::FunctionCallee callee =
        module->getOrInsertFunction(intrinsicName, fTy);

    // Chain: each annotation takes the result of the previous one as its
    // pointer operand.
    llvm::Value *curPtr = ptr;
    for (const std::string &payload : payloads) {
      llvm::Value *annStrAS1 = getOrCreateAS1MetadataStringPtr(
          *module, payload, ".str.cachecontrol");

      curPtr = builder.CreateCall(
          callee, {curPtr, annStrAS1, fileStrAS1, lineVal, nullAS1},
          "annotated.ptr");
    }

    inst->setOperand(ptrIdx, curPtr);
    return mlir::success();
  }
};
} // namespace

void mlir::registerXeVMDialectTranslation(::mlir::DialectRegistry &registry) {
  registry.insert<xevm::XeVMDialect>();
  registry.addExtension(+[](MLIRContext *ctx, xevm::XeVMDialect *dialect) {
    dialect->addInterfaces<XeVMDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerXeVMDialectTranslation(::mlir::MLIRContext &context) {
  DialectRegistry registry;
  registerXeVMDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
