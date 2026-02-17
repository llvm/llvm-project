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
// Extract the source filename from the debug location of \p Inst, if available.
static std::string getSourceFilename(const llvm::Instruction *Inst) {
  if (const llvm::DebugLoc &DbgLoc = Inst->getDebugLoc()) {
    if (auto *Loc = DbgLoc.get()) {
      if (llvm::DIFile *File = Loc->getFile()) {
        if (!File->getDirectory().empty())
          return (File->getDirectory() + "/" + File->getFilename()).str();
        return File->getFilename().str();
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
buildCacheControlPayloads(llvm::ArrayRef<mlir::Attribute> Attrs) {
  llvm::SmallVector<std::string> Payloads;
  llvm::StringMap<bool> Seen;

  for (mlir::Attribute A : Attrs) {
    auto Arr = mlir::dyn_cast<mlir::ArrayAttr>(A);
    if (!Arr)
      continue;

    auto Vals = Arr.getValue();
    // Assert that the attribute has exactly 3 integer values: [SPIR-V token, L1
    // value, L3 value].
    llvm::assert(Vals.size() == 3 &&
                 "Expected 3 integer values in cache control attribute.");

    auto FirstAttr = mlir::dyn_cast<mlir::IntegerAttr>(Vals[0]);
    auto SecondAttr = mlir::dyn_cast<mlir::IntegerAttr>(Vals[1]);
    auto ThirdAttr = mlir::dyn_cast<mlir::IntegerAttr>(Vals[2]);

    if (!FirstAttr || !SecondAttr || !ThirdAttr)
      continue;

    uint64_t First = FirstAttr.getValue().getZExtValue();
    uint64_t Second = SecondAttr.getValue().getZExtValue();
    uint64_t Third = ThirdAttr.getValue().getZExtValue();

    // Produce: {FIRST:\22SECOND,THIRD\22}
    // where \22 is the 0x22 byte ("), which LLVM IR prints as \22.
    // The null terminator (\00) is added by ConstantDataArray::getString.
    std::string Entry;
    Entry.push_back('{');
    Entry += std::to_string(First);
    Entry.push_back(':');
    Entry.push_back('"'); // 0x22 → prints as \22 in LLVM IR
    Entry += std::to_string(Second);
    Entry.push_back(',');
    Entry += std::to_string(Third);
    Entry.push_back('"'); // 0x22 → prints as \22 in LLVM IR
    Entry.push_back('}');

    // Skip duplicates — identical annotations on the same pointer are
    // redundant.
    if (!Seen.insert({Entry, true}).second)
      continue;

    Payloads.push_back(std::move(Entry));
  }

  return Payloads;
}

// Create (or reuse) an addrspace(1) global string in section "llvm.metadata"
// and return an i8 addrspace(1)* pointer to its first element.
//
// A module-level StringMap caches previously created globals so that
// identical strings are not duplicated.
static llvm::Value *getOrCreateAS1MetadataStringPtr(llvm::Module &M,
                                                    llvm::StringRef Str,
                                                    llvm::StringRef NameHint) {
  // Per-module cache keyed on the string content.
  static llvm::DenseMap<llvm::Module *, llvm::StringMap<llvm::Value *>> Cache;
  auto &ModuleCache = Cache[&M];

  auto It = ModuleCache.find(Str);
  if (It != ModuleCache.end())
    return It->second;

  llvm::LLVMContext &Ctx = M.getContext();

  llvm::Constant *Arr =
      llvm::ConstantDataArray::getString(Ctx, Str, /*AddNull=*/true);
  auto *ArrTy = llvm::cast<llvm::ArrayType>(Arr->getType());

  auto *GV = new llvm::GlobalVariable(
      M, ArrTy,
      /*isConstant=*/true, llvm::GlobalValue::PrivateLinkage, Arr, NameHint,
      /*InsertBefore=*/nullptr, llvm::GlobalValue::NotThreadLocal,
      /*AddressSpace=*/1);

  GV->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
  GV->setSection("llvm.metadata");
  GV->setAlignment(llvm::Align(1));

  llvm::Constant *Zero = llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), 0);
  llvm::Constant *Idxs[] = {Zero, Zero};

  llvm::Constant *GEP =
      llvm::ConstantExpr::getInBoundsGetElementPtr(ArrTy, GV, Idxs);

  auto *AS1PtrTy = llvm::PointerType::get(Ctx, /*AddrSpace=*/1);
  llvm::Value *Result = llvm::ConstantExpr::getBitCast(GEP, AS1PtrTy);

  ModuleCache[Str] = Result;
  return Result;
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
  static LogicalResult handleDecorationCacheControl(llvm::Instruction *inst,
                                                    ArrayRef<Attribute> attrs) {
    SmallVector<llvm::Metadata *> decorations;
    llvm::LLVMContext &ctx = inst->getContext();
    llvm::Type *i32Ty = llvm::IntegerType::getInt32Ty(ctx);
    llvm::transform(
        attrs, std::back_inserter(decorations),
        [&ctx, i32Ty](Attribute attr) -> llvm::Metadata * {
          auto valuesArray = dyn_cast<ArrayAttr>(attr).getValue();
          std::array<llvm::Metadata *, 3> metadata;
          llvm::transform(
              valuesArray, metadata.begin(), [i32Ty](Attribute valueAttr) {
                return llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    i32Ty, cast<IntegerAttr>(valueAttr).getValue()));
              });
          return llvm::MDNode::get(ctx, metadata);
        });
    constexpr llvm::StringLiteral decorationCacheControlMDName =
        "spirv.DecorationCacheControlINTEL";
    inst->setMetadata(decorationCacheControlMDName,
                      llvm::MDNode::get(ctx, decorations));
    return success();
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
