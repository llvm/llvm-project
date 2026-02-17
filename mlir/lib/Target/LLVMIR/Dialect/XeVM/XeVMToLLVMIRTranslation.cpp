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
  /// Attach cache-control metadata to the pointer operand of \p Inst.
  ///
  /// Each attribute in \p Attrs becomes a separate \c llvm.ptr.annotation.p1.p1
  /// call.  When there are multiple attributes the calls are chained: the
  /// result of the first annotation is fed as the pointer into the second, and
  /// so on, so that every annotation is associated with the same logical
  /// pointer.
  ///
  /// @todo: We need to handle all the supported address spaces and not just
  /// addrspace(1).  This protocol is currently only supports addrspace(1).
  /// However, the SPV_INTEL_cache_control extension allows cache control
  /// decorations on pointers in different address spaces. See
  /// https://github.khronos.org/SPIRV-Registry/extensions/INTEL/SPV_INTEL_cache_controls.html
  /// for more details. We should extend this protocol to support other address
  /// spaces as needed.
  /// See:
  /// https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/main/docs/SPIRVRepresentationInLLVM.rst#address-spaces
  /// for mapping between LLVM address spaces and SPIR-V storage classes.
  ///
  /// \code
  ///   %a1 = call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(
  ///              ptr addrspace(1) %ptr, ... "{6442:\220,1\22}" ...)
  ///   %a2 = call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(
  ///              ptr addrspace(1) %a1,  ... "{6442:\221,1\22}" ...)
  ///   ; instruction now uses %a2
  /// \endcode
  static mlir::LogicalResult
  handleDecorationCacheControl(llvm::Instruction *Inst,
                               llvm::ArrayRef<mlir::Attribute> Attrs) {
    llvm::Module *M = Inst->getModule();
    if (!M)
      return mlir::failure();

    // @TODO: Expect the pointer to be the first operand, but this is not a
    // strict requirement of the protocol; it can be adjusted if needed.
    constexpr unsigned PtrIdx = 0;
    llvm::Value *Ptr = Inst->getOperand(PtrIdx);

    if (!Ptr || !Ptr->getType()->isPointerTy())
      return mlir::success();

    auto *PtrTy = llvm::cast<llvm::PointerType>(Ptr->getType());
    if (PtrTy->getAddressSpace() != 1)
      return mlir::success(); // protocol expects addrspace(1)

    llvm::SmallVector<std::string> Payloads = buildCacheControlPayloads(Attrs);
    if (Payloads.empty())
      return mlir::success();

    llvm::LLVMContext &Ctx = M->getContext();
    llvm::IRBuilder<> B(Inst);
    auto *AS1PtrTy = llvm::PointerType::get(Ctx, 1);

    // Shared across all annotations for this instruction.
    std::string FileName = getSourceFilename(Inst);
    llvm::Value *FileStrAS1 = getOrCreateAS1MetadataStringPtr(
        *M, FileName.empty() ? llvm::StringRef("") : llvm::StringRef(FileName),
        ".str.file");

    unsigned Line = 0;
    if (llvm::DILocation *Loc = Inst->getDebugLoc().get())
      Line = Loc->getLine();
    llvm::Value *LineV =
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), Line);

    llvm::Value *NullAS1 =
        llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(AS1PtrTy));

    llvm::FunctionType *FTy = llvm::FunctionType::get(
        PtrTy,
        {PtrTy, AS1PtrTy, AS1PtrTy, llvm::Type::getInt32Ty(Ctx), AS1PtrTy},
        /*isVarArg=*/false);

    llvm::FunctionCallee Callee =
        M->getOrInsertFunction("llvm.ptr.annotation.p1.p1", FTy);

    // Chain: each annotation takes the result of the previous one as its
    // pointer operand.
    llvm::Value *CurPtr = Ptr;
    for (const std::string &Payload : Payloads) {
      llvm::Value *AnnStrAS1 =
          getOrCreateAS1MetadataStringPtr(*M, Payload, ".str.cachecontrol");

      CurPtr =
          B.CreateCall(Callee, {CurPtr, AnnStrAS1, FileStrAS1, LineV, NullAS1},
                       "annotated.ptr");
    }

    Inst->setOperand(PtrIdx, CurPtr);
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
