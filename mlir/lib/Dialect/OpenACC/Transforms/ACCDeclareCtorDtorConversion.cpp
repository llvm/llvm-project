//===- ACCDeclareCtorDtorConversion.cpp - Declare ctor/dtor to LLVM -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convert ACC declare global constructors and destructors to LLVM functions
// registered in llvm.mlir.global_ctors / llvm.mlir.global_dtors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"

#include <string>
#include <utility>

namespace llvm::cl {
template <>
class parser<std::pair<std::string, bool>>
    : public basic_parser<std::pair<std::string, bool>> {
public:
  parser(Option &option) : basic_parser(option) {}

  bool parse(Option &option, StringRef argName, StringRef arg,
             std::pair<std::string, bool> &value) {
    auto [name, flagStr] = arg.rsplit(':');
    if (name.empty() || flagStr.empty())
      return option.error("expected <name>:<bool>", argName);

    bool ifMain = false;
    if (flagStr.equals_insensitive("true") || flagStr == "1")
      ifMain = true;
    else if (flagStr.equals_insensitive("false") || flagStr == "0")
      ifMain = false;
    else
      return option.error("invalid boolean in extra constructor mapping",
                          argName);

    value = {name.str(), ifMain};
    return false;
  }

  StringRef getValueName() const override { return "name:bool"; }

  static void print(raw_ostream &os,
                    const std::pair<std::string, bool> &value) {
    os << value.first << ':' << (value.second ? "true" : "false");
  }
};
} // namespace llvm::cl

namespace mlir {
namespace acc {
#define GEN_PASS_DEF_ACCDECLARECTORDTORCONVERSION
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"
} // namespace acc
} // namespace mlir

using namespace mlir;

namespace {

static void collectExistingGlobalCtors(
    ModuleOp mod, SmallVectorImpl<Attribute> &ctors,
    SmallVectorImpl<int32_t> &priorities, SmallVectorImpl<Attribute> &data,
    SmallVectorImpl<LLVM::GlobalCtorsOp> &globalCtorsOps) {
  for (auto globalCtors : mod.getOps<LLVM::GlobalCtorsOp>()) {
    ctors.append(globalCtors.getCtors().begin(), globalCtors.getCtors().end());
    for (Attribute attr : globalCtors.getPriorities())
      priorities.push_back(cast<IntegerAttr>(attr).getInt());
    data.append(globalCtors.getData().begin(), globalCtors.getData().end());
    globalCtorsOps.push_back(globalCtors);
  }
}

static void collectExistingGlobalDtors(
    ModuleOp mod, SmallVectorImpl<Attribute> &dtors,
    SmallVectorImpl<int32_t> &priorities, SmallVectorImpl<Attribute> &data,
    SmallVectorImpl<LLVM::GlobalDtorsOp> &globalDtorsOps) {
  for (auto globalDtors : mod.getOps<LLVM::GlobalDtorsOp>()) {
    dtors.append(globalDtors.getDtors().begin(), globalDtors.getDtors().end());
    for (Attribute attr : globalDtors.getPriorities())
      priorities.push_back(cast<IntegerAttr>(attr).getInt());
    data.append(globalDtors.getData().begin(), globalDtors.getData().end());
    globalDtorsOps.push_back(globalDtors);
  }
}

static void replaceGlobalCtors(ModuleOp mod, OpBuilder &builder,
                               ArrayRef<Attribute> ctors,
                               ArrayRef<int32_t> priorities,
                               ArrayRef<Attribute> data,
                               MutableArrayRef<LLVM::GlobalCtorsOp> oldOps) {
  for (auto globalCtors : oldOps)
    globalCtors.erase();
  if (ctors.empty())
    return;

  builder.setInsertionPointToEnd(mod.getBody());
  LLVM::GlobalCtorsOp::create(
      builder, mod.getLoc(), builder.getArrayAttr(ctors),
      builder.getI32ArrayAttr(priorities), builder.getArrayAttr(data));
}

static void replaceGlobalDtors(ModuleOp mod, OpBuilder &builder,
                               ArrayRef<Attribute> dtors,
                               ArrayRef<int32_t> priorities,
                               ArrayRef<Attribute> data,
                               MutableArrayRef<LLVM::GlobalDtorsOp> oldOps) {
  for (auto globalDtors : oldOps)
    globalDtors.erase();
  if (dtors.empty())
    return;

  builder.setInsertionPointToEnd(mod.getBody());
  LLVM::GlobalDtorsOp::create(
      builder, mod.getLoc(), builder.getArrayAttr(dtors),
      builder.getI32ArrayAttr(priorities), builder.getArrayAttr(data));
}

/// Create an llvm.func from an acc.global_ctor / acc.global_dtor region.
/// Nested operations are cloned unchanged for later lowering.
static LLVM::LLVMFuncOp createLLVMFunctionFromRegion(StringRef symName,
                                                     Region &region,
                                                     ModuleOp mod,
                                                     OpBuilder &builder) {
  auto llvmVoidTy = LLVM::LLVMVoidType::get(mod.getContext());
  auto funcTy = LLVM::LLVMFunctionType::get(llvmVoidTy, {}, /*isVarArg=*/false);
  builder.setInsertionPointToEnd(mod.getBody());
  auto newFunc = LLVM::LLVMFuncOp::create(builder, mod.getLoc(), symName,
                                          funcTy, LLVM::Linkage::Internal);

  Block *entry = newFunc.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry);

  IRMapping mapping;
  mapping.map(region.front().getArguments(), entry->getArguments());

  for (Operation &op : region.front()) {
    Operation *clonedOp = builder.clone(op, mapping);
    mapping.map(op.getResults(), clonedOp->getResults());
  }

  Operation *accTerm = entry->getTerminator();
  LLVM::ReturnOp::create(builder, mod.getLoc(), ValueRange{});
  accTerm->erase();

  return newFunc;
}

static constexpr llvm::StringRef kExtraCtorName{"__openaccExtraConstructor"};

/// Declare extra runtime functions and call them from a defined llvm.func
/// registered in llvm.mlir.global_ctors. The extra functions themselves stay
/// declarations: llvm.mlir.global_ctors requires a function with a body.
static LLVM::LLVMFuncOp
createExtraConstructorCaller(ModuleOp mod, OpBuilder &builder,
                             ArrayRef<std::string> extraNames) {
  auto llvmVoidTy = LLVM::LLVMVoidType::get(mod.getContext());
  auto funcTy = LLVM::LLVMFunctionType::get(llvmVoidTy, {}, /*isVarArg=*/false);

  builder.setInsertionPointToEnd(mod.getBody());
  for (const std::string &name : extraNames) {
    if (mod.lookupSymbol<LLVM::LLVMFuncOp>(name))
      continue;
    auto decl = LLVM::LLVMFuncOp::create(builder, mod.getLoc(), name, funcTy);
    decl.setVisibility(SymbolTable::Visibility::Private);
  }

  auto wrapper = LLVM::LLVMFuncOp::create(builder, mod.getLoc(), kExtraCtorName,
                                          funcTy, LLVM::Linkage::Internal);
  Block *entry = wrapper.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry);
  for (const std::string &name : extraNames)
    LLVM::CallOp::create(builder, mod.getLoc(), funcTy,
                         FlatSymbolRefAttr::get(mod.getContext(), name));
  LLVM::ReturnOp::create(builder, mod.getLoc(), ValueRange{});
  return wrapper;
}

struct ACCDeclareCtorDtorConversion
    : public acc::impl::ACCDeclareCtorDtorConversionBase<
          ACCDeclareCtorDtorConversion> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    OpBuilder builder{mod.getBodyRegion()};
    SmallVector<Operation *> worklist;

    SmallVector<Attribute, 8> allCtors;
    SmallVector<int32_t, 8> ctorPriorities;
    SmallVector<Attribute, 8> ctorData;
    SmallVector<LLVM::GlobalCtorsOp, 4> globalCtorsOps;
    collectExistingGlobalCtors(mod, allCtors, ctorPriorities, ctorData,
                               globalCtorsOps);
    size_t existingCtorCount = allCtors.size();

    SmallVector<Attribute, 8> allDtors;
    SmallVector<int32_t, 8> dtorPriorities;
    SmallVector<Attribute, 8> dtorData;
    SmallVector<LLVM::GlobalDtorsOp, 4> globalDtorsOps;
    collectExistingGlobalDtors(mod, allDtors, dtorPriorities, dtorData,
                               globalDtorsOps);
    size_t existingDtorCount = allDtors.size();

    mod.walk([&](acc::GlobalConstructorOp op) {
      LLVM::LLVMFuncOp newCtor = createLLVMFunctionFromRegion(
          op.getSymName(), op.getRegion(), mod, builder);
      allCtors.push_back(
          FlatSymbolRefAttr::get(mod.getContext(), newCtor.getSymName()));
      ctorPriorities.push_back(priority);
      // Null associated data: constructor always runs at load time.
      ctorData.push_back(LLVM::ZeroAttr::get(builder.getContext()));
      worklist.push_back(op.getOperation());
    });

    mod.walk([&](acc::GlobalDestructorOp op) {
      if (generateDtors) {
        LLVM::LLVMFuncOp newDtor = createLLVMFunctionFromRegion(
            op.getSymName(), op.getRegion(), mod, builder);
        allDtors.push_back(
            FlatSymbolRefAttr::get(mod.getContext(), newDtor.getSymName()));
        dtorPriorities.push_back(priority);
        // Null associated data: destructor always runs at unload time.
        dtorData.push_back(LLVM::ZeroAttr::get(builder.getContext()));
      }
      worklist.push_back(op.getOperation());
    });

    bool hasProgramEntry =
        !programEntryName.empty() &&
        static_cast<bool>(mod.lookupSymbol(programEntryName));
    SmallVector<std::string, 4> extraNames;
    for (const auto &[funcName, ifMain] : extraConstructors) {
      if (!ifMain || hasProgramEntry)
        extraNames.push_back(funcName);
    }
    if (!extraNames.empty()) {
      LLVM::LLVMFuncOp extraCtor =
          createExtraConstructorCaller(mod, builder, extraNames);
      allCtors.push_back(
          FlatSymbolRefAttr::get(mod.getContext(), extraCtor.getSymName()));
      ctorPriorities.push_back(priority);
      ctorData.push_back(LLVM::ZeroAttr::get(builder.getContext()));
    }

    if (allCtors.size() > existingCtorCount)
      replaceGlobalCtors(mod, builder, allCtors, ctorPriorities, ctorData,
                         globalCtorsOps);
    if (allDtors.size() > existingDtorCount)
      replaceGlobalDtors(mod, builder, allDtors, dtorPriorities, dtorData,
                         globalDtorsOps);

    for (Operation *op : worklist)
      op->erase();
  }
};

} // namespace
