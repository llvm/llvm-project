//===- WindowsArm64.cpp - Windows on Arm execution engine tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(_WIN32) && (defined(_M_ARM64) || defined(__aarch64__))

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <system_error>

using namespace mlir;

namespace {
constexpr uintptr_t FourGiB = uintptr_t{4} << 30;
constexpr uintptr_t EightGiB = uintptr_t{8} << 30;

static int FarData;
static int CtorCount;
static int DtorCount;

static void ctorCallback() { ++CtorCount; }
static void dtorCallback() { ++DtorCount; }

// Places JIT allocations at least 8 GiB from Anchor. This makes the test
// independent of Windows ASLR's usual (but not guaranteed) separation of the
// module-image, heap, and VirtualAlloc regions.
class FarMemoryMapper final : public llvm::SectionMemoryManager::MemoryMapper {
public:
  explicit FarMemoryMapper(uintptr_t Anchor)
      : Anchor(Anchor),
        NextHint(Anchor > EightGiB ? Anchor - EightGiB : Anchor + EightGiB) {}

  llvm::sys::MemoryBlock
  allocateMappedMemory(llvm::SectionMemoryManager::AllocationPurpose,
                       size_t NumBytes, const llvm::sys::MemoryBlock *,
                       unsigned Flags, std::error_code &EC) override {
    // LLVM's Windows mapper passes the end of Near as a VirtualAlloc address
    // hint. A zero-sized fake block therefore gives this test direct control
    // of the requested address without committing a multi-gigabyte range.
    llvm::sys::MemoryBlock Hint(reinterpret_cast<void *>(NextHint), 0);
    llvm::sys::MemoryBlock Block =
        llvm::sys::Memory::allocateMappedMemory(NumBytes, &Hint, Flags, EC);
    if (EC)
      return Block;

    uintptr_t Address = reinterpret_cast<uintptr_t>(Block.base());
    uintptr_t Distance = Address > Anchor ? Address - Anchor : Anchor - Address;
    if (Distance <= FourGiB) {
      llvm::sys::Memory::releaseMappedMemory(Block);
      EC = std::make_error_code(std::errc::not_enough_memory);
      return {};
    }

    MinimumDistance = std::min(MinimumDistance, Distance);
    // Keep all JIT objects close to each other while preserving their distance
    // from the external symbols in this executable.
    NextHint = Address + Block.allocatedSize();
    return Block;
  }

  std::error_code protectMappedMemory(const llvm::sys::MemoryBlock &Block,
                                      unsigned Flags) override {
    return llvm::sys::Memory::protectMappedMemory(Block, Flags);
  }

  std::error_code releaseMappedMemory(llvm::sys::MemoryBlock &Block) override {
    return llvm::sys::Memory::releaseMappedMemory(Block);
  }

  uintptr_t getMinimumDistance() const { return MinimumDistance; }

private:
  uintptr_t Anchor;
  uintptr_t NextHint;
  uintptr_t MinimumDistance = std::numeric_limits<uintptr_t>::max();
};

static std::unique_ptr<llvm::Module>
buildFarAddressModule(Operation *, llvm::LLVMContext &context) {
  auto module = std::make_unique<llvm::Module>("windows-arm64-far", context);
  llvm::IRBuilder<> builder(context);
  auto *voidFnTy = llvm::FunctionType::get(builder.getVoidTy(), false);

  auto addCtorOrDtor = [&](llvm::StringRef callbackName,
                           llvm::StringRef functionName, bool isCtor) {
    auto *callback = llvm::Function::Create(
        voidFnTy, llvm::GlobalValue::ExternalLinkage, callbackName, *module);
    auto *function = llvm::Function::Create(
        voidFnTy, llvm::GlobalValue::InternalLinkage, functionName, *module);
    builder.SetInsertPoint(
        llvm::BasicBlock::Create(context, "entry", function));
    builder.CreateCall(callback);
    builder.CreateRetVoid();
    if (isCtor)
      llvm::appendToGlobalCtors(*module, function, 0);
    else
      llvm::appendToGlobalDtors(*module, function, 0);
  };
  addCtorOrDtor("ctor_callback", "ctor", true);
  addCtorOrDtor("dtor_callback", "dtor", false);

  auto *farData = new llvm::GlobalVariable(*module, builder.getInt32Ty(), false,
                                           llvm::GlobalValue::ExternalLinkage,
                                           nullptr, "far_data");
  auto *addressFnTy = llvm::FunctionType::get(builder.getInt64Ty(), false);
  auto *addressFn =
      llvm::Function::Create(addressFnTy, llvm::GlobalValue::ExternalLinkage,
                             "get_far_data_address", *module);
  builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", addressFn));
  builder.CreateRet(builder.CreatePtrToInt(farData, builder.getInt64Ty()));
  return module;
}

TEST(MLIRExecutionEngine, ConstructorsAndFarExternalAddressOnWindowsArm64) {
  CtorCount = 0;
  DtorCount = 0;

  MLIRContext context;
  OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(&context));
  FarMemoryMapper mapper(reinterpret_cast<uintptr_t>(&FarData));
  ExecutionEngineOptions options;
  options.llvmModuleBuilder = buildFarAddressModule;
  options.sectionMemoryMapper = &mapper;

  auto jitOrError = ExecutionEngine::create(*module, options);
  if (!jitOrError)
    FAIL() << llvm::toString(jitOrError.takeError());
  std::unique_ptr<ExecutionEngine> jit = std::move(*jitOrError);
  jit->registerSymbols([](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbols;
    symbols[interner("far_data")] = {llvm::orc::ExecutorAddr::fromPtr(&FarData),
                                     llvm::JITSymbolFlags::Exported};
    symbols[interner("ctor_callback")] = {
        llvm::orc::ExecutorAddr::fromPtr(&ctorCallback),
        llvm::JITSymbolFlags::Exported};
    symbols[interner("dtor_callback")] = {
        llvm::orc::ExecutorAddr::fromPtr(&dtorCallback),
        llvm::JITSymbolFlags::Exported};
    return symbols;
  });

  EXPECT_EQ(CtorCount, 0);
  jit->initialize();
  EXPECT_EQ(CtorCount, 1);

  auto addressOrError = jit->lookup("get_far_data_address");
  if (!addressOrError)
    FAIL() << llvm::toString(addressOrError.takeError());
  auto getAddress = reinterpret_cast<uintptr_t (*)()>(*addressOrError);
  EXPECT_EQ(getAddress(), reinterpret_cast<uintptr_t>(&FarData));
  EXPECT_GT(mapper.getMinimumDistance(), FourGiB);

  jit.reset();
  EXPECT_EQ(DtorCount, 1);
}
} // namespace

#endif // Windows/AArch64
