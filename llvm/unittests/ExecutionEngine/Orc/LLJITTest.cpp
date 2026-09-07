//===------------------------- LLJITTest.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

// Compile and run a trivial module through J and check it returns 42. The
// function is marked nounwind so the Win64 backend does not emit .pdata/.xdata
// unwind info: that info carries image-base-relative (ADDR32NB) relocations
// which would require __ImageBase to be defined -- something a bare
// ObjectLinkingLayer (no COFFPlatform) does not provide.
static void compileRunAndCheck42(LLJIT &J) {
  auto Ctx = std::make_unique<LLVMContext>();
  auto M = std::make_unique<Module>("<test>", *Ctx);
  M->setDataLayout(J.getDataLayout());

  // int f() { return 42; }
  auto *Int32Ty = Type::getInt32Ty(*Ctx);
  auto *F = Function::Create(FunctionType::get(Int32Ty, false),
                             Function::ExternalLinkage, "f", M.get());
  F->setDoesNotThrow();
  IRBuilder<> B(BasicBlock::Create(*Ctx, "entry", F));
  B.CreateRet(ConstantInt::get(Int32Ty, 42));

  ASSERT_THAT_ERROR(
      J.addIRModule(ThreadSafeModule(std::move(M), std::move(Ctx))),
      Succeeded());

  auto FSym = J.lookup("f");
  ASSERT_THAT_EXPECTED(FSym, Succeeded());

  auto *FPtr = FSym->toPtr<int (*)()>();
  EXPECT_EQ(FPtr(), 42);
}

// Build an LLJIT that uses the per-JITDylib-colocating slab allocator with the
// host's default linker, then compile and run a trivial module through it. On
// hosts where JITLink is the default (ELF/MachO) this exercises the slab
// allocator end to end; on hosts that still default to RuntimeDyld (COFF today)
// the allocator is constructed but unused, so this confirms the opt-in does no
// harm there.
TEST(LLJITTest, ColocatingSlabAllocator) {
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();

  auto J = LLJITBuilder().setColocatingSlabAllocator().create();
  if (!J) {
    // No JIT support for the host (or native target not built in).
    consumeError(J.takeError());
    GTEST_SKIP() << "Could not create an LLJIT for the host";
  }

  compileRunAndCheck42(**J);
}

// Force the JITLink ObjectLinkingLayer regardless of the host's default linker,
// so that on COFF (which otherwise uses RuntimeDyld and would ignore a
// JITLinkMemoryManager) the colocating slab allocator is actually used. This is
// the configuration that the eventual COFF-defaults-to-JITLink flip will make
// the default.
TEST(LLJITTest, ColocatingSlabAllocatorForcesJITLink) {
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();

  auto J =
      LLJITBuilder()
          .setColocatingSlabAllocator()
          .setObjectLinkingLayerCreator(
              [](ExecutionSession &ES, jitlink::JITLinkMemoryManager &MemMgr)
                  -> Expected<std::unique_ptr<ObjectLayer>> {
                return std::make_unique<ObjectLinkingLayer>(ES, MemMgr);
              })
          .create();
  if (!J) {
    consumeError(J.takeError());
    GTEST_SKIP() << "Could not create a JITLink-based LLJIT for the host";
  }

  compileRunAndCheck42(**J);
}

} // namespace
