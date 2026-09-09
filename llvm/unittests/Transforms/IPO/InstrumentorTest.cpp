//===- InstrumentorTest.cpp - Unit tests for InstrumentorPass -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/Instrumentor.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::instrumentor;

namespace {

class PreRuntimeLinkConfig final : public InstrumentationConfig {
public:
  explicit PreRuntimeLinkConfig(StringRef RuntimePath)
      : RuntimePath(RuntimePath) {}

  void populate(InstrumentorIRBuilderTy &) override {
    RuntimeBitcodes->setStringList({StringRef(RuntimePath)});
  }

  bool instrumentBeforeRuntimeLink(Module &M,
                                   InstrumentorIRBuilderTy &) override {
    LLVMContext &Ctx = M.getContext();
    Type *Ty = Type::getInt1Ty(Ctx);
    Constant *SawRuntime =
        ConstantInt::get(Ty, M.getNamedGlobal("runtime_marker") != nullptr);
    new GlobalVariable(M, Ty, false, GlobalValue::ExternalLinkage, SawRuntime,
                       "hook_saw_runtime");
    return true;
  }

private:
  std::string RuntimePath;
};

std::unique_ptr<Module> parseModule(StringRef IR, LLVMContext &Ctx) {
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(IR, Err, Ctx);
  EXPECT_TRUE(M);
  return M;
}

TEST(InstrumentorTest, RunsHookBeforeRuntimeLink) {
  SmallString<128> RuntimePath;
  int FD;
  ASSERT_FALSE(sys::fs::createTemporaryFile("instrumentor-runtime", "ll", FD,
                                            RuntimePath));
  scope_exit RemoveRuntime([&] { sys::fs::remove(RuntimePath); });

  raw_fd_ostream OS(FD, true);
  OS << "@runtime_marker = global i32 0\n";
  OS.close();

  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseModule(R"ir(
    define void @test() {
    entry:
      ret void
    }
  )ir",
                                          Ctx);
  ASSERT_TRUE(M);

  PreRuntimeLinkConfig Config(RuntimePath);
  ModuleAnalysisManager MAM;
  InstrumentorPass Pass(/*FS=*/nullptr, &Config, /*IIRB=*/nullptr);
  Pass.run(*M, MAM);

  GlobalVariable *HookSawRuntime = M->getNamedGlobal("hook_saw_runtime");
  ASSERT_NE(HookSawRuntime, nullptr);
  auto *Initializer = dyn_cast<ConstantInt>(HookSawRuntime->getInitializer());
  ASSERT_NE(Initializer, nullptr);
  EXPECT_TRUE(Initializer->isZero());
  EXPECT_NE(M->getNamedGlobal("runtime_marker"), nullptr);
}

} // namespace
