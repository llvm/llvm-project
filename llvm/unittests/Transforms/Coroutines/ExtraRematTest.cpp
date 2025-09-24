//===- ExtraRematTest.cpp - Coroutines unit tests -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Transforms/Coroutines/ABI.h"
#include "llvm/Transforms/Coroutines/CoroSplit.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

struct ExtraRematTest : public testing::Test {
  LLVMContext Ctx;
  ModulePassManager MPM;
  PassBuilder PB;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  LLVMContext Context;
  std::unique_ptr<Module> M;

  ExtraRematTest() {
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  }

  BasicBlock *getBasicBlockByName(Function *F, StringRef Name) const {
    for (BasicBlock &BB : *F) {
      if (BB.getName() == Name)
        return &BB;
    }
    return nullptr;
  }

  CallInst *getCallByName(BasicBlock *BB, StringRef Name) const {
    for (Instruction &I : *BB) {
      if (CallInst *CI = dyn_cast<CallInst>(&I))
        if (CI->getCalledFunction()->getName() == Name)
          return CI;
    }
    return nullptr;
  }

  void ParseAssembly(const StringRef IR) {
    SMDiagnostic Error;
    M = parseAssemblyString(IR, Error, Context);
    std::string errMsg;
    raw_string_ostream os(errMsg);
    Error.print("", os);

    // A failure here means that the test itself is buggy.
    if (!M)
      report_fatal_error(errMsg.c_str());
  }
};

StringRef Text = R"(
    define ptr @f(i32 %n) presplitcoroutine {
    entry:
      %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
      %size = call i32 @llvm.coro.size.i32()
      %alloc = call ptr @malloc(i32 %size)
      %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)

      %inc1 = add i32 %n, 1
      %val2 = call i32 @should.remat(i32 %inc1)
      %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
      switch i8 %sp1, label %suspend [i8 0, label %resume1
                                      i8 1, label %cleanup]
    resume1:
      %inc2 = add i32 %val2, 1
      %sp2 = call i8 @llvm.coro.suspend(token none, i1 false)
      switch i8 %sp1, label %suspend [i8 0, label %resume2
                                      i8 1, label %cleanup]

    resume2:
      call void @print(i32 %val2)
      call void @print(i32 %inc2)
      br label %cleanup

    cleanup:
      %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
      call void @free(ptr %mem)
      br label %suspend
    suspend:
      call void @llvm.coro.end(ptr %hdl, i1 0)
      ret ptr %hdl
    }

    declare ptr @llvm.coro.free(token, ptr)
    declare i32 @llvm.coro.size.i32()
    declare i8  @llvm.coro.suspend(token, i1)
    declare void @llvm.coro.resume(ptr)
    declare void @llvm.coro.destroy(ptr)

    declare token @llvm.coro.id(i32, ptr, ptr, ptr)
    declare i1 @llvm.coro.alloc(token)
    declare ptr @llvm.coro.begin(token, ptr)
    declare void @llvm.coro.end(ptr, i1)

    declare i32 @should.remat(i32)

    declare noalias ptr @malloc(i32)
    declare void @print(i32)
    declare void @free(ptr)
  )";

// Materializable callback with extra rematerialization
bool ExtraMaterializable(Instruction &I) {
  if (isa<CastInst>(&I) || isa<GetElementPtrInst>(&I) ||
      isa<BinaryOperator>(&I) || isa<CmpInst>(&I) || isa<SelectInst>(&I))
    return true;

  if (auto *CI = dyn_cast<CallInst>(&I)) {
    auto *CalledFunc = CI->getCalledFunction();
    if (CalledFunc && CalledFunc->getName().starts_with("should.remat"))
      return true;
  }

  return false;
}

TEST_F(ExtraRematTest, TestCoroRematDefault) {
  ParseAssembly(Text);

  ASSERT_TRUE(M);

  CGSCCPassManager CGPM;
  CGPM.addPass(CoroSplitPass());
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  MPM.run(*M, MAM);

  // Verify that extra rematerializable instruction has been rematerialized
  Function *F = M->getFunction("f.resume");
  ASSERT_TRUE(F) << "could not find split function f.resume";

  BasicBlock *Resume1 = getBasicBlockByName(F, "resume1");
  ASSERT_TRUE(Resume1)
      << "could not find expected BB resume1 in split function";

  // With default materialization the intrinsic should not have been
  // rematerialized
  CallInst *CI = getCallByName(Resume1, "should.remat");
  ASSERT_FALSE(CI);
}

TEST_F(ExtraRematTest, TestCoroRematWithCallback) {
  ParseAssembly(Text);

  ASSERT_TRUE(M);

  CGSCCPassManager CGPM;
  CGPM.addPass(
      CoroSplitPass(std::function<bool(Instruction &)>(ExtraMaterializable)));
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  MPM.run(*M, MAM);

  // Verify that extra rematerializable instruction has been rematerialized
  Function *F = M->getFunction("f.resume");
  ASSERT_TRUE(F) << "could not find split function f.resume";

  BasicBlock *Resume1 = getBasicBlockByName(F, "resume1");
  ASSERT_TRUE(Resume1)
      << "could not find expected BB resume1 in split function";

  // With callback the extra rematerialization of the function should have
  // happened
  CallInst *CI = getCallByName(Resume1, "should.remat");
  ASSERT_TRUE(CI);
}

StringRef TextCoroBeginCustomABI = R"(
    define ptr @f(i32 %n) presplitcoroutine {
    entry:
      %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
      %size = call i32 @llvm.coro.size.i32()
      %alloc = call ptr @malloc(i32 %size)
      %hdl = call ptr @llvm.coro.begin.custom.abi(token %id, ptr %alloc, i32 0)

      %inc1 = add i32 %n, 1
      %val2 = call i32 @should.remat(i32 %inc1)
      %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
      switch i8 %sp1, label %suspend [i8 0, label %resume1
                                      i8 1, label %cleanup]
    resume1:
      %inc2 = add i32 %val2, 1
      %sp2 = call i8 @llvm.coro.suspend(token none, i1 false)
      switch i8 %sp1, label %suspend [i8 0, label %resume2
                                      i8 1, label %cleanup]

    resume2:
      call void @print(i32 %val2)
      call void @print(i32 %inc2)
      br label %cleanup

    cleanup:
      %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
      call void @free(ptr %mem)
      br label %suspend
    suspend:
      call void @llvm.coro.end(ptr %hdl, i1 0)
      ret ptr %hdl
    }

    declare ptr @llvm.coro.free(token, ptr)
    declare i32 @llvm.coro.size.i32()
    declare i8  @llvm.coro.suspend(token, i1)
    declare void @llvm.coro.resume(ptr)
    declare void @llvm.coro.destroy(ptr)

    declare token @llvm.coro.id(i32, ptr, ptr, ptr)
    declare i1 @llvm.coro.alloc(token)
    declare ptr @llvm.coro.begin.custom.abi(token, ptr, i32)
    declare void @llvm.coro.end(ptr, i1)

    declare i32 @should.remat(i32)

    declare noalias ptr @malloc(i32)
    declare void @print(i32)
    declare void @free(ptr)
  )";

// SwitchABI with overridden isMaterializable
class ExtraCustomABI : public coro::SwitchABI {
public:
  ExtraCustomABI(Function &F, coro::Shape &S)
      : coro::SwitchABI(F, S, ExtraMaterializable) {}
};

TEST_F(ExtraRematTest, TestCoroRematWithCustomABI) {
  ParseAssembly(TextCoroBeginCustomABI);

  ASSERT_TRUE(M);

  CoroSplitPass::BaseABITy GenCustomABI = [](Function &F, coro::Shape &S) {
    return std::make_unique<ExtraCustomABI>(F, S);
  };

  CGSCCPassManager CGPM;
  CGPM.addPass(CoroSplitPass({GenCustomABI}));
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  MPM.run(*M, MAM);

  // Verify that extra rematerializable instruction has been rematerialized
  Function *F = M->getFunction("f.resume");
  ASSERT_TRUE(F) << "could not find split function f.resume";

  BasicBlock *Resume1 = getBasicBlockByName(F, "resume1");
  ASSERT_TRUE(Resume1)
      << "could not find expected BB resume1 in split function";

  // With callback the extra rematerialization of the function should have
  // happened
  CallInst *CI = getCallByName(Resume1, "should.remat");
  ASSERT_TRUE(CI);
}

} // namespace
