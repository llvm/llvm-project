//=========- MemTransferLowerTest.cpp - MemTransferLower unit tests -=========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"
#include "llvm/Transforms/Vectorize/LoopVectorize.h"

#include "gtest/gtest-spi.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
struct ForwardingPass : public PassInfoMixin<ForwardingPass> {
  template <typename T> ForwardingPass(T &&Arg) : Func(std::forward<T>(Arg)) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
    return Func(F, FAM);
  }

  std::function<PreservedAnalyses(Function &, FunctionAnalysisManager &)> Func;
};

struct MemTransferLowerTest : public testing::Test {
  PassBuilder PB;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  ModulePassManager MPM;
  LLVMContext Context;
  std::unique_ptr<Module> M;

  MemTransferLowerTest() {
    // Register all the basic analyses with the managers.
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  }

  BasicBlock *getBasicBlockByName(Function &F, StringRef Name) const {
    for (BasicBlock &BB : F) {
      if (BB.getName() == Name)
        return &BB;
    }
    return nullptr;
  }

  Instruction *getInstructionByOpcode(BasicBlock &BB, unsigned Opcode,
                                      unsigned Number) const {
    unsigned CurrNumber = 0;
    for (Instruction &I : BB)
      if (I.getOpcode() == Opcode) {
        ++CurrNumber;
        if (CurrNumber == Number)
          return &I;
      }
    return nullptr;
  }

  void ParseAssembly(const char *IR) {
    SMDiagnostic Error;
    M = parseAssemblyString(IR, Error, Context);
    std::string errMsg;
    raw_string_ostream os(errMsg);
    Error.print("", os);

    // A failure here means that the test itself is buggy.
    if (!M)
      report_fatal_error(os.str().c_str());
  }
};

// By semantics source and destination of llvm.memcpy.* intrinsic
// are either equal or don't overlap. Once the intrinsic is lowered
// to a loop it can be hard or impossible to reason about these facts.
// For that reason expandMemCpyAsLoop is expected to  explicitly mark
// loads from source and stores to destination as not aliasing.
TEST_F(MemTransferLowerTest, MemCpyKnownLength) {
  ParseAssembly("declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8 *, i64, i1)\n"
                "define void @foo(i8* %dst, i8* %src, i64 %n) optsize {\n"
                "entry:\n"
                "  %is_not_equal = icmp ne i8* %dst, %src\n"
                "  br i1 %is_not_equal, label %memcpy, label %exit\n"
                "memcpy:\n"
                "  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, "
                "i64 1024, i1 false)\n"
                "  br label %exit\n"
                "exit:\n"
                "  ret void\n"
                "}\n");

  FunctionPassManager FPM;
  FPM.addPass(ForwardingPass(
      [=](Function &F, FunctionAnalysisManager &FAM) -> PreservedAnalyses {
        TargetTransformInfo TTI(M->getDataLayout());
        auto *MemCpyBB = getBasicBlockByName(F, "memcpy");
        Instruction *Inst = &MemCpyBB->front();
        MemCpyInst *MemCpyI = cast<MemCpyInst>(Inst);
        auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
        expandMemCpyAsLoop(MemCpyI, TTI, &SE);
        auto *CopyLoopBB = getBasicBlockByName(F, "load-store-loop");
        Instruction *LoadInst =
            getInstructionByOpcode(*CopyLoopBB, Instruction::Load, 1);
        EXPECT_NE(nullptr, LoadInst->getMetadata(LLVMContext::MD_alias_scope));
        Instruction *StoreInst =
            getInstructionByOpcode(*CopyLoopBB, Instruction::Store, 1);
        EXPECT_NE(nullptr, StoreInst->getMetadata(LLVMContext::MD_noalias));
        return PreservedAnalyses::none();
      }));
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));

  MPM.run(*M, MAM);
}

// This test indirectly checks that loads and stores (generated as a result of
// llvm.memcpy lowering) doesn't alias by making sure the loop can be
// successfully vectorized without additional runtime checks.
TEST_F(MemTransferLowerTest, VecMemCpyKnownLength) {
  ParseAssembly("declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8 *, i64, i1)\n"
                "define void @foo(i8* %dst, i8* %src, i64 %n) optsize {\n"
                "entry:\n"
                "  %is_not_equal = icmp ne i8* %dst, %src\n"
                "  br i1 %is_not_equal, label %memcpy, label %exit\n"
                "memcpy:\n"
                "  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, "
                "i64 1024, i1 false)\n"
                "  br label %exit\n"
                "exit:\n"
                "  ret void\n"
                "}\n");

  FunctionPassManager FPM;
  FPM.addPass(ForwardingPass(
      [=](Function &F, FunctionAnalysisManager &FAM) -> PreservedAnalyses {
        TargetTransformInfo TTI(M->getDataLayout());
        auto *MemCpyBB = getBasicBlockByName(F, "memcpy");
        Instruction *Inst = &MemCpyBB->front();
        MemCpyInst *MemCpyI = cast<MemCpyInst>(Inst);
        auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
        expandMemCpyAsLoop(MemCpyI, TTI, &SE);
        return PreservedAnalyses::none();
      }));
  FPM.addPass(LoopVectorizePass(LoopVectorizeOptions()));
  FPM.addPass(ForwardingPass(
      [=](Function &F, FunctionAnalysisManager &FAM) -> PreservedAnalyses {
        auto *TargetBB = getBasicBlockByName(F, "vector.body");
        EXPECT_NE(nullptr, TargetBB);
        return PreservedAnalyses::all();
      }));
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));

  MPM.run(*M, MAM);
}

TEST_F(MemTransferLowerTest, AtomicMemCpyKnownLength) {
  ParseAssembly("declare void "
                "@llvm.memcpy.element.unordered.atomic.p0i32.p0i32.i64(i32*, "
                "i32 *, i64, i32)\n"
                "define void @foo(i32* %dst, i32* %src, i64 %n) optsize {\n"
                "entry:\n"
                "  %is_not_equal = icmp ne i32* %dst, %src\n"
                "  br i1 %is_not_equal, label %memcpy, label %exit\n"
                "memcpy:\n"
                "  call void "
                "@llvm.memcpy.element.unordered.atomic.p0i32.p0i32.i64(i32* "
                "%dst, i32* %src, "
                "i64 1024, i32 4)\n"
                "  br label %exit\n"
                "exit:\n"
                "  ret void\n"
                "}\n");

  FunctionPassManager FPM;
  FPM.addPass(ForwardingPass(
      [=](Function &F, FunctionAnalysisManager &FAM) -> PreservedAnalyses {
        TargetTransformInfo TTI(M->getDataLayout());
        auto *MemCpyBB = getBasicBlockByName(F, "memcpy");
        Instruction *Inst = &MemCpyBB->front();
        assert(isa<AtomicMemCpyInst>(Inst) &&
               "Expecting llvm.memcpy.p0i8.i64 instructon");
        AtomicMemCpyInst *MemCpyI = cast<AtomicMemCpyInst>(Inst);
        auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
        expandAtomicMemCpyAsLoop(MemCpyI, TTI, &SE);
        auto *CopyLoopBB = getBasicBlockByName(F, "load-store-loop");
        Instruction *LoadInst =
            getInstructionByOpcode(*CopyLoopBB, Instruction::Load, 1);
        EXPECT_TRUE(LoadInst->isAtomic());
        EXPECT_NE(LoadInst->getMetadata(LLVMContext::MD_alias_scope), nullptr);
        Instruction *StoreInst =
            getInstructionByOpcode(*CopyLoopBB, Instruction::Store, 1);
        EXPECT_TRUE(StoreInst->isAtomic());
        EXPECT_NE(StoreInst->getMetadata(LLVMContext::MD_noalias), nullptr);
        return PreservedAnalyses::none();
      }));
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));

  MPM.run(*M, MAM);
}

TEST_F(MemTransferLowerTest, AtomicMemCpyUnKnownLength) {
  ParseAssembly("declare void "
                "@llvm.memcpy.element.unordered.atomic.p0i32.p0i32.i64(i32*, "
                "i32 *, i64, i32)\n"
                "define void @foo(i32* %dst, i32* %src, i64 %n) optsize {\n"
                "entry:\n"
                "  %is_not_equal = icmp ne i32* %dst, %src\n"
                "  br i1 %is_not_equal, label %memcpy, label %exit\n"
                "memcpy:\n"
                "  call void "
                "@llvm.memcpy.element.unordered.atomic.p0i32.p0i32.i64(i32* "
                "%dst, i32* %src, "
                "i64 %n, i32 4)\n"
                "  br label %exit\n"
                "exit:\n"
                "  ret void\n"
                "}\n");

  FunctionPassManager FPM;
  FPM.addPass(ForwardingPass(
      [=](Function &F, FunctionAnalysisManager &FAM) -> PreservedAnalyses {
        TargetTransformInfo TTI(M->getDataLayout());
        auto *MemCpyBB = getBasicBlockByName(F, "memcpy");
        Instruction *Inst = &MemCpyBB->front();
        assert(isa<AtomicMemCpyInst>(Inst) &&
               "Expecting llvm.memcpy.p0i8.i64 instructon");
        AtomicMemCpyInst *MemCpyI = cast<AtomicMemCpyInst>(Inst);
        auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
        expandAtomicMemCpyAsLoop(MemCpyI, TTI, &SE);
        auto *CopyLoopBB = getBasicBlockByName(F, "loop-memcpy-expansion");
        Instruction *LoadInst =
            getInstructionByOpcode(*CopyLoopBB, Instruction::Load, 1);
        EXPECT_TRUE(LoadInst->isAtomic());
        EXPECT_NE(LoadInst->getMetadata(LLVMContext::MD_alias_scope), nullptr);
        Instruction *StoreInst =
            getInstructionByOpcode(*CopyLoopBB, Instruction::Store, 1);
        EXPECT_TRUE(StoreInst->isAtomic());
        EXPECT_NE(StoreInst->getMetadata(LLVMContext::MD_noalias), nullptr);
        return PreservedAnalyses::none();
      }));
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));

  MPM.run(*M, MAM);
}
} // namespace
