//===- SandboxIRBench.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These tests measure the performance of some core SandboxIR functions and
// compare them against LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "benchmark/benchmark.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/SandboxIR/SandboxIR.h"
#include "llvm/Support/SourceMgr.h"
#include <memory>
#include <sstream>

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(IR, Err, C);
  if (!M)
    Err.print("SandboxIRBench", errs());
  return M;
}

enum class IR {
  LLVM,
  SBox,
};
// Traits to get llvm::BasicBlock/sandboxir::BasicBlock from IR::LLVM/IR::SBox.
template <IR IRTy> struct TypeSelect {};
template <> struct TypeSelect<IR::LLVM> {
  using BasicBlock = llvm::BasicBlock;
};
template <> struct TypeSelect<IR::SBox> {
  using BasicBlock = sandboxir::BasicBlock;
};

template <IR IRTy>
static typename TypeSelect<IRTy>::BasicBlock *
genIR(std::unique_ptr<llvm::Module> &LLVMM, LLVMContext &LLVMCtx,
      sandboxir::Context &Ctx,
      std::function<std::string(unsigned)> GenerateIRStr,
      unsigned NumInstrs = 0u) {
  std::string IRStr = GenerateIRStr(NumInstrs);
  LLVMM = parseIR(LLVMCtx, IRStr.c_str());
  llvm::Function *LLVMF = &*LLVMM->getFunction("foo");
  llvm::BasicBlock *LLVMBB = &*LLVMF->begin();

  sandboxir::Function *F = Ctx.createFunction(LLVMF);
  sandboxir::BasicBlock *BB = &*F->begin();
  if constexpr (IRTy == IR::LLVM)
    return LLVMBB;
  else
    return BB;
}

static std::string generateBBWalkIR(unsigned Size) {
  std::stringstream SS;
  SS << "define void @foo(i32 %v1, i32 %v2) {\n";
  for (auto Cnt : seq<unsigned>(0, Size))
    SS << "  %add" << Cnt << " = add i32 %v1, %v2\n";
  SS << "ret void";
  SS << "}";
  return SS.str();
}

template <IR IRTy> static void BBWalk(benchmark::State &State) {
  LLVMContext LLVMCtx;
  sandboxir::Context Ctx(LLVMCtx);
  unsigned NumInstrs = State.range(0);
  std::unique_ptr<llvm::Module> LLVMM;
  auto *BB = genIR<IRTy>(LLVMM, LLVMCtx, Ctx, generateBBWalkIR, NumInstrs);
  for (auto _ : State) {
    // Walk LLVM Instructions.
    for (auto &I : *BB)
      benchmark::DoNotOptimize(I);
  }
}

static std::string generateGetTypeIR(unsigned Size) {
  return R"IR(
define void @foo(i32 %v1, i32 %v2) {
  %add = add i32 %v1, %v2
  ret void
}
)IR";
}

template <IR IRTy> static void GetType(benchmark::State &State) {
  LLVMContext LLVMCtx;
  sandboxir::Context Ctx(LLVMCtx);
  std::unique_ptr<llvm::Module> LLVMM;
  auto *BB = genIR<IRTy>(LLVMM, LLVMCtx, Ctx, generateGetTypeIR);
  auto *I = &*BB->begin();
  for (auto _ : State)
    benchmark::DoNotOptimize(I->getType());
}

static std::string generateRAUWIR(unsigned Size) {
  std::stringstream SS;
  SS << "define void @foo(i32 %v1, i32 %v2) {\n";
  SS << "  %def1 = add i32 %v1, %v2\n";
  SS << "  %def2 = add i32 %v1, %v2\n";
  for (auto Cnt : seq<unsigned>(0, Size))
    SS << "  %add" << Cnt << " = add i32 %def1, %def1\n";
  SS << "ret void";
  SS << "}";
  return SS.str();
}

template <IR IRTy> static void RAUW(benchmark::State &State) {
  LLVMContext LLVMCtx;
  sandboxir::Context Ctx(LLVMCtx);
  std::unique_ptr<llvm::Module> LLVMM;
  unsigned NumInstrs = State.range(0);
  auto *BB = genIR<IRTy>(LLVMM, LLVMCtx, Ctx, generateRAUWIR, NumInstrs);
  auto It = BB->begin();
  auto *Def1 = &*It++;
  auto *Def2 = &*It++;
  for (auto _ : State) {
    Def1->replaceAllUsesWith(Def2);
    Def2->replaceAllUsesWith(Def1);
  }
}

BENCHMARK(GetType<IR::LLVM>);
BENCHMARK(GetType<IR::SBox>);

BENCHMARK(BBWalk<IR::LLVM>)->Args({1024});
BENCHMARK(BBWalk<IR::SBox>)->Args({1024});

BENCHMARK(RAUW<IR::LLVM>)->Args({512});
BENCHMARK(RAUW<IR::SBox>)->Args({512});

BENCHMARK_MAIN();
