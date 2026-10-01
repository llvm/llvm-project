//===- KnownBitsDataflowTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/KnownBitsDataflow.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
static Instruction *findInstructionByName(Function *F, StringRef Name) {
  for (Instruction &I : instructions(F))
    if (I.getName() == Name)
      return &I;
  return nullptr;
}

std::unique_ptr<Module> parseIR(LLVMContext &Ctx, StringRef Assembly) {
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(Assembly, Err, Ctx);
  if (!M)
    Err.print(__FILE__, errs());
  return M;
}

template <typename RangeT>
static auto make_knownbits_range(RangeT &&R) { // NOLINT
  return make_filter_range(R, [](const auto &V) {
    return V->getType()->getScalarType()->isIntOrPtrTy();
  });
}

/// Simply exposes some routines in KnownBitsDataflow, and adds a few for
/// testing.
struct DataflowForTest : public KnownBitsDataflow {
private:
  const Function &F;

public:
  DataflowForTest(const Function &F) : F(F) {}
  SmallVector<KnownBitsVH, 16> computeRoots() const {
    return KnownBitsDataflow::computeRoots(F);
  }
  bool contains(const Value *V) const { return KnownBitsDataflow::contains(V); }
  KnownBits at(const Value *V) const {
    auto It = find_as(V);
    assert(It != end() && "Expected key in Map");
    return It->second;
  }
  // Returns an unordered list.
  auto computeLeaves() const {
    auto IsLeaf = [](const Value *V) {
      return make_knownbits_range(V->users()).empty();
    };
    return make_filter_range(keys(), [&IsLeaf](const KnownBitsVH &V) {
      return IsLeaf(V.getValPtr());
    });
  }
  void setKB(const Value *V, KnownBits Known) {
    value_as(V) = {Known, nullptr};
  }
  void setAllZero(const Value *V) { value_as(V).setAllZero(); }
  void setAllOnes(const Value *V) { value_as(V).setAllOnes(); }
  void invalidate(const Value *V) {
    return KnownBitsDataflow::invalidate(key_as(V));
  }
  bool isAllOnes(const Value *V) const { return at(V).isAllOnes(); }
  void print(raw_ostream &OS) { KnownBitsDataflow::print(F, OS); }
};

/// Additionally initializes the map with the entire graph.
struct DataflowInitializerForTest : public DataflowForTest {
  DataflowInitializerForTest(Function &F) : DataflowForTest(F) {
    initializeEntireGraph(F);
  }
  void initializeEntireGraph(const Function &F) {
    // We emplace all-conflict that's different from unknown for testing. This
    // is because unknown KnownBits are used to represent invalidated values.
    auto GetAllConflict = [](unsigned BitWidth) {
      KnownBits Known(BitWidth);
      Known.setAllConflict();
      return Known;
    };
    for (const Value *V : make_knownbits_range(make_pointer_range(F.args())))
      emplace_as(V, GetAllConflict(KnownBitsDataflow::getBitWidth(
                        V->getType(), F.getDataLayout())));

    // Now collect all Instructions that aren't reachable from the function's
    // arguments, updating Roots, as we test for unreachability.
    for (const BasicBlock &BB : F) {
      for (const Value *V : make_knownbits_range(make_pointer_range(BB))) {
        emplace_as(V, GetAllConflict(KnownBitsDataflow::getBitWidth(
                          V->getType(), F.getDataLayout())));
      }
    }
  }
};

TEST(KnownBitsDataflow, BasicConstruction) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
define void @test(i32 %n) {
entry:
  br label %loop
loop:
  %phi_counter = phi i32 [ 0, %entry ], [ %next_counter, %loop ]
  %phi_result = phi i32 [ 1, %entry ], [ %result, %loop ]
  %counter = add i32 %phi_counter, 1
  %result = mul i32 %phi_result, 2
  %next_counter = add i32 %counter, 1
  %cond = icmp slt i32 %next_counter, %n
  br i1 %cond, label %loop, label %exit
exit:
  store i32 %result, ptr poison
  ret void
})");
  Function *F = M->getFunction("test");
  DataflowInitializerForTest KBCache(*F);
  Argument *ArgN = &*F->arg_begin();
  Instruction *Counter = findInstructionByName(F, "counter");
  Instruction *NextCounter = findInstructionByName(F, "next_counter");
  Instruction *Result = findInstructionByName(F, "result");
  Instruction *PhiCounter = findInstructionByName(F, "phi_counter");
  Instruction *PhiResult = findInstructionByName(F, "phi_result");
  Instruction *Cond = findInstructionByName(F, "cond");

  EXPECT_TRUE(KBCache.contains(ArgN));
  EXPECT_TRUE(KBCache.contains(PhiCounter));
  EXPECT_TRUE(KBCache.contains(PhiResult));
  EXPECT_TRUE(KBCache.contains(Counter));
  EXPECT_TRUE(KBCache.contains(Result));
  EXPECT_TRUE(KBCache.contains(NextCounter));
  EXPECT_TRUE(KBCache.contains(Cond));
  EXPECT_EQ(KBCache.size(), 7u);
}

TEST(KnownBitsDataflow, ConstructionWithIntAndPtr) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
define void @test(i32 %int_arg, float %float_arg, ptr %ptr_arg, <2 x i32> %vec_int_arg, <2 x ptr> %vec_ptr_arg) {
entry:
  br i1 poison, label %then, label %else
then:
  %int_val = add i32 %int_arg, 1
  %float_val = fadd float %float_arg, 1.0
  %vec_val = add <2 x i32> %vec_int_arg, <i32 1, i32 2>
  br label %merge
else:
  %fpconv = fptoui float %float_arg to i32
  %int_val2 = mul i32 %int_arg, %fpconv
  %ptr_val = getelementptr i8, ptr %ptr_arg, i32 4
  %vec_val2 = mul <2 x i32> %vec_int_arg, <i32 3, i32 4>
  br label %merge
merge:
  %phi_int = phi i32 [ %int_val, %then ], [ %int_val2, %else ]
  %phi_float = phi float [ %float_val, %then ], [ %float_arg, %else ]
  %phi_ptr = phi ptr [ %ptr_arg, %then ], [ %ptr_val, %else ]
  %phi_vec = phi <2 x i32> [ %vec_val, %then ], [ %vec_val2, %else ]
  %final_int = add i32 %phi_int, 5
  %vec_ptr_conv = ptrtoint <2 x ptr> %vec_ptr_arg to <2 x i32>
  %final_vec = add <2 x i32> %phi_vec, %vec_ptr_conv
  store float %phi_float, ptr %phi_ptr
  ret void
})");
  Function *F = M->getFunction("test");
  DataflowInitializerForTest KBCache(*F);
  auto *ArgIt = F->arg_begin();
  Argument *IntArg = &*ArgIt++;
  Argument *FloatArg = &*ArgIt++;
  Argument *PtrArg = &*ArgIt++;
  Argument *VecIntArg = &*ArgIt++;
  Argument *VecPtrArg = &*ArgIt++;
  Instruction *IntVal = findInstructionByName(F, "int_val");
  Instruction *FloatVal = findInstructionByName(F, "float_val");
  Instruction *VecVal = findInstructionByName(F, "vec_val");
  Instruction *IntVal2 = findInstructionByName(F, "int_val2");
  Instruction *PtrVal = findInstructionByName(F, "ptr_val");
  Instruction *VecVal2 = findInstructionByName(F, "vec_val2");
  Instruction *PhiInt = findInstructionByName(F, "phi_int");
  Instruction *PhiFloat = findInstructionByName(F, "phi_float");
  Instruction *PhiPtr = findInstructionByName(F, "phi_ptr");
  Instruction *PhiVec = findInstructionByName(F, "phi_vec");
  Instruction *FinalInt = findInstructionByName(F, "final_int");
  Instruction *FPConv = findInstructionByName(F, "fpconv");
  Instruction *VecPtrConv = findInstructionByName(F, "vec_ptr_conv");
  Instruction *FinalVec = findInstructionByName(F, "final_vec");

  EXPECT_THAT(
      KBCache.computeRoots(),
      ::testing::ElementsAre(IntArg, PtrArg, VecIntArg, VecPtrArg, FPConv));
  EXPECT_THAT(KBCache.computeLeaves(),
              ::testing::UnorderedElementsAre(FinalInt, PhiPtr, FinalVec));

  EXPECT_TRUE(KBCache.contains(IntArg));
  EXPECT_FALSE(KBCache.contains(FloatArg));
  EXPECT_TRUE(KBCache.contains(PtrArg));
  EXPECT_TRUE(KBCache.contains(VecIntArg));
  EXPECT_TRUE(KBCache.contains(VecPtrArg));
  EXPECT_TRUE(KBCache.contains(IntVal));
  EXPECT_TRUE(KBCache.contains(IntVal2));
  EXPECT_TRUE(KBCache.contains(PhiInt));
  EXPECT_TRUE(KBCache.contains(FinalInt));
  EXPECT_TRUE(KBCache.contains(VecVal));
  EXPECT_TRUE(KBCache.contains(VecVal2));
  EXPECT_TRUE(KBCache.contains(PhiVec));
  EXPECT_TRUE(KBCache.contains(FPConv));
  EXPECT_TRUE(KBCache.contains(VecPtrConv));
  EXPECT_TRUE(KBCache.contains(FinalVec));
  EXPECT_FALSE(KBCache.contains(FloatVal));
  EXPECT_TRUE(KBCache.contains(PtrVal));
  EXPECT_FALSE(KBCache.contains(PhiFloat));
  EXPECT_TRUE(KBCache.contains(PhiPtr));

  EXPECT_EQ(KBCache.size(), 16u);
}

TEST(KnownBitsDataflow, ConstructionWithNestedLoop) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
define void @test(i32 %n, i32 %m) {
entry:
  br label %outer_loop
outer_loop:
  %outer_phi = phi i32 [ 0, %entry ], [ %outer_next, %outer_latch ]
  br label %inner_loop
inner_loop:
  %inner_phi = phi i32 [ 0, %outer_loop ], [ %inner_next, %inner_loop ]
  %inner_next = add i32 %inner_phi, 1
  %inner_cond = icmp slt i32 %inner_next, %m
  br i1 %inner_cond, label %inner_loop, label %outer_latch
outer_latch:
  %outer_next = add i32 %outer_phi, 1
  %outer_cond = icmp slt i32 %outer_next, %n
  br i1 %outer_cond, label %outer_loop, label %exit
exit:
  ret void
})");
  Function *F = M->getFunction("test");
  DataflowInitializerForTest KBCache(*F);
  auto *ArgIt = F->arg_begin();
  Argument *ArgN = &*ArgIt++;
  Argument *ArgM = &*ArgIt;
  Instruction *OuterPHI = findInstructionByName(F, "outer_phi");
  Instruction *InnerPHI = findInstructionByName(F, "inner_phi");
  Instruction *InnerNext = findInstructionByName(F, "inner_next");
  Instruction *OuterNext = findInstructionByName(F, "outer_next");
  Instruction *InnerCond = findInstructionByName(F, "inner_cond");
  Instruction *OuterCond = findInstructionByName(F, "outer_cond");

  EXPECT_THAT(KBCache.computeRoots(),
              ::testing::ElementsAre(ArgN, ArgM, OuterPHI, InnerPHI));
  EXPECT_THAT(KBCache.computeLeaves(),
              ::testing::UnorderedElementsAre(OuterCond, InnerCond));

  EXPECT_TRUE(KBCache.contains(ArgN));
  EXPECT_TRUE(KBCache.contains(ArgM));
  EXPECT_TRUE(KBCache.contains(OuterPHI));
  EXPECT_TRUE(KBCache.contains(InnerPHI));
  EXPECT_TRUE(KBCache.contains(InnerNext));
  EXPECT_TRUE(KBCache.contains(OuterNext));
  EXPECT_TRUE(KBCache.contains(InnerCond));
  EXPECT_TRUE(KBCache.contains(OuterCond));
  EXPECT_EQ(KBCache.size(), 8u);
}

TEST(KnownBitsDataflow, ForwardDataflowBitsSingleBB) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
define void @test(i32 %arg, <2 x i32> %vec_arg) {
  %counter = add i32 %arg, 1
  %result = mul i32 %counter, 2
  %next_counter = add i32 %result, 3
  %branch_val = sub i32 %next_counter, 1
  %merge_val = add i32 %branch_val, 5
  store i32 %merge_val, ptr poison
  ret void
})");
  Function *F = M->getFunction("test");
  DataflowInitializerForTest KBCache(*F);
  Argument *Arg = &*F->arg_begin();
  Argument *VecArg = &*F->arg_begin();
  Instruction *Counter = findInstructionByName(F, "counter");
  Instruction *NextCounter = findInstructionByName(F, "next_counter");
  Instruction *Result = findInstructionByName(F, "result");
  Instruction *BranchVal = findInstructionByName(F, "branch_val");
  Instruction *MergeVal = findInstructionByName(F, "merge_val");
  KnownBits Known32(32);
  Known32.setAllOnes();

  KBCache.setKB(Arg, Known32);
  KBCache.setKB(Counter, Known32);
  KBCache.setKB(Result, Known32);
  KBCache.setKB(NextCounter, Known32);
  KBCache.setKB(BranchVal, Known32);
  KBCache.setKB(MergeVal, Known32);
  KBCache.setAllOnes(VecArg);

  EXPECT_TRUE(KBCache.isAllOnes(Arg));
  EXPECT_TRUE(KBCache.isAllOnes(Counter));
  EXPECT_TRUE(KBCache.isAllOnes(Result));
  EXPECT_TRUE(KBCache.isAllOnes(NextCounter));
  EXPECT_TRUE(KBCache.isAllOnes(BranchVal));
  EXPECT_TRUE(KBCache.isAllOnes(MergeVal));
  EXPECT_TRUE(KBCache.isAllOnes(VecArg));

  KBCache.invalidate(Counter);

  EXPECT_TRUE(KBCache.isAllOnes(Arg));
  EXPECT_TRUE(KBCache.isAllOnes(VecArg));
  EXPECT_FALSE(KBCache.lookup(Counter));
  EXPECT_FALSE(KBCache.lookup(Result));
  EXPECT_FALSE(KBCache.lookup(NextCounter));
  EXPECT_FALSE(KBCache.lookup(BranchVal));
  EXPECT_FALSE(KBCache.lookup(MergeVal));
}

TEST(KnownBitsDataflow, ForwardDataflowMultipleBBs) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
define void @test(i32 %n, i1 %cond) {
entry:
  %counter = add i32 %n, 1
  br i1 %cond, label %then, label %else
then:
  %branch_val = mul i32 %counter, 2
  br label %merge
else:
  %result = add i32 %counter, 3
  br label %merge
merge:
  %merge_val = phi i32 [ %branch_val, %then ], [ %result, %else ]
  %next_counter = add i32 %merge_val, 1
  store i32 %next_counter, ptr poison
  ret void
})");
  Function *F = M->getFunction("test");
  DataflowInitializerForTest KBCache(*F);
  auto *ArgIt = F->arg_begin();
  Argument *ArgN = &*ArgIt++;
  Argument *ArgCond = &*ArgIt;
  Instruction *Counter = findInstructionByName(F, "counter");
  Instruction *NextCounter = findInstructionByName(F, "next_counter");
  Instruction *Result = findInstructionByName(F, "result");
  Instruction *BranchVal = findInstructionByName(F, "branch_val");
  Instruction *MergeVal = findInstructionByName(F, "merge_val");
  KnownBits Known32(32);
  Known32.setAllOnes();

  KBCache.setKB(ArgN, Known32);
  KBCache.setKB(Counter, Known32);
  KBCache.setKB(BranchVal, Known32);
  KBCache.setKB(Result, Known32);
  KBCache.setKB(MergeVal, Known32);
  KBCache.setKB(NextCounter, Known32);
  KBCache.setAllOnes(ArgCond);

  EXPECT_TRUE(KBCache.isAllOnes(Counter));
  EXPECT_TRUE(KBCache.isAllOnes(BranchVal));
  EXPECT_TRUE(KBCache.isAllOnes(Result));
  EXPECT_TRUE(KBCache.isAllOnes(MergeVal));
  EXPECT_TRUE(KBCache.isAllOnes(NextCounter));

  KBCache.invalidate(Result);

  EXPECT_TRUE(KBCache.isAllOnes(ArgN));
  EXPECT_TRUE(KBCache.isAllOnes(ArgCond));
  EXPECT_TRUE(KBCache.isAllOnes(Counter));
  EXPECT_TRUE(KBCache.isAllOnes(BranchVal));
  EXPECT_FALSE(KBCache.lookup(Result));
  EXPECT_FALSE(KBCache.lookup(MergeVal));
  EXPECT_FALSE(KBCache.lookup(NextCounter));
}

TEST(KnownBitsDataflow, ForwardDataflowPartialInitialization) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
define void @test(i32 %n, i1 %cond) {
entry:
  %counter = add i32 %n, 1
  br i1 %cond, label %then, label %else
then:
  %branch_val = mul i32 %counter, 2
  br label %merge
else:
  %result = add i32 %counter, 3
  br label %merge
merge:
  %merge_val = phi i32 [ %branch_val, %then ], [ %result, %else ]
  %next_counter = add i32 %merge_val, 1
  store i32 %next_counter, ptr poison
  ret void
})");
  Function *F = M->getFunction("test");
  DataflowForTest KBCache(*F);
  auto *ArgIt = F->arg_begin();
  Argument *ArgN = &*ArgIt++;
  Argument *ArgCond = &*ArgIt;
  Instruction *Counter = findInstructionByName(F, "counter");
  Instruction *NextCounter = findInstructionByName(F, "next_counter");
  Instruction *Result = findInstructionByName(F, "result");
  Instruction *BranchVal = findInstructionByName(F, "branch_val");
  Instruction *MergeVal = findInstructionByName(F, "merge_val");
  KnownBits Known32(32);
  Known32.setAllOnes();

  KBCache.emplace_as(ArgN, Known32);
  KBCache.emplace_as(BranchVal, Known32);
  KBCache.emplace_as(NextCounter, Known32);
  EXPECT_EQ(KBCache.size(), 3u);

  EXPECT_FALSE(KBCache.contains(ArgCond));
  EXPECT_FALSE(KBCache.contains(Result));
  EXPECT_FALSE(KBCache.contains(Counter));
  EXPECT_FALSE(KBCache.contains(MergeVal));

  EXPECT_TRUE(KBCache.isAllOnes(ArgN));
  EXPECT_TRUE(KBCache.isAllOnes(BranchVal));
  EXPECT_TRUE(KBCache.isAllOnes(NextCounter));

  KBCache.invalidate(BranchVal);

  EXPECT_FALSE(KBCache.lookup(BranchVal));
  EXPECT_FALSE(KBCache.lookup(NextCounter));
  EXPECT_EQ(KBCache.size(), 3u);

  std::string ActualOutput;
  raw_string_ostream OS(ActualOutput);
  KBCache.print(OS);
  std::string ExpectedOutput =
      R"(^ i32 %n | 11111111111111111111111111111111
$   %next_counter = add i32 %merge_val, 1 | ????????????????????????????????
    %branch_val = mul i32 %counter, 2 | ????????????????????????????????
)";
  EXPECT_EQ(ActualOutput, ExpectedOutput);
}

TEST(KnownBitsDataflow, IRManipulation) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
define void @test(i32 %int_arg, float %float_arg, ptr %ptr_arg, <2 x i32> %vec_int_arg, <2 x ptr> %vec_ptr_arg) {
entry:
  br i1 poison, label %then, label %else
then:
  %int_val = add i32 %int_arg, 1
  %float_val = fadd float %float_arg, 1.0
  %vec_val = add <2 x i32> %vec_int_arg, <i32 1, i32 2>
  br label %merge
else:
  %fpconv = fptoui float %float_arg to i32
  %int_val2 = mul i32 %int_arg, %fpconv
  %ptr_val = getelementptr i8, ptr %ptr_arg, i32 4
  %vec_val2 = mul <2 x i32> %vec_int_arg, <i32 3, i32 4>
  br label %merge
merge:
  %phi_int = phi i32 [ %int_val, %then ], [ %int_val2, %else ]
  %phi_float = phi float [ %float_val, %then ], [ %float_arg, %else ]
  %phi_ptr = phi ptr [ %ptr_arg, %then ], [ %ptr_val, %else ]
  %phi_vec = phi <2 x i32> [ %vec_val, %then ], [ %vec_val2, %else ]
  %final_int = add i32 %phi_int, 5
  %vec_ptr_conv = ptrtoint <2 x ptr> %vec_ptr_arg to <2 x i32>
  %final_vec = add <2 x i32> %phi_vec, %vec_ptr_conv
  %final_dead = mul <2 x i32> %final_vec, splat(i32 2)
  store float %phi_float, ptr %phi_ptr
  ret void
})");
  Function *F = M->getFunction("test");
  DataflowInitializerForTest KBCache(*F);
  Instruction *PhiInt = findInstructionByName(F, "phi_int");
  Instruction *PhiPtr = findInstructionByName(F, "phi_ptr");
  Instruction *FinalVec = findInstructionByName(F, "final_vec");
  Instruction *FinalInt = findInstructionByName(F, "final_int");
  Instruction *VecPtrConv = findInstructionByName(F, "vec_ptr_conv");
  Instruction *FinalDead = findInstructionByName(F, "final_dead");

  EXPECT_EQ(KBCache.size(), 17u);
  EXPECT_THAT(KBCache.computeLeaves(),
              ::testing::UnorderedElementsAre(FinalInt, PhiPtr, FinalDead));
  FinalInt->eraseFromParent();
  FinalDead->eraseFromParent();
  EXPECT_EQ(KBCache.size(), 15u);
  EXPECT_THAT(KBCache.computeLeaves(),
              ::testing::UnorderedElementsAre(PhiPtr, PhiInt, FinalVec));
  VecPtrConv->replaceAllUsesWith(PoisonValue::get(VecPtrConv->getType()));
  EXPECT_EQ(KBCache.size(), 15u);
}

TEST(KnownBitsDataflow, Print) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
define void @test(i32 %n) {
entry:
  br label %loop
loop:
  %phi_counter = phi i32 [ 0, %entry ], [ %next_counter, %loop ]
  %counter = add i32 %phi_counter, 1
  %result = mul i32 %counter, 2
  %next_counter = add i32 %result, 1
  %cond = icmp slt i32 %next_counter, %n
  br i1 %cond, label %loop, label %exit
exit:
  ret void
})");
  Function *F = M->getFunction("test");
  DataflowInitializerForTest KBCache(*F);
  Instruction *Result = findInstructionByName(F, "result");
  KBCache.setAllZero(Result);
  std::string ActualOutput;
  raw_string_ostream OS(ActualOutput);
  KBCache.print(OS);
  std::string ExpectedOutput =
      R"(^ i32 %n | !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
$   %cond = icmp slt i32 %next_counter, %n | !
^   %phi_counter = phi i32 [ 0, %entry ], [ %next_counter, %loop ] | !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    %counter = add i32 %phi_counter, 1 | !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    %result = mul i32 %counter, 2 | 00000000000000000000000000000000
    %next_counter = add i32 %result, 1 | !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
)";
  EXPECT_EQ(ActualOutput, ExpectedOutput);
}
} // namespace
