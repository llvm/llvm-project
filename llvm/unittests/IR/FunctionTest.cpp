//===- FunctionTest.cpp - Function unit tests -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Function.h"
#include "llvm-c/Core.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace {

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("InstructionsTests", errs());
  return Mod;
}

static BasicBlock *getBBWithName(Function *F, StringRef Name) {
  auto It = find_if(
      *F, [&Name](const BasicBlock &BB) { return BB.getName() == Name; });
  assert(It != F->end() && "Not found!");
  return &*It;
}

TEST(FunctionTest, hasLazyArguments) {
  LLVMContext C;

  Type *ArgTypes[] = {Type::getInt8Ty(C), Type::getInt32Ty(C)};
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(C), ArgTypes, false);

  // Functions start out with lazy arguments.
  std::unique_ptr<Function> F(
      Function::Create(FTy, GlobalValue::ExternalLinkage, "F"));
  EXPECT_TRUE(F->hasLazyArguments());

  // Checking for empty or size shouldn't force arguments to be instantiated.
  EXPECT_FALSE(F->arg_empty());
  EXPECT_TRUE(F->hasLazyArguments());
  EXPECT_EQ(2u, F->arg_size());
  EXPECT_TRUE(F->hasLazyArguments());

  // The argument list should be populated at first access.
  (void)F->arg_begin();
  EXPECT_FALSE(F->hasLazyArguments());

  // Checking that getArg gets the arguments from F1 in the correct order.
  unsigned i = 0;
  for (Argument &A : F->args()) {
    EXPECT_EQ(&A, F->getArg(i));
    ++i;
  }
  EXPECT_FALSE(F->hasLazyArguments());
}

TEST(FunctionTest, stealArgumentListFrom) {
  LLVMContext C;

  Type *ArgTypes[] = {Type::getInt8Ty(C), Type::getInt32Ty(C)};
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(C), ArgTypes, false);
  std::unique_ptr<Function> F1(
      Function::Create(FTy, GlobalValue::ExternalLinkage, "F1"));
  std::unique_ptr<Function> F2(
      Function::Create(FTy, GlobalValue::ExternalLinkage, "F1"));
  EXPECT_TRUE(F1->hasLazyArguments());
  EXPECT_TRUE(F2->hasLazyArguments());

  // Steal arguments before they've been accessed.  Nothing should change; both
  // functions should still have lazy arguments.
  //
  //   steal(empty); drop (empty)
  F1->stealArgumentListFrom(*F2);
  EXPECT_TRUE(F1->hasLazyArguments());
  EXPECT_TRUE(F2->hasLazyArguments());

  // Save arguments from F1 for later assertions.  F1 won't have lazy arguments
  // anymore.
  SmallVector<Argument *, 4> Args;
  for (Argument &A : F1->args())
    Args.push_back(&A);
  EXPECT_EQ(2u, Args.size());
  EXPECT_FALSE(F1->hasLazyArguments());

  // Steal arguments from F1 to F2.  F1's arguments should be lazy again.
  //
  //   steal(real); drop (empty)
  F2->stealArgumentListFrom(*F1);
  EXPECT_TRUE(F1->hasLazyArguments());
  EXPECT_FALSE(F2->hasLazyArguments());
  unsigned I = 0;
  for (Argument &A : F2->args()) {
    EXPECT_EQ(Args[I], &A);
    I++;
  }
  EXPECT_EQ(2u, I);

  // Check that arguments in F1 don't have pointer equality with the saved ones.
  // This also instantiates F1's arguments.
  I = 0;
  for (Argument &A : F1->args()) {
    EXPECT_NE(Args[I], &A);
    I++;
  }
  EXPECT_EQ(2u, I);
  EXPECT_FALSE(F1->hasLazyArguments());
  EXPECT_FALSE(F2->hasLazyArguments());

  // Steal back from F2.  F2's arguments should be lazy again.
  //
  //   steal(real); drop (real)
  F1->stealArgumentListFrom(*F2);
  EXPECT_FALSE(F1->hasLazyArguments());
  EXPECT_TRUE(F2->hasLazyArguments());
  I = 0;
  for (Argument &A : F1->args()) {
    EXPECT_EQ(Args[I], &A);
    I++;
  }
  EXPECT_EQ(2u, I);

  // Steal from F2 a second time.  Now both functions should have lazy
  // arguments.
  //
  //   steal(empty); drop (real)
  F1->stealArgumentListFrom(*F2);
  EXPECT_TRUE(F1->hasLazyArguments());
  EXPECT_TRUE(F2->hasLazyArguments());
}

// Test setting and removing section information
TEST(FunctionTest, setSection) {
  LLVMContext C;
  Module M("test", C);

  llvm::Function *F =
      Function::Create(llvm::FunctionType::get(llvm::Type::getVoidTy(C), false),
                       llvm::GlobalValue::ExternalLinkage, "F", &M);

  F->setSection(".text.test");
  EXPECT_TRUE(F->getSection() == ".text.test");
  EXPECT_TRUE(F->hasSection());
  F->setSection("");
  EXPECT_FALSE(F->hasSection());
  F->setSection(".text.test");
  F->setSection(".text.test2");
  EXPECT_TRUE(F->getSection() == ".text.test2");
  EXPECT_TRUE(F->hasSection());
}

TEST(FunctionTest, GetPointerAlignment) {
  LLVMContext Context;
  Type *VoidType(Type::getVoidTy(Context));
  FunctionType *FuncType(FunctionType::get(VoidType, false));
  std::unique_ptr<Function> Func(Function::Create(
      FuncType, GlobalValue::ExternalLinkage));
  EXPECT_EQ(Align(1), Func->getPointerAlignment(DataLayout("")));
  EXPECT_EQ(Align(1), Func->getPointerAlignment(DataLayout("Fi8")));
  EXPECT_EQ(Align(1), Func->getPointerAlignment(DataLayout("Fn8")));
  EXPECT_EQ(Align(2), Func->getPointerAlignment(DataLayout("Fi16")));
  EXPECT_EQ(Align(2), Func->getPointerAlignment(DataLayout("Fn16")));
  EXPECT_EQ(Align(4), Func->getPointerAlignment(DataLayout("Fi32")));
  EXPECT_EQ(Align(4), Func->getPointerAlignment(DataLayout("Fn32")));

  Func->setAlignment(Align(4));

  EXPECT_EQ(Align(1), Func->getPointerAlignment(DataLayout("")));
  EXPECT_EQ(Align(1), Func->getPointerAlignment(DataLayout("Fi8")));
  EXPECT_EQ(Align(4), Func->getPointerAlignment(DataLayout("Fn8")));
  EXPECT_EQ(Align(2), Func->getPointerAlignment(DataLayout("Fi16")));
  EXPECT_EQ(Align(4), Func->getPointerAlignment(DataLayout("Fn16")));
  EXPECT_EQ(Align(4), Func->getPointerAlignment(DataLayout("Fi32")));
  EXPECT_EQ(Align(4), Func->getPointerAlignment(DataLayout("Fn32")));
}

TEST(FunctionTest, InsertBasicBlockAt) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"(
define void @foo(i32 %a, i32 %b) {
foo_bb0:
  ret void
}

define void @bar() {
bar_bb0:
  br label %bar_bb1
bar_bb1:
  br label %bar_bb2
bar_bb2:
  ret void
}
)");
  Function *FooF = M->getFunction("foo");
  BasicBlock *FooBB0 = getBBWithName(FooF, "foo_bb0");

  Function *BarF = M->getFunction("bar");
  BasicBlock *BarBB0 = getBBWithName(BarF, "bar_bb0");
  BasicBlock *BarBB1 = getBBWithName(BarF, "bar_bb1");
  BasicBlock *BarBB2 = getBBWithName(BarF, "bar_bb2");

  // Insert foo_bb0 into bar() at the very top.
  FooBB0->removeFromParent();
  auto It = BarF->insert(BarF->begin(), FooBB0);
  EXPECT_EQ(BarBB0->getPrevNode(), FooBB0);
  EXPECT_EQ(It, FooBB0->getIterator());

  // Insert foo_bb0 into bar() at the very end.
  FooBB0->removeFromParent();
  It = BarF->insert(BarF->end(), FooBB0);
  EXPECT_EQ(FooBB0->getPrevNode(), BarBB2);
  EXPECT_EQ(FooBB0->getNextNode(), nullptr);
  EXPECT_EQ(It, FooBB0->getIterator());

  // Insert foo_bb0 into bar() just before bar_bb0.
  FooBB0->removeFromParent();
  It = BarF->insert(BarBB0->getIterator(), FooBB0);
  EXPECT_EQ(FooBB0->getPrevNode(), nullptr);
  EXPECT_EQ(FooBB0->getNextNode(), BarBB0);
  EXPECT_EQ(It, FooBB0->getIterator());

  // Insert foo_bb0 into bar() just before bar_bb1.
  FooBB0->removeFromParent();
  It = BarF->insert(BarBB1->getIterator(), FooBB0);
  EXPECT_EQ(FooBB0->getPrevNode(), BarBB0);
  EXPECT_EQ(FooBB0->getNextNode(), BarBB1);
  EXPECT_EQ(It, FooBB0->getIterator());

  // Insert foo_bb0 into bar() just before bar_bb2.
  FooBB0->removeFromParent();
  It = BarF->insert(BarBB2->getIterator(), FooBB0);
  EXPECT_EQ(FooBB0->getPrevNode(), BarBB1);
  EXPECT_EQ(FooBB0->getNextNode(), BarBB2);
  EXPECT_EQ(It, FooBB0->getIterator());
}

TEST(FunctionTest, SpliceOneBB) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
    define void @from() {
     from_bb1:
       br label %from_bb2
     from_bb2:
       br label %from_bb3
     from_bb3:
       ret void
    }
    define void @to() {
     to_bb1:
       br label %to_bb2
     to_bb2:
       br label %to_bb3
     to_bb3:
       ret void
    }
)");
  Function *FromF = M->getFunction("from");
  BasicBlock *FromBB1 = getBBWithName(FromF, "from_bb1");
  BasicBlock *FromBB2 = getBBWithName(FromF, "from_bb2");
  BasicBlock *FromBB3 = getBBWithName(FromF, "from_bb3");

  Function *ToF = M->getFunction("to");
  BasicBlock *ToBB1 = getBBWithName(ToF, "to_bb1");
  BasicBlock *ToBB2 = getBBWithName(ToF, "to_bb2");
  BasicBlock *ToBB3 = getBBWithName(ToF, "to_bb3");

  // Move from_bb2 before to_bb1.
  ToF->splice(ToBB1->getIterator(), FromF, FromBB2->getIterator());
  EXPECT_EQ(FromF->size(), 2u);
  EXPECT_EQ(ToF->size(), 4u);

  auto It = FromF->begin();
  EXPECT_EQ(&*It++, FromBB1);
  EXPECT_EQ(&*It++, FromBB3);

  It = ToF->begin();
  EXPECT_EQ(&*It++, FromBB2);
  EXPECT_EQ(&*It++, ToBB1);
  EXPECT_EQ(&*It++, ToBB2);
  EXPECT_EQ(&*It++, ToBB3);

  // Cleanup to avoid "Uses remain when a value is destroyed!".
  FromF->splice(FromBB3->getIterator(), ToF, FromBB2->getIterator());
}

TEST(FunctionTest, SpliceOneBBWhenFromIsSameAsTo) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
    define void @fromto() {
     bb1:
       br label %bb2
     bb2:
       ret void
    }
)");
  Function *F = M->getFunction("fromto");
  BasicBlock *BB1 = getBBWithName(F, "bb1");
  BasicBlock *BB2 = getBBWithName(F, "bb2");

  // According to ilist's splice() a single-element splice where dst == src
  // should be a noop.
  F->splice(BB1->getIterator(), F, BB1->getIterator());

  auto It = F->begin();
  EXPECT_EQ(&*It++, BB1);
  EXPECT_EQ(&*It++, BB2);
  EXPECT_EQ(F->size(), 2u);
}

TEST(FunctionTest, SpliceLastBB) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
    define void @from() {
     from_bb1:
       br label %from_bb2
     from_bb2:
       br label %from_bb3
     from_bb3:
       ret void
    }
    define void @to() {
     to_bb1:
       br label %to_bb2
     to_bb2:
       br label %to_bb3
     to_bb3:
       ret void
    }
)");

  Function *FromF = M->getFunction("from");
  BasicBlock *FromBB1 = getBBWithName(FromF, "from_bb1");
  BasicBlock *FromBB2 = getBBWithName(FromF, "from_bb2");
  BasicBlock *FromBB3 = getBBWithName(FromF, "from_bb3");

  Function *ToF = M->getFunction("to");
  BasicBlock *ToBB1 = getBBWithName(ToF, "to_bb1");
  BasicBlock *ToBB2 = getBBWithName(ToF, "to_bb2");
  BasicBlock *ToBB3 = getBBWithName(ToF, "to_bb3");

  // Move from_bb2 before to_bb1.
  auto ToMove = FromBB2->getIterator();
  ToF->splice(ToBB1->getIterator(), FromF, ToMove, std::next(ToMove));

  EXPECT_EQ(FromF->size(), 2u);
  auto It = FromF->begin();
  EXPECT_EQ(&*It++, FromBB1);
  EXPECT_EQ(&*It++, FromBB3);

  EXPECT_EQ(ToF->size(), 4u);
  It = ToF->begin();
  EXPECT_EQ(&*It++, FromBB2);
  EXPECT_EQ(&*It++, ToBB1);
  EXPECT_EQ(&*It++, ToBB2);
  EXPECT_EQ(&*It++, ToBB3);

  // Cleanup to avoid "Uses remain when a value is destroyed!".
  FromF->splice(FromBB3->getIterator(), ToF, ToMove);
}

TEST(FunctionTest, SpliceBBRange) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
    define void @from() {
     from_bb1:
       br label %from_bb2
     from_bb2:
       br label %from_bb3
     from_bb3:
       ret void
    }
    define void @to() {
     to_bb1:
       br label %to_bb2
     to_bb2:
       br label %to_bb3
     to_bb3:
       ret void
    }
)");

  Function *FromF = M->getFunction("from");
  BasicBlock *FromBB1 = getBBWithName(FromF, "from_bb1");
  BasicBlock *FromBB2 = getBBWithName(FromF, "from_bb2");
  BasicBlock *FromBB3 = getBBWithName(FromF, "from_bb3");

  Function *ToF = M->getFunction("to");
  BasicBlock *ToBB1 = getBBWithName(ToF, "to_bb1");
  BasicBlock *ToBB2 = getBBWithName(ToF, "to_bb2");
  BasicBlock *ToBB3 = getBBWithName(ToF, "to_bb3");

  // Move all BBs from @from to @to.
  ToF->splice(ToBB2->getIterator(), FromF, FromF->begin(), FromF->end());

  EXPECT_EQ(FromF->size(), 0u);

  EXPECT_EQ(ToF->size(), 6u);
  auto It = ToF->begin();
  EXPECT_EQ(&*It++, ToBB1);
  EXPECT_EQ(&*It++, FromBB1);
  EXPECT_EQ(&*It++, FromBB2);
  EXPECT_EQ(&*It++, FromBB3);
  EXPECT_EQ(&*It++, ToBB2);
  EXPECT_EQ(&*It++, ToBB3);
}

#ifdef EXPENSIVE_CHECKS
TEST(FunctionTest, SpliceEndBeforeBegin) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
    define void @from() {
     from_bb1:
       br label %from_bb2
     from_bb2:
       br label %from_bb3
     from_bb3:
       ret void
    }
    define void @to() {
     to_bb1:
       br label %to_bb2
     to_bb2:
       br label %to_bb3
     to_bb3:
       ret void
    }
)");

  Function *FromF = M->getFunction("from");
  BasicBlock *FromBB1 = getBBWithName(FromF, "from_bb1");
  BasicBlock *FromBB2 = getBBWithName(FromF, "from_bb2");

  Function *ToF = M->getFunction("to");
  BasicBlock *ToBB2 = getBBWithName(ToF, "to_bb2");

  EXPECT_DEATH(ToF->splice(ToBB2->getIterator(), FromF, FromBB2->getIterator(),
                           FromBB1->getIterator()),
               "FromBeginIt not before FromEndIt!");
}
#endif //EXPENSIVE_CHECKS

TEST(FunctionTest, EraseBBs) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
    define void @foo() {
     bb1:
       br label %bb2
     bb2:
       br label %bb3
     bb3:
       br label %bb4
     bb4:
       br label %bb5
     bb5:
       ret void
    }
)");

  Function *F = M->getFunction("foo");
  BasicBlock *BB1 = getBBWithName(F, "bb1");
  BasicBlock *BB2 = getBBWithName(F, "bb2");
  BasicBlock *BB3 = getBBWithName(F, "bb3");
  BasicBlock *BB4 = getBBWithName(F, "bb4");
  BasicBlock *BB5 = getBBWithName(F, "bb5");
  EXPECT_EQ(F->size(), 5u);

  // Erase BB2.
  BB1->getTerminator()->eraseFromParent();
  auto It = F->erase(BB2->getIterator(), std::next(BB2->getIterator()));
  EXPECT_EQ(F->size(), 4u);
  // Check that the iterator returned matches the node after the erased one.
  EXPECT_EQ(It, BB3->getIterator());

  It = F->begin();
  EXPECT_EQ(&*It++, BB1);
  EXPECT_EQ(&*It++, BB3);
  EXPECT_EQ(&*It++, BB4);
  EXPECT_EQ(&*It++, BB5);

  // Erase all BBs.
  It = F->erase(F->begin(), F->end());
  EXPECT_EQ(F->size(), 0u);
}

TEST(FunctionTest, BasicBlockNumbers) {
  LLVMContext Context;
  Type *VoidType = Type::getVoidTy(Context);
  FunctionType *FuncType = FunctionType::get(VoidType, false);
  std::unique_ptr<Function> Func(
      Function::Create(FuncType, GlobalValue::ExternalLinkage));

  EXPECT_EQ(Func->getBlockNumberEpoch(), 0u);
  EXPECT_EQ(Func->getMaxBlockNumber(), 0u);

  BasicBlock *BB1 = BasicBlock::Create(Context, "bb1", Func.get());
  EXPECT_EQ(BB1->getNumber(), 0u);
  EXPECT_EQ(Func->getMaxBlockNumber(), 1u);
  BasicBlock *BB2 = BasicBlock::Create(Context, "bb2", Func.get());
  EXPECT_EQ(BB2->getNumber(), 1u);
  EXPECT_EQ(Func->getMaxBlockNumber(), 2u);
  BasicBlock *BB3 = BasicBlock::Create(Context, "bb3", Func.get());
  EXPECT_EQ(BB3->getNumber(), 2u);
  EXPECT_EQ(Func->getMaxBlockNumber(), 3u);

  BB2->eraseFromParent();
  // Erasing doesn't trigger renumbering
  EXPECT_EQ(BB1->getNumber(), 0u);
  EXPECT_EQ(BB3->getNumber(), 2u);
  EXPECT_EQ(Func->getMaxBlockNumber(), 3u);
  // ... and number are assigned monotonically increasing
  BasicBlock *BB4 = BasicBlock::Create(Context, "bb4", Func.get());
  EXPECT_EQ(BB4->getNumber(), 3u);
  EXPECT_EQ(Func->getMaxBlockNumber(), 4u);
  // ... even if inserted not at the end
  BasicBlock *BB5 = BasicBlock::Create(Context, "bb5", Func.get(), BB1);
  EXPECT_EQ(BB5->getNumber(), 4u);
  EXPECT_EQ(Func->getMaxBlockNumber(), 5u);

  // Func is now: bb5, bb1, bb3, bb4
  // Renumbering assigns numbers in their order in the function
  EXPECT_EQ(Func->getBlockNumberEpoch(), 0u);
  Func->renumberBlocks();
  EXPECT_EQ(Func->getBlockNumberEpoch(), 1u);
  EXPECT_EQ(BB5->getNumber(), 0u);
  EXPECT_EQ(BB1->getNumber(), 1u);
  EXPECT_EQ(BB3->getNumber(), 2u);
  EXPECT_EQ(BB4->getNumber(), 3u);
  EXPECT_EQ(Func->getMaxBlockNumber(), 4u);

  // Moving a block inside the function doesn't change numbers
  BB1->moveBefore(BB5);
  EXPECT_EQ(BB5->getNumber(), 0u);
  EXPECT_EQ(BB1->getNumber(), 1u);
  EXPECT_EQ(BB3->getNumber(), 2u);
  EXPECT_EQ(BB4->getNumber(), 3u);
  EXPECT_EQ(Func->getMaxBlockNumber(), 4u);

  // Removing a block and adding it back assigns a new number, because the
  // block was temporarily without a parent.
  BB4->removeFromParent();
  BB4->insertInto(Func.get());
  EXPECT_EQ(BB5->getNumber(), 0u);
  EXPECT_EQ(BB1->getNumber(), 1u);
  EXPECT_EQ(BB3->getNumber(), 2u);
  EXPECT_EQ(BB4->getNumber(), 4u);
  EXPECT_EQ(Func->getMaxBlockNumber(), 5u);

  std::unique_ptr<Function> Func2(
      Function::Create(FuncType, GlobalValue::ExternalLinkage));
  BasicBlock *BB6 = BasicBlock::Create(Context, "bb6", Func2.get());
  EXPECT_EQ(BB6->getNumber(), 0u);
  EXPECT_EQ(Func2->getMaxBlockNumber(), 1u);
  // Moving a block to a different function assigns a new number
  BB3->removeFromParent();
  BB3->insertInto(Func2.get(), BB6);
  EXPECT_EQ(BB3->getParent(), Func2.get());
  EXPECT_EQ(BB3->getNumber(), 1u);
  EXPECT_EQ(Func2->getMaxBlockNumber(), 2u);

  EXPECT_EQ(Func2->getBlockNumberEpoch(), 0u);
  Func2->renumberBlocks();
  EXPECT_EQ(Func2->getBlockNumberEpoch(), 1u);
  EXPECT_EQ(BB3->getNumber(), 0u);
  EXPECT_EQ(BB6->getNumber(), 1u);
  EXPECT_EQ(Func2->getMaxBlockNumber(), 2u);

  // splice works as expected and assigns new numbers
  Func->splice(Func->end(), Func2.get());
  EXPECT_EQ(BB5->getNumber(), 0u);
  EXPECT_EQ(BB1->getNumber(), 1u);
  EXPECT_EQ(BB4->getNumber(), 4u);
  EXPECT_EQ(BB3->getNumber(), 5u);
  EXPECT_EQ(BB6->getNumber(), 6u);
  EXPECT_EQ(Func->getMaxBlockNumber(), 7u);
}

TEST(FunctionTest, UWTable) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
    define void @foo() {
     bb1:
       ret void
    }
)");

  Function &F = *M->getFunction("foo");

  EXPECT_FALSE(F.hasUWTable());
  EXPECT_TRUE(F.getUWTableKind() == UWTableKind::None);

  F.setUWTableKind(UWTableKind::Async);
  EXPECT_TRUE(F.hasUWTable());
  EXPECT_TRUE(F.getUWTableKind() == UWTableKind::Async);

  F.setUWTableKind(UWTableKind::None);
  EXPECT_FALSE(F.hasUWTable());
  EXPECT_TRUE(F.getUWTableKind() == UWTableKind::None);
}

TEST(FunctionTest, Personality) {
  LLVMContext Ctx;
  Module M("test", Ctx);
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  FunctionType *FTy = FunctionType::get(Int8Ty, false);
  Function *F = Function::Create(FTy, GlobalValue::ExternalLinkage, "F", &M);
  Function *PersonalityFn =
      Function::Create(FTy, GlobalValue::ExternalLinkage, "PersonalityFn", &M);

  EXPECT_FALSE(F->hasPersonalityFn());
  F->setPersonalityFn(PersonalityFn);
  EXPECT_TRUE(F->hasPersonalityFn());
  EXPECT_EQ(F->getPersonalityFn(), PersonalityFn);
  F->setPersonalityFn(nullptr);
  EXPECT_FALSE(F->hasPersonalityFn());

  EXPECT_FALSE(LLVMHasPersonalityFn(wrap(F)));
  LLVMSetPersonalityFn(wrap(F), wrap(PersonalityFn));
  EXPECT_TRUE(LLVMHasPersonalityFn(wrap(F)));
  EXPECT_EQ(LLVMGetPersonalityFn(wrap(F)), wrap(PersonalityFn));
  LLVMSetPersonalityFn(wrap(F), nullptr);
  EXPECT_FALSE(LLVMHasPersonalityFn(wrap(F)));
}

} // end namespace
