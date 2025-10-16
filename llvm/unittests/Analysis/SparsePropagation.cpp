//===- SparsePropagation.cpp - Unit tests for the generic solver ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/SparsePropagation.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace {
/// To enable interprocedural analysis, we assign LLVM values to the following
/// groups. The register group represents SSA registers, the return group
/// represents the return values of functions, and the memory group represents
/// in-memory values. An LLVM Value can technically be in more than one group.
/// It's necessary to distinguish these groups so we can, for example, track a
/// global variable separately from the value stored at its location.
enum class IPOGrouping { Register, Return, Memory };

/// Our LatticeKeys are PointerIntPairs composed of LLVM values and groupings.
/// The PointerIntPair header provides a DenseMapInfo specialization, so using
/// these as LatticeKeys is fine.
using TestLatticeKey = PointerIntPair<Value *, 2, IPOGrouping>;
} // namespace

namespace llvm {
/// A specialization of LatticeKeyInfo for TestLatticeKeys. The generic solver
/// must translate between LatticeKeys and LLVM Values when adding Values to
/// its work list and inspecting the state of control-flow related values.
template <> struct LatticeKeyInfo<TestLatticeKey> {
  static inline Value *getValueFromLatticeKey(TestLatticeKey Key) {
    return Key.getPointer();
  }
  static inline TestLatticeKey getLatticeKeyFromValue(Value *V) {
    return TestLatticeKey(V, IPOGrouping::Register);
  }
};
} // namespace llvm

namespace {
/// This class defines a simple test lattice value that could be used for
/// solving problems similar to constant propagation. The value is maintained
/// as a PointerIntPair.
class TestLatticeVal {
public:
  /// The states of the lattices value. Only the ConstantVal state is
  /// interesting; the rest are special states used by the generic solver. The
  /// UntrackedVal state differs from the other three in that the generic
  /// solver uses it to avoid doing unnecessary work. In particular, when a
  /// value moves to the UntrackedVal state, it's users are not notified.
  enum TestLatticeStateTy {
    UndefinedVal,
    ConstantVal,
    OverdefinedVal,
    UntrackedVal
  };

  TestLatticeVal() : LatticeVal(nullptr, UndefinedVal) {}
  TestLatticeVal(Constant *C, TestLatticeStateTy State)
      : LatticeVal(C, State) {}

  /// Return true if this lattice value is in the Constant state. This is used
  /// for checking the solver results.
  bool isConstant() const { return LatticeVal.getInt() == ConstantVal; }

  /// Return true if this lattice value is in the Overdefined state. This is
  /// used for checking the solver results.
  bool isOverdefined() const { return LatticeVal.getInt() == OverdefinedVal; }

  bool operator==(const TestLatticeVal &RHS) const {
    return LatticeVal == RHS.LatticeVal;
  }

  bool operator!=(const TestLatticeVal &RHS) const {
    return LatticeVal != RHS.LatticeVal;
  }

private:
  /// A simple lattice value type for problems similar to constant propagation.
  /// It holds the constant value and the lattice state.
  PointerIntPair<const Constant *, 2, TestLatticeStateTy> LatticeVal;
};

/// This class defines a simple test lattice function that could be used for
/// solving problems similar to constant propagation. The test lattice differs
/// from a "real" lattice in a few ways. First, it initializes all return
/// values, values stored in global variables, and arguments in the undefined
/// state. This means that there are no limitations on what we can track
/// interprocedurally. For simplicity, all global values in the tests will be
/// given internal linkage, since this is not something this lattice function
/// tracks. Second, it only handles the few instructions necessary for the
/// tests.
class TestLatticeFunc
    : public AbstractLatticeFunction<TestLatticeKey, TestLatticeVal> {
public:
  /// Construct a new test lattice function with special values for the
  /// Undefined, Overdefined, and Untracked states.
  TestLatticeFunc()
      : AbstractLatticeFunction(
            TestLatticeVal(nullptr, TestLatticeVal::UndefinedVal),
            TestLatticeVal(nullptr, TestLatticeVal::OverdefinedVal),
            TestLatticeVal(nullptr, TestLatticeVal::UntrackedVal)) {}

  /// Compute and return a TestLatticeVal for the given TestLatticeKey. For the
  /// test analysis, a LatticeKey will begin in the undefined state, unless it
  /// represents an LLVM Constant in the register grouping.
  TestLatticeVal ComputeLatticeVal(TestLatticeKey Key) override {
    if (Key.getInt() == IPOGrouping::Register)
      if (auto *C = dyn_cast<Constant>(Key.getPointer()))
        return TestLatticeVal(C, TestLatticeVal::ConstantVal);
    return getUndefVal();
  }

  /// Merge the two given lattice values. This merge should be equivalent to
  /// what is done for constant propagation. That is, the resulting lattice
  /// value is constant only if the two given lattice values are constant and
  /// hold the same value.
  TestLatticeVal MergeValues(TestLatticeVal X, TestLatticeVal Y) override {
    if (X == getUntrackedVal() || Y == getUntrackedVal())
      return getUntrackedVal();
    if (X == getOverdefinedVal() || Y == getOverdefinedVal())
      return getOverdefinedVal();
    if (X == getUndefVal() && Y == getUndefVal())
      return getUndefVal();
    if (X == getUndefVal())
      return Y;
    if (Y == getUndefVal())
      return X;
    if (X == Y)
      return X;
    return getOverdefinedVal();
  }

  /// Compute the lattice values that change as a result of executing the given
  /// instruction. We only handle the few instructions needed for the tests.
  void ComputeInstructionState(
      Instruction &I,
      SmallDenseMap<TestLatticeKey, TestLatticeVal, 16> &ChangedValues,
      SparseSolver<TestLatticeKey, TestLatticeVal> &SS) override {
    switch (I.getOpcode()) {
    case Instruction::Call:
      return visitCallBase(cast<CallBase>(I), ChangedValues, SS);
    case Instruction::Ret:
      return visitReturn(*cast<ReturnInst>(&I), ChangedValues, SS);
    case Instruction::Store:
      return visitStore(*cast<StoreInst>(&I), ChangedValues, SS);
    default:
      return visitInst(I, ChangedValues, SS);
    }
  }

private:
  /// Handle call sites. The state of a called function's argument is the merge
  /// of the current formal argument state with the call site's corresponding
  /// actual argument state. The call site state is the merge of the call site
  /// state with the returned value state of the called function.
  void visitCallBase(CallBase &I,
                     SmallDenseMap<TestLatticeKey, TestLatticeVal, 16> &ChangedValues,
                     SparseSolver<TestLatticeKey, TestLatticeVal> &SS) {
    Function *F = I.getCalledFunction();
    auto RegI = TestLatticeKey(&I, IPOGrouping::Register);
    if (!F) {
      ChangedValues[RegI] = getOverdefinedVal();
      return;
    }
    SS.MarkBlockExecutable(&F->front());
    for (Argument &A : F->args()) {
      auto RegFormal = TestLatticeKey(&A, IPOGrouping::Register);
      auto RegActual =
          TestLatticeKey(I.getArgOperand(A.getArgNo()), IPOGrouping::Register);
      ChangedValues[RegFormal] =
          MergeValues(SS.getValueState(RegFormal), SS.getValueState(RegActual));
    }
    auto RetF = TestLatticeKey(F, IPOGrouping::Return);
    ChangedValues[RegI] =
        MergeValues(SS.getValueState(RegI), SS.getValueState(RetF));
  }

  /// Handle return instructions. The function's return state is the merge of
  /// the returned value state and the function's current return state.
  void visitReturn(ReturnInst &I,
                   SmallDenseMap<TestLatticeKey, TestLatticeVal, 16> &ChangedValues,
                   SparseSolver<TestLatticeKey, TestLatticeVal> &SS) {
    Function *F = I.getParent()->getParent();
    if (F->getReturnType()->isVoidTy())
      return;
    auto RegR = TestLatticeKey(I.getReturnValue(), IPOGrouping::Register);
    auto RetF = TestLatticeKey(F, IPOGrouping::Return);
    ChangedValues[RetF] =
        MergeValues(SS.getValueState(RegR), SS.getValueState(RetF));
  }

  /// Handle store instructions. If the pointer operand of the store is a
  /// global variable, we attempt to track the value. The global variable state
  /// is the merge of the stored value state with the current global variable
  /// state.
  void visitStore(StoreInst &I,
                  SmallDenseMap<TestLatticeKey, TestLatticeVal, 16> &ChangedValues,
                  SparseSolver<TestLatticeKey, TestLatticeVal> &SS) {
    auto *GV = dyn_cast<GlobalVariable>(I.getPointerOperand());
    if (!GV)
      return;
    auto RegVal = TestLatticeKey(I.getValueOperand(), IPOGrouping::Register);
    auto MemPtr = TestLatticeKey(GV, IPOGrouping::Memory);
    ChangedValues[MemPtr] =
        MergeValues(SS.getValueState(RegVal), SS.getValueState(MemPtr));
  }

  /// Handle all other instructions. All other instructions are marked
  /// overdefined.
  void visitInst(Instruction &I,
                 SmallDenseMap<TestLatticeKey, TestLatticeVal, 16> &ChangedValues,
                 SparseSolver<TestLatticeKey, TestLatticeVal> &SS) {
    auto RegI = TestLatticeKey(&I, IPOGrouping::Register);
    ChangedValues[RegI] = getOverdefinedVal();
  }
};

/// This class defines the common data used for all of the tests. The tests
/// should add code to the module and then run the solver.
class SparsePropagationTest : public testing::Test {
protected:
  LLVMContext Context;
  Module M;
  IRBuilder<> Builder;
  TestLatticeFunc Lattice;
  SparseSolver<TestLatticeKey, TestLatticeVal> Solver;

public:
  SparsePropagationTest()
      : M("", Context), Builder(Context), Solver(&Lattice) {}
};
} // namespace

/// Test that we mark discovered functions executable.
///
/// define internal void @f() {
///   call void @g()
///   ret void
/// }
///
/// define internal void @g() {
///   call void @f()
///   ret void
/// }
///
/// For this test, we initially mark "f" executable, and the solver discovers
/// "g" because of the call in "f". The mutually recursive call in "g" also
/// tests that we don't add a block to the basic block work list if it is
/// already executable. Doing so would put the solver into an infinite loop.
TEST_F(SparsePropagationTest, MarkBlockExecutable) {
  Function *F = Function::Create(FunctionType::get(Builder.getVoidTy(), false),
                                 GlobalValue::InternalLinkage, "f", &M);
  Function *G = Function::Create(FunctionType::get(Builder.getVoidTy(), false),
                                 GlobalValue::InternalLinkage, "g", &M);
  BasicBlock *FEntry = BasicBlock::Create(Context, "", F);
  BasicBlock *GEntry = BasicBlock::Create(Context, "", G);
  Builder.SetInsertPoint(FEntry);
  Builder.CreateCall(G);
  Builder.CreateRetVoid();
  Builder.SetInsertPoint(GEntry);
  Builder.CreateCall(F);
  Builder.CreateRetVoid();

  Solver.MarkBlockExecutable(FEntry);
  Solver.Solve();

  EXPECT_TRUE(Solver.isBlockExecutable(GEntry));
}

/// Test that we propagate information through global variables.
///
/// @gv = internal global i64
///
/// define internal void @f() {
///   store i64 1, i64* @gv
///   ret void
/// }
///
/// define internal void @g() {
///   store i64 1, i64* @gv
///   ret void
/// }
///
/// For this test, we initially mark both "f" and "g" executable, and the
/// solver computes the lattice state of the global variable as constant.
TEST_F(SparsePropagationTest, GlobalVariableConstant) {
  Function *F = Function::Create(FunctionType::get(Builder.getVoidTy(), false),
                                 GlobalValue::InternalLinkage, "f", &M);
  Function *G = Function::Create(FunctionType::get(Builder.getVoidTy(), false),
                                 GlobalValue::InternalLinkage, "g", &M);
  GlobalVariable *GV =
      new GlobalVariable(M, Builder.getInt64Ty(), false,
                         GlobalValue::InternalLinkage, nullptr, "gv");
  BasicBlock *FEntry = BasicBlock::Create(Context, "", F);
  BasicBlock *GEntry = BasicBlock::Create(Context, "", G);
  Builder.SetInsertPoint(FEntry);
  Builder.CreateStore(Builder.getInt64(1), GV);
  Builder.CreateRetVoid();
  Builder.SetInsertPoint(GEntry);
  Builder.CreateStore(Builder.getInt64(1), GV);
  Builder.CreateRetVoid();

  Solver.MarkBlockExecutable(FEntry);
  Solver.MarkBlockExecutable(GEntry);
  Solver.Solve();

  auto MemGV = TestLatticeKey(GV, IPOGrouping::Memory);
  EXPECT_TRUE(Solver.getExistingValueState(MemGV).isConstant());
}

/// Test that we propagate information through global variables.
///
/// @gv = internal global i64
///
/// define internal void @f() {
///   store i64 0, i64* @gv
///   ret void
/// }
///
/// define internal void @g() {
///   store i64 1, i64* @gv
///   ret void
/// }
///
/// For this test, we initially mark both "f" and "g" executable, and the
/// solver computes the lattice state of the global variable as overdefined.
TEST_F(SparsePropagationTest, GlobalVariableOverDefined) {
  Function *F = Function::Create(FunctionType::get(Builder.getVoidTy(), false),
                                 GlobalValue::InternalLinkage, "f", &M);
  Function *G = Function::Create(FunctionType::get(Builder.getVoidTy(), false),
                                 GlobalValue::InternalLinkage, "g", &M);
  GlobalVariable *GV =
      new GlobalVariable(M, Builder.getInt64Ty(), false,
                         GlobalValue::InternalLinkage, nullptr, "gv");
  BasicBlock *FEntry = BasicBlock::Create(Context, "", F);
  BasicBlock *GEntry = BasicBlock::Create(Context, "", G);
  Builder.SetInsertPoint(FEntry);
  Builder.CreateStore(Builder.getInt64(0), GV);
  Builder.CreateRetVoid();
  Builder.SetInsertPoint(GEntry);
  Builder.CreateStore(Builder.getInt64(1), GV);
  Builder.CreateRetVoid();

  Solver.MarkBlockExecutable(FEntry);
  Solver.MarkBlockExecutable(GEntry);
  Solver.Solve();

  auto MemGV = TestLatticeKey(GV, IPOGrouping::Memory);
  EXPECT_TRUE(Solver.getExistingValueState(MemGV).isOverdefined());
}

/// Test that we propagate information through function returns.
///
/// define internal i64 @f(i1* %cond) {
/// if:
///   %0 = load i1, i1* %cond
///   br i1 %0, label %then, label %else
///
/// then:
///   ret i64 1
///
/// else:
///   ret i64 1
/// }
///
/// For this test, we initially mark "f" executable, and the solver computes
/// the return value of the function as constant.
TEST_F(SparsePropagationTest, FunctionDefined) {
  Function *F =
      Function::Create(FunctionType::get(Builder.getInt64Ty(),
                                         {PointerType::get(Context, 0)}, false),
                       GlobalValue::InternalLinkage, "f", &M);
  BasicBlock *If = BasicBlock::Create(Context, "if", F);
  BasicBlock *Then = BasicBlock::Create(Context, "then", F);
  BasicBlock *Else = BasicBlock::Create(Context, "else", F);
  F->arg_begin()->setName("cond");
  Builder.SetInsertPoint(If);
  LoadInst *Cond = Builder.CreateLoad(Type::getInt1Ty(Context), F->arg_begin());
  Builder.CreateCondBr(Cond, Then, Else);
  Builder.SetInsertPoint(Then);
  Builder.CreateRet(Builder.getInt64(1));
  Builder.SetInsertPoint(Else);
  Builder.CreateRet(Builder.getInt64(1));

  Solver.MarkBlockExecutable(If);
  Solver.Solve();

  auto RetF = TestLatticeKey(F, IPOGrouping::Return);
  EXPECT_TRUE(Solver.getExistingValueState(RetF).isConstant());
}

/// Test that we propagate information through function returns.
///
/// define internal i64 @f(i1* %cond) {
/// if:
///   %0 = load i1, i1* %cond
///   br i1 %0, label %then, label %else
///
/// then:
///   ret i64 0
///
/// else:
///   ret i64 1
/// }
///
/// For this test, we initially mark "f" executable, and the solver computes
/// the return value of the function as overdefined.
TEST_F(SparsePropagationTest, FunctionOverDefined) {
  Function *F =
      Function::Create(FunctionType::get(Builder.getInt64Ty(),
                                         {PointerType::get(Context, 0)}, false),
                       GlobalValue::InternalLinkage, "f", &M);
  BasicBlock *If = BasicBlock::Create(Context, "if", F);
  BasicBlock *Then = BasicBlock::Create(Context, "then", F);
  BasicBlock *Else = BasicBlock::Create(Context, "else", F);
  F->arg_begin()->setName("cond");
  Builder.SetInsertPoint(If);
  LoadInst *Cond = Builder.CreateLoad(Type::getInt1Ty(Context), F->arg_begin());
  Builder.CreateCondBr(Cond, Then, Else);
  Builder.SetInsertPoint(Then);
  Builder.CreateRet(Builder.getInt64(0));
  Builder.SetInsertPoint(Else);
  Builder.CreateRet(Builder.getInt64(1));

  Solver.MarkBlockExecutable(If);
  Solver.Solve();

  auto RetF = TestLatticeKey(F, IPOGrouping::Return);
  EXPECT_TRUE(Solver.getExistingValueState(RetF).isOverdefined());
}

/// Test that we propagate information through arguments.
///
/// define internal void @f() {
///   call void @g(i64 0, i64 1)
///   call void @g(i64 1, i64 1)
///   ret void
/// }
///
/// define internal void @g(i64 %a, i64 %b) {
///   ret void
/// }
///
/// For this test, we initially mark "f" executable, and the solver discovers
/// "g" because of the calls in "f". The solver computes the state of argument
/// "a" as overdefined and the state of "b" as constant.
///
/// In addition, this test demonstrates that ComputeInstructionState can alter
/// the state of multiple lattice values, in addition to the one associated
/// with the instruction definition. Each call instruction in this test updates
/// the state of arguments "a" and "b".
TEST_F(SparsePropagationTest, ComputeInstructionState) {
  Function *F = Function::Create(FunctionType::get(Builder.getVoidTy(), false),
                                 GlobalValue::InternalLinkage, "f", &M);
  Function *G = Function::Create(
      FunctionType::get(Builder.getVoidTy(),
                        {Builder.getInt64Ty(), Builder.getInt64Ty()}, false),
      GlobalValue::InternalLinkage, "g", &M);
  Argument *A = G->arg_begin();
  Argument *B = std::next(G->arg_begin());
  A->setName("a");
  B->setName("b");
  BasicBlock *FEntry = BasicBlock::Create(Context, "", F);
  BasicBlock *GEntry = BasicBlock::Create(Context, "", G);
  Builder.SetInsertPoint(FEntry);
  Builder.CreateCall(G, {Builder.getInt64(0), Builder.getInt64(1)});
  Builder.CreateCall(G, {Builder.getInt64(1), Builder.getInt64(1)});
  Builder.CreateRetVoid();
  Builder.SetInsertPoint(GEntry);
  Builder.CreateRetVoid();

  Solver.MarkBlockExecutable(FEntry);
  Solver.Solve();

  auto RegA = TestLatticeKey(A, IPOGrouping::Register);
  auto RegB = TestLatticeKey(B, IPOGrouping::Register);
  EXPECT_TRUE(Solver.getExistingValueState(RegA).isOverdefined());
  EXPECT_TRUE(Solver.getExistingValueState(RegB).isConstant());
}

/// Test that we can handle exceptional terminator instructions.
///
/// declare internal void @p()
///
/// declare internal void @g()
///
/// define internal void @f() personality ptr @p {
/// entry:
///   invoke void @g()
///           to label %exit unwind label %catch.pad
///
/// catch.pad:
///   %0 = catchswitch within none [label %catch.body] unwind to caller
///
/// catch.body:
///   %1 = catchpad within %0 []
///   catchret from %1 to label %exit
///
/// exit:
///   ret void
/// }
///
/// For this test, we initially mark the entry block executable. The solver
/// then discovers the rest of the blocks in the function are executable.
TEST_F(SparsePropagationTest, ExceptionalTerminatorInsts) {
  Function *P = Function::Create(FunctionType::get(Builder.getVoidTy(), false),
                                 GlobalValue::InternalLinkage, "p", &M);
  Function *G = Function::Create(FunctionType::get(Builder.getVoidTy(), false),
                                 GlobalValue::InternalLinkage, "g", &M);
  Function *F = Function::Create(FunctionType::get(Builder.getVoidTy(), false),
                                 GlobalValue::InternalLinkage, "f", &M);
  F->setPersonalityFn(P);
  BasicBlock *Entry = BasicBlock::Create(Context, "entry", F);
  BasicBlock *Pad = BasicBlock::Create(Context, "catch.pad", F);
  BasicBlock *Body = BasicBlock::Create(Context, "catch.body", F);
  BasicBlock *Exit = BasicBlock::Create(Context, "exit", F);
  Builder.SetInsertPoint(Entry);
  Builder.CreateInvoke(G, Exit, Pad);
  Builder.SetInsertPoint(Pad);
  CatchSwitchInst *CatchSwitch =
      Builder.CreateCatchSwitch(ConstantTokenNone::get(Context), nullptr, 1);
  CatchSwitch->addHandler(Body);
  Builder.SetInsertPoint(Body);
  CatchPadInst *CatchPad = Builder.CreateCatchPad(CatchSwitch, {});
  Builder.CreateCatchRet(CatchPad, Exit);
  Builder.SetInsertPoint(Exit);
  Builder.CreateRetVoid();

  Solver.MarkBlockExecutable(Entry);
  Solver.Solve();

  EXPECT_TRUE(Solver.isBlockExecutable(Pad));
  EXPECT_TRUE(Solver.isBlockExecutable(Body));
  EXPECT_TRUE(Solver.isBlockExecutable(Exit));
}
