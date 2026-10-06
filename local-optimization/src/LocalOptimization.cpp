//===- LocalOptimization.cpp - Five local optimizations -------------------===//
//
// Compiler certification assignment: five local optimizations, each in its
// own function, executed by run(). "Local" = one basic block at a time.
//
//   opt -load-pass-plugin=LocalOptimization.so -passes=local-opt in.ll
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"

using namespace llvm;

namespace {

struct LocalOptimizationPass : OptionalPassInfoMixin<LocalOptimizationPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &);

  bool constantPropagation(Function &F);
  bool instCombine(Function &F);
  bool deadCodeElimination(Function &F);
  bool strengthReduction(Function &F);
  bool commonSubexpressionElimination(Function &F);
};

// Replace all uses of I with V, then delete I.
void replace(Instruction &I, Value *V) {
  I.replaceAllUsesWith(V);
  I.eraseFromParent();
}

// Is V the integer constant N?
bool isConst(Value *V, uint64_t N) {
  auto *C = dyn_cast<ConstantInt>(V);
  return C && C->getValue() == N;
}

// A store directly into a local variable (alloca), or nullptr.
StoreInst *storeToVariable(Instruction &I) {
  auto *SI = dyn_cast<StoreInst>(&I);
  return SI && isa<AllocaInst>(SI->getPointerOperand()) ? SI : nullptr;
}

// x * 2^N --> x << N.  Used by both instCombine and strengthReduction.
Value *mulToShift(BinaryOperator &BO) {
  if (BO.getOpcode() != Instruction::Mul)
    return nullptr;
  Value *X = BO.getOperand(0);
  auto *C = dyn_cast<ConstantInt>(BO.getOperand(1));
  if (!C) { // constant on the left: 2 * c
    X = BO.getOperand(1);
    C = dyn_cast<ConstantInt>(BO.getOperand(0));
  }
  if (!C || !C->getValue().isPowerOf2())
    return nullptr;
  auto *N = ConstantInt::get(BO.getType(), C->getValue().logBase2());
  auto *Shl = BinaryOperator::CreateShl(X, N, "", BO.getIterator());
  Shl->takeName(&BO);
  return Shl;
}

//===----------------------------------------------------------------------===//
// 1. Constant propagation: a load of a variable is replaced by the value
//    last stored to it (a constant, or another variable's value = copy
//    propagation).
//      store i32 2, ptr %a
//      %0 = load i32, ptr %a      -->   uses of %0 now use 2
//===----------------------------------------------------------------------===//
bool LocalOptimizationPass::constantPropagation(Function &F) {
  bool Changed = false;
  for (BasicBlock &BB : F) {
    DenseMap<Value *, Value *> Holds; // variable -> value it holds
    for (Instruction &I : make_early_inc_range(BB)) {
      if (StoreInst *SI = storeToVariable(I)) {
        Holds[SI->getPointerOperand()] = SI->getValueOperand();
      } else if (auto *LI = dyn_cast<LoadInst>(&I)) {
        Value *V = Holds.lookup(LI->getPointerOperand());
        if (V && V->getType() == LI->getType()) {
          replace(*LI, V);
          Changed = true;
        }
      } else if (I.mayWriteToMemory()) {
        Holds.clear(); // e.g. a call, which may change any variable
      }
    }
  }
  return Changed;
}

//===----------------------------------------------------------------------===//
// 2. Instruction combining:
//    a. constant folding       2 + 3 --> 5
//    b. algebraic identities   x - x --> 0,  x / x --> 1,  x * 1 --> x, ...
//    c. power of 2 as shift    x * 8 --> x << 3
//===----------------------------------------------------------------------===//
Value *foldConstants(BinaryOperator &BO) {
  auto *A = dyn_cast<ConstantInt>(BO.getOperand(0));
  auto *B = dyn_cast<ConstantInt>(BO.getOperand(1));
  if (!A || !B)
    return nullptr;
  const APInt &X = A->getValue(), &Y = B->getValue();
  switch (BO.getOpcode()) {
  case Instruction::Add:
    return ConstantInt::get(BO.getType(), X + Y);
  case Instruction::Sub:
    return ConstantInt::get(BO.getType(), X - Y);
  case Instruction::Mul:
    return ConstantInt::get(BO.getType(), X * Y);
  case Instruction::SDiv:
    if (Y.isZero() || (X.isMinSignedValue() && Y.isAllOnes()))
      return nullptr; // undefined: x / 0, INT_MIN / -1
    return ConstantInt::get(BO.getType(), X.sdiv(Y));
  default:
    return nullptr;
  }
}

Value *simplifyIdentity(BinaryOperator &BO) {
  Value *X = BO.getOperand(0), *Y = BO.getOperand(1);
  switch (BO.getOpcode()) {
  case Instruction::Add: // x + 0 --> x
    return isConst(Y, 0) ? X : isConst(X, 0) ? Y : nullptr;
  case Instruction::Sub: // x - 0 --> x,  x - x --> 0
    if (isConst(Y, 0))
      return X;
    return X == Y ? ConstantInt::get(BO.getType(), 0) : nullptr;
  case Instruction::Mul: // x * 1 --> x
    return isConst(Y, 1) ? X : isConst(X, 1) ? Y : nullptr;
  case Instruction::SDiv: // x / 1 --> x,  x / x --> 1  (x == 0 is undefined)
    if (isConst(Y, 1))
      return X;
    return X == Y ? ConstantInt::get(BO.getType(), 1) : nullptr;
  default:
    return nullptr;
  }
}

bool LocalOptimizationPass::instCombine(Function &F) {
  bool Changed = false;
  for (Instruction &I : make_early_inc_range(instructions(F))) {
    auto *BO = dyn_cast<BinaryOperator>(&I);
    if (!BO)
      continue;
    Value *New = foldConstants(*BO);
    if (!New)
      New = simplifyIdentity(*BO);
    if (!New)
      New = mulToShift(*BO);
    if (New) {
      replace(*BO, New);
      Changed = true;
    }
  }
  return Changed;
}

//===----------------------------------------------------------------------===//
// 3. Dead code elimination:
//    a. unused + no side effects --> removed  (an unused printf() is kept)
//    b. redundant assignment: storing the value the variable already holds
//         a = 3; a = 3;   -->   a = 3;
//===----------------------------------------------------------------------===//
bool LocalOptimizationPass::deadCodeElimination(Function &F) {
  bool Changed = false;
  for (BasicBlock &BB : F) {
    DenseMap<Value *, Value *> Holds; // variable -> value it holds
    for (Instruction &I : make_early_inc_range(BB)) {
      if (StoreInst *SI = storeToVariable(I)) {
        Value *&Held = Holds[SI->getPointerOperand()];
        if (Held == SI->getValueOperand()) {
          SI->eraseFromParent();
          Changed = true;
        } else {
          Held = SI->getValueOperand();
        }
      } else if (I.use_empty() && !I.isTerminator() && !I.isEHPad() &&
                 !I.mayHaveSideEffects()) {
        I.eraseFromParent();
        Changed = true;
      } else if (I.mayWriteToMemory()) {
        Holds.clear();
      }
    }
  }
  return Changed;
}

//===----------------------------------------------------------------------===//
// 4. Strength reduction: multiply by a power of 2 --> cheaper left shift
//      2 * c --> c << 1,   f * 8 --> f << 3
//===----------------------------------------------------------------------===//
bool LocalOptimizationPass::strengthReduction(Function &F) {
  bool Changed = false;
  for (Instruction &I : make_early_inc_range(instructions(F))) {
    auto *BO = dyn_cast<BinaryOperator>(&I);
    if (Value *Shl = BO ? mulToShift(*BO) : nullptr) {
      replace(*BO, Shl);
      Changed = true;
    }
  }
  return Changed;
}

//===----------------------------------------------------------------------===//
// 5. Common subexpression elimination: an expression identical to one
//    computed earlier in the block reuses that result.
//      %a = add i32 %b, %c
//      %d = add i32 %b, %c        -->   uses of %d now use %a
//===----------------------------------------------------------------------===//
bool LocalOptimizationPass::commonSubexpressionElimination(Function &F) {
  bool Changed = false;
  for (BasicBlock &BB : F) {
    SmallVector<Instruction *, 16> Seen;
    for (Instruction &I : make_early_inc_range(BB)) {
      if (!isa<BinaryOperator>(I))
        continue;
      auto It = find_if(Seen, [&](Instruction *E) { return E->isIdenticalTo(&I); });
      if (It != Seen.end()) {
        replace(I, *It);
        Changed = true;
      } else {
        Seen.push_back(&I);
      }
    }
  }
  return Changed;
}

//===----------------------------------------------------------------------===//
// run(): the optimizations create work for each other (propagation exposes
// constants to fold, folding leaves unused loads, ...), so repeat them all
// until nothing changes.
//===----------------------------------------------------------------------===//
PreservedAnalyses LocalOptimizationPass::run(Function &F,
                                             FunctionAnalysisManager &) {
  bool Changed = false, Progress;
  do {
    Progress = constantPropagation(F);
    Progress |= instCombine(F);
    Progress |= deadCodeElimination(F);
    Progress |= strengthReduction(F);
    Progress |= commonSubexpressionElimination(F);
    Changed |= Progress;
  } while (Progress);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

} // namespace

// Makes "-passes=local-opt" available to opt.
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "LocalOptimization", "1.0",
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name != "local-opt")
                    return false;
                  FPM.addPass(LocalOptimizationPass());
                  return true;
                });
          }};
}
