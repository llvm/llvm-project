//===- LocalOptimization.cpp - Five local optimizations as one LLVM pass --===//
//
// Compiler certification assignment.
//
//   1. constantPropagation  - forward the value last stored to a local
//                             variable into later loads of it (constant and
//                             copy propagation).
//   2. instCombine          - constant folding, algebraic identities, and
//                             multiplication by a power of 2 as a shift.
//   3. deadCodeElimination  - remove unused side-effect-free instructions,
//                             redundant assignments, and stores to variables
//                             that are never read.
//   4. strengthReduction    - replace mul/udiv/urem by a power of 2 with a
//                             cheaper shift or mask.
//   5. commonSubexpressionElimination - reuse an identical expression already
//                             computed earlier in the same basic block.
//
// "Local" means each optimization looks at one basic block at a time; nothing
// is assumed about values flowing in from other blocks.
//
// The pass works directly on clang -O0 output, where every C variable is an
// alloca that is accessed through load and store instructions.
//
//===----------------------------------------------------------------------===//

#include "LocalOptimization.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// A local variable (alloca) whose address never escapes: every use is a
/// plain load from it or a plain store *to* it. Then nothing else - not even
/// a function call - can read or change it, so its contents can be tracked.
static bool isTrackableVariable(const AllocaInst *AI) {
  for (const User *U : AI->users()) {
    if (const auto *LI = dyn_cast<LoadInst>(U)) {
      if (!LI->isSimple()) // volatile or atomic
        return false;
    } else if (const auto *SI = dyn_cast<StoreInst>(U)) {
      // Storing the variable's address somewhere makes it escape.
      if (!SI->isSimple() || SI->getValueOperand() == AI)
        return false;
    } else {
      return false; // e.g. passed to a call: scanf("%d", &x)
    }
  }
  return true;
}

/// Caches isTrackableVariable() for the duration of one optimization.
class VariableTracker {
  DenseMap<const AllocaInst *, bool> Cache;

public:
  /// The trackable variable Ptr points to, or nullptr.
  AllocaInst *get(Value *Ptr) {
    auto *AI = dyn_cast<AllocaInst>(Ptr);
    if (!AI)
      return nullptr;
    auto [It, Inserted] = Cache.try_emplace(AI, false);
    if (Inserted)
      It->second = isTrackableVariable(AI);
    return It->second ? AI : nullptr;
  }
};

static bool isConstInt(const Value *V, uint64_t N) {
  const auto *C = dyn_cast<ConstantInt>(V);
  return C && C->getValue() == N;
}

static bool isAllOnes(const Value *V) {
  const auto *C = dyn_cast<ConstantInt>(V);
  return C && C->getValue().isAllOnes();
}

/// X * 2^N  -->  X << N   (the constant may be either operand).
/// Used by both instCombine and strengthReduction. Returns a new, not yet
/// inserted instruction, or nullptr.
static Instruction *mulByPowerOf2ToShift(BinaryOperator &BO) {
  if (BO.getOpcode() != Instruction::Mul)
    return nullptr;

  Value *X = BO.getOperand(0);
  auto *C = dyn_cast<ConstantInt>(BO.getOperand(1));
  if (!C) {
    X = BO.getOperand(1);
    C = dyn_cast<ConstantInt>(BO.getOperand(0));
  }
  if (!C || !C->getValue().isPowerOf2())
    return nullptr;

  unsigned N = C->getValue().logBase2();
  auto *Shl = BinaryOperator::CreateShl(X, ConstantInt::get(BO.getType(), N));
  // nuw carries over unchanged. nsw does not when 2^N is the sign bit
  // (e.g. i32 2^31): as a signed number that multiplier is negative.
  Shl->setHasNoUnsignedWrap(BO.hasNoUnsignedWrap());
  Shl->setHasNoSignedWrap(BO.hasNoSignedWrap() &&
                          N < C->getBitWidth() - 1);
  return Shl;
}

static std::string toString(const Value &V) {
  std::string S;
  raw_string_ostream OS(S);
  if (isa<Instruction>(V))
    V.print(OS);
  else
    V.printAsOperand(OS, /*PrintType=*/true);
  return StringRef(S).trim().str();
}

void LocalOptimizationPass::replace(StringRef Opt, Instruction &I,
                                    Value *With) {
  std::string Before = Opts.Verbose ? toString(I) : "";
  if (auto *NewI = dyn_cast<Instruction>(With); NewI && !NewI->getParent()) {
    NewI->insertBefore(I.getIterator());
    NewI->takeName(&I);
  }
  if (Opts.Verbose)
    errs() << "[" << Opt << "] " << Before << "   -->   " << toString(*With)
           << "\n";
  I.replaceAllUsesWith(With);
  I.eraseFromParent();
}

void LocalOptimizationPass::erase(StringRef Opt, Instruction &I,
                                  StringRef Why) {
  if (Opts.Verbose)
    errs() << "[" << Opt << "] " << toString(I) << "   -->   removed (" << Why
           << ")\n";
  I.eraseFromParent();
}

//===----------------------------------------------------------------------===//
// 1. Constant propagation
//===----------------------------------------------------------------------===//

/// Walk each basic block remembering the value each local variable currently
/// holds, and replace loads of the variable with that value:
///
///   store i32 2, ptr %a                store i32 2, ptr %a
///   %0 = load i32, ptr %a       -->    %add = add nsw i32 4, 2
///   %add = add nsw i32 4, %0
///
/// When the stored value is a constant this is constant propagation. When it
/// is another variable's value (c = d; e = c + b) it is copy propagation:
/// the load of c is replaced by the value loaded from d.
bool LocalOptimizationPass::constantPropagation(Function &F) {
  bool Changed = false;
  VariableTracker Vars;

  for (BasicBlock &BB : F) {
    // Variable -> the value it currently holds. Starts empty for each block:
    // what a variable holds on entry depends on the predecessor blocks.
    DenseMap<AllocaInst *, Value *> Holds;

    for (Instruction &I : make_early_inc_range(BB)) {
      if (auto *SI = dyn_cast<StoreInst>(&I)) {
        if (AllocaInst *Var = Vars.get(SI->getPointerOperand()))
          Holds[Var] = SI->getValueOperand();
        continue;
      }

      auto *LI = dyn_cast<LoadInst>(&I);
      if (!LI)
        continue;
      AllocaInst *Var = Vars.get(LI->getPointerOperand());
      if (!Var)
        continue;
      auto It = Holds.find(Var);
      if (It == Holds.end()) {
        // Unknown so far, but from now on the variable is known to hold
        // whatever this load produced; later loads can reuse it.
        Holds[Var] = LI;
      } else if (It->second->getType() == LI->getType()) {
        replace("constprop", *LI, It->second);
        Changed = true;
      }
    }
  }
  return Changed;
}

//===----------------------------------------------------------------------===//
// 2. Instruction combining
//===----------------------------------------------------------------------===//

/// Constant folding: C1 op C2 --> C. Returns nullptr when the result is not
/// well defined (division by zero, INT_MIN / -1, shift by >= bit width):
/// that is undefined behaviour and must not be replaced by a made-up value.
static Value *foldConstants(BinaryOperator &BO) {
  auto *C0 = dyn_cast<ConstantInt>(BO.getOperand(0));
  auto *C1 = dyn_cast<ConstantInt>(BO.getOperand(1));
  if (!C0 || !C1)
    return nullptr;

  const APInt &A = C0->getValue();
  const APInt &B = C1->getValue();
  APInt R;
  switch (BO.getOpcode()) {
  case Instruction::Add:
    R = A + B;
    break;
  case Instruction::Sub:
    R = A - B;
    break;
  case Instruction::Mul:
    R = A * B;
    break;
  case Instruction::SDiv:
    if (B.isZero() || (A.isMinSignedValue() && B.isAllOnes()))
      return nullptr;
    R = A.sdiv(B);
    break;
  case Instruction::UDiv:
    if (B.isZero())
      return nullptr;
    R = A.udiv(B);
    break;
  case Instruction::SRem:
    if (B.isZero() || (A.isMinSignedValue() && B.isAllOnes()))
      return nullptr;
    R = A.srem(B);
    break;
  case Instruction::URem:
    if (B.isZero())
      return nullptr;
    R = A.urem(B);
    break;
  case Instruction::Shl:
    if (B.uge(A.getBitWidth()))
      return nullptr;
    R = A.shl(B);
    break;
  case Instruction::LShr:
    if (B.uge(A.getBitWidth()))
      return nullptr;
    R = A.lshr(B);
    break;
  case Instruction::AShr:
    if (B.uge(A.getBitWidth()))
      return nullptr;
    R = A.ashr(B);
    break;
  case Instruction::And:
    R = A & B;
    break;
  case Instruction::Or:
    R = A | B;
    break;
  case Instruction::Xor:
    R = A ^ B;
    break;
  default:
    return nullptr;
  }
  return ConstantInt::get(BO.getType(), R);
}

/// Algebraic identities: the simpler value BO equals, or nullptr.
///
/// X / X --> 1 is correct even though 0 / 0 is not 1: in LLVM IR (and C)
/// division by zero is undefined behaviour, so the compiler may assume X != 0.
static Value *simplifyAlgebraicIdentity(BinaryOperator &BO) {
  Value *X = BO.getOperand(0);
  Value *Y = BO.getOperand(1);
  Type *Ty = BO.getType();

  switch (BO.getOpcode()) {
  case Instruction::Add:
    if (isConstInt(Y, 0)) // X + 0 --> X
      return X;
    if (isConstInt(X, 0)) // 0 + X --> X
      return Y;
    break;
  case Instruction::Sub:
    if (isConstInt(Y, 0)) // X - 0 --> X
      return X;
    if (X == Y) // X - X --> 0
      return ConstantInt::get(Ty, 0);
    break;
  case Instruction::Mul:
    if (isConstInt(Y, 1)) // X * 1 --> X
      return X;
    if (isConstInt(X, 1)) // 1 * X --> X
      return Y;
    if (isConstInt(X, 0) || isConstInt(Y, 0)) // X * 0 --> 0
      return ConstantInt::get(Ty, 0);
    break;
  case Instruction::SDiv:
  case Instruction::UDiv:
    if (isConstInt(Y, 1)) // X / 1 --> X
      return X;
    if (X == Y) // X / X --> 1
      return ConstantInt::get(Ty, 1);
    if (isConstInt(X, 0)) // 0 / X --> 0
      return ConstantInt::get(Ty, 0);
    break;
  case Instruction::SRem:
  case Instruction::URem:
    // X % 1 --> 0,   X % X --> 0,   0 % X --> 0
    if (isConstInt(Y, 1) || X == Y || isConstInt(X, 0))
      return ConstantInt::get(Ty, 0);
    break;
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    if (isConstInt(Y, 0)) // X << 0 --> X
      return X;
    if (isConstInt(X, 0)) // 0 << X --> 0
      return ConstantInt::get(Ty, 0);
    break;
  case Instruction::And:
    if (X == Y || isAllOnes(Y)) // X & X --> X,   X & -1 --> X
      return X;
    if (isAllOnes(X))
      return Y;
    if (isConstInt(X, 0) || isConstInt(Y, 0)) // X & 0 --> 0
      return ConstantInt::get(Ty, 0);
    break;
  case Instruction::Or:
    if (X == Y || isConstInt(Y, 0)) // X | X --> X,   X | 0 --> X
      return X;
    if (isConstInt(X, 0))
      return Y;
    if (isAllOnes(X) || isAllOnes(Y)) // X | -1 --> -1
      return ConstantInt::getAllOnesValue(Ty);
    break;
  case Instruction::Xor:
    if (X == Y) // X ^ X --> 0
      return ConstantInt::get(Ty, 0);
    if (isConstInt(Y, 0)) // X ^ 0 --> X
      return X;
    if (isConstInt(X, 0))
      return Y;
    break;
  default:
    break;
  }
  return nullptr;
}

/// For every integer binary operation, try in order:
///   a. constant folding          2 + 3 --> 5
///   b. algebraic identities      b - b --> 0,  a / a --> 1,  x * 1 --> x
///   c. power of 2 as a shift     x * 8 --> x << 3
bool LocalOptimizationPass::instCombine(Function &F) {
  bool Changed = false;
  for (Instruction &I : make_early_inc_range(instructions(F))) {
    auto *BO = dyn_cast<BinaryOperator>(&I);
    if (!BO || !BO->getType()->isIntegerTy())
      continue;

    Value *New = foldConstants(*BO);
    if (!New)
      New = simplifyAlgebraicIdentity(*BO);
    if (!New)
      New = mulByPowerOf2ToShift(*BO);
    if (New) {
      replace("instcombine", *BO, New);
      Changed = true;
    }
  }
  return Changed;
}

//===----------------------------------------------------------------------===//
// 3. Dead code elimination
//===----------------------------------------------------------------------===//

/// Unused + no side effects --> dead.  Unused + side effects --> keep.
/// (A call to printf is unused but prints, so it is never dead.)
static bool isDead(const Instruction &I) {
  return I.use_empty() && !I.isTerminator() && !I.isEHPad() &&
         !I.mayHaveSideEffects();
}

bool LocalOptimizationPass::deadCodeElimination(Function &F) {
  bool Changed = false;
  VariableTracker Vars;

  // a. Redundant assignments: a store that writes the value the variable
  //    already holds has no effect.   int a = 3; a = 3;   -->   int a = 3;
  for (BasicBlock &BB : F) {
    DenseMap<AllocaInst *, Value *> Holds;
    for (Instruction &I : make_early_inc_range(BB)) {
      if (auto *SI = dyn_cast<StoreInst>(&I)) {
        AllocaInst *Var = Vars.get(SI->getPointerOperand());
        if (!Var)
          continue;
        if (Holds.lookup(Var) == SI->getValueOperand()) {
          erase("dce", *SI, "redundant assignment");
          Changed = true;
        } else {
          Holds[Var] = SI->getValueOperand();
        }
      } else if (auto *LI = dyn_cast<LoadInst>(&I)) {
        if (AllocaInst *Var = Vars.get(LI->getPointerOperand()))
          Holds.try_emplace(Var, LI);
      }
    }
  }

  // b. A store is a side effect, so the rule above never removes one. But a
  //    store to a variable that is never read can not be observed, so the
  //    stores and the variable itself are dead. Their stored values (e.g.
  //    "c = a + b" with c unused) may then become dead too.
  SmallVector<AllocaInst *, 8> NeverRead;
  for (Instruction &I : instructions(F))
    if (auto *AI = dyn_cast<AllocaInst>(&I))
      if (Vars.get(AI) &&
          none_of(AI->users(), [](User *U) { return isa<LoadInst>(U); }))
        NeverRead.push_back(AI);

  SmallSetVector<Instruction *, 16> Worklist;
  for (AllocaInst *AI : NeverRead) {
    for (User *U : make_early_inc_range(AI->users())) {
      auto *SI = cast<StoreInst>(U);
      if (auto *Stored = dyn_cast<Instruction>(SI->getValueOperand()))
        Worklist.insert(Stored);
      erase("dce", *SI, "variable is never read");
    }
    erase("dce", *AI, "variable is never read");
    Changed = true;
  }

  // c. Classic DCE. Removing an instruction may leave its operands unused,
  //    so they are re-examined.
  for (Instruction &I : instructions(F))
    if (isDead(I))
      Worklist.insert(&I);

  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();
    if (!isDead(*I))
      continue;
    SmallVector<Instruction *, 4> Operands;
    for (Value *Op : I->operands())
      if (auto *OpI = dyn_cast<Instruction>(Op))
        Operands.push_back(OpI);
    erase("dce", *I, "unused");
    Changed = true;
    for (Instruction *OpI : Operands)
      if (isDead(*OpI))
        Worklist.insert(OpI);
  }
  return Changed;
}

//===----------------------------------------------------------------------===//
// 4. Strength reduction
//===----------------------------------------------------------------------===//

/// Replace an expensive operation by a power of two with a cheap one:
///   X * 2^N     -->  X << N           2 * c --> c << 1,   f * 8 --> f << 3
///   X udiv 2^N  -->  X >> N           (unsigned only)
///   X urem 2^N  -->  X & (2^N - 1)    (unsigned only)
///
/// Signed division is not changed: -7 / 2 is -3 in C (rounds toward zero),
/// but -7 >> 1 is -4 (rounds down).
bool LocalOptimizationPass::strengthReduction(Function &F) {
  bool Changed = false;
  for (Instruction &I : make_early_inc_range(instructions(F))) {
    auto *BO = dyn_cast<BinaryOperator>(&I);
    if (!BO || !BO->getType()->isIntegerTy())
      continue;

    Instruction *New = mulByPowerOf2ToShift(*BO);
    auto *C = dyn_cast<ConstantInt>(BO->getOperand(1));
    if (!New && C && C->getValue().isPowerOf2()) {
      Value *X = BO->getOperand(0);
      const APInt &P = C->getValue();
      if (BO->getOpcode() == Instruction::UDiv) {
        New = BinaryOperator::CreateLShr(
            X, ConstantInt::get(BO->getType(), P.logBase2()));
        New->setIsExact(BO->isExact());
      } else if (BO->getOpcode() == Instruction::URem) {
        New = BinaryOperator::CreateAnd(X, ConstantInt::get(BO->getType(), P - 1));
      }
    }
    if (New) {
      replace("strength-reduction", *BO, New);
      Changed = true;
    }
  }
  return Changed;
}

//===----------------------------------------------------------------------===//
// 5. Common subexpression elimination
//===----------------------------------------------------------------------===//

/// Pure expressions: the result depends only on the operands, and computing
/// it reads no memory and has no side effects.
static bool isExpression(const Instruction &I) {
  return isa<BinaryOperator, UnaryOperator, CmpInst, CastInst,
             GetElementPtrInst, SelectInst>(I);
}

/// Does New compute the same thing as Old?  b + c also matches c + b.
static bool isSameExpression(Instruction &Old, Instruction &New) {
  if (Old.isIdenticalToWhenDefined(&New))
    return true;
  return isa<BinaryOperator>(New) && New.isCommutative() &&
         Old.getOpcode() == New.getOpcode() &&
         Old.getType() == New.getType() &&
         Old.getOperand(0) == New.getOperand(1) &&
         Old.getOperand(1) == New.getOperand(0);
}

/// In SSA form an operand never changes after it is defined, so if an
/// identical expression was computed earlier in the block, reuse it:
///
///   %a = add i32 %b, %c                %a = add i32 %b, %c
///   %d = add i32 %b, %c        -->     (uses of %d now use %a)
bool LocalOptimizationPass::commonSubexpressionElimination(Function &F) {
  bool Changed = false;
  for (BasicBlock &BB : F) {
    SmallVector<Instruction *, 32> Available;
    for (Instruction &I : make_early_inc_range(BB)) {
      if (!isExpression(I))
        continue;
      auto It = find_if(Available,
                        [&](Instruction *E) { return isSameExpression(*E, I); });
      if (It == Available.end()) {
        Available.push_back(&I);
        continue;
      }
      // The earlier copy now stands for both, so it may only keep the
      // no-overflow flags (nsw, nuw, ...) that both copies had.
      (*It)->andIRFlags(&I);
      replace("cse", I, *It);
      Changed = true;
    }
  }
  return Changed;
}

//===----------------------------------------------------------------------===//
// run()
//===----------------------------------------------------------------------===//

/// Runs the five optimizations in the order the assignment lists them.
///
/// They feed each other: propagation exposes constants to fold, folding
/// makes the original loads dead, and so on. So the whole sequence is
/// repeated until a round changes nothing. That always ends, because every
/// transformation removes an instruction or replaces a mul/udiv/urem with a
/// shift or mask, and neither can happen forever.
PreservedAnalyses LocalOptimizationPass::run(Function &F,
                                             FunctionAnalysisManager &) {
  bool Changed = false;
  bool Progress;
  do {
    Progress = false;
    if (Opts.ConstantPropagation)
      Progress |= constantPropagation(F);
    if (Opts.InstCombine)
      Progress |= instCombine(F);
    if (Opts.DeadCodeElimination)
      Progress |= deadCodeElimination(F);
    if (Opts.StrengthReduction)
      Progress |= strengthReduction(F);
    if (Opts.CommonSubexpressionElimination)
      Progress |= commonSubexpressionElimination(F);
    Changed |= Progress;
  } while (Progress);

  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>(); // Instructions change, basic blocks do not.
  return PA;
}

//===----------------------------------------------------------------------===//
// Plugin registration
//===----------------------------------------------------------------------===//

/// Parses the parameter list of "local-opt<...>". No parameters (or only
/// "verbose") means all five optimizations; otherwise only those listed.
static Expected<LocalOptimizationOptions> parseOptions(StringRef Params) {
  LocalOptimizationOptions All, Selected;
  Selected.ConstantPropagation = Selected.InstCombine =
      Selected.DeadCodeElimination = Selected.StrengthReduction =
          Selected.CommonSubexpressionElimination = false;
  bool AnySelected = false;

  while (!Params.empty()) {
    StringRef Name;
    std::tie(Name, Params) = Params.split(';');
    if (Name == "verbose") {
      All.Verbose = Selected.Verbose = true;
      continue;
    }
    AnySelected = true;
    if (Name == "constprop")
      Selected.ConstantPropagation = true;
    else if (Name == "instcombine")
      Selected.InstCombine = true;
    else if (Name == "dce")
      Selected.DeadCodeElimination = true;
    else if (Name == "strength-reduction")
      Selected.StrengthReduction = true;
    else if (Name == "cse")
      Selected.CommonSubexpressionElimination = true;
    else
      return make_error<StringError>(
          "invalid local-opt parameter '" + Name +
              "' (expected constprop, instcombine, dce, strength-reduction, "
              "cse or verbose)",
          inconvertibleErrorCode());
  }
  return AnySelected ? Selected : All;
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "LocalOptimization", "1.0",
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (!PassBuilder::checkParametrizedPassName(Name,
                                                              "local-opt"))
                    return false;
                  auto Opts = PassBuilder::parsePassParameters(
                      parseOptions, Name, "local-opt");
                  if (!Opts)
                    reportFatalUsageError(Opts.takeError());
                  FPM.addPass(LocalOptimizationPass(*Opts));
                  return true;
                });
          }};
}
