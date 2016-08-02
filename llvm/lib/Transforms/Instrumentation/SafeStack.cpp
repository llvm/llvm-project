//===-- SafeStack.cpp - Safe Stack Insertion ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass splits the stack into the safe stack (kept as-is for LLVM backend)
// and the unsafe stack (explicitly allocated and managed through the runtime
// support library).
//
// http://clang.llvm.org/docs/SafeStack.html
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "safestack"

enum UnsafeStackPtrStorageVal { ThreadLocalUSP, SingleThreadUSP };

static cl::opt<UnsafeStackPtrStorageVal> USPStorage("safe-stack-usp-storage",
    cl::Hidden, cl::init(ThreadLocalUSP),
    cl::desc("Type of storage for the unsafe stack pointer"),
    cl::values(clEnumValN(ThreadLocalUSP, "thread-local",
                          "Thread-local storage"),
               clEnumValN(SingleThreadUSP, "single-thread",
                          "Non-thread-local storage"),
               clEnumValEnd));

namespace llvm {

STATISTIC(NumFunctions, "Total number of functions");
STATISTIC(NumUnsafeStackFunctions, "Number of functions with unsafe stack");
STATISTIC(NumUnsafeStackRestorePointsFunctions,
          "Number of functions that use setjmp or exceptions");

STATISTIC(NumAllocas, "Total number of allocas");
STATISTIC(NumUnsafeStaticAllocas, "Number of unsafe static allocas");
STATISTIC(NumUnsafeDynamicAllocas, "Number of unsafe dynamic allocas");
STATISTIC(NumUnsafeByValArguments, "Number of unsafe byval arguments");
STATISTIC(NumUnsafeStackRestorePoints, "Number of setjmps and landingpads");

} // namespace llvm

namespace {

/// Rewrite an SCEV expression for a memory access address to an expression that
/// represents offset from the given alloca.
///
/// The implementation simply replaces all mentions of the alloca with zero.
class AllocaOffsetRewriter : public SCEVRewriteVisitor<AllocaOffsetRewriter> {
  const Value *AllocaPtr;

public:
  AllocaOffsetRewriter(ScalarEvolution &SE, const Value *AllocaPtr)
      : SCEVRewriteVisitor(SE), AllocaPtr(AllocaPtr) {}

  const SCEV *visitUnknown(const SCEVUnknown *Expr) {
    if (Expr->getValue() == AllocaPtr)
      return SE.getZero(Expr->getType());
    return Expr;
  }
};

/// The SafeStack pass splits the stack of each function into the safe
/// stack, which is only accessed through memory safe dereferences (as
/// determined statically), and the unsafe stack, which contains all
/// local variables that are accessed in ways that we can't prove to
/// be safe.
class SafeStack : public FunctionPass {
  const TargetMachine *TM;
  const TargetLoweringBase *TL;
  const DataLayout *DL;
  ScalarEvolution *SE;

  Type *StackPtrTy;
  Type *IntPtrTy;
  Type *Int32Ty;
  Type *Int8Ty;

  Value *UnsafeStackPtr = nullptr;

  /// Unsafe stack alignment. Each stack frame must ensure that the stack is
  /// aligned to this value. We need to re-align the unsafe stack if the
  /// alignment of any object on the stack exceeds this value.
  ///
  /// 16 seems like a reasonable upper bound on the alignment of objects that we
  /// might expect to appear on the stack on most common targets.
  enum { StackAlignment = 16 };

  /// \brief Build a value representing a pointer to the unsafe stack pointer.
  Value *getOrCreateUnsafeStackPtr(IRBuilder<> &IRB, Function &F);

  /// \brief Find all static allocas, dynamic allocas, return instructions and
  /// stack restore points (exception unwind blocks and setjmp calls) in the
  /// given function and append them to the respective vectors.
  void findInsts(Function &F, SmallVectorImpl<AllocaInst *> &StaticAllocas,
                 SmallVectorImpl<AllocaInst *> &DynamicAllocas,
                 SmallVectorImpl<Argument *> &ByValArguments,
                 SmallVectorImpl<ReturnInst *> &Returns,
                 SmallVectorImpl<Instruction *> &StackRestorePoints);

  /// \brief Calculate the allocation size of a given alloca. Returns 0 if the
  /// size can not be statically determined.
  uint64_t getStaticAllocaAllocationSize(const AllocaInst* AI);

  /// \brief Allocate space for all static allocas in \p StaticAllocas,
  /// replace allocas with pointers into the unsafe stack and generate code to
  /// restore the stack pointer before all return instructions in \p Returns.
  ///
  /// \returns A pointer to the top of the unsafe stack after all unsafe static
  /// allocas are allocated.
  Value *moveStaticAllocasToUnsafeStack(IRBuilder<> &IRB, Function &F,
                                        ArrayRef<AllocaInst *> StaticAllocas,
                                        ArrayRef<Argument *> ByValArguments,
                                        ArrayRef<ReturnInst *> Returns);

  /// \brief Generate code to restore the stack after all stack restore points
  /// in \p StackRestorePoints.
  ///
  /// \returns A local variable in which to maintain the dynamic top of the
  /// unsafe stack if needed.
  AllocaInst *
  createStackRestorePoints(IRBuilder<> &IRB, Function &F,
                           ArrayRef<Instruction *> StackRestorePoints,
                           Value *StaticTop, bool NeedDynamicTop);

  /// \brief Replace all allocas in \p DynamicAllocas with code to allocate
  /// space dynamically on the unsafe stack and store the dynamic unsafe stack
  /// top to \p DynamicTop if non-null.
  void moveDynamicAllocasToUnsafeStack(Function &F, Value *UnsafeStackPtr,
                                       AllocaInst *DynamicTop,
                                       ArrayRef<AllocaInst *> DynamicAllocas);

  bool IsSafeStackAlloca(const Value *AllocaPtr, uint64_t AllocaSize);

  bool IsMemIntrinsicSafe(const MemIntrinsic *MI, const Use &U,
                          const Value *AllocaPtr, uint64_t AllocaSize);
  bool IsAccessSafe(Value *Addr, uint64_t Size, const Value *AllocaPtr,
                    uint64_t AllocaSize);

public:
  static char ID; // Pass identification, replacement for typeid.
  SafeStack(const TargetMachine *TM)
      : FunctionPass(ID), TM(TM), TL(nullptr), DL(nullptr) {
    initializeSafeStackPass(*PassRegistry::getPassRegistry());
  }
  SafeStack() : SafeStack(nullptr) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<ScalarEvolutionWrapperPass>();
  }

  bool doInitialization(Module &M) override {
    DL = &M.getDataLayout();

    StackPtrTy = Type::getInt8PtrTy(M.getContext());
    IntPtrTy = DL->getIntPtrType(M.getContext());
    Int32Ty = Type::getInt32Ty(M.getContext());
    Int8Ty = Type::getInt8Ty(M.getContext());

    return false;
  }

  bool runOnFunction(Function &F) override;
}; // class SafeStack

uint64_t SafeStack::getStaticAllocaAllocationSize(const AllocaInst* AI) {
  uint64_t Size = DL->getTypeAllocSize(AI->getAllocatedType());
  if (AI->isArrayAllocation()) {
    auto C = dyn_cast<ConstantInt>(AI->getArraySize());
    if (!C)
      return 0;
    Size *= C->getZExtValue();
  }
  return Size;
}

bool SafeStack::IsAccessSafe(Value *Addr, uint64_t AccessSize,
                             const Value *AllocaPtr, uint64_t AllocaSize) {
  AllocaOffsetRewriter Rewriter(*SE, AllocaPtr);
  const SCEV *Expr = Rewriter.visit(SE->getSCEV(Addr));

  uint64_t BitWidth = SE->getTypeSizeInBits(Expr->getType());
  ConstantRange AccessStartRange = SE->getUnsignedRange(Expr);
  ConstantRange SizeRange =
      ConstantRange(APInt(BitWidth, 0), APInt(BitWidth, AccessSize));
  ConstantRange AccessRange = AccessStartRange.add(SizeRange);
  ConstantRange AllocaRange =
      ConstantRange(APInt(BitWidth, 0), APInt(BitWidth, AllocaSize));
  bool Safe = AllocaRange.contains(AccessRange);

  DEBUG(dbgs() << "[SafeStack] "
               << (isa<AllocaInst>(AllocaPtr) ? "Alloca " : "ByValArgument ")
               << *AllocaPtr << "\n"
               << "            Access " << *Addr << "\n"
               << "            SCEV " << *Expr
               << " U: " << SE->getUnsignedRange(Expr)
               << ", S: " << SE->getSignedRange(Expr) << "\n"
               << "            Range " << AccessRange << "\n"
               << "            AllocaRange " << AllocaRange << "\n"
               << "            " << (Safe ? "safe" : "unsafe") << "\n");

  return Safe;
}

bool SafeStack::IsMemIntrinsicSafe(const MemIntrinsic *MI, const Use &U,
                                   const Value *AllocaPtr,
                                   uint64_t AllocaSize) {
  // All MemIntrinsics have destination address in Arg0 and size in Arg2.
  if (MI->getRawDest() != U) return true;
  const auto *Len = dyn_cast<ConstantInt>(MI->getLength());
  // Non-constant size => unsafe. FIXME: try SCEV getRange.
  if (!Len) return false;
  return IsAccessSafe(U, Len->getZExtValue(), AllocaPtr, AllocaSize);
}

/// Check whether a given allocation must be put on the safe
/// stack or not. The function analyzes all uses of AI and checks whether it is
/// only accessed in a memory safe way (as decided statically).
bool SafeStack::IsSafeStackAlloca(const Value *AllocaPtr, uint64_t AllocaSize) {
  // Go through all uses of this alloca and check whether all accesses to the
  // allocated object are statically known to be memory safe and, hence, the
  // object can be placed on the safe stack.
  SmallPtrSet<const Value *, 16> Visited;
  SmallVector<const Value *, 8> WorkList;
  WorkList.push_back(AllocaPtr);

  // A DFS search through all uses of the alloca in bitcasts/PHI/GEPs/etc.
  while (!WorkList.empty()) {
    const Value *V = WorkList.pop_back_val();
    for (const Use &UI : V->uses()) {
      auto I = cast<const Instruction>(UI.getUser());
      assert(V == UI.get());

      switch (I->getOpcode()) {
      case Instruction::Load: {
        if (!IsAccessSafe(UI, DL->getTypeStoreSize(I->getType()), AllocaPtr,
                          AllocaSize))
          return false;
        break;
      }
      case Instruction::VAArg:
        // "va-arg" from a pointer is safe.
        break;
      case Instruction::Store: {
        if (V == I->getOperand(0)) {
          // Stored the pointer - conservatively assume it may be unsafe.
          DEBUG(dbgs() << "[SafeStack] Unsafe alloca: " << *AllocaPtr
                       << "\n            store of address: " << *I << "\n");
          return false;
        }

        if (!IsAccessSafe(UI, DL->getTypeStoreSize(I->getOperand(0)->getType()),
                          AllocaPtr, AllocaSize))
          return false;
        break;
      }
      case Instruction::Ret: {
        // Information leak.
        return false;
      }

      case Instruction::Call:
      case Instruction::Invoke: {
        ImmutableCallSite CS(I);

        if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
          if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
              II->getIntrinsicID() == Intrinsic::lifetime_end)
            continue;
        }

        if (const MemIntrinsic *MI = dyn_cast<MemIntrinsic>(I)) {
          if (!IsMemIntrinsicSafe(MI, UI, AllocaPtr, AllocaSize)) {
            DEBUG(dbgs() << "[SafeStack] Unsafe alloca: " << *AllocaPtr
                         << "\n            unsafe memintrinsic: " << *I
                         << "\n");
            return false;
          }
          continue;
        }

        // LLVM 'nocapture' attribute is only set for arguments whose address
        // is not stored, passed around, or used in any other non-trivial way.
        // We assume that passing a pointer to an object as a 'nocapture
        // readnone' argument is safe.
        // FIXME: a more precise solution would require an interprocedural
        // analysis here, which would look at all uses of an argument inside
        // the function being called.
        ImmutableCallSite::arg_iterator B = CS.arg_begin(), E = CS.arg_end();
        for (ImmutableCallSite::arg_iterator A = B; A != E; ++A)
          if (A->get() == V)
            if (!(CS.doesNotCapture(A - B) && (CS.doesNotAccessMemory(A - B) ||
                                               CS.doesNotAccessMemory()))) {
              DEBUG(dbgs() << "[SafeStack] Unsafe alloca: " << *AllocaPtr
                           << "\n            unsafe call: " << *I << "\n");
              return false;
            }
        continue;
      }

      default:
        if (Visited.insert(I).second)
          WorkList.push_back(cast<const Instruction>(I));
      }
    }
  }

  // All uses of the alloca are safe, we can place it on the safe stack.
  return true;
}

Value *SafeStack::getOrCreateUnsafeStackPtr(IRBuilder<> &IRB, Function &F) {
  // Check if there is a target-specific location for the unsafe stack pointer.
  if (TL)
    if (Value *V = TL->getSafeStackPointerLocation(IRB))
      return V;

  // Otherwise, assume the target links with compiler-rt, which provides a
  // thread-local variable with a magic name.
  Module &M = *F.getParent();
  const char *UnsafeStackPtrVar = "__safestack_unsafe_stack_ptr";
  auto UnsafeStackPtr =
      dyn_cast_or_null<GlobalVariable>(M.getNamedValue(UnsafeStackPtrVar));

  bool UseTLS = USPStorage == ThreadLocalUSP;

  if (!UnsafeStackPtr) {
    auto TLSModel = UseTLS ?
        GlobalValue::InitialExecTLSModel :
        GlobalValue::NotThreadLocal;
    // The global variable is not defined yet, define it ourselves.
    // We use the initial-exec TLS model because we do not support the
    // variable living anywhere other than in the main executable.
    UnsafeStackPtr = new GlobalVariable(
        M, StackPtrTy, false, GlobalValue::ExternalLinkage, nullptr,
        UnsafeStackPtrVar, nullptr, TLSModel);
  } else {
    // The variable exists, check its type and attributes.
    if (UnsafeStackPtr->getValueType() != StackPtrTy)
      report_fatal_error(Twine(UnsafeStackPtrVar) + " must have void* type");
    if (UseTLS != UnsafeStackPtr->isThreadLocal())
      report_fatal_error(Twine(UnsafeStackPtrVar) + " must " +
                         (UseTLS ? "" : "not ") + "be thread-local");
  }
  return UnsafeStackPtr;
}

void SafeStack::findInsts(Function &F,
                          SmallVectorImpl<AllocaInst *> &StaticAllocas,
                          SmallVectorImpl<AllocaInst *> &DynamicAllocas,
                          SmallVectorImpl<Argument *> &ByValArguments,
                          SmallVectorImpl<ReturnInst *> &Returns,
                          SmallVectorImpl<Instruction *> &StackRestorePoints) {
  for (Instruction &I : instructions(&F)) {
    if (auto AI = dyn_cast<AllocaInst>(&I)) {
      ++NumAllocas;

      uint64_t Size = getStaticAllocaAllocationSize(AI);
      if (IsSafeStackAlloca(AI, Size))
        continue;

      if (AI->isStaticAlloca()) {
        ++NumUnsafeStaticAllocas;
        StaticAllocas.push_back(AI);
      } else {
        ++NumUnsafeDynamicAllocas;
        DynamicAllocas.push_back(AI);
      }
    } else if (auto RI = dyn_cast<ReturnInst>(&I)) {
      Returns.push_back(RI);
    } else if (auto CI = dyn_cast<CallInst>(&I)) {
      // setjmps require stack restore.
      if (CI->getCalledFunction() && CI->canReturnTwice())
        StackRestorePoints.push_back(CI);
    } else if (auto LP = dyn_cast<LandingPadInst>(&I)) {
      // Exception landing pads require stack restore.
      StackRestorePoints.push_back(LP);
    } else if (auto II = dyn_cast<IntrinsicInst>(&I)) {
      if (II->getIntrinsicID() == Intrinsic::gcroot)
        llvm::report_fatal_error(
            "gcroot intrinsic not compatible with safestack attribute");
    }
  }
  for (Argument &Arg : F.args()) {
    if (!Arg.hasByValAttr())
      continue;
    uint64_t Size =
        DL->getTypeStoreSize(Arg.getType()->getPointerElementType());
    if (IsSafeStackAlloca(&Arg, Size))
      continue;

    ++NumUnsafeByValArguments;
    ByValArguments.push_back(&Arg);
  }
}

AllocaInst *
SafeStack::createStackRestorePoints(IRBuilder<> &IRB, Function &F,
                                    ArrayRef<Instruction *> StackRestorePoints,
                                    Value *StaticTop, bool NeedDynamicTop) {
  if (StackRestorePoints.empty())
    return nullptr;

  // We need the current value of the shadow stack pointer to restore
  // after longjmp or exception catching.

  // FIXME: On some platforms this could be handled by the longjmp/exception
  // runtime itself.

  AllocaInst *DynamicTop = nullptr;
  if (NeedDynamicTop)
    // If we also have dynamic alloca's, the stack pointer value changes
    // throughout the function. For now we store it in an alloca.
    DynamicTop = IRB.CreateAlloca(StackPtrTy, /*ArraySize=*/nullptr,
                                  "unsafe_stack_dynamic_ptr");

  if (!StaticTop)
    // We need the original unsafe stack pointer value, even if there are
    // no unsafe static allocas.
    StaticTop = IRB.CreateLoad(UnsafeStackPtr, false, "unsafe_stack_ptr");

  if (NeedDynamicTop)
    IRB.CreateStore(StaticTop, DynamicTop);

  // Restore current stack pointer after longjmp/exception catch.
  for (Instruction *I : StackRestorePoints) {
    ++NumUnsafeStackRestorePoints;

    IRB.SetInsertPoint(I->getNextNode());
    Value *CurrentTop = DynamicTop ? IRB.CreateLoad(DynamicTop) : StaticTop;
    IRB.CreateStore(CurrentTop, UnsafeStackPtr);
  }

  return DynamicTop;
}

Value *SafeStack::moveStaticAllocasToUnsafeStack(
    IRBuilder<> &IRB, Function &F, ArrayRef<AllocaInst *> StaticAllocas,
    ArrayRef<Argument *> ByValArguments, ArrayRef<ReturnInst *> Returns) {
  if (StaticAllocas.empty() && ByValArguments.empty())
    return nullptr;

  DIBuilder DIB(*F.getParent());

  // We explicitly compute and set the unsafe stack layout for all unsafe
  // static alloca instructions. We save the unsafe "base pointer" in the
  // prologue into a local variable and restore it in the epilogue.

  // Load the current stack pointer (we'll also use it as a base pointer).
  // FIXME: use a dedicated register for it ?
  Instruction *BasePointer =
      IRB.CreateLoad(UnsafeStackPtr, false, "unsafe_stack_ptr");
  assert(BasePointer->getType() == StackPtrTy);

  for (ReturnInst *RI : Returns) {
    IRB.SetInsertPoint(RI);
    IRB.CreateStore(BasePointer, UnsafeStackPtr);
  }

  // Compute maximum alignment among static objects on the unsafe stack.
  unsigned MaxAlignment = 0;
  for (Argument *Arg : ByValArguments) {
    Type *Ty = Arg->getType()->getPointerElementType();
    unsigned Align = std::max((unsigned)DL->getPrefTypeAlignment(Ty),
                              Arg->getParamAlignment());
    if (Align > MaxAlignment)
      MaxAlignment = Align;
  }
  for (AllocaInst *AI : StaticAllocas) {
    Type *Ty = AI->getAllocatedType();
    unsigned Align =
        std::max((unsigned)DL->getPrefTypeAlignment(Ty), AI->getAlignment());
    if (Align > MaxAlignment)
      MaxAlignment = Align;
  }

  if (MaxAlignment > StackAlignment) {
    // Re-align the base pointer according to the max requested alignment.
    assert(isPowerOf2_32(MaxAlignment));
    IRB.SetInsertPoint(BasePointer->getNextNode());
    BasePointer = cast<Instruction>(IRB.CreateIntToPtr(
        IRB.CreateAnd(IRB.CreatePtrToInt(BasePointer, IntPtrTy),
                      ConstantInt::get(IntPtrTy, ~uint64_t(MaxAlignment - 1))),
        StackPtrTy));
  }

  int64_t StaticOffset = 0; // Current stack top.
  IRB.SetInsertPoint(BasePointer->getNextNode());

  for (Argument *Arg : ByValArguments) {
    Type *Ty = Arg->getType()->getPointerElementType();

    uint64_t Size = DL->getTypeStoreSize(Ty);
    if (Size == 0)
      Size = 1; // Don't create zero-sized stack objects.

    // Ensure the object is properly aligned.
    unsigned Align = std::max((unsigned)DL->getPrefTypeAlignment(Ty),
                              Arg->getParamAlignment());

    // Add alignment.
    // NOTE: we ensure that BasePointer itself is aligned to >= Align.
    StaticOffset += Size;
    StaticOffset = alignTo(StaticOffset, Align);

    Value *Off = IRB.CreateGEP(BasePointer, // BasePointer is i8*
                               ConstantInt::get(Int32Ty, -StaticOffset));
    Value *NewArg = IRB.CreateBitCast(Off, Arg->getType(),
                                     Arg->getName() + ".unsafe-byval");

    // Replace alloc with the new location.
    replaceDbgDeclare(Arg, BasePointer, BasePointer->getNextNode(), DIB,
                      /*Deref=*/true, -StaticOffset);
    Arg->replaceAllUsesWith(NewArg);
    IRB.SetInsertPoint(cast<Instruction>(NewArg)->getNextNode());
    IRB.CreateMemCpy(Off, Arg, Size, Arg->getParamAlignment());
  }

  // Allocate space for every unsafe static AllocaInst on the unsafe stack.
  for (AllocaInst *AI : StaticAllocas) {
    IRB.SetInsertPoint(AI);

    Type *Ty = AI->getAllocatedType();
    uint64_t Size = getStaticAllocaAllocationSize(AI);
    if (Size == 0)
      Size = 1; // Don't create zero-sized stack objects.

    // Ensure the object is properly aligned.
    unsigned Align =
        std::max((unsigned)DL->getPrefTypeAlignment(Ty), AI->getAlignment());

    // Add alignment.
    // NOTE: we ensure that BasePointer itself is aligned to >= Align.
    StaticOffset += Size;
    StaticOffset = alignTo(StaticOffset, Align);

    Value *Off = IRB.CreateGEP(BasePointer, // BasePointer is i8*
                               ConstantInt::get(Int32Ty, -StaticOffset));
    Value *NewAI = IRB.CreateBitCast(Off, AI->getType(), AI->getName());
    if (AI->hasName() && isa<Instruction>(NewAI))
      cast<Instruction>(NewAI)->takeName(AI);

    // Replace alloc with the new location.
    replaceDbgDeclareForAlloca(AI, BasePointer, DIB, /*Deref=*/true, -StaticOffset);
    AI->replaceAllUsesWith(NewAI);
    AI->eraseFromParent();
  }

  // Re-align BasePointer so that our callees would see it aligned as
  // expected.
  // FIXME: no need to update BasePointer in leaf functions.
  StaticOffset = alignTo(StaticOffset, StackAlignment);

  // Update shadow stack pointer in the function epilogue.
  IRB.SetInsertPoint(BasePointer->getNextNode());

  Value *StaticTop =
      IRB.CreateGEP(BasePointer, ConstantInt::get(Int32Ty, -StaticOffset),
                    "unsafe_stack_static_top");
  IRB.CreateStore(StaticTop, UnsafeStackPtr);
  return StaticTop;
}

void SafeStack::moveDynamicAllocasToUnsafeStack(
    Function &F, Value *UnsafeStackPtr, AllocaInst *DynamicTop,
    ArrayRef<AllocaInst *> DynamicAllocas) {
  DIBuilder DIB(*F.getParent());

  for (AllocaInst *AI : DynamicAllocas) {
    IRBuilder<> IRB(AI);

    // Compute the new SP value (after AI).
    Value *ArraySize = AI->getArraySize();
    if (ArraySize->getType() != IntPtrTy)
      ArraySize = IRB.CreateIntCast(ArraySize, IntPtrTy, false);

    Type *Ty = AI->getAllocatedType();
    uint64_t TySize = DL->getTypeAllocSize(Ty);
    Value *Size = IRB.CreateMul(ArraySize, ConstantInt::get(IntPtrTy, TySize));

    Value *SP = IRB.CreatePtrToInt(IRB.CreateLoad(UnsafeStackPtr), IntPtrTy);
    SP = IRB.CreateSub(SP, Size);

    // Align the SP value to satisfy the AllocaInst, type and stack alignments.
    unsigned Align = std::max(
        std::max((unsigned)DL->getPrefTypeAlignment(Ty), AI->getAlignment()),
        (unsigned)StackAlignment);

    assert(isPowerOf2_32(Align));
    Value *NewTop = IRB.CreateIntToPtr(
        IRB.CreateAnd(SP, ConstantInt::get(IntPtrTy, ~uint64_t(Align - 1))),
        StackPtrTy);

    // Save the stack pointer.
    IRB.CreateStore(NewTop, UnsafeStackPtr);
    if (DynamicTop)
      IRB.CreateStore(NewTop, DynamicTop);

    Value *NewAI = IRB.CreatePointerCast(NewTop, AI->getType());
    if (AI->hasName() && isa<Instruction>(NewAI))
      NewAI->takeName(AI);

    replaceDbgDeclareForAlloca(AI, NewAI, DIB, /*Deref=*/true);
    AI->replaceAllUsesWith(NewAI);
    AI->eraseFromParent();
  }

  if (!DynamicAllocas.empty()) {
    // Now go through the instructions again, replacing stacksave/stackrestore.
    for (inst_iterator It = inst_begin(&F), Ie = inst_end(&F); It != Ie;) {
      Instruction *I = &*(It++);
      auto II = dyn_cast<IntrinsicInst>(I);
      if (!II)
        continue;

      if (II->getIntrinsicID() == Intrinsic::stacksave) {
        IRBuilder<> IRB(II);
        Instruction *LI = IRB.CreateLoad(UnsafeStackPtr);
        LI->takeName(II);
        II->replaceAllUsesWith(LI);
        II->eraseFromParent();
      } else if (II->getIntrinsicID() == Intrinsic::stackrestore) {
        IRBuilder<> IRB(II);
        Instruction *SI = IRB.CreateStore(II->getArgOperand(0), UnsafeStackPtr);
        SI->takeName(II);
        assert(II->use_empty());
        II->eraseFromParent();
      }
    }
  }
}

bool SafeStack::runOnFunction(Function &F) {
  DEBUG(dbgs() << "[SafeStack] Function: " << F.getName() << "\n");

  if (!F.hasFnAttribute(Attribute::SafeStack)) {
    DEBUG(dbgs() << "[SafeStack]     safestack is not requested"
                    " for this function\n");
    return false;
  }

  if (F.isDeclaration()) {
    DEBUG(dbgs() << "[SafeStack]     function definition"
                    " is not available\n");
    return false;
  }

  TL = TM ? TM->getSubtargetImpl(F)->getTargetLowering() : nullptr;
  SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();

  {
    // Make sure the regular stack protector won't run on this function
    // (safestack attribute takes precedence).
    AttrBuilder B;
    B.addAttribute(Attribute::StackProtect)
        .addAttribute(Attribute::StackProtectReq)
        .addAttribute(Attribute::StackProtectStrong);
    F.removeAttributes(
        AttributeSet::FunctionIndex,
        AttributeSet::get(F.getContext(), AttributeSet::FunctionIndex, B));
  }

  ++NumFunctions;

  SmallVector<AllocaInst *, 16> StaticAllocas;
  SmallVector<AllocaInst *, 4> DynamicAllocas;
  SmallVector<Argument *, 4> ByValArguments;
  SmallVector<ReturnInst *, 4> Returns;

  // Collect all points where stack gets unwound and needs to be restored
  // This is only necessary because the runtime (setjmp and unwind code) is
  // not aware of the unsafe stack and won't unwind/restore it prorerly.
  // To work around this problem without changing the runtime, we insert
  // instrumentation to restore the unsafe stack pointer when necessary.
  SmallVector<Instruction *, 4> StackRestorePoints;

  // Find all static and dynamic alloca instructions that must be moved to the
  // unsafe stack, all return instructions and stack restore points.
  findInsts(F, StaticAllocas, DynamicAllocas, ByValArguments, Returns,
            StackRestorePoints);

  if (StaticAllocas.empty() && DynamicAllocas.empty() &&
      ByValArguments.empty() && StackRestorePoints.empty())
    return false; // Nothing to do in this function.

  if (!StaticAllocas.empty() || !DynamicAllocas.empty() ||
      !ByValArguments.empty())
    ++NumUnsafeStackFunctions; // This function has the unsafe stack.

  if (!StackRestorePoints.empty())
    ++NumUnsafeStackRestorePointsFunctions;

  IRBuilder<> IRB(&F.front(), F.begin()->getFirstInsertionPt());
  UnsafeStackPtr = getOrCreateUnsafeStackPtr(IRB, F);

  // The top of the unsafe stack after all unsafe static allocas are allocated.
  Value *StaticTop = moveStaticAllocasToUnsafeStack(IRB, F, StaticAllocas,
                                                    ByValArguments, Returns);

  // Safe stack object that stores the current unsafe stack top. It is updated
  // as unsafe dynamic (non-constant-sized) allocas are allocated and freed.
  // This is only needed if we need to restore stack pointer after longjmp
  // or exceptions, and we have dynamic allocations.
  // FIXME: a better alternative might be to store the unsafe stack pointer
  // before setjmp / invoke instructions.
  AllocaInst *DynamicTop = createStackRestorePoints(
      IRB, F, StackRestorePoints, StaticTop, !DynamicAllocas.empty());

  // Handle dynamic allocas.
  moveDynamicAllocasToUnsafeStack(F, UnsafeStackPtr, DynamicTop,
                                  DynamicAllocas);

  DEBUG(dbgs() << "[SafeStack]     safestack applied\n");
  return true;
}

} // anonymous namespace

char SafeStack::ID = 0;
INITIALIZE_TM_PASS_BEGIN(SafeStack, "safe-stack",
                         "Safe Stack instrumentation pass", false, false)
INITIALIZE_TM_PASS_END(SafeStack, "safe-stack",
                       "Safe Stack instrumentation pass", false, false)

FunctionPass *llvm::createSafeStackPass(const llvm::TargetMachine *TM) {
  return new SafeStack(TM);
}
