//===- ObjCARC.cpp - ObjC ARC Optimization --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines ObjC ARC optimizations. ARC stands for
// Automatic Reference Counting and is a system for managing reference counts
// for objects in Objective C.
//
// The optimizations performed include elimination of redundant, partially
// redundant, and inconsequential reference count operations, elimination of
// redundant weak pointer operations, pattern-matching and replacement of
// low-level operations into higher-level operations, and numerous minor
// simplifications.
//
// This file also defines a simple ARC-aware AliasAnalysis.
//
// WARNING: This file knows about certain library functions. It recognizes them
// by name, and hardwires knowledge of their semantics.
//
// WARNING: This file knows about how certain Objective-C library functions are
// used. Naive LLVM IR transformations which would otherwise be
// behavior-preserving may break these assumptions.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "objc-arc"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/DenseMap.h"
using namespace llvm;

// A handy option to enable/disable all optimizations in this file.
static cl::opt<bool> EnableARCOpts("enable-objc-arc-opts", cl::init(true));

//===----------------------------------------------------------------------===//
// Misc. Utilities
//===----------------------------------------------------------------------===//

namespace {
  /// MapVector - An associative container with fast insertion-order
  /// (deterministic) iteration over its elements. Plus the special
  /// blot operation.
  template<class KeyT, class ValueT>
  class MapVector {
    /// Map - Map keys to indices in Vector.
    typedef DenseMap<KeyT, size_t> MapTy;
    MapTy Map;

    /// Vector - Keys and values.
    typedef std::vector<std::pair<KeyT, ValueT> > VectorTy;
    VectorTy Vector;

  public:
    typedef typename VectorTy::iterator iterator;
    typedef typename VectorTy::const_iterator const_iterator;
    iterator begin() { return Vector.begin(); }
    iterator end() { return Vector.end(); }
    const_iterator begin() const { return Vector.begin(); }
    const_iterator end() const { return Vector.end(); }

#ifdef XDEBUG
    ~MapVector() {
      assert(Vector.size() >= Map.size()); // May differ due to blotting.
      for (typename MapTy::const_iterator I = Map.begin(), E = Map.end();
           I != E; ++I) {
        assert(I->second < Vector.size());
        assert(Vector[I->second].first == I->first);
      }
      for (typename VectorTy::const_iterator I = Vector.begin(),
           E = Vector.end(); I != E; ++I)
        assert(!I->first ||
               (Map.count(I->first) &&
                Map[I->first] == size_t(I - Vector.begin())));
    }
#endif

    ValueT &operator[](const KeyT &Arg) {
      std::pair<typename MapTy::iterator, bool> Pair =
        Map.insert(std::make_pair(Arg, size_t(0)));
      if (Pair.second) {
        size_t Num = Vector.size();
        Pair.first->second = Num;
        Vector.push_back(std::make_pair(Arg, ValueT()));
        return Vector[Num].second;
      }
      return Vector[Pair.first->second].second;
    }

    std::pair<iterator, bool>
    insert(const std::pair<KeyT, ValueT> &InsertPair) {
      std::pair<typename MapTy::iterator, bool> Pair =
        Map.insert(std::make_pair(InsertPair.first, size_t(0)));
      if (Pair.second) {
        size_t Num = Vector.size();
        Pair.first->second = Num;
        Vector.push_back(InsertPair);
        return std::make_pair(Vector.begin() + Num, true);
      }
      return std::make_pair(Vector.begin() + Pair.first->second, false);
    }

    const_iterator find(const KeyT &Key) const {
      typename MapTy::const_iterator It = Map.find(Key);
      if (It == Map.end()) return Vector.end();
      return Vector.begin() + It->second;
    }

    /// blot - This is similar to erase, but instead of removing the element
    /// from the vector, it just zeros out the key in the vector. This leaves
    /// iterators intact, but clients must be prepared for zeroed-out keys when
    /// iterating.
    void blot(const KeyT &Key) {
      typename MapTy::iterator It = Map.find(Key);
      if (It == Map.end()) return;
      Vector[It->second].first = KeyT();
      Map.erase(It);
    }

    void clear() {
      Map.clear();
      Vector.clear();
    }
  };
}

//===----------------------------------------------------------------------===//
// ARC Utilities.
//===----------------------------------------------------------------------===//

#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/CallSite.h"
#include "llvm/ADT/StringSwitch.h"

namespace {
  /// InstructionClass - A simple classification for instructions.
  enum InstructionClass {
    IC_Retain,              ///< objc_retain
    IC_RetainRV,            ///< objc_retainAutoreleasedReturnValue
    IC_RetainBlock,         ///< objc_retainBlock
    IC_Release,             ///< objc_release
    IC_Autorelease,         ///< objc_autorelease
    IC_AutoreleaseRV,       ///< objc_autoreleaseReturnValue
    IC_AutoreleasepoolPush, ///< objc_autoreleasePoolPush
    IC_AutoreleasepoolPop,  ///< objc_autoreleasePoolPop
    IC_NoopCast,            ///< objc_retainedObject, etc.
    IC_FusedRetainAutorelease, ///< objc_retainAutorelease
    IC_FusedRetainAutoreleaseRV, ///< objc_retainAutoreleaseReturnValue
    IC_LoadWeakRetained,    ///< objc_loadWeakRetained (primitive)
    IC_StoreWeak,           ///< objc_storeWeak (primitive)
    IC_InitWeak,            ///< objc_initWeak (derived)
    IC_LoadWeak,            ///< objc_loadWeak (derived)
    IC_MoveWeak,            ///< objc_moveWeak (derived)
    IC_CopyWeak,            ///< objc_copyWeak (derived)
    IC_DestroyWeak,         ///< objc_destroyWeak (derived)
    IC_StoreStrong,         ///< objc_storeStrong (derived)
    IC_CallOrUser,          ///< could call objc_release and/or "use" pointers
    IC_Call,                ///< could call objc_release
    IC_User,                ///< could "use" a pointer
    IC_None                 ///< anything else
  };
}

/// IsPotentialUse - Test whether the given value is possible a
/// reference-counted pointer.
static bool IsPotentialUse(const Value *Op) {
  // Pointers to static or stack storage are not reference-counted pointers.
  if (isa<Constant>(Op) || isa<AllocaInst>(Op))
    return false;
  // Special arguments are not reference-counted.
  if (const Argument *Arg = dyn_cast<Argument>(Op))
    if (Arg->hasByValAttr() ||
        Arg->hasNestAttr() ||
        Arg->hasStructRetAttr())
      return false;
  // Only consider values with pointer types.
  // It seemes intuitive to exclude function pointer types as well, since
  // functions are never reference-counted, however clang occasionally
  // bitcasts reference-counted pointers to function-pointer type
  // temporarily.
  PointerType *Ty = dyn_cast<PointerType>(Op->getType());
  if (!Ty)
    return false;
  // Conservatively assume anything else is a potential use.
  return true;
}

/// GetCallSiteClass - Helper for GetInstructionClass. Determines what kind
/// of construct CS is.
static InstructionClass GetCallSiteClass(ImmutableCallSite CS) {
  for (ImmutableCallSite::arg_iterator I = CS.arg_begin(), E = CS.arg_end();
       I != E; ++I)
    if (IsPotentialUse(*I))
      return CS.onlyReadsMemory() ? IC_User : IC_CallOrUser;

  return CS.onlyReadsMemory() ? IC_None : IC_Call;
}

/// GetFunctionClass - Determine if F is one of the special known Functions.
/// If it isn't, return IC_CallOrUser.
static InstructionClass GetFunctionClass(const Function *F) {
  Function::const_arg_iterator AI = F->arg_begin(), AE = F->arg_end();

  // No arguments.
  if (AI == AE)
    return StringSwitch<InstructionClass>(F->getName())
      .Case("objc_autoreleasePoolPush",  IC_AutoreleasepoolPush)
      .Default(IC_CallOrUser);

  // One argument.
  const Argument *A0 = AI++;
  if (AI == AE)
    // Argument is a pointer.
    if (PointerType *PTy = dyn_cast<PointerType>(A0->getType())) {
      Type *ETy = PTy->getElementType();
      // Argument is i8*.
      if (ETy->isIntegerTy(8))
        return StringSwitch<InstructionClass>(F->getName())
          .Case("objc_retain",                IC_Retain)
          .Case("objc_retainAutoreleasedReturnValue", IC_RetainRV)
          .Case("objc_retainBlock",           IC_RetainBlock)
          .Case("objc_release",               IC_Release)
          .Case("objc_autorelease",           IC_Autorelease)
          .Case("objc_autoreleaseReturnValue", IC_AutoreleaseRV)
          .Case("objc_autoreleasePoolPop",    IC_AutoreleasepoolPop)
          .Case("objc_retainedObject",        IC_NoopCast)
          .Case("objc_unretainedObject",      IC_NoopCast)
          .Case("objc_unretainedPointer",     IC_NoopCast)
          .Case("objc_retain_autorelease",    IC_FusedRetainAutorelease)
          .Case("objc_retainAutorelease",     IC_FusedRetainAutorelease)
          .Case("objc_retainAutoreleaseReturnValue",IC_FusedRetainAutoreleaseRV)
          .Default(IC_CallOrUser);

      // Argument is i8**
      if (PointerType *Pte = dyn_cast<PointerType>(ETy))
        if (Pte->getElementType()->isIntegerTy(8))
          return StringSwitch<InstructionClass>(F->getName())
            .Case("objc_loadWeakRetained",      IC_LoadWeakRetained)
            .Case("objc_loadWeak",              IC_LoadWeak)
            .Case("objc_destroyWeak",           IC_DestroyWeak)
            .Default(IC_CallOrUser);
    }

  // Two arguments, first is i8**.
  const Argument *A1 = AI++;
  if (AI == AE)
    if (PointerType *PTy = dyn_cast<PointerType>(A0->getType()))
      if (PointerType *Pte = dyn_cast<PointerType>(PTy->getElementType()))
        if (Pte->getElementType()->isIntegerTy(8))
          if (PointerType *PTy1 = dyn_cast<PointerType>(A1->getType())) {
            Type *ETy1 = PTy1->getElementType();
            // Second argument is i8*
            if (ETy1->isIntegerTy(8))
              return StringSwitch<InstructionClass>(F->getName())
                     .Case("objc_storeWeak",             IC_StoreWeak)
                     .Case("objc_initWeak",              IC_InitWeak)
                     .Case("objc_storeStrong",           IC_StoreStrong)
                     .Default(IC_CallOrUser);
            // Second argument is i8**.
            if (PointerType *Pte1 = dyn_cast<PointerType>(ETy1))
              if (Pte1->getElementType()->isIntegerTy(8))
                return StringSwitch<InstructionClass>(F->getName())
                       .Case("objc_moveWeak",              IC_MoveWeak)
                       .Case("objc_copyWeak",              IC_CopyWeak)
                       .Default(IC_CallOrUser);
          }

  // Anything else.
  return IC_CallOrUser;
}

/// GetInstructionClass - Determine what kind of construct V is.
static InstructionClass GetInstructionClass(const Value *V) {
  if (const Instruction *I = dyn_cast<Instruction>(V)) {
    // Any instruction other than bitcast and gep with a pointer operand have a
    // use of an objc pointer. Bitcasts, GEPs, Selects, PHIs transfer a pointer
    // to a subsequent use, rather than using it themselves, in this sense.
    // As a short cut, several other opcodes are known to have no pointer
    // operands of interest. And ret is never followed by a release, so it's
    // not interesting to examine.
    switch (I->getOpcode()) {
    case Instruction::Call: {
      const CallInst *CI = cast<CallInst>(I);
      // Check for calls to special functions.
      if (const Function *F = CI->getCalledFunction()) {
        InstructionClass Class = GetFunctionClass(F);
        if (Class != IC_CallOrUser)
          return Class;

        // None of the intrinsic functions do objc_release. For intrinsics, the
        // only question is whether or not they may be users.
        switch (F->getIntrinsicID()) {
        case Intrinsic::returnaddress: case Intrinsic::frameaddress:
        case Intrinsic::stacksave: case Intrinsic::stackrestore:
        case Intrinsic::vastart: case Intrinsic::vacopy: case Intrinsic::vaend:
        case Intrinsic::objectsize: case Intrinsic::prefetch:
        case Intrinsic::stackprotector:
        case Intrinsic::eh_return_i32: case Intrinsic::eh_return_i64:
        case Intrinsic::eh_typeid_for: case Intrinsic::eh_dwarf_cfa:
        case Intrinsic::eh_sjlj_lsda: case Intrinsic::eh_sjlj_functioncontext:
        case Intrinsic::init_trampoline: case Intrinsic::adjust_trampoline:
        case Intrinsic::lifetime_start: case Intrinsic::lifetime_end:
        case Intrinsic::invariant_start: case Intrinsic::invariant_end:
        // Don't let dbg info affect our results.
        case Intrinsic::dbg_declare: case Intrinsic::dbg_value:
          // Short cut: Some intrinsics obviously don't use ObjC pointers.
          return IC_None;
        default:
          break;
        }
      }
      return GetCallSiteClass(CI);
    }
    case Instruction::Invoke:
      return GetCallSiteClass(cast<InvokeInst>(I));
    case Instruction::BitCast:
    case Instruction::GetElementPtr:
    case Instruction::Select: case Instruction::PHI:
    case Instruction::Ret: case Instruction::Br:
    case Instruction::Switch: case Instruction::IndirectBr:
    case Instruction::Alloca: case Instruction::VAArg:
    case Instruction::Add: case Instruction::FAdd:
    case Instruction::Sub: case Instruction::FSub:
    case Instruction::Mul: case Instruction::FMul:
    case Instruction::SDiv: case Instruction::UDiv: case Instruction::FDiv:
    case Instruction::SRem: case Instruction::URem: case Instruction::FRem:
    case Instruction::Shl: case Instruction::LShr: case Instruction::AShr:
    case Instruction::And: case Instruction::Or: case Instruction::Xor:
    case Instruction::SExt: case Instruction::ZExt: case Instruction::Trunc:
    case Instruction::IntToPtr: case Instruction::FCmp:
    case Instruction::FPTrunc: case Instruction::FPExt:
    case Instruction::FPToUI: case Instruction::FPToSI:
    case Instruction::UIToFP: case Instruction::SIToFP:
    case Instruction::InsertElement: case Instruction::ExtractElement:
    case Instruction::ShuffleVector:
    case Instruction::ExtractValue:
      break;
    case Instruction::ICmp:
      // Comparing a pointer with null, or any other constant, isn't an
      // interesting use, because we don't care what the pointer points to, or
      // about the values of any other dynamic reference-counted pointers.
      if (IsPotentialUse(I->getOperand(1)))
        return IC_User;
      break;
    default:
      // For anything else, check all the operands.
      // Note that this includes both operands of a Store: while the first
      // operand isn't actually being dereferenced, it is being stored to
      // memory where we can no longer track who might read it and dereference
      // it, so we have to consider it potentially used.
      for (User::const_op_iterator OI = I->op_begin(), OE = I->op_end();
           OI != OE; ++OI)
        if (IsPotentialUse(*OI))
          return IC_User;
    }
  }

  // Otherwise, it's totally inert for ARC purposes.
  return IC_None;
}

/// GetBasicInstructionClass - Determine what kind of construct V is. This is
/// similar to GetInstructionClass except that it only detects objc runtine
/// calls. This allows it to be faster.
static InstructionClass GetBasicInstructionClass(const Value *V) {
  if (const CallInst *CI = dyn_cast<CallInst>(V)) {
    if (const Function *F = CI->getCalledFunction())
      return GetFunctionClass(F);
    // Otherwise, be conservative.
    return IC_CallOrUser;
  }

  // Otherwise, be conservative.
  return isa<InvokeInst>(V) ? IC_CallOrUser : IC_User;
}

/// IsRetain - Test if the given class is objc_retain or
/// equivalent.
static bool IsRetain(InstructionClass Class) {
  return Class == IC_Retain ||
         Class == IC_RetainRV;
}

/// IsAutorelease - Test if the given class is objc_autorelease or
/// equivalent.
static bool IsAutorelease(InstructionClass Class) {
  return Class == IC_Autorelease ||
         Class == IC_AutoreleaseRV;
}

/// IsForwarding - Test if the given class represents instructions which return
/// their argument verbatim.
static bool IsForwarding(InstructionClass Class) {
  // objc_retainBlock technically doesn't always return its argument
  // verbatim, but it doesn't matter for our purposes here.
  return Class == IC_Retain ||
         Class == IC_RetainRV ||
         Class == IC_Autorelease ||
         Class == IC_AutoreleaseRV ||
         Class == IC_RetainBlock ||
         Class == IC_NoopCast;
}

/// IsNoopOnNull - Test if the given class represents instructions which do
/// nothing if passed a null pointer.
static bool IsNoopOnNull(InstructionClass Class) {
  return Class == IC_Retain ||
         Class == IC_RetainRV ||
         Class == IC_Release ||
         Class == IC_Autorelease ||
         Class == IC_AutoreleaseRV ||
         Class == IC_RetainBlock;
}

/// IsAlwaysTail - Test if the given class represents instructions which are
/// always safe to mark with the "tail" keyword.
static bool IsAlwaysTail(InstructionClass Class) {
  // IC_RetainBlock may be given a stack argument.
  return Class == IC_Retain ||
         Class == IC_RetainRV ||
         Class == IC_Autorelease ||
         Class == IC_AutoreleaseRV;
}

/// IsNoThrow - Test if the given class represents instructions which are always
/// safe to mark with the nounwind attribute..
static bool IsNoThrow(InstructionClass Class) {
  // objc_retainBlock is not nounwind because it calls user copy constructors
  // which could theoretically throw.
  return Class == IC_Retain ||
         Class == IC_RetainRV ||
         Class == IC_Release ||
         Class == IC_Autorelease ||
         Class == IC_AutoreleaseRV ||
         Class == IC_AutoreleasepoolPush ||
         Class == IC_AutoreleasepoolPop;
}

/// EraseInstruction - Erase the given instruction. Many ObjC calls return their
/// argument verbatim, so if it's such a call and the return value has users,
/// replace them with the argument value.
static void EraseInstruction(Instruction *CI) {
  Value *OldArg = cast<CallInst>(CI)->getArgOperand(0);

  bool Unused = CI->use_empty();

  if (!Unused) {
    // Replace the return value with the argument.
    assert(IsForwarding(GetBasicInstructionClass(CI)) &&
           "Can't delete non-forwarding instruction with users!");
    CI->replaceAllUsesWith(OldArg);
  }

  CI->eraseFromParent();

  if (Unused)
    RecursivelyDeleteTriviallyDeadInstructions(OldArg);
}

/// GetUnderlyingObjCPtr - This is a wrapper around getUnderlyingObject which
/// also knows how to look through objc_retain and objc_autorelease calls, which
/// we know to return their argument verbatim.
static const Value *GetUnderlyingObjCPtr(const Value *V) {
  for (;;) {
    V = GetUnderlyingObject(V);
    if (!IsForwarding(GetBasicInstructionClass(V)))
      break;
    V = cast<CallInst>(V)->getArgOperand(0);
  }

  return V;
}

/// StripPointerCastsAndObjCCalls - This is a wrapper around
/// Value::stripPointerCasts which also knows how to look through objc_retain
/// and objc_autorelease calls, which we know to return their argument verbatim.
static const Value *StripPointerCastsAndObjCCalls(const Value *V) {
  for (;;) {
    V = V->stripPointerCasts();
    if (!IsForwarding(GetBasicInstructionClass(V)))
      break;
    V = cast<CallInst>(V)->getArgOperand(0);
  }
  return V;
}

/// StripPointerCastsAndObjCCalls - This is a wrapper around
/// Value::stripPointerCasts which also knows how to look through objc_retain
/// and objc_autorelease calls, which we know to return their argument verbatim.
static Value *StripPointerCastsAndObjCCalls(Value *V) {
  for (;;) {
    V = V->stripPointerCasts();
    if (!IsForwarding(GetBasicInstructionClass(V)))
      break;
    V = cast<CallInst>(V)->getArgOperand(0);
  }
  return V;
}

/// GetObjCArg - Assuming the given instruction is one of the special calls such
/// as objc_retain or objc_release, return the argument value, stripped of no-op
/// casts and forwarding calls.
static Value *GetObjCArg(Value *Inst) {
  return StripPointerCastsAndObjCCalls(cast<CallInst>(Inst)->getArgOperand(0));
}

/// IsObjCIdentifiedObject - This is similar to AliasAnalysis'
/// isObjCIdentifiedObject, except that it uses special knowledge of
/// ObjC conventions...
static bool IsObjCIdentifiedObject(const Value *V) {
  // Assume that call results and arguments have their own "provenance".
  // Constants (including GlobalVariables) and Allocas are never
  // reference-counted.
  if (isa<CallInst>(V) || isa<InvokeInst>(V) ||
      isa<Argument>(V) || isa<Constant>(V) ||
      isa<AllocaInst>(V))
    return true;

  if (const LoadInst *LI = dyn_cast<LoadInst>(V)) {
    const Value *Pointer =
      StripPointerCastsAndObjCCalls(LI->getPointerOperand());
    if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(Pointer)) {
      // A constant pointer can't be pointing to an object on the heap. It may
      // be reference-counted, but it won't be deleted.
      if (GV->isConstant())
        return true;
      StringRef Name = GV->getName();
      // These special variables are known to hold values which are not
      // reference-counted pointers.
      if (Name.startswith("\01L_OBJC_SELECTOR_REFERENCES_") ||
          Name.startswith("\01L_OBJC_CLASSLIST_REFERENCES_") ||
          Name.startswith("\01L_OBJC_CLASSLIST_SUP_REFS_$_") ||
          Name.startswith("\01L_OBJC_METH_VAR_NAME_") ||
          Name.startswith("\01l_objc_msgSend_fixup_"))
        return true;
    }
  }

  return false;
}

/// FindSingleUseIdentifiedObject - This is similar to
/// StripPointerCastsAndObjCCalls but it stops as soon as it finds a value
/// with multiple uses.
static const Value *FindSingleUseIdentifiedObject(const Value *Arg) {
  if (Arg->hasOneUse()) {
    if (const BitCastInst *BC = dyn_cast<BitCastInst>(Arg))
      return FindSingleUseIdentifiedObject(BC->getOperand(0));
    if (const GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Arg))
      if (GEP->hasAllZeroIndices())
        return FindSingleUseIdentifiedObject(GEP->getPointerOperand());
    if (IsForwarding(GetBasicInstructionClass(Arg)))
      return FindSingleUseIdentifiedObject(
               cast<CallInst>(Arg)->getArgOperand(0));
    if (!IsObjCIdentifiedObject(Arg))
      return 0;
    return Arg;
  }

  // If we found an identifiable object but it has multiple uses, but they are
  // trivial uses, we can still consider this to be a single-use value.
  if (IsObjCIdentifiedObject(Arg)) {
    for (Value::const_use_iterator UI = Arg->use_begin(), UE = Arg->use_end();
         UI != UE; ++UI) {
      const User *U = *UI;
      if (!U->use_empty() || StripPointerCastsAndObjCCalls(U) != Arg)
         return 0;
    }

    return Arg;
  }

  return 0;
}

/// ModuleHasARC - Test if the given module looks interesting to run ARC
/// optimization on.
static bool ModuleHasARC(const Module &M) {
  return
    M.getNamedValue("objc_retain") ||
    M.getNamedValue("objc_release") ||
    M.getNamedValue("objc_autorelease") ||
    M.getNamedValue("objc_retainAutoreleasedReturnValue") ||
    M.getNamedValue("objc_retainBlock") ||
    M.getNamedValue("objc_autoreleaseReturnValue") ||
    M.getNamedValue("objc_autoreleasePoolPush") ||
    M.getNamedValue("objc_loadWeakRetained") ||
    M.getNamedValue("objc_loadWeak") ||
    M.getNamedValue("objc_destroyWeak") ||
    M.getNamedValue("objc_storeWeak") ||
    M.getNamedValue("objc_initWeak") ||
    M.getNamedValue("objc_moveWeak") ||
    M.getNamedValue("objc_copyWeak") ||
    M.getNamedValue("objc_retainedObject") ||
    M.getNamedValue("objc_unretainedObject") ||
    M.getNamedValue("objc_unretainedPointer");
}

/// DoesObjCBlockEscape - Test whether the given pointer, which is an
/// Objective C block pointer, does not "escape". This differs from regular
/// escape analysis in that a use as an argument to a call is not considered
/// an escape.
static bool DoesObjCBlockEscape(const Value *BlockPtr) {
  // Walk the def-use chains.
  SmallVector<const Value *, 4> Worklist;
  Worklist.push_back(BlockPtr);
  do {
    const Value *V = Worklist.pop_back_val();
    for (Value::const_use_iterator UI = V->use_begin(), UE = V->use_end();
         UI != UE; ++UI) {
      const User *UUser = *UI;
      // Special - Use by a call (callee or argument) is not considered
      // to be an escape.
      switch (GetBasicInstructionClass(UUser)) {
      case IC_StoreWeak:
      case IC_InitWeak:
      case IC_StoreStrong:
      case IC_Autorelease:
      case IC_AutoreleaseRV:
        // These special functions make copies of their pointer arguments.
        return true;
      case IC_User:
      case IC_None:
        // Use by an instruction which copies the value is an escape if the
        // result is an escape.
        if (isa<BitCastInst>(UUser) || isa<GetElementPtrInst>(UUser) ||
            isa<PHINode>(UUser) || isa<SelectInst>(UUser)) {
          Worklist.push_back(UUser);
          continue;
        }
        // Use by a load is not an escape.
        if (isa<LoadInst>(UUser))
          continue;
        // Use by a store is not an escape if the use is the address.
        if (const StoreInst *SI = dyn_cast<StoreInst>(UUser))
          if (V != SI->getValueOperand())
            continue;
        break;
      default:
        // Regular calls and other stuff are not considered escapes.
        continue;
      }
      // Otherwise, conservatively assume an escape.
      return true;
    }
  } while (!Worklist.empty());

  // No escapes found.
  return false;
}

//===----------------------------------------------------------------------===//
// ARC AliasAnalysis.
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Passes.h"

namespace {
  /// ObjCARCAliasAnalysis - This is a simple alias analysis
  /// implementation that uses knowledge of ARC constructs to answer queries.
  ///
  /// TODO: This class could be generalized to know about other ObjC-specific
  /// tricks. Such as knowing that ivars in the non-fragile ABI are non-aliasing
  /// even though their offsets are dynamic.
  class ObjCARCAliasAnalysis : public ImmutablePass,
                               public AliasAnalysis {
  public:
    static char ID; // Class identification, replacement for typeinfo
    ObjCARCAliasAnalysis() : ImmutablePass(ID) {
      initializeObjCARCAliasAnalysisPass(*PassRegistry::getPassRegistry());
    }

  private:
    virtual void initializePass() {
      InitializeAliasAnalysis(this);
    }

    /// getAdjustedAnalysisPointer - This method is used when a pass implements
    /// an analysis interface through multiple inheritance.  If needed, it
    /// should override this to adjust the this pointer as needed for the
    /// specified pass info.
    virtual void *getAdjustedAnalysisPointer(const void *PI) {
      if (PI == &AliasAnalysis::ID)
        return static_cast<AliasAnalysis *>(this);
      return this;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual AliasResult alias(const Location &LocA, const Location &LocB);
    virtual bool pointsToConstantMemory(const Location &Loc, bool OrLocal);
    virtual ModRefBehavior getModRefBehavior(ImmutableCallSite CS);
    virtual ModRefBehavior getModRefBehavior(const Function *F);
    virtual ModRefResult getModRefInfo(ImmutableCallSite CS,
                                       const Location &Loc);
    virtual ModRefResult getModRefInfo(ImmutableCallSite CS1,
                                       ImmutableCallSite CS2);
  };
}  // End of anonymous namespace

// Register this pass...
char ObjCARCAliasAnalysis::ID = 0;
INITIALIZE_AG_PASS(ObjCARCAliasAnalysis, AliasAnalysis, "objc-arc-aa",
                   "ObjC-ARC-Based Alias Analysis", false, true, false)

ImmutablePass *llvm::createObjCARCAliasAnalysisPass() {
  return new ObjCARCAliasAnalysis();
}

void
ObjCARCAliasAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AliasAnalysis::getAnalysisUsage(AU);
}

AliasAnalysis::AliasResult
ObjCARCAliasAnalysis::alias(const Location &LocA, const Location &LocB) {
  if (!EnableARCOpts)
    return AliasAnalysis::alias(LocA, LocB);

  // First, strip off no-ops, including ObjC-specific no-ops, and try making a
  // precise alias query.
  const Value *SA = StripPointerCastsAndObjCCalls(LocA.Ptr);
  const Value *SB = StripPointerCastsAndObjCCalls(LocB.Ptr);
  AliasResult Result =
    AliasAnalysis::alias(Location(SA, LocA.Size, LocA.TBAATag),
                         Location(SB, LocB.Size, LocB.TBAATag));
  if (Result != MayAlias)
    return Result;

  // If that failed, climb to the underlying object, including climbing through
  // ObjC-specific no-ops, and try making an imprecise alias query.
  const Value *UA = GetUnderlyingObjCPtr(SA);
  const Value *UB = GetUnderlyingObjCPtr(SB);
  if (UA != SA || UB != SB) {
    Result = AliasAnalysis::alias(Location(UA), Location(UB));
    // We can't use MustAlias or PartialAlias results here because
    // GetUnderlyingObjCPtr may return an offsetted pointer value.
    if (Result == NoAlias)
      return NoAlias;
  }

  // If that failed, fail. We don't need to chain here, since that's covered
  // by the earlier precise query.
  return MayAlias;
}

bool
ObjCARCAliasAnalysis::pointsToConstantMemory(const Location &Loc,
                                             bool OrLocal) {
  if (!EnableARCOpts)
    return AliasAnalysis::pointsToConstantMemory(Loc, OrLocal);

  // First, strip off no-ops, including ObjC-specific no-ops, and try making
  // a precise alias query.
  const Value *S = StripPointerCastsAndObjCCalls(Loc.Ptr);
  if (AliasAnalysis::pointsToConstantMemory(Location(S, Loc.Size, Loc.TBAATag),
                                            OrLocal))
    return true;

  // If that failed, climb to the underlying object, including climbing through
  // ObjC-specific no-ops, and try making an imprecise alias query.
  const Value *U = GetUnderlyingObjCPtr(S);
  if (U != S)
    return AliasAnalysis::pointsToConstantMemory(Location(U), OrLocal);

  // If that failed, fail. We don't need to chain here, since that's covered
  // by the earlier precise query.
  return false;
}

AliasAnalysis::ModRefBehavior
ObjCARCAliasAnalysis::getModRefBehavior(ImmutableCallSite CS) {
  // We have nothing to do. Just chain to the next AliasAnalysis.
  return AliasAnalysis::getModRefBehavior(CS);
}

AliasAnalysis::ModRefBehavior
ObjCARCAliasAnalysis::getModRefBehavior(const Function *F) {
  if (!EnableARCOpts)
    return AliasAnalysis::getModRefBehavior(F);

  switch (GetFunctionClass(F)) {
  case IC_NoopCast:
    return DoesNotAccessMemory;
  default:
    break;
  }

  return AliasAnalysis::getModRefBehavior(F);
}

AliasAnalysis::ModRefResult
ObjCARCAliasAnalysis::getModRefInfo(ImmutableCallSite CS, const Location &Loc) {
  if (!EnableARCOpts)
    return AliasAnalysis::getModRefInfo(CS, Loc);

  switch (GetBasicInstructionClass(CS.getInstruction())) {
  case IC_Retain:
  case IC_RetainRV:
  case IC_Autorelease:
  case IC_AutoreleaseRV:
  case IC_NoopCast:
  case IC_AutoreleasepoolPush:
  case IC_FusedRetainAutorelease:
  case IC_FusedRetainAutoreleaseRV:
    // These functions don't access any memory visible to the compiler.
    // Note that this doesn't include objc_retainBlock, because it updates
    // pointers when it copies block data.
    return NoModRef;
  default:
    break;
  }

  return AliasAnalysis::getModRefInfo(CS, Loc);
}

AliasAnalysis::ModRefResult
ObjCARCAliasAnalysis::getModRefInfo(ImmutableCallSite CS1,
                                    ImmutableCallSite CS2) {
  // TODO: Theoretically we could check for dependencies between objc_* calls
  // and OnlyAccessesArgumentPointees calls or other well-behaved calls.
  return AliasAnalysis::getModRefInfo(CS1, CS2);
}

//===----------------------------------------------------------------------===//
// ARC expansion.
//===----------------------------------------------------------------------===//

#include "llvm/Support/InstIterator.h"
#include "llvm/Transforms/Scalar.h"

namespace {
  /// ObjCARCExpand - Early ARC transformations.
  class ObjCARCExpand : public FunctionPass {
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual bool doInitialization(Module &M);
    virtual bool runOnFunction(Function &F);

    /// Run - A flag indicating whether this optimization pass should run.
    bool Run;

  public:
    static char ID;
    ObjCARCExpand() : FunctionPass(ID) {
      initializeObjCARCExpandPass(*PassRegistry::getPassRegistry());
    }
  };
}

char ObjCARCExpand::ID = 0;
INITIALIZE_PASS(ObjCARCExpand,
                "objc-arc-expand", "ObjC ARC expansion", false, false)

Pass *llvm::createObjCARCExpandPass() {
  return new ObjCARCExpand();
}

void ObjCARCExpand::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
}

bool ObjCARCExpand::doInitialization(Module &M) {
  Run = ModuleHasARC(M);
  return false;
}

bool ObjCARCExpand::runOnFunction(Function &F) {
  if (!EnableARCOpts)
    return false;

  // If nothing in the Module uses ARC, don't do anything.
  if (!Run)
    return false;

  bool Changed = false;

  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
    Instruction *Inst = &*I;

    switch (GetBasicInstructionClass(Inst)) {
    case IC_Retain:
    case IC_RetainRV:
    case IC_Autorelease:
    case IC_AutoreleaseRV:
    case IC_FusedRetainAutorelease:
    case IC_FusedRetainAutoreleaseRV:
      // These calls return their argument verbatim, as a low-level
      // optimization. However, this makes high-level optimizations
      // harder. Undo any uses of this optimization that the front-end
      // emitted here. We'll redo them in the contract pass.
      Changed = true;
      Inst->replaceAllUsesWith(cast<CallInst>(Inst)->getArgOperand(0));
      break;
    default:
      break;
    }
  }

  return Changed;
}

//===----------------------------------------------------------------------===//
// ARC autorelease pool elimination.
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/ADT/STLExtras.h"

namespace {
  /// ObjCARCAPElim - Autorelease pool elimination.
  class ObjCARCAPElim : public ModulePass {
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual bool runOnModule(Module &M);

    static bool MayAutorelease(ImmutableCallSite CS, unsigned Depth = 0);
    static bool OptimizeBB(BasicBlock *BB);

  public:
    static char ID;
    ObjCARCAPElim() : ModulePass(ID) {
      initializeObjCARCAPElimPass(*PassRegistry::getPassRegistry());
    }
  };
}

char ObjCARCAPElim::ID = 0;
INITIALIZE_PASS(ObjCARCAPElim,
                "objc-arc-apelim",
                "ObjC ARC autorelease pool elimination",
                false, false)

Pass *llvm::createObjCARCAPElimPass() {
  return new ObjCARCAPElim();
}

void ObjCARCAPElim::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
}

/// MayAutorelease - Interprocedurally determine if calls made by the
/// given call site can possibly produce autoreleases.
bool ObjCARCAPElim::MayAutorelease(ImmutableCallSite CS, unsigned Depth) {
  if (const Function *Callee = CS.getCalledFunction()) {
    if (Callee->isDeclaration() || Callee->mayBeOverridden())
      return true;
    for (Function::const_iterator I = Callee->begin(), E = Callee->end();
         I != E; ++I) {
      const BasicBlock *BB = I;
      for (BasicBlock::const_iterator J = BB->begin(), F = BB->end();
           J != F; ++J)
        if (ImmutableCallSite JCS = ImmutableCallSite(J))
          // This recursion depth limit is arbitrary. It's just great
          // enough to cover known interesting testcases.
          if (Depth < 3 &&
              !JCS.onlyReadsMemory() &&
              MayAutorelease(JCS, Depth + 1))
            return true;
    }
    return false;
  }

  return true;
}

bool ObjCARCAPElim::OptimizeBB(BasicBlock *BB) {
  bool Changed = false;

  Instruction *Push = 0;
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ) {
    Instruction *Inst = I++;
    switch (GetBasicInstructionClass(Inst)) {
    case IC_AutoreleasepoolPush:
      Push = Inst;
      break;
    case IC_AutoreleasepoolPop:
      // If this pop matches a push and nothing in between can autorelease,
      // zap the pair.
      if (Push && cast<CallInst>(Inst)->getArgOperand(0) == Push) {
        Changed = true;
        Inst->eraseFromParent();
        Push->eraseFromParent();
      }
      Push = 0;
      break;
    case IC_CallOrUser:
      if (MayAutorelease(ImmutableCallSite(Inst)))
        Push = 0;
      break;
    default:
      break;
    }
  }

  return Changed;
}

bool ObjCARCAPElim::runOnModule(Module &M) {
  if (!EnableARCOpts)
    return false;

  // If nothing in the Module uses ARC, don't do anything.
  if (!ModuleHasARC(M))
    return false;

  // Find the llvm.global_ctors variable, as the first step in
  // identifying the global constructors. In theory, unnecessary autorelease
  // pools could occur anywhere, but in practice it's pretty rare. Global
  // ctors are a place where autorelease pools get inserted automatically,
  // so it's pretty common for them to be unnecessary, and it's pretty
  // profitable to eliminate them.
  GlobalVariable *GV = M.getGlobalVariable("llvm.global_ctors");
  if (!GV)
    return false;

  assert(GV->hasDefinitiveInitializer() &&
         "llvm.global_ctors is uncooperative!");

  bool Changed = false;

  // Dig the constructor functions out of GV's initializer.
  ConstantArray *Init = cast<ConstantArray>(GV->getInitializer());
  for (User::op_iterator OI = Init->op_begin(), OE = Init->op_end();
       OI != OE; ++OI) {
    Value *Op = *OI;
    // llvm.global_ctors is an array of pairs where the second members
    // are constructor functions.
    Function *F = dyn_cast<Function>(cast<ConstantStruct>(Op)->getOperand(1));
    // If the user used a constructor function with the wrong signature and
    // it got bitcasted or whatever, look the other way.
    if (!F)
      continue;
    // Only look at function definitions.
    if (F->isDeclaration())
      continue;
    // Only look at functions with one basic block.
    if (llvm::next(F->begin()) != F->end())
      continue;
    // Ok, a single-block constructor function definition. Try to optimize it.
    Changed |= OptimizeBB(F->begin());
  }

  return Changed;
}

//===----------------------------------------------------------------------===//
// ARC optimization.
//===----------------------------------------------------------------------===//

// TODO: On code like this:
//
// objc_retain(%x)
// stuff_that_cannot_release()
// objc_autorelease(%x)
// stuff_that_cannot_release()
// objc_retain(%x)
// stuff_that_cannot_release()
// objc_autorelease(%x)
//
// The second retain and autorelease can be deleted.

// TODO: It should be possible to delete
// objc_autoreleasePoolPush and objc_autoreleasePoolPop
// pairs if nothing is actually autoreleased between them. Also, autorelease
// calls followed by objc_autoreleasePoolPop calls (perhaps in ObjC++ code
// after inlining) can be turned into plain release calls.

// TODO: Critical-edge splitting. If the optimial insertion point is
// a critical edge, the current algorithm has to fail, because it doesn't
// know how to split edges. It should be possible to make the optimizer
// think in terms of edges, rather than blocks, and then split critical
// edges on demand.

// TODO: OptimizeSequences could generalized to be Interprocedural.

// TODO: Recognize that a bunch of other objc runtime calls have
// non-escaping arguments and non-releasing arguments, and may be
// non-autoreleasing.

// TODO: Sink autorelease calls as far as possible. Unfortunately we
// usually can't sink them past other calls, which would be the main
// case where it would be useful.

// TODO: The pointer returned from objc_loadWeakRetained is retained.

// TODO: Delete release+retain pairs (rare).

#include "llvm/LLVMContext.h"
#include "llvm/Support/CFG.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/SmallPtrSet.h"

STATISTIC(NumNoops,       "Number of no-op objc calls eliminated");
STATISTIC(NumPartialNoops, "Number of partially no-op objc calls eliminated");
STATISTIC(NumAutoreleases,"Number of autoreleases converted to releases");
STATISTIC(NumRets,        "Number of return value forwarding "
                          "retain+autoreleaes eliminated");
STATISTIC(NumRRs,         "Number of retain+release paths eliminated");
STATISTIC(NumPeeps,       "Number of calls peephole-optimized");

namespace {
  /// ProvenanceAnalysis - This is similar to BasicAliasAnalysis, and it
  /// uses many of the same techniques, except it uses special ObjC-specific
  /// reasoning about pointer relationships.
  class ProvenanceAnalysis {
    AliasAnalysis *AA;

    typedef std::pair<const Value *, const Value *> ValuePairTy;
    typedef DenseMap<ValuePairTy, bool> CachedResultsTy;
    CachedResultsTy CachedResults;

    bool relatedCheck(const Value *A, const Value *B);
    bool relatedSelect(const SelectInst *A, const Value *B);
    bool relatedPHI(const PHINode *A, const Value *B);

    void operator=(const ProvenanceAnalysis &) LLVM_DELETED_FUNCTION;
    ProvenanceAnalysis(const ProvenanceAnalysis &) LLVM_DELETED_FUNCTION;

  public:
    ProvenanceAnalysis() {}

    void setAA(AliasAnalysis *aa) { AA = aa; }

    AliasAnalysis *getAA() const { return AA; }

    bool related(const Value *A, const Value *B);

    void clear() {
      CachedResults.clear();
    }
  };
}

bool ProvenanceAnalysis::relatedSelect(const SelectInst *A, const Value *B) {
  // If the values are Selects with the same condition, we can do a more precise
  // check: just check for relations between the values on corresponding arms.
  if (const SelectInst *SB = dyn_cast<SelectInst>(B))
    if (A->getCondition() == SB->getCondition())
      return related(A->getTrueValue(), SB->getTrueValue()) ||
             related(A->getFalseValue(), SB->getFalseValue());

  // Check both arms of the Select node individually.
  return related(A->getTrueValue(), B) ||
         related(A->getFalseValue(), B);
}

bool ProvenanceAnalysis::relatedPHI(const PHINode *A, const Value *B) {
  // If the values are PHIs in the same block, we can do a more precise as well
  // as efficient check: just check for relations between the values on
  // corresponding edges.
  if (const PHINode *PNB = dyn_cast<PHINode>(B))
    if (PNB->getParent() == A->getParent()) {
      for (unsigned i = 0, e = A->getNumIncomingValues(); i != e; ++i)
        if (related(A->getIncomingValue(i),
                    PNB->getIncomingValueForBlock(A->getIncomingBlock(i))))
          return true;
      return false;
    }

  // Check each unique source of the PHI node against B.
  SmallPtrSet<const Value *, 4> UniqueSrc;
  for (unsigned i = 0, e = A->getNumIncomingValues(); i != e; ++i) {
    const Value *PV1 = A->getIncomingValue(i);
    if (UniqueSrc.insert(PV1) && related(PV1, B))
      return true;
  }

  // All of the arms checked out.
  return false;
}

/// isStoredObjCPointer - Test if the value of P, or any value covered by its
/// provenance, is ever stored within the function (not counting callees).
static bool isStoredObjCPointer(const Value *P) {
  SmallPtrSet<const Value *, 8> Visited;
  SmallVector<const Value *, 8> Worklist;
  Worklist.push_back(P);
  Visited.insert(P);
  do {
    P = Worklist.pop_back_val();
    for (Value::const_use_iterator UI = P->use_begin(), UE = P->use_end();
         UI != UE; ++UI) {
      const User *Ur = *UI;
      if (isa<StoreInst>(Ur)) {
        if (UI.getOperandNo() == 0)
          // The pointer is stored.
          return true;
        // The pointed is stored through.
        continue;
      }
      if (isa<CallInst>(Ur))
        // The pointer is passed as an argument, ignore this.
        continue;
      if (isa<PtrToIntInst>(P))
        // Assume the worst.
        return true;
      if (Visited.insert(Ur))
        Worklist.push_back(Ur);
    }
  } while (!Worklist.empty());

  // Everything checked out.
  return false;
}

bool ProvenanceAnalysis::relatedCheck(const Value *A, const Value *B) {
  // Skip past provenance pass-throughs.
  A = GetUnderlyingObjCPtr(A);
  B = GetUnderlyingObjCPtr(B);

  // Quick check.
  if (A == B)
    return true;

  // Ask regular AliasAnalysis, for a first approximation.
  switch (AA->alias(A, B)) {
  case AliasAnalysis::NoAlias:
    return false;
  case AliasAnalysis::MustAlias:
  case AliasAnalysis::PartialAlias:
    return true;
  case AliasAnalysis::MayAlias:
    break;
  }

  bool AIsIdentified = IsObjCIdentifiedObject(A);
  bool BIsIdentified = IsObjCIdentifiedObject(B);

  // An ObjC-Identified object can't alias a load if it is never locally stored.
  if (AIsIdentified) {
    // Check for an obvious escape.
    if (isa<LoadInst>(B))
      return isStoredObjCPointer(A);
    if (BIsIdentified) {
      // Check for an obvious escape.
      if (isa<LoadInst>(A))
        return isStoredObjCPointer(B);
      // Both pointers are identified and escapes aren't an evident problem.
      return false;
    }
  } else if (BIsIdentified) {
    // Check for an obvious escape.
    if (isa<LoadInst>(A))
      return isStoredObjCPointer(B);
  }

   // Special handling for PHI and Select.
  if (const PHINode *PN = dyn_cast<PHINode>(A))
    return relatedPHI(PN, B);
  if (const PHINode *PN = dyn_cast<PHINode>(B))
    return relatedPHI(PN, A);
  if (const SelectInst *S = dyn_cast<SelectInst>(A))
    return relatedSelect(S, B);
  if (const SelectInst *S = dyn_cast<SelectInst>(B))
    return relatedSelect(S, A);

  // Conservative.
  return true;
}

bool ProvenanceAnalysis::related(const Value *A, const Value *B) {
  // Begin by inserting a conservative value into the map. If the insertion
  // fails, we have the answer already. If it succeeds, leave it there until we
  // compute the real answer to guard against recursive queries.
  if (A > B) std::swap(A, B);
  std::pair<CachedResultsTy::iterator, bool> Pair =
    CachedResults.insert(std::make_pair(ValuePairTy(A, B), true));
  if (!Pair.second)
    return Pair.first->second;

  bool Result = relatedCheck(A, B);
  CachedResults[ValuePairTy(A, B)] = Result;
  return Result;
}

namespace {
  // Sequence - A sequence of states that a pointer may go through in which an
  // objc_retain and objc_release are actually needed.
  enum Sequence {
    S_None,
    S_Retain,         ///< objc_retain(x)
    S_CanRelease,     ///< foo(x) -- x could possibly see a ref count decrement
    S_Use,            ///< any use of x
    S_Stop,           ///< like S_Release, but code motion is stopped
    S_Release,        ///< objc_release(x)
    S_MovableRelease  ///< objc_release(x), !clang.imprecise_release
  };
}

static Sequence MergeSeqs(Sequence A, Sequence B, bool TopDown) {
  // The easy cases.
  if (A == B)
    return A;
  if (A == S_None || B == S_None)
    return S_None;

  if (A > B) std::swap(A, B);
  if (TopDown) {
    // Choose the side which is further along in the sequence.
    if ((A == S_Retain || A == S_CanRelease) &&
        (B == S_CanRelease || B == S_Use))
      return B;
  } else {
    // Choose the side which is further along in the sequence.
    if ((A == S_Use || A == S_CanRelease) &&
        (B == S_Use || B == S_Release || B == S_Stop || B == S_MovableRelease))
      return A;
    // If both sides are releases, choose the more conservative one.
    if (A == S_Stop && (B == S_Release || B == S_MovableRelease))
      return A;
    if (A == S_Release && B == S_MovableRelease)
      return A;
  }

  return S_None;
}

namespace {
  /// RRInfo - Unidirectional information about either a
  /// retain-decrement-use-release sequence or release-use-decrement-retain
  /// reverese sequence.
  struct RRInfo {
    /// KnownSafe - After an objc_retain, the reference count of the referenced
    /// object is known to be positive. Similarly, before an objc_release, the
    /// reference count of the referenced object is known to be positive. If
    /// there are retain-release pairs in code regions where the retain count
    /// is known to be positive, they can be eliminated, regardless of any side
    /// effects between them.
    ///
    /// Also, a retain+release pair nested within another retain+release
    /// pair all on the known same pointer value can be eliminated, regardless
    /// of any intervening side effects.
    ///
    /// KnownSafe is true when either of these conditions is satisfied.
    bool KnownSafe;

    /// IsRetainBlock - True if the Calls are objc_retainBlock calls (as
    /// opposed to objc_retain calls).
    bool IsRetainBlock;

    /// IsTailCallRelease - True of the objc_release calls are all marked
    /// with the "tail" keyword.
    bool IsTailCallRelease;

    /// ReleaseMetadata - If the Calls are objc_release calls and they all have
    /// a clang.imprecise_release tag, this is the metadata tag.
    MDNode *ReleaseMetadata;

    /// Calls - For a top-down sequence, the set of objc_retains or
    /// objc_retainBlocks. For bottom-up, the set of objc_releases.
    SmallPtrSet<Instruction *, 2> Calls;

    /// ReverseInsertPts - The set of optimal insert positions for
    /// moving calls in the opposite sequence.
    SmallPtrSet<Instruction *, 2> ReverseInsertPts;

    RRInfo() :
      KnownSafe(false), IsRetainBlock(false),
      IsTailCallRelease(false),
      ReleaseMetadata(0) {}

    void clear();
  };
}

void RRInfo::clear() {
  KnownSafe = false;
  IsRetainBlock = false;
  IsTailCallRelease = false;
  ReleaseMetadata = 0;
  Calls.clear();
  ReverseInsertPts.clear();
}

namespace {
  /// PtrState - This class summarizes several per-pointer runtime properties
  /// which are propogated through the flow graph.
  class PtrState {
    /// KnownPositiveRefCount - True if the reference count is known to
    /// be incremented.
    bool KnownPositiveRefCount;

    /// Partial - True of we've seen an opportunity for partial RR elimination,
    /// such as pushing calls into a CFG triangle or into one side of a
    /// CFG diamond.
    bool Partial;

    /// Seq - The current position in the sequence.
    Sequence Seq : 8;

  public:
    /// RRI - Unidirectional information about the current sequence.
    /// TODO: Encapsulate this better.
    RRInfo RRI;

    PtrState() : KnownPositiveRefCount(false), Partial(false),
                 Seq(S_None) {}

    void SetKnownPositiveRefCount() {
      KnownPositiveRefCount = true;
    }

    void ClearRefCount() {
      KnownPositiveRefCount = false;
    }

    bool IsKnownIncremented() const {
      return KnownPositiveRefCount;
    }

    void SetSeq(Sequence NewSeq) {
      Seq = NewSeq;
    }

    Sequence GetSeq() const {
      return Seq;
    }

    void ClearSequenceProgress() {
      ResetSequenceProgress(S_None);
    }

    void ResetSequenceProgress(Sequence NewSeq) {
      Seq = NewSeq;
      Partial = false;
      RRI.clear();
    }

    void Merge(const PtrState &Other, bool TopDown);
  };
}

void
PtrState::Merge(const PtrState &Other, bool TopDown) {
  Seq = MergeSeqs(Seq, Other.Seq, TopDown);
  KnownPositiveRefCount = KnownPositiveRefCount && Other.KnownPositiveRefCount;

  // We can't merge a plain objc_retain with an objc_retainBlock.
  if (RRI.IsRetainBlock != Other.RRI.IsRetainBlock)
    Seq = S_None;

  // If we're not in a sequence (anymore), drop all associated state.
  if (Seq == S_None) {
    Partial = false;
    RRI.clear();
  } else if (Partial || Other.Partial) {
    // If we're doing a merge on a path that's previously seen a partial
    // merge, conservatively drop the sequence, to avoid doing partial
    // RR elimination. If the branch predicates for the two merge differ,
    // mixing them is unsafe.
    ClearSequenceProgress();
  } else {
    // Conservatively merge the ReleaseMetadata information.
    if (RRI.ReleaseMetadata != Other.RRI.ReleaseMetadata)
      RRI.ReleaseMetadata = 0;

    RRI.KnownSafe = RRI.KnownSafe && Other.RRI.KnownSafe;
    RRI.IsTailCallRelease = RRI.IsTailCallRelease &&
                            Other.RRI.IsTailCallRelease;
    RRI.Calls.insert(Other.RRI.Calls.begin(), Other.RRI.Calls.end());

    // Merge the insert point sets. If there are any differences,
    // that makes this a partial merge.
    Partial = RRI.ReverseInsertPts.size() != Other.RRI.ReverseInsertPts.size();
    for (SmallPtrSet<Instruction *, 2>::const_iterator
         I = Other.RRI.ReverseInsertPts.begin(),
         E = Other.RRI.ReverseInsertPts.end(); I != E; ++I)
      Partial |= RRI.ReverseInsertPts.insert(*I);
  }
}

namespace {
  /// BBState - Per-BasicBlock state.
  class BBState {
    /// TopDownPathCount - The number of unique control paths from the entry
    /// which can reach this block.
    unsigned TopDownPathCount;

    /// BottomUpPathCount - The number of unique control paths to exits
    /// from this block.
    unsigned BottomUpPathCount;

    /// MapTy - A type for PerPtrTopDown and PerPtrBottomUp.
    typedef MapVector<const Value *, PtrState> MapTy;

    /// PerPtrTopDown - The top-down traversal uses this to record information
    /// known about a pointer at the bottom of each block.
    MapTy PerPtrTopDown;

    /// PerPtrBottomUp - The bottom-up traversal uses this to record information
    /// known about a pointer at the top of each block.
    MapTy PerPtrBottomUp;

    /// Preds, Succs - Effective successors and predecessors of the current
    /// block (this ignores ignorable edges and ignored backedges).
    SmallVector<BasicBlock *, 2> Preds;
    SmallVector<BasicBlock *, 2> Succs;

  public:
    BBState() : TopDownPathCount(0), BottomUpPathCount(0) {}

    typedef MapTy::iterator ptr_iterator;
    typedef MapTy::const_iterator ptr_const_iterator;

    ptr_iterator top_down_ptr_begin() { return PerPtrTopDown.begin(); }
    ptr_iterator top_down_ptr_end() { return PerPtrTopDown.end(); }
    ptr_const_iterator top_down_ptr_begin() const {
      return PerPtrTopDown.begin();
    }
    ptr_const_iterator top_down_ptr_end() const {
      return PerPtrTopDown.end();
    }

    ptr_iterator bottom_up_ptr_begin() { return PerPtrBottomUp.begin(); }
    ptr_iterator bottom_up_ptr_end() { return PerPtrBottomUp.end(); }
    ptr_const_iterator bottom_up_ptr_begin() const {
      return PerPtrBottomUp.begin();
    }
    ptr_const_iterator bottom_up_ptr_end() const {
      return PerPtrBottomUp.end();
    }

    /// SetAsEntry - Mark this block as being an entry block, which has one
    /// path from the entry by definition.
    void SetAsEntry() { TopDownPathCount = 1; }

    /// SetAsExit - Mark this block as being an exit block, which has one
    /// path to an exit by definition.
    void SetAsExit()  { BottomUpPathCount = 1; }

    PtrState &getPtrTopDownState(const Value *Arg) {
      return PerPtrTopDown[Arg];
    }

    PtrState &getPtrBottomUpState(const Value *Arg) {
      return PerPtrBottomUp[Arg];
    }

    void clearBottomUpPointers() {
      PerPtrBottomUp.clear();
    }

    void clearTopDownPointers() {
      PerPtrTopDown.clear();
    }

    void InitFromPred(const BBState &Other);
    void InitFromSucc(const BBState &Other);
    void MergePred(const BBState &Other);
    void MergeSucc(const BBState &Other);

    /// GetAllPathCount - Return the number of possible unique paths from an
    /// entry to an exit which pass through this block. This is only valid
    /// after both the top-down and bottom-up traversals are complete.
    unsigned GetAllPathCount() const {
      assert(TopDownPathCount != 0);
      assert(BottomUpPathCount != 0);
      return TopDownPathCount * BottomUpPathCount;
    }

    // Specialized CFG utilities.
    typedef SmallVectorImpl<BasicBlock *>::const_iterator edge_iterator;
    edge_iterator pred_begin() { return Preds.begin(); }
    edge_iterator pred_end() { return Preds.end(); }
    edge_iterator succ_begin() { return Succs.begin(); }
    edge_iterator succ_end() { return Succs.end(); }

    void addSucc(BasicBlock *Succ) { Succs.push_back(Succ); }
    void addPred(BasicBlock *Pred) { Preds.push_back(Pred); }

    bool isExit() const { return Succs.empty(); }
  };
}

void BBState::InitFromPred(const BBState &Other) {
  PerPtrTopDown = Other.PerPtrTopDown;
  TopDownPathCount = Other.TopDownPathCount;
}

void BBState::InitFromSucc(const BBState &Other) {
  PerPtrBottomUp = Other.PerPtrBottomUp;
  BottomUpPathCount = Other.BottomUpPathCount;
}

/// MergePred - The top-down traversal uses this to merge information about
/// predecessors to form the initial state for a new block.
void BBState::MergePred(const BBState &Other) {
  // Other.TopDownPathCount can be 0, in which case it is either dead or a
  // loop backedge. Loop backedges are special.
  TopDownPathCount += Other.TopDownPathCount;

  // Check for overflow. If we have overflow, fall back to conservative behavior.
  if (TopDownPathCount < Other.TopDownPathCount) {
    clearTopDownPointers();
    return;
  }

  // For each entry in the other set, if our set has an entry with the same key,
  // merge the entries. Otherwise, copy the entry and merge it with an empty
  // entry.
  for (ptr_const_iterator MI = Other.top_down_ptr_begin(),
       ME = Other.top_down_ptr_end(); MI != ME; ++MI) {
    std::pair<ptr_iterator, bool> Pair = PerPtrTopDown.insert(*MI);
    Pair.first->second.Merge(Pair.second ? PtrState() : MI->second,
                             /*TopDown=*/true);
  }

  // For each entry in our set, if the other set doesn't have an entry with the
  // same key, force it to merge with an empty entry.
  for (ptr_iterator MI = top_down_ptr_begin(),
       ME = top_down_ptr_end(); MI != ME; ++MI)
    if (Other.PerPtrTopDown.find(MI->first) == Other.PerPtrTopDown.end())
      MI->second.Merge(PtrState(), /*TopDown=*/true);
}

/// MergeSucc - The bottom-up traversal uses this to merge information about
/// successors to form the initial state for a new block.
void BBState::MergeSucc(const BBState &Other) {
  // Other.BottomUpPathCount can be 0, in which case it is either dead or a
  // loop backedge. Loop backedges are special.
  BottomUpPathCount += Other.BottomUpPathCount;

  // Check for overflow. If we have overflow, fall back to conservative behavior.
  if (BottomUpPathCount < Other.BottomUpPathCount) {
    clearBottomUpPointers();
    return;
  }

  // For each entry in the other set, if our set has an entry with the
  // same key, merge the entries. Otherwise, copy the entry and merge
  // it with an empty entry.
  for (ptr_const_iterator MI = Other.bottom_up_ptr_begin(),
       ME = Other.bottom_up_ptr_end(); MI != ME; ++MI) {
    std::pair<ptr_iterator, bool> Pair = PerPtrBottomUp.insert(*MI);
    Pair.first->second.Merge(Pair.second ? PtrState() : MI->second,
                             /*TopDown=*/false);
  }

  // For each entry in our set, if the other set doesn't have an entry
  // with the same key, force it to merge with an empty entry.
  for (ptr_iterator MI = bottom_up_ptr_begin(),
       ME = bottom_up_ptr_end(); MI != ME; ++MI)
    if (Other.PerPtrBottomUp.find(MI->first) == Other.PerPtrBottomUp.end())
      MI->second.Merge(PtrState(), /*TopDown=*/false);
}

namespace {
  /// ObjCARCOpt - The main ARC optimization pass.
  class ObjCARCOpt : public FunctionPass {
    bool Changed;
    ProvenanceAnalysis PA;

    /// Run - A flag indicating whether this optimization pass should run.
    bool Run;

    /// RetainRVCallee, etc. - Declarations for ObjC runtime
    /// functions, for use in creating calls to them. These are initialized
    /// lazily to avoid cluttering up the Module with unused declarations.
    Constant *RetainRVCallee, *AutoreleaseRVCallee, *ReleaseCallee,
             *RetainCallee, *RetainBlockCallee, *AutoreleaseCallee;

    /// UsedInThisFunciton - Flags which determine whether each of the
    /// interesting runtine functions is in fact used in the current function.
    unsigned UsedInThisFunction;

    /// ImpreciseReleaseMDKind - The Metadata Kind for clang.imprecise_release
    /// metadata.
    unsigned ImpreciseReleaseMDKind;

    /// CopyOnEscapeMDKind - The Metadata Kind for clang.arc.copy_on_escape
    /// metadata.
    unsigned CopyOnEscapeMDKind;

    /// NoObjCARCExceptionsMDKind - The Metadata Kind for
    /// clang.arc.no_objc_arc_exceptions metadata.
    unsigned NoObjCARCExceptionsMDKind;

    Constant *getRetainRVCallee(Module *M);
    Constant *getAutoreleaseRVCallee(Module *M);
    Constant *getReleaseCallee(Module *M);
    Constant *getRetainCallee(Module *M);
    Constant *getRetainBlockCallee(Module *M);
    Constant *getAutoreleaseCallee(Module *M);

    bool IsRetainBlockOptimizable(const Instruction *Inst);

    void OptimizeRetainCall(Function &F, Instruction *Retain);
    bool OptimizeRetainRVCall(Function &F, Instruction *RetainRV);
    void OptimizeAutoreleaseRVCall(Function &F, Instruction *AutoreleaseRV);
    void OptimizeIndividualCalls(Function &F);

    void CheckForCFGHazards(const BasicBlock *BB,
                            DenseMap<const BasicBlock *, BBState> &BBStates,
                            BBState &MyStates) const;
    bool VisitInstructionBottomUp(Instruction *Inst,
                                  BasicBlock *BB,
                                  MapVector<Value *, RRInfo> &Retains,
                                  BBState &MyStates);
    bool VisitBottomUp(BasicBlock *BB,
                       DenseMap<const BasicBlock *, BBState> &BBStates,
                       MapVector<Value *, RRInfo> &Retains);
    bool VisitInstructionTopDown(Instruction *Inst,
                                 DenseMap<Value *, RRInfo> &Releases,
                                 BBState &MyStates);
    bool VisitTopDown(BasicBlock *BB,
                      DenseMap<const BasicBlock *, BBState> &BBStates,
                      DenseMap<Value *, RRInfo> &Releases);
    bool Visit(Function &F,
               DenseMap<const BasicBlock *, BBState> &BBStates,
               MapVector<Value *, RRInfo> &Retains,
               DenseMap<Value *, RRInfo> &Releases);

    void MoveCalls(Value *Arg, RRInfo &RetainsToMove, RRInfo &ReleasesToMove,
                   MapVector<Value *, RRInfo> &Retains,
                   DenseMap<Value *, RRInfo> &Releases,
                   SmallVectorImpl<Instruction *> &DeadInsts,
                   Module *M);

    bool PerformCodePlacement(DenseMap<const BasicBlock *, BBState> &BBStates,
                              MapVector<Value *, RRInfo> &Retains,
                              DenseMap<Value *, RRInfo> &Releases,
                              Module *M);

    void OptimizeWeakCalls(Function &F);

    bool OptimizeSequences(Function &F);

    void OptimizeReturns(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual bool doInitialization(Module &M);
    virtual bool runOnFunction(Function &F);
    virtual void releaseMemory();

  public:
    static char ID;
    ObjCARCOpt() : FunctionPass(ID) {
      initializeObjCARCOptPass(*PassRegistry::getPassRegistry());
    }
  };
}

char ObjCARCOpt::ID = 0;
INITIALIZE_PASS_BEGIN(ObjCARCOpt,
                      "objc-arc", "ObjC ARC optimization", false, false)
INITIALIZE_PASS_DEPENDENCY(ObjCARCAliasAnalysis)
INITIALIZE_PASS_END(ObjCARCOpt,
                    "objc-arc", "ObjC ARC optimization", false, false)

Pass *llvm::createObjCARCOptPass() {
  return new ObjCARCOpt();
}

void ObjCARCOpt::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<ObjCARCAliasAnalysis>();
  AU.addRequired<AliasAnalysis>();
  // ARC optimization doesn't currently split critical edges.
  AU.setPreservesCFG();
}

bool ObjCARCOpt::IsRetainBlockOptimizable(const Instruction *Inst) {
  // Without the magic metadata tag, we have to assume this might be an
  // objc_retainBlock call inserted to convert a block pointer to an id,
  // in which case it really is needed.
  if (!Inst->getMetadata(CopyOnEscapeMDKind))
    return false;

  // If the pointer "escapes" (not including being used in a call),
  // the copy may be needed.
  if (DoesObjCBlockEscape(Inst))
    return false;

  // Otherwise, it's not needed.
  return true;
}

Constant *ObjCARCOpt::getRetainRVCallee(Module *M) {
  if (!RetainRVCallee) {
    LLVMContext &C = M->getContext();
    Type *I8X = PointerType::getUnqual(Type::getInt8Ty(C));
    Type *Params[] = { I8X };
    FunctionType *FTy = FunctionType::get(I8X, Params, /*isVarArg=*/false);
    AttrListPtr Attributes =
      AttrListPtr().addAttr(M->getContext(), AttrListPtr::FunctionIndex,
                            Attributes::get(C, Attributes::NoUnwind));
    RetainRVCallee =
      M->getOrInsertFunction("objc_retainAutoreleasedReturnValue", FTy,
                             Attributes);
  }
  return RetainRVCallee;
}

Constant *ObjCARCOpt::getAutoreleaseRVCallee(Module *M) {
  if (!AutoreleaseRVCallee) {
    LLVMContext &C = M->getContext();
    Type *I8X = PointerType::getUnqual(Type::getInt8Ty(C));
    Type *Params[] = { I8X };
    FunctionType *FTy = FunctionType::get(I8X, Params, /*isVarArg=*/false);
    AttrListPtr Attributes =
      AttrListPtr().addAttr(M->getContext(), AttrListPtr::FunctionIndex,
                            Attributes::get(C, Attributes::NoUnwind));
    AutoreleaseRVCallee =
      M->getOrInsertFunction("objc_autoreleaseReturnValue", FTy,
                             Attributes);
  }
  return AutoreleaseRVCallee;
}

Constant *ObjCARCOpt::getReleaseCallee(Module *M) {
  if (!ReleaseCallee) {
    LLVMContext &C = M->getContext();
    Type *Params[] = { PointerType::getUnqual(Type::getInt8Ty(C)) };
    AttrListPtr Attributes =
      AttrListPtr().addAttr(M->getContext(), AttrListPtr::FunctionIndex,
                            Attributes::get(C, Attributes::NoUnwind));
    ReleaseCallee =
      M->getOrInsertFunction(
        "objc_release",
        FunctionType::get(Type::getVoidTy(C), Params, /*isVarArg=*/false),
        Attributes);
  }
  return ReleaseCallee;
}

Constant *ObjCARCOpt::getRetainCallee(Module *M) {
  if (!RetainCallee) {
    LLVMContext &C = M->getContext();
    Type *Params[] = { PointerType::getUnqual(Type::getInt8Ty(C)) };
    AttrListPtr Attributes =
      AttrListPtr().addAttr(M->getContext(), AttrListPtr::FunctionIndex,
                            Attributes::get(C, Attributes::NoUnwind));
    RetainCallee =
      M->getOrInsertFunction(
        "objc_retain",
        FunctionType::get(Params[0], Params, /*isVarArg=*/false),
        Attributes);
  }
  return RetainCallee;
}

Constant *ObjCARCOpt::getRetainBlockCallee(Module *M) {
  if (!RetainBlockCallee) {
    LLVMContext &C = M->getContext();
    Type *Params[] = { PointerType::getUnqual(Type::getInt8Ty(C)) };
    // objc_retainBlock is not nounwind because it calls user copy constructors
    // which could theoretically throw.
    RetainBlockCallee =
      M->getOrInsertFunction(
        "objc_retainBlock",
        FunctionType::get(Params[0], Params, /*isVarArg=*/false),
        AttrListPtr());
  }
  return RetainBlockCallee;
}

Constant *ObjCARCOpt::getAutoreleaseCallee(Module *M) {
  if (!AutoreleaseCallee) {
    LLVMContext &C = M->getContext();
    Type *Params[] = { PointerType::getUnqual(Type::getInt8Ty(C)) };
    AttrListPtr Attributes =
      AttrListPtr().addAttr(M->getContext(), AttrListPtr::FunctionIndex,
                            Attributes::get(C, Attributes::NoUnwind));
    AutoreleaseCallee =
      M->getOrInsertFunction(
        "objc_autorelease",
        FunctionType::get(Params[0], Params, /*isVarArg=*/false),
        Attributes);
  }
  return AutoreleaseCallee;
}

/// IsPotentialUse - Test whether the given value is possible a
/// reference-counted pointer, including tests which utilize AliasAnalysis.
static bool IsPotentialUse(const Value *Op, AliasAnalysis &AA) {
  // First make the rudimentary check.
  if (!IsPotentialUse(Op))
    return false;

  // Objects in constant memory are not reference-counted.
  if (AA.pointsToConstantMemory(Op))
    return false;

  // Pointers in constant memory are not pointing to reference-counted objects.
  if (const LoadInst *LI = dyn_cast<LoadInst>(Op))
    if (AA.pointsToConstantMemory(LI->getPointerOperand()))
      return false;

  // Otherwise assume the worst.
  return true;
}

/// CanAlterRefCount - Test whether the given instruction can result in a
/// reference count modification (positive or negative) for the pointer's
/// object.
static bool
CanAlterRefCount(const Instruction *Inst, const Value *Ptr,
                 ProvenanceAnalysis &PA, InstructionClass Class) {
  switch (Class) {
  case IC_Autorelease:
  case IC_AutoreleaseRV:
  case IC_User:
    // These operations never directly modify a reference count.
    return false;
  default: break;
  }

  ImmutableCallSite CS = static_cast<const Value *>(Inst);
  assert(CS && "Only calls can alter reference counts!");

  // See if AliasAnalysis can help us with the call.
  AliasAnalysis::ModRefBehavior MRB = PA.getAA()->getModRefBehavior(CS);
  if (AliasAnalysis::onlyReadsMemory(MRB))
    return false;
  if (AliasAnalysis::onlyAccessesArgPointees(MRB)) {
    for (ImmutableCallSite::arg_iterator I = CS.arg_begin(), E = CS.arg_end();
         I != E; ++I) {
      const Value *Op = *I;
      if (IsPotentialUse(Op, *PA.getAA()) && PA.related(Ptr, Op))
        return true;
    }
    return false;
  }

  // Assume the worst.
  return true;
}

/// CanUse - Test whether the given instruction can "use" the given pointer's
/// object in a way that requires the reference count to be positive.
static bool
CanUse(const Instruction *Inst, const Value *Ptr, ProvenanceAnalysis &PA,
       InstructionClass Class) {
  // IC_Call operations (as opposed to IC_CallOrUser) never "use" objc pointers.
  if (Class == IC_Call)
    return false;

  // Consider various instructions which may have pointer arguments which are
  // not "uses".
  if (const ICmpInst *ICI = dyn_cast<ICmpInst>(Inst)) {
    // Comparing a pointer with null, or any other constant, isn't really a use,
    // because we don't care what the pointer points to, or about the values
    // of any other dynamic reference-counted pointers.
    if (!IsPotentialUse(ICI->getOperand(1), *PA.getAA()))
      return false;
  } else if (ImmutableCallSite CS = static_cast<const Value *>(Inst)) {
    // For calls, just check the arguments (and not the callee operand).
    for (ImmutableCallSite::arg_iterator OI = CS.arg_begin(),
         OE = CS.arg_end(); OI != OE; ++OI) {
      const Value *Op = *OI;
      if (IsPotentialUse(Op, *PA.getAA()) && PA.related(Ptr, Op))
        return true;
    }
    return false;
  } else if (const StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
    // Special-case stores, because we don't care about the stored value, just
    // the store address.
    const Value *Op = GetUnderlyingObjCPtr(SI->getPointerOperand());
    // If we can't tell what the underlying object was, assume there is a
    // dependence.
    return IsPotentialUse(Op, *PA.getAA()) && PA.related(Op, Ptr);
  }

  // Check each operand for a match.
  for (User::const_op_iterator OI = Inst->op_begin(), OE = Inst->op_end();
       OI != OE; ++OI) {
    const Value *Op = *OI;
    if (IsPotentialUse(Op, *PA.getAA()) && PA.related(Ptr, Op))
      return true;
  }
  return false;
}

/// CanInterruptRV - Test whether the given instruction can autorelease
/// any pointer or cause an autoreleasepool pop.
static bool
CanInterruptRV(InstructionClass Class) {
  switch (Class) {
  case IC_AutoreleasepoolPop:
  case IC_CallOrUser:
  case IC_Call:
  case IC_Autorelease:
  case IC_AutoreleaseRV:
  case IC_FusedRetainAutorelease:
  case IC_FusedRetainAutoreleaseRV:
    return true;
  default:
    return false;
  }
}

namespace {
  /// DependenceKind - There are several kinds of dependence-like concepts in
  /// use here.
  enum DependenceKind {
    NeedsPositiveRetainCount,
    AutoreleasePoolBoundary,
    CanChangeRetainCount,
    RetainAutoreleaseDep,       ///< Blocks objc_retainAutorelease.
    RetainAutoreleaseRVDep,     ///< Blocks objc_retainAutoreleaseReturnValue.
    RetainRVDep                 ///< Blocks objc_retainAutoreleasedReturnValue.
  };
}

/// Depends - Test if there can be dependencies on Inst through Arg. This
/// function only tests dependencies relevant for removing pairs of calls.
static bool
Depends(DependenceKind Flavor, Instruction *Inst, const Value *Arg,
        ProvenanceAnalysis &PA) {
  // If we've reached the definition of Arg, stop.
  if (Inst == Arg)
    return true;

  switch (Flavor) {
  case NeedsPositiveRetainCount: {
    InstructionClass Class = GetInstructionClass(Inst);
    switch (Class) {
    case IC_AutoreleasepoolPop:
    case IC_AutoreleasepoolPush:
    case IC_None:
      return false;
    default:
      return CanUse(Inst, Arg, PA, Class);
    }
  }

  case AutoreleasePoolBoundary: {
    InstructionClass Class = GetInstructionClass(Inst);
    switch (Class) {
    case IC_AutoreleasepoolPop:
    case IC_AutoreleasepoolPush:
      // These mark the end and begin of an autorelease pool scope.
      return true;
    default:
      // Nothing else does this.
      return false;
    }
  }

  case CanChangeRetainCount: {
    InstructionClass Class = GetInstructionClass(Inst);
    switch (Class) {
    case IC_AutoreleasepoolPop:
      // Conservatively assume this can decrement any count.
      return true;
    case IC_AutoreleasepoolPush:
    case IC_None:
      return false;
    default:
      return CanAlterRefCount(Inst, Arg, PA, Class);
    }
  }

  case RetainAutoreleaseDep:
    switch (GetBasicInstructionClass(Inst)) {
    case IC_AutoreleasepoolPop:
    case IC_AutoreleasepoolPush:
      // Don't merge an objc_autorelease with an objc_retain inside a different
      // autoreleasepool scope.
      return true;
    case IC_Retain:
    case IC_RetainRV:
      // Check for a retain of the same pointer for merging.
      return GetObjCArg(Inst) == Arg;
    default:
      // Nothing else matters for objc_retainAutorelease formation.
      return false;
    }

  case RetainAutoreleaseRVDep: {
    InstructionClass Class = GetBasicInstructionClass(Inst);
    switch (Class) {
    case IC_Retain:
    case IC_RetainRV:
      // Check for a retain of the same pointer for merging.
      return GetObjCArg(Inst) == Arg;
    default:
      // Anything that can autorelease interrupts
      // retainAutoreleaseReturnValue formation.
      return CanInterruptRV(Class);
    }
  }

  case RetainRVDep:
    return CanInterruptRV(GetBasicInstructionClass(Inst));
  }

  llvm_unreachable("Invalid dependence flavor");
}

/// FindDependencies - Walk up the CFG from StartPos (which is in StartBB) and
/// find local and non-local dependencies on Arg.
/// TODO: Cache results?
static void
FindDependencies(DependenceKind Flavor,
                 const Value *Arg,
                 BasicBlock *StartBB, Instruction *StartInst,
                 SmallPtrSet<Instruction *, 4> &DependingInstructions,
                 SmallPtrSet<const BasicBlock *, 4> &Visited,
                 ProvenanceAnalysis &PA) {
  BasicBlock::iterator StartPos = StartInst;

  SmallVector<std::pair<BasicBlock *, BasicBlock::iterator>, 4> Worklist;
  Worklist.push_back(std::make_pair(StartBB, StartPos));
  do {
    std::pair<BasicBlock *, BasicBlock::iterator> Pair =
      Worklist.pop_back_val();
    BasicBlock *LocalStartBB = Pair.first;
    BasicBlock::iterator LocalStartPos = Pair.second;
    BasicBlock::iterator StartBBBegin = LocalStartBB->begin();
    for (;;) {
      if (LocalStartPos == StartBBBegin) {
        pred_iterator PI(LocalStartBB), PE(LocalStartBB, false);
        if (PI == PE)
          // If we've reached the function entry, produce a null dependence.
          DependingInstructions.insert(0);
        else
          // Add the predecessors to the worklist.
          do {
            BasicBlock *PredBB = *PI;
            if (Visited.insert(PredBB))
              Worklist.push_back(std::make_pair(PredBB, PredBB->end()));
          } while (++PI != PE);
        break;
      }

      Instruction *Inst = --LocalStartPos;
      if (Depends(Flavor, Inst, Arg, PA)) {
        DependingInstructions.insert(Inst);
        break;
      }
    }
  } while (!Worklist.empty());

  // Determine whether the original StartBB post-dominates all of the blocks we
  // visited. If not, insert a sentinal indicating that most optimizations are
  // not safe.
  for (SmallPtrSet<const BasicBlock *, 4>::const_iterator I = Visited.begin(),
       E = Visited.end(); I != E; ++I) {
    const BasicBlock *BB = *I;
    if (BB == StartBB)
      continue;
    const TerminatorInst *TI = cast<TerminatorInst>(&BB->back());
    for (succ_const_iterator SI(TI), SE(TI, false); SI != SE; ++SI) {
      const BasicBlock *Succ = *SI;
      if (Succ != StartBB && !Visited.count(Succ)) {
        DependingInstructions.insert(reinterpret_cast<Instruction *>(-1));
        return;
      }
    }
  }
}

static bool isNullOrUndef(const Value *V) {
  return isa<ConstantPointerNull>(V) || isa<UndefValue>(V);
}

static bool isNoopInstruction(const Instruction *I) {
  return isa<BitCastInst>(I) ||
         (isa<GetElementPtrInst>(I) &&
          cast<GetElementPtrInst>(I)->hasAllZeroIndices());
}

/// OptimizeRetainCall - Turn objc_retain into
/// objc_retainAutoreleasedReturnValue if the operand is a return value.
void
ObjCARCOpt::OptimizeRetainCall(Function &F, Instruction *Retain) {
  ImmutableCallSite CS(GetObjCArg(Retain));
  const Instruction *Call = CS.getInstruction();
  if (!Call) return;
  if (Call->getParent() != Retain->getParent()) return;

  // Check that the call is next to the retain.
  BasicBlock::const_iterator I = Call;
  ++I;
  while (isNoopInstruction(I)) ++I;
  if (&*I != Retain)
    return;

  // Turn it to an objc_retainAutoreleasedReturnValue..
  Changed = true;
  ++NumPeeps;
  cast<CallInst>(Retain)->setCalledFunction(getRetainRVCallee(F.getParent()));
}

/// OptimizeRetainRVCall - Turn objc_retainAutoreleasedReturnValue into
/// objc_retain if the operand is not a return value.  Or, if it can be paired
/// with an objc_autoreleaseReturnValue, delete the pair and return true.
bool
ObjCARCOpt::OptimizeRetainRVCall(Function &F, Instruction *RetainRV) {
  // Check for the argument being from an immediately preceding call or invoke.
  const Value *Arg = GetObjCArg(RetainRV);
  ImmutableCallSite CS(Arg);
  if (const Instruction *Call = CS.getInstruction()) {
    if (Call->getParent() == RetainRV->getParent()) {
      BasicBlock::const_iterator I = Call;
      ++I;
      while (isNoopInstruction(I)) ++I;
      if (&*I == RetainRV)
        return false;
    } else if (const InvokeInst *II = dyn_cast<InvokeInst>(Call)) {
      BasicBlock *RetainRVParent = RetainRV->getParent();
      if (II->getNormalDest() == RetainRVParent) {
        BasicBlock::const_iterator I = RetainRVParent->begin();
        while (isNoopInstruction(I)) ++I;
        if (&*I == RetainRV)
          return false;
      }
    }
  }

  // Check for being preceded by an objc_autoreleaseReturnValue on the same
  // pointer. In this case, we can delete the pair.
  BasicBlock::iterator I = RetainRV, Begin = RetainRV->getParent()->begin();
  if (I != Begin) {
    do --I; while (I != Begin && isNoopInstruction(I));
    if (GetBasicInstructionClass(I) == IC_AutoreleaseRV &&
        GetObjCArg(I) == Arg) {
      Changed = true;
      ++NumPeeps;
      EraseInstruction(I);
      EraseInstruction(RetainRV);
      return true;
    }
  }

  // Turn it to a plain objc_retain.
  Changed = true;
  ++NumPeeps;
  cast<CallInst>(RetainRV)->setCalledFunction(getRetainCallee(F.getParent()));
  return false;
}

/// OptimizeAutoreleaseRVCall - Turn objc_autoreleaseReturnValue into
/// objc_autorelease if the result is not used as a return value.
void
ObjCARCOpt::OptimizeAutoreleaseRVCall(Function &F, Instruction *AutoreleaseRV) {
  // Check for a return of the pointer value.
  const Value *Ptr = GetObjCArg(AutoreleaseRV);
  SmallVector<const Value *, 2> Users;
  Users.push_back(Ptr);
  do {
    Ptr = Users.pop_back_val();
    for (Value::const_use_iterator UI = Ptr->use_begin(), UE = Ptr->use_end();
         UI != UE; ++UI) {
      const User *I = *UI;
      if (isa<ReturnInst>(I) || GetBasicInstructionClass(I) == IC_RetainRV)
        return;
      if (isa<BitCastInst>(I))
        Users.push_back(I);
    }
  } while (!Users.empty());

  Changed = true;
  ++NumPeeps;
  cast<CallInst>(AutoreleaseRV)->
    setCalledFunction(getAutoreleaseCallee(F.getParent()));
}

/// OptimizeIndividualCalls - Visit each call, one at a time, and make
/// simplifications without doing any additional analysis.
void ObjCARCOpt::OptimizeIndividualCalls(Function &F) {
  // Reset all the flags in preparation for recomputing them.
  UsedInThisFunction = 0;

  // Visit all objc_* calls in F.
  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ) {
    Instruction *Inst = &*I++;
    InstructionClass Class = GetBasicInstructionClass(Inst);

    switch (Class) {
    default: break;

    // Delete no-op casts. These function calls have special semantics, but
    // the semantics are entirely implemented via lowering in the front-end,
    // so by the time they reach the optimizer, they are just no-op calls
    // which return their argument.
    //
    // There are gray areas here, as the ability to cast reference-counted
    // pointers to raw void* and back allows code to break ARC assumptions,
    // however these are currently considered to be unimportant.
    case IC_NoopCast:
      Changed = true;
      ++NumNoops;
      EraseInstruction(Inst);
      continue;

    // If the pointer-to-weak-pointer is null, it's undefined behavior.
    case IC_StoreWeak:
    case IC_LoadWeak:
    case IC_LoadWeakRetained:
    case IC_InitWeak:
    case IC_DestroyWeak: {
      CallInst *CI = cast<CallInst>(Inst);
      if (isNullOrUndef(CI->getArgOperand(0))) {
        Changed = true;
        Type *Ty = CI->getArgOperand(0)->getType();
        new StoreInst(UndefValue::get(cast<PointerType>(Ty)->getElementType()),
                      Constant::getNullValue(Ty),
                      CI);
        CI->replaceAllUsesWith(UndefValue::get(CI->getType()));
        CI->eraseFromParent();
        continue;
      }
      break;
    }
    case IC_CopyWeak:
    case IC_MoveWeak: {
      CallInst *CI = cast<CallInst>(Inst);
      if (isNullOrUndef(CI->getArgOperand(0)) ||
          isNullOrUndef(CI->getArgOperand(1))) {
        Changed = true;
        Type *Ty = CI->getArgOperand(0)->getType();
        new StoreInst(UndefValue::get(cast<PointerType>(Ty)->getElementType()),
                      Constant::getNullValue(Ty),
                      CI);
        CI->replaceAllUsesWith(UndefValue::get(CI->getType()));
        CI->eraseFromParent();
        continue;
      }
      break;
    }
    case IC_Retain:
      OptimizeRetainCall(F, Inst);
      break;
    case IC_RetainRV:
      if (OptimizeRetainRVCall(F, Inst))
        continue;
      break;
    case IC_AutoreleaseRV:
      OptimizeAutoreleaseRVCall(F, Inst);
      break;
    }

    // objc_autorelease(x) -> objc_release(x) if x is otherwise unused.
    if (IsAutorelease(Class) && Inst->use_empty()) {
      CallInst *Call = cast<CallInst>(Inst);
      const Value *Arg = Call->getArgOperand(0);
      Arg = FindSingleUseIdentifiedObject(Arg);
      if (Arg) {
        Changed = true;
        ++NumAutoreleases;

        // Create the declaration lazily.
        LLVMContext &C = Inst->getContext();
        CallInst *NewCall =
          CallInst::Create(getReleaseCallee(F.getParent()),
                           Call->getArgOperand(0), "", Call);
        NewCall->setMetadata(ImpreciseReleaseMDKind,
                             MDNode::get(C, ArrayRef<Value *>()));
        EraseInstruction(Call);
        Inst = NewCall;
        Class = IC_Release;
      }
    }

    // For functions which can never be passed stack arguments, add
    // a tail keyword.
    if (IsAlwaysTail(Class)) {
      Changed = true;
      cast<CallInst>(Inst)->setTailCall();
    }

    // Set nounwind as needed.
    if (IsNoThrow(Class)) {
      Changed = true;
      cast<CallInst>(Inst)->setDoesNotThrow();
    }

    if (!IsNoopOnNull(Class)) {
      UsedInThisFunction |= 1 << Class;
      continue;
    }

    const Value *Arg = GetObjCArg(Inst);

    // ARC calls with null are no-ops. Delete them.
    if (isNullOrUndef(Arg)) {
      Changed = true;
      ++NumNoops;
      EraseInstruction(Inst);
      continue;
    }

    // Keep track of which of retain, release, autorelease, and retain_block
    // are actually present in this function.
    UsedInThisFunction |= 1 << Class;

    // If Arg is a PHI, and one or more incoming values to the
    // PHI are null, and the call is control-equivalent to the PHI, and there
    // are no relevant side effects between the PHI and the call, the call
    // could be pushed up to just those paths with non-null incoming values.
    // For now, don't bother splitting critical edges for this.
    SmallVector<std::pair<Instruction *, const Value *>, 4> Worklist;
    Worklist.push_back(std::make_pair(Inst, Arg));
    do {
      std::pair<Instruction *, const Value *> Pair = Worklist.pop_back_val();
      Inst = Pair.first;
      Arg = Pair.second;

      const PHINode *PN = dyn_cast<PHINode>(Arg);
      if (!PN) continue;

      // Determine if the PHI has any null operands, or any incoming
      // critical edges.
      bool HasNull = false;
      bool HasCriticalEdges = false;
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
        Value *Incoming =
          StripPointerCastsAndObjCCalls(PN->getIncomingValue(i));
        if (isNullOrUndef(Incoming))
          HasNull = true;
        else if (cast<TerminatorInst>(PN->getIncomingBlock(i)->back())
                   .getNumSuccessors() != 1) {
          HasCriticalEdges = true;
          break;
        }
      }
      // If we have null operands and no critical edges, optimize.
      if (!HasCriticalEdges && HasNull) {
        SmallPtrSet<Instruction *, 4> DependingInstructions;
        SmallPtrSet<const BasicBlock *, 4> Visited;

        // Check that there is nothing that cares about the reference
        // count between the call and the phi.
        switch (Class) {
        case IC_Retain:
        case IC_RetainBlock:
          // These can always be moved up.
          break;
        case IC_Release:
          // These can't be moved across things that care about the retain
          // count.
          FindDependencies(NeedsPositiveRetainCount, Arg,
                           Inst->getParent(), Inst,
                           DependingInstructions, Visited, PA);
          break;
        case IC_Autorelease:
          // These can't be moved across autorelease pool scope boundaries.
          FindDependencies(AutoreleasePoolBoundary, Arg,
                           Inst->getParent(), Inst,
                           DependingInstructions, Visited, PA);
          break;
        case IC_RetainRV:
        case IC_AutoreleaseRV:
          // Don't move these; the RV optimization depends on the autoreleaseRV
          // being tail called, and the retainRV being immediately after a call
          // (which might still happen if we get lucky with codegen layout, but
          // it's not worth taking the chance).
          continue;
        default:
          llvm_unreachable("Invalid dependence flavor");
        }

        if (DependingInstructions.size() == 1 &&
            *DependingInstructions.begin() == PN) {
          Changed = true;
          ++NumPartialNoops;
          // Clone the call into each predecessor that has a non-null value.
          CallInst *CInst = cast<CallInst>(Inst);
          Type *ParamTy = CInst->getArgOperand(0)->getType();
          for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
            Value *Incoming =
              StripPointerCastsAndObjCCalls(PN->getIncomingValue(i));
            if (!isNullOrUndef(Incoming)) {
              CallInst *Clone = cast<CallInst>(CInst->clone());
              Value *Op = PN->getIncomingValue(i);
              Instruction *InsertPos = &PN->getIncomingBlock(i)->back();
              if (Op->getType() != ParamTy)
                Op = new BitCastInst(Op, ParamTy, "", InsertPos);
              Clone->setArgOperand(0, Op);
              Clone->insertBefore(InsertPos);
              Worklist.push_back(std::make_pair(Clone, Incoming));
            }
          }
          // Erase the original call.
          EraseInstruction(CInst);
          continue;
        }
      }
    } while (!Worklist.empty());
  }
}

/// CheckForCFGHazards - Check for critical edges, loop boundaries, irreducible
/// control flow, or other CFG structures where moving code across the edge
/// would result in it being executed more.
void
ObjCARCOpt::CheckForCFGHazards(const BasicBlock *BB,
                               DenseMap<const BasicBlock *, BBState> &BBStates,
                               BBState &MyStates) const {
  // If any top-down local-use or possible-dec has a succ which is earlier in
  // the sequence, forget it.
  for (BBState::ptr_iterator I = MyStates.top_down_ptr_begin(),
       E = MyStates.top_down_ptr_end(); I != E; ++I)
    switch (I->second.GetSeq()) {
    default: break;
    case S_Use: {
      const Value *Arg = I->first;
      const TerminatorInst *TI = cast<TerminatorInst>(&BB->back());
      bool SomeSuccHasSame = false;
      bool AllSuccsHaveSame = true;
      PtrState &S = I->second;
      succ_const_iterator SI(TI), SE(TI, false);

      // If the terminator is an invoke marked with the
      // clang.arc.no_objc_arc_exceptions metadata, the unwind edge can be
      // ignored, for ARC purposes.
      if (isa<InvokeInst>(TI) && TI->getMetadata(NoObjCARCExceptionsMDKind))
        --SE;

      for (; SI != SE; ++SI) {
        Sequence SuccSSeq = S_None;
        bool SuccSRRIKnownSafe = false;
        // If VisitBottomUp has pointer information for this successor, take
        // what we know about it.
        DenseMap<const BasicBlock *, BBState>::iterator BBI =
          BBStates.find(*SI);
        assert(BBI != BBStates.end());
        const PtrState &SuccS = BBI->second.getPtrBottomUpState(Arg);
        SuccSSeq = SuccS.GetSeq();
        SuccSRRIKnownSafe = SuccS.RRI.KnownSafe;
        switch (SuccSSeq) {
        case S_None:
        case S_CanRelease: {
          if (!S.RRI.KnownSafe && !SuccSRRIKnownSafe) {
            S.ClearSequenceProgress();
            break;
          }
          continue;
        }
        case S_Use:
          SomeSuccHasSame = true;
          break;
        case S_Stop:
        case S_Release:
        case S_MovableRelease:
          if (!S.RRI.KnownSafe && !SuccSRRIKnownSafe)
            AllSuccsHaveSame = false;
          break;
        case S_Retain:
          llvm_unreachable("bottom-up pointer in retain state!");
        }
      }
      // If the state at the other end of any of the successor edges
      // matches the current state, require all edges to match. This
      // guards against loops in the middle of a sequence.
      if (SomeSuccHasSame && !AllSuccsHaveSame)
        S.ClearSequenceProgress();
      break;
    }
    case S_CanRelease: {
      const Value *Arg = I->first;
      const TerminatorInst *TI = cast<TerminatorInst>(&BB->back());
      bool SomeSuccHasSame = false;
      bool AllSuccsHaveSame = true;
      PtrState &S = I->second;
      succ_const_iterator SI(TI), SE(TI, false);

      // If the terminator is an invoke marked with the
      // clang.arc.no_objc_arc_exceptions metadata, the unwind edge can be
      // ignored, for ARC purposes.
      if (isa<InvokeInst>(TI) && TI->getMetadata(NoObjCARCExceptionsMDKind))
        --SE;

      for (; SI != SE; ++SI) {
        Sequence SuccSSeq = S_None;
        bool SuccSRRIKnownSafe = false;
        // If VisitBottomUp has pointer information for this successor, take
        // what we know about it.
        DenseMap<const BasicBlock *, BBState>::iterator BBI =
          BBStates.find(*SI);
        assert(BBI != BBStates.end());
        const PtrState &SuccS = BBI->second.getPtrBottomUpState(Arg);
        SuccSSeq = SuccS.GetSeq();
        SuccSRRIKnownSafe = SuccS.RRI.KnownSafe;
        switch (SuccSSeq) {
        case S_None: {
          if (!S.RRI.KnownSafe && !SuccSRRIKnownSafe) {
            S.ClearSequenceProgress();
            break;
          }
          continue;
        }
        case S_CanRelease:
          SomeSuccHasSame = true;
          break;
        case S_Stop:
        case S_Release:
        case S_MovableRelease:
        case S_Use:
          if (!S.RRI.KnownSafe && !SuccSRRIKnownSafe)
            AllSuccsHaveSame = false;
          break;
        case S_Retain:
          llvm_unreachable("bottom-up pointer in retain state!");
        }
      }
      // If the state at the other end of any of the successor edges
      // matches the current state, require all edges to match. This
      // guards against loops in the middle of a sequence.
      if (SomeSuccHasSame && !AllSuccsHaveSame)
        S.ClearSequenceProgress();
      break;
    }
    }
}

bool
ObjCARCOpt::VisitInstructionBottomUp(Instruction *Inst,
                                     BasicBlock *BB,
                                     MapVector<Value *, RRInfo> &Retains,
                                     BBState &MyStates) {
  bool NestingDetected = false;
  InstructionClass Class = GetInstructionClass(Inst);
  const Value *Arg = 0;

  switch (Class) {
  case IC_Release: {
    Arg = GetObjCArg(Inst);

    PtrState &S = MyStates.getPtrBottomUpState(Arg);

    // If we see two releases in a row on the same pointer. If so, make
    // a note, and we'll cicle back to revisit it after we've
    // hopefully eliminated the second release, which may allow us to
    // eliminate the first release too.
    // Theoretically we could implement removal of nested retain+release
    // pairs by making PtrState hold a stack of states, but this is
    // simple and avoids adding overhead for the non-nested case.
    if (S.GetSeq() == S_Release || S.GetSeq() == S_MovableRelease)
      NestingDetected = true;

    MDNode *ReleaseMetadata = Inst->getMetadata(ImpreciseReleaseMDKind);
    S.ResetSequenceProgress(ReleaseMetadata ? S_MovableRelease : S_Release);
    S.RRI.ReleaseMetadata = ReleaseMetadata;
    S.RRI.KnownSafe = S.IsKnownIncremented();
    S.RRI.IsTailCallRelease = cast<CallInst>(Inst)->isTailCall();
    S.RRI.Calls.insert(Inst);

    S.SetKnownPositiveRefCount();
    break;
  }
  case IC_RetainBlock:
    // An objc_retainBlock call with just a use may need to be kept,
    // because it may be copying a block from the stack to the heap.
    if (!IsRetainBlockOptimizable(Inst))
      break;
    // FALLTHROUGH
  case IC_Retain:
  case IC_RetainRV: {
    Arg = GetObjCArg(Inst);

    PtrState &S = MyStates.getPtrBottomUpState(Arg);
    S.SetKnownPositiveRefCount();

    switch (S.GetSeq()) {
    case S_Stop:
    case S_Release:
    case S_MovableRelease:
    case S_Use:
      S.RRI.ReverseInsertPts.clear();
      // FALL THROUGH
    case S_CanRelease:
      // Don't do retain+release tracking for IC_RetainRV, because it's
      // better to let it remain as the first instruction after a call.
      if (Class != IC_RetainRV) {
        S.RRI.IsRetainBlock = Class == IC_RetainBlock;
        Retains[Inst] = S.RRI;
      }
      S.ClearSequenceProgress();
      break;
    case S_None:
      break;
    case S_Retain:
      llvm_unreachable("bottom-up pointer in retain state!");
    }
    return NestingDetected;
  }
  case IC_AutoreleasepoolPop:
    // Conservatively, clear MyStates for all known pointers.
    MyStates.clearBottomUpPointers();
    return NestingDetected;
  case IC_AutoreleasepoolPush:
  case IC_None:
    // These are irrelevant.
    return NestingDetected;
  default:
    break;
  }

  // Consider any other possible effects of this instruction on each
  // pointer being tracked.
  for (BBState::ptr_iterator MI = MyStates.bottom_up_ptr_begin(),
       ME = MyStates.bottom_up_ptr_end(); MI != ME; ++MI) {
    const Value *Ptr = MI->first;
    if (Ptr == Arg)
      continue; // Handled above.
    PtrState &S = MI->second;
    Sequence Seq = S.GetSeq();

    // Check for possible releases.
    if (CanAlterRefCount(Inst, Ptr, PA, Class)) {
      S.ClearRefCount();
      switch (Seq) {
      case S_Use:
        S.SetSeq(S_CanRelease);
        continue;
      case S_CanRelease:
      case S_Release:
      case S_MovableRelease:
      case S_Stop:
      case S_None:
        break;
      case S_Retain:
        llvm_unreachable("bottom-up pointer in retain state!");
      }
    }

    // Check for possible direct uses.
    switch (Seq) {
    case S_Release:
    case S_MovableRelease:
      if (CanUse(Inst, Ptr, PA, Class)) {
        assert(S.RRI.ReverseInsertPts.empty());
        // If this is an invoke instruction, we're scanning it as part of
        // one of its successor blocks, since we can't insert code after it
        // in its own block, and we don't want to split critical edges.
        if (isa<InvokeInst>(Inst))
          S.RRI.ReverseInsertPts.insert(BB->getFirstInsertionPt());
        else
          S.RRI.ReverseInsertPts.insert(llvm::next(BasicBlock::iterator(Inst)));
        S.SetSeq(S_Use);
      } else if (Seq == S_Release &&
                 (Class == IC_User || Class == IC_CallOrUser)) {
        // Non-movable releases depend on any possible objc pointer use.
        S.SetSeq(S_Stop);
        assert(S.RRI.ReverseInsertPts.empty());
        // As above; handle invoke specially.
        if (isa<InvokeInst>(Inst))
          S.RRI.ReverseInsertPts.insert(BB->getFirstInsertionPt());
        else
          S.RRI.ReverseInsertPts.insert(llvm::next(BasicBlock::iterator(Inst)));
      }
      break;
    case S_Stop:
      if (CanUse(Inst, Ptr, PA, Class))
        S.SetSeq(S_Use);
      break;
    case S_CanRelease:
    case S_Use:
    case S_None:
      break;
    case S_Retain:
      llvm_unreachable("bottom-up pointer in retain state!");
    }
  }

  return NestingDetected;
}

bool
ObjCARCOpt::VisitBottomUp(BasicBlock *BB,
                          DenseMap<const BasicBlock *, BBState> &BBStates,
                          MapVector<Value *, RRInfo> &Retains) {
  bool NestingDetected = false;
  BBState &MyStates = BBStates[BB];

  // Merge the states from each successor to compute the initial state
  // for the current block.
  BBState::edge_iterator SI(MyStates.succ_begin()),
                         SE(MyStates.succ_end());
  if (SI != SE) {
    const BasicBlock *Succ = *SI;
    DenseMap<const BasicBlock *, BBState>::iterator I = BBStates.find(Succ);
    assert(I != BBStates.end());
    MyStates.InitFromSucc(I->second);
    ++SI;
    for (; SI != SE; ++SI) {
      Succ = *SI;
      I = BBStates.find(Succ);
      assert(I != BBStates.end());
      MyStates.MergeSucc(I->second);
    }
  }

  // Visit all the instructions, bottom-up.
  for (BasicBlock::iterator I = BB->end(), E = BB->begin(); I != E; --I) {
    Instruction *Inst = llvm::prior(I);

    // Invoke instructions are visited as part of their successors (below).
    if (isa<InvokeInst>(Inst))
      continue;

    NestingDetected |= VisitInstructionBottomUp(Inst, BB, Retains, MyStates);
  }

  // If there's a predecessor with an invoke, visit the invoke as if it were
  // part of this block, since we can't insert code after an invoke in its own
  // block, and we don't want to split critical edges.
  for (BBState::edge_iterator PI(MyStates.pred_begin()),
       PE(MyStates.pred_end()); PI != PE; ++PI) {
    BasicBlock *Pred = *PI;
    if (InvokeInst *II = dyn_cast<InvokeInst>(&Pred->back()))
      NestingDetected |= VisitInstructionBottomUp(II, BB, Retains, MyStates);
  }

  return NestingDetected;
}

bool
ObjCARCOpt::VisitInstructionTopDown(Instruction *Inst,
                                    DenseMap<Value *, RRInfo> &Releases,
                                    BBState &MyStates) {
  bool NestingDetected = false;
  InstructionClass Class = GetInstructionClass(Inst);
  const Value *Arg = 0;

  switch (Class) {
  case IC_RetainBlock:
    // An objc_retainBlock call with just a use may need to be kept,
    // because it may be copying a block from the stack to the heap.
    if (!IsRetainBlockOptimizable(Inst))
      break;
    // FALLTHROUGH
  case IC_Retain:
  case IC_RetainRV: {
    Arg = GetObjCArg(Inst);

    PtrState &S = MyStates.getPtrTopDownState(Arg);

    // Don't do retain+release tracking for IC_RetainRV, because it's
    // better to let it remain as the first instruction after a call.
    if (Class != IC_RetainRV) {
      // If we see two retains in a row on the same pointer. If so, make
      // a note, and we'll cicle back to revisit it after we've
      // hopefully eliminated the second retain, which may allow us to
      // eliminate the first retain too.
      // Theoretically we could implement removal of nested retain+release
      // pairs by making PtrState hold a stack of states, but this is
      // simple and avoids adding overhead for the non-nested case.
      if (S.GetSeq() == S_Retain)
        NestingDetected = true;

      S.ResetSequenceProgress(S_Retain);
      S.RRI.IsRetainBlock = Class == IC_RetainBlock;
      S.RRI.KnownSafe = S.IsKnownIncremented();
      S.RRI.Calls.insert(Inst);
    }

    S.SetKnownPositiveRefCount();

    // A retain can be a potential use; procede to the generic checking
    // code below.
    break;
  }
  case IC_Release: {
    Arg = GetObjCArg(Inst);

    PtrState &S = MyStates.getPtrTopDownState(Arg);
    S.ClearRefCount();

    switch (S.GetSeq()) {
    case S_Retain:
    case S_CanRelease:
      S.RRI.ReverseInsertPts.clear();
      // FALL THROUGH
    case S_Use:
      S.RRI.ReleaseMetadata = Inst->getMetadata(ImpreciseReleaseMDKind);
      S.RRI.IsTailCallRelease = cast<CallInst>(Inst)->isTailCall();
      Releases[Inst] = S.RRI;
      S.ClearSequenceProgress();
      break;
    case S_None:
      break;
    case S_Stop:
    case S_Release:
    case S_MovableRelease:
      llvm_unreachable("top-down pointer in release state!");
    }
    break;
  }
  case IC_AutoreleasepoolPop:
    // Conservatively, clear MyStates for all known pointers.
    MyStates.clearTopDownPointers();
    return NestingDetected;
  case IC_AutoreleasepoolPush:
  case IC_None:
    // These are irrelevant.
    return NestingDetected;
  default:
    break;
  }

  // Consider any other possible effects of this instruction on each
  // pointer being tracked.
  for (BBState::ptr_iterator MI = MyStates.top_down_ptr_begin(),
       ME = MyStates.top_down_ptr_end(); MI != ME; ++MI) {
    const Value *Ptr = MI->first;
    if (Ptr == Arg)
      continue; // Handled above.
    PtrState &S = MI->second;
    Sequence Seq = S.GetSeq();

    // Check for possible releases.
    if (CanAlterRefCount(Inst, Ptr, PA, Class)) {
      S.ClearRefCount();
      switch (Seq) {
      case S_Retain:
        S.SetSeq(S_CanRelease);
        assert(S.RRI.ReverseInsertPts.empty());
        S.RRI.ReverseInsertPts.insert(Inst);

        // One call can't cause a transition from S_Retain to S_CanRelease
        // and S_CanRelease to S_Use. If we've made the first transition,
        // we're done.
        continue;
      case S_Use:
      case S_CanRelease:
      case S_None:
        break;
      case S_Stop:
      case S_Release:
      case S_MovableRelease:
        llvm_unreachable("top-down pointer in release state!");
      }
    }

    // Check for possible direct uses.
    switch (Seq) {
    case S_CanRelease:
      if (CanUse(Inst, Ptr, PA, Class))
        S.SetSeq(S_Use);
      break;
    case S_Retain:
    case S_Use:
    case S_None:
      break;
    case S_Stop:
    case S_Release:
    case S_MovableRelease:
      llvm_unreachable("top-down pointer in release state!");
    }
  }

  return NestingDetected;
}

bool
ObjCARCOpt::VisitTopDown(BasicBlock *BB,
                         DenseMap<const BasicBlock *, BBState> &BBStates,
                         DenseMap<Value *, RRInfo> &Releases) {
  bool NestingDetected = false;
  BBState &MyStates = BBStates[BB];

  // Merge the states from each predecessor to compute the initial state
  // for the current block.
  BBState::edge_iterator PI(MyStates.pred_begin()),
                         PE(MyStates.pred_end());
  if (PI != PE) {
    const BasicBlock *Pred = *PI;
    DenseMap<const BasicBlock *, BBState>::iterator I = BBStates.find(Pred);
    assert(I != BBStates.end());
    MyStates.InitFromPred(I->second);
    ++PI;
    for (; PI != PE; ++PI) {
      Pred = *PI;
      I = BBStates.find(Pred);
      assert(I != BBStates.end());
      MyStates.MergePred(I->second);
    }
  }

  // Visit all the instructions, top-down.
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
    Instruction *Inst = I;
    NestingDetected |= VisitInstructionTopDown(Inst, Releases, MyStates);
  }

  CheckForCFGHazards(BB, BBStates, MyStates);
  return NestingDetected;
}

static void
ComputePostOrders(Function &F,
                  SmallVectorImpl<BasicBlock *> &PostOrder,
                  SmallVectorImpl<BasicBlock *> &ReverseCFGPostOrder,
                  unsigned NoObjCARCExceptionsMDKind,
                  DenseMap<const BasicBlock *, BBState> &BBStates) {
  /// Visited - The visited set, for doing DFS walks.
  SmallPtrSet<BasicBlock *, 16> Visited;

  // Do DFS, computing the PostOrder.
  SmallPtrSet<BasicBlock *, 16> OnStack;
  SmallVector<std::pair<BasicBlock *, succ_iterator>, 16> SuccStack;

  // Functions always have exactly one entry block, and we don't have
  // any other block that we treat like an entry block.
  BasicBlock *EntryBB = &F.getEntryBlock();
  BBState &MyStates = BBStates[EntryBB];
  MyStates.SetAsEntry();
  TerminatorInst *EntryTI = cast<TerminatorInst>(&EntryBB->back());
  SuccStack.push_back(std::make_pair(EntryBB, succ_iterator(EntryTI)));
  Visited.insert(EntryBB);
  OnStack.insert(EntryBB);
  do {
  dfs_next_succ:
    BasicBlock *CurrBB = SuccStack.back().first;
    TerminatorInst *TI = cast<TerminatorInst>(&CurrBB->back());
    succ_iterator SE(TI, false);

    // If the terminator is an invoke marked with the
    // clang.arc.no_objc_arc_exceptions metadata, the unwind edge can be
    // ignored, for ARC purposes.
    if (isa<InvokeInst>(TI) && TI->getMetadata(NoObjCARCExceptionsMDKind))
      --SE;

    while (SuccStack.back().second != SE) {
      BasicBlock *SuccBB = *SuccStack.back().second++;
      if (Visited.insert(SuccBB)) {
        TerminatorInst *TI = cast<TerminatorInst>(&SuccBB->back());
        SuccStack.push_back(std::make_pair(SuccBB, succ_iterator(TI)));
        BBStates[CurrBB].addSucc(SuccBB);
        BBState &SuccStates = BBStates[SuccBB];
        SuccStates.addPred(CurrBB);
        OnStack.insert(SuccBB);
        goto dfs_next_succ;
      }

      if (!OnStack.count(SuccBB)) {
        BBStates[CurrBB].addSucc(SuccBB);
        BBStates[SuccBB].addPred(CurrBB);
      }
    }
    OnStack.erase(CurrBB);
    PostOrder.push_back(CurrBB);
    SuccStack.pop_back();
  } while (!SuccStack.empty());

  Visited.clear();

  // Do reverse-CFG DFS, computing the reverse-CFG PostOrder.
  // Functions may have many exits, and there also blocks which we treat
  // as exits due to ignored edges.
  SmallVector<std::pair<BasicBlock *, BBState::edge_iterator>, 16> PredStack;
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    BasicBlock *ExitBB = I;
    BBState &MyStates = BBStates[ExitBB];
    if (!MyStates.isExit())
      continue;

    MyStates.SetAsExit();

    PredStack.push_back(std::make_pair(ExitBB, MyStates.pred_begin()));
    Visited.insert(ExitBB);
    while (!PredStack.empty()) {
    reverse_dfs_next_succ:
      BBState::edge_iterator PE = BBStates[PredStack.back().first].pred_end();
      while (PredStack.back().second != PE) {
        BasicBlock *BB = *PredStack.back().second++;
        if (Visited.insert(BB)) {
          PredStack.push_back(std::make_pair(BB, BBStates[BB].pred_begin()));
          goto reverse_dfs_next_succ;
        }
      }
      ReverseCFGPostOrder.push_back(PredStack.pop_back_val().first);
    }
  }
}

// Visit - Visit the function both top-down and bottom-up.
bool
ObjCARCOpt::Visit(Function &F,
                  DenseMap<const BasicBlock *, BBState> &BBStates,
                  MapVector<Value *, RRInfo> &Retains,
                  DenseMap<Value *, RRInfo> &Releases) {

  // Use reverse-postorder traversals, because we magically know that loops
  // will be well behaved, i.e. they won't repeatedly call retain on a single
  // pointer without doing a release. We can't use the ReversePostOrderTraversal
  // class here because we want the reverse-CFG postorder to consider each
  // function exit point, and we want to ignore selected cycle edges.
  SmallVector<BasicBlock *, 16> PostOrder;
  SmallVector<BasicBlock *, 16> ReverseCFGPostOrder;
  ComputePostOrders(F, PostOrder, ReverseCFGPostOrder,
                    NoObjCARCExceptionsMDKind,
                    BBStates);

  // Use reverse-postorder on the reverse CFG for bottom-up.
  bool BottomUpNestingDetected = false;
  for (SmallVectorImpl<BasicBlock *>::const_reverse_iterator I =
       ReverseCFGPostOrder.rbegin(), E = ReverseCFGPostOrder.rend();
       I != E; ++I)
    BottomUpNestingDetected |= VisitBottomUp(*I, BBStates, Retains);

  // Use reverse-postorder for top-down.
  bool TopDownNestingDetected = false;
  for (SmallVectorImpl<BasicBlock *>::const_reverse_iterator I =
       PostOrder.rbegin(), E = PostOrder.rend();
       I != E; ++I)
    TopDownNestingDetected |= VisitTopDown(*I, BBStates, Releases);

  return TopDownNestingDetected && BottomUpNestingDetected;
}

/// MoveCalls - Move the calls in RetainsToMove and ReleasesToMove.
void ObjCARCOpt::MoveCalls(Value *Arg,
                           RRInfo &RetainsToMove,
                           RRInfo &ReleasesToMove,
                           MapVector<Value *, RRInfo> &Retains,
                           DenseMap<Value *, RRInfo> &Releases,
                           SmallVectorImpl<Instruction *> &DeadInsts,
                           Module *M) {
  Type *ArgTy = Arg->getType();
  Type *ParamTy = PointerType::getUnqual(Type::getInt8Ty(ArgTy->getContext()));

  // Insert the new retain and release calls.
  for (SmallPtrSet<Instruction *, 2>::const_iterator
       PI = ReleasesToMove.ReverseInsertPts.begin(),
       PE = ReleasesToMove.ReverseInsertPts.end(); PI != PE; ++PI) {
    Instruction *InsertPt = *PI;
    Value *MyArg = ArgTy == ParamTy ? Arg :
                   new BitCastInst(Arg, ParamTy, "", InsertPt);
    CallInst *Call =
      CallInst::Create(RetainsToMove.IsRetainBlock ?
                         getRetainBlockCallee(M) : getRetainCallee(M),
                       MyArg, "", InsertPt);
    Call->setDoesNotThrow();
    if (RetainsToMove.IsRetainBlock)
      Call->setMetadata(CopyOnEscapeMDKind,
                        MDNode::get(M->getContext(), ArrayRef<Value *>()));
    else
      Call->setTailCall();
  }
  for (SmallPtrSet<Instruction *, 2>::const_iterator
       PI = RetainsToMove.ReverseInsertPts.begin(),
       PE = RetainsToMove.ReverseInsertPts.end(); PI != PE; ++PI) {
    Instruction *InsertPt = *PI;
    Value *MyArg = ArgTy == ParamTy ? Arg :
                   new BitCastInst(Arg, ParamTy, "", InsertPt);
    CallInst *Call = CallInst::Create(getReleaseCallee(M), MyArg,
                                      "", InsertPt);
    // Attach a clang.imprecise_release metadata tag, if appropriate.
    if (MDNode *M = ReleasesToMove.ReleaseMetadata)
      Call->setMetadata(ImpreciseReleaseMDKind, M);
    Call->setDoesNotThrow();
    if (ReleasesToMove.IsTailCallRelease)
      Call->setTailCall();
  }

  // Delete the original retain and release calls.
  for (SmallPtrSet<Instruction *, 2>::const_iterator
       AI = RetainsToMove.Calls.begin(),
       AE = RetainsToMove.Calls.end(); AI != AE; ++AI) {
    Instruction *OrigRetain = *AI;
    Retains.blot(OrigRetain);
    DeadInsts.push_back(OrigRetain);
  }
  for (SmallPtrSet<Instruction *, 2>::const_iterator
       AI = ReleasesToMove.Calls.begin(),
       AE = ReleasesToMove.Calls.end(); AI != AE; ++AI) {
    Instruction *OrigRelease = *AI;
    Releases.erase(OrigRelease);
    DeadInsts.push_back(OrigRelease);
  }
}

/// PerformCodePlacement - Identify pairings between the retains and releases,
/// and delete and/or move them.
bool
ObjCARCOpt::PerformCodePlacement(DenseMap<const BasicBlock *, BBState>
                                   &BBStates,
                                 MapVector<Value *, RRInfo> &Retains,
                                 DenseMap<Value *, RRInfo> &Releases,
                                 Module *M) {
  bool AnyPairsCompletelyEliminated = false;
  RRInfo RetainsToMove;
  RRInfo ReleasesToMove;
  SmallVector<Instruction *, 4> NewRetains;
  SmallVector<Instruction *, 4> NewReleases;
  SmallVector<Instruction *, 8> DeadInsts;

  // Visit each retain.
  for (MapVector<Value *, RRInfo>::const_iterator I = Retains.begin(),
       E = Retains.end(); I != E; ++I) {
    Value *V = I->first;
    if (!V) continue; // blotted

    Instruction *Retain = cast<Instruction>(V);
    Value *Arg = GetObjCArg(Retain);

    // If the object being released is in static or stack storage, we know it's
    // not being managed by ObjC reference counting, so we can delete pairs
    // regardless of what possible decrements or uses lie between them.
    bool KnownSafe = isa<Constant>(Arg) || isa<AllocaInst>(Arg);

    // A constant pointer can't be pointing to an object on the heap. It may
    // be reference-counted, but it won't be deleted.
    if (const LoadInst *LI = dyn_cast<LoadInst>(Arg))
      if (const GlobalVariable *GV =
            dyn_cast<GlobalVariable>(
              StripPointerCastsAndObjCCalls(LI->getPointerOperand())))
        if (GV->isConstant())
          KnownSafe = true;

    // If a pair happens in a region where it is known that the reference count
    // is already incremented, we can similarly ignore possible decrements.
    bool KnownSafeTD = true, KnownSafeBU = true;

    // Connect the dots between the top-down-collected RetainsToMove and
    // bottom-up-collected ReleasesToMove to form sets of related calls.
    // This is an iterative process so that we connect multiple releases
    // to multiple retains if needed.
    unsigned OldDelta = 0;
    unsigned NewDelta = 0;
    unsigned OldCount = 0;
    unsigned NewCount = 0;
    bool FirstRelease = true;
    bool FirstRetain = true;
    NewRetains.push_back(Retain);
    for (;;) {
      for (SmallVectorImpl<Instruction *>::const_iterator
           NI = NewRetains.begin(), NE = NewRetains.end(); NI != NE; ++NI) {
        Instruction *NewRetain = *NI;
        MapVector<Value *, RRInfo>::const_iterator It = Retains.find(NewRetain);
        assert(It != Retains.end());
        const RRInfo &NewRetainRRI = It->second;
        KnownSafeTD &= NewRetainRRI.KnownSafe;
        for (SmallPtrSet<Instruction *, 2>::const_iterator
             LI = NewRetainRRI.Calls.begin(),
             LE = NewRetainRRI.Calls.end(); LI != LE; ++LI) {
          Instruction *NewRetainRelease = *LI;
          DenseMap<Value *, RRInfo>::const_iterator Jt =
            Releases.find(NewRetainRelease);
          if (Jt == Releases.end())
            goto next_retain;
          const RRInfo &NewRetainReleaseRRI = Jt->second;
          assert(NewRetainReleaseRRI.Calls.count(NewRetain));
          if (ReleasesToMove.Calls.insert(NewRetainRelease)) {
            OldDelta -=
              BBStates[NewRetainRelease->getParent()].GetAllPathCount();

            // Merge the ReleaseMetadata and IsTailCallRelease values.
            if (FirstRelease) {
              ReleasesToMove.ReleaseMetadata =
                NewRetainReleaseRRI.ReleaseMetadata;
              ReleasesToMove.IsTailCallRelease =
                NewRetainReleaseRRI.IsTailCallRelease;
              FirstRelease = false;
            } else {
              if (ReleasesToMove.ReleaseMetadata !=
                    NewRetainReleaseRRI.ReleaseMetadata)
                ReleasesToMove.ReleaseMetadata = 0;
              if (ReleasesToMove.IsTailCallRelease !=
                    NewRetainReleaseRRI.IsTailCallRelease)
                ReleasesToMove.IsTailCallRelease = false;
            }

            // Collect the optimal insertion points.
            if (!KnownSafe)
              for (SmallPtrSet<Instruction *, 2>::const_iterator
                   RI = NewRetainReleaseRRI.ReverseInsertPts.begin(),
                   RE = NewRetainReleaseRRI.ReverseInsertPts.end();
                   RI != RE; ++RI) {
                Instruction *RIP = *RI;
                if (ReleasesToMove.ReverseInsertPts.insert(RIP))
                  NewDelta -= BBStates[RIP->getParent()].GetAllPathCount();
              }
            NewReleases.push_back(NewRetainRelease);
          }
        }
      }
      NewRetains.clear();
      if (NewReleases.empty()) break;

      // Back the other way.
      for (SmallVectorImpl<Instruction *>::const_iterator
           NI = NewReleases.begin(), NE = NewReleases.end(); NI != NE; ++NI) {
        Instruction *NewRelease = *NI;
        DenseMap<Value *, RRInfo>::const_iterator It =
          Releases.find(NewRelease);
        assert(It != Releases.end());
        const RRInfo &NewReleaseRRI = It->second;
        KnownSafeBU &= NewReleaseRRI.KnownSafe;
        for (SmallPtrSet<Instruction *, 2>::const_iterator
             LI = NewReleaseRRI.Calls.begin(),
             LE = NewReleaseRRI.Calls.end(); LI != LE; ++LI) {
          Instruction *NewReleaseRetain = *LI;
          MapVector<Value *, RRInfo>::const_iterator Jt =
            Retains.find(NewReleaseRetain);
          if (Jt == Retains.end())
            goto next_retain;
          const RRInfo &NewReleaseRetainRRI = Jt->second;
          assert(NewReleaseRetainRRI.Calls.count(NewRelease));
          if (RetainsToMove.Calls.insert(NewReleaseRetain)) {
            unsigned PathCount =
              BBStates[NewReleaseRetain->getParent()].GetAllPathCount();
            OldDelta += PathCount;
            OldCount += PathCount;

            // Merge the IsRetainBlock values.
            if (FirstRetain) {
              RetainsToMove.IsRetainBlock = NewReleaseRetainRRI.IsRetainBlock;
              FirstRetain = false;
            } else if (ReleasesToMove.IsRetainBlock !=
                       NewReleaseRetainRRI.IsRetainBlock)
              // It's not possible to merge the sequences if one uses
              // objc_retain and the other uses objc_retainBlock.
              goto next_retain;

            // Collect the optimal insertion points.
            if (!KnownSafe)
              for (SmallPtrSet<Instruction *, 2>::const_iterator
                   RI = NewReleaseRetainRRI.ReverseInsertPts.begin(),
                   RE = NewReleaseRetainRRI.ReverseInsertPts.end();
                   RI != RE; ++RI) {
                Instruction *RIP = *RI;
                if (RetainsToMove.ReverseInsertPts.insert(RIP)) {
                  PathCount = BBStates[RIP->getParent()].GetAllPathCount();
                  NewDelta += PathCount;
                  NewCount += PathCount;
                }
              }
            NewRetains.push_back(NewReleaseRetain);
          }
        }
      }
      NewReleases.clear();
      if (NewRetains.empty()) break;
    }

    // If the pointer is known incremented or nested, we can safely delete the
    // pair regardless of what's between them.
    if (KnownSafeTD || KnownSafeBU) {
      RetainsToMove.ReverseInsertPts.clear();
      ReleasesToMove.ReverseInsertPts.clear();
      NewCount = 0;
    } else {
      // Determine whether the new insertion points we computed preserve the
      // balance of retain and release calls through the program.
      // TODO: If the fully aggressive solution isn't valid, try to find a
      // less aggressive solution which is.
      if (NewDelta != 0)
        goto next_retain;
    }

    // Determine whether the original call points are balanced in the retain and
    // release calls through the program. If not, conservatively don't touch
    // them.
    // TODO: It's theoretically possible to do code motion in this case, as
    // long as the existing imbalances are maintained.
    if (OldDelta != 0)
      goto next_retain;

    // Ok, everything checks out and we're all set. Let's move some code!
    Changed = true;
    assert(OldCount != 0 && "Unreachable code?");
    AnyPairsCompletelyEliminated = NewCount == 0;
    NumRRs += OldCount - NewCount;
    MoveCalls(Arg, RetainsToMove, ReleasesToMove,
              Retains, Releases, DeadInsts, M);

  next_retain:
    NewReleases.clear();
    NewRetains.clear();
    RetainsToMove.clear();
    ReleasesToMove.clear();
  }

  // Now that we're done moving everything, we can delete the newly dead
  // instructions, as we no longer need them as insert points.
  while (!DeadInsts.empty())
    EraseInstruction(DeadInsts.pop_back_val());

  return AnyPairsCompletelyEliminated;
}

/// OptimizeWeakCalls - Weak pointer optimizations.
void ObjCARCOpt::OptimizeWeakCalls(Function &F) {
  // First, do memdep-style RLE and S2L optimizations. We can't use memdep
  // itself because it uses AliasAnalysis and we need to do provenance
  // queries instead.
  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ) {
    Instruction *Inst = &*I++;
    InstructionClass Class = GetBasicInstructionClass(Inst);
    if (Class != IC_LoadWeak && Class != IC_LoadWeakRetained)
      continue;

    // Delete objc_loadWeak calls with no users.
    if (Class == IC_LoadWeak && Inst->use_empty()) {
      Inst->eraseFromParent();
      continue;
    }

    // TODO: For now, just look for an earlier available version of this value
    // within the same block. Theoretically, we could do memdep-style non-local
    // analysis too, but that would want caching. A better approach would be to
    // use the technique that EarlyCSE uses.
    inst_iterator Current = llvm::prior(I);
    BasicBlock *CurrentBB = Current.getBasicBlockIterator();
    for (BasicBlock::iterator B = CurrentBB->begin(),
                              J = Current.getInstructionIterator();
         J != B; --J) {
      Instruction *EarlierInst = &*llvm::prior(J);
      InstructionClass EarlierClass = GetInstructionClass(EarlierInst);
      switch (EarlierClass) {
      case IC_LoadWeak:
      case IC_LoadWeakRetained: {
        // If this is loading from the same pointer, replace this load's value
        // with that one.
        CallInst *Call = cast<CallInst>(Inst);
        CallInst *EarlierCall = cast<CallInst>(EarlierInst);
        Value *Arg = Call->getArgOperand(0);
        Value *EarlierArg = EarlierCall->getArgOperand(0);
        switch (PA.getAA()->alias(Arg, EarlierArg)) {
        case AliasAnalysis::MustAlias:
          Changed = true;
          // If the load has a builtin retain, insert a plain retain for it.
          if (Class == IC_LoadWeakRetained) {
            CallInst *CI =
              CallInst::Create(getRetainCallee(F.getParent()), EarlierCall,
                               "", Call);
            CI->setTailCall();
          }
          // Zap the fully redundant load.
          Call->replaceAllUsesWith(EarlierCall);
          Call->eraseFromParent();
          goto clobbered;
        case AliasAnalysis::MayAlias:
        case AliasAnalysis::PartialAlias:
          goto clobbered;
        case AliasAnalysis::NoAlias:
          break;
        }
        break;
      }
      case IC_StoreWeak:
      case IC_InitWeak: {
        // If this is storing to the same pointer and has the same size etc.
        // replace this load's value with the stored value.
        CallInst *Call = cast<CallInst>(Inst);
        CallInst *EarlierCall = cast<CallInst>(EarlierInst);
        Value *Arg = Call->getArgOperand(0);
        Value *EarlierArg = EarlierCall->getArgOperand(0);
        switch (PA.getAA()->alias(Arg, EarlierArg)) {
        case AliasAnalysis::MustAlias:
          Changed = true;
          // If the load has a builtin retain, insert a plain retain for it.
          if (Class == IC_LoadWeakRetained) {
            CallInst *CI =
              CallInst::Create(getRetainCallee(F.getParent()), EarlierCall,
                               "", Call);
            CI->setTailCall();
          }
          // Zap the fully redundant load.
          Call->replaceAllUsesWith(EarlierCall->getArgOperand(1));
          Call->eraseFromParent();
          goto clobbered;
        case AliasAnalysis::MayAlias:
        case AliasAnalysis::PartialAlias:
          goto clobbered;
        case AliasAnalysis::NoAlias:
          break;
        }
        break;
      }
      case IC_MoveWeak:
      case IC_CopyWeak:
        // TOOD: Grab the copied value.
        goto clobbered;
      case IC_AutoreleasepoolPush:
      case IC_None:
      case IC_User:
        // Weak pointers are only modified through the weak entry points
        // (and arbitrary calls, which could call the weak entry points).
        break;
      default:
        // Anything else could modify the weak pointer.
        goto clobbered;
      }
    }
  clobbered:;
  }

  // Then, for each destroyWeak with an alloca operand, check to see if
  // the alloca and all its users can be zapped.
  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ) {
    Instruction *Inst = &*I++;
    InstructionClass Class = GetBasicInstructionClass(Inst);
    if (Class != IC_DestroyWeak)
      continue;

    CallInst *Call = cast<CallInst>(Inst);
    Value *Arg = Call->getArgOperand(0);
    if (AllocaInst *Alloca = dyn_cast<AllocaInst>(Arg)) {
      for (Value::use_iterator UI = Alloca->use_begin(),
           UE = Alloca->use_end(); UI != UE; ++UI) {
        const Instruction *UserInst = cast<Instruction>(*UI);
        switch (GetBasicInstructionClass(UserInst)) {
        case IC_InitWeak:
        case IC_StoreWeak:
        case IC_DestroyWeak:
          continue;
        default:
          goto done;
        }
      }
      Changed = true;
      for (Value::use_iterator UI = Alloca->use_begin(),
           UE = Alloca->use_end(); UI != UE; ) {
        CallInst *UserInst = cast<CallInst>(*UI++);
        switch (GetBasicInstructionClass(UserInst)) {
        case IC_InitWeak:
        case IC_StoreWeak:
          // These functions return their second argument.
          UserInst->replaceAllUsesWith(UserInst->getArgOperand(1));
          break;
        case IC_DestroyWeak:
          // No return value.
          break;
        default:
          llvm_unreachable("alloca really is used!");
        }
        UserInst->eraseFromParent();
      }
      Alloca->eraseFromParent();
    done:;
    }
  }
}

/// OptimizeSequences - Identify program paths which execute sequences of
/// retains and releases which can be eliminated.
bool ObjCARCOpt::OptimizeSequences(Function &F) {
  /// Releases, Retains - These are used to store the results of the main flow
  /// analysis. These use Value* as the key instead of Instruction* so that the
  /// map stays valid when we get around to rewriting code and calls get
  /// replaced by arguments.
  DenseMap<Value *, RRInfo> Releases;
  MapVector<Value *, RRInfo> Retains;

  /// BBStates, This is used during the traversal of the function to track the
  /// states for each identified object at each block.
  DenseMap<const BasicBlock *, BBState> BBStates;

  // Analyze the CFG of the function, and all instructions.
  bool NestingDetected = Visit(F, BBStates, Retains, Releases);

  // Transform.
  return PerformCodePlacement(BBStates, Retains, Releases, F.getParent()) &&
         NestingDetected;
}

/// OptimizeReturns - Look for this pattern:
/// \code
///    %call = call i8* @something(...)
///    %2 = call i8* @objc_retain(i8* %call)
///    %3 = call i8* @objc_autorelease(i8* %2)
///    ret i8* %3
/// \endcode
/// And delete the retain and autorelease.
///
/// Otherwise if it's just this:
/// \code
///    %3 = call i8* @objc_autorelease(i8* %2)
///    ret i8* %3
/// \endcode
/// convert the autorelease to autoreleaseRV.
void ObjCARCOpt::OptimizeReturns(Function &F) {
  if (!F.getReturnType()->isPointerTy())
    return;

  SmallPtrSet<Instruction *, 4> DependingInstructions;
  SmallPtrSet<const BasicBlock *, 4> Visited;
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI) {
    BasicBlock *BB = FI;
    ReturnInst *Ret = dyn_cast<ReturnInst>(&BB->back());
    if (!Ret) continue;

    const Value *Arg = StripPointerCastsAndObjCCalls(Ret->getOperand(0));
    FindDependencies(NeedsPositiveRetainCount, Arg,
                     BB, Ret, DependingInstructions, Visited, PA);
    if (DependingInstructions.size() != 1)
      goto next_block;

    {
      CallInst *Autorelease =
        dyn_cast_or_null<CallInst>(*DependingInstructions.begin());
      if (!Autorelease)
        goto next_block;
      InstructionClass AutoreleaseClass = GetBasicInstructionClass(Autorelease);
      if (!IsAutorelease(AutoreleaseClass))
        goto next_block;
      if (GetObjCArg(Autorelease) != Arg)
        goto next_block;

      DependingInstructions.clear();
      Visited.clear();

      // Check that there is nothing that can affect the reference
      // count between the autorelease and the retain.
      FindDependencies(CanChangeRetainCount, Arg,
                       BB, Autorelease, DependingInstructions, Visited, PA);
      if (DependingInstructions.size() != 1)
        goto next_block;

      {
        CallInst *Retain =
          dyn_cast_or_null<CallInst>(*DependingInstructions.begin());

        // Check that we found a retain with the same argument.
        if (!Retain ||
            !IsRetain(GetBasicInstructionClass(Retain)) ||
            GetObjCArg(Retain) != Arg)
          goto next_block;

        DependingInstructions.clear();
        Visited.clear();

        // Convert the autorelease to an autoreleaseRV, since it's
        // returning the value.
        if (AutoreleaseClass == IC_Autorelease) {
          Autorelease->setCalledFunction(getAutoreleaseRVCallee(F.getParent()));
          AutoreleaseClass = IC_AutoreleaseRV;
        }

        // Check that there is nothing that can affect the reference
        // count between the retain and the call.
        // Note that Retain need not be in BB.
        FindDependencies(CanChangeRetainCount, Arg, Retain->getParent(), Retain,
                         DependingInstructions, Visited, PA);
        if (DependingInstructions.size() != 1)
          goto next_block;

        {
          CallInst *Call =
            dyn_cast_or_null<CallInst>(*DependingInstructions.begin());

          // Check that the pointer is the return value of the call.
          if (!Call || Arg != Call)
            goto next_block;

          // Check that the call is a regular call.
          InstructionClass Class = GetBasicInstructionClass(Call);
          if (Class != IC_CallOrUser && Class != IC_Call)
            goto next_block;

          // If so, we can zap the retain and autorelease.
          Changed = true;
          ++NumRets;
          EraseInstruction(Retain);
          EraseInstruction(Autorelease);
        }
      }
    }

  next_block:
    DependingInstructions.clear();
    Visited.clear();
  }
}

bool ObjCARCOpt::doInitialization(Module &M) {
  if (!EnableARCOpts)
    return false;

  // If nothing in the Module uses ARC, don't do anything.
  Run = ModuleHasARC(M);
  if (!Run)
    return false;

  // Identify the imprecise release metadata kind.
  ImpreciseReleaseMDKind =
    M.getContext().getMDKindID("clang.imprecise_release");
  CopyOnEscapeMDKind =
    M.getContext().getMDKindID("clang.arc.copy_on_escape");
  NoObjCARCExceptionsMDKind =
    M.getContext().getMDKindID("clang.arc.no_objc_arc_exceptions");

  // Intuitively, objc_retain and others are nocapture, however in practice
  // they are not, because they return their argument value. And objc_release
  // calls finalizers which can have arbitrary side effects.

  // These are initialized lazily.
  RetainRVCallee = 0;
  AutoreleaseRVCallee = 0;
  ReleaseCallee = 0;
  RetainCallee = 0;
  RetainBlockCallee = 0;
  AutoreleaseCallee = 0;

  return false;
}

bool ObjCARCOpt::runOnFunction(Function &F) {
  if (!EnableARCOpts)
    return false;

  // If nothing in the Module uses ARC, don't do anything.
  if (!Run)
    return false;

  Changed = false;

  PA.setAA(&getAnalysis<AliasAnalysis>());

  // This pass performs several distinct transformations. As a compile-time aid
  // when compiling code that isn't ObjC, skip these if the relevant ObjC
  // library functions aren't declared.

  // Preliminary optimizations. This also computs UsedInThisFunction.
  OptimizeIndividualCalls(F);

  // Optimizations for weak pointers.
  if (UsedInThisFunction & ((1 << IC_LoadWeak) |
                            (1 << IC_LoadWeakRetained) |
                            (1 << IC_StoreWeak) |
                            (1 << IC_InitWeak) |
                            (1 << IC_CopyWeak) |
                            (1 << IC_MoveWeak) |
                            (1 << IC_DestroyWeak)))
    OptimizeWeakCalls(F);

  // Optimizations for retain+release pairs.
  if (UsedInThisFunction & ((1 << IC_Retain) |
                            (1 << IC_RetainRV) |
                            (1 << IC_RetainBlock)))
    if (UsedInThisFunction & (1 << IC_Release))
      // Run OptimizeSequences until it either stops making changes or
      // no retain+release pair nesting is detected.
      while (OptimizeSequences(F)) {}

  // Optimizations if objc_autorelease is used.
  if (UsedInThisFunction & ((1 << IC_Autorelease) |
                            (1 << IC_AutoreleaseRV)))
    OptimizeReturns(F);

  return Changed;
}

void ObjCARCOpt::releaseMemory() {
  PA.clear();
}

//===----------------------------------------------------------------------===//
// ARC contraction.
//===----------------------------------------------------------------------===//

// TODO: ObjCARCContract could insert PHI nodes when uses aren't
// dominated by single calls.

#include "llvm/Operator.h"
#include "llvm/InlineAsm.h"
#include "llvm/Analysis/Dominators.h"

STATISTIC(NumStoreStrongs, "Number objc_storeStrong calls formed");

namespace {
  /// ObjCARCContract - Late ARC optimizations.  These change the IR in a way
  /// that makes it difficult to be analyzed by ObjCARCOpt, so it's run late.
  class ObjCARCContract : public FunctionPass {
    bool Changed;
    AliasAnalysis *AA;
    DominatorTree *DT;
    ProvenanceAnalysis PA;

    /// Run - A flag indicating whether this optimization pass should run.
    bool Run;

    /// StoreStrongCallee, etc. - Declarations for ObjC runtime
    /// functions, for use in creating calls to them. These are initialized
    /// lazily to avoid cluttering up the Module with unused declarations.
    Constant *StoreStrongCallee,
             *RetainAutoreleaseCallee, *RetainAutoreleaseRVCallee;

    /// RetainRVMarker - The inline asm string to insert between calls and
    /// RetainRV calls to make the optimization work on targets which need it.
    const MDString *RetainRVMarker;

    /// StoreStrongCalls - The set of inserted objc_storeStrong calls. If
    /// at the end of walking the function we have found no alloca
    /// instructions, these calls can be marked "tail".
    SmallPtrSet<CallInst *, 8> StoreStrongCalls;

    Constant *getStoreStrongCallee(Module *M);
    Constant *getRetainAutoreleaseCallee(Module *M);
    Constant *getRetainAutoreleaseRVCallee(Module *M);

    bool ContractAutorelease(Function &F, Instruction *Autorelease,
                             InstructionClass Class,
                             SmallPtrSet<Instruction *, 4>
                               &DependingInstructions,
                             SmallPtrSet<const BasicBlock *, 4>
                               &Visited);

    void ContractRelease(Instruction *Release,
                         inst_iterator &Iter);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual bool doInitialization(Module &M);
    virtual bool runOnFunction(Function &F);

  public:
    static char ID;
    ObjCARCContract() : FunctionPass(ID) {
      initializeObjCARCContractPass(*PassRegistry::getPassRegistry());
    }
  };
}

char ObjCARCContract::ID = 0;
INITIALIZE_PASS_BEGIN(ObjCARCContract,
                      "objc-arc-contract", "ObjC ARC contraction", false, false)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_END(ObjCARCContract,
                    "objc-arc-contract", "ObjC ARC contraction", false, false)

Pass *llvm::createObjCARCContractPass() {
  return new ObjCARCContract();
}

void ObjCARCContract::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<AliasAnalysis>();
  AU.addRequired<DominatorTree>();
  AU.setPreservesCFG();
}

Constant *ObjCARCContract::getStoreStrongCallee(Module *M) {
  if (!StoreStrongCallee) {
    LLVMContext &C = M->getContext();
    Type *I8X = PointerType::getUnqual(Type::getInt8Ty(C));
    Type *I8XX = PointerType::getUnqual(I8X);
    Type *Params[] = { I8XX, I8X };

    AttrListPtr Attributes = AttrListPtr()
      .addAttr(M->getContext(), AttrListPtr::FunctionIndex,
               Attributes::get(C, Attributes::NoUnwind))
      .addAttr(M->getContext(), 1, Attributes::get(C, Attributes::NoCapture));

    StoreStrongCallee =
      M->getOrInsertFunction(
        "objc_storeStrong",
        FunctionType::get(Type::getVoidTy(C), Params, /*isVarArg=*/false),
        Attributes);
  }
  return StoreStrongCallee;
}

Constant *ObjCARCContract::getRetainAutoreleaseCallee(Module *M) {
  if (!RetainAutoreleaseCallee) {
    LLVMContext &C = M->getContext();
    Type *I8X = PointerType::getUnqual(Type::getInt8Ty(C));
    Type *Params[] = { I8X };
    FunctionType *FTy = FunctionType::get(I8X, Params, /*isVarArg=*/false);
    AttrListPtr Attributes =
      AttrListPtr().addAttr(M->getContext(), AttrListPtr::FunctionIndex,
                            Attributes::get(C, Attributes::NoUnwind));
    RetainAutoreleaseCallee =
      M->getOrInsertFunction("objc_retainAutorelease", FTy, Attributes);
  }
  return RetainAutoreleaseCallee;
}

Constant *ObjCARCContract::getRetainAutoreleaseRVCallee(Module *M) {
  if (!RetainAutoreleaseRVCallee) {
    LLVMContext &C = M->getContext();
    Type *I8X = PointerType::getUnqual(Type::getInt8Ty(C));
    Type *Params[] = { I8X };
    FunctionType *FTy = FunctionType::get(I8X, Params, /*isVarArg=*/false);
    AttrListPtr Attributes =
      AttrListPtr().addAttr(M->getContext(), AttrListPtr::FunctionIndex,
                            Attributes::get(C, Attributes::NoUnwind));
    RetainAutoreleaseRVCallee =
      M->getOrInsertFunction("objc_retainAutoreleaseReturnValue", FTy,
                             Attributes);
  }
  return RetainAutoreleaseRVCallee;
}

/// ContractAutorelease - Merge an autorelease with a retain into a fused call.
bool
ObjCARCContract::ContractAutorelease(Function &F, Instruction *Autorelease,
                                     InstructionClass Class,
                                     SmallPtrSet<Instruction *, 4>
                                       &DependingInstructions,
                                     SmallPtrSet<const BasicBlock *, 4>
                                       &Visited) {
  const Value *Arg = GetObjCArg(Autorelease);

  // Check that there are no instructions between the retain and the autorelease
  // (such as an autorelease_pop) which may change the count.
  CallInst *Retain = 0;
  if (Class == IC_AutoreleaseRV)
    FindDependencies(RetainAutoreleaseRVDep, Arg,
                     Autorelease->getParent(), Autorelease,
                     DependingInstructions, Visited, PA);
  else
    FindDependencies(RetainAutoreleaseDep, Arg,
                     Autorelease->getParent(), Autorelease,
                     DependingInstructions, Visited, PA);

  Visited.clear();
  if (DependingInstructions.size() != 1) {
    DependingInstructions.clear();
    return false;
  }

  Retain = dyn_cast_or_null<CallInst>(*DependingInstructions.begin());
  DependingInstructions.clear();

  if (!Retain ||
      GetBasicInstructionClass(Retain) != IC_Retain ||
      GetObjCArg(Retain) != Arg)
    return false;

  Changed = true;
  ++NumPeeps;

  if (Class == IC_AutoreleaseRV)
    Retain->setCalledFunction(getRetainAutoreleaseRVCallee(F.getParent()));
  else
    Retain->setCalledFunction(getRetainAutoreleaseCallee(F.getParent()));

  EraseInstruction(Autorelease);
  return true;
}

/// ContractRelease - Attempt to merge an objc_release with a store, load, and
/// objc_retain to form an objc_storeStrong. This can be a little tricky because
/// the instructions don't always appear in order, and there may be unrelated
/// intervening instructions.
void ObjCARCContract::ContractRelease(Instruction *Release,
                                      inst_iterator &Iter) {
  LoadInst *Load = dyn_cast<LoadInst>(GetObjCArg(Release));
  if (!Load || !Load->isSimple()) return;

  // For now, require everything to be in one basic block.
  BasicBlock *BB = Release->getParent();
  if (Load->getParent() != BB) return;

  // Walk down to find the store and the release, which may be in either order.
  BasicBlock::iterator I = Load, End = BB->end();
  ++I;
  AliasAnalysis::Location Loc = AA->getLocation(Load);
  StoreInst *Store = 0;
  bool SawRelease = false;
  for (; !Store || !SawRelease; ++I) {
    if (I == End)
      return;

    Instruction *Inst = I;
    if (Inst == Release) {
      SawRelease = true;
      continue;
    }

    InstructionClass Class = GetBasicInstructionClass(Inst);

    // Unrelated retains are harmless.
    if (IsRetain(Class))
      continue;

    if (Store) {
      // The store is the point where we're going to put the objc_storeStrong,
      // so make sure there are no uses after it.
      if (CanUse(Inst, Load, PA, Class))
        return;
    } else if (AA->getModRefInfo(Inst, Loc) & AliasAnalysis::Mod) {
      // We are moving the load down to the store, so check for anything
      // else which writes to the memory between the load and the store.
      Store = dyn_cast<StoreInst>(Inst);
      if (!Store || !Store->isSimple()) return;
      if (Store->getPointerOperand() != Loc.Ptr) return;
    }
  }

  Value *New = StripPointerCastsAndObjCCalls(Store->getValueOperand());

  // Walk up to find the retain.
  I = Store;
  BasicBlock::iterator Begin = BB->begin();
  while (I != Begin && GetBasicInstructionClass(I) != IC_Retain)
    --I;
  Instruction *Retain = I;
  if (GetBasicInstructionClass(Retain) != IC_Retain) return;
  if (GetObjCArg(Retain) != New) return;

  Changed = true;
  ++NumStoreStrongs;

  LLVMContext &C = Release->getContext();
  Type *I8X = PointerType::getUnqual(Type::getInt8Ty(C));
  Type *I8XX = PointerType::getUnqual(I8X);

  Value *Args[] = { Load->getPointerOperand(), New };
  if (Args[0]->getType() != I8XX)
    Args[0] = new BitCastInst(Args[0], I8XX, "", Store);
  if (Args[1]->getType() != I8X)
    Args[1] = new BitCastInst(Args[1], I8X, "", Store);
  CallInst *StoreStrong =
    CallInst::Create(getStoreStrongCallee(BB->getParent()->getParent()),
                     Args, "", Store);
  StoreStrong->setDoesNotThrow();
  StoreStrong->setDebugLoc(Store->getDebugLoc());

  // We can't set the tail flag yet, because we haven't yet determined
  // whether there are any escaping allocas. Remember this call, so that
  // we can set the tail flag once we know it's safe.
  StoreStrongCalls.insert(StoreStrong);

  if (&*Iter == Store) ++Iter;
  Store->eraseFromParent();
  Release->eraseFromParent();
  EraseInstruction(Retain);
  if (Load->use_empty())
    Load->eraseFromParent();
}

bool ObjCARCContract::doInitialization(Module &M) {
  // If nothing in the Module uses ARC, don't do anything.
  Run = ModuleHasARC(M);
  if (!Run)
    return false;

  // These are initialized lazily.
  StoreStrongCallee = 0;
  RetainAutoreleaseCallee = 0;
  RetainAutoreleaseRVCallee = 0;

  // Initialize RetainRVMarker.
  RetainRVMarker = 0;
  if (NamedMDNode *NMD =
        M.getNamedMetadata("clang.arc.retainAutoreleasedReturnValueMarker"))
    if (NMD->getNumOperands() == 1) {
      const MDNode *N = NMD->getOperand(0);
      if (N->getNumOperands() == 1)
        if (const MDString *S = dyn_cast<MDString>(N->getOperand(0)))
          RetainRVMarker = S;
    }

  return false;
}

bool ObjCARCContract::runOnFunction(Function &F) {
  if (!EnableARCOpts)
    return false;

  // If nothing in the Module uses ARC, don't do anything.
  if (!Run)
    return false;

  Changed = false;
  AA = &getAnalysis<AliasAnalysis>();
  DT = &getAnalysis<DominatorTree>();

  PA.setAA(&getAnalysis<AliasAnalysis>());

  // Track whether it's ok to mark objc_storeStrong calls with the "tail"
  // keyword. Be conservative if the function has variadic arguments.
  // It seems that functions which "return twice" are also unsafe for the
  // "tail" argument, because they are setjmp, which could need to
  // return to an earlier stack state.
  bool TailOkForStoreStrongs = !F.isVarArg() &&
                               !F.callsFunctionThatReturnsTwice();

  // For ObjC library calls which return their argument, replace uses of the
  // argument with uses of the call return value, if it dominates the use. This
  // reduces register pressure.
  SmallPtrSet<Instruction *, 4> DependingInstructions;
  SmallPtrSet<const BasicBlock *, 4> Visited;
  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ) {
    Instruction *Inst = &*I++;

    // Only these library routines return their argument. In particular,
    // objc_retainBlock does not necessarily return its argument.
    InstructionClass Class = GetBasicInstructionClass(Inst);
    switch (Class) {
    case IC_Retain:
    case IC_FusedRetainAutorelease:
    case IC_FusedRetainAutoreleaseRV:
      break;
    case IC_Autorelease:
    case IC_AutoreleaseRV:
      if (ContractAutorelease(F, Inst, Class, DependingInstructions, Visited))
        continue;
      break;
    case IC_RetainRV: {
      // If we're compiling for a target which needs a special inline-asm
      // marker to do the retainAutoreleasedReturnValue optimization,
      // insert it now.
      if (!RetainRVMarker)
        break;
      BasicBlock::iterator BBI = Inst;
      BasicBlock *InstParent = Inst->getParent();

      // Step up to see if the call immediately precedes the RetainRV call.
      // If it's an invoke, we have to cross a block boundary. And we have
      // to carefully dodge no-op instructions.
      do {
        if (&*BBI == InstParent->begin()) {
          BasicBlock *Pred = InstParent->getSinglePredecessor();
          if (!Pred)
            goto decline_rv_optimization;
          BBI = Pred->getTerminator();
          break;
        }
        --BBI;
      } while (isNoopInstruction(BBI));

      if (&*BBI == GetObjCArg(Inst)) {
        Changed = true;
        InlineAsm *IA =
          InlineAsm::get(FunctionType::get(Type::getVoidTy(Inst->getContext()),
                                           /*isVarArg=*/false),
                         RetainRVMarker->getString(),
                         /*Constraints=*/"", /*hasSideEffects=*/true);
        CallInst::Create(IA, "", Inst);
      }
    decline_rv_optimization:
      break;
    }
    case IC_InitWeak: {
      // objc_initWeak(p, null) => *p = null
      CallInst *CI = cast<CallInst>(Inst);
      if (isNullOrUndef(CI->getArgOperand(1))) {
        Value *Null =
          ConstantPointerNull::get(cast<PointerType>(CI->getType()));
        Changed = true;
        new StoreInst(Null, CI->getArgOperand(0), CI);
        CI->replaceAllUsesWith(Null);
        CI->eraseFromParent();
      }
      continue;
    }
    case IC_Release:
      ContractRelease(Inst, I);
      continue;
    case IC_User:
      // Be conservative if the function has any alloca instructions.
      // Technically we only care about escaping alloca instructions,
      // but this is sufficient to handle some interesting cases.
      if (isa<AllocaInst>(Inst))
        TailOkForStoreStrongs = false;
      continue;
    default:
      continue;
    }

    // Don't use GetObjCArg because we don't want to look through bitcasts
    // and such; to do the replacement, the argument must have type i8*.
    const Value *Arg = cast<CallInst>(Inst)->getArgOperand(0);
    for (;;) {
      // If we're compiling bugpointed code, don't get in trouble.
      if (!isa<Instruction>(Arg) && !isa<Argument>(Arg))
        break;
      // Look through the uses of the pointer.
      for (Value::const_use_iterator UI = Arg->use_begin(), UE = Arg->use_end();
           UI != UE; ) {
        Use &U = UI.getUse();
        unsigned OperandNo = UI.getOperandNo();
        ++UI; // Increment UI now, because we may unlink its element.

        // If the call's return value dominates a use of the call's argument
        // value, rewrite the use to use the return value. We check for
        // reachability here because an unreachable call is considered to
        // trivially dominate itself, which would lead us to rewriting its
        // argument in terms of its return value, which would lead to
        // infinite loops in GetObjCArg.
        if (DT->isReachableFromEntry(U) && DT->dominates(Inst, U)) {
          Changed = true;
          Instruction *Replacement = Inst;
          Type *UseTy = U.get()->getType();
          if (PHINode *PHI = dyn_cast<PHINode>(U.getUser())) {
            // For PHI nodes, insert the bitcast in the predecessor block.
            unsigned ValNo = PHINode::getIncomingValueNumForOperand(OperandNo);
            BasicBlock *BB = PHI->getIncomingBlock(ValNo);
            if (Replacement->getType() != UseTy)
              Replacement = new BitCastInst(Replacement, UseTy, "",
                                            &BB->back());
            // While we're here, rewrite all edges for this PHI, rather
            // than just one use at a time, to minimize the number of
            // bitcasts we emit.
            for (unsigned i = 0, e = PHI->getNumIncomingValues(); i != e; ++i)
              if (PHI->getIncomingBlock(i) == BB) {
                // Keep the UI iterator valid.
                if (&PHI->getOperandUse(
                      PHINode::getOperandNumForIncomingValue(i)) ==
                    &UI.getUse())
                  ++UI;
                PHI->setIncomingValue(i, Replacement);
              }
          } else {
            if (Replacement->getType() != UseTy)
              Replacement = new BitCastInst(Replacement, UseTy, "",
                                            cast<Instruction>(U.getUser()));
            U.set(Replacement);
          }
        }
      }

      // If Arg is a no-op casted pointer, strip one level of casts and iterate.
      if (const BitCastInst *BI = dyn_cast<BitCastInst>(Arg))
        Arg = BI->getOperand(0);
      else if (isa<GEPOperator>(Arg) &&
               cast<GEPOperator>(Arg)->hasAllZeroIndices())
        Arg = cast<GEPOperator>(Arg)->getPointerOperand();
      else if (isa<GlobalAlias>(Arg) &&
               !cast<GlobalAlias>(Arg)->mayBeOverridden())
        Arg = cast<GlobalAlias>(Arg)->getAliasee();
      else
        break;
    }
  }

  // If this function has no escaping allocas or suspicious vararg usage,
  // objc_storeStrong calls can be marked with the "tail" keyword.
  if (TailOkForStoreStrongs)
    for (SmallPtrSet<CallInst *, 8>::iterator I = StoreStrongCalls.begin(),
         E = StoreStrongCalls.end(); I != E; ++I)
      (*I)->setTailCall();
  StoreStrongCalls.clear();

  return Changed;
}
