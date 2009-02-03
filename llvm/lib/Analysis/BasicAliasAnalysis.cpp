//===- BasicAliasAnalysis.cpp - Local Alias Analysis Impl -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the default implementation of the Alias Analysis interface
// that simply implements a few identities (two different globals cannot alias,
// etc), but otherwise does no analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/ManagedStatic.h"
#include <algorithm>
using namespace llvm;

//===----------------------------------------------------------------------===//
// Useful predicates
//===----------------------------------------------------------------------===//

static const User *isGEP(const Value *V) {
  if (isa<GetElementPtrInst>(V) ||
      (isa<ConstantExpr>(V) &&
       cast<ConstantExpr>(V)->getOpcode() == Instruction::GetElementPtr))
    return cast<User>(V);
  return 0;
}

static const Value *GetGEPOperands(const Value *V, 
                                   SmallVector<Value*, 16> &GEPOps) {
  assert(GEPOps.empty() && "Expect empty list to populate!");
  GEPOps.insert(GEPOps.end(), cast<User>(V)->op_begin()+1,
                cast<User>(V)->op_end());

  // Accumulate all of the chained indexes into the operand array
  V = cast<User>(V)->getOperand(0);

  while (const User *G = isGEP(V)) {
    if (!isa<Constant>(GEPOps[0]) || isa<GlobalValue>(GEPOps[0]) ||
        !cast<Constant>(GEPOps[0])->isNullValue())
      break;  // Don't handle folding arbitrary pointer offsets yet...
    GEPOps.erase(GEPOps.begin());   // Drop the zero index
    GEPOps.insert(GEPOps.begin(), G->op_begin()+1, G->op_end());
    V = G->getOperand(0);
  }
  return V;
}

/// isKnownNonNull - Return true if we know that the specified value is never
/// null.
static bool isKnownNonNull(const Value *V) {
  // Alloca never returns null, malloc might.
  if (isa<AllocaInst>(V)) return true;
  
  // A byval argument is never null.
  if (const Argument *A = dyn_cast<Argument>(V))
    return A->hasByValAttr();

  // Global values are not null unless extern weak.
  if (const GlobalValue *GV = dyn_cast<GlobalValue>(V))
    return !GV->hasExternalWeakLinkage();
  return false;
}

/// isNonEscapingLocalObject - Return true if the pointer is to a function-local
/// object that never escapes from the function.
static bool isNonEscapingLocalObject(const Value *V) {
  // If this is a local allocation, check to see if it escapes.
  if (isa<AllocationInst>(V) || isNoAliasCall(V))
    return !PointerMayBeCaptured(V, false);

  // If this is an argument that corresponds to a byval or noalias argument,
  // then it has not escaped before entering the function.  Check if it escapes
  // inside the function.
  if (const Argument *A = dyn_cast<Argument>(V))
    if (A->hasByValAttr() || A->hasNoAliasAttr()) {
      // Don't bother analyzing arguments already known not to escape.
      if (A->hasNoCaptureAttr())
        return true;
      return !PointerMayBeCaptured(V, false);
    }
  return false;
}


/// isObjectSmallerThan - Return true if we can prove that the object specified
/// by V is smaller than Size.
static bool isObjectSmallerThan(const Value *V, unsigned Size,
                                const TargetData &TD) {
  const Type *AccessTy;
  if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
    AccessTy = GV->getType()->getElementType();
  } else if (const AllocationInst *AI = dyn_cast<AllocationInst>(V)) {
    if (!AI->isArrayAllocation())
      AccessTy = AI->getType()->getElementType();
    else
      return false;
  } else if (const Argument *A = dyn_cast<Argument>(V)) {
    if (A->hasByValAttr())
      AccessTy = cast<PointerType>(A->getType())->getElementType();
    else
      return false;
  } else {
    return false;
  }
  
  if (AccessTy->isSized())
    return TD.getTypePaddedSize(AccessTy) < Size;
  return false;
}

//===----------------------------------------------------------------------===//
// NoAA Pass
//===----------------------------------------------------------------------===//

namespace {
  /// NoAA - This class implements the -no-aa pass, which always returns "I
  /// don't know" for alias queries.  NoAA is unlike other alias analysis
  /// implementations, in that it does not chain to a previous analysis.  As
  /// such it doesn't follow many of the rules that other alias analyses must.
  ///
  struct VISIBILITY_HIDDEN NoAA : public ImmutablePass, public AliasAnalysis {
    static char ID; // Class identification, replacement for typeinfo
    NoAA() : ImmutablePass(&ID) {}
    explicit NoAA(void *PID) : ImmutablePass(PID) { }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<TargetData>();
    }

    virtual void initializePass() {
      TD = &getAnalysis<TargetData>();
    }

    virtual AliasResult alias(const Value *V1, unsigned V1Size,
                              const Value *V2, unsigned V2Size) {
      return MayAlias;
    }

    virtual ModRefBehavior getModRefBehavior(Function *F, CallSite CS,
                                         std::vector<PointerAccessInfo> *Info) {
      return UnknownModRefBehavior;
    }

    virtual void getArgumentAccesses(Function *F, CallSite CS,
                                     std::vector<PointerAccessInfo> &Info) {
      assert(0 && "This method may not be called on this function!");
    }

    virtual void getMustAliases(Value *P, std::vector<Value*> &RetVals) { }
    virtual bool pointsToConstantMemory(const Value *P) { return false; }
    virtual ModRefResult getModRefInfo(CallSite CS, Value *P, unsigned Size) {
      return ModRef;
    }
    virtual ModRefResult getModRefInfo(CallSite CS1, CallSite CS2) {
      return ModRef;
    }
    virtual bool hasNoModRefInfoForCalls() const { return true; }

    virtual void deleteValue(Value *V) {}
    virtual void copyValue(Value *From, Value *To) {}
  };
}  // End of anonymous namespace

// Register this pass...
char NoAA::ID = 0;
static RegisterPass<NoAA>
U("no-aa", "No Alias Analysis (always returns 'may' alias)", true, true);

// Declare that we implement the AliasAnalysis interface
static RegisterAnalysisGroup<AliasAnalysis> V(U);

ImmutablePass *llvm::createNoAAPass() { return new NoAA(); }

//===----------------------------------------------------------------------===//
// BasicAA Pass
//===----------------------------------------------------------------------===//

namespace {
  /// BasicAliasAnalysis - This is the default alias analysis implementation.
  /// Because it doesn't chain to a previous alias analysis (like -no-aa), it
  /// derives from the NoAA class.
  struct VISIBILITY_HIDDEN BasicAliasAnalysis : public NoAA {
    static char ID; // Class identification, replacement for typeinfo
    BasicAliasAnalysis() : NoAA(&ID) {}
    AliasResult alias(const Value *V1, unsigned V1Size,
                      const Value *V2, unsigned V2Size);

    ModRefResult getModRefInfo(CallSite CS, Value *P, unsigned Size);
    ModRefResult getModRefInfo(CallSite CS1, CallSite CS2);
    
    /// hasNoModRefInfoForCalls - We can provide mod/ref information against
    /// non-escaping allocations.
    virtual bool hasNoModRefInfoForCalls() const { return false; }

    /// pointsToConstantMemory - Chase pointers until we find a (constant
    /// global) or not.
    bool pointsToConstantMemory(const Value *P);

  private:
    // CheckGEPInstructions - Check two GEP instructions with known
    // must-aliasing base pointers.  This checks to see if the index expressions
    // preclude the pointers from aliasing...
    AliasResult
    CheckGEPInstructions(const Type* BasePtr1Ty,
                         Value **GEP1Ops, unsigned NumGEP1Ops, unsigned G1Size,
                         const Type *BasePtr2Ty,
                         Value **GEP2Ops, unsigned NumGEP2Ops, unsigned G2Size);
  };
}  // End of anonymous namespace

// Register this pass...
char BasicAliasAnalysis::ID = 0;
static RegisterPass<BasicAliasAnalysis>
X("basicaa", "Basic Alias Analysis (default AA impl)", false, true);

// Declare that we implement the AliasAnalysis interface
static RegisterAnalysisGroup<AliasAnalysis, true> Y(X);

ImmutablePass *llvm::createBasicAliasAnalysisPass() {
  return new BasicAliasAnalysis();
}


/// pointsToConstantMemory - Chase pointers until we find a (constant
/// global) or not.
bool BasicAliasAnalysis::pointsToConstantMemory(const Value *P) {
  if (const GlobalVariable *GV = 
        dyn_cast<GlobalVariable>(P->getUnderlyingObject()))
    return GV->isConstant();
  return false;
}

// getModRefInfo - Check to see if the specified callsite can clobber the
// specified memory object.  Since we only look at local properties of this
// function, we really can't say much about this query.  We do, however, use
// simple "address taken" analysis on local objects.
//
AliasAnalysis::ModRefResult
BasicAliasAnalysis::getModRefInfo(CallSite CS, Value *P, unsigned Size) {
  if (!isa<Constant>(P)) {
    const Value *Object = P->getUnderlyingObject();
    
    // If this is a tail call and P points to a stack location, we know that
    // the tail call cannot access or modify the local stack.
    // We cannot exclude byval arguments here; these belong to the caller of
    // the current function not to the current function, and a tail callee
    // may reference them.
    if (isa<AllocaInst>(Object))
      if (CallInst *CI = dyn_cast<CallInst>(CS.getInstruction()))
        if (CI->isTailCall())
          return NoModRef;
    
    // If the pointer is to a locally allocated object that does not escape,
    // then the call can not mod/ref the pointer unless the call takes the
    // argument without capturing it.
    if (isNonEscapingLocalObject(Object)) {
      bool passedAsArg = false;
      // TODO: Eventually only check 'nocapture' arguments.
      for (CallSite::arg_iterator CI = CS.arg_begin(), CE = CS.arg_end();
           CI != CE; ++CI)
        if (isa<PointerType>((*CI)->getType()) &&
            alias(cast<Value>(CI), ~0U, P, ~0U) != NoAlias)
          passedAsArg = true;
      
      if (!passedAsArg)
        return NoModRef;
    }
  }

  // The AliasAnalysis base class has some smarts, lets use them.
  return AliasAnalysis::getModRefInfo(CS, P, Size);
}


AliasAnalysis::ModRefResult 
BasicAliasAnalysis::getModRefInfo(CallSite CS1, CallSite CS2) {
  // If CS1 or CS2 are readnone, they don't interact.
  ModRefBehavior CS1B = AliasAnalysis::getModRefBehavior(CS1);
  if (CS1B == DoesNotAccessMemory) return NoModRef;
  
  ModRefBehavior CS2B = AliasAnalysis::getModRefBehavior(CS2);
  if (CS2B == DoesNotAccessMemory) return NoModRef;
  
  // If they both only read from memory, just return ref.
  if (CS1B == OnlyReadsMemory && CS2B == OnlyReadsMemory)
    return Ref;
  
  // Otherwise, fall back to NoAA (mod+ref).
  return NoAA::getModRefInfo(CS1, CS2);
}


// alias - Provide a bunch of ad-hoc rules to disambiguate in common cases, such
// as array references.
//
AliasAnalysis::AliasResult
BasicAliasAnalysis::alias(const Value *V1, unsigned V1Size,
                          const Value *V2, unsigned V2Size) {
  // Strip off any constant expression casts if they exist
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(V1))
    if (CE->isCast() && isa<PointerType>(CE->getOperand(0)->getType()))
      V1 = CE->getOperand(0);
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(V2))
    if (CE->isCast() && isa<PointerType>(CE->getOperand(0)->getType()))
      V2 = CE->getOperand(0);

  // Are we checking for alias of the same value?
  if (V1 == V2) return MustAlias;

  if (!isa<PointerType>(V1->getType()) || !isa<PointerType>(V2->getType()))
    return NoAlias;  // Scalars cannot alias each other

  // Strip off cast instructions.   Since V1 and V2 are pointers, they must be
  // pointer<->pointer bitcasts.
  if (const BitCastInst *I = dyn_cast<BitCastInst>(V1))
    return alias(I->getOperand(0), V1Size, V2, V2Size);
  if (const BitCastInst *I = dyn_cast<BitCastInst>(V2))
    return alias(V1, V1Size, I->getOperand(0), V2Size);

  // Figure out what objects these things are pointing to if we can.
  const Value *O1 = V1->getUnderlyingObject();
  const Value *O2 = V2->getUnderlyingObject();

  if (O1 != O2) {
    // If V1/V2 point to two different objects we know that we have no alias.
    if (isIdentifiedObject(O1) && isIdentifiedObject(O2))
      return NoAlias;
  
    // Arguments can't alias with local allocations or noalias calls.
    if ((isa<Argument>(O1) && (isa<AllocationInst>(O2) || isNoAliasCall(O2))) ||
        (isa<Argument>(O2) && (isa<AllocationInst>(O1) || isNoAliasCall(O1))))
      return NoAlias;

    // Most objects can't alias null.
    if ((isa<ConstantPointerNull>(V2) && isKnownNonNull(O1)) ||
        (isa<ConstantPointerNull>(V1) && isKnownNonNull(O2)))
      return NoAlias;
  }
  
  // If the size of one access is larger than the entire object on the other
  // side, then we know such behavior is undefined and can assume no alias.
  const TargetData &TD = getTargetData();
  if ((V1Size != ~0U && isObjectSmallerThan(O2, V1Size, TD)) ||
      (V2Size != ~0U && isObjectSmallerThan(O1, V2Size, TD)))
    return NoAlias;
  
  // If one pointer is the result of a call/invoke and the other is a
  // non-escaping local object, then we know the object couldn't escape to a
  // point where the call could return it.
  if ((isa<CallInst>(O1) || isa<InvokeInst>(O1)) &&
      isNonEscapingLocalObject(O2))
    return NoAlias;
  if ((isa<CallInst>(O2) || isa<InvokeInst>(O2)) &&
      isNonEscapingLocalObject(O1))
    return NoAlias;
  
  // If we have two gep instructions with must-alias'ing base pointers, figure
  // out if the indexes to the GEP tell us anything about the derived pointer.
  // Note that we also handle chains of getelementptr instructions as well as
  // constant expression getelementptrs here.
  //
  if (isGEP(V1) && isGEP(V2)) {
    const User *GEP1 = cast<User>(V1);
    const User *GEP2 = cast<User>(V2);
    
    // If V1 and V2 are identical GEPs, just recurse down on both of them.
    // This allows us to analyze things like:
    //   P = gep A, 0, i, 1
    //   Q = gep B, 0, i, 1
    // by just analyzing A and B.  This is even safe for variable indices.
    if (GEP1->getType() == GEP2->getType() &&
        GEP1->getNumOperands() == GEP2->getNumOperands() &&
        GEP1->getOperand(0)->getType() == GEP2->getOperand(0)->getType() &&
        // All operands are the same, ignoring the base.
        std::equal(GEP1->op_begin()+1, GEP1->op_end(), GEP2->op_begin()+1))
      return alias(GEP1->getOperand(0), V1Size, GEP2->getOperand(0), V2Size);
    
    
    // Drill down into the first non-gep value, to test for must-aliasing of
    // the base pointers.
    while (isGEP(GEP1->getOperand(0)) &&
           GEP1->getOperand(1) ==
           Constant::getNullValue(GEP1->getOperand(1)->getType()))
      GEP1 = cast<User>(GEP1->getOperand(0));
    const Value *BasePtr1 = GEP1->getOperand(0);

    while (isGEP(GEP2->getOperand(0)) &&
           GEP2->getOperand(1) ==
           Constant::getNullValue(GEP2->getOperand(1)->getType()))
      GEP2 = cast<User>(GEP2->getOperand(0));
    const Value *BasePtr2 = GEP2->getOperand(0);

    // Do the base pointers alias?
    AliasResult BaseAlias = alias(BasePtr1, ~0U, BasePtr2, ~0U);
    if (BaseAlias == NoAlias) return NoAlias;
    if (BaseAlias == MustAlias) {
      // If the base pointers alias each other exactly, check to see if we can
      // figure out anything about the resultant pointers, to try to prove
      // non-aliasing.

      // Collect all of the chained GEP operands together into one simple place
      SmallVector<Value*, 16> GEP1Ops, GEP2Ops;
      BasePtr1 = GetGEPOperands(V1, GEP1Ops);
      BasePtr2 = GetGEPOperands(V2, GEP2Ops);

      // If GetGEPOperands were able to fold to the same must-aliased pointer,
      // do the comparison.
      if (BasePtr1 == BasePtr2) {
        AliasResult GAlias =
          CheckGEPInstructions(BasePtr1->getType(),
                               &GEP1Ops[0], GEP1Ops.size(), V1Size,
                               BasePtr2->getType(),
                               &GEP2Ops[0], GEP2Ops.size(), V2Size);
        if (GAlias != MayAlias)
          return GAlias;
      }
    }
  }

  // Check to see if these two pointers are related by a getelementptr
  // instruction.  If one pointer is a GEP with a non-zero index of the other
  // pointer, we know they cannot alias.
  //
  if (isGEP(V2)) {
    std::swap(V1, V2);
    std::swap(V1Size, V2Size);
  }

  if (V1Size != ~0U && V2Size != ~0U)
    if (isGEP(V1)) {
      SmallVector<Value*, 16> GEPOperands;
      const Value *BasePtr = GetGEPOperands(V1, GEPOperands);

      AliasResult R = alias(BasePtr, V1Size, V2, V2Size);
      if (R == MustAlias) {
        // If there is at least one non-zero constant index, we know they cannot
        // alias.
        bool ConstantFound = false;
        bool AllZerosFound = true;
        for (unsigned i = 0, e = GEPOperands.size(); i != e; ++i)
          if (const Constant *C = dyn_cast<Constant>(GEPOperands[i])) {
            if (!C->isNullValue()) {
              ConstantFound = true;
              AllZerosFound = false;
              break;
            }
          } else {
            AllZerosFound = false;
          }

        // If we have getelementptr <ptr>, 0, 0, 0, 0, ... and V2 must aliases
        // the ptr, the end result is a must alias also.
        if (AllZerosFound)
          return MustAlias;

        if (ConstantFound) {
          if (V2Size <= 1 && V1Size <= 1)  // Just pointer check?
            return NoAlias;

          // Otherwise we have to check to see that the distance is more than
          // the size of the argument... build an index vector that is equal to
          // the arguments provided, except substitute 0's for any variable
          // indexes we find...
          if (cast<PointerType>(
                BasePtr->getType())->getElementType()->isSized()) {
            for (unsigned i = 0; i != GEPOperands.size(); ++i)
              if (!isa<ConstantInt>(GEPOperands[i]))
                GEPOperands[i] =
                  Constant::getNullValue(GEPOperands[i]->getType());
            int64_t Offset =
              getTargetData().getIndexedOffset(BasePtr->getType(),
                                               &GEPOperands[0],
                                               GEPOperands.size());

            if (Offset >= (int64_t)V2Size || Offset <= -(int64_t)V1Size)
              return NoAlias;
          }
        }
      }
    }

  return MayAlias;
}

// This function is used to determine if the indices of two GEP instructions are
// equal. V1 and V2 are the indices.
static bool IndexOperandsEqual(Value *V1, Value *V2) {
  if (V1->getType() == V2->getType())
    return V1 == V2;
  if (Constant *C1 = dyn_cast<Constant>(V1))
    if (Constant *C2 = dyn_cast<Constant>(V2)) {
      // Sign extend the constants to long types, if necessary
      if (C1->getType() != Type::Int64Ty)
        C1 = ConstantExpr::getSExt(C1, Type::Int64Ty);
      if (C2->getType() != Type::Int64Ty) 
        C2 = ConstantExpr::getSExt(C2, Type::Int64Ty);
      return C1 == C2;
    }
  return false;
}

/// CheckGEPInstructions - Check two GEP instructions with known must-aliasing
/// base pointers.  This checks to see if the index expressions preclude the
/// pointers from aliasing...
AliasAnalysis::AliasResult 
BasicAliasAnalysis::CheckGEPInstructions(
  const Type* BasePtr1Ty, Value **GEP1Ops, unsigned NumGEP1Ops, unsigned G1S,
  const Type *BasePtr2Ty, Value **GEP2Ops, unsigned NumGEP2Ops, unsigned G2S) {
  // We currently can't handle the case when the base pointers have different
  // primitive types.  Since this is uncommon anyway, we are happy being
  // extremely conservative.
  if (BasePtr1Ty != BasePtr2Ty)
    return MayAlias;

  const PointerType *GEPPointerTy = cast<PointerType>(BasePtr1Ty);

  // Find the (possibly empty) initial sequence of equal values... which are not
  // necessarily constants.
  unsigned NumGEP1Operands = NumGEP1Ops, NumGEP2Operands = NumGEP2Ops;
  unsigned MinOperands = std::min(NumGEP1Operands, NumGEP2Operands);
  unsigned MaxOperands = std::max(NumGEP1Operands, NumGEP2Operands);
  unsigned UnequalOper = 0;
  while (UnequalOper != MinOperands &&
         IndexOperandsEqual(GEP1Ops[UnequalOper], GEP2Ops[UnequalOper])) {
    // Advance through the type as we go...
    ++UnequalOper;
    if (const CompositeType *CT = dyn_cast<CompositeType>(BasePtr1Ty))
      BasePtr1Ty = CT->getTypeAtIndex(GEP1Ops[UnequalOper-1]);
    else {
      // If all operands equal each other, then the derived pointers must
      // alias each other...
      BasePtr1Ty = 0;
      assert(UnequalOper == NumGEP1Operands && UnequalOper == NumGEP2Operands &&
             "Ran out of type nesting, but not out of operands?");
      return MustAlias;
    }
  }

  // If we have seen all constant operands, and run out of indexes on one of the
  // getelementptrs, check to see if the tail of the leftover one is all zeros.
  // If so, return mustalias.
  if (UnequalOper == MinOperands) {
    if (NumGEP1Ops < NumGEP2Ops) {
      std::swap(GEP1Ops, GEP2Ops);
      std::swap(NumGEP1Ops, NumGEP2Ops);
    }

    bool AllAreZeros = true;
    for (unsigned i = UnequalOper; i != MaxOperands; ++i)
      if (!isa<Constant>(GEP1Ops[i]) ||
          !cast<Constant>(GEP1Ops[i])->isNullValue()) {
        AllAreZeros = false;
        break;
      }
    if (AllAreZeros) return MustAlias;
  }


  // So now we know that the indexes derived from the base pointers,
  // which are known to alias, are different.  We can still determine a
  // no-alias result if there are differing constant pairs in the index
  // chain.  For example:
  //        A[i][0] != A[j][1] iff (&A[0][1]-&A[0][0] >= std::max(G1S, G2S))
  //
  // We have to be careful here about array accesses.  In particular, consider:
  //        A[1][0] vs A[0][i]
  // In this case, we don't *know* that the array will be accessed in bounds:
  // the index could even be negative.  Because of this, we have to
  // conservatively *give up* and return may alias.  We disregard differing
  // array subscripts that are followed by a variable index without going
  // through a struct.
  //
  unsigned SizeMax = std::max(G1S, G2S);
  if (SizeMax == ~0U) return MayAlias; // Avoid frivolous work.

  // Scan for the first operand that is constant and unequal in the
  // two getelementptrs...
  unsigned FirstConstantOper = UnequalOper;
  for (; FirstConstantOper != MinOperands; ++FirstConstantOper) {
    const Value *G1Oper = GEP1Ops[FirstConstantOper];
    const Value *G2Oper = GEP2Ops[FirstConstantOper];

    if (G1Oper != G2Oper)   // Found non-equal constant indexes...
      if (Constant *G1OC = dyn_cast<ConstantInt>(const_cast<Value*>(G1Oper)))
        if (Constant *G2OC = dyn_cast<ConstantInt>(const_cast<Value*>(G2Oper))){
          if (G1OC->getType() != G2OC->getType()) {
            // Sign extend both operands to long.
            if (G1OC->getType() != Type::Int64Ty)
              G1OC = ConstantExpr::getSExt(G1OC, Type::Int64Ty);
            if (G2OC->getType() != Type::Int64Ty) 
              G2OC = ConstantExpr::getSExt(G2OC, Type::Int64Ty);
            GEP1Ops[FirstConstantOper] = G1OC;
            GEP2Ops[FirstConstantOper] = G2OC;
          }
          
          if (G1OC != G2OC) {
            // Handle the "be careful" case above: if this is an array/vector
            // subscript, scan for a subsequent variable array index.
            if (isa<SequentialType>(BasePtr1Ty))  {
              const Type *NextTy =
                cast<SequentialType>(BasePtr1Ty)->getElementType();
              bool isBadCase = false;
              
              for (unsigned Idx = FirstConstantOper+1;
                   Idx != MinOperands && isa<SequentialType>(NextTy); ++Idx) {
                const Value *V1 = GEP1Ops[Idx], *V2 = GEP2Ops[Idx];
                if (!isa<Constant>(V1) || !isa<Constant>(V2)) {
                  isBadCase = true;
                  break;
                }
                NextTy = cast<SequentialType>(NextTy)->getElementType();
              }
              
              if (isBadCase) G1OC = 0;
            }

            // Make sure they are comparable (ie, not constant expressions), and
            // make sure the GEP with the smaller leading constant is GEP1.
            if (G1OC) {
              Constant *Compare = ConstantExpr::getICmp(ICmpInst::ICMP_SGT, 
                                                        G1OC, G2OC);
              if (ConstantInt *CV = dyn_cast<ConstantInt>(Compare)) {
                if (CV->getZExtValue()) {  // If they are comparable and G2 > G1
                  std::swap(GEP1Ops, GEP2Ops);  // Make GEP1 < GEP2
                  std::swap(NumGEP1Ops, NumGEP2Ops);
                }
                break;
              }
            }
          }
        }
    BasePtr1Ty = cast<CompositeType>(BasePtr1Ty)->getTypeAtIndex(G1Oper);
  }

  // No shared constant operands, and we ran out of common operands.  At this
  // point, the GEP instructions have run through all of their operands, and we
  // haven't found evidence that there are any deltas between the GEP's.
  // However, one GEP may have more operands than the other.  If this is the
  // case, there may still be hope.  Check this now.
  if (FirstConstantOper == MinOperands) {
    // Make GEP1Ops be the longer one if there is a longer one.
    if (NumGEP1Ops < NumGEP2Ops) {
      std::swap(GEP1Ops, GEP2Ops);
      std::swap(NumGEP1Ops, NumGEP2Ops);
    }

    // Is there anything to check?
    if (NumGEP1Ops > MinOperands) {
      for (unsigned i = FirstConstantOper; i != MaxOperands; ++i)
        if (isa<ConstantInt>(GEP1Ops[i]) && 
            !cast<ConstantInt>(GEP1Ops[i])->isZero()) {
          // Yup, there's a constant in the tail.  Set all variables to
          // constants in the GEP instruction to make it suitable for
          // TargetData::getIndexedOffset.
          for (i = 0; i != MaxOperands; ++i)
            if (!isa<ConstantInt>(GEP1Ops[i]))
              GEP1Ops[i] = Constant::getNullValue(GEP1Ops[i]->getType());
          // Okay, now get the offset.  This is the relative offset for the full
          // instruction.
          const TargetData &TD = getTargetData();
          int64_t Offset1 = TD.getIndexedOffset(GEPPointerTy, GEP1Ops,
                                                NumGEP1Ops);

          // Now check without any constants at the end.
          int64_t Offset2 = TD.getIndexedOffset(GEPPointerTy, GEP1Ops,
                                                MinOperands);

          // Make sure we compare the absolute difference.
          if (Offset1 > Offset2)
            std::swap(Offset1, Offset2);

          // If the tail provided a bit enough offset, return noalias!
          if ((uint64_t)(Offset2-Offset1) >= SizeMax)
            return NoAlias;
          // Otherwise break - we don't look for another constant in the tail.
          break;
        }
    }

    // Couldn't find anything useful.
    return MayAlias;
  }

  // If there are non-equal constants arguments, then we can figure
  // out a minimum known delta between the two index expressions... at
  // this point we know that the first constant index of GEP1 is less
  // than the first constant index of GEP2.

  // Advance BasePtr[12]Ty over this first differing constant operand.
  BasePtr2Ty = cast<CompositeType>(BasePtr1Ty)->
      getTypeAtIndex(GEP2Ops[FirstConstantOper]);
  BasePtr1Ty = cast<CompositeType>(BasePtr1Ty)->
      getTypeAtIndex(GEP1Ops[FirstConstantOper]);

  // We are going to be using TargetData::getIndexedOffset to determine the
  // offset that each of the GEP's is reaching.  To do this, we have to convert
  // all variable references to constant references.  To do this, we convert the
  // initial sequence of array subscripts into constant zeros to start with.
  const Type *ZeroIdxTy = GEPPointerTy;
  for (unsigned i = 0; i != FirstConstantOper; ++i) {
    if (!isa<StructType>(ZeroIdxTy))
      GEP1Ops[i] = GEP2Ops[i] = Constant::getNullValue(Type::Int32Ty);

    if (const CompositeType *CT = dyn_cast<CompositeType>(ZeroIdxTy))
      ZeroIdxTy = CT->getTypeAtIndex(GEP1Ops[i]);
  }

  // We know that GEP1Ops[FirstConstantOper] & GEP2Ops[FirstConstantOper] are ok

  // Loop over the rest of the operands...
  for (unsigned i = FirstConstantOper+1; i != MaxOperands; ++i) {
    const Value *Op1 = i < NumGEP1Ops ? GEP1Ops[i] : 0;
    const Value *Op2 = i < NumGEP2Ops ? GEP2Ops[i] : 0;
    // If they are equal, use a zero index...
    if (Op1 == Op2 && BasePtr1Ty == BasePtr2Ty) {
      if (!isa<ConstantInt>(Op1))
        GEP1Ops[i] = GEP2Ops[i] = Constant::getNullValue(Op1->getType());
      // Otherwise, just keep the constants we have.
    } else {
      if (Op1) {
        if (const ConstantInt *Op1C = dyn_cast<ConstantInt>(Op1)) {
          // If this is an array index, make sure the array element is in range.
          if (const ArrayType *AT = dyn_cast<ArrayType>(BasePtr1Ty)) {
            if (Op1C->getZExtValue() >= AT->getNumElements())
              return MayAlias;  // Be conservative with out-of-range accesses
          } else if (const VectorType *VT = dyn_cast<VectorType>(BasePtr1Ty)) {
            if (Op1C->getZExtValue() >= VT->getNumElements())
              return MayAlias;  // Be conservative with out-of-range accesses
          }
          
        } else {
          // GEP1 is known to produce a value less than GEP2.  To be
          // conservatively correct, we must assume the largest possible
          // constant is used in this position.  This cannot be the initial
          // index to the GEP instructions (because we know we have at least one
          // element before this one with the different constant arguments), so
          // we know that the current index must be into either a struct or
          // array.  Because we know it's not constant, this cannot be a
          // structure index.  Because of this, we can calculate the maximum
          // value possible.
          //
          if (const ArrayType *AT = dyn_cast<ArrayType>(BasePtr1Ty))
            GEP1Ops[i] = ConstantInt::get(Type::Int64Ty,AT->getNumElements()-1);
          else if (const VectorType *VT = dyn_cast<VectorType>(BasePtr1Ty))
            GEP1Ops[i] = ConstantInt::get(Type::Int64Ty,VT->getNumElements()-1);
        }
      }

      if (Op2) {
        if (const ConstantInt *Op2C = dyn_cast<ConstantInt>(Op2)) {
          // If this is an array index, make sure the array element is in range.
          if (const ArrayType *AT = dyn_cast<ArrayType>(BasePtr2Ty)) {
            if (Op2C->getZExtValue() >= AT->getNumElements())
              return MayAlias;  // Be conservative with out-of-range accesses
          } else if (const VectorType *VT = dyn_cast<VectorType>(BasePtr2Ty)) {
            if (Op2C->getZExtValue() >= VT->getNumElements())
              return MayAlias;  // Be conservative with out-of-range accesses
          }
        } else {  // Conservatively assume the minimum value for this index
          GEP2Ops[i] = Constant::getNullValue(Op2->getType());
        }
      }
    }

    if (BasePtr1Ty && Op1) {
      if (const CompositeType *CT = dyn_cast<CompositeType>(BasePtr1Ty))
        BasePtr1Ty = CT->getTypeAtIndex(GEP1Ops[i]);
      else
        BasePtr1Ty = 0;
    }

    if (BasePtr2Ty && Op2) {
      if (const CompositeType *CT = dyn_cast<CompositeType>(BasePtr2Ty))
        BasePtr2Ty = CT->getTypeAtIndex(GEP2Ops[i]);
      else
        BasePtr2Ty = 0;
    }
  }

  if (GEPPointerTy->getElementType()->isSized()) {
    int64_t Offset1 =
      getTargetData().getIndexedOffset(GEPPointerTy, GEP1Ops, NumGEP1Ops);
    int64_t Offset2 = 
      getTargetData().getIndexedOffset(GEPPointerTy, GEP2Ops, NumGEP2Ops);
    assert(Offset1 != Offset2 &&
           "There is at least one different constant here!");
    
    // Make sure we compare the absolute difference.
    if (Offset1 > Offset2)
      std::swap(Offset1, Offset2);
    
    if ((uint64_t)(Offset2-Offset1) >= SizeMax) {
      //cerr << "Determined that these two GEP's don't alias ["
      //     << SizeMax << " bytes]: \n" << *GEP1 << *GEP2;
      return NoAlias;
    }
  }
  return MayAlias;
}

// Make sure that anything that uses AliasAnalysis pulls in this file...
DEFINING_FILE_FOR(BasicAliasAnalysis)
