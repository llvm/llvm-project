//===-- DeadArgumentElimination.cpp - Eliminate dead arguments ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass deletes dead arguments from internal functions.  Dead argument
// elimination removes arguments which are directly dead, as well as arguments
// only passed into function calls as dead arguments of other functions.  This
// pass also deletes dead return values in a similar way.
//
// This pass is often useful as a cleanup pass to run after aggressive
// interprocedural passes, which add possibly-dead arguments or return values.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "deadargelim"
#include "llvm/Transforms/IPO.h"
#include "llvm/CallingConv.h"
#include "llvm/Constant.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Compiler.h"
#include <map>
#include <set>
using namespace llvm;

STATISTIC(NumArgumentsEliminated, "Number of unread args removed");
STATISTIC(NumRetValsEliminated  , "Number of unused return values removed");

namespace {
  /// DAE - The dead argument elimination pass.
  ///
  class VISIBILITY_HIDDEN DAE : public ModulePass {
  public:

    /// Struct that represents (part of) either a return value or a function
    /// argument.  Used so that arguments and return values can be used
    /// interchangably.
    struct RetOrArg {
      RetOrArg(const Function* F, unsigned Idx, bool IsArg) : F(F), Idx(Idx),
               IsArg(IsArg) {}
      const Function *F;
      unsigned Idx;
      bool IsArg;

      /// Make RetOrArg comparable, so we can put it into a map.
      bool operator<(const RetOrArg &O) const {
        if (F != O.F)
          return F < O.F;
        else if (Idx != O.Idx)
          return Idx < O.Idx;
        else
          return IsArg < O.IsArg;
      }

      /// Make RetOrArg comparable, so we can easily iterate the multimap.
      bool operator==(const RetOrArg &O) const {
        return F == O.F && Idx == O.Idx && IsArg == O.IsArg;
      }

      std::string getDescription() const {
        return std::string((IsArg ? "Argument #" : "Return value #")) 
               + utostr(Idx) + " of function " + F->getName();
      }
    };

    /// Liveness enum - During our initial pass over the program, we determine
    /// that things are either alive or maybe alive. We don't mark anything
    /// explicitly dead (even if we know they are), since anything not alive
    /// with no registered uses (in Uses) will never be marked alive and will
    /// thus become dead in the end.
    enum Liveness { Live, MaybeLive };

    /// Convenience wrapper
    RetOrArg CreateRet(const Function *F, unsigned Idx) {
      return RetOrArg(F, Idx, false);
    }
    /// Convenience wrapper
    RetOrArg CreateArg(const Function *F, unsigned Idx) {
      return RetOrArg(F, Idx, true);
    }

    typedef std::multimap<RetOrArg, RetOrArg> UseMap;
    /// This maps a return value or argument to any MaybeLive return values or
    /// arguments it uses. This allows the MaybeLive values to be marked live
    /// when any of its users is marked live.
    /// For example (indices are left out for clarity):
    ///  - Uses[ret F] = ret G
    ///    This means that F calls G, and F returns the value returned by G.
    ///  - Uses[arg F] = ret G
    ///    This means that some function calls G and passes its result as an
    ///    argument to F.
    ///  - Uses[ret F] = arg F
    ///    This means that F returns one of its own arguments.
    ///  - Uses[arg F] = arg G
    ///    This means that G calls F and passes one of its own (G's) arguments
    ///    directly to F.
    UseMap Uses;

    typedef std::set<RetOrArg> LiveSet;
    typedef std::set<const Function*> LiveFuncSet;

    /// This set contains all values that have been determined to be live.
    LiveSet LiveValues;
    /// This set contains all values that are cannot be changed in any way.
    LiveFuncSet LiveFunctions;

    typedef SmallVector<RetOrArg, 5> UseVector;

  public:
    static char ID; // Pass identification, replacement for typeid
    DAE() : ModulePass(&ID) {}
    bool runOnModule(Module &M);

    virtual bool ShouldHackArguments() const { return false; }

  private:
    Liveness MarkIfNotLive(RetOrArg Use, UseVector &MaybeLiveUses);
    Liveness SurveyUse(Value::use_iterator U, UseVector &MaybeLiveUses,
                       unsigned RetValNum = 0);
    Liveness SurveyUses(Value *V, UseVector &MaybeLiveUses);

    void SurveyFunction(Function &F);
    void MarkValue(const RetOrArg &RA, Liveness L,
                   const UseVector &MaybeLiveUses);
    void MarkLive(const RetOrArg &RA);
    void MarkLive(const Function &F);
    void PropagateLiveness(const RetOrArg &RA);
    bool RemoveDeadStuffFromFunction(Function *F);
    bool DeleteDeadVarargs(Function &Fn);
  };
}


char DAE::ID = 0;
static RegisterPass<DAE>
X("deadargelim", "Dead Argument Elimination");

namespace {
  /// DAH - DeadArgumentHacking pass - Same as dead argument elimination, but
  /// deletes arguments to functions which are external.  This is only for use
  /// by bugpoint.
  struct DAH : public DAE {
    static char ID;
    virtual bool ShouldHackArguments() const { return true; }
  };
}

char DAH::ID = 0;
static RegisterPass<DAH>
Y("deadarghaX0r", "Dead Argument Hacking (BUGPOINT USE ONLY; DO NOT USE)");

/// createDeadArgEliminationPass - This pass removes arguments from functions
/// which are not used by the body of the function.
///
ModulePass *llvm::createDeadArgEliminationPass() { return new DAE(); }
ModulePass *llvm::createDeadArgHackingPass() { return new DAH(); }

/// DeleteDeadVarargs - If this is an function that takes a ... list, and if
/// llvm.vastart is never called, the varargs list is dead for the function.
bool DAE::DeleteDeadVarargs(Function &Fn) {
  assert(Fn.getFunctionType()->isVarArg() && "Function isn't varargs!");
  if (Fn.isDeclaration() || !Fn.hasLocalLinkage()) return false;

  // Ensure that the function is only directly called.
  for (Value::use_iterator I = Fn.use_begin(), E = Fn.use_end(); I != E; ++I) {
    // If this use is anything other than a call site, give up.
    CallSite CS = CallSite::get(*I);
    Instruction *TheCall = CS.getInstruction();
    if (!TheCall) return false;   // Not a direct call site?

    // The addr of this function is passed to the call.
    if (!CS.isCallee(I)) return false;
  }

  // Okay, we know we can transform this function if safe.  Scan its body
  // looking for calls to llvm.vastart.
  for (Function::iterator BB = Fn.begin(), E = Fn.end(); BB != E; ++BB) {
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
      if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
        if (II->getIntrinsicID() == Intrinsic::vastart)
          return false;
      }
    }
  }

  // If we get here, there are no calls to llvm.vastart in the function body,
  // remove the "..." and adjust all the calls.

  // Start by computing a new prototype for the function, which is the same as
  // the old function, but doesn't have isVarArg set.
  const FunctionType *FTy = Fn.getFunctionType();
  std::vector<const Type*> Params(FTy->param_begin(), FTy->param_end());
  FunctionType *NFTy = FunctionType::get(FTy->getReturnType(), Params, false);
  unsigned NumArgs = Params.size();

  // Create the new function body and insert it into the module...
  Function *NF = Function::Create(NFTy, Fn.getLinkage());
  NF->copyAttributesFrom(&Fn);
  Fn.getParent()->getFunctionList().insert(&Fn, NF);
  NF->takeName(&Fn);

  // Loop over all of the callers of the function, transforming the call sites
  // to pass in a smaller number of arguments into the new function.
  //
  std::vector<Value*> Args;
  while (!Fn.use_empty()) {
    CallSite CS = CallSite::get(Fn.use_back());
    Instruction *Call = CS.getInstruction();

    // Pass all the same arguments.
    Args.assign(CS.arg_begin(), CS.arg_begin()+NumArgs);

    // Drop any attributes that were on the vararg arguments.
    AttrListPtr PAL = CS.getAttributes();
    if (!PAL.isEmpty() && PAL.getSlot(PAL.getNumSlots() - 1).Index > NumArgs) {
      SmallVector<AttributeWithIndex, 8> AttributesVec;
      for (unsigned i = 0; PAL.getSlot(i).Index <= NumArgs; ++i)
        AttributesVec.push_back(PAL.getSlot(i));
      if (Attributes FnAttrs = PAL.getFnAttributes()) 
        AttributesVec.push_back(AttributeWithIndex::get(~0, FnAttrs));
      PAL = AttrListPtr::get(AttributesVec.begin(), AttributesVec.end());
    }

    Instruction *New;
    if (InvokeInst *II = dyn_cast<InvokeInst>(Call)) {
      New = InvokeInst::Create(NF, II->getNormalDest(), II->getUnwindDest(),
                               Args.begin(), Args.end(), "", Call);
      cast<InvokeInst>(New)->setCallingConv(CS.getCallingConv());
      cast<InvokeInst>(New)->setAttributes(PAL);
    } else {
      New = CallInst::Create(NF, Args.begin(), Args.end(), "", Call);
      cast<CallInst>(New)->setCallingConv(CS.getCallingConv());
      cast<CallInst>(New)->setAttributes(PAL);
      if (cast<CallInst>(Call)->isTailCall())
        cast<CallInst>(New)->setTailCall();
    }
    Args.clear();

    if (!Call->use_empty())
      Call->replaceAllUsesWith(New);

    New->takeName(Call);

    // Finally, remove the old call from the program, reducing the use-count of
    // F.
    Call->eraseFromParent();
  }

  // Since we have now created the new function, splice the body of the old
  // function right into the new function, leaving the old rotting hulk of the
  // function empty.
  NF->getBasicBlockList().splice(NF->begin(), Fn.getBasicBlockList());

  // Loop over the argument list, transfering uses of the old arguments over to
  // the new arguments, also transfering over the names as well.  While we're at
  // it, remove the dead arguments from the DeadArguments list.
  //
  for (Function::arg_iterator I = Fn.arg_begin(), E = Fn.arg_end(),
       I2 = NF->arg_begin(); I != E; ++I, ++I2) {
    // Move the name and users over to the new version.
    I->replaceAllUsesWith(I2);
    I2->takeName(I);
  }

  // Finally, nuke the old function.
  Fn.eraseFromParent();
  return true;
}

/// Convenience function that returns the number of return values. It returns 0
/// for void functions and 1 for functions not returning a struct. It returns
/// the number of struct elements for functions returning a struct.
static unsigned NumRetVals(const Function *F) {
  if (F->getReturnType() == Type::VoidTy)
    return 0;
  else if (const StructType *STy = dyn_cast<StructType>(F->getReturnType()))
    return STy->getNumElements();
  else
    return 1;
}

/// MarkIfNotLive - This checks Use for liveness in LiveValues. If Use is not
/// live, it adds Use to the MaybeLiveUses argument. Returns the determined
/// liveness of Use.
DAE::Liveness DAE::MarkIfNotLive(RetOrArg Use, UseVector &MaybeLiveUses) {
  // We're live if our use or its Function is already marked as live.
  if (LiveFunctions.count(Use.F) || LiveValues.count(Use))
    return Live;

  // We're maybe live otherwise, but remember that we must become live if
  // Use becomes live.
  MaybeLiveUses.push_back(Use);
  return MaybeLive;
}


/// SurveyUse - This looks at a single use of an argument or return value
/// and determines if it should be alive or not. Adds this use to MaybeLiveUses
/// if it causes the used value to become MaybeAlive.
///
/// RetValNum is the return value number to use when this use is used in a
/// return instruction. This is used in the recursion, you should always leave
/// it at 0.
DAE::Liveness DAE::SurveyUse(Value::use_iterator U, UseVector &MaybeLiveUses,
                             unsigned RetValNum) {
    Value *V = *U;
    if (ReturnInst *RI = dyn_cast<ReturnInst>(V)) {
      // The value is returned from a function. It's only live when the
      // function's return value is live. We use RetValNum here, for the case
      // that U is really a use of an insertvalue instruction that uses the
      // orginal Use.
      RetOrArg Use = CreateRet(RI->getParent()->getParent(), RetValNum);
      // We might be live, depending on the liveness of Use.
      return MarkIfNotLive(Use, MaybeLiveUses);
    }
    if (InsertValueInst *IV = dyn_cast<InsertValueInst>(V)) {
      if (U.getOperandNo() != InsertValueInst::getAggregateOperandIndex()
          && IV->hasIndices())
        // The use we are examining is inserted into an aggregate. Our liveness
        // depends on all uses of that aggregate, but if it is used as a return
        // value, only index at which we were inserted counts.
        RetValNum = *IV->idx_begin();

      // Note that if we are used as the aggregate operand to the insertvalue,
      // we don't change RetValNum, but do survey all our uses.

      Liveness Result = MaybeLive;
      for (Value::use_iterator I = IV->use_begin(),
           E = V->use_end(); I != E; ++I) {
        Result = SurveyUse(I, MaybeLiveUses, RetValNum);
        if (Result == Live)
          break;
      }
      return Result;
    }
    CallSite CS = CallSite::get(V);
    if (CS.getInstruction()) {
      Function *F = CS.getCalledFunction();
      if (F) {
        // Used in a direct call.
  
        // Find the argument number. We know for sure that this use is an
        // argument, since if it was the function argument this would be an
        // indirect call and the we know can't be looking at a value of the
        // label type (for the invoke instruction).
        unsigned ArgNo = CS.getArgumentNo(U.getOperandNo());

        if (ArgNo >= F->getFunctionType()->getNumParams())
          // The value is passed in through a vararg! Must be live.
          return Live;

        assert(CS.getArgument(ArgNo) 
               == CS.getInstruction()->getOperand(U.getOperandNo()) 
               && "Argument is not where we expected it");

        // Value passed to a normal call. It's only live when the corresponding
        // argument to the called function turns out live.
        RetOrArg Use = CreateArg(F, ArgNo);
        return MarkIfNotLive(Use, MaybeLiveUses);
      }
    }
    // Used in any other way? Value must be live.
    return Live;
}

/// SurveyUses - This looks at all the uses of the given value
/// Returns the Liveness deduced from the uses of this value.
///
/// Adds all uses that cause the result to be MaybeLive to MaybeLiveRetUses. If
/// the result is Live, MaybeLiveUses might be modified but its content should
/// be ignored (since it might not be complete).
DAE::Liveness DAE::SurveyUses(Value *V, UseVector &MaybeLiveUses) {
  // Assume it's dead (which will only hold if there are no uses at all..).
  Liveness Result = MaybeLive;
  // Check each use.
  for (Value::use_iterator I = V->use_begin(),
       E = V->use_end(); I != E; ++I) {
    Result = SurveyUse(I, MaybeLiveUses);
    if (Result == Live)
      break;
  }
  return Result;
}

// SurveyFunction - This performs the initial survey of the specified function,
// checking out whether or not it uses any of its incoming arguments or whether
// any callers use the return value.  This fills in the LiveValues set and Uses
// map.
//
// We consider arguments of non-internal functions to be intrinsically alive as
// well as arguments to functions which have their "address taken".
//
void DAE::SurveyFunction(Function &F) {
  unsigned RetCount = NumRetVals(&F);
  // Assume all return values are dead
  typedef SmallVector<Liveness, 5> RetVals;
  RetVals RetValLiveness(RetCount, MaybeLive);

  typedef SmallVector<UseVector, 5> RetUses;
  // These vectors map each return value to the uses that make it MaybeLive, so
  // we can add those to the Uses map if the return value really turns out to be
  // MaybeLive. Initialized to a list of RetCount empty lists.
  RetUses MaybeLiveRetUses(RetCount);

  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    if (ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator()))
      if (RI->getNumOperands() != 0 && RI->getOperand(0)->getType()
          != F.getFunctionType()->getReturnType()) {
        // We don't support old style multiple return values.
        MarkLive(F);
        return;
      }

  if (!F.hasLocalLinkage() && (!ShouldHackArguments() || F.isIntrinsic())) {
    MarkLive(F);
    return;
  }

  DOUT << "DAE - Inspecting callers for fn: " << F.getName() << "\n";
  // Keep track of the number of live retvals, so we can skip checks once all
  // of them turn out to be live.
  unsigned NumLiveRetVals = 0;
  const Type *STy = dyn_cast<StructType>(F.getReturnType());
  // Loop all uses of the function.
  for (Value::use_iterator I = F.use_begin(), E = F.use_end(); I != E; ++I) {
    // If the function is PASSED IN as an argument, its address has been
    // taken.
    CallSite CS = CallSite::get(*I);
    if (!CS.getInstruction() || !CS.isCallee(I)) {
      MarkLive(F);
      return;
    }

    // If this use is anything other than a call site, the function is alive.
    Instruction *TheCall = CS.getInstruction();
    if (!TheCall) {   // Not a direct call site?
      MarkLive(F);
      return;
    }

    // If we end up here, we are looking at a direct call to our function.

    // Now, check how our return value(s) is/are used in this caller. Don't
    // bother checking return values if all of them are live already.
    if (NumLiveRetVals != RetCount) {
      if (STy) {
        // Check all uses of the return value.
        for (Value::use_iterator I = TheCall->use_begin(),
             E = TheCall->use_end(); I != E; ++I) {
          ExtractValueInst *Ext = dyn_cast<ExtractValueInst>(*I);
          if (Ext && Ext->hasIndices()) {
            // This use uses a part of our return value, survey the uses of
            // that part and store the results for this index only.
            unsigned Idx = *Ext->idx_begin();
            if (RetValLiveness[Idx] != Live) {
              RetValLiveness[Idx] = SurveyUses(Ext, MaybeLiveRetUses[Idx]);
              if (RetValLiveness[Idx] == Live)
                NumLiveRetVals++;
            }
          } else {
            // Used by something else than extractvalue. Mark all return
            // values as live.
            for (unsigned i = 0; i != RetCount; ++i )
              RetValLiveness[i] = Live;
            NumLiveRetVals = RetCount;
            break;
          }
        }
      } else {
        // Single return value
        RetValLiveness[0] = SurveyUses(TheCall, MaybeLiveRetUses[0]);
        if (RetValLiveness[0] == Live)
          NumLiveRetVals = RetCount;
      }
    }
  }

  // Now we've inspected all callers, record the liveness of our return values.
  for (unsigned i = 0; i != RetCount; ++i)
    MarkValue(CreateRet(&F, i), RetValLiveness[i], MaybeLiveRetUses[i]);

  DOUT << "DAE - Inspecting args for fn: " << F.getName() << "\n";

  // Now, check all of our arguments.
  unsigned i = 0;
  UseVector MaybeLiveArgUses;
  for (Function::arg_iterator AI = F.arg_begin(),
       E = F.arg_end(); AI != E; ++AI, ++i) {
    // See what the effect of this use is (recording any uses that cause
    // MaybeLive in MaybeLiveArgUses).
    Liveness Result = SurveyUses(AI, MaybeLiveArgUses);
    // Mark the result.
    MarkValue(CreateArg(&F, i), Result, MaybeLiveArgUses);
    // Clear the vector again for the next iteration.
    MaybeLiveArgUses.clear();
  }
}

/// MarkValue - This function marks the liveness of RA depending on L. If L is
/// MaybeLive, it also takes all uses in MaybeLiveUses and records them in Uses,
/// such that RA will be marked live if any use in MaybeLiveUses gets marked
/// live later on.
void DAE::MarkValue(const RetOrArg &RA, Liveness L,
                    const UseVector &MaybeLiveUses) {
  switch (L) {
    case Live: MarkLive(RA); break;
    case MaybeLive:
    {
      // Note any uses of this value, so this return value can be
      // marked live whenever one of the uses becomes live.
      for (UseVector::const_iterator UI = MaybeLiveUses.begin(),
           UE = MaybeLiveUses.end(); UI != UE; ++UI)
        Uses.insert(std::make_pair(*UI, RA));
      break;
    }
  }
}

/// MarkLive - Mark the given Function as alive, meaning that it cannot be
/// changed in any way. Additionally,
/// mark any values that are used as this function's parameters or by its return
/// values (according to Uses) live as well.
void DAE::MarkLive(const Function &F) {
    DOUT << "DAE - Intrinsically live fn: " << F.getName() << "\n";
    // Mark the function as live.
    LiveFunctions.insert(&F);
    // Mark all arguments as live.
    for (unsigned i = 0, e = F.arg_size(); i != e; ++i)
      PropagateLiveness(CreateArg(&F, i));
    // Mark all return values as live.
    for (unsigned i = 0, e = NumRetVals(&F); i != e; ++i)
      PropagateLiveness(CreateRet(&F, i));
}

/// MarkLive - Mark the given return value or argument as live. Additionally,
/// mark any values that are used by this value (according to Uses) live as
/// well.
void DAE::MarkLive(const RetOrArg &RA) {
  if (LiveFunctions.count(RA.F))
    return; // Function was already marked Live.

  if (!LiveValues.insert(RA).second)
    return; // We were already marked Live.

  DOUT << "DAE - Marking " << RA.getDescription() << " live\n";
  PropagateLiveness(RA);
}

/// PropagateLiveness - Given that RA is a live value, propagate it's liveness
/// to any other values it uses (according to Uses).
void DAE::PropagateLiveness(const RetOrArg &RA) {
  // We don't use upper_bound (or equal_range) here, because our recursive call
  // to ourselves is likely to cause the upper_bound (which is the first value
  // not belonging to RA) to become erased and the iterator invalidated.
  UseMap::iterator Begin = Uses.lower_bound(RA);
  UseMap::iterator E = Uses.end();
  UseMap::iterator I;
  for (I = Begin; I != E && I->first == RA; ++I)
    MarkLive(I->second);

  // Erase RA from the Uses map (from the lower bound to wherever we ended up
  // after the loop).
  Uses.erase(Begin, I);
}

// RemoveDeadStuffFromFunction - Remove any arguments and return values from F
// that are not in LiveValues. Transform the function and all of the callees of
// the function to not have these arguments and return values.
//
bool DAE::RemoveDeadStuffFromFunction(Function *F) {
  // Don't modify fully live functions
  if (LiveFunctions.count(F))
    return false;

  // Start by computing a new prototype for the function, which is the same as
  // the old function, but has fewer arguments and a different return type.
  const FunctionType *FTy = F->getFunctionType();
  std::vector<const Type*> Params;

  // Set up to build a new list of parameter attributes.
  SmallVector<AttributeWithIndex, 8> AttributesVec;
  const AttrListPtr &PAL = F->getAttributes();

  // The existing function return attributes.
  Attributes RAttrs = PAL.getRetAttributes();
  Attributes FnAttrs = PAL.getFnAttributes();

  // Find out the new return value.

  const Type *RetTy = FTy->getReturnType();
  const Type *NRetTy = NULL;
  unsigned RetCount = NumRetVals(F);
  // -1 means unused, other numbers are the new index
  SmallVector<int, 5> NewRetIdxs(RetCount, -1);
  std::vector<const Type*> RetTypes;
  if (RetTy == Type::VoidTy) {
    NRetTy = Type::VoidTy;
  } else {
    const StructType *STy = dyn_cast<StructType>(RetTy);
    if (STy)
      // Look at each of the original return values individually.
      for (unsigned i = 0; i != RetCount; ++i) {
        RetOrArg Ret = CreateRet(F, i);
        if (LiveValues.erase(Ret)) {
          RetTypes.push_back(STy->getElementType(i));
          NewRetIdxs[i] = RetTypes.size() - 1;
        } else {
          ++NumRetValsEliminated;
          DOUT << "DAE - Removing return value " << i << " from "
               << F->getNameStart() << "\n";
        }
      }
    else
      // We used to return a single value.
      if (LiveValues.erase(CreateRet(F, 0))) {
        RetTypes.push_back(RetTy);
        NewRetIdxs[0] = 0;
      } else {
        DOUT << "DAE - Removing return value from " << F->getNameStart()
             << "\n";
        ++NumRetValsEliminated;
      }
    if (RetTypes.size() > 1)
      // More than one return type? Return a struct with them. Also, if we used
      // to return a struct and didn't change the number of return values,
      // return a struct again. This prevents changing {something} into
      // something and {} into void.
      // Make the new struct packed if we used to return a packed struct
      // already.
      NRetTy = StructType::get(RetTypes, STy->isPacked());
    else if (RetTypes.size() == 1)
      // One return type? Just a simple value then, but only if we didn't use to
      // return a struct with that simple value before.
      NRetTy = RetTypes.front();
    else if (RetTypes.size() == 0)
      // No return types? Make it void, but only if we didn't use to return {}.
      NRetTy = Type::VoidTy;
  }

  assert(NRetTy && "No new return type found?");

  // Remove any incompatible attributes, but only if we removed all return
  // values. Otherwise, ensure that we don't have any conflicting attributes
  // here. Currently, this should not be possible, but special handling might be
  // required when new return value attributes are added.
  if (NRetTy == Type::VoidTy)
    RAttrs &= ~Attribute::typeIncompatible(NRetTy);
  else
    assert((RAttrs & Attribute::typeIncompatible(NRetTy)) == 0 
           && "Return attributes no longer compatible?");

  if (RAttrs)
    AttributesVec.push_back(AttributeWithIndex::get(0, RAttrs));

  // Remember which arguments are still alive.
  SmallVector<bool, 10> ArgAlive(FTy->getNumParams(), false);
  // Construct the new parameter list from non-dead arguments. Also construct
  // a new set of parameter attributes to correspond. Skip the first parameter
  // attribute, since that belongs to the return value.
  unsigned i = 0;
  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end();
       I != E; ++I, ++i) {
    RetOrArg Arg = CreateArg(F, i);
    if (LiveValues.erase(Arg)) {
      Params.push_back(I->getType());
      ArgAlive[i] = true;

      // Get the original parameter attributes (skipping the first one, that is
      // for the return value.
      if (Attributes Attrs = PAL.getParamAttributes(i + 1))
        AttributesVec.push_back(AttributeWithIndex::get(Params.size(), Attrs));
    } else {
      ++NumArgumentsEliminated;
      DOUT << "DAE - Removing argument " << i << " (" << I->getNameStart()
           << ") from " << F->getNameStart() << "\n";
    }
  }

  if (FnAttrs != Attribute::None) 
    AttributesVec.push_back(AttributeWithIndex::get(~0, FnAttrs));

  // Reconstruct the AttributesList based on the vector we constructed.
  AttrListPtr NewPAL = AttrListPtr::get(AttributesVec.begin(), AttributesVec.end());

  // Work around LLVM bug PR56: the CWriter cannot emit varargs functions which
  // have zero fixed arguments.
  //
  // Note that we apply this hack for a vararg fuction that does not have any
  // arguments anymore, but did have them before (so don't bother fixing
  // functions that were already broken wrt CWriter).
  bool ExtraArgHack = false;
  if (Params.empty() && FTy->isVarArg() && FTy->getNumParams() != 0) {
    ExtraArgHack = true;
    Params.push_back(Type::Int32Ty);
  }

  // Create the new function type based on the recomputed parameters.
  FunctionType *NFTy = FunctionType::get(NRetTy, Params, FTy->isVarArg());

  // No change?
  if (NFTy == FTy)
    return false;

  // Create the new function body and insert it into the module...
  Function *NF = Function::Create(NFTy, F->getLinkage());
  NF->copyAttributesFrom(F);
  NF->setAttributes(NewPAL);
  // Insert the new function before the old function, so we won't be processing
  // it again.
  F->getParent()->getFunctionList().insert(F, NF);
  NF->takeName(F);

  // Loop over all of the callers of the function, transforming the call sites
  // to pass in a smaller number of arguments into the new function.
  //
  std::vector<Value*> Args;
  while (!F->use_empty()) {
    CallSite CS = CallSite::get(F->use_back());
    Instruction *Call = CS.getInstruction();

    AttributesVec.clear();
    const AttrListPtr &CallPAL = CS.getAttributes();

    // The call return attributes.
    Attributes RAttrs = CallPAL.getRetAttributes();
    Attributes FnAttrs = CallPAL.getFnAttributes();
    // Adjust in case the function was changed to return void.
    RAttrs &= ~Attribute::typeIncompatible(NF->getReturnType());
    if (RAttrs)
      AttributesVec.push_back(AttributeWithIndex::get(0, RAttrs));

    // Declare these outside of the loops, so we can reuse them for the second
    // loop, which loops the varargs.
    CallSite::arg_iterator I = CS.arg_begin();
    unsigned i = 0;
    // Loop over those operands, corresponding to the normal arguments to the
    // original function, and add those that are still alive.
    for (unsigned e = FTy->getNumParams(); i != e; ++I, ++i)
      if (ArgAlive[i]) {
        Args.push_back(*I);
        // Get original parameter attributes, but skip return attributes.
        if (Attributes Attrs = CallPAL.getParamAttributes(i + 1))
          AttributesVec.push_back(AttributeWithIndex::get(Args.size(), Attrs));
      }

    if (ExtraArgHack)
      Args.push_back(UndefValue::get(Type::Int32Ty));

    // Push any varargs arguments on the list. Don't forget their attributes.
    for (CallSite::arg_iterator E = CS.arg_end(); I != E; ++I, ++i) {
      Args.push_back(*I);
      if (Attributes Attrs = CallPAL.getParamAttributes(i + 1))
        AttributesVec.push_back(AttributeWithIndex::get(Args.size(), Attrs));
    }

    if (FnAttrs != Attribute::None)
      AttributesVec.push_back(AttributeWithIndex::get(~0, FnAttrs));

    // Reconstruct the AttributesList based on the vector we constructed.
    AttrListPtr NewCallPAL = AttrListPtr::get(AttributesVec.begin(),
                                              AttributesVec.end());

    Instruction *New;
    if (InvokeInst *II = dyn_cast<InvokeInst>(Call)) {
      New = InvokeInst::Create(NF, II->getNormalDest(), II->getUnwindDest(),
                               Args.begin(), Args.end(), "", Call);
      cast<InvokeInst>(New)->setCallingConv(CS.getCallingConv());
      cast<InvokeInst>(New)->setAttributes(NewCallPAL);
    } else {
      New = CallInst::Create(NF, Args.begin(), Args.end(), "", Call);
      cast<CallInst>(New)->setCallingConv(CS.getCallingConv());
      cast<CallInst>(New)->setAttributes(NewCallPAL);
      if (cast<CallInst>(Call)->isTailCall())
        cast<CallInst>(New)->setTailCall();
    }
    Args.clear();

    if (!Call->use_empty()) {
      if (New->getType() == Call->getType()) {
        // Return type not changed? Just replace users then.
        Call->replaceAllUsesWith(New);
        New->takeName(Call);
      } else if (New->getType() == Type::VoidTy) {
        // Our return value has uses, but they will get removed later on.
        // Replace by null for now.
        Call->replaceAllUsesWith(Constant::getNullValue(Call->getType()));
      } else {
        assert(isa<StructType>(RetTy) && "Return type changed, but not into a"
                                         "void. The old return type must have"
                                         "been a struct!");
        // We used to return a struct. Instead of doing smart stuff with all the
        // uses of this struct, we will just rebuild it using
        // extract/insertvalue chaining and let instcombine clean that up.
        //
        // Start out building up our return value from undef
        Value *RetVal = llvm::UndefValue::get(RetTy);
        for (unsigned i = 0; i != RetCount; ++i)
          if (NewRetIdxs[i] != -1) {
            Value *V;
            if (RetTypes.size() > 1)
              // We are still returning a struct, so extract the value from our
              // return value
              V = ExtractValueInst::Create(New, NewRetIdxs[i], "newret", Call);
            else
              // We are now returning a single element, so just insert that
              V = New;
            // Insert the value at the old position
            RetVal = InsertValueInst::Create(RetVal, V, i, "oldret", Call);
          }
        // Now, replace all uses of the old call instruction with the return
        // struct we built
        Call->replaceAllUsesWith(RetVal);
        New->takeName(Call);
      }
    }

    // Finally, remove the old call from the program, reducing the use-count of
    // F.
    Call->eraseFromParent();
  }

  // Since we have now created the new function, splice the body of the old
  // function right into the new function, leaving the old rotting hulk of the
  // function empty.
  NF->getBasicBlockList().splice(NF->begin(), F->getBasicBlockList());

  // Loop over the argument list, transfering uses of the old arguments over to
  // the new arguments, also transfering over the names as well.
  i = 0;
  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(),
       I2 = NF->arg_begin(); I != E; ++I, ++i)
    if (ArgAlive[i]) {
      // If this is a live argument, move the name and users over to the new
      // version.
      I->replaceAllUsesWith(I2);
      I2->takeName(I);
      ++I2;
    } else {
      // If this argument is dead, replace any uses of it with null constants
      // (these are guaranteed to become unused later on).
      I->replaceAllUsesWith(Constant::getNullValue(I->getType()));
    }

  // If we change the return value of the function we must rewrite any return
  // instructions.  Check this now.
  if (F->getReturnType() != NF->getReturnType())
    for (Function::iterator BB = NF->begin(), E = NF->end(); BB != E; ++BB)
      if (ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator())) {
        Value *RetVal;

        if (NFTy->getReturnType() == Type::VoidTy) {
          RetVal = 0;
        } else {
          assert (isa<StructType>(RetTy));
          // The original return value was a struct, insert
          // extractvalue/insertvalue chains to extract only the values we need
          // to return and insert them into our new result.
          // This does generate messy code, but we'll let it to instcombine to
          // clean that up.
          Value *OldRet = RI->getOperand(0);
          // Start out building up our return value from undef
          RetVal = llvm::UndefValue::get(NRetTy);
          for (unsigned i = 0; i != RetCount; ++i)
            if (NewRetIdxs[i] != -1) {
              ExtractValueInst *EV = ExtractValueInst::Create(OldRet, i,
                                                              "oldret", RI);
              if (RetTypes.size() > 1) {
                // We're still returning a struct, so reinsert the value into
                // our new return value at the new index

                RetVal = InsertValueInst::Create(RetVal, EV, NewRetIdxs[i],
                                                 "newret", RI);
              } else {
                // We are now only returning a simple value, so just return the
                // extracted value.
                RetVal = EV;
              }
            }
        }
        // Replace the return instruction with one returning the new return
        // value (possibly 0 if we became void).
        ReturnInst::Create(RetVal, RI);
        BB->getInstList().erase(RI);
      }

  // Now that the old function is dead, delete it.
  F->eraseFromParent();

  return true;
}

bool DAE::runOnModule(Module &M) {
  bool Changed = false;

  // First pass: Do a simple check to see if any functions can have their "..."
  // removed.  We can do this if they never call va_start.  This loop cannot be
  // fused with the next loop, because deleting a function invalidates
  // information computed while surveying other functions.
  DOUT << "DAE - Deleting dead varargs\n";
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ) {
    Function &F = *I++;
    if (F.getFunctionType()->isVarArg())
      Changed |= DeleteDeadVarargs(F);
  }

  // Second phase:loop through the module, determining which arguments are live.
  // We assume all arguments are dead unless proven otherwise (allowing us to
  // determine that dead arguments passed into recursive functions are dead).
  //
  DOUT << "DAE - Determining liveness\n";
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    SurveyFunction(*I);
  
  // Now, remove all dead arguments and return values from each function in
  // turn
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ) {
    // Increment now, because the function will probably get removed (ie
    // replaced by a new one).
    Function *F = I++;
    Changed |= RemoveDeadStuffFromFunction(F);
  }
  return Changed;
}
