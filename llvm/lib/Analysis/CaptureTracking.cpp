//===--- CaptureTracking.cpp - Determine whether a pointer is captured ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains routines that help determine which pointers are captured.
// A pointer value is captured if the function makes a copy of any part of the
// pointer that outlives the call.  Not being captured means, more or less, that
// the pointer is only dereferenced and not stored in a global.  Returning part
// of the pointer as the function return value may or may not count as capturing
// the pointer, depending on the context.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CaptureTracking.h"
using namespace llvm;

CaptureTracker::~CaptureTracker() {}

bool CaptureTracker::shouldExplore(Use *U) { return true; }

namespace {
  struct SimpleCaptureTracker : public CaptureTracker {
    explicit SimpleCaptureTracker(bool ReturnCaptures)
      : ReturnCaptures(ReturnCaptures), Captured(false) {}

    void tooManyUses() { Captured = true; }

    bool captured(Use *U) {
      if (isa<ReturnInst>(U->getUser()) && !ReturnCaptures)
        return false;

      Captured = true;
      return true;
    }

    bool ReturnCaptures;

    bool Captured;
  };
}

/// PointerMayBeCaptured - Return true if this pointer value may be captured
/// by the enclosing function (which is required to exist).  This routine can
/// be expensive, so consider caching the results.  The boolean ReturnCaptures
/// specifies whether returning the value (or part of it) from the function
/// counts as capturing it or not.  The boolean StoreCaptures specified whether
/// storing the value (or part of it) into memory anywhere automatically
/// counts as capturing it or not.
bool llvm::PointerMayBeCaptured(const Value *V,
                                bool ReturnCaptures, bool StoreCaptures) {
  assert(!isa<GlobalValue>(V) &&
         "It doesn't make sense to ask whether a global is captured.");

  // TODO: If StoreCaptures is not true, we could do Fancy analysis
  // to determine whether this store is not actually an escape point.
  // In that case, BasicAliasAnalysis should be updated as well to
  // take advantage of this.
  (void)StoreCaptures;

  SimpleCaptureTracker SCT(ReturnCaptures);
  PointerMayBeCaptured(V, &SCT);
  return SCT.Captured;
}

/// TODO: Write a new FunctionPass AliasAnalysis so that it can keep
/// a cache. Then we can move the code from BasicAliasAnalysis into
/// that path, and remove this threshold.
static int const Threshold = 20;

void llvm::PointerMayBeCaptured(const Value *V, CaptureTracker *Tracker) {
  assert(V->getType()->isPointerTy() && "Capture is for pointers only!");
  SmallVector<Use*, Threshold> Worklist;
  SmallSet<Use*, Threshold> Visited;
  int Count = 0;

  for (Value::const_use_iterator UI = V->use_begin(), UE = V->use_end();
       UI != UE; ++UI) {
    // If there are lots of uses, conservatively say that the value
    // is captured to avoid taking too much compile time.
    if (Count++ >= Threshold)
      return Tracker->tooManyUses();

    Use *U = &UI.getUse();
    if (!Tracker->shouldExplore(U)) continue;
    Visited.insert(U);
    Worklist.push_back(U);
  }

  while (!Worklist.empty()) {
    Use *U = Worklist.pop_back_val();
    Instruction *I = cast<Instruction>(U->getUser());
    V = U->get();

    switch (I->getOpcode()) {
    case Instruction::Call:
    case Instruction::Invoke: {
      CallSite CS(I);
      // Not captured if the callee is readonly, doesn't return a copy through
      // its return value and doesn't unwind (a readonly function can leak bits
      // by throwing an exception or not depending on the input value).
      if (CS.onlyReadsMemory() && CS.doesNotThrow() && I->getType()->isVoidTy())
        break;

      // Not captured if only passed via 'nocapture' arguments.  Note that
      // calling a function pointer does not in itself cause the pointer to
      // be captured.  This is a subtle point considering that (for example)
      // the callee might return its own address.  It is analogous to saying
      // that loading a value from a pointer does not cause the pointer to be
      // captured, even though the loaded value might be the pointer itself
      // (think of self-referential objects).
      CallSite::arg_iterator B = CS.arg_begin(), E = CS.arg_end();
      for (CallSite::arg_iterator A = B; A != E; ++A)
        if (A->get() == V && !CS.doesNotCapture(A - B))
          // The parameter is not marked 'nocapture' - captured.
          if (Tracker->captured(U))
            return;
      break;
    }
    case Instruction::Load:
      // Loading from a pointer does not cause it to be captured.
      break;
    case Instruction::VAArg:
      // "va-arg" from a pointer does not cause it to be captured.
      break;
    case Instruction::Store:
      if (V == I->getOperand(0))
        // Stored the pointer - conservatively assume it may be captured.
        if (Tracker->captured(U))
          return;
      // Storing to the pointee does not cause the pointer to be captured.
      break;
    case Instruction::BitCast:
    case Instruction::GetElementPtr:
    case Instruction::PHI:
    case Instruction::Select:
      // The original value is not captured via this if the new value isn't.
      for (Instruction::use_iterator UI = I->use_begin(), UE = I->use_end();
           UI != UE; ++UI) {
        Use *U = &UI.getUse();
        if (Visited.insert(U))
          if (Tracker->shouldExplore(U))
            Worklist.push_back(U);
      }
      break;
    case Instruction::ICmp:
      // Don't count comparisons of a no-alias return value against null as
      // captures. This allows us to ignore comparisons of malloc results
      // with null, for example.
      if (isNoAliasCall(V->stripPointerCasts()))
        if (ConstantPointerNull *CPN =
              dyn_cast<ConstantPointerNull>(I->getOperand(1)))
          if (CPN->getType()->getAddressSpace() == 0)
            break;
      // Otherwise, be conservative. There are crazy ways to capture pointers
      // using comparisons.
      if (Tracker->captured(U))
        return;
      break;
    default:
      // Something else - be conservative and say it is captured.
      if (Tracker->captured(U))
        return;
      break;
    }
  }

  // All uses examined.
}
