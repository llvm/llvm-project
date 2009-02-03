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

#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Instructions.h"
#include "llvm/Value.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CallSite.h"
using namespace llvm;

/// PointerMayBeCaptured - Return true if this pointer value may be captured
/// by the enclosing function (which is required to exist).  This routine can
/// be expensive, so consider caching the results.  The boolean ReturnCaptures
/// specifies whether returning the value (or part of it) from the function
/// counts as capturing it or not.
bool llvm::PointerMayBeCaptured(const Value *V, bool ReturnCaptures) {
  assert(isa<PointerType>(V->getType()) && "Capture is for pointers only!");
  SmallVector<Use*, 16> Worklist;
  SmallSet<Use*, 16> Visited;

  for (Value::use_const_iterator UI = V->use_begin(), UE = V->use_end();
       UI != UE; ++UI) {
    Use *U = &UI.getUse();
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
      CallSite CS = CallSite::get(I);
      // Not captured if the callee is readonly and doesn't return a copy
      // through its return value.
      if (CS.onlyReadsMemory() && I->getType() == Type::VoidTy)
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
        if (A->get() == V && !CS.paramHasAttr(A - B + 1, Attribute::NoCapture))
          // The parameter is not marked 'nocapture' - captured.
          return true;
      // Only passed via 'nocapture' arguments, or is the called function - not
      // captured.
      break;
    }
    case Instruction::Free:
      // Freeing a pointer does not cause it to be captured.
      break;
    case Instruction::Load:
      // Loading from a pointer does not cause it to be captured.
      break;
    case Instruction::Ret:
      if (ReturnCaptures)
        return true;
      break;
    case Instruction::Store:
      if (V == I->getOperand(0))
        // Stored the pointer - it may be captured.
        return true;
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
          Worklist.push_back(U);
      }
      break;
    default:
      // Something else - be conservative and say it is captured.
      return true;
    }
  }

  // All uses examined - not captured.
  return false;
}
