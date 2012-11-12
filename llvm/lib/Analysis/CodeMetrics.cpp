//===- CodeMetrics.cpp - Code cost measurements ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements code cost measurement utilities.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Function.h"
#include "llvm/Support/CallSite.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/DataLayout.h"

using namespace llvm;

/// callIsSmall - If a call is likely to lower to a single target instruction,
/// or is otherwise deemed small return true.
/// TODO: Perhaps calls like memcpy, strcpy, etc?
bool llvm::callIsSmall(ImmutableCallSite CS) {
  if (isa<IntrinsicInst>(CS.getInstruction()))
    return true;

  const Function *F = CS.getCalledFunction();
  if (!F) return false;

  if (F->hasLocalLinkage()) return false;

  if (!F->hasName()) return false;

  StringRef Name = F->getName();

  // These will all likely lower to a single selection DAG node.
  if (Name == "copysign" || Name == "copysignf" || Name == "copysignl" ||
      Name == "fabs" || Name == "fabsf" || Name == "fabsl" ||
      Name == "sin" || Name == "sinf" || Name == "sinl" ||
      Name == "cos" || Name == "cosf" || Name == "cosl" ||
      Name == "sqrt" || Name == "sqrtf" || Name == "sqrtl" )
    return true;

  // These are all likely to be optimized into something smaller.
  if (Name == "pow" || Name == "powf" || Name == "powl" ||
      Name == "exp2" || Name == "exp2l" || Name == "exp2f" ||
      Name == "floor" || Name == "floorf" || Name == "ceil" ||
      Name == "round" || Name == "ffs" || Name == "ffsl" ||
      Name == "abs" || Name == "labs" || Name == "llabs")
    return true;

  return false;
}

bool llvm::isInstructionFree(const Instruction *I, const DataLayout *TD) {
  if (isa<PHINode>(I))
    return true;

  // If a GEP has all constant indices, it will probably be folded with
  // a load/store.
  if (const GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I))
    return GEP->hasAllConstantIndices();

  if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
    switch (II->getIntrinsicID()) {
    default:
      return false;
    case Intrinsic::dbg_declare:
    case Intrinsic::dbg_value:
    case Intrinsic::invariant_start:
    case Intrinsic::invariant_end:
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
    case Intrinsic::objectsize:
    case Intrinsic::ptr_annotation:
    case Intrinsic::var_annotation:
      // These intrinsics don't count as size.
      return true;
    }
  }

  if (const CastInst *CI = dyn_cast<CastInst>(I)) {
    // Noop casts, including ptr <-> int,  don't count.
    if (CI->isLosslessCast())
      return true;

    Value *Op = CI->getOperand(0);
    // An inttoptr cast is free so long as the input is a legal integer type
    // which doesn't contain values outside the range of a pointer.
    if (isa<IntToPtrInst>(CI) && TD &&
        TD->isLegalInteger(Op->getType()->getScalarSizeInBits()) &&
        Op->getType()->getScalarSizeInBits() <= TD->getPointerSizeInBits())
      return true;

    // A ptrtoint cast is free so long as the result is large enough to store
    // the pointer, and a legal integer type.
    if (isa<PtrToIntInst>(CI) && TD &&
        TD->isLegalInteger(Op->getType()->getScalarSizeInBits()) &&
        Op->getType()->getScalarSizeInBits() >= TD->getPointerSizeInBits())
      return true;

    // trunc to a native type is free (assuming the target has compare and
    // shift-right of the same width).
    if (TD && isa<TruncInst>(CI) &&
        TD->isLegalInteger(TD->getTypeSizeInBits(CI->getType())))
      return true;
    // Result of a cmp instruction is often extended (to be used by other
    // cmp instructions, logical or return instructions). These are usually
    // nop on most sane targets.
    if (isa<CmpInst>(CI->getOperand(0)))
      return true;
  }

  return false;
}

/// analyzeBasicBlock - Fill in the current structure with information gleaned
/// from the specified block.
void CodeMetrics::analyzeBasicBlock(const BasicBlock *BB,
                                    const DataLayout *TD) {
  ++NumBlocks;
  unsigned NumInstsBeforeThisBB = NumInsts;
  for (BasicBlock::const_iterator II = BB->begin(), E = BB->end();
       II != E; ++II) {
    if (isInstructionFree(II, TD))
      continue;

    // Special handling for calls.
    if (isa<CallInst>(II) || isa<InvokeInst>(II)) {
      ImmutableCallSite CS(cast<Instruction>(II));

      if (const Function *F = CS.getCalledFunction()) {
        // If a function is both internal and has a single use, then it is
        // extremely likely to get inlined in the future (it was probably
        // exposed by an interleaved devirtualization pass).
        if (!CS.isNoInline() && F->hasInternalLinkage() && F->hasOneUse())
          ++NumInlineCandidates;

        // If this call is to function itself, then the function is recursive.
        // Inlining it into other functions is a bad idea, because this is
        // basically just a form of loop peeling, and our metrics aren't useful
        // for that case.
        if (F == BB->getParent())
          isRecursive = true;
      }

      if (!callIsSmall(CS)) {
        // Each argument to a call takes on average one instruction to set up.
        NumInsts += CS.arg_size();

        // We don't want inline asm to count as a call - that would prevent loop
        // unrolling. The argument setup cost is still real, though.
        if (!isa<InlineAsm>(CS.getCalledValue()))
          ++NumCalls;
      }
    }

    if (const AllocaInst *AI = dyn_cast<AllocaInst>(II)) {
      if (!AI->isStaticAlloca())
        this->usesDynamicAlloca = true;
    }

    if (isa<ExtractElementInst>(II) || II->getType()->isVectorTy())
      ++NumVectorInsts;

    ++NumInsts;
  }

  if (isa<ReturnInst>(BB->getTerminator()))
    ++NumRets;

  // We never want to inline functions that contain an indirectbr.  This is
  // incorrect because all the blockaddress's (in static global initializers
  // for example) would be referring to the original function, and this indirect
  // jump would jump from the inlined copy of the function into the original
  // function which is extremely undefined behavior.
  // FIXME: This logic isn't really right; we can safely inline functions
  // with indirectbr's as long as no other function or global references the
  // blockaddress of a block within the current function.  And as a QOI issue,
  // if someone is using a blockaddress without an indirectbr, and that
  // reference somehow ends up in another function or global, we probably
  // don't want to inline this function.
  if (isa<IndirectBrInst>(BB->getTerminator()))
    containsIndirectBr = true;

  // Remember NumInsts for this BB.
  NumBBInsts[BB] = NumInsts - NumInstsBeforeThisBB;
}

void CodeMetrics::analyzeFunction(Function *F, const DataLayout *TD) {
  // If this function contains a call that "returns twice" (e.g., setjmp or
  // _setjmp) and it isn't marked with "returns twice" itself, never inline it.
  // This is a hack because we depend on the user marking their local variables
  // as volatile if they are live across a setjmp call, and they probably
  // won't do this in callers.
  exposesReturnsTwice = F->callsFunctionThatReturnsTwice() &&
    !F->getFnAttributes().hasAttribute(Attributes::ReturnsTwice);

  // Look at the size of the callee.
  for (Function::const_iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
    analyzeBasicBlock(&*BB, TD);
}
