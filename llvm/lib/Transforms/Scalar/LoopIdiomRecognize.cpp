//===-- LoopIdiomRecognize.cpp - Loop idiom recognition -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass implements an idiom recognizer that transforms simple loops into a
// non-loop form.  In cases that this kicks in, it can be a significant
// performance win.
//
//===----------------------------------------------------------------------===//
//
// TODO List:
//
// Future loop memory idioms to recognize:
//   memcmp, memmove, strlen, etc.
// Future floating point idioms to recognize in -ffast-math mode:
//   fpowi
// Future integer operation idioms to recognize:
//   ctpop, ctlz, cttz
//
// Beware that isel's default lowering for ctpop is highly inefficient for
// i64 and larger types when i64 is legal and the value has few bits set.  It
// would be good to enhance isel to emit a loop for ctpop in this case.
//
// We should enhance the memset/memcpy recognition to handle multiple stores in
// the loop.  This would handle things like:
//   void foo(_Complex float *P)
//     for (i) { __real__(*P) = 0;  __imag__(*P) = 0; }
//
// We should enhance this to handle negative strides through memory.
// Alternatively (and perhaps better) we could rely on an earlier pass to force
// forward iteration through memory, which is generally better for cache
// behavior.  Negative strides *do* happen for memset/memcpy loops.
//
// This could recognize common matrix multiplies and dot product idioms and
// replace them with calls to BLAS (if linked in??).
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-idiom"
#include "llvm/Transforms/Scalar.h"
#include "llvm/IRBuilder.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Module.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;

STATISTIC(NumMemSet, "Number of memset's formed from loop stores");
STATISTIC(NumMemCpy, "Number of memcpy's formed from loop load+stores");

namespace {
  class LoopIdiomRecognize : public LoopPass {
    Loop *CurLoop;
    const DataLayout *TD;
    DominatorTree *DT;
    ScalarEvolution *SE;
    TargetLibraryInfo *TLI;
  public:
    static char ID;
    explicit LoopIdiomRecognize() : LoopPass(ID) {
      initializeLoopIdiomRecognizePass(*PassRegistry::getPassRegistry());
    }

    bool runOnLoop(Loop *L, LPPassManager &LPM);
    bool runOnLoopBlock(BasicBlock *BB, const SCEV *BECount,
                        SmallVectorImpl<BasicBlock*> &ExitBlocks);

    bool processLoopStore(StoreInst *SI, const SCEV *BECount);
    bool processLoopMemSet(MemSetInst *MSI, const SCEV *BECount);

    bool processLoopStridedStore(Value *DestPtr, unsigned StoreSize,
                                 unsigned StoreAlignment,
                                 Value *SplatValue, Instruction *TheStore,
                                 const SCEVAddRecExpr *Ev,
                                 const SCEV *BECount);
    bool processLoopStoreOfLoopLoad(StoreInst *SI, unsigned StoreSize,
                                    const SCEVAddRecExpr *StoreEv,
                                    const SCEVAddRecExpr *LoadEv,
                                    const SCEV *BECount);

    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG.
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<LoopInfo>();
      AU.addPreserved<LoopInfo>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.addRequiredID(LCSSAID);
      AU.addPreservedID(LCSSAID);
      AU.addRequired<AliasAnalysis>();
      AU.addPreserved<AliasAnalysis>();
      AU.addRequired<ScalarEvolution>();
      AU.addPreserved<ScalarEvolution>();
      AU.addPreserved<DominatorTree>();
      AU.addRequired<DominatorTree>();
      AU.addRequired<TargetLibraryInfo>();
    }
  };
}

char LoopIdiomRecognize::ID = 0;
INITIALIZE_PASS_BEGIN(LoopIdiomRecognize, "loop-idiom", "Recognize loop idioms",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LCSSA)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfo)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(LoopIdiomRecognize, "loop-idiom", "Recognize loop idioms",
                    false, false)

Pass *llvm::createLoopIdiomPass() { return new LoopIdiomRecognize(); }

/// deleteDeadInstruction - Delete this instruction.  Before we do, go through
/// and zero out all the operands of this instruction.  If any of them become
/// dead, delete them and the computation tree that feeds them.
///
static void deleteDeadInstruction(Instruction *I, ScalarEvolution &SE,
                                  const TargetLibraryInfo *TLI) {
  SmallVector<Instruction*, 32> NowDeadInsts;

  NowDeadInsts.push_back(I);

  // Before we touch this instruction, remove it from SE!
  do {
    Instruction *DeadInst = NowDeadInsts.pop_back_val();

    // This instruction is dead, zap it, in stages.  Start by removing it from
    // SCEV.
    SE.forgetValue(DeadInst);

    for (unsigned op = 0, e = DeadInst->getNumOperands(); op != e; ++op) {
      Value *Op = DeadInst->getOperand(op);
      DeadInst->setOperand(op, 0);

      // If this operand just became dead, add it to the NowDeadInsts list.
      if (!Op->use_empty()) continue;

      if (Instruction *OpI = dyn_cast<Instruction>(Op))
        if (isInstructionTriviallyDead(OpI, TLI))
          NowDeadInsts.push_back(OpI);
    }

    DeadInst->eraseFromParent();

  } while (!NowDeadInsts.empty());
}

/// deleteIfDeadInstruction - If the specified value is a dead instruction,
/// delete it and any recursively used instructions.
static void deleteIfDeadInstruction(Value *V, ScalarEvolution &SE,
                                    const TargetLibraryInfo *TLI) {
  if (Instruction *I = dyn_cast<Instruction>(V))
    if (isInstructionTriviallyDead(I, TLI))
      deleteDeadInstruction(I, SE, TLI);
}

bool LoopIdiomRecognize::runOnLoop(Loop *L, LPPassManager &LPM) {
  CurLoop = L;

  // If the loop could not be converted to canonical form, it must have an
  // indirectbr in it, just give up.
  if (!L->getLoopPreheader())
    return false;

  // Disable loop idiom recognition if the function's name is a common idiom.
  StringRef Name = L->getHeader()->getParent()->getName();
  if (Name == "memset" || Name == "memcpy")
    return false;

  // The trip count of the loop must be analyzable.
  SE = &getAnalysis<ScalarEvolution>();
  if (!SE->hasLoopInvariantBackedgeTakenCount(L))
    return false;
  const SCEV *BECount = SE->getBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(BECount)) return false;

  // If this loop executes exactly one time, then it should be peeled, not
  // optimized by this pass.
  if (const SCEVConstant *BECst = dyn_cast<SCEVConstant>(BECount))
    if (BECst->getValue()->getValue() == 0)
      return false;

  // We require target data for now.
  TD = getAnalysisIfAvailable<DataLayout>();
  if (TD == 0) return false;

  DT = &getAnalysis<DominatorTree>();
  LoopInfo &LI = getAnalysis<LoopInfo>();
  TLI = &getAnalysis<TargetLibraryInfo>();

  SmallVector<BasicBlock*, 8> ExitBlocks;
  CurLoop->getUniqueExitBlocks(ExitBlocks);

  DEBUG(dbgs() << "loop-idiom Scanning: F["
               << L->getHeader()->getParent()->getName()
               << "] Loop %" << L->getHeader()->getName() << "\n");

  bool MadeChange = false;
  // Scan all the blocks in the loop that are not in subloops.
  for (Loop::block_iterator BI = L->block_begin(), E = L->block_end(); BI != E;
       ++BI) {
    // Ignore blocks in subloops.
    if (LI.getLoopFor(*BI) != CurLoop)
      continue;

    MadeChange |= runOnLoopBlock(*BI, BECount, ExitBlocks);
  }
  return MadeChange;
}

/// runOnLoopBlock - Process the specified block, which lives in a counted loop
/// with the specified backedge count.  This block is known to be in the current
/// loop and not in any subloops.
bool LoopIdiomRecognize::runOnLoopBlock(BasicBlock *BB, const SCEV *BECount,
                                     SmallVectorImpl<BasicBlock*> &ExitBlocks) {
  // We can only promote stores in this block if they are unconditionally
  // executed in the loop.  For a block to be unconditionally executed, it has
  // to dominate all the exit blocks of the loop.  Verify this now.
  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i)
    if (!DT->dominates(BB, ExitBlocks[i]))
      return false;

  bool MadeChange = false;
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ) {
    Instruction *Inst = I++;
    // Look for store instructions, which may be optimized to memset/memcpy.
    if (StoreInst *SI = dyn_cast<StoreInst>(Inst))  {
      WeakVH InstPtr(I);
      if (!processLoopStore(SI, BECount)) continue;
      MadeChange = true;

      // If processing the store invalidated our iterator, start over from the
      // top of the block.
      if (InstPtr == 0)
        I = BB->begin();
      continue;
    }

    // Look for memset instructions, which may be optimized to a larger memset.
    if (MemSetInst *MSI = dyn_cast<MemSetInst>(Inst))  {
      WeakVH InstPtr(I);
      if (!processLoopMemSet(MSI, BECount)) continue;
      MadeChange = true;

      // If processing the memset invalidated our iterator, start over from the
      // top of the block.
      if (InstPtr == 0)
        I = BB->begin();
      continue;
    }
  }

  return MadeChange;
}


/// processLoopStore - See if this store can be promoted to a memset or memcpy.
bool LoopIdiomRecognize::processLoopStore(StoreInst *SI, const SCEV *BECount) {
  if (!SI->isSimple()) return false;

  Value *StoredVal = SI->getValueOperand();
  Value *StorePtr = SI->getPointerOperand();

  // Reject stores that are so large that they overflow an unsigned.
  uint64_t SizeInBits = TD->getTypeSizeInBits(StoredVal->getType());
  if ((SizeInBits & 7) || (SizeInBits >> 32) != 0)
    return false;

  // See if the pointer expression is an AddRec like {base,+,1} on the current
  // loop, which indicates a strided store.  If we have something else, it's a
  // random store we can't handle.
  const SCEVAddRecExpr *StoreEv =
    dyn_cast<SCEVAddRecExpr>(SE->getSCEV(StorePtr));
  if (StoreEv == 0 || StoreEv->getLoop() != CurLoop || !StoreEv->isAffine())
    return false;

  // Check to see if the stride matches the size of the store.  If so, then we
  // know that every byte is touched in the loop.
  unsigned StoreSize = (unsigned)SizeInBits >> 3;
  const SCEVConstant *Stride = dyn_cast<SCEVConstant>(StoreEv->getOperand(1));

  if (Stride == 0 || StoreSize != Stride->getValue()->getValue()) {
    // TODO: Could also handle negative stride here someday, that will require
    // the validity check in mayLoopAccessLocation to be updated though.
    // Enable this to print exact negative strides.
    if (0 && Stride && StoreSize == -Stride->getValue()->getValue()) {
      dbgs() << "NEGATIVE STRIDE: " << *SI << "\n";
      dbgs() << "BB: " << *SI->getParent();
    }

    return false;
  }

  // See if we can optimize just this store in isolation.
  if (processLoopStridedStore(StorePtr, StoreSize, SI->getAlignment(),
                              StoredVal, SI, StoreEv, BECount))
    return true;

  // If the stored value is a strided load in the same loop with the same stride
  // this this may be transformable into a memcpy.  This kicks in for stuff like
  //   for (i) A[i] = B[i];
  if (LoadInst *LI = dyn_cast<LoadInst>(StoredVal)) {
    const SCEVAddRecExpr *LoadEv =
      dyn_cast<SCEVAddRecExpr>(SE->getSCEV(LI->getOperand(0)));
    if (LoadEv && LoadEv->getLoop() == CurLoop && LoadEv->isAffine() &&
        StoreEv->getOperand(1) == LoadEv->getOperand(1) && LI->isSimple())
      if (processLoopStoreOfLoopLoad(SI, StoreSize, StoreEv, LoadEv, BECount))
        return true;
  }
  //errs() << "UNHANDLED strided store: " << *StoreEv << " - " << *SI << "\n";

  return false;
}

/// processLoopMemSet - See if this memset can be promoted to a large memset.
bool LoopIdiomRecognize::
processLoopMemSet(MemSetInst *MSI, const SCEV *BECount) {
  // We can only handle non-volatile memsets with a constant size.
  if (MSI->isVolatile() || !isa<ConstantInt>(MSI->getLength())) return false;

  // If we're not allowed to hack on memset, we fail.
  if (!TLI->has(LibFunc::memset))
    return false;

  Value *Pointer = MSI->getDest();

  // See if the pointer expression is an AddRec like {base,+,1} on the current
  // loop, which indicates a strided store.  If we have something else, it's a
  // random store we can't handle.
  const SCEVAddRecExpr *Ev = dyn_cast<SCEVAddRecExpr>(SE->getSCEV(Pointer));
  if (Ev == 0 || Ev->getLoop() != CurLoop || !Ev->isAffine())
    return false;

  // Reject memsets that are so large that they overflow an unsigned.
  uint64_t SizeInBytes = cast<ConstantInt>(MSI->getLength())->getZExtValue();
  if ((SizeInBytes >> 32) != 0)
    return false;

  // Check to see if the stride matches the size of the memset.  If so, then we
  // know that every byte is touched in the loop.
  const SCEVConstant *Stride = dyn_cast<SCEVConstant>(Ev->getOperand(1));

  // TODO: Could also handle negative stride here someday, that will require the
  // validity check in mayLoopAccessLocation to be updated though.
  if (Stride == 0 || MSI->getLength() != Stride->getValue())
    return false;

  return processLoopStridedStore(Pointer, (unsigned)SizeInBytes,
                                 MSI->getAlignment(), MSI->getValue(),
                                 MSI, Ev, BECount);
}


/// mayLoopAccessLocation - Return true if the specified loop might access the
/// specified pointer location, which is a loop-strided access.  The 'Access'
/// argument specifies what the verboten forms of access are (read or write).
static bool mayLoopAccessLocation(Value *Ptr,AliasAnalysis::ModRefResult Access,
                                  Loop *L, const SCEV *BECount,
                                  unsigned StoreSize, AliasAnalysis &AA,
                                  Instruction *IgnoredStore) {
  // Get the location that may be stored across the loop.  Since the access is
  // strided positively through memory, we say that the modified location starts
  // at the pointer and has infinite size.
  uint64_t AccessSize = AliasAnalysis::UnknownSize;

  // If the loop iterates a fixed number of times, we can refine the access size
  // to be exactly the size of the memset, which is (BECount+1)*StoreSize
  if (const SCEVConstant *BECst = dyn_cast<SCEVConstant>(BECount))
    AccessSize = (BECst->getValue()->getZExtValue()+1)*StoreSize;

  // TODO: For this to be really effective, we have to dive into the pointer
  // operand in the store.  Store to &A[i] of 100 will always return may alias
  // with store of &A[100], we need to StoreLoc to be "A" with size of 100,
  // which will then no-alias a store to &A[100].
  AliasAnalysis::Location StoreLoc(Ptr, AccessSize);

  for (Loop::block_iterator BI = L->block_begin(), E = L->block_end(); BI != E;
       ++BI)
    for (BasicBlock::iterator I = (*BI)->begin(), E = (*BI)->end(); I != E; ++I)
      if (&*I != IgnoredStore &&
          (AA.getModRefInfo(I, StoreLoc) & Access))
        return true;

  return false;
}

/// getMemSetPatternValue - If a strided store of the specified value is safe to
/// turn into a memset_pattern16, return a ConstantArray of 16 bytes that should
/// be passed in.  Otherwise, return null.
///
/// Note that we don't ever attempt to use memset_pattern8 or 4, because these
/// just replicate their input array and then pass on to memset_pattern16.
static Constant *getMemSetPatternValue(Value *V, const DataLayout &TD) {
  // If the value isn't a constant, we can't promote it to being in a constant
  // array.  We could theoretically do a store to an alloca or something, but
  // that doesn't seem worthwhile.
  Constant *C = dyn_cast<Constant>(V);
  if (C == 0) return 0;

  // Only handle simple values that are a power of two bytes in size.
  uint64_t Size = TD.getTypeSizeInBits(V->getType());
  if (Size == 0 || (Size & 7) || (Size & (Size-1)))
    return 0;

  // Don't care enough about darwin/ppc to implement this.
  if (TD.isBigEndian())
    return 0;

  // Convert to size in bytes.
  Size /= 8;

  // TODO: If CI is larger than 16-bytes, we can try slicing it in half to see
  // if the top and bottom are the same (e.g. for vectors and large integers).
  if (Size > 16) return 0;

  // If the constant is exactly 16 bytes, just use it.
  if (Size == 16) return C;

  // Otherwise, we'll use an array of the constants.
  unsigned ArraySize = 16/Size;
  ArrayType *AT = ArrayType::get(V->getType(), ArraySize);
  return ConstantArray::get(AT, std::vector<Constant*>(ArraySize, C));
}


/// processLoopStridedStore - We see a strided store of some value.  If we can
/// transform this into a memset or memset_pattern in the loop preheader, do so.
bool LoopIdiomRecognize::
processLoopStridedStore(Value *DestPtr, unsigned StoreSize,
                        unsigned StoreAlignment, Value *StoredVal,
                        Instruction *TheStore, const SCEVAddRecExpr *Ev,
                        const SCEV *BECount) {

  // If the stored value is a byte-wise value (like i32 -1), then it may be
  // turned into a memset of i8 -1, assuming that all the consecutive bytes
  // are stored.  A store of i32 0x01020304 can never be turned into a memset,
  // but it can be turned into memset_pattern if the target supports it.
  Value *SplatValue = isBytewiseValue(StoredVal);
  Constant *PatternValue = 0;

  // If we're allowed to form a memset, and the stored value would be acceptable
  // for memset, use it.
  if (SplatValue && TLI->has(LibFunc::memset) &&
      // Verify that the stored value is loop invariant.  If not, we can't
      // promote the memset.
      CurLoop->isLoopInvariant(SplatValue)) {
    // Keep and use SplatValue.
    PatternValue = 0;
  } else if (TLI->has(LibFunc::memset_pattern16) &&
             (PatternValue = getMemSetPatternValue(StoredVal, *TD))) {
    // It looks like we can use PatternValue!
    SplatValue = 0;
  } else {
    // Otherwise, this isn't an idiom we can transform.  For example, we can't
    // do anything with a 3-byte store.
    return false;
  }

  // The trip count of the loop and the base pointer of the addrec SCEV is
  // guaranteed to be loop invariant, which means that it should dominate the
  // header.  This allows us to insert code for it in the preheader.
  BasicBlock *Preheader = CurLoop->getLoopPreheader();
  IRBuilder<> Builder(Preheader->getTerminator());
  SCEVExpander Expander(*SE, "loop-idiom");

  // Okay, we have a strided store "p[i]" of a splattable value.  We can turn
  // this into a memset in the loop preheader now if we want.  However, this
  // would be unsafe to do if there is anything else in the loop that may read
  // or write to the aliased location.  Check for any overlap by generating the
  // base pointer and checking the region.
  unsigned AddrSpace = cast<PointerType>(DestPtr->getType())->getAddressSpace();
  Value *BasePtr =
    Expander.expandCodeFor(Ev->getStart(), Builder.getInt8PtrTy(AddrSpace),
                           Preheader->getTerminator());


  if (mayLoopAccessLocation(BasePtr, AliasAnalysis::ModRef,
                            CurLoop, BECount,
                            StoreSize, getAnalysis<AliasAnalysis>(), TheStore)){
    Expander.clear();
    // If we generated new code for the base pointer, clean up.
    deleteIfDeadInstruction(BasePtr, *SE, TLI);
    return false;
  }

  // Okay, everything looks good, insert the memset.

  // The # stored bytes is (BECount+1)*Size.  Expand the trip count out to
  // pointer size if it isn't already.
  Type *IntPtr = TD->getIntPtrType(DestPtr->getContext());
  BECount = SE->getTruncateOrZeroExtend(BECount, IntPtr);

  const SCEV *NumBytesS = SE->getAddExpr(BECount, SE->getConstant(IntPtr, 1),
                                         SCEV::FlagNUW);
  if (StoreSize != 1)
    NumBytesS = SE->getMulExpr(NumBytesS, SE->getConstant(IntPtr, StoreSize),
                               SCEV::FlagNUW);

  Value *NumBytes =
    Expander.expandCodeFor(NumBytesS, IntPtr, Preheader->getTerminator());

  CallInst *NewCall;
  if (SplatValue)
    NewCall = Builder.CreateMemSet(BasePtr, SplatValue,NumBytes,StoreAlignment);
  else {
    Module *M = TheStore->getParent()->getParent()->getParent();
    Value *MSP = M->getOrInsertFunction("memset_pattern16",
                                        Builder.getVoidTy(),
                                        Builder.getInt8PtrTy(),
                                        Builder.getInt8PtrTy(), IntPtr,
                                        (void*)0);

    // Otherwise we should form a memset_pattern16.  PatternValue is known to be
    // an constant array of 16-bytes.  Plop the value into a mergable global.
    GlobalVariable *GV = new GlobalVariable(*M, PatternValue->getType(), true,
                                            GlobalValue::InternalLinkage,
                                            PatternValue, ".memset_pattern");
    GV->setUnnamedAddr(true); // Ok to merge these.
    GV->setAlignment(16);
    Value *PatternPtr = ConstantExpr::getBitCast(GV, Builder.getInt8PtrTy());
    NewCall = Builder.CreateCall3(MSP, BasePtr, PatternPtr, NumBytes);
  }

  DEBUG(dbgs() << "  Formed memset: " << *NewCall << "\n"
               << "    from store to: " << *Ev << " at: " << *TheStore << "\n");
  NewCall->setDebugLoc(TheStore->getDebugLoc());

  // Okay, the memset has been formed.  Zap the original store and anything that
  // feeds into it.
  deleteDeadInstruction(TheStore, *SE, TLI);
  ++NumMemSet;
  return true;
}

/// processLoopStoreOfLoopLoad - We see a strided store whose value is a
/// same-strided load.
bool LoopIdiomRecognize::
processLoopStoreOfLoopLoad(StoreInst *SI, unsigned StoreSize,
                           const SCEVAddRecExpr *StoreEv,
                           const SCEVAddRecExpr *LoadEv,
                           const SCEV *BECount) {
  // If we're not allowed to form memcpy, we fail.
  if (!TLI->has(LibFunc::memcpy))
    return false;

  LoadInst *LI = cast<LoadInst>(SI->getValueOperand());

  // The trip count of the loop and the base pointer of the addrec SCEV is
  // guaranteed to be loop invariant, which means that it should dominate the
  // header.  This allows us to insert code for it in the preheader.
  BasicBlock *Preheader = CurLoop->getLoopPreheader();
  IRBuilder<> Builder(Preheader->getTerminator());
  SCEVExpander Expander(*SE, "loop-idiom");

  // Okay, we have a strided store "p[i]" of a loaded value.  We can turn
  // this into a memcpy in the loop preheader now if we want.  However, this
  // would be unsafe to do if there is anything else in the loop that may read
  // or write the memory region we're storing to.  This includes the load that
  // feeds the stores.  Check for an alias by generating the base address and
  // checking everything.
  Value *StoreBasePtr =
    Expander.expandCodeFor(StoreEv->getStart(),
                           Builder.getInt8PtrTy(SI->getPointerAddressSpace()),
                           Preheader->getTerminator());

  if (mayLoopAccessLocation(StoreBasePtr, AliasAnalysis::ModRef,
                            CurLoop, BECount, StoreSize,
                            getAnalysis<AliasAnalysis>(), SI)) {
    Expander.clear();
    // If we generated new code for the base pointer, clean up.
    deleteIfDeadInstruction(StoreBasePtr, *SE, TLI);
    return false;
  }

  // For a memcpy, we have to make sure that the input array is not being
  // mutated by the loop.
  Value *LoadBasePtr =
    Expander.expandCodeFor(LoadEv->getStart(),
                           Builder.getInt8PtrTy(LI->getPointerAddressSpace()),
                           Preheader->getTerminator());

  if (mayLoopAccessLocation(LoadBasePtr, AliasAnalysis::Mod, CurLoop, BECount,
                            StoreSize, getAnalysis<AliasAnalysis>(), SI)) {
    Expander.clear();
    // If we generated new code for the base pointer, clean up.
    deleteIfDeadInstruction(LoadBasePtr, *SE, TLI);
    deleteIfDeadInstruction(StoreBasePtr, *SE, TLI);
    return false;
  }

  // Okay, everything is safe, we can transform this!


  // The # stored bytes is (BECount+1)*Size.  Expand the trip count out to
  // pointer size if it isn't already.
  Type *IntPtr = TD->getIntPtrType(SI->getContext());
  BECount = SE->getTruncateOrZeroExtend(BECount, IntPtr);

  const SCEV *NumBytesS = SE->getAddExpr(BECount, SE->getConstant(IntPtr, 1),
                                         SCEV::FlagNUW);
  if (StoreSize != 1)
    NumBytesS = SE->getMulExpr(NumBytesS, SE->getConstant(IntPtr, StoreSize),
                               SCEV::FlagNUW);

  Value *NumBytes =
    Expander.expandCodeFor(NumBytesS, IntPtr, Preheader->getTerminator());

  CallInst *NewCall =
    Builder.CreateMemCpy(StoreBasePtr, LoadBasePtr, NumBytes,
                         std::min(SI->getAlignment(), LI->getAlignment()));
  NewCall->setDebugLoc(SI->getDebugLoc());

  DEBUG(dbgs() << "  Formed memcpy: " << *NewCall << "\n"
               << "    from load ptr=" << *LoadEv << " at: " << *LI << "\n"
               << "    from store ptr=" << *StoreEv << " at: " << *SI << "\n");


  // Okay, the memset has been formed.  Zap the original store and anything that
  // feeds into it.
  deleteDeadInstruction(SI, *SE, TLI);
  ++NumMemCpy;
  return true;
}
