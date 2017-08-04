//===----- LoadStoreVectorizer.cpp - GPU Load & Store Vectorizer ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/OrderedBasicBlock.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Vectorize.h"

using namespace llvm;

#define DEBUG_TYPE "load-store-vectorizer"
STATISTIC(NumVectorInstructions, "Number of vector accesses generated");
STATISTIC(NumScalarsVectorized, "Number of scalar accesses vectorized");

namespace {

// FIXME: Assuming stack alignment of 4 is always good enough
static const unsigned StackAdjustedAlignment = 4;
typedef SmallVector<Instruction *, 8> InstrList;
typedef MapVector<Value *, InstrList> InstrListMap;

class Vectorizer {
  Function &F;
  AliasAnalysis &AA;
  DominatorTree &DT;
  ScalarEvolution &SE;
  TargetTransformInfo &TTI;
  const DataLayout &DL;
  IRBuilder<> Builder;

public:
  Vectorizer(Function &F, AliasAnalysis &AA, DominatorTree &DT,
             ScalarEvolution &SE, TargetTransformInfo &TTI)
      : F(F), AA(AA), DT(DT), SE(SE), TTI(TTI),
        DL(F.getParent()->getDataLayout()), Builder(SE.getContext()) {}

  bool run();

private:
  Value *getPointerOperand(Value *I) const;

  GetElementPtrInst *getSourceGEP(Value *Src) const;

  unsigned getPointerAddressSpace(Value *I);

  unsigned getAlignment(LoadInst *LI) const {
    unsigned Align = LI->getAlignment();
    if (Align != 0)
      return Align;

    return DL.getABITypeAlignment(LI->getType());
  }

  unsigned getAlignment(StoreInst *SI) const {
    unsigned Align = SI->getAlignment();
    if (Align != 0)
      return Align;

    return DL.getABITypeAlignment(SI->getValueOperand()->getType());
  }

  bool isConsecutiveAccess(Value *A, Value *B);

  /// After vectorization, reorder the instructions that I depends on
  /// (the instructions defining its operands), to ensure they dominate I.
  void reorder(Instruction *I);

  /// Returns the first and the last instructions in Chain.
  std::pair<BasicBlock::iterator, BasicBlock::iterator>
  getBoundaryInstrs(ArrayRef<Instruction *> Chain);

  /// Erases the original instructions after vectorizing.
  void eraseInstructions(ArrayRef<Instruction *> Chain);

  /// "Legalize" the vector type that would be produced by combining \p
  /// ElementSizeBits elements in \p Chain. Break into two pieces such that the
  /// total size of each piece is 1, 2 or a multiple of 4 bytes. \p Chain is
  /// expected to have more than 4 elements.
  std::pair<ArrayRef<Instruction *>, ArrayRef<Instruction *>>
  splitOddVectorElts(ArrayRef<Instruction *> Chain, unsigned ElementSizeBits);

  /// Finds the largest prefix of Chain that's vectorizable, checking for
  /// intervening instructions which may affect the memory accessed by the
  /// instructions within Chain.
  ///
  /// The elements of \p Chain must be all loads or all stores and must be in
  /// address order.
  ArrayRef<Instruction *> getVectorizablePrefix(ArrayRef<Instruction *> Chain);

  /// Collects load and store instructions to vectorize.
  std::pair<InstrListMap, InstrListMap> collectInstructions(BasicBlock *BB);

  /// Processes the collected instructions, the \p Map. The values of \p Map
  /// should be all loads or all stores.
  bool vectorizeChains(InstrListMap &Map);

  /// Finds the load/stores to consecutive memory addresses and vectorizes them.
  bool vectorizeInstructions(ArrayRef<Instruction *> Instrs);

  /// Vectorizes the load instructions in Chain.
  bool
  vectorizeLoadChain(ArrayRef<Instruction *> Chain,
                     SmallPtrSet<Instruction *, 16> *InstructionsProcessed);

  /// Vectorizes the store instructions in Chain.
  bool
  vectorizeStoreChain(ArrayRef<Instruction *> Chain,
                      SmallPtrSet<Instruction *, 16> *InstructionsProcessed);

  /// Check if this load/store access is misaligned accesses.
  bool accessIsMisaligned(unsigned SzInBytes, unsigned AddressSpace,
                          unsigned Alignment);
};

class LoadStoreVectorizer : public FunctionPass {
public:
  static char ID;

  LoadStoreVectorizer() : FunctionPass(ID) {
    initializeLoadStoreVectorizerPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override {
    return "GPU Load and Store Vectorizer";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.setPreservesCFG();
  }
};
}

INITIALIZE_PASS_BEGIN(LoadStoreVectorizer, DEBUG_TYPE,
                      "Vectorize load and Store instructions", false, false)
INITIALIZE_PASS_DEPENDENCY(SCEVAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(GlobalsAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(LoadStoreVectorizer, DEBUG_TYPE,
                    "Vectorize load and store instructions", false, false)

char LoadStoreVectorizer::ID = 0;

Pass *llvm::createLoadStoreVectorizerPass() {
  return new LoadStoreVectorizer();
}

// The real propagateMetadata expects a SmallVector<Value*>, but we deal in
// vectors of Instructions.
static void propagateMetadata(Instruction *I, ArrayRef<Instruction *> IL) {
  SmallVector<Value *, 8> VL(IL.begin(), IL.end());
  propagateMetadata(I, VL);
}

bool LoadStoreVectorizer::runOnFunction(Function &F) {
  // Don't vectorize when the attribute NoImplicitFloat is used.
  if (skipFunction(F) || F.hasFnAttribute(Attribute::NoImplicitFloat))
    return false;

  AliasAnalysis &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  ScalarEvolution &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  TargetTransformInfo &TTI =
      getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);

  Vectorizer V(F, AA, DT, SE, TTI);
  return V.run();
}

// Vectorizer Implementation
bool Vectorizer::run() {
  bool Changed = false;

  // Scan the blocks in the function in post order.
  for (BasicBlock *BB : post_order(&F)) {
    InstrListMap LoadRefs, StoreRefs;
    std::tie(LoadRefs, StoreRefs) = collectInstructions(BB);
    Changed |= vectorizeChains(LoadRefs);
    Changed |= vectorizeChains(StoreRefs);
  }

  return Changed;
}

Value *Vectorizer::getPointerOperand(Value *I) const {
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return LI->getPointerOperand();
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return SI->getPointerOperand();
  return nullptr;
}

unsigned Vectorizer::getPointerAddressSpace(Value *I) {
  if (LoadInst *L = dyn_cast<LoadInst>(I))
    return L->getPointerAddressSpace();
  if (StoreInst *S = dyn_cast<StoreInst>(I))
    return S->getPointerAddressSpace();
  return -1;
}

GetElementPtrInst *Vectorizer::getSourceGEP(Value *Src) const {
  // First strip pointer bitcasts. Make sure pointee size is the same with
  // and without casts.
  // TODO: a stride set by the add instruction below can match the difference
  // in pointee type size here. Currently it will not be vectorized.
  Value *SrcPtr = getPointerOperand(Src);
  Value *SrcBase = SrcPtr->stripPointerCasts();
  if (DL.getTypeStoreSize(SrcPtr->getType()->getPointerElementType()) ==
      DL.getTypeStoreSize(SrcBase->getType()->getPointerElementType()))
    SrcPtr = SrcBase;
  return dyn_cast<GetElementPtrInst>(SrcPtr);
}

// FIXME: Merge with llvm::isConsecutiveAccess
bool Vectorizer::isConsecutiveAccess(Value *A, Value *B) {
  Value *PtrA = getPointerOperand(A);
  Value *PtrB = getPointerOperand(B);
  unsigned ASA = getPointerAddressSpace(A);
  unsigned ASB = getPointerAddressSpace(B);

  // Check that the address spaces match and that the pointers are valid.
  if (!PtrA || !PtrB || (ASA != ASB))
    return false;

  // Make sure that A and B are different pointers of the same size type.
  unsigned PtrBitWidth = DL.getPointerSizeInBits(ASA);
  Type *PtrATy = PtrA->getType()->getPointerElementType();
  Type *PtrBTy = PtrB->getType()->getPointerElementType();
  if (PtrA == PtrB ||
      DL.getTypeStoreSize(PtrATy) != DL.getTypeStoreSize(PtrBTy) ||
      DL.getTypeStoreSize(PtrATy->getScalarType()) !=
          DL.getTypeStoreSize(PtrBTy->getScalarType()))
    return false;

  APInt Size(PtrBitWidth, DL.getTypeStoreSize(PtrATy));

  APInt OffsetA(PtrBitWidth, 0), OffsetB(PtrBitWidth, 0);
  PtrA = PtrA->stripAndAccumulateInBoundsConstantOffsets(DL, OffsetA);
  PtrB = PtrB->stripAndAccumulateInBoundsConstantOffsets(DL, OffsetB);

  APInt OffsetDelta = OffsetB - OffsetA;

  // Check if they are based on the same pointer. That makes the offsets
  // sufficient.
  if (PtrA == PtrB)
    return OffsetDelta == Size;

  // Compute the necessary base pointer delta to have the necessary final delta
  // equal to the size.
  APInt BaseDelta = Size - OffsetDelta;

  // Compute the distance with SCEV between the base pointers.
  const SCEV *PtrSCEVA = SE.getSCEV(PtrA);
  const SCEV *PtrSCEVB = SE.getSCEV(PtrB);
  const SCEV *C = SE.getConstant(BaseDelta);
  const SCEV *X = SE.getAddExpr(PtrSCEVA, C);
  if (X == PtrSCEVB)
    return true;

  // Sometimes even this doesn't work, because SCEV can't always see through
  // patterns that look like (gep (ext (add (shl X, C1), C2))). Try checking
  // things the hard way.

  // Look through GEPs after checking they're the same except for the last
  // index.
  GetElementPtrInst *GEPA = getSourceGEP(A);
  GetElementPtrInst *GEPB = getSourceGEP(B);
  if (!GEPA || !GEPB || GEPA->getNumOperands() != GEPB->getNumOperands())
    return false;
  unsigned FinalIndex = GEPA->getNumOperands() - 1;
  for (unsigned i = 0; i < FinalIndex; i++)
    if (GEPA->getOperand(i) != GEPB->getOperand(i))
      return false;

  Instruction *OpA = dyn_cast<Instruction>(GEPA->getOperand(FinalIndex));
  Instruction *OpB = dyn_cast<Instruction>(GEPB->getOperand(FinalIndex));
  if (!OpA || !OpB || OpA->getOpcode() != OpB->getOpcode() ||
      OpA->getType() != OpB->getType())
    return false;

  // Only look through a ZExt/SExt.
  if (!isa<SExtInst>(OpA) && !isa<ZExtInst>(OpA))
    return false;

  bool Signed = isa<SExtInst>(OpA);

  OpA = dyn_cast<Instruction>(OpA->getOperand(0));
  OpB = dyn_cast<Instruction>(OpB->getOperand(0));
  if (!OpA || !OpB || OpA->getType() != OpB->getType())
    return false;

  // Now we need to prove that adding 1 to OpA won't overflow.
  bool Safe = false;
  // First attempt: if OpB is an add with NSW/NUW, and OpB is 1 added to OpA,
  // we're okay.
  if (OpB->getOpcode() == Instruction::Add &&
      isa<ConstantInt>(OpB->getOperand(1)) &&
      cast<ConstantInt>(OpB->getOperand(1))->getSExtValue() > 0) {
    if (Signed)
      Safe = cast<BinaryOperator>(OpB)->hasNoSignedWrap();
    else
      Safe = cast<BinaryOperator>(OpB)->hasNoUnsignedWrap();
  }

  unsigned BitWidth = OpA->getType()->getScalarSizeInBits();

  // Second attempt:
  // If any bits are known to be zero other than the sign bit in OpA, we can
  // add 1 to it while guaranteeing no overflow of any sort.
  if (!Safe) {
    KnownBits Known(BitWidth);
    computeKnownBits(OpA, Known, DL, 0, nullptr, OpA, &DT);
    if (Known.countMaxTrailingOnes() < (BitWidth - 1))
      Safe = true;
  }

  if (!Safe)
    return false;

  const SCEV *OffsetSCEVA = SE.getSCEV(OpA);
  const SCEV *OffsetSCEVB = SE.getSCEV(OpB);
  const SCEV *One = SE.getConstant(APInt(BitWidth, 1));
  const SCEV *X2 = SE.getAddExpr(OffsetSCEVA, One);
  return X2 == OffsetSCEVB;
}

void Vectorizer::reorder(Instruction *I) {
  OrderedBasicBlock OBB(I->getParent());
  SmallPtrSet<Instruction *, 16> InstructionsToMove;
  SmallVector<Instruction *, 16> Worklist;

  Worklist.push_back(I);
  while (!Worklist.empty()) {
    Instruction *IW = Worklist.pop_back_val();
    int NumOperands = IW->getNumOperands();
    for (int i = 0; i < NumOperands; i++) {
      Instruction *IM = dyn_cast<Instruction>(IW->getOperand(i));
      if (!IM || IM->getOpcode() == Instruction::PHI)
        continue;

      // If IM is in another BB, no need to move it, because this pass only
      // vectorizes instructions within one BB.
      if (IM->getParent() != I->getParent())
        continue;

      if (!OBB.dominates(IM, I)) {
        InstructionsToMove.insert(IM);
        Worklist.push_back(IM);
      }
    }
  }

  // All instructions to move should follow I. Start from I, not from begin().
  for (auto BBI = I->getIterator(), E = I->getParent()->end(); BBI != E;
       ++BBI) {
    if (!InstructionsToMove.count(&*BBI))
      continue;
    Instruction *IM = &*BBI;
    --BBI;
    IM->removeFromParent();
    IM->insertBefore(I);
  }
}

std::pair<BasicBlock::iterator, BasicBlock::iterator>
Vectorizer::getBoundaryInstrs(ArrayRef<Instruction *> Chain) {
  Instruction *C0 = Chain[0];
  BasicBlock::iterator FirstInstr = C0->getIterator();
  BasicBlock::iterator LastInstr = C0->getIterator();

  BasicBlock *BB = C0->getParent();
  unsigned NumFound = 0;
  for (Instruction &I : *BB) {
    if (!is_contained(Chain, &I))
      continue;

    ++NumFound;
    if (NumFound == 1) {
      FirstInstr = I.getIterator();
    }
    if (NumFound == Chain.size()) {
      LastInstr = I.getIterator();
      break;
    }
  }

  // Range is [first, last).
  return std::make_pair(FirstInstr, ++LastInstr);
}

void Vectorizer::eraseInstructions(ArrayRef<Instruction *> Chain) {
  SmallVector<Instruction *, 16> Instrs;
  for (Instruction *I : Chain) {
    Value *PtrOperand = getPointerOperand(I);
    assert(PtrOperand && "Instruction must have a pointer operand.");
    Instrs.push_back(I);
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(PtrOperand))
      Instrs.push_back(GEP);
  }

  // Erase instructions.
  for (Instruction *I : Instrs)
    if (I->use_empty())
      I->eraseFromParent();
}

std::pair<ArrayRef<Instruction *>, ArrayRef<Instruction *>>
Vectorizer::splitOddVectorElts(ArrayRef<Instruction *> Chain,
                               unsigned ElementSizeBits) {
  unsigned ElementSizeBytes = ElementSizeBits / 8;
  unsigned SizeBytes = ElementSizeBytes * Chain.size();
  unsigned NumLeft = (SizeBytes - (SizeBytes % 4)) / ElementSizeBytes;
  if (NumLeft == Chain.size()) {
    if ((NumLeft & 1) == 0)
      NumLeft /= 2; // Split even in half
    else
      --NumLeft;    // Split off last element
  } else if (NumLeft == 0)
    NumLeft = 1;
  return std::make_pair(Chain.slice(0, NumLeft), Chain.slice(NumLeft));
}

ArrayRef<Instruction *>
Vectorizer::getVectorizablePrefix(ArrayRef<Instruction *> Chain) {
  // These are in BB order, unlike Chain, which is in address order.
  SmallVector<Instruction *, 16> MemoryInstrs;
  SmallVector<Instruction *, 16> ChainInstrs;

  bool IsLoadChain = isa<LoadInst>(Chain[0]);
  DEBUG({
    for (Instruction *I : Chain) {
      if (IsLoadChain)
        assert(isa<LoadInst>(I) &&
               "All elements of Chain must be loads, or all must be stores.");
      else
        assert(isa<StoreInst>(I) &&
               "All elements of Chain must be loads, or all must be stores.");
    }
  });

  for (Instruction &I : make_range(getBoundaryInstrs(Chain))) {
    if (isa<LoadInst>(I) || isa<StoreInst>(I)) {
      if (!is_contained(Chain, &I))
        MemoryInstrs.push_back(&I);
      else
        ChainInstrs.push_back(&I);
    } else if (IsLoadChain && (I.mayWriteToMemory() || I.mayThrow())) {
      DEBUG(dbgs() << "LSV: Found may-write/throw operation: " << I << '\n');
      break;
    } else if (!IsLoadChain && (I.mayReadOrWriteMemory() || I.mayThrow())) {
      DEBUG(dbgs() << "LSV: Found may-read/write/throw operation: " << I
                   << '\n');
      break;
    }
  }

  OrderedBasicBlock OBB(Chain[0]->getParent());

  // Loop until we find an instruction in ChainInstrs that we can't vectorize.
  unsigned ChainInstrIdx = 0;
  Instruction *BarrierMemoryInstr = nullptr;

  for (unsigned E = ChainInstrs.size(); ChainInstrIdx < E; ++ChainInstrIdx) {
    Instruction *ChainInstr = ChainInstrs[ChainInstrIdx];

    // If a barrier memory instruction was found, chain instructions that follow
    // will not be added to the valid prefix.
    if (BarrierMemoryInstr && OBB.dominates(BarrierMemoryInstr, ChainInstr))
      break;

    // Check (in BB order) if any instruction prevents ChainInstr from being
    // vectorized. Find and store the first such "conflicting" instruction.
    for (Instruction *MemInstr : MemoryInstrs) {
      // If a barrier memory instruction was found, do not check past it.
      if (BarrierMemoryInstr && OBB.dominates(BarrierMemoryInstr, MemInstr))
        break;

      if (isa<LoadInst>(MemInstr) && isa<LoadInst>(ChainInstr))
        continue;

      // We can ignore the alias as long as the load comes before the store,
      // because that means we won't be moving the load past the store to
      // vectorize it (the vectorized load is inserted at the location of the
      // first load in the chain).
      if (isa<StoreInst>(MemInstr) && isa<LoadInst>(ChainInstr) &&
          OBB.dominates(ChainInstr, MemInstr))
        continue;

      // Same case, but in reverse.
      if (isa<LoadInst>(MemInstr) && isa<StoreInst>(ChainInstr) &&
          OBB.dominates(MemInstr, ChainInstr))
        continue;

      if (!AA.isNoAlias(MemoryLocation::get(MemInstr),
                        MemoryLocation::get(ChainInstr))) {
        DEBUG({
          dbgs() << "LSV: Found alias:\n"
                    "  Aliasing instruction and pointer:\n"
                 << "  " << *MemInstr << '\n'
                 << "  " << *getPointerOperand(MemInstr) << '\n'
                 << "  Aliased instruction and pointer:\n"
                 << "  " << *ChainInstr << '\n'
                 << "  " << *getPointerOperand(ChainInstr) << '\n';
        });
        // Save this aliasing memory instruction as a barrier, but allow other
        // instructions that precede the barrier to be vectorized with this one.
        BarrierMemoryInstr = MemInstr;
        break;
      }
    }
    // Continue the search only for store chains, since vectorizing stores that
    // precede an aliasing load is valid. Conversely, vectorizing loads is valid
    // up to an aliasing store, but should not pull loads from further down in
    // the basic block.
    if (IsLoadChain && BarrierMemoryInstr) {
      // The BarrierMemoryInstr is a store that precedes ChainInstr.
      assert(OBB.dominates(BarrierMemoryInstr, ChainInstr));
      break;
    }
  }

  // Find the largest prefix of Chain whose elements are all in
  // ChainInstrs[0, ChainInstrIdx).  This is the largest vectorizable prefix of
  // Chain.  (Recall that Chain is in address order, but ChainInstrs is in BB
  // order.)
  SmallPtrSet<Instruction *, 8> VectorizableChainInstrs(
      ChainInstrs.begin(), ChainInstrs.begin() + ChainInstrIdx);
  unsigned ChainIdx = 0;
  for (unsigned ChainLen = Chain.size(); ChainIdx < ChainLen; ++ChainIdx) {
    if (!VectorizableChainInstrs.count(Chain[ChainIdx]))
      break;
  }
  return Chain.slice(0, ChainIdx);
}

std::pair<InstrListMap, InstrListMap>
Vectorizer::collectInstructions(BasicBlock *BB) {
  InstrListMap LoadRefs;
  InstrListMap StoreRefs;

  for (Instruction &I : *BB) {
    if (!I.mayReadOrWriteMemory())
      continue;

    if (LoadInst *LI = dyn_cast<LoadInst>(&I)) {
      if (!LI->isSimple())
        continue;

      // Skip if it's not legal.
      if (!TTI.isLegalToVectorizeLoad(LI))
        continue;

      Type *Ty = LI->getType();
      if (!VectorType::isValidElementType(Ty->getScalarType()))
        continue;

      // Skip weird non-byte sizes. They probably aren't worth the effort of
      // handling correctly.
      unsigned TySize = DL.getTypeSizeInBits(Ty);
      if (TySize < 8)
        continue;

      Value *Ptr = LI->getPointerOperand();
      unsigned AS = Ptr->getType()->getPointerAddressSpace();
      unsigned VecRegSize = TTI.getLoadStoreVecRegBitWidth(AS);

      // No point in looking at these if they're too big to vectorize.
      if (TySize > VecRegSize / 2)
        continue;

      // Make sure all the users of a vector are constant-index extracts.
      if (isa<VectorType>(Ty) && !all_of(LI->users(), [](const User *U) {
            const ExtractElementInst *EEI = dyn_cast<ExtractElementInst>(U);
            return EEI && isa<ConstantInt>(EEI->getOperand(1));
          }))
        continue;

      // Save the load locations.
      Value *ObjPtr = GetUnderlyingObject(Ptr, DL);
      LoadRefs[ObjPtr].push_back(LI);

    } else if (StoreInst *SI = dyn_cast<StoreInst>(&I)) {
      if (!SI->isSimple())
        continue;

      // Skip if it's not legal.
      if (!TTI.isLegalToVectorizeStore(SI))
        continue;

      Type *Ty = SI->getValueOperand()->getType();
      if (!VectorType::isValidElementType(Ty->getScalarType()))
        continue;

      // Skip weird non-byte sizes. They probably aren't worth the effort of
      // handling correctly.
      unsigned TySize = DL.getTypeSizeInBits(Ty);
      if (TySize < 8)
        continue;

      Value *Ptr = SI->getPointerOperand();
      unsigned AS = Ptr->getType()->getPointerAddressSpace();
      unsigned VecRegSize = TTI.getLoadStoreVecRegBitWidth(AS);
      if (TySize > VecRegSize / 2)
        continue;

      if (isa<VectorType>(Ty) && !all_of(SI->users(), [](const User *U) {
            const ExtractElementInst *EEI = dyn_cast<ExtractElementInst>(U);
            return EEI && isa<ConstantInt>(EEI->getOperand(1));
          }))
        continue;

      // Save store location.
      Value *ObjPtr = GetUnderlyingObject(Ptr, DL);
      StoreRefs[ObjPtr].push_back(SI);
    }
  }

  return {LoadRefs, StoreRefs};
}

bool Vectorizer::vectorizeChains(InstrListMap &Map) {
  bool Changed = false;

  for (const std::pair<Value *, InstrList> &Chain : Map) {
    unsigned Size = Chain.second.size();
    if (Size < 2)
      continue;

    DEBUG(dbgs() << "LSV: Analyzing a chain of length " << Size << ".\n");

    // Process the stores in chunks of 64.
    for (unsigned CI = 0, CE = Size; CI < CE; CI += 64) {
      unsigned Len = std::min<unsigned>(CE - CI, 64);
      ArrayRef<Instruction *> Chunk(&Chain.second[CI], Len);
      Changed |= vectorizeInstructions(Chunk);
    }
  }

  return Changed;
}

bool Vectorizer::vectorizeInstructions(ArrayRef<Instruction *> Instrs) {
  DEBUG(dbgs() << "LSV: Vectorizing " << Instrs.size() << " instructions.\n");
  SmallVector<int, 16> Heads, Tails;
  int ConsecutiveChain[64];

  // Do a quadratic search on all of the given stores and find all of the pairs
  // of stores that follow each other.
  for (int i = 0, e = Instrs.size(); i < e; ++i) {
    ConsecutiveChain[i] = -1;
    for (int j = e - 1; j >= 0; --j) {
      if (i == j)
        continue;

      if (isConsecutiveAccess(Instrs[i], Instrs[j])) {
        if (ConsecutiveChain[i] != -1) {
          int CurDistance = std::abs(ConsecutiveChain[i] - i);
          int NewDistance = std::abs(ConsecutiveChain[i] - j);
          if (j < i || NewDistance > CurDistance)
            continue; // Should not insert.
        }

        Tails.push_back(j);
        Heads.push_back(i);
        ConsecutiveChain[i] = j;
      }
    }
  }

  bool Changed = false;
  SmallPtrSet<Instruction *, 16> InstructionsProcessed;

  for (int Head : Heads) {
    if (InstructionsProcessed.count(Instrs[Head]))
      continue;
    bool LongerChainExists = false;
    for (unsigned TIt = 0; TIt < Tails.size(); TIt++)
      if (Head == Tails[TIt] &&
          !InstructionsProcessed.count(Instrs[Heads[TIt]])) {
        LongerChainExists = true;
        break;
      }
    if (LongerChainExists)
      continue;

    // We found an instr that starts a chain. Now follow the chain and try to
    // vectorize it.
    SmallVector<Instruction *, 16> Operands;
    int I = Head;
    while (I != -1 && (is_contained(Tails, I) || is_contained(Heads, I))) {
      if (InstructionsProcessed.count(Instrs[I]))
        break;

      Operands.push_back(Instrs[I]);
      I = ConsecutiveChain[I];
    }

    bool Vectorized = false;
    if (isa<LoadInst>(*Operands.begin()))
      Vectorized = vectorizeLoadChain(Operands, &InstructionsProcessed);
    else
      Vectorized = vectorizeStoreChain(Operands, &InstructionsProcessed);

    Changed |= Vectorized;
  }

  return Changed;
}

bool Vectorizer::vectorizeStoreChain(
    ArrayRef<Instruction *> Chain,
    SmallPtrSet<Instruction *, 16> *InstructionsProcessed) {
  StoreInst *S0 = cast<StoreInst>(Chain[0]);

  // If the vector has an int element, default to int for the whole load.
  Type *StoreTy;
  for (Instruction *I : Chain) {
    StoreTy = cast<StoreInst>(I)->getValueOperand()->getType();
    if (StoreTy->isIntOrIntVectorTy())
      break;

    if (StoreTy->isPtrOrPtrVectorTy()) {
      StoreTy = Type::getIntNTy(F.getParent()->getContext(),
                                DL.getTypeSizeInBits(StoreTy));
      break;
    }
  }

  unsigned Sz = DL.getTypeSizeInBits(StoreTy);
  unsigned AS = S0->getPointerAddressSpace();
  unsigned VecRegSize = TTI.getLoadStoreVecRegBitWidth(AS);
  unsigned VF = VecRegSize / Sz;
  unsigned ChainSize = Chain.size();
  unsigned Alignment = getAlignment(S0);

  if (!isPowerOf2_32(Sz) || VF < 2 || ChainSize < 2) {
    InstructionsProcessed->insert(Chain.begin(), Chain.end());
    return false;
  }

  ArrayRef<Instruction *> NewChain = getVectorizablePrefix(Chain);
  if (NewChain.empty()) {
    // No vectorization possible.
    InstructionsProcessed->insert(Chain.begin(), Chain.end());
    return false;
  }
  if (NewChain.size() == 1) {
    // Failed after the first instruction. Discard it and try the smaller chain.
    InstructionsProcessed->insert(NewChain.front());
    return false;
  }

  // Update Chain to the valid vectorizable subchain.
  Chain = NewChain;
  ChainSize = Chain.size();

  // Check if it's legal to vectorize this chain. If not, split the chain and
  // try again.
  unsigned EltSzInBytes = Sz / 8;
  unsigned SzInBytes = EltSzInBytes * ChainSize;
  if (!TTI.isLegalToVectorizeStoreChain(SzInBytes, Alignment, AS)) {
    auto Chains = splitOddVectorElts(Chain, Sz);
    return vectorizeStoreChain(Chains.first, InstructionsProcessed) |
           vectorizeStoreChain(Chains.second, InstructionsProcessed);
  }

  VectorType *VecTy;
  VectorType *VecStoreTy = dyn_cast<VectorType>(StoreTy);
  if (VecStoreTy)
    VecTy = VectorType::get(StoreTy->getScalarType(),
                            Chain.size() * VecStoreTy->getNumElements());
  else
    VecTy = VectorType::get(StoreTy, Chain.size());

  // If it's more than the max vector size or the target has a better
  // vector factor, break it into two pieces.
  unsigned TargetVF = TTI.getStoreVectorFactor(VF, Sz, SzInBytes, VecTy);
  if (ChainSize > VF || (VF != TargetVF && TargetVF < ChainSize)) {
    DEBUG(dbgs() << "LSV: Chain doesn't match with the vector factor."
                    " Creating two separate arrays.\n");
    return vectorizeStoreChain(Chain.slice(0, TargetVF),
                               InstructionsProcessed) |
           vectorizeStoreChain(Chain.slice(TargetVF), InstructionsProcessed);
  }

  DEBUG({
    dbgs() << "LSV: Stores to vectorize:\n";
    for (Instruction *I : Chain)
      dbgs() << "  " << *I << "\n";
  });

  // We won't try again to vectorize the elements of the chain, regardless of
  // whether we succeed below.
  InstructionsProcessed->insert(Chain.begin(), Chain.end());

  // If the store is going to be misaligned, don't vectorize it.
  if (accessIsMisaligned(SzInBytes, AS, Alignment)) {
    if (S0->getPointerAddressSpace() != 0)
      return false;

    unsigned NewAlign = getOrEnforceKnownAlignment(S0->getPointerOperand(),
                                                   StackAdjustedAlignment,
                                                   DL, S0, nullptr, &DT);
    if (NewAlign < StackAdjustedAlignment)
      return false;
  }

  BasicBlock::iterator First, Last;
  std::tie(First, Last) = getBoundaryInstrs(Chain);
  Builder.SetInsertPoint(&*Last);

  Value *Vec = UndefValue::get(VecTy);

  if (VecStoreTy) {
    unsigned VecWidth = VecStoreTy->getNumElements();
    for (unsigned I = 0, E = Chain.size(); I != E; ++I) {
      StoreInst *Store = cast<StoreInst>(Chain[I]);
      for (unsigned J = 0, NE = VecStoreTy->getNumElements(); J != NE; ++J) {
        unsigned NewIdx = J + I * VecWidth;
        Value *Extract = Builder.CreateExtractElement(Store->getValueOperand(),
                                                      Builder.getInt32(J));
        if (Extract->getType() != StoreTy->getScalarType())
          Extract = Builder.CreateBitCast(Extract, StoreTy->getScalarType());

        Value *Insert =
            Builder.CreateInsertElement(Vec, Extract, Builder.getInt32(NewIdx));
        Vec = Insert;
      }
    }
  } else {
    for (unsigned I = 0, E = Chain.size(); I != E; ++I) {
      StoreInst *Store = cast<StoreInst>(Chain[I]);
      Value *Extract = Store->getValueOperand();
      if (Extract->getType() != StoreTy->getScalarType())
        Extract =
            Builder.CreateBitOrPointerCast(Extract, StoreTy->getScalarType());

      Value *Insert =
          Builder.CreateInsertElement(Vec, Extract, Builder.getInt32(I));
      Vec = Insert;
    }
  }

  // This cast is safe because Builder.CreateStore() always creates a bona fide
  // StoreInst.
  StoreInst *SI = cast<StoreInst>(
      Builder.CreateStore(Vec, Builder.CreateBitCast(S0->getPointerOperand(),
                                                     VecTy->getPointerTo(AS))));
  propagateMetadata(SI, Chain);
  SI->setAlignment(Alignment);

  eraseInstructions(Chain);
  ++NumVectorInstructions;
  NumScalarsVectorized += Chain.size();
  return true;
}

bool Vectorizer::vectorizeLoadChain(
    ArrayRef<Instruction *> Chain,
    SmallPtrSet<Instruction *, 16> *InstructionsProcessed) {
  LoadInst *L0 = cast<LoadInst>(Chain[0]);

  // If the vector has an int element, default to int for the whole load.
  Type *LoadTy;
  for (const auto &V : Chain) {
    LoadTy = cast<LoadInst>(V)->getType();
    if (LoadTy->isIntOrIntVectorTy())
      break;

    if (LoadTy->isPtrOrPtrVectorTy()) {
      LoadTy = Type::getIntNTy(F.getParent()->getContext(),
                               DL.getTypeSizeInBits(LoadTy));
      break;
    }
  }

  unsigned Sz = DL.getTypeSizeInBits(LoadTy);
  unsigned AS = L0->getPointerAddressSpace();
  unsigned VecRegSize = TTI.getLoadStoreVecRegBitWidth(AS);
  unsigned VF = VecRegSize / Sz;
  unsigned ChainSize = Chain.size();
  unsigned Alignment = getAlignment(L0);

  if (!isPowerOf2_32(Sz) || VF < 2 || ChainSize < 2) {
    InstructionsProcessed->insert(Chain.begin(), Chain.end());
    return false;
  }

  ArrayRef<Instruction *> NewChain = getVectorizablePrefix(Chain);
  if (NewChain.empty()) {
    // No vectorization possible.
    InstructionsProcessed->insert(Chain.begin(), Chain.end());
    return false;
  }
  if (NewChain.size() == 1) {
    // Failed after the first instruction. Discard it and try the smaller chain.
    InstructionsProcessed->insert(NewChain.front());
    return false;
  }

  // Update Chain to the valid vectorizable subchain.
  Chain = NewChain;
  ChainSize = Chain.size();

  // Check if it's legal to vectorize this chain. If not, split the chain and
  // try again.
  unsigned EltSzInBytes = Sz / 8;
  unsigned SzInBytes = EltSzInBytes * ChainSize;
  if (!TTI.isLegalToVectorizeLoadChain(SzInBytes, Alignment, AS)) {
    auto Chains = splitOddVectorElts(Chain, Sz);
    return vectorizeLoadChain(Chains.first, InstructionsProcessed) |
           vectorizeLoadChain(Chains.second, InstructionsProcessed);
  }

  VectorType *VecTy;
  VectorType *VecLoadTy = dyn_cast<VectorType>(LoadTy);
  if (VecLoadTy)
    VecTy = VectorType::get(LoadTy->getScalarType(),
                            Chain.size() * VecLoadTy->getNumElements());
  else
    VecTy = VectorType::get(LoadTy, Chain.size());

  // If it's more than the max vector size or the target has a better
  // vector factor, break it into two pieces.
  unsigned TargetVF = TTI.getLoadVectorFactor(VF, Sz, SzInBytes, VecTy);
  if (ChainSize > VF || (VF != TargetVF && TargetVF < ChainSize)) {
    DEBUG(dbgs() << "LSV: Chain doesn't match with the vector factor."
                    " Creating two separate arrays.\n");
    return vectorizeLoadChain(Chain.slice(0, TargetVF), InstructionsProcessed) |
           vectorizeLoadChain(Chain.slice(TargetVF), InstructionsProcessed);
  }

  // We won't try again to vectorize the elements of the chain, regardless of
  // whether we succeed below.
  InstructionsProcessed->insert(Chain.begin(), Chain.end());

  // If the load is going to be misaligned, don't vectorize it.
  if (accessIsMisaligned(SzInBytes, AS, Alignment)) {
    if (L0->getPointerAddressSpace() != 0)
      return false;

    unsigned NewAlign = getOrEnforceKnownAlignment(L0->getPointerOperand(),
                                                   StackAdjustedAlignment,
                                                   DL, L0, nullptr, &DT);
    if (NewAlign < StackAdjustedAlignment)
      return false;

    Alignment = NewAlign;
  }

  DEBUG({
    dbgs() << "LSV: Loads to vectorize:\n";
    for (Instruction *I : Chain)
      I->dump();
  });

  // getVectorizablePrefix already computed getBoundaryInstrs.  The value of
  // Last may have changed since then, but the value of First won't have.  If it
  // matters, we could compute getBoundaryInstrs only once and reuse it here.
  BasicBlock::iterator First, Last;
  std::tie(First, Last) = getBoundaryInstrs(Chain);
  Builder.SetInsertPoint(&*First);

  Value *Bitcast =
      Builder.CreateBitCast(L0->getPointerOperand(), VecTy->getPointerTo(AS));
  // This cast is safe because Builder.CreateLoad always creates a bona fide
  // LoadInst.
  LoadInst *LI = cast<LoadInst>(Builder.CreateLoad(Bitcast));
  propagateMetadata(LI, Chain);
  LI->setAlignment(Alignment);

  if (VecLoadTy) {
    SmallVector<Instruction *, 16> InstrsToErase;

    unsigned VecWidth = VecLoadTy->getNumElements();
    for (unsigned I = 0, E = Chain.size(); I != E; ++I) {
      for (auto Use : Chain[I]->users()) {
        // All users of vector loads are ExtractElement instructions with
        // constant indices, otherwise we would have bailed before now.
        Instruction *UI = cast<Instruction>(Use);
        unsigned Idx = cast<ConstantInt>(UI->getOperand(1))->getZExtValue();
        unsigned NewIdx = Idx + I * VecWidth;
        Value *V = Builder.CreateExtractElement(LI, Builder.getInt32(NewIdx),
                                                UI->getName());
        if (V->getType() != UI->getType())
          V = Builder.CreateBitCast(V, UI->getType());

        // Replace the old instruction.
        UI->replaceAllUsesWith(V);
        InstrsToErase.push_back(UI);
      }
    }

    // Bitcast might not be an Instruction, if the value being loaded is a
    // constant.  In that case, no need to reorder anything.
    if (Instruction *BitcastInst = dyn_cast<Instruction>(Bitcast))
      reorder(BitcastInst);

    for (auto I : InstrsToErase)
      I->eraseFromParent();
  } else {
    for (unsigned I = 0, E = Chain.size(); I != E; ++I) {
      Value *CV = Chain[I];
      Value *V =
          Builder.CreateExtractElement(LI, Builder.getInt32(I), CV->getName());
      if (V->getType() != CV->getType()) {
        V = Builder.CreateBitOrPointerCast(V, CV->getType());
      }

      // Replace the old instruction.
      CV->replaceAllUsesWith(V);
    }

    if (Instruction *BitcastInst = dyn_cast<Instruction>(Bitcast))
      reorder(BitcastInst);
  }

  eraseInstructions(Chain);

  ++NumVectorInstructions;
  NumScalarsVectorized += Chain.size();
  return true;
}

bool Vectorizer::accessIsMisaligned(unsigned SzInBytes, unsigned AddressSpace,
                                    unsigned Alignment) {
  if (Alignment % SzInBytes == 0)
    return false;

  bool Fast = false;
  bool Allows = TTI.allowsMisalignedMemoryAccesses(F.getParent()->getContext(),
                                                   SzInBytes * 8, AddressSpace,
                                                   Alignment, &Fast);
  DEBUG(dbgs() << "LSV: Target said misaligned is allowed? " << Allows
               << " and fast? " << Fast << "\n";);
  return !Allows || !Fast;
}
