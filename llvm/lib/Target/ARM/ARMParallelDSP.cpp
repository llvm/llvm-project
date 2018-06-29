//===- ParallelDSP.cpp - Parallel DSP Pass --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Armv6 introduced instructions to perform 32-bit SIMD operations. The
/// purpose of this pass is do some IR pattern matching to create ACLE
/// DSP intrinsics, which map on these 32-bit SIMD operations.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/Debug.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "ARM.h"
#include "ARMSubtarget.h"

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "parallel-dsp"

namespace {
  struct ParallelMAC;
  struct Reduction;

  using ParallelMACList = SmallVector<ParallelMAC, 8>;
  using ReductionList   = SmallVector<Reduction, 8>;
  using ValueList       = SmallVector<Value*, 8>;
  using LoadInstList    = SmallVector<LoadInst*, 8>;
  using PMACPair        = std::pair<ParallelMAC*,ParallelMAC*>;
  using PMACPairList    = SmallVector<PMACPair, 8>;
  using Instructions    = SmallVector<Instruction*,16>;
  using MemLocList      = SmallVector<MemoryLocation, 4>;

  // 'ParallelMAC' and 'Reduction' are just some bookkeeping data structures.
  // 'Reduction' contains the phi-node and accumulator statement from where we
  // start pattern matching, and 'ParallelMAC' the multiplication
  // instructions that are candidates for parallel execution.
  struct ParallelMAC {
    Instruction *Mul;
    ValueList    VL;        // List of all (narrow) operands of this Mul
    LoadInstList VecLd;     // List of all load instructions of this Mul
    MemLocList   MemLocs;   // All memory locations read by this Mul

    ParallelMAC(Instruction *I, ValueList &V) : Mul(I), VL(V) {};
  };

  struct Reduction {
    PHINode         *Phi;             // The Phi-node from where we start
                                      // pattern matching.
    Instruction     *AccIntAdd;       // The accumulating integer add statement,
                                      // i.e, the reduction statement.

    Reduction (PHINode *P, Instruction *Acc) : Phi(P), AccIntAdd(Acc) { };
  };

  class ARMParallelDSP : public LoopPass {
    ScalarEvolution   *SE;
    AliasAnalysis     *AA;
    TargetLibraryInfo *TLI;
    DominatorTree     *DT;
    LoopInfo          *LI;
    Loop              *L;
    const DataLayout  *DL;
    Module            *M;

    bool InsertParallelMACs(Reduction &Reduction, PMACPairList &PMACPairs);
    bool AreSequentialLoads(LoadInst *Ld0, LoadInst *Ld1, LoadInstList &VecLd);
    PMACPairList CreateParallelMACPairs(ParallelMACList &Candidates);
    Instruction *CreateSMLADCall(LoadInst *VecLd0, LoadInst *VecLd1,
                                 Instruction *Acc, Instruction *InsertAfter);

    /// Try to match and generate: SMLAD, SMLADX - Signed Multiply Accumulate
    /// Dual performs two signed 16x16-bit multiplications. It adds the
    /// products to a 32-bit accumulate operand. Optionally, the instruction can
    /// exchange the halfwords of the second operand before performing the
    /// arithmetic.
    bool MatchSMLAD(Function &F);

  public:
    static char ID;

    ARMParallelDSP() : LoopPass(ID) { }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      LoopPass::getAnalysisUsage(AU);
      AU.addRequired<AssumptionCacheTracker>();
      AU.addRequired<ScalarEvolutionWrapperPass>();
      AU.addRequired<AAResultsWrapperPass>();
      AU.addRequired<TargetLibraryInfoWrapperPass>();
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<TargetPassConfig>();
      AU.addPreserved<LoopInfoWrapperPass>();
      AU.setPreservesCFG();
    }

    bool runOnLoop(Loop *TheLoop, LPPassManager &) override {
      L = TheLoop;
      SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
      AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
      TLI = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
      DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
      LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
      auto &TPC = getAnalysis<TargetPassConfig>();

      BasicBlock *Header = TheLoop->getHeader();
      if (!Header)
        return false;

      // TODO: We assume the loop header and latch to be the same block.
      // This is not a fundamental restriction, but lifting this would just
      // require more work to do the transformation and then patch up the CFG.
      if (Header != TheLoop->getLoopLatch()) {
        LLVM_DEBUG(dbgs() << "The loop header is not the loop latch: not "
                             "running pass ARMParallelDSP\n");
        return false;
      }

      Function &F = *Header->getParent();
      M = F.getParent();
      DL = &M->getDataLayout();

      auto &TM = TPC.getTM<TargetMachine>();
      auto *ST = &TM.getSubtarget<ARMSubtarget>(F);

      if (!ST->allowsUnalignedMem()) {
        LLVM_DEBUG(dbgs() << "Unaligned memory access not supported: not "
                             "running pass ARMParallelDSP\n");
        return false;
      }

      if (!ST->hasDSP()) {
        LLVM_DEBUG(dbgs() << "DSP extension not enabled: not running pass "
                             "ARMParallelDSP\n");
        return false;
      }

      LoopAccessInfo LAI(L, SE, TLI, AA, DT, LI);
      bool Changes = false;

      LLVM_DEBUG(dbgs() << "\n== Parallel DSP pass ==\n\n");
      Changes = MatchSMLAD(F);
      return Changes;
    }
  };
}

template<unsigned BitWidth>
static bool IsNarrowSequence(Value *V, ValueList &VL) {
  LLVM_DEBUG(dbgs() << "Is narrow sequence: "; V->dump());
  ConstantInt *CInt;

  if (match(V, m_ConstantInt(CInt))) {
    // TODO: if a constant is used, it needs to fit within the bit width.
    return false;
  }

  auto *I = dyn_cast<Instruction>(V);
  if (!I)
   return false;

  Value *Val, *LHS, *RHS;
  bool isNarrow = false;

  if (match(V, m_Trunc(m_Value(Val)))) {
    if (cast<TruncInst>(I)->getDestTy()->getIntegerBitWidth() == BitWidth)
      isNarrow = IsNarrowSequence<BitWidth>(Val, VL);
  } else if (match(V, m_Add(m_Value(LHS), m_Value(RHS)))) {
    // TODO: we need to implement sadd16/sadd8 for this, which enables to
    // also do the rewrite for smlad8.ll, but it is unsupported for now.
    isNarrow = false;
  } else if (match(V, m_ZExtOrSExt(m_Value(Val)))) {
    if (cast<CastInst>(I)->getSrcTy()->getIntegerBitWidth() == BitWidth)
      isNarrow = true;
    else
      LLVM_DEBUG(dbgs() << "Wrong SrcTy size of CastInst: " <<
                 cast<CastInst>(I)->getSrcTy()->getIntegerBitWidth());

    if (match(Val, m_Load(m_Value(Val)))) {
      auto *Ld = dyn_cast<LoadInst>(I->getOperand(0));
      LLVM_DEBUG(dbgs() << "Found narrow Load:\t"; Ld->dump());
      VL.push_back(Ld);
      isNarrow = true;
    } else if (!isa<Instruction>(I->getOperand(0)))
      VL.push_back(I->getOperand(0));
  }

  if (isNarrow) {
    LLVM_DEBUG(dbgs() << "Found narrow Op:\t"; I->dump());
    VL.push_back(I);
  } else
    LLVM_DEBUG(dbgs() << "Found unsupported Op:\t"; I->dump());

  return isNarrow;
}

// Element-by-element comparison of Value lists returning true if they are
// instructions with the same opcode or constants with the same value.
static bool AreSymmetrical(const ValueList &VL0,
                           const ValueList &VL1) {
  if (VL0.size() != VL1.size()) {
    LLVM_DEBUG(dbgs() << "Muls are mismatching operand list lengths: "
                      << VL0.size() << " != " << VL1.size() << "\n");
    return false;
  }

  const unsigned Pairs = VL0.size();
  LLVM_DEBUG(dbgs() << "Number of operand pairs: " << Pairs << "\n");

  for (unsigned i = 0; i < Pairs; ++i) {
    const Value *V0 = VL0[i];
    const Value *V1 = VL1[i];
    const auto *Inst0 = dyn_cast<Instruction>(V0);
    const auto *Inst1 = dyn_cast<Instruction>(V1);

    LLVM_DEBUG(dbgs() << "Pair " << i << ":\n";
               dbgs() << "mul1: "; V0->dump();
               dbgs() << "mul2: "; V1->dump());

    if (!Inst0 || !Inst1)
      return false;

    if (Inst0->isSameOperationAs(Inst1)) {
      LLVM_DEBUG(dbgs() << "OK: same operation found!\n");
      continue;
    }

    const APInt *C0, *C1;
    if (!(match(V0, m_APInt(C0)) && match(V1, m_APInt(C1)) && C0 == C1))
      return false;
  }

  LLVM_DEBUG(dbgs() << "OK: found symmetrical operand lists.\n");
  return true;
}

bool ARMParallelDSP::AreSequentialLoads(LoadInst *Ld0, LoadInst *Ld1,
                                        LoadInstList &VecLd) {
  if (!Ld0 || !Ld1)
    return false;

  LLVM_DEBUG(dbgs() << "Are consecutive loads:\n";
    dbgs() << "Ld0:"; Ld0->dump();
    dbgs() << "Ld1:"; Ld1->dump();
  );

  if (!Ld0->isSimple() || !Ld1->isSimple()) {
    LLVM_DEBUG(dbgs() << "No, not touching volatile loads\n");
    return false;
  }
  if (!Ld0->hasOneUse() || !Ld1->hasOneUse()) {
    LLVM_DEBUG(dbgs() << "No, load has more than one use.\n");
    return false;
  }
  if (isConsecutiveAccess(Ld0, Ld1, *DL, *SE)) {
    VecLd.push_back(Ld0);
    VecLd.push_back(Ld1);
    LLVM_DEBUG(dbgs() << "OK: loads are consecutive.\n");
    return true;
  }
  LLVM_DEBUG(dbgs() << "No, Ld0 and Ld1 aren't consecutive.\n");
  return false;
}

PMACPairList
ARMParallelDSP::CreateParallelMACPairs(ParallelMACList &Candidates) {
  const unsigned Elems = Candidates.size();
  PMACPairList PMACPairs;

  if (Elems < 2)
    return PMACPairs;

  // TODO: for now we simply try to match consecutive pairs i and i+1.
  // We can compare all elements, but then we need to compare and evaluate
  // different solutions.
  for(unsigned i=0; i<Elems-1; i+=2) {
    ParallelMAC &PMul0 = Candidates[i];
    ParallelMAC &PMul1 = Candidates[i+1];
    const Instruction *Mul0 = PMul0.Mul;
    const Instruction *Mul1 = PMul1.Mul;

    if (Mul0 == Mul1)
      continue;

    LLVM_DEBUG(dbgs() << "\nCheck parallel muls:\n";
               dbgs() << "- "; Mul0->dump();
               dbgs() << "- "; Mul1->dump());

    const ValueList &VL0 = PMul0.VL;
    const ValueList &VL1 = PMul1.VL;

    if (!AreSymmetrical(VL0, VL1))
      continue;

    LLVM_DEBUG(dbgs() << "OK: mul operands list match:\n");
    // The first elements of each vector should be loads with sexts. If we find
    // that its two pairs of consecutive loads, then these can be transformed
    // into two wider loads and the users can be replaced with DSP
    // intrinsics.
    for (unsigned x = 0; x < VL0.size(); x += 4) {
      auto *Ld0 = dyn_cast<LoadInst>(VL0[x]);
      auto *Ld1 = dyn_cast<LoadInst>(VL1[x]);
      auto *Ld2 = dyn_cast<LoadInst>(VL0[x+2]);
      auto *Ld3 = dyn_cast<LoadInst>(VL1[x+2]);

      LLVM_DEBUG(dbgs() << "Looking at operands " << x << ":\n";
                 dbgs() << "\t mul1: "; VL0[x]->dump();
                 dbgs() << "\t mul2: "; VL1[x]->dump();
                 dbgs() << "and operands " << x + 2 << ":\n";
                 dbgs() << "\t mul1: "; VL0[x+2]->dump();
                 dbgs() << "\t mul2: "; VL1[x+2]->dump());

      if (AreSequentialLoads(Ld0, Ld1, Candidates[i].VecLd) &&
          AreSequentialLoads(Ld2, Ld3, Candidates[i+1].VecLd)) {
        LLVM_DEBUG(dbgs() << "OK: found two pairs of parallel loads!\n");
        PMACPairs.push_back(std::make_pair(&PMul0, &PMul1));
      }
    }
  }
  return PMACPairs;
}

bool ARMParallelDSP::InsertParallelMACs(Reduction &Reduction,
                                        PMACPairList &PMACPairs) {
  Instruction *Acc = Reduction.Phi;
  Instruction *InsertAfter = Reduction.AccIntAdd;

  for (auto &Pair : PMACPairs) {
    LLVM_DEBUG(dbgs() << "Found parallel MACs!!\n";
               dbgs() << "- "; Pair.first->Mul->dump();
               dbgs() << "- "; Pair.second->Mul->dump());
    Acc = CreateSMLADCall(Pair.first->VecLd[0], Pair.second->VecLd[0], Acc,
	                        InsertAfter);
    InsertAfter = Acc;
  }

  if (Acc != Reduction.Phi) {
    LLVM_DEBUG(dbgs() << "Replace Accumulate: "; Acc->dump());
    Reduction.AccIntAdd->replaceAllUsesWith(Acc);
    return true;
  }
  return false;
}

static ReductionList MatchReductions(Function &F, Loop *TheLoop,
                                     BasicBlock *Header) {
  ReductionList Reductions;
  RecurrenceDescriptor RecDesc;
  const bool HasFnNoNaNAttr =
    F.getFnAttribute("no-nans-fp-math").getValueAsString() == "true";
  const BasicBlock *Latch = TheLoop->getLoopLatch();

  // We need a preheader as getIncomingValueForBlock assumes there is one.
  if (!TheLoop->getLoopPreheader())
    return Reductions;

  for (PHINode &Phi : Header->phis()) {
    const auto *Ty = Phi.getType();
    if (!Ty->isIntegerTy(32))
      continue;

    const bool IsReduction =
      RecurrenceDescriptor::AddReductionVar(&Phi,
                                            RecurrenceDescriptor::RK_IntegerAdd,
                                            TheLoop, HasFnNoNaNAttr, RecDesc);
    if (!IsReduction)
      continue;

    Instruction *Acc = dyn_cast<Instruction>(Phi.getIncomingValueForBlock(Latch));
    if (!Acc)
      continue;

    Reductions.push_back(Reduction(&Phi, Acc));
  }

  LLVM_DEBUG(
    dbgs() << "\nAccumulating integer additions (reductions) found:\n";
    for (auto R : Reductions) {
      dbgs() << "-  "; R.Phi->dump();
      dbgs() << "-> "; R.AccIntAdd->dump();
    }
  );
  return Reductions;
}

static void AddCandidateMAC(ParallelMACList &Candidates, const Instruction *Acc,
                            Value *MulOp0, Value *MulOp1, int MulOpNum) {
  Instruction *Mul = dyn_cast<Instruction>(Acc->getOperand(MulOpNum));
  LLVM_DEBUG(dbgs() << "OK, found acc mul:\t"; Mul->dump());
  ValueList VL;
  if (IsNarrowSequence<16>(MulOp0, VL) &&
      IsNarrowSequence<16>(MulOp1, VL)) {
    LLVM_DEBUG(dbgs() << "OK, found narrow mul: "; Mul->dump());
    Candidates.push_back(ParallelMAC(Mul, VL));
  }
}

static ParallelMACList MatchParallelMACs(Reduction &R) {
  ParallelMACList Candidates;
  const Instruction *Acc = R.AccIntAdd;
  Value *A, *MulOp0, *MulOp1;
  LLVM_DEBUG(dbgs() << "\n- Analysing:\t"; Acc->dump());

  // Pattern 1: the accumulator is the RHS of the mul.
  while(match(Acc, m_Add(m_Mul(m_Value(MulOp0), m_Value(MulOp1)),
                         m_Value(A)))){
    AddCandidateMAC(Candidates, Acc, MulOp0, MulOp1, 0);
    Acc = dyn_cast<Instruction>(A);
  }
  // Pattern 2: the accumulator is the LHS of the mul.
  while(match(Acc, m_Add(m_Value(A),
                         m_Mul(m_Value(MulOp0), m_Value(MulOp1))))) {
    AddCandidateMAC(Candidates, Acc, MulOp0, MulOp1, 1);
    Acc = dyn_cast<Instruction>(A);
  }

  // The last mul in the chain has a slightly different pattern:
  // the mul is the first operand
  if (match(Acc, m_Add(m_Mul(m_Value(MulOp0), m_Value(MulOp1)), m_Value(A))))
    AddCandidateMAC(Candidates, Acc, MulOp0, MulOp1, 0);

  // Because we start at the bottom of the chain, and we work our way up,
  // the muls are added in reverse program order to the list.
  std::reverse(Candidates.begin(), Candidates.end());
  return Candidates;
}

// Collects all instructions that are not part of the MAC chains, which is the
// set of instructions that can potentially alias with the MAC operands.
static Instructions AliasCandidates(BasicBlock *Header,
                                    ParallelMACList &MACCandidates) {
  Instructions Aliases;
  auto IsMACCandidate = [] (Instruction *I, ParallelMACList &MACCandidates) {
    for (auto &MAC : MACCandidates)
      for (auto *Val : MAC.VL)
        if (I == MAC.Mul || Val == I)
          return true;
   return false;
  };

  std::for_each(Header->begin(), Header->end(),
                [&Aliases, &MACCandidates, &IsMACCandidate] (Instruction &I) {
                  if (I.mayReadOrWriteMemory() &&
                      !IsMACCandidate(&I, MACCandidates))
                    Aliases.push_back(&I); });
  return Aliases;
}

// This compares all instructions from the "alias candidates" set, i.e., all
// instructions that are not part of the MAC-chain, with all instructions in
// the MAC candidate set, to see if instructions are aliased.
static bool AreAliased(AliasAnalysis *AA, Instructions AliasCandidates,
                       ParallelMACList &MACCandidates) {
  LLVM_DEBUG(dbgs() << "Alias checks:\n");
  for (auto *I : AliasCandidates) {
    LLVM_DEBUG(dbgs() << "- "; I->dump());
    for (auto &MAC : MACCandidates) {
      LLVM_DEBUG(dbgs() << "mul: "; MAC.Mul->dump());
      assert(MAC.MemLocs.size() >= 2 && "expecting at least 2 memlocs");
      for (auto &MemLoc : MAC.MemLocs) {
        if (isModOrRefSet(intersectModRef(AA->getModRefInfo(I, MemLoc),
                                          ModRefInfo::ModRef))) {
          LLVM_DEBUG(dbgs() << "Yes, aliases found\n");
          return true;
        }
      }
    }
  }
  LLVM_DEBUG(dbgs() << "OK: no aliases found!\n");
  return false;
}

static bool SetMemoryLocations(ParallelMACList &Candidates) {
  const auto Size = MemoryLocation::UnknownSize;
  for (auto &C : Candidates) {
    // A mul has 2 operands, and a narrow op consist of sext and a load; thus
    // we expect at least 4 items in this operand value list.
    if (C.VL.size() < 4) {
      LLVM_DEBUG(dbgs() << "Operand list too short.\n");
      return false;
    }

    for (unsigned i = 0; i < C.VL.size(); i += 4) {
      auto *LdOp0 = dyn_cast<LoadInst>(C.VL[i]);
      auto *LdOp1 = dyn_cast<LoadInst>(C.VL[i+2]);
      if (!LdOp0 || !LdOp1)
        return false;

      C.MemLocs.push_back(MemoryLocation(LdOp0->getPointerOperand(), Size));
      C.MemLocs.push_back(MemoryLocation(LdOp1->getPointerOperand(), Size));
    }
  }
  return true;
}

// Loop Pass that needs to identify integer add/sub reductions of 16-bit vector
// multiplications.
// To use SMLAD:
// 1) we first need to find integer add reduction PHIs,
// 2) then from the PHI, look for this pattern:
//
// acc0 = phi i32 [0, %entry], [%acc1, %loop.body]
// ld0 = load i16
// sext0 = sext i16 %ld0 to i32
// ld1 = load i16
// sext1 = sext i16 %ld1 to i32
// mul0 = mul %sext0, %sext1
// ld2 = load i16
// sext2 = sext i16 %ld2 to i32
// ld3 = load i16
// sext3 = sext i16 %ld3 to i32
// mul1 = mul i32 %sext2, %sext3
// add0 = add i32 %mul0, %acc0
// acc1 = add i32 %add0, %mul1
//
// Which can be selected to:
//
// ldr.h r0
// ldr.h r1
// smlad r2, r0, r1, r2
//
// If constants are used instead of loads, these will need to be hoisted
// out and into a register.
//
// If loop invariants are used instead of loads, these need to be packed
// before the loop begins.
//
// Can only be enabled for cores which support unaligned accesses.
//
bool ARMParallelDSP::MatchSMLAD(Function &F) {
  BasicBlock *Header = L->getHeader();
  LLVM_DEBUG(dbgs() << "= Matching SMLAD =\n";
             dbgs() << "Header block:\n"; Header->dump();
             dbgs() << "Loop info:\n\n"; L->dump());

  bool Changed = false;
  ReductionList Reductions = MatchReductions(F, L, Header);

  for (auto &R : Reductions) {
    ParallelMACList MACCandidates = MatchParallelMACs(R);
    if (!SetMemoryLocations(MACCandidates))
      continue;
    Instructions Aliases = AliasCandidates(Header, MACCandidates);
    if (AreAliased(AA, Aliases, MACCandidates))
      continue;
    PMACPairList PMACPairs = CreateParallelMACPairs(MACCandidates);
    Changed = InsertParallelMACs(R, PMACPairs) || Changed;
  }

  LLVM_DEBUG(if (Changed) dbgs() << "Header block:\n"; Header->dump(););
  return Changed;
}

static void CreateLoadIns(IRBuilder<NoFolder> &IRB, Instruction *Acc,
                          LoadInst **VecLd) {
  const Type *AccTy = Acc->getType();
  const unsigned AddrSpace = (*VecLd)->getPointerAddressSpace();

  Value *VecPtr = IRB.CreateBitCast((*VecLd)->getPointerOperand(),
                                    AccTy->getPointerTo(AddrSpace));
  *VecLd = IRB.CreateAlignedLoad(VecPtr, (*VecLd)->getAlignment());
}

Instruction *ARMParallelDSP::CreateSMLADCall(LoadInst *VecLd0, LoadInst *VecLd1,
                                             Instruction *Acc,
                                             Instruction *InsertAfter) {
  LLVM_DEBUG(dbgs() << "Create SMLAD intrinsic using:\n";
             dbgs() << "- "; VecLd0->dump();
             dbgs() << "- "; VecLd1->dump();
             dbgs() << "- "; Acc->dump());

  IRBuilder<NoFolder> Builder(InsertAfter->getParent(),
                              ++BasicBlock::iterator(InsertAfter));

  // Replace the reduction chain with an intrinsic call
  CreateLoadIns(Builder, Acc, &VecLd0);
  CreateLoadIns(Builder, Acc, &VecLd1);
  Value* Args[] = { VecLd0, VecLd1, Acc };
  Function *SMLAD = Intrinsic::getDeclaration(M, Intrinsic::arm_smlad);
  CallInst *Call = Builder.CreateCall(SMLAD, Args);
  return Call;
}

Pass *llvm::createARMParallelDSPPass() {
  return new ARMParallelDSP();
}

char ARMParallelDSP::ID = 0;

INITIALIZE_PASS_BEGIN(ARMParallelDSP, "parallel-dsp",
                "Transform loops to use DSP intrinsics", false, false)
INITIALIZE_PASS_END(ARMParallelDSP, "parallel-dsp",
                "Transform loops to use DSP intrinsics", false, false)
