//===---- LoongArchMemoryBarrierOpt.cpp - Memory barrier Optimization -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This pass removes or merges redundant memory barrier instructions.
///
/// - DBAR x + DBAR y -> DBAR (x & y)
/// - DBAR x + AMO_DB -> AMO_DB
/// - DBAR x + AMO    -> AMO_DB
/// - DBAR x + LL     -> LL
/// - AMO_DB + DBAR x -> AMO_DB
/// - AMO    + DBAR x -> AMO_DB
/// - SC     + DBAR x -> SC
///
//===----------------------------------------------------------------------===//

#include "LoongArch.h"
#include "LoongArchInstrInfo.h"
#include "LoongArchSubtarget.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "loongarch-memory-barrier-opt"
#define LOONGARCH_MEMORY_BARRIER_OPT_NAME                                      \
  "LoongArch Memory Barrier Optimisation pass"

static cl::opt<bool> RequireNoPathBypass(
    "loongarch-require-no-path-bypass",
    cl::desc("Optimize only when no paths bypass either memory barrier"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> MergeAMOWithMB(
    "loongarch-merge-amo-with-dbar",
    cl::desc("Merge AMOs with DBARs into AMO_DB during optimization"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> DisableInlineAsm(
    "loongarch-disable-inline-asm-barrier-opt",
    cl::desc("Disable optimization of memory barriers in InlineAsm"),
    cl::init(false), cl::Hidden);

static cl::opt<bool> ReplaceEliminatedMBToNop(
    "loongarch-replace-eliminated-dbar-to-nop",
    cl::desc("Replace eliminated DBARs with NOPs to preserve code layout"),
    cl::init(false), cl::Hidden);

namespace {

#define AMO_CASES                                                              \
  CASE(AMSWAP, B)                                                              \
  CASE(AMSWAP, H)                                                              \
  CASE(AMSWAP, W)                                                              \
  CASE(AMSWAP, D)                                                              \
  CASE(AMADD, B)                                                               \
  CASE(AMADD, H)                                                               \
  CASE(AMADD, W)                                                               \
  CASE(AMADD, D)                                                               \
  CASE(AMAND, W)                                                               \
  CASE(AMAND, D)                                                               \
  CASE(AMOR, W)                                                                \
  CASE(AMOR, D)                                                                \
  CASE(AMXOR, W)                                                               \
  CASE(AMXOR, D)                                                               \
  CASE(AMMAX, W)                                                               \
  CASE(AMMAX, D)                                                               \
  CASE(AMMAX, WU)                                                              \
  CASE(AMMAX, DU)                                                              \
  CASE(AMMIN, W)                                                               \
  CASE(AMMIN, D)                                                               \
  CASE(AMMIN, WU)                                                              \
  CASE(AMMIN, DU)                                                              \
  CASE(AMCAS, B)                                                               \
  CASE(AMCAS, H)                                                               \
  CASE(AMCAS, W)                                                               \
  CASE(AMCAS, D)

static std::optional<std::pair<StringRef, StringRef>> parseMB(StringRef Asm) {
  auto T1 = llvm::getToken(Asm);
  if (!T1.first.equals_insensitive("dbar"))
    return std::nullopt;
  auto T2 = llvm::getToken(T1.second);
  if (T2.first.empty())
    return std::nullopt;
  auto T3 = llvm::getToken(T2.second);
  if (T3.first.trim().empty() || T3.first.starts_with('#'))
    return std::pair(T1.first, T2.first);
  return std::nullopt;
}

static std::optional<std::pair<StringRef, StringRef>>
isAsmMB(const MachineInstr &MI) {
  if (DisableInlineAsm)
    return std::nullopt;
  if (!MI.isInlineAsm())
    return std::nullopt;
  auto Asm = MI.getOperand(InlineAsm::MIOp_AsmString).getSymbolName();
  return parseMB(Asm);
}

static StringRef getAMDB(StringRef Name) {
#define CASE(Name, Suffix)                                                     \
  .Case(#Name "." #Suffix, MergeAMOWithMB ? #Name "_DB." #Suffix : "")         \
      .Case(#Name "_DB." #Suffix, #Name "_DB." #Suffix)
  return StringSwitch<StringRef>(Name.upper()) AMO_CASES.Default({});
#undef CASE
}

static std::optional<std::pair<StringRef, StringRef>> parseAM(StringRef Asm) {
  auto T1 = llvm::getToken(Asm);
  auto OpName = getAMDB(T1.first);
  if (OpName.empty())
    return std::nullopt;
  auto T2 = llvm::getToken(T1.second, ",");
  if (T2.first.empty())
    return std::nullopt;
  auto T3 = llvm::getToken(T2.second, ",");
  if (T3.first.empty())
    return std::nullopt;
  auto T4 = llvm::getToken(T3.second, ",");
  if (T4.first.empty())
    return std::nullopt;
  auto T5 = llvm::getToken(T4.second);
  if (T5.first.trim().empty() || T5.first.starts_with('#')) {
    StringRef Operands(T2.first.data(),
                       T4.first.data() + T4.first.size() - T2.first.data());
    return std::pair(OpName, Operands);
  }
  return std::nullopt;
}

static std::optional<std::pair<StringRef, StringRef>>
isAsmAM(const MachineInstr &MI) {
  if (DisableInlineAsm)
    return std::nullopt;
  if (!MI.isInlineAsm())
    return std::nullopt;
  auto Asm = MI.getOperand(InlineAsm::MIOp_AsmString).getSymbolName();
  return parseAM(Asm);
}

static bool isMB(const MachineInstr &MI) {
  return MI.getOpcode() == LoongArch::DBAR;
}

static bool isLL(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case LoongArch::LL_W:
  case LoongArch::LL_D:
    return true;
  default:
    return false;
  }
}

static bool isSC(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case LoongArch::SC_W:
  case LoongArch::SC_D:
  case LoongArch::SC_Q:
    return true;
  default:
    return false;
  }
}

static std::optional<unsigned> isAM(const MachineInstr &MI) {
#define CASE(Name, Suffix)                                                     \
  case LoongArch::Name##_##Suffix:                                             \
    if (!MergeAMOWithMB)                                                       \
      return std::nullopt;                                                     \
    [[fallthrough]];                                                           \
  case LoongArch::Name##__DB_##Suffix:                                         \
    return LoongArch::Name##__DB_##Suffix;
  switch (MI.getOpcode()) {
    AMO_CASES
  default:
    return std::nullopt;
  }
#undef CASE
}

static bool isSafeToSkip(const MachineInstr &MI) {
  if (MI.mayLoadOrStore())
    return false;
  if (MI.isCall() || MI.isReturn())
    return false;
  if (MI.isInlineAsm())
    return isAsmMB(MI) != std::nullopt;
  if (MI.hasUnmodeledSideEffects())
    return isMB(MI);
  return true;
}

struct BarrierHint {
  BarrierHint(unsigned Hint) : Hint(Hint) {}

  bool subsumes(const BarrierHint &O) const { return (Hint & O.Hint) == Hint; }

  BarrierHint merge(const BarrierHint &O) const {
    return BarrierHint(Hint & O.Hint);
  }

  static inline bool isValid(unsigned Hint) { return (Hint & ~0x1f) == 0; }

  unsigned Hint;
};

struct InstBarrier {
  InstBarrier(MachineInstr &MI)
      : MI(&MI), Pre(0), Post(0), Data(0), IsMB(false), IsAM(false),
        IsAsm(false) {
    if (isMB(MI)) {
      unsigned Hint = MI.getOperand(0).getImm();
      if (!BarrierHint::isValid(Hint))
        return;
      IsMB = true;
      Pre = Post = BarrierHint(Hint);
    } else if (isLL(MI)) {
      IsAM = true;
      Pre = BarrierHint(0b10000);
      Post = BarrierHint(0b11111);
    } else if (isSC(MI)) {
      IsAM = true;
      Pre = BarrierHint(0b11111);
      Post = BarrierHint(0b10000);
    } else if (auto R = isAM(MI)) {
      IsAM = true;
      OpcAMDB = *R;
      Pre = Post = BarrierHint(0b10000);
    } else if (auto R = isAsmMB(MI)) {
      OpName = (*R).first;
      Operands = (*R).second;
      auto B = parseAsmMB(Operands, MI);
      if (!B || !BarrierHint::isValid((*B).first))
        return;
      IsMB = true;
      IsAsm = true;
      HintOff = (*B).second;
      Pre = Post = BarrierHint((*B).first);
    } else if (auto R = isAsmAM(MI)) {
      OpName = (*R).first;
      Operands = (*R).second;
      IsAM = true;
      IsAsm = true;
      Pre = Post = BarrierHint(0b10000);
    }
  }

  static std::optional<std::pair<unsigned, unsigned>>
  parseAsmMB(StringRef Operand, MachineInstr &MI) {
    unsigned Hint, HintOff = 0;
    // DBAR N | 0xN
    if (!Operand.starts_with('$')) {
      if (Operand.getAsInteger(0, Hint))
        return std::nullopt;
      return std::pair(Hint, HintOff);
    }
    // DBAR $N
    unsigned N = 0, Off, AsmDescOp;
    if (Operand.drop_front().getAsInteger(0, Off))
      return std::nullopt;
    AsmDescOp = InlineAsm::MIOp_FirstOperand;
    while (AsmDescOp != MI.getNumOperands()) {
      const MachineOperand &MO = MI.getOperand(AsmDescOp);
      assert(MO.isImm() && "Unexpected operand type!");
      const InlineAsm::Flag F(MO.getImm());
      if (N == Off) {
        assert(F.isImmKind() && "Unexpected flag kind!");
        HintOff = AsmDescOp + 1;
        Hint = MI.getOperand(HintOff).getImm();
        return std::pair(Hint, HintOff);
      }
      AsmDescOp += 1 + F.getNumOperandRegisters();
      ++N;
    }
    return std::nullopt;
  }

  MachineInstr *MI;
  BarrierHint Pre;
  BarrierHint Post;
  StringRef OpName;
  StringRef Operands;
  union {
    unsigned OpcAMDB;
    unsigned HintOff;
    unsigned Data;
  };
  bool IsMB;
  bool IsAM;
  bool IsAsm;
};

class LoongArchMemoryBarrierOpt : public MachineFunctionPass {
public:
  static char ID;

  LoongArchMemoryBarrierOpt() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return LOONGARCH_MEMORY_BARRIER_OPT_NAME;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<MachineDominatorTreeWrapperPass>();
    AU.addRequired<MachinePostDominatorTreeWrapperPass>();
    AU.addPreserved<MachinePostDominatorTreeWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &Fn) override;

private:
  enum : unsigned {
    CandidateA = 1u << 0,
    CandidateB = 1u << 1,
  };

  unsigned resolveBarrierRedundancy(const MachineInstr *A,
                                    const MachineInstr *B) const;
  bool eliminateRedundantBarrier(InstBarrier &IA, InstBarrier &IB) const;

  MachineFunction *MF;
  const MachineDominatorTree *MDT;
  const MachinePostDominatorTree *MPDT;
};

static bool checkAllPathSafe(const MachineBasicBlock *MBBA,
                             const MachineBasicBlock *MBBB, bool IsAToB) {
  const MachineBasicBlock *Start = IsAToB ? MBBA : MBBB;
  const MachineBasicBlock *End = IsAToB ? MBBB : MBBA;

  SmallVector<const MachineBasicBlock *, 16> Worklist;
  DenseSet<const MachineBasicBlock *> Visited;

  Worklist.push_back(Start);
  Visited.insert(Start);

  while (!Worklist.empty()) {
    const MachineBasicBlock *BB = Worklist.pop_back_val();

    if (BB == End)
      continue;

    if (BB != Start) {
      for (const MachineInstr &MI : *BB) {
        if (!isSafeToSkip(MI))
          return false;
      }
    }

    if (IsAToB) {
      for (const MachineBasicBlock *Succ : BB->successors()) {
        if (Visited.insert(Succ).second)
          Worklist.push_back(Succ);
      }
    } else {
      for (const MachineBasicBlock *Pred : BB->predecessors()) {
        if (Visited.insert(Pred).second)
          Worklist.push_back(Pred);
      }
    }
  }

  return true;
}

// Returns a bitmask indicating removal candidates: A (bit 1) and B (bit 2).
unsigned LoongArchMemoryBarrierOpt::resolveBarrierRedundancy(
    const MachineInstr *A, const MachineInstr *B) const {
  const MachineBasicBlock *MBBA = A->getParent();
  const MachineBasicBlock *MBBB = B->getParent();

  if (MBBA == MBBB) {
    /* A -> B */
    for (auto It = std::next(A->getIterator()); It != MBBA->end(); ++It) {
      if (It == B->getIterator())
        return CandidateA | CandidateB;
      if (!isSafeToSkip(*It))
        return 0;
    }
    return 0;
  }

  // Cross-block walk
  bool ADomB = MDT->dominates(MBBA, MBBB);
  bool BPostDomA = MPDT->dominates(MBBB, MBBA);
  unsigned Mask = 0;
  if (!ADomB && !BPostDomA)
    return 0;

  /* A -> MBBA->end() */
  for (auto It = std::next(A->getIterator()); It != MBBA->end(); ++It)
    if (!isSafeToSkip(*It))
      return 0;
  /* B -> MBBB->begin() */
  for (auto It = MBBB->begin(); It != B->getIterator(); ++It)
    if (!isSafeToSkip(*It))
      return 0;

  /* MBBA -> MBBB */
  if (BPostDomA)
    if (checkAllPathSafe(MBBA, MBBB, true /*IsAToB*/))
      Mask |= CandidateA;

  /* MBBB -> MBBA */
  if (ADomB)
    if (checkAllPathSafe(MBBA, MBBB, false /*IsAToB*/))
      Mask |= CandidateB;

  return Mask;
}

// Update DBAR hint
static void updateMB(InstBarrier &I, BarrierHint Hint, MachineFunction *MF) {
  assert(I.IsMB && "Unexpected!");
  I.Pre = I.Post = Hint;
  if (!I.IsAsm) {
    I.MI->getOperand(0).setImm(Hint.Hint);
    return;
  }
  if (I.HintOff) {
    I.MI->getOperand(I.HintOff).setImm(Hint.Hint);
    return;
  }
  MachineOperand &MO = I.MI->getOperand(InlineAsm::MIOp_AsmString);
  auto New = I.OpName.str() + " " + llvm::utostr(Hint.Hint);
  auto Sym = MF->createExternalSymbolName(New);
  MO = MachineOperand::CreateES(Sym);
}

// Replace AMO to AMO_DB
static void replaceAM(InstBarrier &I, MachineFunction *MF) {
  if (!I.IsAM)
    return;
  if (I.OpcAMDB) {
    auto &ST = MF->getSubtarget<LoongArchSubtarget>();
    I.MI->setDesc(ST.getInstrInfo()->get(I.OpcAMDB));
    return;
  }
  if (!I.IsAsm)
    return;
  MachineOperand &MO = I.MI->getOperand(InlineAsm::MIOp_AsmString);
  auto New = I.OpName.str() + " " + I.Operands.str();
  auto Sym = MF->createExternalSymbolName(New);
  MO = MachineOperand::CreateES(Sym);
}

bool LoongArchMemoryBarrierOpt::eliminateRedundantBarrier(
    InstBarrier &IA, InstBarrier &IB) const {
  MachineInstr *A = IA.MI;
  MachineInstr *B = IB.MI;

  if (!A || !B)
    return false; // Already erased
  if (A == B)
    return false;

  unsigned Mask = resolveBarrierRedundancy(A, B);
  if (!Mask)
    return false;

  auto eraseOrReplaceWithNop = [&](MachineInstr *MI) {
    if (ReplaceEliminatedMBToNop) {
      auto &ST = MF->getSubtarget<LoongArchSubtarget>();
      BuildMI(*MI->getParent(), MI->getIterator(), MI->getDebugLoc(),
              ST.getInstrInfo()->get(LoongArch::ANDI), LoongArch::R0)
          .addReg(LoongArch::R0)
          .addImm(0);
    }
    MI->eraseFromParent();
  };

  // A        B
  // DBAR x + DBAR y -> DBAR (x & y)
  // DBAR x + AMO_DB -> AMO_DB
  // DBAR x + AMO    -> AMO_DB
  // DBAR x + LL     -> LL
  if ((Mask & CandidateA) && IA.IsMB) {
    if (!IB.Pre.subsumes(IA.Post)) {
      if (!IB.IsMB || (RequireNoPathBypass && !(Mask & CandidateB)))
        return false;
      updateMB(IB, IB.Pre.merge(IA.Post), MF);
    }
    replaceAM(IB, MF);
    eraseOrReplaceWithNop(A);
    IA.MI = nullptr;
    return true;
  }

  // A        B
  // DBAR x + DBAR y -> DBAR (x & y)
  // AMO_DB + DBAR x -> AMO_DB
  // AMO    + DBAR x -> AMO_DB
  // SC     + DBAR x -> SC
  if ((Mask & CandidateB) && IB.IsMB) {
    if (!IA.Post.subsumes(IB.Pre)) {
      if (!IA.IsMB || (RequireNoPathBypass && !(Mask & CandidateA)))
        return false;
      updateMB(IA, IA.Post.merge(IB.Pre), MF);
    }
    replaceAM(IA, MF);
    eraseOrReplaceWithNop(B);
    IB.MI = nullptr;
    return true;
  }

  return false;
}

bool LoongArchMemoryBarrierOpt::runOnMachineFunction(MachineFunction &Fn) {
  if (skipFunction(Fn.getFunction()))
    return false;

  MF = &Fn;
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  MPDT = &getAnalysis<MachinePostDominatorTreeWrapperPass>().getPostDomTree();

  SmallVector<InstBarrier, 32> Sites;
  bool Changed = false;

  for (MachineBasicBlock &MBB : Fn)
    for (MachineInstr &MI : MBB) {
      InstBarrier IB(MI);
      if (IB.IsMB || IB.IsAM)
        Sites.push_back(IB);
    }

  for (size_t a = 0; a < Sites.size(); ++a) {
    for (size_t b = a + 1; b < Sites.size(); ++b) {
      InstBarrier &IA = Sites[a];
      InstBarrier &IB = Sites[b];
      Changed |= eliminateRedundantBarrier(IA, IB);
      Changed |= eliminateRedundantBarrier(IB, IA);
    }
  }

  return Changed;
}
} // namespace

char LoongArchMemoryBarrierOpt::ID = 0;
INITIALIZE_PASS_BEGIN(LoongArchMemoryBarrierOpt, DEBUG_TYPE,
                      LOONGARCH_MEMORY_BARRIER_OPT_NAME, false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTreeWrapperPass)
INITIALIZE_PASS_END(LoongArchMemoryBarrierOpt, DEBUG_TYPE,
                    LOONGARCH_MEMORY_BARRIER_OPT_NAME, false, false)

FunctionPass *llvm::createLoongArchMemoryBarrierOptPass() {
  return new LoongArchMemoryBarrierOpt();
}
