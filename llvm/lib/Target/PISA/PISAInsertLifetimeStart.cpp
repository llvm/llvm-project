//=== PISAInsertLifetimeStart.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Inserts "lifetime.start %R;" markers at loop headers.
//
// What the marker means
// ---------------------
// A "lifetime.start %R;" at a loop header H asserts that the value register R
// holds on entry to the loop from the back-edge (the value carried from the
// previous iteration) is never used. In other words R is NOT loop-carried: each
// iteration produces a fresh R before reading it, so no value flows from one
// iteration to the next through R. Without the marker R looks alive across
// iterations and must be conservatively preserved; with it, R may be treated as
// undefined on entry to the loop.
//
// The pattern we detect
// ---------------------
// We look for a register R with:
//   - an undefined seed before the loop (an IMPLICIT_DEF, so R starts the loop
//     with no real value), and
//   - inside the loop, a real definition D that always runs before any use U of
//     R in the same iteration.
// When D always precedes U, every U reads the value D just produced this
// iteration, never the value carried across the back-edge -- so marking R is
// sound.
//
// Why this needs more than dominance
// ----------------------------------
// D need not dominate U. A common shape guards the definition block and the use
// block with the SAME predicate, so the loop header can branch around the
// definition block:
//
//     H:  goto.cond p -> M     ; p true skips the definition
//     D:  R = ...              ; runs only when p is false
//     M:  goto.cond p -> L     ; the same p skips the use
//     U:  ... = R              ; runs only when p is false
//     L:  latch
//
// Here the path H -> M -> U skips D, so D does not dominate U; but that path
// needs p both true (to skip D) and false (to reach U), which is impossible.
// So whenever U runs, D ran first. We capture this by comparing the predicate
// guards of D and U: U is covered by D when D runs before U and every predicate
// guarding D also guards U (so U running implies D ran). Values may reach U
// through pure repacks (copy / insert / extract / mov); we follow R forward
// through those to its real uses.
//
// Detection flow (per register R, per loop L with header H)
// ---------------------------------------------------------
//   1. R has an IMPLICIT_DEF seed.
//   2. R is live across the back-edge (live-in to H, live-out of a latch).
//   3. R's value does not escape the loop: neither R nor a value repacked from
//      it is observed after the loop (its carried value must not leak out).
//   4. The value reaching H from outside L is the IMPLICIT_DEF seed, not a real
//      value (an IMPLICIT_DEF dominates H, no real def dominates H, L has a
//      unique preheader).
//   5. Following R forward through repack ops, every real use U is inside L and
//      is covered by a real def D of R that runs before U (by dominance, or by
//      a shared predicate guard with D ordered before U).
// If all hold, emit the marker at H.
//
// Pass placement and liveness
// ---------------------------
// Runs in addPreEmitPass(), after register coalescing, so the marked vreg name
// matches the emitted name; gated to -O != none. LiveVariables cannot run here
// (the MIR is non-SSA at this slot), so liveness is recomputed by a small
// backward block-level dataflow. MachineLoopInfo / MachineDominatorTree /
// MachinePostDominatorTree are pure-CFG and valid here.
//===----------------------------------------------------------------------===//

#include "PISA.h"
#include "PISAInstrInfo.h"
#include "PISARegisterInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

// Generated register-class -> lifetime.start opcode lookup.
namespace llvm {
namespace PISA {
struct LifetimeStartEntry {
  // Since upstream commit 92f01b267efe ([TableGen] Use StringTable for
  // searchable tables), string columns in SearchableTables are emitted as
  // StringTable offsets (unsigned) rather than const char* pointers.
  unsigned RegClassName;
  unsigned Opcode;
};

#define GET_LifetimeStartTable_DECL
#define GET_LifetimeStartTable_IMPL
#include "PISAGenSearchableTables.inc"

} // namespace PISA
} // namespace llvm

#define DEBUG_TYPE "pisa-insert-lifetime-start"

STATISTIC(NumLifetimeMarkers, "Number of lifetime.start markers inserted");

// NB: the option name must differ from the pass registration name
// ("pisa-insert-lifetime-start", used by -run-pass / -start-after /
// -print-after and the DEBUG_TYPE). The legacy PassNameParser registers every
// pass's arg name as a CLI literal, so a cl::opt of the same name aborts every
// tool at startup ("registered more than once"). Hence the "-enable" suffix.
static cl::opt<bool> LifetimeStartOpt(
    "pisa-insert-lifetime-start-enable",
    cl::desc("Insert lifetime.start liveness markers at loop headers"),
    cl::init(true), cl::Hidden);

namespace {

// Per-block liveness as a BitVector indexed by Register::virtReg2Index (only
// virtual registers are tracked).
using LiveMap = DenseMap<const MachineBasicBlock *, BitVector>;

// A predicated branch (predgoto) "goto.cond <negate><cond>, <label>". The block
// ending in it has two CFG successors: the label target (taken when cond equals
// !negate) and the fall-through (taken when cond equals negate). PredCtrl
// records what is needed to attribute each outgoing edge to a (predicate,
// polarity) pair.
struct PredCtrl {
  MachineBasicBlock *Block; // block containing predgoto (source block)
  MachineBasicBlock *Label; // the explicit branch target operand
  Register Pred;            // the predicate register (cond)
  bool Negate;              // the negate flag (operand 0)
};

// One packed (predicate vreg, polarity) guard key, used for set membership.
using PredKey = uint64_t;

// Pack (predicate vreg, polarity) into one key for set membership.
inline PredKey predKey(Register P, bool Pol) {
  return (static_cast<PredKey>(P.id()) << 1) | (Pol ? 1u : 0u);
}

class PISAInsertLifetimeStart : public MachineFunctionPass {
public:
  static char ID;

  PISAInsertLifetimeStart();

  StringRef getPassName() const override {
    return "PISA Insert lifetime.start";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  MachineRegisterInfo *MRI = nullptr;
  const TargetInstrInfo *TII = nullptr;
  MachineDominatorTree *MDT = nullptr;
  MachinePostDominatorTree *MPDT = nullptr;
  // Every predicated branch in the function. Control-dependence is a whole-CFG
  // property, so this is gathered once and reused for all candidates.
  SmallVector<PredCtrl, 16> PredCtrls;

  void computeBlockLiveness(MachineFunction &MF, LiveMap &LiveIn,
                            LiveMap &LiveOut) const;

  bool isBackEdgeLive(Register R, const MachineLoop *L, const LiveMap &LiveIn,
                      const LiveMap &LiveOut) const;

  bool isLiveOutOfLoop(Register R, const MachineLoop *L,
                       const LiveMap &LiveIn) const;

  bool implicitDefSeedReachesHeader(const MachineRegisterInfo &MRI, Register R,
                                    const MachineLoop *L,
                                    const MachineDominatorTree &MDT) const;

  SmallVector<std::pair<const MachineBasicBlock *, PredKey>, 8>
  directCtrlDeps(const MachineBasicBlock *B) const;

  DenseSet<PredKey> transitiveCtrlSet(const MachineBasicBlock *B) const;

  bool isRealFullDef(const MachineInstr &MI, Register R) const;

  bool defRunsBeforeUse(const MachineInstr &D, const MachineInstr &U) const;

  bool reachesWithinIteration(const MachineBasicBlock *From,
                              const MachineBasicBlock *To,
                              const MachineLoop *L) const;

  bool hasTransitiveControlDependence(Register R, const MachineLoop *L) const;

  // Returns true iff any marker was inserted.
  bool run(MachineFunction &MF, MachineLoopInfo &MLI);
};

} // namespace

// Pick the typed marker variant for R's register class; 0 if none (then the
// caller skips R). The register-class -> opcode mapping is generated from
// VTs.LifetimeTypes (see LifetimeStartTable in PISAInstrInfo.td).
static unsigned pickLifetimeStartOpcode(const TargetRegisterInfo &TRI,
                                        const TargetRegisterClass *RC) {
  const auto *Entry =
      PISA::lookupLifetimeStartByRegClass(TRI.getRegClassName(RC));
  return Entry ? Entry->Opcode : 0;
}

// True if MI only repacks its operand bits -- copy / undef / insert / extract /
// mov -- so the value flows through it without being observed.
static bool isForwardingOpcode(const MachineInstr &MI,
                               const TargetInstrInfo &TII) {
  if (MI.isCopy() || MI.isImplicitDef())
    return true;
  StringRef Name = TII.getName(MI.getOpcode());
  return Name.starts_with("insert_") || Name.starts_with("extract_") ||
         Name.starts_with("mov_");
}

void PISAInsertLifetimeStart::getAnalysisUsage(AnalysisUsage &AU) const {
  // Both are pure-CFG / SSA-independent and valid at addPreEmitPass.
  // LiveVariables is deliberately NOT required: it rejects the non-SSA
  // MIR present at this slot (see file header) -- liveness is recomputed
  // manually.
  AU.addRequired<MachineLoopInfoWrapperPass>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  // Post-dominators back the control-dependence test.
  AU.addRequired<MachinePostDominatorTreeWrapperPass>();
  AU.setPreservesCFG(); // we only ever insert marker MIs
  MachineFunctionPass::getAnalysisUsage(AU);
}

PISAInsertLifetimeStart::PISAInsertLifetimeStart() : MachineFunctionPass(ID) {
  initializePISAInsertLifetimeStartPass(*PassRegistry::getPassRegistry());
}

// Manual block-level liveness over virtual registers (NOT LiveVariables --
// invalid at this slot, see file header). Fills LiveIn/LiveOut, each
// BitVector sized to MRI.getNumVirtRegs() and indexed by
// Register::virtReg2Index.
void PISAInsertLifetimeStart::computeBlockLiveness(MachineFunction &MF,
                                                   LiveMap &LiveIn,
                                                   LiveMap &LiveOut) const {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const unsigned NumV = MRI.getNumVirtRegs();

  // Local transfer functions, computed once per block:
  //   Def[B]   = registers fully defined somewhere in B.
  //   UpUse[B] = upward-exposed uses (read in B before any def in B).
  LiveMap Def, UpUse;
  for (MachineBasicBlock &MBB : MF) {
    BitVector DefB(NumV), UseB(NumV), DefSoFar(NumV);
    for (MachineInstr &MI : MBB) {
      if (MI.isDebugInstr())
        continue;
      // Uses first: a use reads the value defined by an EARLIER instruction, so
      // it is upward-exposed unless this block already (fully) defined it.
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.getReg().isVirtual() || !MO.readsReg())
          continue;
        unsigned I = MO.getReg().virtRegIndex();
        if (!DefSoFar.test(I))
          UseB.set(I);
      }
      // Then defs. A full def (subreg 0) kills upward liveness; a sub-register
      // def preserves the other bits, so it does not kill (and its read of the
      // remaining bits was already counted as a use above via readsReg()).
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.isDef() || !MO.getReg().isVirtual())
          continue;
        if (MO.getSubReg() != PISA::NoSubRegister)
          continue; // partial def: not a kill
        unsigned I = MO.getReg().virtRegIndex();
        DefB.set(I);
        DefSoFar.set(I);
      }
    }
    Def[&MBB] = std::move(DefB);
    UpUse[&MBB] = std::move(UseB);
    LiveIn[&MBB] = BitVector(NumV);
    LiveOut[&MBB] = BitVector(NumV);
  }

  // Backward iterative dataflow to fixpoint:
  //   LiveOut[B] = U_{S in succ(B)} LiveIn[S]
  //   LiveIn[B]  = UpUse[B] U (LiveOut[B] \ Def[B])
  bool Changed = true;
  while (Changed) {
    Changed = false;
    // Reverse layout order converges faster for mostly-forward CFGs.
    for (MachineBasicBlock &MBB : reverse(MF)) {
      BitVector Out(NumV);
      for (const MachineBasicBlock *Succ : MBB.successors())
        Out |= LiveIn[Succ];

      BitVector In = Out;
      In.reset(Def[&MBB]); // Out & ~Def
      In |= UpUse[&MBB];

      if (Out != LiveOut[&MBB]) {
        LiveOut[&MBB] = std::move(Out);
        Changed = true;
      }
      if (In != LiveIn[&MBB]) {
        LiveIn[&MBB] = std::move(In);
        Changed = true;
      }
    }
  }
}

// True iff R is live across L's back-edge: live-in to the header and live-out
// of at least one latch.
bool PISAInsertLifetimeStart::isBackEdgeLive(Register R, const MachineLoop *L,
                                             const LiveMap &LiveIn,
                                             const LiveMap &LiveOut) const {
  const unsigned I = R.virtRegIndex();
  const MachineBasicBlock *H = L->getHeader();
  auto HIt = LiveIn.find(H);
  if (HIt == LiveIn.end() || !HIt->second.test(I))
    return false; // not live-in to the header

  SmallVector<MachineBasicBlock *, 4> Latches;
  L->getLoopLatches(Latches);
  for (const MachineBasicBlock *Latch : Latches) {
    auto LIt = LiveOut.find(Latch);
    if (LIt != LiveOut.end() && LIt->second.test(I))
      return true; // live across the latch -> header back-edge
  }
  return false;
}

// True if R's value is observed after the loop (R live-in to a loop-exit
// block); such a register must not be marked because its carried value
// escapes.
bool PISAInsertLifetimeStart::isLiveOutOfLoop(Register R, const MachineLoop *L,
                                              const LiveMap &LiveIn) const {
  const unsigned I = R.virtRegIndex();
  SmallVector<MachineBasicBlock *, 4> ExitBlocks;
  L->getExitBlocks(ExitBlocks);
  for (const MachineBasicBlock *S : ExitBlocks) {
    auto It = LiveIn.find(S);
    if (It != LiveIn.end() && It->second.test(I))
      return true; // value escapes the loop -> observed after the loop
  }
  return false;
}

// True iff it is SOUND to assert R dead at L's header: the value of R that
// reaches the header from OUTSIDE the loop is the IMPLICIT_DEF undef seed,
// not a real (live) value. This uses a dominance approximation; it also
// subsumes the verifier requirement that the inserted header USE be dominated
// by a def.
//
// Requires ALL of:
//   (1) an IMPLICIT_DEF def of R STRICTLY dominates the header -- the undef
//       seed (a def in H itself sits at/after the insertion point at this
//       post-PHIElim slot, so it does not dominate the marker use);
//   (2) NO non-IMPLICIT_DEF def of R dominates the header -- a real
//       dominating def (e.g. a preheader load that seeds a loop-carried
//       reduction) is the live value entering the loop, so "dead at header"
//       would delete a live seed and miscompile; and
//   (3) the loop has a UNIQUE preheader, so the single loop-entry path makes
//       the closest dominating def the reaching def at the header. Without
//       it, a real def on one of several entry edges need not dominate the
//       header, so (2) could pass while a live value still reaches it.
//
// Defs INSIDE the loop body (the loop-carried redefinition the marker is
// entitled to discard) do not dominate the header and are correctly ignored.
bool PISAInsertLifetimeStart::implicitDefSeedReachesHeader(
    const MachineRegisterInfo &MRI, Register R, const MachineLoop *L,
    const MachineDominatorTree &MDT) const {
  const MachineBasicBlock *H = L->getHeader();

  // (3) Unique preheader -> a single loop-entry path into the header, so the
  // closest def dominating H is exactly the value reaching it from outside.
  if (!L->getLoopPreheader())
    return false;

  bool ImplicitDefDominates = false;
  for (const MachineInstr &DefMI : MRI.def_instructions(R)) {
    const MachineBasicBlock *DefBB = DefMI.getParent();
    // A def in H itself sits at or after the insertion point (this slot is
    // post-PHIElim / NoPHIs, so getFirstNonPHI() is the block top), so it does
    // not dominate the inserted use; only STRICT dominators of H carry a value
    // into the loop along the entry path.
    if (DefBB == H || !MDT.dominates(DefBB, H))
      continue;
    if (DefMI.isImplicitDef())
      ImplicitDefDominates = true; // (1) the undef seed dominates the header
    else
      return false; // (2) a real value reaches the header -> not dead on entry
  }
  return ImplicitDefDominates;
}

// Collect direct (immediate) control dependences of block B: which predicated
// branches directly decide whether B runs, together with the predicate
// polarity that forces B to execute. B is control-dependent on edge (A -> C)
// iff B post-dominates C and B does not (reflexively) post-dominate A.
SmallVector<std::pair<const MachineBasicBlock *, PredKey>, 8>
PISAInsertLifetimeStart::directCtrlDeps(const MachineBasicBlock *B) const {
  SmallVector<std::pair<const MachineBasicBlock *, PredKey>, 8> Deps;
  for (const PredCtrl &PC : PredCtrls) {
    if (MPDT->dominates(B, PC.Block))
      continue; // B post-dominates the branch site -> not control-dependent
    for (MachineBasicBlock *C : PC.Block->successors()) {
      if (!MPDT->dominates(B, C))
        continue;
      bool ToLabel = (C == PC.Label);
      // Edge to label is taken when cond == !negate; fall-through when
      // ==negate.
      bool Pol = ToLabel ? !PC.Negate : PC.Negate;
      Deps.push_back({PC.Block, predKey(PC.Pred, Pol)});
    }
  }
  return Deps;
}

// Transitive guard set of block B: every (pred,pol) that must hold for B to
// run, following the control-dependence chain. A block inherits the guards of
// each branch site it is control-dependent on. The Seen set makes the closure
// terminate even with cyclic control dependence (loops).
DenseSet<PredKey>
PISAInsertLifetimeStart::transitiveCtrlSet(const MachineBasicBlock *B) const {
  DenseSet<PredKey> S;
  SmallVector<const MachineBasicBlock *, 8> Work{B};
  DenseSet<const MachineBasicBlock *> Seen{B};
  while (!Work.empty()) {
    const MachineBasicBlock *X = Work.pop_back_val();
    for (auto [Site, Key] : directCtrlDeps(X)) {
      S.insert(Key);
      if (Seen.insert(Site).second)
        Work.push_back(Site);
    }
  }
  return S;
}

// A real (non-undef, non-debug) full (subreg-0) def of R. A repack op that
// fully defines R counts; what matters for soundness is ordering, not the
// opcode.
bool PISAInsertLifetimeStart::isRealFullDef(const MachineInstr &MI,
                                            Register R) const {
  if (MI.isImplicitDef() || MI.isDebugInstr())
    return false;
  for (const MachineOperand &MO : MI.operands())
    if (MO.isReg() && MO.isDef() && MO.getReg() == R &&
        MO.getSubReg() == PISA::NoSubRegister)
      return true;
  return false;
}

// True iff D is executed before U within an iteration: D's block dominates
// U's block, or (same block) D is the earlier instruction. With a shared
// guard (or on its own) this proves R holds a same-iteration value where U
// observes it.
bool PISAInsertLifetimeStart::defRunsBeforeUse(const MachineInstr &D,
                                               const MachineInstr &U) const {
  const MachineBasicBlock *DB = D.getParent(), *UB = U.getParent();
  if (DB != UB)
    return MDT->dominates(DB, UB);
  for (const MachineInstr &MI : *DB) {
    if (&MI == &D)
      return true;
    if (&MI == &U)
      return false;
  }
  return false;
}

// True if block To is reachable from block From along intra-loop edges
// without passing through L's header -- i.e. To runs after From in the
// iteration (the header is the only re-entry point, so excluding it cuts
// every back-edge of L). Used by the cover below to reject a def that,
// although it shares the observation's guard, runs after the observation (so
// the observation still sees the back-edge value).
bool PISAInsertLifetimeStart::reachesWithinIteration(
    const MachineBasicBlock *From, const MachineBasicBlock *To,
    const MachineLoop *L) const {
  const MachineBasicBlock *H = L->getHeader();
  SmallVector<const MachineBasicBlock *, 8> Work;
  DenseSet<const MachineBasicBlock *> Seen;
  for (const MachineBasicBlock *S : From->successors())
    if (L->contains(S) && S != H && Seen.insert(S).second)
      Work.push_back(S);
  while (!Work.empty()) {
    const MachineBasicBlock *B = Work.pop_back_val();
    if (B == To)
      return true;
    for (const MachineBasicBlock *S : B->successors())
      if (L->contains(S) && S != H && Seen.insert(S).second)
        Work.push_back(S);
  }
  return false;
}

// Per-use cover test. R's value reaches its real observations through
// forwarding ops (copy/insert/extract/mov) that only repack it, so we follow
// R forward through them and require that every real use U of a value derived
// from R is covered: a real def D of R runs before U and D's guards are
// implied by U's guards, so U always sees this iteration's value rather than
// the back-edge value. D ordered before U is checked by dominance or, for a
// shared guard, by D not running after U within the iteration
// (reachesWithinIteration) -- without that ordering U could read the
// back-edge value before D rewrites it. Guards are compared with the
// transitive guard set, so a deep observation links back to the predicate
// guarding the def even when intervening blocks test other predicates.
//
// A register with no real def (pass-through), or an observation no def
// covers, is rejected. A use OUTSIDE the loop L is always rejected: a value
// derived from R that is read after the loop is the carried value escaping.
// This catches escapes through a different register than R itself, which the
// isLiveOutOfLoop(R) gate -- keyed on R alone -- cannot see.
bool PISAInsertLifetimeStart::hasTransitiveControlDependence(
    Register R, const MachineLoop *L) const {
  SmallVector<const MachineInstr *, 4> RDefs;
  for (const MachineInstr &MI : MRI->def_instructions(R))
    if (isRealFullDef(MI, R))
      RDefs.push_back(&MI);
  if (RDefs.empty())
    return false; // pass-through: nothing redefines R in the loop

  SmallVector<Register, 8> Work{R};
  DenseSet<Register> Visited;
  bool SawObservation = false;
  while (!Work.empty()) {
    Register V = Work.pop_back_val();
    if (!Visited.insert(V).second)
      continue; // already processed
    for (const MachineInstr &U : MRI->use_instructions(V)) {
      if (U.isDebugInstr())
        continue;
      if (isForwardingOpcode(U, *TII)) {
        // Repack: follow the produced virtual reg(s); it observes nothing.
        for (const MachineOperand &MO : U.operands())
          if (MO.isReg() && MO.isDef() && MO.getReg().isVirtual())
            Work.push_back(MO.getReg());
        continue;
      }
      // Real observation of a value derived from R.
      SawObservation = true;
      if (!L->contains(U.getParent()))
        return false; // observed after the loop -> escaping back-edge value
      DenseSet<PredKey> CU = transitiveCtrlSet(U.getParent());
      bool Covered = false;
      for (const MachineInstr *D : RDefs) {
        // A def cannot cover its OWN read: a read-modify-write of R (e.g. an
        // accumulator `R = iadd R, x`) reads the incoming/back-edge value
        // before this instruction's def, so it must stay an uncovered
        // observation.
        if (D == &U)
          continue;
        if (defRunsBeforeUse(*D, U)) { // dominated -> fresh, predicate-free
          Covered = true;
          break;
        }
        // Guard-containment cover: D and U live in different blocks. Only
        // sound if D is NOT after U within the iteration -- otherwise U reads
        // the back-edge value before D rewrites it. (Same-block D-after-U was
        // already settled false by defRunsBeforeUse, and
        // reachesWithinIteration is block-granular, so skip it there.)
        const MachineBasicBlock *DB = D->getParent(), *UB = U.getParent();
        if (DB == UB || reachesWithinIteration(UB, DB, L))
          continue; // D runs after U -> cannot cover this observation
        DenseSet<PredKey> CD = transitiveCtrlSet(DB);
        if (CD.empty())
          continue; // unconditional D is covered only via the dominance
                    // branch
        // Containment: every guard of D also guards U, so U running implies D
        // ran the same iteration.
        bool Implies = true;
        for (PredKey K : CD) {
          // predKey identifies a guard by (predicate vreg, polarity) only. That
          // is a value identity ONLY when the predicate vreg has a single
          // reaching def: this MIR is non-SSA (post-PHIElim / coalescing), so a
          // predicate vreg may be redefined and hold different values at
          // different branch sites. If D's guard predicate is multiply defined,
          // a matching key in CU can come from a branch testing a DIFFERENT
          // value, so "U ran => D ran" no longer follows. Refuse the cover.
          Register P = Register(static_cast<unsigned>(K >> 1));
          if (!MRI->hasOneDef(P)) {
            Implies = false;
            break;
          }
          if (!CU.contains(K)) {
            Implies = false;
            break;
          }
        }
        if (Implies)
          Covered = true;
        if (Covered)
          break;
      }
      if (!Covered)
        return false; // observed outside R's def guard -> back-edge value
                      // live
    }
  }
  // SawObservation stays false only for an unobservable register -> reject.
  return SawObservation;
}

bool PISAInsertLifetimeStart::run(MachineFunction &MF, MachineLoopInfo &MLI) {
  const unsigned NumV = MRI->getNumVirtRegs();
  if (NumV == 0 || MLI.empty())
    return false;

  // Candidate signal: a vreg with an IMPLICIT_DEF def (undef-on-entry seed).
  BitVector HasImplicitDef(NumV);
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (MI.isImplicitDef()) {
        Register R = MI.getOperand(0).getReg();
        if (R.isVirtual())
          HasImplicitDef.set(R.virtRegIndex());
      }
  if (HasImplicitDef.none())
    return false;

  // Collect every predicated branch in the function. Control-dependence is a
  // whole-CFG property, so this is gathered once and reused for all candidates.
  PredCtrls.clear();
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (MI.getOpcode() == PISA::predgoto) {
        Register P = MI.getOperand(1).getReg();
        if (!P.isVirtual())
          continue;
        PredCtrls.push_back({&MBB, MI.getOperand(2).getMBB(), P,
                             MI.getOperand(0).getImm() != 0});
      }
  if (PredCtrls.empty())
    return false;

  // Visit loops innermost-first, dedup globally.
  SmallVector<MachineLoop *, 8> Loops;
  for (MachineLoop *TopL : MLI) {
    SmallVector<MachineLoop *, 8> WL{TopL};
    while (!WL.empty()) {
      MachineLoop *L = WL.pop_back_val();
      Loops.push_back(L);
      WL.append(L->begin(), L->end());
    }
  }
  llvm::stable_sort(Loops, [](const MachineLoop *A, const MachineLoop *B) {
    return A->getLoopDepth() > B->getLoopDepth();
  });

  LiveMap LiveIn, LiveOut;
  computeBlockLiveness(MF, LiveIn, LiveOut);

  BitVector Marked(NumV);
  bool Changed = false;

  for (MachineLoop *L : Loops) {
    MachineBasicBlock *H = L->getHeader();
    MachineBasicBlock::iterator InsertPt = H->getFirstNonPHI();
    for (unsigned I : HasImplicitDef.set_bits()) {
      if (Marked.test(I))
        continue; // already processed
      Register R = Register::index2VirtReg(I);
      if (!isBackEdgeLive(R, L, LiveIn, LiveOut))
        continue;
      if (isLiveOutOfLoop(R, L, LiveIn)) {
        // R escapes the loop: its exit value is observed after the loop and is
        // (or depends on) the carried value, so "dead at the header" is
        // unsound.
        LLVM_DEBUG(dbgs() << "[" DEBUG_TYPE "] skip " << printReg(R, nullptr)
                          << ": live-out of the loop (escapes to a post-loop "
                             "use)\n");
        continue;
      }
      if (!implicitDefSeedReachesHeader(*MRI, R, L, *MDT)) {
        // Either the header marker would be a use not dominated by any def (R's
        // only seed/def is inside the loop body), or a REAL def reaches the
        // header from outside the loop -- i.e. a live value enters the loop and
        // "dead at header" would be a miscompile.
        LLVM_DEBUG(dbgs() << "[" DEBUG_TYPE "] skip " << printReg(R, nullptr)
                          << ": IMPLICIT_DEF seed does not reach the loop "
                             "header (real def reaches it, or no dominating "
                             "seed / no unique preheader)\n");
        continue;
      }
      if (!hasTransitiveControlDependence(R, L)) {
        LLVM_DEBUG(dbgs() << "[" DEBUG_TYPE "] skip " << printReg(R, nullptr)
                          << ": an observation is not covered by a guarded "
                             "preceding def (transitive)\n");
        continue;
      }
      unsigned Opc = pickLifetimeStartOpcode(*MRI->getTargetRegisterInfo(),
                                             MRI->getRegClass(R));
      if (!Opc) {
        LLVM_DEBUG(dbgs() << "[" DEBUG_TYPE "] skip " << printReg(R, nullptr)
                          << ": no lifetime.start variant for its reg class\n");
        continue;
      }
      BuildMI(*H, InsertPt, DebugLoc(), TII->get(Opc)).addDef(R);
      Marked.set(I);
      ++NumLifetimeMarkers;
      Changed = true;
      LLVM_DEBUG(dbgs() << "[" DEBUG_TYPE "] mark " << printReg(R, nullptr)
                        << " at " << printMBBReference(*H) << " (loop depth "
                        << L->getLoopDepth() << ")\n");
    }
  }

  LLVM_DEBUG(dbgs() << "[" DEBUG_TYPE "] " << MF.getName() << ": "
                    << Marked.count() << " marker(s) inserted\n");
  return Changed;
}

bool PISAInsertLifetimeStart::runOnMachineFunction(MachineFunction &MF) {
  if (!LifetimeStartOpt)
    return false; // default: insert nothing

  MRI = &MF.getRegInfo();
  TII = MF.getSubtarget().getInstrInfo();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  MPDT = &getAnalysis<MachinePostDominatorTreeWrapperPass>().getPostDomTree();
  MachineLoopInfo &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  return run(MF, MLI);
}

char PISAInsertLifetimeStart::ID = 0;
INITIALIZE_PASS_BEGIN(PISAInsertLifetimeStart, DEBUG_TYPE,
                      "PISA insert lifetime.start", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTreeWrapperPass)
INITIALIZE_PASS_END(PISAInsertLifetimeStart, DEBUG_TYPE,
                    "PISA insert lifetime.start", false, false)

namespace llvm {
FunctionPass *createPISAInsertLifetimeStart() {
  return new PISAInsertLifetimeStart();
}
} // namespace llvm
