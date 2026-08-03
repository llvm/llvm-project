//=== PISALegalizePredicates.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unified pre-legalization strategy for bitwise operations on predicate
// (i1) values. PISA has no bitwise operations on predicate registers, and
// `cmp` instructions yield i32 -1/0 values. The legalizer would
// widen i1 chains piecewise (i1 -> i32 for bitwise, i8/i16 for other sub-i16
// types), leaving redundant ext/trunc pairs and missing fusion opportunities
// (e.g. bfn).
//
// This pass walks forward from every G_ICMP/G_FCMP result and rewrites any
// chain of bitwise-on-i1, sext/zext/anyext-of-i1, vector-extract from a
// vector-of-i1, and trunc-back-to-i1 to operate uniformly on i32. Consumers
// that still need an i1 (G_BRCOND, G_SELECT condition, etc.) are fed via a
// single G_ICMP ne 0 restoration at the boundary.
//
// Pipeline placement: scheduled in addPreLegalizeMachineIR(); the pass only
// rewrites scalar/vector s1 chains and leaves other pre-legalization results
// undisturbed.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/PISAMCTargetDesc.h"
#include "PISA.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "pisa-legalize-predicates"
#define DEBUG_NAME "PISA Legalize Predicates"

using namespace llvm;

STATISTIC(NumPromoted, "Number of registers promoted to s32");
STATISTIC(NumRestorations, "Number of sink restorations inserted");
STATISTIC(NumErased, "Number of dead original instructions erased");

namespace {

// =====================  Pass boilerplate  =====================

class PISALegalizePredicates : public MachineFunctionPass {
public:
  static char ID;

  PISALegalizePredicates() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return DEBUG_NAME; }
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;
};

// =====================  Implementation  =====================

class PISALegalizePredicatesImpl {
  MachineFunction &MF;
  MachineRegisterInfo &MRI;
  MachineIRBuilder B;

  // Map from each promotable original Register to its s32 replacement.
  // Scalar s1 -> s32; <Nxs1> -> <Nxs32>; sN (N>1) coming from sext-of-s1 ->
  // s32.
  DenseMap<Register, Register> Promoted;

  // Cache: classification status for registers we've already inspected.
  enum Status { Unknown, Yes, No };
  DenseMap<Register, Status> Cache;

  // Magnitude domain of a register's *original* value. The internal promoted
  // form is always the SEXT representation ({0, -1}); the domain records what
  // the value was before promotion so the exact bits can be restored at a
  // non-i1 (integer) sink:
  //   Sext -- true == all-ones ({0, -1}); e.g. OpenCL vector relational.
  //   Zext -- true == 1        ({0,  1}); e.g. OpenCL scalar relational.
  //   Any  -- no magnitude (a 1-bit value or a literal 0); matches either side
  //           of a bitwise op.
  // A bitwise op mixing Sext and Zext operands has an ambiguous result
  // magnitude and is excluded from promotion (tryGetDomain returns false).
  enum class Domain : unsigned char { Any, Sext, Zext };
  // Memo codes: 0 = visiting (cycle guard), 1 = Any, 2 = Sext, 3 = Zext,
  // 4 = bail.
  DenseMap<Register, unsigned char> DomMemo;

public:
  PISALegalizePredicatesImpl(MachineFunction &MF)
      : MF(MF), MRI(MF.getRegInfo()) {
    B.setMF(MF);
  }

  bool run();

private:
  // ---- Type helpers ----
  // Returns true for s1 or <N x s1>; getScalarSizeInBits() returns 1 in
  // either case.
  static bool isBoolType(LLT Ty) { return Ty.getScalarSizeInBits() == 1; }
  static LLT promotedType(LLT Ty) {
    // s1 -> s32; <Nxs1> -> <Nxs32>; sN -> s32; <NxsM> -> <Nxs32>.
    LLT S32 = LLT::integer(32);
    if (Ty.isVector())
      return LLT::fixed_vector(Ty.getNumElements(), S32);
    return S32;
  }

  // Opcodes whose originals stay alive in MIR after rewrite (the new s32
  // chain is built alongside, not in place).
  static bool isKeptAliveOpcode(unsigned Opc) {
    return Opc == TargetOpcode::G_ICMP || Opc == TargetOpcode::G_FCMP ||
           Opc == TargetOpcode::G_CONSTANT || Opc == TargetOpcode::COPY;
  }

  // ---- Constant value helpers ----
  // Boolean-domain test: returns true iff the constant is 0 (false),
  // 1 (interpreted as true on s1), or all-ones (-1 on wider). Other constant
  // values disqualify promotion because they aren't predicate values. Caller
  // must guarantee MI is G_CONSTANT.
  static bool isBoolConst(const MachineInstr &MI, int64_t &PromotedVal) {
    assert(MI.getOpcode() == TargetOpcode::G_CONSTANT);
    const APInt &V = MI.getOperand(1).getCImm()->getValue();
    if (V.isZero()) {
      PromotedVal = 0;
      return true;
    }
    // For wider types -1 is "all-ones"; for i1 the bit "true" is value 1 and
    // also already covered by isAllOnes (1-bit all-ones == 1). Either way map
    // to -1 in the promoted s32 form.
    if (V.isAllOnes()) {
      PromotedVal = -1;
      return true;
    }
    return false;
  }

  // ---- Classification (Phase 1) ----
  // Returns true if R is part of the promotable chain rooted at some cmp.
  bool isPromotable(Register R);

  // ---- Use validation (Phase 2) ----
  // Returns true if a use of R inside UseInstr can be handled (either the use
  // is itself promotable, or we know how to insert a sink restoration).
  bool isHandledUse(MachineInstr &UseInstr, Register R);

  // Drop R from the promotion set transitively if any of its uses is
  // unhandled. Re-iterates to fix-point.
  void invalidateUnhandled();

  // Compute the original magnitude domain of R (see Domain above). Returns
  // false if the magnitude is ambiguous (a bitwise op mixing Sext and Zext
  // operands), meaning R cannot be safely restored at an integer sink and must
  // not be promoted. Memoized in DomMemo.
  bool tryGetDomain(Register R, Domain &Out);

  // ---- Rewrite (Phase 3) ----
  Register getOrBuildPromoted(Register R);
  Register restoreS1(Register PromotedReg, MachineInstr &InsertBeforeMI);
  void eraseDeadOriginals();

  // Inserts B at the right point for MI's *next* instruction.
  void setInsertPointAfter(MachineInstr &MI) {
    B.setInsertPt(*MI.getParent(), std::next(MI.getIterator()));
  }
  // Inserts B right before MI.
  void setInsertPointBefore(MachineInstr &MI) {
    B.setInsertPt(*MI.getParent(), MI.getIterator());
  }
};

// -----------------------------------------------------------------------------
// Phase 1: classification.
// -----------------------------------------------------------------------------

bool PISALegalizePredicatesImpl::isPromotable(Register R) {
  auto It = Cache.find(R);
  if (It != Cache.end())
    return It->second == Yes;

  // Mark as "currently visiting" by inserting No first. Cycles (e.g. through
  // a G_PHI) thus terminate as not-promotable.
  Cache[R] = No;

  MachineInstr *Def = MRI.getVRegDef(R);
  assert(Def && "expected a def for every virtual register in SSA MIR");

  LLT Ty = MRI.getType(R);
  bool Result = false;

  switch (Def->getOpcode()) {
  case TargetOpcode::G_ICMP:
  case TargetOpcode::G_FCMP:
    // Seed: a cmp whose result is i1 (scalar or vector); isBoolType covers
    // both since getScalarSizeInBits == 1 for s1 and <N x s1>.
    Result = isBoolType(Ty);
    break;
  case TargetOpcode::G_AND:
  case TargetOpcode::G_OR:
  case TargetOpcode::G_XOR:
    // Promotable if both operands are promotable (or boolean-domain consts).
    Result = isPromotable(Def->getOperand(1).getReg()) &&
             isPromotable(Def->getOperand(2).getReg());
    break;
  case TargetOpcode::G_SEXT:
  case TargetOpcode::G_ZEXT:
  case TargetOpcode::G_ANYEXT:
  case TargetOpcode::G_TRUNC:
  case TargetOpcode::COPY:
    Result = isPromotable(Def->getOperand(1).getReg());
    break;
  case TargetOpcode::G_EXTRACT_VECTOR_ELT:
    Result = isPromotable(Def->getOperand(1).getReg());
    break;
  case TargetOpcode::G_CONSTANT: {
    int64_t Dummy;
    Result = isBoolConst(*Def, Dummy);
    break;
  }
  default:
    break;
  }

  Cache[R] = Result ? Yes : No;
  return Result;
}

// -----------------------------------------------------------------------------
// Phase 2: validate uses; demote any register whose uses we can't handle.
// -----------------------------------------------------------------------------

bool PISALegalizePredicatesImpl::isHandledUse(MachineInstr &UseInstr,
                                              Register R) {
  // If UseInstr is itself promotable (its def is in the chain), the operand
  // will be rewritten in lock-step with the def.
  if (UseInstr.getNumDefs() >= 1 &&
      Cache.lookup(UseInstr.getOperand(0).getReg()) == Yes)
    return true;

  // Otherwise, this is a sink. We need to be able to restore the original
  // value at the boundary. For vector promotable defs feeding a
  // non-promotable user, we don't try to vector-restore in v1. Bail.
  if (MRI.getType(R).isVector())
    return false;

  // Scalar restoration: G_ICMP ne s32, 0 -> s1, or G_TRUNC s32 -> sN.
  // Both are always materializable, so any scalar use is "handled" in
  // principle, but we still skip a few opcodes that need special care:
  switch (UseInstr.getOpcode()) {
  case TargetOpcode::G_PHI:
    // PHI of i1 across blocks requires inserting restoration in the
    // predecessor blocks. v1 conservatively skips.
    return false;
  default:
    return true;
  }
}

void PISALegalizePredicatesImpl::invalidateUnhandled() {
  bool Changed = true;
  while (Changed) {
    Changed = false;
    for (auto &E : Cache) {
      if (E.second != Yes)
        continue;
      Register R = E.first;

      // (a) All uses of R must be handled (in-set or known sink).
      bool OK = true;
      for (MachineOperand &U : MRI.use_nodbg_operands(R)) {
        MachineInstr *UseInstr = U.getParent();
        if (!isHandledUse(*UseInstr, R)) {
          OK = false;
          break;
        }
      }
      if (!OK) {
        E.second = No;
        Changed = true;
        continue;
      }

      // (b) All required operands of R's def must still be promotable.
      // If an operand was just demoted above, cascade the demotion so that
      // Phase 3 doesn't recursively try to promote a non-promotable reg.
      MachineInstr *Def = MRI.getVRegDef(R);
      assert(Def);
      auto IsOperandPromotable = [&](Register Op) {
        return Cache.lookup(Op) == Yes;
      };
      switch (Def->getOpcode()) {
      case TargetOpcode::G_AND:
      case TargetOpcode::G_OR:
      case TargetOpcode::G_XOR: {
        if (!IsOperandPromotable(Def->getOperand(1).getReg()) ||
            !IsOperandPromotable(Def->getOperand(2).getReg())) {
          E.second = No;
          Changed = true;
        }
        break;
      }
      case TargetOpcode::G_SEXT:
      case TargetOpcode::G_ZEXT:
      case TargetOpcode::G_ANYEXT:
      case TargetOpcode::G_TRUNC:
      case TargetOpcode::G_EXTRACT_VECTOR_ELT:
      case TargetOpcode::COPY: {
        if (!IsOperandPromotable(Def->getOperand(1).getReg())) {
          E.second = No;
          Changed = true;
        }
        break;
      }
      // Seeds (G_ICMP, G_FCMP) and G_CONSTANT don't depend on the
      // promotability of any operand.
      default:
        break;
      }
    }
  }
}

bool PISALegalizePredicatesImpl::tryGetDomain(Register R, Domain &Out) {
  auto It = DomMemo.find(R);
  if (It != DomMemo.end()) {
    switch (It->second) {
    case 1:
      Out = Domain::Any;
      return true;
    case 2:
      Out = Domain::Sext;
      return true;
    case 3:
      Out = Domain::Zext;
      return true;
    default: // 0 (cycle) or 4 (bail)
      return false;
    }
  }
  DomMemo[R] = 0; // visiting -- a cycle reaching back here resolves to bail.
  auto Set = [&](Domain D) {
    DomMemo[R] = D == Domain::Any ? 1 : D == Domain::Sext ? 2 : 3;
    Out = D;
    return true;
  };
  auto Bail = [&]() {
    DomMemo[R] = 4;
    return false;
  };

  // A 1-bit value (scalar s1 or <N x s1>) carries no magnitude; the extend
  // that consumes it decides the domain.
  if (MRI.getType(R).getScalarSizeInBits() == 1)
    return Set(Domain::Any);

  MachineInstr *Def = MRI.getVRegDef(R);
  switch (Def->getOpcode()) {
  case TargetOpcode::G_ZEXT:
    return Set(Domain::Zext);
  case TargetOpcode::G_SEXT:
  case TargetOpcode::G_ANYEXT:
    // ANYEXT leaves the high bits undefined, so the sext form ({0, -1}) is an
    // acceptable refinement.
    return Set(Domain::Sext);
  case TargetOpcode::G_CONSTANT: {
    const APInt &V = Def->getOperand(1).getCImm()->getValue();
    // 0 matches any domain; all-ones is a true value of -1 (Sext). Other
    // constants are not classified promotable, so never reach here.
    return Set(V.isZero() ? Domain::Any : Domain::Sext);
  }
  case TargetOpcode::G_TRUNC:
  case TargetOpcode::COPY:
  case TargetOpcode::G_EXTRACT_VECTOR_ELT: {
    Domain D;
    if (!tryGetDomain(Def->getOperand(1).getReg(), D))
      return Bail();
    return Set(D);
  }
  case TargetOpcode::G_AND:
  case TargetOpcode::G_OR:
  case TargetOpcode::G_XOR: {
    Domain A, BD;
    if (!tryGetDomain(Def->getOperand(1).getReg(), A) ||
        !tryGetDomain(Def->getOperand(2).getReg(), BD))
      return Bail();
    if (A == Domain::Any)
      return Set(BD);
    if (BD == Domain::Any)
      return Set(A);
    if (A == BD)
      return Set(A);
    return Bail(); // mixed sext/zext -- ambiguous magnitude.
  }
  default:
    // tryGetDomain is only ever called on registers in the promotable closure,
    // whose defs are one of the opcodes handled above (i1-typed values, incl.
    // G_ICMP/G_FCMP results, are handled before the switch).
    llvm_unreachable("magnitude domain queried for a non-promotable opcode");
  }
}

// -----------------------------------------------------------------------------
// Phase 3: rewrite.
// -----------------------------------------------------------------------------

Register PISALegalizePredicatesImpl::getOrBuildPromoted(Register R) {
  auto It = Promoted.find(R);
  if (It != Promoted.end())
    return It->second;

  assert(isPromotable(R) && "asked to promote a non-promotable reg");

  MachineInstr *Def = MRI.getVRegDef(R);
  assert(Def && "expected a def");
  LLT OrigTy = MRI.getType(R);
  LLT NewTy = promotedType(OrigTy);

  // Allocate the destination s32 register up front so cyclic chains (none
  // expected, but safe) terminate cleanly via Promoted lookup.
  Register Dst = MRI.createGenericVirtualRegister(NewTy);
  Promoted[R] = Dst;
  LLVM_DEBUG(dbgs() << "  Promoting %" << R.virtRegIndex() << " ("
                    << MRI.getType(R) << " -> " << NewTy << "): " << *Def);
  ++NumPromoted;

  switch (Def->getOpcode()) {
  case TargetOpcode::G_ICMP: {
    // Peephole for the common reduction pattern:
    //   %p = ... (promotable, in s32 represents 0 or -1)
    //   %r = G_ICMP eq/ne %p, <0 or all-ones>
    // The cmp's promoted s32 form is just Promoted(%p) (or its XOR -1)
    // depending on the predicate and constant -- skip the
    // trunc -> icmp -> sext detour the legalizer would otherwise produce.
    // The eq/ne peephole below materializes scalar values (a scalar -1 / xor
    // and a scalar i1 compare), so it only applies to a scalar cmp result. A
    // vector cmp (<N x i1>) falls through to the vector-safe G_SEXT default.
    CmpInst::Predicate Pred =
        (CmpInst::Predicate)Def->getOperand(1).getPredicate();
    if (!MRI.getType(R).isVector() &&
        (Pred == CmpInst::ICMP_EQ || Pred == CmpInst::ICMP_NE)) {
      Register LhsReg = Def->getOperand(2).getReg();
      Register RhsReg = Def->getOperand(3).getReg();

      auto ClassifyOperands = [&](Register A, Register B, Register &PromOp,
                                  bool &IsAllOnes) -> bool {
        if (!isPromotable(A))
          return false;
        MachineInstr *BDef = MRI.getVRegDef(B);
        if (BDef->getOpcode() != TargetOpcode::G_CONSTANT)
          return false;
        const APInt &V = BDef->getOperand(1).getCImm()->getValue();
        if (V.isZero()) {
          PromOp = A;
          IsAllOnes = false;
          return true;
        }
        if (V.isAllOnes()) {
          PromOp = A;
          IsAllOnes = true;
          return true;
        }
        return false;
      };

      Register PromOp;
      bool IsAllOnes = false;
      if (ClassifyOperands(LhsReg, RhsReg, PromOp, IsAllOnes) ||
          ClassifyOperands(RhsReg, LhsReg, PromOp, IsAllOnes)) {
        Register PromS32 = getOrBuildPromoted(PromOp);
        // eq(p, all_ones) == p ; eq(p, 0) == ~p
        // ne(p, 0)        == p ; ne(p, all_ones) == ~p
        bool Negate = (Pred == CmpInst::ICMP_EQ) ^ IsAllOnes;
        if (!Negate) {
          Promoted[R] = PromS32;
          return PromS32;
        }
        setInsertPointAfter(*Def);
        auto AllOnesC = B.buildConstant(LLT::integer(32), -1);
        B.buildXor(Dst, PromS32, AllOnesC.getReg(0));
        break;
      }

      // Peephole: both operands are promotable (e.g. icmp eq of two AND
      // chains). Compare the promoted s32 forms directly. Both are in the
      // {0, -1} domain, so eq/ne is preserved.
      if (isPromotable(LhsReg) && isPromotable(RhsReg)) {
        Register PromLHS = getOrBuildPromoted(LhsReg);
        Register PromRHS = getOrBuildPromoted(RhsReg);
        setInsertPointAfter(*Def);
        Register CmpS32 = MRI.createGenericVirtualRegister(LLT::integer(1));
        B.buildICmp(Pred, CmpS32, PromLHS, PromRHS);
        B.buildSExt(Dst, CmpS32);
        break;
      }
    }
    // Default: no peephole matched; promote via sext.
    setInsertPointAfter(*Def);
    B.buildSExt(Dst, R);
    break;
  }
  case TargetOpcode::G_FCMP: {
    // Materialize one G_SEXT of the cmp's i1 def to s32 (scalar or vector),
    // right after the cmp. We keep the original cmp alive -- any non-promoted
    // sink (e.g. G_BRCOND) keeps using its i1 def directly.
    setInsertPointAfter(*Def);
    B.buildSExt(Dst, R);
    break;
  }
  case TargetOpcode::G_CONSTANT: {
    int64_t Val = 0;
    bool OK = isBoolConst(*Def, Val);
    (void)OK;
    assert(OK);
    setInsertPointAfter(*Def);
    auto NewC = B.buildConstant(NewTy, Val);
    Promoted[R] = NewC.getReg(0);
    return NewC.getReg(0);
  }
  case TargetOpcode::G_AND:
  case TargetOpcode::G_OR:
  case TargetOpcode::G_XOR: {
    Register LHS = getOrBuildPromoted(Def->getOperand(1).getReg());
    Register RHS = getOrBuildPromoted(Def->getOperand(2).getReg());
    setInsertPointBefore(*Def);
    B.buildInstr(Def->getOpcode(), {Dst}, {LHS, RHS});
    break;
  }
  case TargetOpcode::G_SEXT:
  case TargetOpcode::G_ZEXT:
  case TargetOpcode::G_ANYEXT:
  case TargetOpcode::G_TRUNC: {
    // The promoted source is already an s32 (or <Nxs32>) representation of
    // a predicate-domain value (0 / all-ones). The corresponding promoted
    // form for *this* instruction is the same s32 representation, so just
    // alias to the source's promoted reg.
    Register Src = getOrBuildPromoted(Def->getOperand(1).getReg());
    Promoted[R] = Src;
    return Src;
  }
  case TargetOpcode::G_EXTRACT_VECTOR_ELT: {
    // Replace the vector source with its <Nxs32> promoted form and extract
    // the same index at s32.
    Register VecS32 = getOrBuildPromoted(Def->getOperand(1).getReg());
    Register Idx = Def->getOperand(2).getReg();
    setInsertPointBefore(*Def);
    B.buildExtractVectorElement(Dst, VecS32, Idx);
    break;
  }
  case TargetOpcode::COPY: {
    Register Src = getOrBuildPromoted(Def->getOperand(1).getReg());
    Promoted[R] = Src;
    return Src;
  }
  }
  // The assert at the top of this function -- isPromotable(R) -- guarantees
  // the opcode is one of the cases handled above.

  return Dst;
}

Register PISALegalizePredicatesImpl::restoreS1(Register PromotedReg,
                                               MachineInstr &InsertBeforeMI) {
  // Build the restoration just before the sink so it trivially dominates.
  // Each sink gets its own restoration -- sharing across sinks would need
  // dominance analysis and is not worth the complexity in v1.
  setInsertPointBefore(InsertBeforeMI);
  auto Zero = B.buildConstant(LLT::integer(32), 0);
  Register Dst = MRI.createGenericVirtualRegister(LLT::integer(1));
  B.buildICmp(CmpInst::ICMP_NE, Dst, PromotedReg, Zero.getReg(0));
  return Dst;
}

bool PISALegalizePredicatesImpl::run() {
  // -- Phase 1: classify every reachable register.
  // Seed with every cmp def, then transitively classify any user that reads
  // those defs, etc. We rely on the memoized isPromotable() to do the work
  // recursively when we encounter a use whose own operands haven't been
  // classified yet.
  SmallVector<MachineInstr *, 32> Cmps;
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      unsigned Opc = MI.getOpcode();
      if (Opc != TargetOpcode::G_ICMP && Opc != TargetOpcode::G_FCMP)
        continue;
      // isPromotable below filters out cmps with non-i1 result; no need to
      // pre-filter here.
      Cmps.push_back(&MI);
    }
  }

  if (Cmps.empty())
    return false;

  // BFS forward from each cmp to discover the promotable closure. The
  // memoized isPromotable() returns by classifying defs; here we also need
  // to populate the Cache for downstream uses that don't get directly
  // queried. We do this by walking users.
  SmallVector<Register, 64> Worklist;
  for (MachineInstr *MI : Cmps) {
    Register R = MI->getOperand(0).getReg();
    if (isPromotable(R))
      Worklist.push_back(R);
  }
  while (!Worklist.empty()) {
    Register R = Worklist.pop_back_val();
    for (MachineOperand &U : MRI.use_nodbg_operands(R)) {
      MachineInstr *UI = U.getParent();
      // Sinks (G_BRCOND/G_STORE/G_RETURN) have no def -- Phase 2 deals with
      // them. operand(0).isReg() is guaranteed by MIR semantics whenever
      // getNumDefs >= 1.
      if (UI->getNumDefs() < 1)
        continue;
      Register UR = UI->getOperand(0).getReg();
      auto It = Cache.find(UR);
      if (It != Cache.end())
        continue; // already classified
      if (isPromotable(UR))
        Worklist.push_back(UR);
    }
  }

  // -- Phase 2: invalidate any promotable reg whose uses we can't handle.
  invalidateUnhandled();

  // -- Phase 2b: drop chains whose original integer magnitude is ambiguous
  // (a bitwise op mixing sext- and zext-rooted operands). They cannot be
  // restored exactly at an integer sink. Re-run use validation after each
  // demotion round so the cascade reaches consumers of the dropped regs.
  {
    bool Changed = true;
    while (Changed) {
      Changed = false;
      DomMemo.clear();
      for (auto &E : Cache) {
        if (E.second != Yes)
          continue;
        Domain D;
        if (!tryGetDomain(E.first, D)) {
          E.second = No;
          Changed = true;
        }
      }
      if (Changed)
        invalidateUnhandled();
    }
  }

  // Collect the final promote set and decide whether the rewrite is
  // profitable. A single G_XOR-with-true (NOT of a cmp) feeding a branch is
  // already handled by the existing opt_brcond_by_inverting_cond combiner --
  // promoting it would only block that rule. We require *either*
  //   - at least one G_EXTRACT_VECTOR_ELT (a vectorized-cmp reduction), or
  //   - two or more bitwise ops in the promote set (a real reduction chain
  //     that the legalizer would otherwise widen piecewise).
  //
  // We additionally require the chain to genuinely *escape* the boolean
  // domain at one of its *sinks* (see the HasEscapingSink scan below for the
  // precise test). A pure boolean reduction whose every sink is an i1 consumer
  // (G_BRCOND / G_SELECT condition) is deliberately left to the legalizer,
  // which widens-and-fuses such chains optimally. Promoting them here instead
  // restores the result with a `sext` + `icmp ne 0` round trip that is a no-op
  // on a boolean (sext then "!= 0" is the identity). Some downstream consumers
  // fold that pair away, but others do not -- where it survives, the round trip
  // defeats `bfn`-with-flag fusion and emits extra scalar ops across many
  // shaders. Keying the gate on the sinks (rather than on whether some internal
  // node happens to be integer-typed) keeps the unified strategy where it helps
  // -- vector reductions, genuine integer escapes, and the round-trip-free
  // eq/ne operand-redirect fast path -- without regressing pure boolean chains.
  SmallVector<MachineInstr *, 32> ToRewriteDefs;
  bool HasVectorExtract = false;
  unsigned NumBitwise = 0;
  for (auto &E : Cache) {
    if (E.second != Yes)
      continue;
    MachineInstr *Def = MRI.getVRegDef(E.first);
    assert(Def);
    ToRewriteDefs.push_back(Def);
    switch (Def->getOpcode()) {
    case TargetOpcode::G_XOR: {
      // A NOT (xor with a boolean constant) is not a real reduction: promoting
      // it just materializes an explicit `not` that the comparison-inverting
      // combiner would otherwise fold into the cmp predicate (e.g. slt -> sge).
      // Only count xors of two non-constant predicate values toward
      // profitability. (Once promotion is decided by genuine reduction ops, a
      // NOT inside the chain is still rewritten and is absorbed by `bfn`.)
      Register X = Def->getOperand(1).getReg();
      Register Y = Def->getOperand(2).getReg();
      if (MRI.getVRegDef(X)->getOpcode() != TargetOpcode::G_CONSTANT &&
          MRI.getVRegDef(Y)->getOpcode() != TargetOpcode::G_CONSTANT)
        ++NumBitwise;
      break;
    }
    case TargetOpcode::G_AND:
    case TargetOpcode::G_OR:
      ++NumBitwise;
      break;
    case TargetOpcode::G_EXTRACT_VECTOR_ELT:
      HasVectorExtract = true;
      break;
    default:
      break;
    }
  }

  // Decide whether the chain genuinely escapes the boolean domain by examining
  // its *sinks* (uses outside the promote set, plus the kept-alive cmps that
  // stay live). A sink "escapes" iff promoting it does not introduce the
  // sext + `icmp ne 0` round trip:
  //   - integer sink: a non-bool consumer reading a promoted value at width
  //     > 1 (restored via trunc/sext/and, i.e. consumed as a number), or
  //   - clean eq/ne sink: a G_ICMP eq/ne both of whose operands are promoted,
  //     which is redirected to the s32 forms with no restore at all.
  // Any other out-of-set consumer of a bool-typed promoted value is a
  // *predicate sink* (G_BRCOND, G_SELECT condition, an `icmp ==/!= 0` flag
  // test, an i1 store, ...) that must be fed an s1 reconstructed via the
  // sext + `icmp ne/eq 0` round trip.
  //
  // Vector extracts are tracked separately above. A chain whose every sink is
  // a predicate sink has only round-trip restorations and is left unpromoted
  // (the i1-sink case). A chain that has *both* an escaping sink and a
  // predicate sink is also left unpromoted on the scalar path: promoting it
  // would still emit the round-trip restore at the predicate sink, and that
  // restore defeats the backend's `bfn`-with-flag fusion (a single
  // `bfn3.(...)::eq ...,0x1` becomes `bfn2 ... + cmp ::eq 0`, i.e. +1 op per
  // flag) -- the dominant residual regression after the i1-sink gate. The
  // legalizer widens the integer escape at its boundary and the matcher keeps
  // the fused flag, so deferring the whole mixed chain matches the
  // pre-promotion baseline. Only chains whose every sink escapes (pure integer
  // / clean eq-ne reductions, with no predicate consumer) are promoted on the
  // scalar path.
  bool HasEscapingSink = false;
  bool HasPredicateSink = false;
  for (auto &E : Cache) {
    if (E.second != Yes)
      continue;
    Register R = E.first;
    LLT RTy = MRI.getType(R);
    if (RTy.isVector())
      continue;
    for (MachineOperand &U : MRI.use_nodbg_operands(R)) {
      MachineInstr *UI = U.getParent();
      // In-set users that will be erased are rewritten in lock-step; not sinks.
      bool InSetEraseable = UI->getNumDefs() >= 1 &&
                            Cache.lookup(UI->getOperand(0).getReg()) == Yes &&
                            !isKeptAliveOpcode(UI->getOpcode());
      if (InSetEraseable)
        continue;
      // Clean eq/ne sink: redirected to the s32 promoted forms, no round trip.
      if (UI->getOpcode() == TargetOpcode::G_ICMP) {
        auto Pred = (CmpInst::Predicate)UI->getOperand(1).getPredicate();
        auto Qualifies = [&](Register X) { return Cache.lookup(X) == Yes; };
        if ((Pred == CmpInst::ICMP_EQ || Pred == CmpInst::ICMP_NE) &&
            Qualifies(UI->getOperand(2).getReg()) &&
            Qualifies(UI->getOperand(3).getReg())) {
          HasEscapingSink = true;
          continue;
        }
      }
      // Integer sink: the value is consumed at width > 1, restored as a number
      // rather than via the i1 sext + (!=0) round trip. isBoolType keys off
      // getScalarSizeInBits, which (unlike getSizeInBits) does not assert on a
      // typeless register-class operand such as an inline-asm sink.
      if (!isBoolType(RTy)) {
        HasEscapingSink = true;
        continue;
      }
      // Bool-typed value consumed out of set by anything else: a predicate
      // sink that needs the fusion-defeating round-trip restore.
      HasPredicateSink = true;
    }
  }

  if (!HasVectorExtract &&
      (NumBitwise < 2 || !HasEscapingSink || HasPredicateSink))
    return false;

  LLVM_DEBUG(dbgs() << "PISALegalizePredicates: processing " << MF.getName()
                    << " (" << ToRewriteDefs.size() << " defs to rewrite)\n");

  // -- Phase 3: materialize s32 forms for every promoted def, then rewrite
  // each use.
  for (MachineInstr *Def : ToRewriteDefs)
    (void)getOrBuildPromoted(Def->getOperand(0).getReg());

  // The set of s32 promoted-form registers. Used to recognise an eq/ne operand
  // that an earlier restore iteration already redirected to its s32 form (such
  // a register is a *value* in Promoted, not a key, so Promoted.count() alone
  // would miss it).
  DenseSet<Register> PromotedVals;
  for (auto &P : Promoted)
    PromotedVals.insert(P.second);

  // Walk each promoted register's uses; rewrite out-of-set sinks via a
  // restored s1 (or a trunc to their iN width). In-set users will be
  // erased shortly, so their use of the original need not be touched.
  //
  // Only "eraseable" originals require this -- cmps/constants/copies remain
  // alive in MIR and feed their sinks unchanged.
  for (auto &E : Cache) {
    if (E.second != Yes)
      continue;
    Register OrigReg = E.first;
    Register NewReg = Promoted.lookup(OrigReg);
    // The G_ICMP peephole can lazily classify additional constant operands
    // as promotable when probing operand orientation. Such entries land in
    // Cache as Yes but never go through getOrBuildPromoted (the peephole
    // only needed the bool-ness flag). Skip them here.
    if (!NewReg)
      continue;

    MachineInstr *OrigDef = MRI.getVRegDef(OrigReg);
    assert(OrigDef);

    // Skip kinds whose originals stay alive -- they keep feeding their sinks
    // directly, so we don't need to insert restoration.
    if (isKeptAliveOpcode(OrigDef->getOpcode()))
      continue;

    LLT OrigTy = MRI.getType(OrigReg);

    // Vector sinks not supported in v1 (we bailed in isHandledUse).
    if (OrigTy.isVector())
      continue;

    // Snapshot uses up front -- the iterator is invalidated by rewrites.
    SmallVector<MachineOperand *, 8> Uses;
    for (MachineOperand &U : MRI.use_nodbg_operands(OrigReg))
      Uses.push_back(&U);

    for (MachineOperand *U : Uses) {
      MachineInstr *UI = U->getParent();
      // An in-set user whose own opcode is *also* eraseable will be removed
      // shortly; its new promoted form already reads the right value, so
      // we can leave its original operand alone.
      // An in-set user whose opcode is kept-alive (cmp/const/copy) keeps
      // its original instruction in MIR and still references this orig.
      // We must redirect that operand to a restored value so the orig
      // becomes truly dead and the legalizer doesn't widen it.
      bool InSetEraseable = UI->getNumDefs() >= 1 &&
                            Cache.lookup(UI->getOperand(0).getReg()) == Yes &&
                            !isKeptAliveOpcode(UI->getOpcode());
      if (InSetEraseable)
        continue;

      // Special case: a kept-alive G_ICMP eq/ne both of whose compared operands
      // are promoted. eq/ne is invariant under the {0,-1} vs {0,1}
      // representation (both promoted forms are the canonical {0,-1}), so let
      // the comparison read the s32 promoted forms directly. This avoids
      // restoring each operand back to i1 -- which would cost a ucmp.ne plus a
      // select per operand -- and keeps the cmp's i1 result feeding its sink.
      if (UI->getOpcode() == TargetOpcode::G_ICMP) {
        auto Pred = (CmpInst::Predicate)UI->getOperand(1).getPredicate();
        // An operand qualifies if it is a promoted chain (a key of Promoted) or
        // already its s32 promoted form (redirected in an earlier iteration).
        auto Qualifies = [&](Register X) {
          return Promoted.count(X) || PromotedVals.count(X);
        };
        if ((Pred == CmpInst::ICMP_EQ || Pred == CmpInst::ICMP_NE) &&
            Qualifies(UI->getOperand(2).getReg()) &&
            Qualifies(UI->getOperand(3).getReg())) {
          U->setReg(NewReg);
          ++NumRestorations;
          continue;
        }
      }

      Register Restored;
      if (OrigTy.getSizeInBits() == 1) {
        Restored = restoreS1(NewReg, *UI);
      } else {
        // Restore the *original* integer value at this sink. The promoted form
        // is always SEXT-domain ({0, -1}); the original value may be
        // ZEXT-domain ({0, 1}) if its chain was rooted in zero-extends.
        // Reconstruct exactly from the tracked magnitude domain:
        //   Zext: boolean -> G_ZEXT to width  ({0, 1}),
        //   Sext: trunc/sext of the {0, -1} form (sign bits already correct).
        Domain Dom = Domain::Sext;
        bool OK = tryGetDomain(OrigReg, Dom);
        assert(OK && "promoted reg must have an unambiguous magnitude domain");
        (void)OK;
        if (Dom == Domain::Zext) {
          // ZEXT domain ({0, 1}): mask the low bit of the {0, -1} form and
          // adapt to width. This is cheaper than materializing a predicate and
          // selecting 1/0 -- on an 8-bit sink the predicate+select path expands
          // to ucmp + sel + trunc, whereas `and , 1` keeps the value in a GP
          // register and matches the legalizer's fused boolean-mask lowering.
          setInsertPointBefore(*UI);
          auto One = B.buildConstant(LLT::integer(32), 1);
          Register Masked = MRI.createGenericVirtualRegister(LLT::integer(32));
          B.buildAnd(Masked, NewReg, One.getReg(0));
          unsigned W = OrigTy.getSizeInBits();
          if (W == 32) {
            Restored = Masked;
          } else {
            Restored = MRI.createGenericVirtualRegister(OrigTy);
            if (W < 32)
              B.buildTrunc(Restored, Masked);
            else
              B.buildZExt(Restored, Masked);
          }
        } else if (OrigTy.getSizeInBits() == 32) {
          Restored = NewReg;
        } else {
          setInsertPointBefore(*UI);
          Restored = MRI.createGenericVirtualRegister(OrigTy);
          if (OrigTy.getSizeInBits() < 32)
            B.buildTrunc(Restored, NewReg);
          else
            B.buildSExt(Restored, NewReg);
        }
      }
      U->setReg(Restored);
      LLVM_DEBUG(dbgs() << "  Restored use in: " << *UI);
      ++NumRestorations;
    }
  }

  // Erase dead originals (skip the cmps -- they're still alive for s1 sinks
  // and may also be DCEd by later passes if they end up unused).
  eraseDeadOriginals();

  return true;
}

void PISALegalizePredicatesImpl::eraseDeadOriginals() {
  // Collect erasable candidates into a worklist. Iterate until no more
  // instructions become dead (removing a def may free its operands).
  SmallVector<Register, 16> Worklist;
  for (auto &E : Cache) {
    if (E.second != Yes)
      continue;
    Register R = E.first;
    MachineInstr *Def = MRI.getVRegDef(R);
    assert(Def && "every classified vreg has a def in SSA MIR");
    unsigned Opc = Def->getOpcode();
    if (Opc == TargetOpcode::G_ICMP || Opc == TargetOpcode::G_FCMP ||
        Opc == TargetOpcode::G_CONSTANT)
      continue;
    Worklist.push_back(R);
  }

  bool Changed = true;
  while (Changed) {
    Changed = false;
    for (unsigned I = 0; I < Worklist.size(); ++I) {
      Register R = Worklist[I];
      MachineInstr *Def = MRI.getVRegDef(R);
      if (!Def)
        continue;
      if (!MRI.use_nodbg_empty(R))
        continue;
      LLVM_DEBUG(dbgs() << "  Erasing dead original: " << *Def);
      ++NumErased;
      Def->eraseFromParent();
      Changed = true;
    }
  }
}

} // end anonymous namespace

// -----------------------------------------------------------------------------
// Pass plumbing.
// -----------------------------------------------------------------------------

void PISALegalizePredicates::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool PISALegalizePredicates::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  // No FailedISel guard: this pass runs in addPreLegalizeMachineIR(), before
  // instruction selection, so that property is never set at this stage.
  if (MF.getTarget().getOptLevel() == CodeGenOptLevel::None)
    return false;
  PISALegalizePredicatesImpl Impl(MF);
  return Impl.run();
}

char PISALegalizePredicates::ID = 0;
INITIALIZE_PASS(PISALegalizePredicates, DEBUG_TYPE, DEBUG_NAME, false, false)

MachineFunctionPass *llvm::createPISALegalizePredicatesPass() {
  return new PISALegalizePredicates();
}
