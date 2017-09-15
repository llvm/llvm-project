//===-- X86OptimizeLEAs.cpp - optimize usage of LEA instructions ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the pass that performs some optimizations with LEA
// instructions in order to improve performance and code size.
// Currently, it does two things:
// 1) If there are two LEA instructions calculating addresses which only differ
//    by displacement inside a basic block, one of them is removed.
// 2) Address calculations in load and store instructions are replaced by
//    existing LEA def registers where possible.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"

using namespace llvm;

#define DEBUG_TYPE "x86-optimize-LEAs"

static cl::opt<bool>
    DisableX86LEAOpt("disable-x86-lea-opt", cl::Hidden,
                     cl::desc("X86: Disable LEA optimizations."),
                     cl::init(false));

STATISTIC(NumSubstLEAs, "Number of LEA instruction substitutions");
STATISTIC(NumFactoredLEAs, "Number of LEAs factorized");
STATISTIC(NumRedundantLEAs, "Number of redundant LEA instructions removed");

/// \brief Returns true if two machine operands are identical and they are not
/// physical registers.
static inline bool isIdenticalOp(const MachineOperand &MO1,
                                 const MachineOperand &MO2);

/// \brief Returns true if two machine instructions have identical operands.
static bool isIdenticalMI(MachineRegisterInfo *MRI, const MachineOperand &MO1,
                          const MachineOperand &MO2);

/// \brief Returns true if two address displacement operands are of the same
/// type and use the same symbol/index/address regardless of the offset.
static bool isSimilarDispOp(const MachineOperand &MO1,
                            const MachineOperand &MO2);

/// \brief Returns true if the instruction is LEA.
static inline bool isLEA(const MachineInstr &MI);

/// \brief Returns true if Definition of Operand is a copylike instruction.
static bool isDefCopyLike(MachineRegisterInfo *MRI, const MachineOperand &Opr);

namespace {
/// A key based on instruction's memory operands.
class MemOpKey {
public:
  MemOpKey(const MachineOperand *Base, const MachineOperand *Scale,
           const MachineOperand *Index, const MachineOperand *Segment,
           const MachineOperand *Disp, bool DispCheck = false)
      : Disp(Disp), DeepCheck(DispCheck) {
    Operands[0] = Base;
    Operands[1] = Scale;
    Operands[2] = Index;
    Operands[3] = Segment;
  }

  /// Checks operands of MemOpKey are identical, if Base or Index
  /// operand definitions are of kind SUBREG_TO_REG then compare
  /// operands of defining MI.
  bool performDeepCheck(const MemOpKey &Other) const {
    MachineInstr *MI = const_cast<MachineInstr *>(Operands[0]->getParent());
    MachineRegisterInfo *MRI = MI->getRegInfo();

    for (int i = 0; i < 4; i++) {
      bool copyLike = isDefCopyLike(MRI, *Operands[i]);
      if (copyLike && !isIdenticalMI(MRI, *Operands[i], *Other.Operands[i]))
        return false;
      else if (!copyLike && !isIdenticalOp(*Operands[i], *Other.Operands[i]))
        return false;
    }
    return isIdenticalOp(*Disp, *Other.Disp);
  }

  bool operator==(const MemOpKey &Other) const {
    if (DeepCheck)
      return performDeepCheck(Other);

    // Addresses' bases, scales, indices and segments must be identical.
    for (int i = 0; i < 4; ++i)
      if (!isIdenticalOp(*Operands[i], *Other.Operands[i]))
        return false;

    // Addresses' displacements don't have to be exactly the same. It only
    // matters that they use the same symbol/index/address. Immediates' or
    // offsets' differences will be taken care of during instruction
    // substitution.
    return isSimilarDispOp(*Disp, *Other.Disp);
  }

  // Address' base, scale, index and segment operands.
  const MachineOperand *Operands[4];

  // Address' displacement operand.
  const MachineOperand *Disp;

  // If true checks Address' base, index, segment and
  // displacement are identical, in additions if base/index
  // are defined by copylike instruction then futher
  // compare the operands of the defining instruction.
  bool DeepCheck;
};
} // end anonymous namespace

/// Provide DenseMapInfo for MemOpKey.
namespace llvm {
template <> struct DenseMapInfo<MemOpKey> {
  typedef DenseMapInfo<const MachineOperand *> PtrInfo;

  static inline MemOpKey getEmptyKey() {
    return MemOpKey(PtrInfo::getEmptyKey(), PtrInfo::getEmptyKey(),
                    PtrInfo::getEmptyKey(), PtrInfo::getEmptyKey(),
                    PtrInfo::getEmptyKey());
  }

  static inline MemOpKey getTombstoneKey() {
    return MemOpKey(PtrInfo::getTombstoneKey(), PtrInfo::getTombstoneKey(),
                    PtrInfo::getTombstoneKey(), PtrInfo::getTombstoneKey(),
                    PtrInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const MemOpKey &Val) {
    // Checking any field of MemOpKey is enough to determine if the key is
    // empty or tombstone.
    hash_code Hash(0);
    assert(Val.Disp != PtrInfo::getEmptyKey() && "Cannot hash the empty key");
    assert(Val.Disp != PtrInfo::getTombstoneKey() &&
           "Cannot hash the tombstone key");

    auto getMIHash = [](MachineInstr *MI) -> hash_code {
      hash_code h(0);
      for (unsigned i = 1, e = MI->getNumOperands(); i < e; i++)
        h = hash_combine(h, MI->getOperand(i));
      return h;
    };

    const MachineOperand &Base = *Val.Operands[0];
    const MachineOperand &Index = *Val.Operands[2];
    MachineInstr *MI = const_cast<MachineInstr *>(Base.getParent());
    MachineRegisterInfo *MRI = MI->getRegInfo();

    if (isDefCopyLike(MRI, Base))
      Hash = getMIHash(MRI->getVRegDef(Base.getReg()));
    else
      Hash = hash_combine(Hash, Base);

    if (isDefCopyLike(MRI, Index))
      Hash = getMIHash(MRI->getVRegDef(Index.getReg()));
    else
      Hash = hash_combine(Hash, Index);

    Hash = hash_combine(Hash, *Val.Operands[1], *Val.Operands[3]);

    // If the address displacement is an immediate, it should not affect the
    // hash so that memory operands which differ only be immediate displacement
    // would have the same hash. If the address displacement is something else,
    // we should reflect symbol/index/address in the hash.
    switch (Val.Disp->getType()) {
    case MachineOperand::MO_Immediate:
      break;
    case MachineOperand::MO_ConstantPoolIndex:
    case MachineOperand::MO_JumpTableIndex:
      Hash = hash_combine(Hash, Val.Disp->getIndex());
      break;
    case MachineOperand::MO_ExternalSymbol:
      Hash = hash_combine(Hash, Val.Disp->getSymbolName());
      break;
    case MachineOperand::MO_GlobalAddress:
      Hash = hash_combine(Hash, Val.Disp->getGlobal());
      break;
    case MachineOperand::MO_BlockAddress:
      Hash = hash_combine(Hash, Val.Disp->getBlockAddress());
      break;
    case MachineOperand::MO_MCSymbol:
      Hash = hash_combine(Hash, Val.Disp->getMCSymbol());
      break;
    case MachineOperand::MO_MachineBasicBlock:
      Hash = hash_combine(Hash, Val.Disp->getMBB());
      break;
    default:
      llvm_unreachable("Invalid address displacement operand");
    }

    return (unsigned)Hash;
  }

  static bool isEqual(const MemOpKey &LHS, const MemOpKey &RHS) {
    // Checking any field of MemOpKey is enough to determine if the key is
    // empty or tombstone.
    if (RHS.Disp == PtrInfo::getEmptyKey())
      return LHS.Disp == PtrInfo::getEmptyKey();
    if (RHS.Disp == PtrInfo::getTombstoneKey())
      return LHS.Disp == PtrInfo::getTombstoneKey();
    return LHS == RHS;
  }
};
}

/// \brief Returns a hash table key based on memory operands of \p MI. The
/// number of the first memory operand of \p MI is specified through \p N.
static inline MemOpKey getMemOpKey(const MachineInstr &MI, unsigned N) {
  assert((isLEA(MI) || MI.mayLoadOrStore()) &&
         "The instruction must be a LEA, a load or a store");
  return MemOpKey(&MI.getOperand(N + X86::AddrBaseReg),
                  &MI.getOperand(N + X86::AddrScaleAmt),
                  &MI.getOperand(N + X86::AddrIndexReg),
                  &MI.getOperand(N + X86::AddrSegmentReg),
                  &MI.getOperand(N + X86::AddrDisp));
}

static inline MemOpKey getMemOpCSEKey(const MachineInstr &MI, unsigned N) {
  static MachineOperand DummyScale = MachineOperand::CreateImm(1);
  assert((isLEA(MI) || MI.mayLoadOrStore()) &&
         "The instruction must be a LEA, a load or a store");
  return MemOpKey(&MI.getOperand(N + X86::AddrBaseReg), &DummyScale,
                  &MI.getOperand(N + X86::AddrIndexReg),
                  &MI.getOperand(N + X86::AddrSegmentReg),
                  &MI.getOperand(N + X86::AddrDisp), true);
}

static inline bool isIdenticalOp(const MachineOperand &MO1,
                                 const MachineOperand &MO2) {
  return MO1.isIdenticalTo(MO2) &&
         (!MO1.isReg() ||
          !TargetRegisterInfo::isPhysicalRegister(MO1.getReg()));
}

static bool isIdenticalMI(MachineRegisterInfo *MRI, const MachineOperand &MO1,
                          const MachineOperand &MO2) {
  MachineInstr *MI1 = nullptr;
  MachineInstr *MI2 = nullptr; 
  if (!MO1.isReg() || !MO2.isReg())
    return false;

  MI1 = MRI->getVRegDef(MO1.getReg());
  MI2 = MRI->getVRegDef(MO2.getReg());
  if (!MI1 || !MI2)
    return false;
  if (MI1->getOpcode() != MI2->getOpcode())
    return false;
  if (MI1->getNumOperands() != MI2->getNumOperands())
    return false;
  for (unsigned i = 1, e = MI1->getNumOperands(); i < e; ++i)
    if (!isIdenticalOp(MI1->getOperand(i), MI2->getOperand(i)))
      return false;
  return true;
}

#ifndef NDEBUG
static bool isValidDispOp(const MachineOperand &MO) {
  return MO.isImm() || MO.isCPI() || MO.isJTI() || MO.isSymbol() ||
         MO.isGlobal() || MO.isBlockAddress() || MO.isMCSymbol() || MO.isMBB();
}
#endif

static bool isSimilarDispOp(const MachineOperand &MO1,
                            const MachineOperand &MO2) {
  assert(isValidDispOp(MO1) && isValidDispOp(MO2) &&
         "Address displacement operand is not valid");
  return (MO1.isImm() && MO2.isImm()) ||
         (MO1.isCPI() && MO2.isCPI() && MO1.getIndex() == MO2.getIndex()) ||
         (MO1.isJTI() && MO2.isJTI() && MO1.getIndex() == MO2.getIndex()) ||
         (MO1.isSymbol() && MO2.isSymbol() &&
          MO1.getSymbolName() == MO2.getSymbolName()) ||
         (MO1.isGlobal() && MO2.isGlobal() &&
          MO1.getGlobal() == MO2.getGlobal()) ||
         (MO1.isBlockAddress() && MO2.isBlockAddress() &&
          MO1.getBlockAddress() == MO2.getBlockAddress()) ||
         (MO1.isMCSymbol() && MO2.isMCSymbol() &&
          MO1.getMCSymbol() == MO2.getMCSymbol()) ||
         (MO1.isMBB() && MO2.isMBB() && MO1.getMBB() == MO2.getMBB());
}

static inline bool isLEA(const MachineInstr &MI) {
  unsigned Opcode = MI.getOpcode();
  return Opcode == X86::LEA16r || Opcode == X86::LEA32r ||
         Opcode == X86::LEA64r || Opcode == X86::LEA64_32r;
}

static bool isDefCopyLike(MachineRegisterInfo *MRI, const MachineOperand &Opr) {
  if (!Opr.isReg() || TargetRegisterInfo::isPhysicalRegister(Opr.getReg()))
    return false;
  MachineInstr *MI = MRI->getVRegDef(Opr.getReg());
  return MI && MI->isCopyLike();
}

namespace {

/// This class captures the functions and attributes
/// needed to factorize LEA within and across basic
/// blocks.LEA instruction with same BASE,OFFSET and
/// INDEX are the candidates for factorization.
class FactorizeLEAOpt {
public:
  using LEAListT = std::list<MachineInstr *>;
  using LEAMapT = DenseMap<MemOpKey, LEAListT>;
  using ValueT = DenseMap<MemOpKey, unsigned>;
  using ScopeEntryT = std::pair<MachineBasicBlock *, ValueT>;
  using ScopeStackT = std::vector<ScopeEntryT>;

  FactorizeLEAOpt() = default;
  FactorizeLEAOpt(const FactorizeLEAOpt &) = delete;
  FactorizeLEAOpt &operator=(const FactorizeLEAOpt &) = delete;

  void performCleanup() {
    for (auto LEA : removedLEAs)
      LEA->eraseFromParent();
    LEAs.clear();
    Stack.clear();
    removedLEAs.clear();
  }

  LEAMapT &getLEAMap() { return LEAs; }
  ScopeEntryT *getTopScope() { return &Stack.back(); }

  void addForLazyRemoval(MachineInstr *Instr) { removedLEAs.insert(Instr); }

  bool checkIfScheduledForRemoval(MachineInstr *Instr) {
    return removedLEAs.find(Instr) != removedLEAs.end();
  }

  /// Push the ScopeEntry for the BasicBlock over Stack.
  /// Also traverses over list of instruction and update
  /// LEAs Map and ScopeEntry for each LEA instruction
  /// found using insertLEA().
  void pushScope(MachineBasicBlock *MBB);

  /// Stores the size of MachineInstr list corrosponding
  /// to key K from LEAs MAP into the ScopeEntry of
  /// the basic block, then insert the LEA at the beginning
  /// of the list.
  void insertLEA(MachineInstr *MI);

  /// Pops out ScopeEntry of top most BasicBlock from the stack
  /// and remove the LEA instructions contained in the scope
  /// from the LEAs Map.
  void popScope();

  /// If LEA contains Physical Registers then its not a candidate
  /// for factorizations since physical registers may violate SSA
  /// semantics of MI.
  bool containsPhyReg(MachineInstr *MI, unsigned RecLevel);

private:
  ScopeStackT Stack;
  LEAMapT LEAs;
  std::set<MachineInstr *> removedLEAs;
};

void FactorizeLEAOpt::pushScope(MachineBasicBlock *MBB) {
  ValueT EmptyMap;
  ScopeEntryT SE = std::make_pair(MBB, EmptyMap);
  Stack.push_back(SE);
  for (auto &MI : *MBB) {
    if (isLEA(MI))
      insertLEA(&MI);
  }
}

void FactorizeLEAOpt::popScope() {
  ScopeEntryT &SE = Stack.back();
  for (auto MapEntry : SE.second) {
    LEAMapT::iterator Itr = LEAs.find(MapEntry.first);
    assert((Itr != LEAs.end()) &&
           "LEAs map must have a node corresponding to ScopeEntry's Key.");

    while (((*Itr).second.size() > MapEntry.second))
      (*Itr).second.pop_front();
    // If list goes empty remove entry from LEAs Map.
    if ((*Itr).second.empty())
      LEAs.erase(Itr);
  }
  Stack.pop_back();
}

bool FactorizeLEAOpt::containsPhyReg(MachineInstr *MI, unsigned RecLevel) {
  if (!MI || !RecLevel)
    return false;

  MachineRegisterInfo *MRI = MI->getRegInfo();
  for (auto Operand : MI->operands()) {
    if (!Operand.isReg())
      continue;
    if (TargetRegisterInfo::isPhysicalRegister(Operand.getReg()))
      return true;
    MachineInstr *OperDefMI = MRI->getVRegDef(Operand.getReg());
    if (OperDefMI && (MI != OperDefMI) && OperDefMI->isCopyLike() &&
        containsPhyReg(OperDefMI, RecLevel - 1))
      return true;
  }
  return false;
}

void FactorizeLEAOpt::insertLEA(MachineInstr *MI) {
  unsigned lsize;
  if (containsPhyReg(MI, 2))
    return;

  MemOpKey Key = getMemOpCSEKey(*MI, 1);
  ScopeEntryT *TopScope = getTopScope();

  LEAMapT::iterator Itr = LEAs.find(Key);
  if (Itr == LEAs.end()) {
    lsize = 0;
    LEAs[Key].push_front(MI);
  } else {
    lsize = (*Itr).second.size();
    (*Itr).second.push_front(MI);
  }
  if (TopScope->second.find(Key) == TopScope->second.end())
    TopScope->second[Key] = lsize;
}

class OptimizeLEAPass : public MachineFunctionPass {
public:
  OptimizeLEAPass() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "X86 LEA Optimize"; }

  /// \brief Loop over all of the basic blocks, replacing address
  /// calculations in load and store instructions, if it's already
  /// been calculated by LEA. Also, remove redundant LEAs.
  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
    AU.addRequired<MachineDominatorTree>();
  }

private:
  typedef DenseMap<MemOpKey, SmallVector<MachineInstr *, 16>> MemOpMap;

  /// \brief Returns a distance between two instructions inside one basic block.
  /// Negative result means, that instructions occur in reverse order.
  int calcInstrDist(const MachineInstr &First, const MachineInstr &Last);

  /// \brief Choose the best \p LEA instruction from the \p List to replace
  /// address calculation in \p MI instruction. Return the address displacement
  /// and the distance between \p MI and the chosen \p BestLEA in
  /// \p AddrDispShift and \p Dist.
  bool chooseBestLEA(const SmallVectorImpl<MachineInstr *> &List,
                     const MachineInstr &MI, MachineInstr *&BestLEA,
                     int64_t &AddrDispShift, int &Dist);

  /// \brief Returns the difference between addresses' displacements of \p MI1
  /// and \p MI2. The numbers of the first memory operands for the instructions
  /// are specified through \p N1 and \p N2.
  int64_t getAddrDispShift(const MachineInstr &MI1, unsigned N1,
                           const MachineInstr &MI2, unsigned N2) const;

  /// \brief Returns true if the \p Last LEA instruction can be replaced by the
  /// \p First. The difference between displacements of the addresses calculated
  /// by these LEAs is returned in \p AddrDispShift. It'll be used for proper
  /// replacement of the \p Last LEA's uses with the \p First's def register.
  bool isReplaceable(const MachineInstr &First, const MachineInstr &Last,
                     int64_t &AddrDispShift) const;

  /// \brief Find all LEA instructions in the basic block. Also, assign position
  /// numbers to all instructions in the basic block to speed up calculation of
  /// distance between them.
  void findLEAs(const MachineBasicBlock &MBB, MemOpMap &LEAs);

  /// \brief Removes redundant address calculations.
  bool removeRedundantAddrCalc(MemOpMap &LEAs);

  /// Replace debug value MI with a new debug value instruction using register
  /// VReg with an appropriate offset and DIExpression to incorporate the
  /// address displacement AddrDispShift. Return new debug value instruction.
  MachineInstr *replaceDebugValue(MachineInstr &MI, unsigned VReg,
                                  int64_t AddrDispShift);

  /// \brief Removes LEAs which calculate similar addresses.
  bool removeRedundantLEAs(MemOpMap &LEAs);

  /// \brief Visit over basic blocks, collect LEAs in a scoped
  ///  hash map (FactorizeLEAOpt::LEAs) and try to factor them out.
  bool FactorizeLEAsAllBasicBlocks(MachineFunction &MF);

  bool FactorizeLEAsBasicBlock(MachineDomTreeNode *DN);

  /// \brief Factor out LEAs which share Base,Index,Offset and Segment.
  bool processBasicBlock(const MachineBasicBlock &MBB);

  /// \brief Try to replace LEA with a lower strength instruction
  /// to improves latency and throughput.
  bool strengthReduceLEAs(MemOpMap &LEAs, const MachineBasicBlock &MBB);

  DenseMap<const MachineInstr *, unsigned> InstrPos;

  FactorizeLEAOpt FactorOpt;

  MachineDominatorTree *DT;
  MachineRegisterInfo *MRI;
  const X86InstrInfo *TII;
  const X86RegisterInfo *TRI;

  static char ID;
};
char OptimizeLEAPass::ID = 0;
}

FunctionPass *llvm::createX86OptimizeLEAs() { return new OptimizeLEAPass(); }

int OptimizeLEAPass::calcInstrDist(const MachineInstr &First,
                                   const MachineInstr &Last) {
  // Both instructions must be in the same basic block and they must be
  // presented in InstrPos.
  assert(Last.getParent() == First.getParent() &&
         "Instructions are in different basic blocks");
  assert(InstrPos.find(&First) != InstrPos.end() &&
         InstrPos.find(&Last) != InstrPos.end() &&
         "Instructions' positions are undefined");

  return InstrPos[&Last] - InstrPos[&First];
}

// Find the best LEA instruction in the List to replace address recalculation in
// MI. Such LEA must meet these requirements:
// 1) The address calculated by the LEA differs only by the displacement from
//    the address used in MI.
// 2) The register class of the definition of the LEA is compatible with the
//    register class of the address base register of MI.
// 3) Displacement of the new memory operand should fit in 1 byte if possible.
// 4) The LEA should be as close to MI as possible, and prior to it if
//    possible.
bool OptimizeLEAPass::chooseBestLEA(const SmallVectorImpl<MachineInstr *> &List,
                                    const MachineInstr &MI,
                                    MachineInstr *&BestLEA,
                                    int64_t &AddrDispShift, int &Dist) {
  const MachineFunction *MF = MI.getParent()->getParent();
  const MCInstrDesc &Desc = MI.getDesc();
  int MemOpNo = X86II::getMemoryOperandNo(Desc.TSFlags) +
                X86II::getOperandBias(Desc);

  BestLEA = nullptr;

  // Loop over all LEA instructions.
  for (auto DefMI : List) {
    // Get new address displacement.
    int64_t AddrDispShiftTemp = getAddrDispShift(MI, MemOpNo, *DefMI, 1);

    // Make sure address displacement fits 4 bytes.
    if (!isInt<32>(AddrDispShiftTemp))
      continue;

    // Check that LEA def register can be used as MI address base. Some
    // instructions can use a limited set of registers as address base, for
    // example MOV8mr_NOREX. We could constrain the register class of the LEA
    // def to suit MI, however since this case is very rare and hard to
    // reproduce in a test it's just more reliable to skip the LEA.
    if (TII->getRegClass(Desc, MemOpNo + X86::AddrBaseReg, TRI, *MF) !=
        MRI->getRegClass(DefMI->getOperand(0).getReg()))
      continue;

    // Choose the closest LEA instruction from the list, prior to MI if
    // possible. Note that we took into account resulting address displacement
    // as well. Also note that the list is sorted by the order in which the LEAs
    // occur, so the break condition is pretty simple.
    int DistTemp = calcInstrDist(*DefMI, MI);
    assert(DistTemp != 0 &&
           "The distance between two different instructions cannot be zero");
    if (DistTemp > 0 || BestLEA == nullptr) {
      // Do not update return LEA, if the current one provides a displacement
      // which fits in 1 byte, while the new candidate does not.
      if (BestLEA != nullptr && !isInt<8>(AddrDispShiftTemp) &&
          isInt<8>(AddrDispShift))
        continue;

      BestLEA = DefMI;
      AddrDispShift = AddrDispShiftTemp;
      Dist = DistTemp;
    }

    // FIXME: Maybe we should not always stop at the first LEA after MI.
    if (DistTemp < 0)
      break;
  }

  return BestLEA != nullptr;
}

// Get the difference between the addresses' displacements of the two
// instructions \p MI1 and \p MI2. The numbers of the first memory operands are
// passed through \p N1 and \p N2.
int64_t OptimizeLEAPass::getAddrDispShift(const MachineInstr &MI1, unsigned N1,
                                          const MachineInstr &MI2,
                                          unsigned N2) const {
  const MachineOperand &Op1 = MI1.getOperand(N1 + X86::AddrDisp);
  const MachineOperand &Op2 = MI2.getOperand(N2 + X86::AddrDisp);

  assert(isSimilarDispOp(Op1, Op2) &&
         "Address displacement operands are not compatible");

  // After the assert above we can be sure that both operands are of the same
  // valid type and use the same symbol/index/address, thus displacement shift
  // calculation is rather simple.
  if (Op1.isJTI())
    return 0;
  return Op1.isImm() ? Op1.getImm() - Op2.getImm()
                     : Op1.getOffset() - Op2.getOffset();
}

// Check that the Last LEA can be replaced by the First LEA. To be so,
// these requirements must be met:
// 1) Addresses calculated by LEAs differ only by displacement.
// 2) Def registers of LEAs belong to the same class.
// 3) All uses of the Last LEA def register are replaceable, thus the
//    register is used only as address base.
bool OptimizeLEAPass::isReplaceable(const MachineInstr &First,
                                    const MachineInstr &Last,
                                    int64_t &AddrDispShift) const {
  assert(isLEA(First) && isLEA(Last) &&
         "The function works only with LEA instructions");

  // Make sure that LEA def registers belong to the same class. There may be
  // instructions (like MOV8mr_NOREX) which allow a limited set of registers to
  // be used as their operands, so we must be sure that replacing one LEA
  // with another won't lead to putting a wrong register in the instruction.
  if (MRI->getRegClass(First.getOperand(0).getReg()) !=
      MRI->getRegClass(Last.getOperand(0).getReg()))
    return false;

  // Get new address displacement.
  AddrDispShift = getAddrDispShift(Last, 1, First, 1);

  // Loop over all uses of the Last LEA to check that its def register is
  // used only as address base for memory accesses. If so, it can be
  // replaced, otherwise - no.
  for (auto &MO : MRI->use_nodbg_operands(Last.getOperand(0).getReg())) {
    MachineInstr &MI = *MO.getParent();

    // Get the number of the first memory operand.
    const MCInstrDesc &Desc = MI.getDesc();
    int MemOpNo = X86II::getMemoryOperandNo(Desc.TSFlags);

    // If the use instruction has no memory operand - the LEA is not
    // replaceable.
    if (MemOpNo < 0)
      return false;

    MemOpNo += X86II::getOperandBias(Desc);

    // If the address base of the use instruction is not the LEA def register -
    // the LEA is not replaceable.
    if (!isIdenticalOp(MI.getOperand(MemOpNo + X86::AddrBaseReg), MO))
      return false;

    // If the LEA def register is used as any other operand of the use
    // instruction - the LEA is not replaceable.
    for (unsigned i = 0; i < MI.getNumOperands(); i++)
      if (i != (unsigned)(MemOpNo + X86::AddrBaseReg) &&
          isIdenticalOp(MI.getOperand(i), MO))
        return false;

    // Check that the new address displacement will fit 4 bytes.
    if (MI.getOperand(MemOpNo + X86::AddrDisp).isImm() &&
        !isInt<32>(MI.getOperand(MemOpNo + X86::AddrDisp).getImm() +
                   AddrDispShift))
      return false;
  }

  return true;
}

void OptimizeLEAPass::findLEAs(const MachineBasicBlock &MBB, MemOpMap &LEAs) {
  unsigned Pos = 0;
  for (auto &MI : MBB) {
    // Assign the position number to the instruction. Note that we are going to
    // move some instructions during the optimization however there will never
    // be a need to move two instructions before any selected instruction. So to
    // avoid multiple positions' updates during moves we just increase position
    // counter by two leaving a free space for instructions which will be moved.
    InstrPos[&MI] = Pos += 2;

    if (isLEA(MI))
      LEAs[getMemOpKey(MI, 1)].push_back(const_cast<MachineInstr *>(&MI));
  }
}

// Try to find load and store instructions which recalculate addresses already
// calculated by some LEA and replace their memory operands with its def
// register.
bool OptimizeLEAPass::removeRedundantAddrCalc(MemOpMap &LEAs) {
  bool Changed = false;

  assert(!LEAs.empty());
  MachineBasicBlock *MBB = (*LEAs.begin()->second.begin())->getParent();

  // Process all instructions in basic block.
  for (auto I = MBB->begin(), E = MBB->end(); I != E;) {
    MachineInstr &MI = *I++;

    // Instruction must be load or store.
    if (!MI.mayLoadOrStore())
      continue;

    // Get the number of the first memory operand.
    const MCInstrDesc &Desc = MI.getDesc();
    int MemOpNo = X86II::getMemoryOperandNo(Desc.TSFlags);

    // If instruction has no memory operand - skip it.
    if (MemOpNo < 0)
      continue;

    MemOpNo += X86II::getOperandBias(Desc);

    // Get the best LEA instruction to replace address calculation.
    MachineInstr *DefMI;
    int64_t AddrDispShift;
    int Dist;
    if (!chooseBestLEA(LEAs[getMemOpKey(MI, MemOpNo)], MI, DefMI, AddrDispShift,
                       Dist))
      continue;

    // If LEA occurs before current instruction, we can freely replace
    // the instruction. If LEA occurs after, we can lift LEA above the
    // instruction and this way to be able to replace it. Since LEA and the
    // instruction have similar memory operands (thus, the same def
    // instructions for these operands), we can always do that, without
    // worries of using registers before their defs.
    if (Dist < 0) {
      DefMI->removeFromParent();
      MBB->insert(MachineBasicBlock::iterator(&MI), DefMI);
      InstrPos[DefMI] = InstrPos[&MI] - 1;

      // Make sure the instructions' position numbers are sane.
      assert(((InstrPos[DefMI] == 1 &&
               MachineBasicBlock::iterator(DefMI) == MBB->begin()) ||
              InstrPos[DefMI] >
                  InstrPos[&*std::prev(MachineBasicBlock::iterator(DefMI))]) &&
             "Instruction positioning is broken");
    }

    // Since we can possibly extend register lifetime, clear kill flags.
    MRI->clearKillFlags(DefMI->getOperand(0).getReg());

    ++NumSubstLEAs;
    DEBUG(dbgs() << "OptimizeLEAs: Candidate to replace: "; MI.dump(););

    // Change instruction operands.
    MI.getOperand(MemOpNo + X86::AddrBaseReg)
        .ChangeToRegister(DefMI->getOperand(0).getReg(), false);
    MI.getOperand(MemOpNo + X86::AddrScaleAmt).ChangeToImmediate(1);
    MI.getOperand(MemOpNo + X86::AddrIndexReg)
        .ChangeToRegister(X86::NoRegister, false);
    MI.getOperand(MemOpNo + X86::AddrDisp).ChangeToImmediate(AddrDispShift);
    MI.getOperand(MemOpNo + X86::AddrSegmentReg)
        .ChangeToRegister(X86::NoRegister, false);

    DEBUG(dbgs() << "OptimizeLEAs: Replaced by: "; MI.dump(););

    Changed = true;
  }

  return Changed;
}

MachineInstr *OptimizeLEAPass::replaceDebugValue(MachineInstr &MI,
                                                 unsigned VReg,
                                                 int64_t AddrDispShift) {
  DIExpression *Expr = const_cast<DIExpression *>(MI.getDebugExpression());

  if (AddrDispShift != 0)
    Expr = DIExpression::prepend(Expr, DIExpression::NoDeref, AddrDispShift,
                                 DIExpression::WithStackValue);

  // Replace DBG_VALUE instruction with modified version.
  MachineBasicBlock *MBB = MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  bool IsIndirect = MI.isIndirectDebugValue();
  const MDNode *Var = MI.getDebugVariable();
  if (IsIndirect)
    assert(MI.getOperand(1).getImm() == 0 && "DBG_VALUE with nonzero offset");
  return BuildMI(*MBB, MBB->erase(&MI), DL, TII->get(TargetOpcode::DBG_VALUE),
                 IsIndirect, VReg, Var, Expr);
}

// Try to find similar LEAs in the list and replace one with another.
bool OptimizeLEAPass::removeRedundantLEAs(MemOpMap &LEAs) {
  bool Changed = false;

  // Loop over all entries in the table.
  for (auto &E : LEAs) {
    auto &List = E.second;

    // Loop over all LEA pairs.
    auto I1 = List.begin();
    while (I1 != List.end()) {
      MachineInstr &First = **I1;
      auto I2 = std::next(I1);
      while (I2 != List.end()) {
        MachineInstr &Last = **I2;
        int64_t AddrDispShift;

        // LEAs should be in occurrence order in the list, so we can freely
        // replace later LEAs with earlier ones.
        assert(calcInstrDist(First, Last) > 0 &&
               "LEAs must be in occurrence order in the list");

        // Check that the Last LEA instruction can be replaced by the First.
        if (!isReplaceable(First, Last, AddrDispShift)) {
          ++I2;
          continue;
        }

        // Loop over all uses of the Last LEA and update their operands. Note
        // that the correctness of this has already been checked in the
        // isReplaceable function.
        unsigned FirstVReg = First.getOperand(0).getReg();
        unsigned LastVReg = Last.getOperand(0).getReg();
        for (auto UI = MRI->use_begin(LastVReg), UE = MRI->use_end();
             UI != UE;) {
          MachineOperand &MO = *UI++;
          MachineInstr &MI = *MO.getParent();

          if (MI.isDebugValue()) {
            // Replace DBG_VALUE instruction with modified version using the
            // register from the replacing LEA and the address displacement
            // between the LEA instructions.
            replaceDebugValue(MI, FirstVReg, AddrDispShift);
            continue;
          }

          // Get the number of the first memory operand.
          const MCInstrDesc &Desc = MI.getDesc();
          int MemOpNo =
              X86II::getMemoryOperandNo(Desc.TSFlags) +
              X86II::getOperandBias(Desc);

          // Update address base.
          MO.setReg(FirstVReg);

          // Update address disp.
          MachineOperand &Op = MI.getOperand(MemOpNo + X86::AddrDisp);
          if (Op.isImm())
            Op.setImm(Op.getImm() + AddrDispShift);
          else if (!Op.isJTI())
            Op.setOffset(Op.getOffset() + AddrDispShift);
        }

        // Since we can possibly extend register lifetime, clear kill flags.
        MRI->clearKillFlags(FirstVReg);

        ++NumRedundantLEAs;
        DEBUG(dbgs() << "OptimizeLEAs: Remove redundant LEA: "; Last.dump(););

        // By this moment, all of the Last LEA's uses must be replaced. So we
        // can freely remove it.
        assert(MRI->use_empty(LastVReg) &&
               "The LEA's def register must have no uses");
        Last.eraseFromParent();

        // Erase removed LEA from the list.
        I2 = List.erase(I2);

        Changed = true;
      }
      ++I1;
    }
  }

  return Changed;
}

static inline int getADDrrFromLEA(int LEAOpcode) {
  switch (LEAOpcode) {
  default:
    llvm_unreachable("Unexpected LEA instruction");
  case X86::LEA16r:
    return X86::ADD16rr;
  case X86::LEA32r:
    return X86::ADD32rr;
  case X86::LEA64_32r:
  case X86::LEA64r:
    return X86::ADD64rr;
  }
}

bool OptimizeLEAPass::strengthReduceLEAs(MemOpMap &LEAs,
                                         const MachineBasicBlock &BB) {
  bool Changed = false;

  // Loop over all entries in the table.
  for (auto &E : LEAs) {
    auto &List = E.second;

    // Loop over all LEA pairs.
    for (auto I1 = List.begin(); I1 != List.end(); I1++) {
      MachineInstrBuilder NewMI;
      MachineInstr &First = **I1;
      MachineOperand &Res = First.getOperand(0);
      MachineOperand &Base = First.getOperand(1);
      MachineOperand &Scale = First.getOperand(2);
      MachineOperand &Index = First.getOperand(3);
      MachineOperand &Offset = First.getOperand(4);

      const MCInstrDesc &ADDrr = TII->get(getADDrrFromLEA(First.getOpcode()));
      const DebugLoc DL = First.getDebugLoc();

      if (!Base.isReg() || !Index.isReg())
        continue;
      if (TargetRegisterInfo::isPhysicalRegister(Res.getReg()) ||
          TargetRegisterInfo::isPhysicalRegister(Base.getReg()) ||
          TargetRegisterInfo::isPhysicalRegister(Index.getReg()))
        continue;

      MachineBasicBlock &MBB = *(const_cast<MachineBasicBlock *>(&BB));
      if (Scale.isImm() && Scale.getImm() == 1) {
        // R = B + I
        if (Offset.isImm() && !Offset.getImm()) {
          NewMI = BuildMI(MBB, &First, DL, ADDrr)
                      .addDef(Res.getReg())
                      .addUse(Base.getReg())
                      .addUse(Index.getReg());
          Changed = NewMI.getInstr() != nullptr;
          First.eraseFromParent();
        }
      }
    }
  }
  return Changed;
}

bool OptimizeLEAPass::processBasicBlock(const MachineBasicBlock &MBB) {
  bool cseDone = false;

  // Legal scale value (1,2,4 & 8) vector.
  int LegalScale[9] = {0, 1, 1, 0, 1, 0, 0, 0, 1};

  auto CompareFn = [](const MachineInstr *Arg1,
                      const MachineInstr *Arg2) -> bool {
    if (Arg1->getOperand(2).getImm() < Arg2->getOperand(2).getImm())
      return false;
    return true;
  };

  // Loop over all entries in the table.
  for (auto &E : FactorOpt.getLEAMap()) {
    auto &List = E.second;
    if (List.size() > 1)
      List.sort(CompareFn);
    
    // Loop over all LEA pairs.
    for (auto Iter1 = List.begin(); Iter1 != List.end(); Iter1++) {
      for (auto Iter2 = std::next(Iter1); Iter2 != List.end(); Iter2++) {
        MachineInstr &LI1 = **Iter1;
        MachineInstr &LI2 = **Iter2;

        if (!DT->dominates(&LI2, &LI1))
          continue;

        int Scale1 = LI1.getOperand(2).getImm();
        int Scale2 = LI2.getOperand(2).getImm();
        assert(LI2.getOperand(0).isReg() && "Result is a VirtualReg");
        DebugLoc DL = LI1.getDebugLoc();

        if (FactorOpt.checkIfScheduledForRemoval(&LI1))
          continue;

        int Factor = Scale1 - Scale2;
        if (Factor > 0 && LegalScale[Factor]) {
          DEBUG(dbgs() << "CSE LEAs: Candidate to replace: "; LI1.dump(););
          MachineInstrBuilder NewMI =
              BuildMI(*(const_cast<MachineBasicBlock *>(&MBB)), &LI1, DL,
                      TII->get(LI1.getOpcode()))
                  .addDef(LI1.getOperand(0).getReg()) // Dst   = Dst of LI1.
                  .addUse(LI2.getOperand(0).getReg()) // Base  = Dst of LI2.
                  .addImm(Factor) // Scale = Diff b/w scales.
                  .addUse(LI1.getOperand(3).getReg()) // Index = Index of LI1.
                  .addImm(0)                          // Disp  = 0
                  .addUse(
                      LI1.getOperand(5).getReg()); // Segment = Segmant of LI1.

          cseDone = NewMI.getInstr() != nullptr;

          /// Lazy removal shall ensure that replaced LEA remains
          /// till we finish processing all the basic block. This shall
          /// provide opportunity for further factorization based on
          /// the replaced LEA which will be legal since it has same
          /// destination as newly formed LEA.
          FactorOpt.addForLazyRemoval(&LI1);

          NumFactoredLEAs++;
          DEBUG(dbgs() << "CSE LEAs: Replaced by: "; NewMI->dump(););
        }
      }
    }
  }
  return cseDone;
}

bool OptimizeLEAPass::FactorizeLEAsBasicBlock(MachineDomTreeNode *DN) {
  bool Changed = false;
  MachineBasicBlock *MBB = DN->getBlock();
  FactorOpt.pushScope(MBB);

  Changed |= processBasicBlock(*MBB);
  for (auto Child : DN->getChildren())
    FactorizeLEAsBasicBlock(Child);

  FactorOpt.popScope();
  return Changed;
}

bool OptimizeLEAPass::FactorizeLEAsAllBasicBlocks(MachineFunction &MF) {
  bool Changed = FactorizeLEAsBasicBlock(DT->getRootNode());
  FactorOpt.performCleanup();
  return Changed;
}

bool OptimizeLEAPass::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;

  if (DisableX86LEAOpt || skipFunction(*MF.getFunction()))
    return false;

  MRI = &MF.getRegInfo();
  TII = MF.getSubtarget<X86Subtarget>().getInstrInfo();
  TRI = MF.getSubtarget<X86Subtarget>().getRegisterInfo();
  DT = &getAnalysis<MachineDominatorTree>();

  // Attempt factorizing LEAs.
  Changed |= FactorizeLEAsAllBasicBlocks(MF);

  // Process all basic blocks.
  for (auto &MBB : MF) {
    MemOpMap LEAs;
    InstrPos.clear();

    // Find all LEA instructions in basic block.
    findLEAs(MBB, LEAs);

    // If current basic block has no LEAs, move on to the next one.
    if (LEAs.empty())
      continue;

    // Remove redundant LEA instructions.
    Changed |= removeRedundantLEAs(LEAs);

    // Strength reduce LEA instructions.
    Changed |= strengthReduceLEAs(LEAs, MBB);

    // Remove redundant address calculations. Do it only for -Os/-Oz since only
    // a code size gain is expected from this part of the pass.
    if (MF.getFunction()->optForSize())
      Changed |= removeRedundantAddrCalc(LEAs);
  }

  return Changed;
}
