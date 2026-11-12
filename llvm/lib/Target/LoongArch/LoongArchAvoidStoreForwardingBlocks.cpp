//===- LoongArchAvoidStoreForwardingBlocks.cpp - Avoid HW Store Forward Block //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass breaks small memcpy-like load/store pairs (LD_D/ST_D, FLD_D/FST_D,
// VLD/VST and XVLD/XVST) into smaller copies when the load is likely to read
// data from in-flight narrower stores. Store-to-load forwarding fails when the
// load is wider than the store, which delays the load by several cycles.
//
// A store blocks a copy if it is executed shortly before the copy's load and
// either:
// - is narrower than the load and writes inside the loaded bytes through the
//   same base register or frame index. The copy is broken around it.
// - belongs to the previous loop iteration, is at least 8 bytes wide and may
//   write the same underlying object, as in LZ77 back-reference copies:
//     do { memcpy(dst, dst - offset, 16); dst += 16; } while (dst < end);
//   The copy is broken into 8-byte pieces.
//
// All partial loads are emitted before all partial stores, so overlapping
// copies (memmove) remain correct.
//
//===----------------------------------------------------------------------===//

#include "LoongArch.h"
#include "LoongArchInstrInfo.h"
#include "LoongArchSubtarget.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include <map>

using namespace llvm;

#define DEBUG_TYPE "loongarch-avoid-sfb"
#define LOONGARCH_AVOID_SFB_NAME "LoongArch Avoid Store Forwarding Blocks"

STATISTIC(NumBrokenCopies, "Number of copies broken into smaller copies");
STATISTIC(NumBrokenScalarCopies,
          "Number of 8-byte scalar copies broken into smaller copies");
STATISTIC(NumLoopCarriedBlocks,
          "Number of copies broken because of loop-carried blocking stores");

static cl::opt<bool> DisableAvoidSFB("loongarch-disable-avoid-sfb", cl::Hidden,
                                     cl::desc("LoongArch: Disable Store "
                                              "Forwarding Blocks fixup."),
                                     cl::init(false));

static cl::opt<unsigned> SFBInspectionLimit(
    "loongarch-sfb-inspection-limit", cl::Hidden, cl::init(20),
    cl::desc("LoongArch: Number of instructions backward to inspect for store "
             "forwarding blocks."));

namespace {

using DisplacementSizeMap = std::map<int64_t, unsigned>;

class LoongArchAvoidSFB : public MachineFunctionPass {
public:
  static char ID;
  LoongArchAvoidSFB() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return LOONGARCH_AVOID_SFB_NAME; }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  MachineRegisterInfo *MRI = nullptr;
  const LoongArchInstrInfo *TII = nullptr;
  AliasAnalysis *AA = nullptr;
  MachineLoopInfo *MLI = nullptr;
  SmallVector<std::pair<MachineInstr *, MachineInstr *>, 2>
      BlockedLoadsStoresPairs;
  SmallVector<MachineInstr *, 2> ForRemoval;

  /// A store that may block a load, and whether it was reached through a
  /// loop back edge (i.e. it belongs to a previous iteration).
  struct PotentialBlocker {
    MachineInstr *MI;
    bool ViaBackEdge;
  };

  /// Returns couples of Load then Store to memory which look like a memcpy.
  void findPotentiallyBlockedCopies(MachineFunction &MF);
  SmallVector<PotentialBlocker, 4>
  findPotentialBlockers(MachineInstr *LoadInst);
  bool mayAliasAcrossIterations(const MachineMemOperand &LdMMO,
                                const MachineMemOperand &StMMO) const;
  /// Break the memcpy's load and store into smaller copies such that each
  /// memory load that was blocked by a smaller store would now be copied
  /// separately.
  void breakBlockedCopies(MachineInstr *LoadInst, MachineInstr *StoreInst,
                          const DisplacementSizeMap &BlockingStoresDispSizeMap);
  /// Break a copy of size Size to smaller copies.
  void buildCopies(int Size, MachineInstr *LoadInst, int64_t LdDispImm,
                   MachineInstr *StoreInst, int64_t StDispImm, int64_t Offset);
  void buildCopy(MachineInstr *LoadInst, unsigned NLoadOpcode, int64_t LoadDisp,
                 MachineInstr *StoreInst, unsigned NStoreOpcode,
                 int64_t StoreDisp, unsigned Size, int64_t Offset);
};

} // end anonymous namespace

char LoongArchAvoidSFB::ID = 0;

INITIALIZE_PASS_BEGIN(LoongArchAvoidSFB, DEBUG_TYPE, LOONGARCH_AVOID_SFB_NAME,
                      false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(LoongArchAvoidSFB, DEBUG_TYPE, LOONGARCH_AVOID_SFB_NAME,
                    false, false)

FunctionPass *llvm::createLoongArchAvoidStoreForwardingBlocksPass() {
  return new LoongArchAvoidSFB();
}

static bool isPotentialBlockedMemCpyLd(unsigned Opcode) {
  return Opcode == LoongArch::VLD || Opcode == LoongArch::XVLD ||
         Opcode == LoongArch::LD_D || Opcode == LoongArch::FLD_D;
}

static bool isPotentialBlockedMemCpyPair(unsigned LdOpcode, unsigned StOpcode) {
  switch (LdOpcode) {
  case LoongArch::VLD:
    return StOpcode == LoongArch::VST;
  case LoongArch::XVLD:
    return StOpcode == LoongArch::XVST;
  case LoongArch::LD_D:
    return StOpcode == LoongArch::ST_D;
  case LoongArch::FLD_D:
    return StOpcode == LoongArch::FST_D;
  default:
    return false;
  }
}

static bool isPotentialBlockingStoreInst(unsigned Opcode, unsigned LoadOpcode) {
  bool PBlock = false;
  PBlock |= Opcode == LoongArch::ST_D || Opcode == LoongArch::ST_W ||
            Opcode == LoongArch::ST_H || Opcode == LoongArch::ST_B ||
            Opcode == LoongArch::STPTR_D || Opcode == LoongArch::STPTR_W ||
            Opcode == LoongArch::FST_D || Opcode == LoongArch::FST_S;
  if (LoadOpcode == LoongArch::XVLD)
    PBlock |= Opcode == LoongArch::VST;
  return PBlock;
}

static const int MOV256SZ = 32;
static const int MOV128SZ = 16;
static const int MOV64SZ = 8;
static const int MOV32SZ = 4;
static const int MOV16SZ = 2;
static const int MOV8SZ = 1;

static unsigned getRegSizeInBytes(const MachineInstr *LoadInst) {
  switch (LoadInst->getOpcode()) {
  case LoongArch::XVLD:
    return MOV256SZ;
  case LoongArch::VLD:
    return MOV128SZ;
  default:
    return MOV64SZ;
  }
}

// All instructions handled here use the (reg, base, imm) operand layout.
static MachineOperand &getBaseOperand(MachineInstr *MI) {
  return MI->getOperand(1);
}

static MachineOperand &getDispOperand(MachineInstr *MI) {
  return MI->getOperand(2);
}

// Relevant addressing modes contain only base register and immediate
// displacement or frameindex and immediate displacement.
static bool isRelevantAddressingMode(MachineInstr *MI) {
  const MachineOperand &Base = getBaseOperand(MI);
  const MachineOperand &Disp = getDispOperand(MI);
  return (Base.isReg() || Base.isFI()) && Disp.isImm();
}

// Every piece of the broken copy must still be encodable as reg+simm12.
static bool hasEncodableDisps(MachineInstr *MI, unsigned Size) {
  int64_t Disp = getDispOperand(MI).getImm();
  return isInt<12>(Disp) && isInt<12>(Disp + Size - 1);
}

// Collect instructions executed shortly before LoadInst: those before it in
// its block and in its predecessors and, if it is in a loop, those after it in
// its block (previous iteration). Stop at calls and after SFBInspectionLimit
// instructions, beyond which a blocked store has likely completed.
SmallVector<LoongArchAvoidSFB::PotentialBlocker, 4>
LoongArchAvoidSFB::findPotentialBlockers(MachineInstr *LoadInst) {
  SmallVector<PotentialBlocker, 4> PotentialBlockers;
  MachineBasicBlock *MBB = LoadInst->getParent();
  MachineLoop *L = MLI->getLoopFor(MBB);
  unsigned InspectionLimit = SFBInspectionLimit;

  // Scan a range of instructions backwards; return false on a call.
  auto ScanBackwards = [&](auto Begin, auto End, unsigned Limit,
                           bool ViaBackEdge, unsigned &Count) {
    for (auto I = Begin; I != End; ++I) {
      if (I->isMetaInstruction())
        continue;
      if (++Count >= Limit)
        break;
      if (I->isCall())
        return false;
      PotentialBlockers.push_back({&*I, ViaBackEdge});
    }
    return true;
  };

  unsigned BlockCount = 0;
  if (!ScanBackwards(std::next(MachineBasicBlock::reverse_iterator(LoadInst)),
                     MBB->rend(), InspectionLimit, false, BlockCount))
    return PotentialBlockers;
  // If we didn't get to the instructions limit try predecessing blocks.
  if (BlockCount >= InspectionLimit)
    return PotentialBlockers;
  unsigned LimitLeft = InspectionLimit - BlockCount;

  if (L) {
    unsigned Count = 0;
    ScanBackwards(MBB->rbegin(), MachineBasicBlock::reverse_iterator(LoadInst),
                  LimitLeft, true, Count);
  }
  for (MachineBasicBlock *PMBB : MBB->predecessors()) {
    // The tail of MBB itself was handled above.
    if (PMBB == MBB)
      continue;
    bool ViaBackEdge = L && L->getHeader() == MBB && L->contains(PMBB);
    unsigned Count = 0;
    ScanBackwards(PMBB->rbegin(), PMBB->rend(), LimitLeft, ViaBackEdge, Count);
  }
  return PotentialBlockers;
}

// Whether the store may write bytes that the load reads in a later iteration.
// Also require a common underlying object, so that copies between different
// buffers keep using vector accesses.
bool LoongArchAvoidSFB::mayAliasAcrossIterations(
    const MachineMemOperand &LdMMO, const MachineMemOperand &StMMO) const {
  const Value *LdVal = LdMMO.getValue();
  const Value *StVal = StMMO.getValue();
  if (!LdVal || !StVal)
    return false;
  if (AA->isNoAlias(MemoryLocation::getBeforeOrAfter(LdVal, LdMMO.getAAInfo()),
                    MemoryLocation::getBeforeOrAfter(StVal, StMMO.getAAInfo())))
    return false;
  // Look through loop header phis.
  SmallVector<const Value *, 4> LdObjs, StObjs;
  getUnderlyingObjects(LdVal, LdObjs);
  getUnderlyingObjects(StVal, StObjs);
  SmallPtrSet<const Value *, 4> LdSet(llvm::from_range, LdObjs);
  return any_of(StObjs, [&](const Value *V) { return LdSet.contains(V); });
}

void LoongArchAvoidSFB::buildCopy(MachineInstr *LoadInst, unsigned NLoadOpcode,
                                  int64_t LoadDisp, MachineInstr *StoreInst,
                                  unsigned NStoreOpcode, int64_t StoreDisp,
                                  unsigned Size, int64_t Offset) {
  MachineOperand &LoadBase = getBaseOperand(LoadInst);
  MachineOperand &StoreBase = getBaseOperand(StoreInst);
  MachineBasicBlock *MBB = LoadInst->getParent();
  MachineFunction *MF = MBB->getParent();
  MachineMemOperand *LMMO = *LoadInst->memoperands_begin();
  MachineMemOperand *SMMO = *StoreInst->memoperands_begin();

  const TargetRegisterClass *RC = NLoadOpcode == LoongArch::VLD
                                      ? &LoongArch::LSX128RegClass
                                      : &LoongArch::GPRRegClass;
  Register Reg = MRI->createVirtualRegister(RC);
  // Loads go to the original load and stores to the original store, which
  // keeps overlapping copies correct.
  MachineInstr *NewLoad =
      BuildMI(*MBB, LoadInst, LoadInst->getDebugLoc(), TII->get(NLoadOpcode),
              Reg)
          .add(LoadBase)
          .addImm(LoadDisp)
          .addMemOperand(MF->getMachineMemOperand(LMMO, Offset, Size));
  if (LoadBase.isReg())
    getBaseOperand(NewLoad).setIsKill(false);
  LLVM_DEBUG(NewLoad->dump());
  MachineInstr *NewStore =
      BuildMI(*MBB, StoreInst, StoreInst->getDebugLoc(), TII->get(NStoreOpcode))
          .addReg(Reg, RegState::Kill)
          .add(StoreBase)
          .addImm(StoreDisp)
          .addMemOperand(MF->getMachineMemOperand(SMMO, Offset, Size));
  if (StoreBase.isReg())
    getBaseOperand(NewStore).setIsKill(false);
  LLVM_DEBUG(NewStore->dump());
}

void LoongArchAvoidSFB::buildCopies(int Size, MachineInstr *LoadInst,
                                    int64_t LdDispImm, MachineInstr *StoreInst,
                                    int64_t StDispImm, int64_t Offset) {
  static const struct {
    int Size;
    unsigned LoadOpc, StoreOpc;
  } Pieces[] = {{MOV128SZ, LoongArch::VLD, LoongArch::VST},
                {MOV64SZ, LoongArch::LD_D, LoongArch::ST_D},
                {MOV32SZ, LoongArch::LD_W, LoongArch::ST_W},
                {MOV16SZ, LoongArch::LD_H, LoongArch::ST_H},
                {MOV8SZ, LoongArch::LD_B, LoongArch::ST_B}};
  int64_t LdDisp = LdDispImm;
  int64_t StDisp = StDispImm;
  while (Size > 0) {
    for (const auto &P : Pieces) {
      // 128-bit pieces are only used to break 256-bit copies.
      if (P.Size == MOV128SZ && LoadInst->getOpcode() != LoongArch::XVLD)
        continue;
      if (Size < P.Size)
        continue;
      buildCopy(LoadInst, P.LoadOpc, LdDisp, StoreInst, P.StoreOpc, StDisp,
                P.Size, Offset);
      Size -= P.Size;
      LdDisp += P.Size;
      StDisp += P.Size;
      Offset += P.Size;
      break;
    }
  }
  assert(Size == 0 && "Wrong size division");
}

void LoongArchAvoidSFB::findPotentiallyBlockedCopies(MachineFunction &MF) {
  for (auto &MBB : MF)
    for (auto &MI : MBB) {
      if (!isPotentialBlockedMemCpyLd(MI.getOpcode()))
        continue;
      Register DefVR = MI.getOperand(0).getReg();
      if (!MRI->hasOneNonDBGUse(DefVR))
        continue;
      MachineInstr &StoreMI = *MRI->use_instr_nodbg_begin(DefVR);
      if (StoreMI.getParent() != MI.getParent() ||
          !isPotentialBlockedMemCpyPair(MI.getOpcode(), StoreMI.getOpcode()) ||
          StoreMI.getOperand(0).getReg() != DefVR ||
          !isRelevantAddressingMode(&MI) ||
          !isRelevantAddressingMode(&StoreMI) ||
          !hasEncodableDisps(&MI, getRegSizeInBytes(&MI)) ||
          !hasEncodableDisps(&StoreMI, getRegSizeInBytes(&MI)) ||
          !MI.hasOneMemOperand() || !StoreMI.hasOneMemOperand())
        continue;
      // Don't split volatile or atomic accesses.
      const MachineMemOperand *LMMO = *MI.memoperands_begin();
      const MachineMemOperand *SMMO = *StoreMI.memoperands_begin();
      if (LMMO->isVolatile() || LMMO->isAtomic() || SMMO->isVolatile() ||
          SMMO->isAtomic())
        continue;
      BlockedLoadsStoresPairs.push_back(std::make_pair(&MI, &StoreMI));
    }
}

void LoongArchAvoidSFB::breakBlockedCopies(
    MachineInstr *LoadInst, MachineInstr *StoreInst,
    const DisplacementSizeMap &BlockingStoresDispSizeMap) {
  int64_t LdDispImm = getDispOperand(LoadInst).getImm();
  int64_t StDispImm = getDispOperand(StoreInst).getImm();
  int64_t Offset = 0;

  int64_t LdDisp1 = LdDispImm;
  int64_t LdDisp2 = 0;
  int64_t StDisp1 = StDispImm;
  int64_t StDisp2 = 0;
  unsigned Size1 = 0;
  unsigned Size2 = 0;
  int64_t LdStDelta = StDispImm - LdDispImm;

  for (const auto &[Disp, Size] : BlockingStoresDispSizeMap) {
    LdDisp2 = Disp;
    StDisp2 = Disp + LdStDelta;
    Size2 = Size;
    // Avoid copying overlapping areas.
    if (LdDisp2 < LdDisp1) {
      int OverlapDelta = LdDisp1 - LdDisp2;
      LdDisp2 += OverlapDelta;
      StDisp2 += OverlapDelta;
      Size2 -= OverlapDelta;
    }
    Size1 = LdDisp2 - LdDisp1;

    // Build a copy for the point until the current blocking store's
    // displacement.
    buildCopies(Size1, LoadInst, LdDisp1, StoreInst, StDisp1, Offset);
    // Build a copy for the current blocking store.
    buildCopies(Size2, LoadInst, LdDisp2, StoreInst, StDisp2, Offset + Size1);
    LdDisp1 = LdDisp2 + Size2;
    StDisp1 = StDisp2 + Size2;
    Offset += Size1 + Size2;
  }
  unsigned Size3 = (LdDispImm + getRegSizeInBytes(LoadInst)) - LdDisp1;
  buildCopies(Size3, LoadInst, LdDisp1, StoreInst, StDisp1, Offset);
}

static bool hasSameBaseOpValue(MachineInstr *LoadInst,
                               MachineInstr *StoreInst) {
  const MachineOperand &LoadBase = getBaseOperand(LoadInst);
  const MachineOperand &StoreBase = getBaseOperand(StoreInst);
  if (LoadBase.isReg() != StoreBase.isReg())
    return false;
  if (LoadBase.isReg())
    return LoadBase.getReg() == StoreBase.getReg();
  return LoadBase.getIndex() == StoreBase.getIndex();
}

static bool isBlockingStore(int64_t LoadDispImm, unsigned LoadSize,
                            int64_t StoreDispImm, unsigned StoreSize) {
  return ((StoreDispImm >= LoadDispImm) &&
          (StoreDispImm <= LoadDispImm + (LoadSize - StoreSize)));
}

// Keep track of all stores blocking a load
static void
updateBlockingStoresDispSizeMap(DisplacementSizeMap &BlockingStoresDispSizeMap,
                                int64_t DispImm, unsigned Size) {
  auto [It, Inserted] = BlockingStoresDispSizeMap.try_emplace(DispImm, Size);
  // Choose the smallest blocking store starting at this displacement.
  if (!Inserted && It->second > Size)
    It->second = Size;
}

// Remove blocking stores contained in each other.
static void
removeRedundantBlockingStores(DisplacementSizeMap &BlockingStoresDispSizeMap) {
  if (BlockingStoresDispSizeMap.size() <= 1)
    return;

  SmallVector<std::pair<int64_t, unsigned>, 0> DispSizeStack;
  for (auto DispSizePair : BlockingStoresDispSizeMap) {
    int64_t CurrDisp = DispSizePair.first;
    unsigned CurrSize = DispSizePair.second;
    while (DispSizeStack.size()) {
      int64_t PrevDisp = DispSizeStack.back().first;
      unsigned PrevSize = DispSizeStack.back().second;
      if (CurrDisp + CurrSize > PrevDisp + PrevSize)
        break;
      DispSizeStack.pop_back();
    }
    DispSizeStack.push_back(DispSizePair);
  }
  BlockingStoresDispSizeMap.clear();
  for (auto Disp : DispSizeStack)
    BlockingStoresDispSizeMap.insert(Disp);
}

bool LoongArchAvoidSFB::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()) || MF.getFunction().hasOptSize())
    return false;

  const auto &ST = MF.getSubtarget<LoongArchSubtarget>();
  if (DisableAvoidSFB || !ST.is64Bit())
    return false;

  MRI = &MF.getRegInfo();
  assert(MRI->isSSA() && "Expected MIR to be in SSA form");
  TII = ST.getInstrInfo();
  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  LLVM_DEBUG(dbgs() << "Start LoongArchAvoidStoreForwardBlocks\n";);
  // Look for a load then a store which look like a memcpy.
  findPotentiallyBlockedCopies(MF);

  bool Changed = false;
  for (auto [LoadInst, StoreInst] : BlockedLoadsStoresPairs) {
    int64_t LdDispImm = getDispOperand(LoadInst).getImm();
    unsigned LdSize = getRegSizeInBytes(LoadInst);
    const MachineMemOperand &LMMO = **LoadInst->memoperands_begin();
    DisplacementSizeMap BlockingStoresDispSizeMap;
    bool LoopCarriedBlock = false;

    for (auto [PBInst, ViaBackEdge] : findPotentialBlockers(LoadInst)) {
      if (!PBInst->mayStore() || !PBInst->hasOneMemOperand())
        continue;
      const MachineMemOperand &PBMMO = **PBInst->memoperands_begin();
      if (isPotentialBlockingStoreInst(PBInst->getOpcode(),
                                       LoadInst->getOpcode()) &&
          isRelevantAddressingMode(PBInst) &&
          hasSameBaseOpValue(LoadInst, PBInst)) {
        // Known displacement: a narrower store inside the loaded bytes.
        int64_t PBstDispImm = getDispOperand(PBInst).getImm();
        unsigned PBstSize = PBMMO.getSize().getValue();
        if (!ViaBackEdge && PBstSize < LdSize &&
            isBlockingStore(LdDispImm, LdSize, PBstDispImm, PBstSize))
          updateBlockingStoresDispSizeMap(BlockingStoresDispSizeMap,
                                          PBstDispImm, PBstSize);
        continue;
      }
      // Unknown displacement: a store of the previous iteration to the same
      // object, wide enough to forward to 8-byte pieces.
      if (ViaBackEdge && LdSize > MOV64SZ && PBMMO.getSize().hasValue() &&
          PBMMO.getSize().getValue() >= MOV64SZ &&
          mayAliasAcrossIterations(LMMO, PBMMO))
        LoopCarriedBlock = true;
    }

    if (LoopCarriedBlock) {
      // The store may hit any part of the load: copy 8 bytes at a time.
      BlockingStoresDispSizeMap.clear();
      for (unsigned I = 0; I < LdSize; I += MOV64SZ)
        BlockingStoresDispSizeMap[LdDispImm + I] = MOV64SZ;
    }

    if (BlockingStoresDispSizeMap.empty())
      continue;

    // Break the copy around the blocking stores.
    LLVM_DEBUG(dbgs() << "Blocked load and store instructions: \n");
    LLVM_DEBUG(LoadInst->dump());
    LLVM_DEBUG(StoreInst->dump());
    LLVM_DEBUG(dbgs() << "Replaced with:\n");
    removeRedundantBlockingStores(BlockingStoresDispSizeMap);
    breakBlockedCopies(LoadInst, StoreInst, BlockingStoresDispSizeMap);
    // The base registers are read again by the new instructions.
    for (MachineInstr *MI : {LoadInst, StoreInst})
      if (getBaseOperand(MI).isReg())
        MRI->clearKillFlags(getBaseOperand(MI).getReg());
    ForRemoval.push_back(LoadInst);
    ForRemoval.push_back(StoreInst);
    ++NumBrokenCopies;
    if (LdSize == MOV64SZ)
      ++NumBrokenScalarCopies;
    if (LoopCarriedBlock)
      ++NumLoopCarriedBlocks;
    Changed = true;
  }
  for (auto *RemovedInst : ForRemoval)
    RemovedInst->eraseFromParent();
  ForRemoval.clear();
  BlockedLoadsStoresPairs.clear();
  LLVM_DEBUG(dbgs() << "End LoongArchAvoidStoreForwardBlocks\n";);

  return Changed;
}
