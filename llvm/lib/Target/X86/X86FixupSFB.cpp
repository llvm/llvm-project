//===- X86FixupSFB.cpp - Avoid HW Store Forward Block issues -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// If a load follows a store and reloads data that the store has written to
// memory, Intel microarchitectures can in many cases forward the data directly
// from the store to the load, This "store forwarding" saves cycles by enabling
// the load to directly obtain the data instead of accessing the data from
// cache or memory.
// A "store forward block" occurs in cases that a store cannot be forwarded to
// the load. The most typical case of store forward block on Intel Core
// microarchitecture that a small store cannot be forwarded to a large load.
// The estimated penalty for a store forward block is ~13 cycles.
//
// This pass tries to recognize and handle cases where "store forward block"
// is created by the compiler when lowering memcpy calls to a sequence
// of a load and a store.
//
// The pass currently only handles cases where memcpy is lowered to
// XMM/YMM registers, it tries to break the memcpy into smaller copies.
// breaking the memcpy should be possible since there is no atomicity
// guarantee for loads and stores to XMM/YMM.
//
// It could be better for performance to solve the problem by loading
// to XMM/YMM then inserting the partial store before storing back from XMM/YMM
// to memory, but this will result in a more conservative optimization since it
// requires we prove that all memory accesses between the blocking store and the
// load must alias/don't alias before we can move the store, whereas the
// transformation done here is correct regardless to other memory accesses.
//===----------------------------------------------------------------------===//

#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/MCInstrDesc.h"

using namespace llvm;

#define DEBUG_TYPE "x86-fixup-SFB"

static cl::opt<bool> DisableX86FixupSFB("disable-fixup-SFB", cl::Hidden,
                                        cl::desc("X86: Disable SFB fixup."),
                                        cl::init(false));
namespace {

class FixupSFBPass : public MachineFunctionPass {
public:
  FixupSFBPass() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "X86 Fixup Store Forward Block";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  MachineRegisterInfo *MRI;
  const X86InstrInfo *TII;
  const X86RegisterInfo *TRI;
  SmallVector<std::pair<MachineInstr *, MachineInstr *>, 2> BlockedLoadsStores;
  SmallVector<MachineInstr *, 2> ForRemoval;
  bool Is64Bit;

  /// \brief Returns couples of Load then Store to memory which look
  ///  like a memcpy.
  void findPotentiallylBlockedCopies(MachineFunction &MF);
  /// \brief Break the memcpy's load and store into smaller copies
  /// such that each memory load that was blocked by a smaller store
  /// would now be copied separately.
  void
  breakBlockedCopies(MachineInstr *LoadInst, MachineInstr *StoreInst,
                     const std::map<int64_t, unsigned> &BlockingStoresDisp);
  /// \brief Break a copy of size Size to smaller copies.
  void buildCopies(int Size, MachineInstr *LoadInst, int64_t LdDispImm,
                   MachineInstr *StoreInst, int64_t StDispImm,
                   int64_t LMMOffset, int64_t SMMOffset);

  void buildCopy(MachineInstr *LoadInst, unsigned NLoadOpcode, int64_t LoadDisp,
                 MachineInstr *StoreInst, unsigned NStoreOpcode,
                 int64_t StoreDisp, unsigned Size, int64_t LMMOffset,
                 int64_t SMMOffset);

  unsigned getRegSizeInBytes(MachineInstr *Inst);
  static char ID;
};

} // end anonymous namespace

char FixupSFBPass::ID = 0;

FunctionPass *llvm::createX86FixupSFB() { return new FixupSFBPass(); }

static bool isXMMLoadOpcode(unsigned Opcode) {
  return Opcode == X86::MOVUPSrm || Opcode == X86::MOVAPSrm ||
         Opcode == X86::VMOVUPSrm || Opcode == X86::VMOVAPSrm ||
         Opcode == X86::VMOVUPDrm || Opcode == X86::VMOVAPDrm ||
         Opcode == X86::VMOVDQUrm || Opcode == X86::VMOVDQArm ||
         Opcode == X86::VMOVUPSZ128rm || Opcode == X86::VMOVAPSZ128rm ||
         Opcode == X86::VMOVUPDZ128rm || Opcode == X86::VMOVAPDZ128rm ||
         Opcode == X86::VMOVDQU64Z128rm || Opcode == X86::VMOVDQA64Z128rm ||
         Opcode == X86::VMOVDQU32Z128rm || Opcode == X86::VMOVDQA32Z128rm;
}
static bool isYMMLoadOpcode(unsigned Opcode) {
  return Opcode == X86::VMOVUPSYrm || Opcode == X86::VMOVAPSYrm ||
         Opcode == X86::VMOVUPDYrm || Opcode == X86::VMOVAPDYrm ||
         Opcode == X86::VMOVDQUYrm || Opcode == X86::VMOVDQAYrm ||
         Opcode == X86::VMOVUPSZ256rm || Opcode == X86::VMOVAPSZ256rm ||
         Opcode == X86::VMOVUPDZ256rm || Opcode == X86::VMOVAPDZ256rm ||
         Opcode == X86::VMOVDQU64Z256rm || Opcode == X86::VMOVDQA64Z256rm ||
         Opcode == X86::VMOVDQU32Z256rm || Opcode == X86::VMOVDQA32Z256rm;
}

static bool isPotentialBlockedMemCpyLd(unsigned Opcode) {
  return isXMMLoadOpcode(Opcode) || isYMMLoadOpcode(Opcode);
}

std::map<unsigned, std::pair<unsigned, unsigned>> PotentialBlockedMemCpy{
    {X86::MOVUPSrm, {X86::MOVUPSmr, X86::MOVAPSmr}},
    {X86::MOVAPSrm, {X86::MOVUPSmr, X86::MOVAPSmr}},
    {X86::VMOVUPSrm, {X86::VMOVUPSmr, X86::VMOVAPSmr}},
    {X86::VMOVAPSrm, {X86::VMOVUPSmr, X86::VMOVAPSmr}},
    {X86::VMOVUPDrm, {X86::VMOVUPDmr, X86::VMOVAPDmr}},
    {X86::VMOVAPDrm, {X86::VMOVUPDmr, X86::VMOVAPDmr}},
    {X86::VMOVDQUrm, {X86::VMOVDQUmr, X86::VMOVDQAmr}},
    {X86::VMOVDQArm, {X86::VMOVDQUmr, X86::VMOVDQAmr}},
    {X86::VMOVUPSZ128rm, {X86::VMOVUPSZ128mr, X86::VMOVAPSZ128mr}},
    {X86::VMOVAPSZ128rm, {X86::VMOVUPSZ128mr, X86::VMOVAPSZ128mr}},
    {X86::VMOVUPDZ128rm, {X86::VMOVUPDZ128mr, X86::VMOVAPDZ128mr}},
    {X86::VMOVAPDZ128rm, {X86::VMOVUPDZ128mr, X86::VMOVAPDZ128mr}},
    {X86::VMOVUPSYrm, {X86::VMOVUPSYmr, X86::VMOVAPSYmr}},
    {X86::VMOVAPSYrm, {X86::VMOVUPSYmr, X86::VMOVAPSYmr}},
    {X86::VMOVUPDYrm, {X86::VMOVUPDYmr, X86::VMOVAPDYmr}},
    {X86::VMOVAPDYrm, {X86::VMOVUPDYmr, X86::VMOVAPDYmr}},
    {X86::VMOVDQUYrm, {X86::VMOVDQUYmr, X86::VMOVDQAYmr}},
    {X86::VMOVDQAYrm, {X86::VMOVDQUYmr, X86::VMOVDQAYmr}},
    {X86::VMOVUPSZ256rm, {X86::VMOVUPSZ256mr, X86::VMOVAPSZ256mr}},
    {X86::VMOVAPSZ256rm, {X86::VMOVUPSZ256mr, X86::VMOVAPSZ256mr}},
    {X86::VMOVUPDZ256rm, {X86::VMOVUPDZ256mr, X86::VMOVAPDZ256mr}},
    {X86::VMOVAPDZ256rm, {X86::VMOVUPDZ256mr, X86::VMOVAPDZ256mr}},
    {X86::VMOVDQU64Z128rm, {X86::VMOVDQU64Z128mr, X86::VMOVDQA64Z128mr}},
    {X86::VMOVDQA64Z128rm, {X86::VMOVDQU64Z128mr, X86::VMOVDQA64Z128mr}},
    {X86::VMOVDQU32Z128rm, {X86::VMOVDQU32Z128mr, X86::VMOVDQA32Z128mr}},
    {X86::VMOVDQA32Z128rm, {X86::VMOVDQU32Z128mr, X86::VMOVDQA32Z128mr}},
    {X86::VMOVDQU64Z256rm, {X86::VMOVDQU64Z256mr, X86::VMOVDQA64Z256mr}},
    {X86::VMOVDQA64Z256rm, {X86::VMOVDQU64Z256mr, X86::VMOVDQA64Z256mr}},
    {X86::VMOVDQU32Z256rm, {X86::VMOVDQU32Z256mr, X86::VMOVDQA32Z256mr}},
    {X86::VMOVDQA32Z256rm, {X86::VMOVDQU32Z256mr, X86::VMOVDQA32Z256mr}},
};

static bool isPotentialBlockedMemCpyPair(unsigned LdOpcode, unsigned StOpcode) {
  auto PotentialStores = PotentialBlockedMemCpy.at(LdOpcode);
  return PotentialStores.first == StOpcode ||
         PotentialStores.second == StOpcode;
}

static bool isPotentialBlockingStoreInst(int Opcode, int LoadOpcode) {
  bool PBlock = false;
  PBlock |= Opcode == X86::MOV64mr || Opcode == X86::MOV64mi32 ||
            Opcode == X86::MOV32mr || Opcode == X86::MOV32mi ||
            Opcode == X86::MOV16mr || Opcode == X86::MOV16mi ||
            Opcode == X86::MOV8mr || Opcode == X86::MOV8mi;
  if (isYMMLoadOpcode(LoadOpcode))
    PBlock |= Opcode == X86::VMOVUPSmr || Opcode == X86::VMOVAPSmr ||
              Opcode == X86::VMOVUPDmr || Opcode == X86::VMOVAPDmr ||
              Opcode == X86::VMOVDQUmr || Opcode == X86::VMOVDQAmr ||
              Opcode == X86::VMOVUPSZ128mr || Opcode == X86::VMOVAPSZ128mr ||
              Opcode == X86::VMOVUPDZ128mr || Opcode == X86::VMOVAPDZ128mr ||
              Opcode == X86::VMOVDQU64Z128mr ||
              Opcode == X86::VMOVDQA64Z128mr ||
              Opcode == X86::VMOVDQU32Z128mr || Opcode == X86::VMOVDQA32Z128mr;
  return PBlock;
}

static const int MOV128SZ = 16;
static const int MOV64SZ = 8;
static const int MOV32SZ = 4;
static const int MOV16SZ = 2;
static const int MOV8SZ = 1;

std::map<unsigned, unsigned> YMMtoXMMLoadMap = {
    {X86::VMOVUPSYrm, X86::VMOVUPSrm},
    {X86::VMOVAPSYrm, X86::VMOVUPSrm},
    {X86::VMOVUPDYrm, X86::VMOVUPDrm},
    {X86::VMOVAPDYrm, X86::VMOVUPDrm},
    {X86::VMOVDQUYrm, X86::VMOVDQUrm},
    {X86::VMOVDQAYrm, X86::VMOVDQUrm},
    {X86::VMOVUPSZ256rm, X86::VMOVUPSZ128rm},
    {X86::VMOVAPSZ256rm, X86::VMOVUPSZ128rm},
    {X86::VMOVUPDZ256rm, X86::VMOVUPDZ128rm},
    {X86::VMOVAPDZ256rm, X86::VMOVUPDZ128rm},
    {X86::VMOVDQU64Z256rm, X86::VMOVDQU64Z128rm},
    {X86::VMOVDQA64Z256rm, X86::VMOVDQU64Z128rm},
    {X86::VMOVDQU32Z256rm, X86::VMOVDQU32Z128rm},
    {X86::VMOVDQA32Z256rm, X86::VMOVDQU32Z128rm},
};

std::map<unsigned, unsigned> YMMtoXMMStoreMap = {
    {X86::VMOVUPSYmr, X86::VMOVUPSmr},
    {X86::VMOVAPSYmr, X86::VMOVUPSmr},
    {X86::VMOVUPDYmr, X86::VMOVUPDmr},
    {X86::VMOVAPDYmr, X86::VMOVUPDmr},
    {X86::VMOVDQUYmr, X86::VMOVDQUmr},
    {X86::VMOVDQAYmr, X86::VMOVDQUmr},
    {X86::VMOVUPSZ256mr, X86::VMOVUPSZ128mr},
    {X86::VMOVAPSZ256mr, X86::VMOVUPSZ128mr},
    {X86::VMOVUPDZ256mr, X86::VMOVUPDZ128mr},
    {X86::VMOVAPDZ256mr, X86::VMOVUPDZ128mr},
    {X86::VMOVDQU64Z256mr, X86::VMOVDQU64Z128mr},
    {X86::VMOVDQA64Z256mr, X86::VMOVDQU64Z128mr},
    {X86::VMOVDQU32Z256mr, X86::VMOVDQU32Z128mr},
    {X86::VMOVDQA32Z256mr, X86::VMOVDQU32Z128mr},
};

static int getAddrOffset(MachineInstr *MI) {
  const MCInstrDesc &Descl = MI->getDesc();
  int AddrOffset = X86II::getMemoryOperandNo(Descl.TSFlags);
  assert(AddrOffset != -1 && "Expected Memory Operand");
  AddrOffset += X86II::getOperandBias(Descl);
  return AddrOffset;
}

static MachineOperand &getBaseOperand(MachineInstr *MI) {
  int AddrOffset = getAddrOffset(MI);
  return MI->getOperand(AddrOffset + X86::AddrBaseReg);
}

static MachineOperand &getDispOperand(MachineInstr *MI) {
  int AddrOffset = getAddrOffset(MI);
  return MI->getOperand(AddrOffset + X86::AddrDisp);
}

// Relevant addressing modes contain only base register and immediate
// displacement or frameindex and immediate displacement.
// TODO: Consider expanding to other addressing modes in the future
static bool isRelevantAddressingMode(MachineInstr *MI) {
  int AddrOffset = getAddrOffset(MI);
  MachineOperand &Base = MI->getOperand(AddrOffset + X86::AddrBaseReg);
  MachineOperand &Disp = MI->getOperand(AddrOffset + X86::AddrDisp);
  MachineOperand &Scale = MI->getOperand(AddrOffset + X86::AddrScaleAmt);
  MachineOperand &Index = MI->getOperand(AddrOffset + X86::AddrIndexReg);
  MachineOperand &Segment = MI->getOperand(AddrOffset + X86::AddrSegmentReg);

  if (!((Base.isReg() && Base.getReg() != X86::NoRegister) || Base.isFI()))
    return false;
  if (!Disp.isImm())
    return false;
  if (Scale.getImm() != 1)
    return false;
  if (!(Index.isReg() && Index.getReg() == X86::NoRegister))
    return false;
  if (!(Segment.isReg() && Segment.getReg() == X86::NoRegister))
    return false;
  return true;
}

// Collect potentially blocking stores.
// Limit the number of instructions backwards we want to inspect
// since the effect of store block won't be visible if the store
// and load instructions have enough instructions in between to
// keep the core busy.
static const unsigned LIMIT = 20;
static SmallVector<MachineInstr *, 2>
findPotentialBlockers(MachineInstr *LoadInst) {
  SmallVector<MachineInstr *, 2> PotentialBlockers;
  unsigned BlockLimit = 0;
  for (MachineBasicBlock::iterator LI = LoadInst,
                                   BB = LoadInst->getParent()->begin();
       LI != BB; --LI) {
    BlockLimit++;
    if (BlockLimit >= LIMIT)
      break;
    MachineInstr &MI = *LI;
    if (MI.getDesc().isCall())
      break;
    PotentialBlockers.push_back(&MI);
  }
  // If we didn't get to the instructions limit try predecessing blocks.
  // Ideally we should traverse the predecessor blocks in depth with some
  // coloring algorithm, but for now let's just look at the first order
  // predecessors.
  if (BlockLimit < LIMIT) {
    MachineBasicBlock *MBB = LoadInst->getParent();
    int LimitLeft = LIMIT - BlockLimit;
    for (MachineBasicBlock::pred_iterator PB = MBB->pred_begin(),
                                          PE = MBB->pred_end();
         PB != PE; ++PB) {
      MachineBasicBlock *PMBB = *PB;
      int PredLimit = 0;
      for (MachineBasicBlock::reverse_iterator PMI = PMBB->rbegin(),
                                               PME = PMBB->rend();
           PMI != PME; ++PMI) {
        PredLimit++;
        if (PredLimit >= LimitLeft)
          break;
        if (PMI->getDesc().isCall())
          break;
        PotentialBlockers.push_back(&*PMI);
      }
    }
  }
  return PotentialBlockers;
}

void FixupSFBPass::buildCopy(MachineInstr *LoadInst, unsigned NLoadOpcode,
                             int64_t LoadDisp, MachineInstr *StoreInst,
                             unsigned NStoreOpcode, int64_t StoreDisp,
                             unsigned Size, int64_t LMMOffset,
                             int64_t SMMOffset) {
  MachineOperand &LoadBase = getBaseOperand(LoadInst);
  MachineOperand &StoreBase = getBaseOperand(StoreInst);
  MachineBasicBlock *MBB = LoadInst->getParent();
  MachineMemOperand *LMMO = *LoadInst->memoperands_begin();
  MachineMemOperand *SMMO = *StoreInst->memoperands_begin();

  unsigned Reg1 = MRI->createVirtualRegister(
      TII->getRegClass(TII->get(NLoadOpcode), 0, TRI, *(MBB->getParent())));
  BuildMI(*MBB, LoadInst, LoadInst->getDebugLoc(), TII->get(NLoadOpcode), Reg1)
      .add(LoadBase)
      .addImm(1)
      .addReg(X86::NoRegister)
      .addImm(LoadDisp)
      .addReg(X86::NoRegister)
      .addMemOperand(
          MBB->getParent()->getMachineMemOperand(LMMO, LMMOffset, Size));
  DEBUG(LoadInst->getPrevNode()->dump());
  // If the load and store are consecutive, use the loadInst location to
  // reduce register pressure.
  MachineInstr *StInst = StoreInst;
  if (StoreInst->getPrevNode() == LoadInst)
    StInst = LoadInst;
  BuildMI(*MBB, StInst, StInst->getDebugLoc(), TII->get(NStoreOpcode))
      .add(StoreBase)
      .addImm(1)
      .addReg(X86::NoRegister)
      .addImm(StoreDisp)
      .addReg(X86::NoRegister)
      .addReg(Reg1)
      .addMemOperand(
          MBB->getParent()->getMachineMemOperand(SMMO, SMMOffset, Size));
  DEBUG(StInst->getPrevNode()->dump());
}

void FixupSFBPass::buildCopies(int Size, MachineInstr *LoadInst,
                               int64_t LdDispImm, MachineInstr *StoreInst,
                               int64_t StDispImm, int64_t LMMOffset,
                               int64_t SMMOffset) {
  int LdDisp = LdDispImm;
  int StDisp = StDispImm;
  while (Size > 0) {
    if ((Size - MOV128SZ >= 0) && isYMMLoadOpcode(LoadInst->getOpcode())) {
      Size = Size - MOV128SZ;
      buildCopy(LoadInst, YMMtoXMMLoadMap.at(LoadInst->getOpcode()), LdDisp,
                StoreInst, YMMtoXMMStoreMap.at(StoreInst->getOpcode()), StDisp,
                MOV128SZ, LMMOffset, SMMOffset);
      LdDisp += MOV128SZ;
      StDisp += MOV128SZ;
      LMMOffset += MOV128SZ;
      SMMOffset += MOV128SZ;
      continue;
    }
    if (Size - MOV64SZ >= 0 && Is64Bit) {
      Size = Size - MOV64SZ;
      buildCopy(LoadInst, X86::MOV64rm, LdDisp, StoreInst, X86::MOV64mr, StDisp,
                MOV64SZ, LMMOffset, SMMOffset);
      LdDisp += MOV64SZ;
      StDisp += MOV64SZ;
      LMMOffset += MOV64SZ;
      SMMOffset += MOV64SZ;
      continue;
    }
    if (Size - MOV32SZ >= 0) {
      Size = Size - MOV32SZ;
      buildCopy(LoadInst, X86::MOV32rm, LdDisp, StoreInst, X86::MOV32mr, StDisp,
                MOV32SZ, LMMOffset, SMMOffset);
      LdDisp += MOV32SZ;
      StDisp += MOV32SZ;
      LMMOffset += MOV32SZ;
      SMMOffset += MOV32SZ;
      continue;
    }
    if (Size - MOV16SZ >= 0) {
      Size = Size - MOV16SZ;
      buildCopy(LoadInst, X86::MOV16rm, LdDisp, StoreInst, X86::MOV16mr, StDisp,
                MOV16SZ, LMMOffset, SMMOffset);
      LdDisp += MOV16SZ;
      StDisp += MOV16SZ;
      LMMOffset += MOV16SZ;
      SMMOffset += MOV16SZ;
      continue;
    }
    if (Size - MOV8SZ >= 0) {
      Size = Size - MOV8SZ;
      buildCopy(LoadInst, X86::MOV8rm, LdDisp, StoreInst, X86::MOV8mr, StDisp,
                MOV8SZ, LMMOffset, SMMOffset);
      LdDisp += MOV8SZ;
      StDisp += MOV8SZ;
      LMMOffset += MOV8SZ;
      SMMOffset += MOV8SZ;
      continue;
    }
  }
  assert(Size == 0 && "Wrong size division");
}

static void updateKillStatus(MachineInstr *LoadInst, MachineInstr *StoreInst) {
  MachineOperand &LoadBase = getBaseOperand(LoadInst);
  MachineOperand &StoreBase = getBaseOperand(StoreInst);
  if (LoadBase.isReg()) {
    MachineInstr *LastLoad = LoadInst->getPrevNode();
    // If the original load and store to xmm/ymm were consecutive
    // then the partial copies were also created in
    // a consecutive order to reduce register pressure,
    // and the location of the last load is before the last store.
    if (StoreInst->getPrevNode() == LoadInst)
      LastLoad = LoadInst->getPrevNode()->getPrevNode();
    getBaseOperand(LastLoad).setIsKill(LoadBase.isKill());
  }
  if (StoreBase.isReg()) {
    MachineInstr *StInst = StoreInst;
    if (StoreInst->getPrevNode() == LoadInst)
      StInst = LoadInst;
    getBaseOperand(StInst->getPrevNode()).setIsKill(StoreBase.isKill());
  }
}

void FixupSFBPass::findPotentiallylBlockedCopies(MachineFunction &MF) {
  for (auto &MBB : MF)
    for (auto &MI : MBB)
      if (isPotentialBlockedMemCpyLd(MI.getOpcode())) {
        int DefVR = MI.getOperand(0).getReg();
        if (MRI->hasOneUse(DefVR))
          for (auto UI = MRI->use_nodbg_begin(DefVR), UE = MRI->use_nodbg_end();
               UI != UE;) {
            MachineOperand &StoreMO = *UI++;
            MachineInstr &StoreMI = *StoreMO.getParent();
            if (isPotentialBlockedMemCpyPair(MI.getOpcode(),
                                             StoreMI.getOpcode()) &&
                (StoreMI.getParent() == MI.getParent()))
              if (isRelevantAddressingMode(&MI) &&
                  isRelevantAddressingMode(&StoreMI))
                BlockedLoadsStores.push_back(
                    std::pair<MachineInstr *, MachineInstr *>(&MI, &StoreMI));
          }
      }
}
unsigned FixupSFBPass::getRegSizeInBytes(MachineInstr *LoadInst) {
  auto TRC = TII->getRegClass(TII->get(LoadInst->getOpcode()), 0, TRI,
                              *LoadInst->getParent()->getParent());
  return TRI->getRegSizeInBits(*TRC) / 8;
}

void FixupSFBPass::breakBlockedCopies(
    MachineInstr *LoadInst, MachineInstr *StoreInst,
    const std::map<int64_t, unsigned> &BlockingStoresDisp) {
  int64_t LdDispImm = getDispOperand(LoadInst).getImm();
  int64_t StDispImm = getDispOperand(StoreInst).getImm();
  int64_t LMMOffset = (*LoadInst->memoperands_begin())->getOffset();
  int64_t SMMOffset = (*StoreInst->memoperands_begin())->getOffset();

  int64_t LdDisp1 = LdDispImm;
  int64_t LdDisp2 = 0;
  int64_t StDisp1 = StDispImm;
  int64_t StDisp2 = 0;
  unsigned Size1 = 0;
  unsigned Size2 = 0;
  int64_t LdStDelta = StDispImm - LdDispImm;
  for (auto inst : BlockingStoresDisp) {
    LdDisp2 = inst.first;
    StDisp2 = inst.first + LdStDelta;
    Size1 = std::abs(std::abs(LdDisp2) - std::abs(LdDisp1));
    Size2 = inst.second;
    buildCopies(Size1, LoadInst, LdDisp1, StoreInst, StDisp1, LMMOffset,
                SMMOffset);
    buildCopies(Size2, LoadInst, LdDisp2, StoreInst, StDisp2, LMMOffset + Size1,
                SMMOffset + Size1);
    LdDisp1 = LdDisp2 + Size2;
    StDisp1 = StDisp2 + Size2;
    LMMOffset += Size1 + Size2;
    SMMOffset += Size1 + Size2;
  }
  unsigned Size3 = (LdDispImm + getRegSizeInBytes(LoadInst)) - LdDisp1;
  buildCopies(Size3, LoadInst, LdDisp1, StoreInst, StDisp1, LMMOffset,
              LMMOffset);
}

bool FixupSFBPass::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;

  if (DisableX86FixupSFB || skipFunction(MF.getFunction()))
    return false;

  MRI = &MF.getRegInfo();
  assert(MRI->isSSA() && "Expected MIR to be in SSA form");
  TII = MF.getSubtarget<X86Subtarget>().getInstrInfo();
  TRI = MF.getSubtarget<X86Subtarget>().getRegisterInfo();
  Is64Bit = MF.getSubtarget<X86Subtarget>().is64Bit();
  DEBUG(dbgs() << "Start X86FixupSFB\n";);
  // Look for a load then a store to XMM/YMM which look like a memcpy
  findPotentiallylBlockedCopies(MF);

  for (auto LoadStoreInst : BlockedLoadsStores) {
    MachineInstr *LoadInst = LoadStoreInst.first;
    SmallVector<MachineInstr *, 2> PotentialBlockers =
        findPotentialBlockers(LoadInst);

    MachineOperand &LoadBase = getBaseOperand(LoadInst);
    int64_t LdDispImm = getDispOperand(LoadInst).getImm();
    std::map<int64_t, unsigned> BlockingStoresDisp;
    int LdBaseReg = LoadBase.isReg() ? LoadBase.getReg() : LoadBase.getIndex();

    for (auto PBInst : PotentialBlockers) {
      if (isPotentialBlockingStoreInst(PBInst->getOpcode(),
                                       LoadInst->getOpcode())) {
        if (!isRelevantAddressingMode(PBInst))
          continue;
        MachineOperand &PBstoreBase = getBaseOperand(PBInst);
        int64_t PBstDispImm = getDispOperand(PBInst).getImm();
        assert(PBInst->hasOneMemOperand() && "Expected One Memory Operand");
        unsigned PBstSize = (*PBInst->memoperands_begin())->getSize();
        int PBstBaseReg =
            PBstoreBase.isReg() ? PBstoreBase.getReg() : PBstoreBase.getIndex();
        // This check doesn't cover all cases, but it will suffice for now.
        // TODO: take branch probability into consideration, if the blocking
        // store is in an unreached block, breaking the memcopy could lose
        // performance.
        if (((LoadBase.isReg() && PBstoreBase.isReg()) ||
             (LoadBase.isFI() && PBstoreBase.isFI())) &&
            LdBaseReg == PBstBaseReg &&
            ((PBstDispImm >= LdDispImm) &&
             (PBstDispImm <=
              LdDispImm + (getRegSizeInBytes(LoadInst) - PBstSize)))) {
          if (BlockingStoresDisp.count(PBstDispImm)) {
            if (BlockingStoresDisp[PBstDispImm] > PBstSize)
              BlockingStoresDisp[PBstDispImm] = PBstSize;

          } else
            BlockingStoresDisp[PBstDispImm] = PBstSize;
        }
      }
    }

    if (BlockingStoresDisp.size() == 0)
      continue;

    // We found a store forward block, break the memcpy's load and store
    // into smaller copies such that each smaller store that was causing
    // a store block would now be copied separately.
    MachineInstr *StoreInst = LoadStoreInst.second;
    DEBUG(dbgs() << "Blocked load and store instructions: \n");
    DEBUG(LoadInst->dump());
    DEBUG(StoreInst->dump());
    DEBUG(dbgs() << "Replaced with:\n");
    breakBlockedCopies(LoadInst, StoreInst, BlockingStoresDisp);
    updateKillStatus(LoadInst, StoreInst);
    ForRemoval.push_back(LoadInst);
    ForRemoval.push_back(StoreInst);
  }
  for (auto RemovedInst : ForRemoval) {
    RemovedInst->eraseFromParent();
  }
  ForRemoval.clear();
  BlockedLoadsStores.clear();
  DEBUG(dbgs() << "End X86FixupSFB\n";);

  return Changed;
}
