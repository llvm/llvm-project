//===-- EZHConstantIslandPass.cpp - EZH Constant Islands ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EZH.h"
#include "EZHBasicBlockInfo.h"
#include "EZHConstantPoolValue.h"
#include "EZHInstrInfo.h"
#include "EZHSubtarget.h"
#include "MCTargetDesc/EZHMCTargetDesc.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Type.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>

#define DEBUG_TYPE "ezh-constant-islands"

#define DEBUG_PRINT(x) LLVM_DEBUG(dbgs() << x)

using namespace llvm;

namespace {
struct CPUser {
  MachineInstr *MI;
  MachineInstr *CPEMI;
  MachineBasicBlock *HighWaterMark;
  unsigned MaxDisp;
  bool NegOk;
  bool IsSoImm;
  bool KnownAlignment = false;

  CPUser(MachineInstr *mi, MachineInstr *cpemi, unsigned maxdisp, bool neg,
         bool soimm)
      : MI(mi), CPEMI(cpemi), MaxDisp(maxdisp), NegOk(neg), IsSoImm(soimm) {
    HighWaterMark = CPEMI->getParent();
  }

  unsigned getMaxDisp() const {
    return (KnownAlignment ? MaxDisp : MaxDisp - 2) - 2;
  }
};

struct CPEntry {
  MachineInstr *CPEMI;
  unsigned CPI;
  unsigned RefCount;

  CPEntry(MachineInstr *cpemi, unsigned cpi, unsigned rc = 0)
      : CPEMI(cpemi), CPI(cpi), RefCount(rc) {}
};

struct ImmBranch {
  MachineInstr *MI;
  unsigned MaxDisp : 31;
  unsigned isCond : 1;
  unsigned UncondBr;

  ImmBranch(MachineInstr *mi, unsigned maxdisp, bool cond, unsigned ubr)
      : MI(mi), MaxDisp(maxdisp), isCond(cond), UncondBr(ubr) {}
};

class EZHConstantIslandPass : public MachineFunctionPass {
  const TargetInstrInfo *TII;
  MachineFunction *MF;
  SmallVector<CPUser, 8> CPUsers;
  SmallVector<SmallVector<CPEntry, 2>, 8> CPEntries;
  DenseMap<int, int> JumpTableEntryIndices;
  DenseMap<int, int> JumpTableUserIndices;
  SmallVector<ImmBranch, 8> ImmBranches;
  SmallVector<MachineBasicBlock *, 8> WaterList;
  std::unique_ptr<EZHBasicBlockUtils> BBUtils = nullptr;

  void initializeFunctionInfo();

public:
  static char ID;
  EZHConstantIslandPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  MachineBasicBlock *findAvailableWater(CPUser &U, unsigned UserOffset);
  bool handleConstantPoolUser(unsigned CPI, bool CloserWater);
  MachineBasicBlock *createNewWater(MachineInstr &MI);
  bool fixupConditionalBr(ImmBranch &Br);
  bool fixupImmediateBr(ImmBranch &Br);
  void initializeWaterList();
  bool decrementCPEReferenceCount(unsigned CPI, MachineInstr *CPEMI);
  void updateForInsertedWaterBlock(MachineBasicBlock *NewBB);
};

char EZHConstantIslandPass::ID = 0;
} // namespace

void EZHConstantIslandPass::initializeWaterList() {
  WaterList.clear();
  for (auto &MBB : *MF) {
    if (!MBB.empty() &&
        (MBB.back().getOpcode() == EZH::GOTO || MBB.back().isReturn())) {
      WaterList.push_back(&MBB);
    }
  }
}

MachineBasicBlock *EZHConstantIslandPass::createNewWater(MachineInstr &MI) {
  MachineBasicBlock *MBB = MI.getParent();

  // Collect liveness information at MI.
  LivePhysRegs LRs(*MF->getSubtarget().getRegisterInfo());
  LRs.addLiveOuts(*MBB);
  auto LivenessEnd = ++MachineBasicBlock::iterator(MI).getReverse();
  for (MachineInstr &LiveMI : make_range(MBB->rbegin(), LivenessEnd))
    LRs.stepBackward(LiveMI);

  // Find split point
  MachineBasicBlock::iterator I = MI.getIterator();

  // Split block before I
  MachineBasicBlock *NewBB = MF->CreateMachineBasicBlock(MBB->getBasicBlock());
  MF->insert(std::next(MBB->getIterator()), NewBB);

  // Update live-in information in the new block.
  MachineRegisterInfo &MRI = MF->getRegInfo();
  for (MCPhysReg L : LRs)
    if (!MRI.isReserved(L))
      NewBB->addLiveIn(L);

  NewBB->splice(NewBB->begin(), MBB, I, MBB->end());

  // If MBB became empty, insert a dummy instruction to prevent it from being
  // removed!
  if (MBB->empty()) {
    const TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();
    BuildMI(MBB, DebugLoc(), TII->get(EZH::ADDri__))
        .addReg(EZH::R0)
        .addReg(EZH::R0)
        .addImm(0);
  }

  // Update CFG
  NewBB->transferSuccessors(MBB);
  MBB->addSuccessor(NewBB);
  BuildMI(MBB, DebugLoc(), TII->get(EZH::GOTO)).addMBB(NewBB);

  MF->RenumberBlocks(NewBB);
  EZHBasicBlockInfo NewBBI;
  BBUtils->insert(NewBB->getNumber(), NewBBI);
  BBUtils->computeBlockSize(MBB);
  BBUtils->computeBlockSize(NewBB);
  BBUtils->adjustBBOffsetsAfter(MBB);

  return NewBB;
}

MachineBasicBlock *
EZHConstantIslandPass::findAvailableWater(CPUser &U, unsigned UserOffset) {
  if (WaterList.empty())
    return nullptr;

  for (MachineBasicBlock *WaterBB : llvm::reverse(WaterList)) {

    if (WaterBB->getNumber() >= U.HighWaterMark->getNumber()) {
      continue; // Maintain monotonic movement!
    }

    unsigned WaterOffset =
        BBUtils->getBBInfo()[WaterBB->getNumber()].postOffset(Align(4));
    unsigned Disp = (UserOffset > WaterOffset) ? (UserOffset - WaterOffset)
                                               : (WaterOffset - UserOffset);

    if (Disp <= U.MaxDisp)
      return WaterBB; // Found the LATEST valid water (since we are searching
                      // backwards!)
  }

  return nullptr;
}

void EZHConstantIslandPass::initializeFunctionInfo() {
  WaterList.clear();
  CPUsers.clear();
  CPEntries.clear();
  ImmBranches.clear();
  JumpTableEntryIndices.clear();
  JumpTableUserIndices.clear();

  const MachineConstantPool *MCP = MF->getConstantPool();
  const std::vector<MachineConstantPoolEntry> &Constants = MCP->getConstants();

  CPEntries.resize(Constants.size());

  // Initial placement at the end of the function
  MachineBasicBlock *EndBB = MF->CreateMachineBasicBlock();
  MF->push_back(EndBB);
  EndBB->setMachineBlockAddressTaken();
  EndBB->setLabelMustBeEmitted();

  const DataLayout &TD = MF->getDataLayout();

  // Find max alignment and bucket sort constants by descending alignment (ARM
  // style!)
  Align MaxAlign(1);
  for (const auto &Constant : Constants)
    MaxAlign = std::max(MaxAlign, Constant.getAlign());
  unsigned MaxLogAlign = Log2(MaxAlign);

  SmallVector<MachineBasicBlock::iterator, 8> InsPoint(MaxLogAlign + 1,
                                                       EndBB->end());

  for (auto [i, Constant] : llvm::enumerate(Constants)) {
    unsigned Size = Constant.getSizeInBytes(TD);
    Align Alignment = Constant.getAlign();

    assert(isAligned(Alignment, Size) &&
           "CP Entry not multiple of its alignment!");

    unsigned LogAlign = Log2(Alignment);
    MachineBasicBlock::iterator InsAt = InsPoint[LogAlign];

    SmallString<32> SymName;
    raw_svector_ostream(SymName)
        << ".LCPI" << MF->getFunctionNumber() << "_" << i;
    MCSymbol *LitSym = MF->getContext().getOrCreateSymbol(SymName);

    MachineInstr *CPEMI =
        BuildMI(*EndBB, InsAt, DebugLoc(), TII->get(EZH::CONSTPOOL_ENTRY))
            .addSym(LitSym)
            .addConstantPoolIndex(i)
            .addImm(Size); // Operand 2: Size!

    CPEntries[i].push_back(CPEntry(CPEMI, i));

    for (unsigned a = LogAlign + 1; a <= MaxLogAlign; ++a) {
      if (InsPoint[a] == InsAt)
        InsPoint[a] = CPEMI;
    }
  }

  // Do the same for Jump Tables
  const MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
  if (MJTI) {
    const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
    unsigned Offset = CPEntries.size();
    CPEntries.resize(Offset + JT.size());

    for (auto [i, JumpTableEntry] : llvm::enumerate(JT)) {
      SmallString<32> SymName;
      raw_svector_ostream(SymName)
          << ".LJTI" << MF->getFunctionNumber() << "_" << i;
      MCSymbol *LitSym = MF->getContext().getOrCreateSymbol(SymName);

      unsigned JTSize = JumpTableEntry.MBBs.size() * 4;
      MachineInstr *CPEMI = BuildMI(*EndBB, EndBB->end(), DebugLoc(),
                                    TII->get(EZH::CONSTPOOL_ENTRY))
                                .addSym(LitSym)
                                .addJumpTableIndex(i)
                                .addImm(JTSize); // Operand 2: Size!

      CPEntries[Offset + i].push_back(CPEntry(CPEMI, i));
      JumpTableEntryIndices[i] = Offset + i;

      // Force label emission for all targets in this jump table!
      const std::vector<MachineBasicBlock *> &MBBs = JumpTableEntry.MBBs;
      for (MachineBasicBlock *MBB : MBBs)
        MBB->setLabelMustBeEmitted();
    }
  }

  for (MachineBasicBlock &MBB : *MF) {
    for (MachineInstr &MI : MBB) {
      unsigned Opc = MI.getOpcode();

      // Scan for branches
      if (MI.isBranch()) {
        bool isCond = false;
        unsigned MaxOffs = 8 * 1024 * 1024; // 8MB segment limit
        unsigned UncondBr = 0;

        switch (Opc) {
        case EZH::GOTO:
          break;
        case EZH::GOTO_ca:
        case EZH::GOTO_nz:
        case EZH::GOTO_ze:
          isCond = true;
          UncondBr = EZH::GOTO;
          break;
        default:
          continue;
        }

        ImmBranches.push_back(ImmBranch(&MI, MaxOffs, isCond, UncondBr));
      }

      // Scan for loads
      if (Opc == EZH::LOAD_CONSTANT) {
        MachineOperand &MO = MI.getOperand(1);
        unsigned MaxDisp = 500; // E_LDR limit with margin for signed 8-bit word
                                // offset (hardware limit is 508 bytes)

        if (MO.isCPI()) {
          unsigned CPI = MO.getIndex();
          MachineInstr *CPEMI = CPEntries[CPI][0].CPEMI;
          CPUsers.push_back(CPUser(&MI, CPEMI, MaxDisp, true, false));
          ++CPEntries[CPI][0].RefCount;
        } else if (MO.isJTI()) {
          unsigned JTI = MO.getIndex();
          unsigned CPI = JumpTableEntryIndices[JTI];
          MachineInstr *CPEMI = CPEntries[CPI][0].CPEMI;
          CPUsers.push_back(CPUser(&MI, CPEMI, MaxDisp, true, false));
          ++CPEntries[CPI][0].RefCount;
          JumpTableUserIndices[JTI] = CPUsers.size() - 1;
        }
      }
    }
  }
}

bool EZHConstantIslandPass::fixupImmediateBr(ImmBranch &Br) {
  if (!Br.MI)
    return false;
  MachineInstr *MI = Br.MI;
  MachineBasicBlock *DestBB = MI->getOperand(0).getMBB();

  unsigned BrOffset = BBUtils->getOffsetOf(MI);
  if (DestBB->getNumber() < 0 ||
      DestBB->getNumber() >= static_cast<int>(BBUtils->getBBInfo().size())) {
    errs() << "CRASH PREVENTED: DestBB number " << DestBB->getNumber()
           << " is out of bounds for BBInfo of size "
           << BBUtils->getBBInfo().size() << "\n";
    errs() << "DestBB: " << *DestBB << "\n";
    errs() << "Function: " << MF->getName() << "\n";

    errs() << "Referencing branches:\n";
    for (auto &MBB : *MF) {
      for (auto &MI : MBB) {
        if (MI.isBranch()) {
          for (auto &MO : MI.operands()) {
            if (MO.isMBB() && MO.getMBB() == DestBB) {
              errs() << "  In block #" << MBB.getNumber() << " "
                     << MBB.getName() << ": " << MI << "\n";
            }
          }
        }
      }
    }

    errs() << "Blocks in function:\n";
    for (auto &MBB : *MF) {
      errs() << "  Block #" << MBB.getNumber() << " " << MBB.getName() << "\n";
    }
    report_fatal_error("prevented crash due to out-of-bounds block number");
  }
  unsigned TargetOffset = BBUtils->getOffsetOf(DestBB);

  // Check if top 9 bits are different (assuming 23 bits of byte range).
  if ((BrOffset >> 23) == (TargetOffset >> 23))
    return false; // Same segment!

  // Out of range! We need to fix it up!

  // Put DestBB in constant pool
  EZHConstantPoolValue *CPV = new EZHConstantPoolValue(
      DestBB, Type::getInt32Ty(MF->getFunction().getContext()));
  unsigned CPI = MF->getConstantPool()->getConstantPoolIndex(CPV, Align(4));

  // Load address directly into PC!
  BuildMI(*MI->getParent(), MI, MI->getDebugLoc(), TII->get(EZH::LOAD_CONSTANT),
          EZH::PC)
      .addImm(CPI);

  // Remove the original branch instruction
  MI->eraseFromParent();
  Br.MI = nullptr;

  return true;
}

void EZHConstantIslandPass::updateForInsertedWaterBlock(
    MachineBasicBlock *NewBB) {
  NewBB->getParent()->RenumberBlocks(NewBB);
  BBUtils->insert(NewBB->getNumber(), EZHBasicBlockInfo());

  auto CompareMBBNumbers = [](const MachineBasicBlock *LHS,
                              const MachineBasicBlock *RHS) {
    return LHS->getNumber() < RHS->getNumber();
  };
  auto IP = lower_bound(WaterList, NewBB, CompareMBBNumbers);
  WaterList.insert(IP, NewBB);
}

bool EZHConstantIslandPass::decrementCPEReferenceCount(unsigned CPI,
                                                       MachineInstr *CPEMI) {
  DEBUG_PRINT(
      "        decrementCPEReferenceCount: CPI=" << CPI << " CPEMI=" << *CPEMI);
  for (auto &Entry : CPEntries[CPI]) {
    if (Entry.CPEMI == CPEMI) {
      DEBUG_PRINT("          Found entry. Old RefCount=" << (Entry.RefCount + 1)
                                                         << "\n");
      if (--Entry.RefCount == 0) {
        DEBUG_PRINT("          RefCount became 0. Erasing instruction...\n");
        MachineBasicBlock *Parent = CPEMI->getParent();
        CPEMI->eraseFromParent();
        Entry.CPEMI = nullptr;
        BBUtils->computeBlockSize(Parent);
        BBUtils->adjustBBOffsetsAfter(Parent);
        DEBUG_PRINT("          Instruction erased and sizes recalculated.\n");
        return true;
      }
      return false;
    }
  }
  llvm_unreachable("CPEntry not found!");
}

bool EZHConstantIslandPass::handleConstantPoolUser(unsigned CPUserIndex,
                                                   bool CloserWater) {
  CPUser &User = CPUsers[CPUserIndex];
  unsigned UserOffset = BBUtils->getOffsetOf(User.MI);
  unsigned CPEOffset = BBUtils->getOffsetOf(User.CPEMI);

  unsigned Disp = (UserOffset > CPEOffset) ? (UserOffset - CPEOffset)
                                           : (CPEOffset - UserOffset);

  DEBUG_PRINT("      handleConstantPoolUser for "
              << MF->getName() << ": User " << CPUserIndex
              << " (Opc=" << User.MI->getOpcode() << ")\n");
  DEBUG_PRINT("        UserOffset=" << UserOffset << " CPEOffset=" << CPEOffset
                                    << " Disp=" << Disp
                                    << " MaxDisp=" << User.MaxDisp << "\n");

  if (Disp <= User.MaxDisp) {
    DEBUG_PRINT("        -> In range!\n");
    return false; // In range!
  }
  DEBUG_PRINT("        -> Out of range!\n");

  // Search for an existing clone
  MachineInstr *Clone = nullptr;
  unsigned OrigCPI = User.CPEMI->getOperand(1).getIndex();
  if (User.CPEMI->getOperand(1).isJTI()) {
    OrigCPI = JumpTableEntryIndices[OrigCPI];
  }

  for (CPEntry &Entry : CPEntries[OrigCPI]) {
    if (!Entry.CPEMI)
      continue;
    unsigned CloneOffset = BBUtils->getOffsetOf(Entry.CPEMI);
    unsigned CloneDisp = (UserOffset > CloneOffset)
                             ? (UserOffset - CloneOffset)
                             : (CloneOffset - UserOffset);
    if (CloneDisp <= User.MaxDisp) {
      Clone = Entry.CPEMI;
      ++Entry.RefCount;
      break;
    }
  }

  if (Clone) {
    DEBUG_PRINT("      Found clone: " << *Clone << "\n");
    DEBUG_PRINT("      Decrementing old CPE refcount...\n");
    decrementCPEReferenceCount(OrigCPI, User.CPEMI);
    User.CPEMI = Clone;
    MCSymbol *CloneSym = Clone->getOperand(0).getMCSymbol();
    User.MI->getOperand(1).ChangeToMCSymbol(CloneSym);
    DEBUG_PRINT("      handleConstantPoolUser finished (reused clone)\n");
    return true;
  }

  // Search for water
  MachineBasicBlock *WaterBB = findAvailableWater(User, UserOffset);

  MachineBasicBlock *NewIsland = MF->CreateMachineBasicBlock();
  MachineBasicBlock *NewMBB = nullptr;

  if (WaterBB) {
    auto IP = std::find(WaterList.begin(), WaterList.end(), WaterBB);
    if (IP != WaterList.end())
      WaterList.erase(IP);
    NewMBB = &*++WaterBB->getIterator();
  } else {
    DEBUG_PRINT("      No water found. Creating new water...\n");
    NewMBB = createNewWater(*User.MI);
    DEBUG_PRINT("      New water created. NewMBB=" << NewMBB->getNumber()
                                                   << "\n");
  }
  LLVM_DEBUG(dbgs() << "  Created new island. WaterBB="
                    << (WaterBB ? WaterBB->getNumber() : -1) << "\n");

  // Always align the new block because CP entries can be smaller than 4 bytes.
  NewIsland->setAlignment(Align(4));

  // Insert NewIsland before NewMBB
  MF->insert(NewMBB->getIterator(), NewIsland);
  updateForInsertedWaterBlock(NewIsland);
  unsigned Size = User.CPEMI->getOperand(2).getImm();
  BBUtils->adjustBBSize(NewIsland, Size); // Use the actual constant size!
  BBUtils->adjustBBOffsetsAfter(&*--NewIsland->getIterator());

  unsigned CloneIdx = CPEntries[OrigCPI].size();
  SmallString<32> SymName;
  raw_svector_ostream(SymName) << ".LCPI" << MF->getFunctionNumber() << "_"
                               << OrigCPI << "_" << CloneIdx;
  MCSymbol *NewLitSym = MF->getContext().getOrCreateSymbol(SymName);

  // Put CONSTPOOL_ENTRY in NewIsland
  MachineInstr *NewCPEMI =
      BuildMI(NewIsland, DebugLoc(), TII->get(EZH::CONSTPOOL_ENTRY))
          .addSym(NewLitSym)
          .add(User.CPEMI->getOperand(1))  // CPI
          .add(User.CPEMI->getOperand(2)); // Size! (Clone size operand)

  CPEntries[OrigCPI].push_back(CPEntry(NewCPEMI, OrigCPI, 1));

  DEBUG_PRINT(
      "      Relocating to new island. Decrementing old CPE refcount...\n");
  decrementCPEReferenceCount(OrigCPI, User.CPEMI);
  User.CPEMI = NewCPEMI;
  User.HighWaterMark = NewIsland;

  User.MI->getOperand(1).ChangeToMCSymbol(NewLitSym);
  DEBUG_PRINT("      handleConstantPoolUser finished (created new island)\n");

  return true;
}

bool EZHConstantIslandPass::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  const EZHSubtarget &STI = MF->getSubtarget<EZHSubtarget>();
  TII = STI.getInstrInfo();
  MF->RenumberBlocks();
  bool Changed = false;

  // 1. Expand PseudoBR_JT instructions late!
  SmallVector<MachineInstr *, 4> Pseudos;
  for (auto &MBB : *MF) {
    for (auto &MI : MBB) {
      if (MI.getOpcode() == EZH::PseudoBR_JT) {
        Pseudos.push_back(&MI);
      }
    }
  }

  for (MachineInstr *MI : Pseudos) {
    MachineBasicBlock *MBB = MI->getParent();
    DebugLoc DL = MI->getDebugLoc();
    Register TableReg = MI->getOperand(0).getReg();
    Register IndexReg = MI->getOperand(1).getReg();

    BuildMI(*MBB, MI, DL, TII->get(EZH::LSL_ADD__), EZH::RA)
        .addReg(TableReg)
        .addReg(IndexReg)
        .addImm(2);

    BuildMI(*MBB, MI, DL, TII->get(EZH::LDR), EZH::PC)
        .addReg(EZH::RA)
        .addImm(0);

    MI->eraseFromParent();
    Changed = true;
  }

  initializeFunctionInfo();

  // Renumber blocks to ensure EndBB (just created) gets a consecutive ID!
  MF->RenumberBlocks();

  // Construct BBUtils now that EndBB and jump tables are created and
  // renumbered!
  BBUtils = std::make_unique<EZHBasicBlockUtils>(*MF);
  BBUtils->computeAllBlockSizes();
  BBUtils->adjustBBOffsetsAfter(&MF->front());
  initializeWaterList();

  for (auto &MBB : *MF) {
    for (auto &MI : MBB) {
      if (MI.isBranch()) {
        for (auto &MO : MI.operands()) {
          if (MO.isMBB()) {
            MO.getMBB()->setLabelMustBeEmitted();
          }
        }
      }
    }
  }

  const MachineConstantPool *MCP = MF->getConstantPool();
  const std::vector<MachineConstantPoolEntry> &Constants = MCP->getConstants();
  for (const MachineConstantPoolEntry &CPE : Constants) {
    if (!CPE.isMachineConstantPoolEntry()) {
      const Constant *CV = CPE.Val.ConstVal;
      SmallVector<const BlockAddress *, 4> BAs;
      if (const BlockAddress *BA = dyn_cast<BlockAddress>(CV)) {
        BAs.push_back(BA);
      } else if (const ConstantArray *CA = dyn_cast<ConstantArray>(CV)) {
        for (const Value *Op : CA->operand_values())
          if (const BlockAddress *BA = dyn_cast<BlockAddress>(Op))
            BAs.push_back(BA);
      }

      for (const BlockAddress *BA : BAs) {
        for (auto &MBB : *MF) {
          if (MBB.getBasicBlock() == BA->getBasicBlock()) {
            MBB.setLabelMustBeEmitted();
            break;
          }
        }
      }
    }
  }

  if (CPUsers.empty())
    return false;

  // 3. Nested Iterative Loop
  bool OverallChange = true;
  unsigned OverallIters = 0;
  while (OverallChange) {
    OverallChange = false;

    // Run Constant Pool pass until convergence
    bool CPChange = true;
    unsigned NoCPIters = 0;
    while (CPChange) {
      CPChange = false;
      BBUtils->computeAllBlockSizes();
      BBUtils->adjustBBOffsetsAfter(&MF->front());
      initializeWaterList();
      for (auto User : llvm::enumerate(CPUsers))
        CPChange |= handleConstantPoolUser(User.index(), NoCPIters >= 5);

      if (CPChange && ++NoCPIters > 100)
        report_fatal_error("constant island pass failed to converge (CP loop)");
    }

    // Run Branch Fixup pass
    bool BRChange = false;
    BBUtils->computeAllBlockSizes();
    BBUtils->adjustBBOffsetsAfter(&MF->front());
    for (ImmBranch &Br : ImmBranches)
      BRChange |= fixupImmediateBr(Br);

    OverallChange = BRChange;
    Changed |= BRChange;

    if (OverallChange && ++OverallIters > 10)
      report_fatal_error(
          "constant island pass failed to converge (overall loop)");
  }

  // Update operands of LOAD instructions to reference the unique symbol of the
  // selected CPEMI
  for (CPUser &User : CPUsers) {
    MCSymbol *LitSym = User.CPEMI->getOperand(0).getMCSymbol();
    User.MI->getOperand(1) = MachineOperand::CreateMCSymbol(LitSym);
  }

  // Clean up redundant branches created by splitting
  for (auto &MBB : *MF) {
    MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();
    if (I != MBB.end() && I->getOpcode() == EZH::GOTO) {
      MachineBasicBlock *TargetMBB = I->getOperand(0).getMBB();
      if (MBB.isLayoutSuccessor(TargetMBB)) {
        // Erase preceding bitslice workaround if present, skipping debug insts
        MachineBasicBlock::iterator Prev = I;
        while (Prev != MBB.begin()) {
          --Prev;
          if (!Prev->isDebugInstr()) {
            if (Prev->getOpcode() == EZH::GOTOL_bs) {
              Prev->eraseFromParent();
            }
            break;
          }
        }
        I->eraseFromParent();
        Changed = true;
      }
    }
  }

  return Changed;
}

namespace llvm {
FunctionPass *createEZHConstantIslandPass() {
  return new EZHConstantIslandPass();
}
} // namespace llvm
