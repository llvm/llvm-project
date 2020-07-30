//===-- FixupStatepointCallerSaved.cpp - Fixup caller saved registers  ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Statepoint instruction in deopt parameters contains values which are
/// meaningful to the runtime and should be able to be read at the moment the
/// call returns. So we can say that we need to encode the fact that these
/// values are "late read" by runtime. If we could express this notion for
/// register allocator it would produce the right form for us.
/// The need to fixup (i.e this pass) is specifically handling the fact that
/// we cannot describe such a late read for the register allocator.
/// Register allocator may put the value on a register clobbered by the call.
/// This pass forces the spill of such registers and replaces corresponding
/// statepoint operands to added spill slots.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/Statepoint.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "fixup-statepoint-caller-saved"
STATISTIC(NumSpilledRegisters, "Number of spilled register");
STATISTIC(NumSpillSlotsAllocated, "Number of spill slots allocated");
STATISTIC(NumSpillSlotsExtended, "Number of spill slots extended");

static cl::opt<bool> FixupSCSExtendSlotSize(
    "fixup-scs-extend-slot-size", cl::Hidden, cl::init(false),
    cl::desc("Allow spill in spill slot of greater size than register size"),
    cl::Hidden);

namespace {

class FixupStatepointCallerSaved : public MachineFunctionPass {
public:
  static char ID;

  FixupStatepointCallerSaved() : MachineFunctionPass(ID) {
    initializeFixupStatepointCallerSavedPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return "Fixup Statepoint Caller Saved";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};
} // End anonymous namespace.

char FixupStatepointCallerSaved::ID = 0;
char &llvm::FixupStatepointCallerSavedID = FixupStatepointCallerSaved::ID;

INITIALIZE_PASS_BEGIN(FixupStatepointCallerSaved, DEBUG_TYPE,
                      "Fixup Statepoint Caller Saved", false, false)
INITIALIZE_PASS_END(FixupStatepointCallerSaved, DEBUG_TYPE,
                    "Fixup Statepoint Caller Saved", false, false)

// Utility function to get size of the register.
static unsigned getRegisterSize(const TargetRegisterInfo &TRI, Register Reg) {
  const TargetRegisterClass *RC = TRI.getMinimalPhysRegClass(Reg);
  return TRI.getSpillSize(*RC);
}

namespace {
// Cache used frame indexes during statepoint re-write to re-use them in
// processing next statepoint instruction.
// Two strategies. One is to preserve the size of spill slot while another one
// extends the size of spill slots to reduce the number of them, causing
// the less total frame size. But unspill will have "implicit" any extend.
class FrameIndexesCache {
private:
  struct FrameIndexesPerSize {
    // List of used frame indexes during processing previous statepoints.
    SmallVector<int, 8> Slots;
    // Current index of un-used yet frame index.
    unsigned Index = 0;
  };
  MachineFrameInfo &MFI;
  const TargetRegisterInfo &TRI;
  // Map size to list of frame indexes of this size. If the mode is
  // FixupSCSExtendSlotSize then the key 0 is used to keep all frame indexes.
  // If the size of required spill slot is greater than in a cache then the
  // size will be increased.
  DenseMap<unsigned, FrameIndexesPerSize> Cache;

public:
  FrameIndexesCache(MachineFrameInfo &MFI, const TargetRegisterInfo &TRI)
      : MFI(MFI), TRI(TRI) {}
  // Reset the current state of used frame indexes. After invocation of
  // this function all frame indexes are available for allocation.
  void reset() {
    for (auto &It : Cache)
      It.second.Index = 0;
  }
  // Get frame index to spill the register.
  int getFrameIndex(Register Reg) {
    unsigned Size = getRegisterSize(TRI, Reg);
    // In FixupSCSExtendSlotSize mode the bucket with 0 index is used
    // for all sizes.
    unsigned Bucket = FixupSCSExtendSlotSize ? 0 : Size;
    FrameIndexesPerSize &Line = Cache[Bucket];
    if (Line.Index < Line.Slots.size()) {
      int FI = Line.Slots[Line.Index++];
      // If all sizes are kept together we probably need to extend the
      // spill slot size.
      if (MFI.getObjectSize(FI) < Size) {
        MFI.setObjectSize(FI, Size);
        MFI.setObjectAlignment(FI, Align(Size));
        NumSpillSlotsExtended++;
      }
      return FI;
    }
    int FI = MFI.CreateSpillStackObject(Size, Align(Size));
    NumSpillSlotsAllocated++;
    Line.Slots.push_back(FI);
    ++Line.Index;
    return FI;
  }
  // Sort all registers to spill in descendent order. In the
  // FixupSCSExtendSlotSize mode it will minimize the total frame size.
  // In non FixupSCSExtendSlotSize mode we can skip this step.
  void sortRegisters(SmallVectorImpl<Register> &Regs) {
    if (!FixupSCSExtendSlotSize)
      return;
    llvm::sort(Regs.begin(), Regs.end(), [&](Register &A, Register &B) {
      return getRegisterSize(TRI, A) > getRegisterSize(TRI, B);
    });
  }
};

// Describes the state of the current processing statepoint instruction.
class StatepointState {
private:
  // statepoint instruction.
  MachineInstr &MI;
  MachineFunction &MF;
  const TargetRegisterInfo &TRI;
  const TargetInstrInfo &TII;
  MachineFrameInfo &MFI;
  // Mask with callee saved registers.
  const uint32_t *Mask;
  // Cache of frame indexes used on previous instruction processing.
  FrameIndexesCache &CacheFI;
  // Operands with physical registers requiring spilling.
  SmallVector<unsigned, 8> OpsToSpill;
  // Set of register to spill.
  SmallVector<Register, 8> RegsToSpill;
  // Map Register to Frame Slot index.
  DenseMap<Register, int> RegToSlotIdx;

public:
  StatepointState(MachineInstr &MI, const uint32_t *Mask,
                  FrameIndexesCache &CacheFI)
      : MI(MI), MF(*MI.getMF()), TRI(*MF.getSubtarget().getRegisterInfo()),
        TII(*MF.getSubtarget().getInstrInfo()), MFI(MF.getFrameInfo()),
        Mask(Mask), CacheFI(CacheFI) {}
  // Return true if register is callee saved.
  bool isCalleeSaved(Register Reg) { return (Mask[Reg / 32] >> Reg % 32) & 1; }
  // Iterates over statepoint meta args to find caller saver registers.
  // Also cache the size of found registers.
  // Returns true if caller save registers found.
  bool findRegistersToSpill() {
    SmallSet<Register, 8> VisitedRegs;
    for (unsigned Idx = StatepointOpers(&MI).getVarIdx(),
                  EndIdx = MI.getNumOperands();
         Idx < EndIdx; ++Idx) {
      MachineOperand &MO = MI.getOperand(Idx);
      if (!MO.isReg() || MO.isImplicit())
        continue;
      Register Reg = MO.getReg();
      assert(Reg.isPhysical() && "Only physical regs are expected");
      if (isCalleeSaved(Reg))
        continue;
      if (VisitedRegs.insert(Reg).second)
        RegsToSpill.push_back(Reg);
      OpsToSpill.push_back(Idx);
    }
    CacheFI.sortRegisters(RegsToSpill);
    return !RegsToSpill.empty();
  }
  // Spill all caller saved registers right before statepoint instruction.
  // Remember frame index where register is spilled.
  void spillRegisters() {
    for (Register Reg : RegsToSpill) {
      int FI = CacheFI.getFrameIndex(Reg);
      const TargetRegisterClass *RC = TRI.getMinimalPhysRegClass(Reg);
      TII.storeRegToStackSlot(*MI.getParent(), MI, Reg, true /*is_Kill*/, FI,
                              RC, &TRI);
      NumSpilledRegisters++;
      RegToSlotIdx[Reg] = FI;
    }
  }
  // Re-write statepoint machine instruction to replace caller saved operands
  // with indirect memory location (frame index).
  void rewriteStatepoint() {
    MachineInstr *NewMI =
        MF.CreateMachineInstr(TII.get(MI.getOpcode()), MI.getDebugLoc(), true);
    MachineInstrBuilder MIB(MF, NewMI);

    // Add End marker.
    OpsToSpill.push_back(MI.getNumOperands());
    unsigned CurOpIdx = 0;

    for (unsigned I = 0; I < MI.getNumOperands(); ++I) {
      MachineOperand &MO = MI.getOperand(I);
      if (I == OpsToSpill[CurOpIdx]) {
        int FI = RegToSlotIdx[MO.getReg()];
        MIB.addImm(StackMaps::IndirectMemRefOp);
        MIB.addImm(getRegisterSize(TRI, MO.getReg()));
        assert(MO.isReg() && "Should be register");
        assert(MO.getReg().isPhysical() && "Should be physical register");
        MIB.addFrameIndex(FI);
        MIB.addImm(0);
        ++CurOpIdx;
      } else
        MIB.add(MO);
    }
    assert(CurOpIdx == (OpsToSpill.size() - 1) && "Not all operands processed");
    // Add mem operands.
    NewMI->setMemRefs(MF, MI.memoperands());
    for (auto It : RegToSlotIdx) {
      int FrameIndex = It.second;
      auto PtrInfo = MachinePointerInfo::getFixedStack(MF, FrameIndex);
      auto *MMO = MF.getMachineMemOperand(PtrInfo, MachineMemOperand::MOLoad,
                                          getRegisterSize(TRI, It.first),
                                          MFI.getObjectAlign(FrameIndex));
      NewMI->addMemOperand(MF, MMO);
    }
    // Insert new statepoint and erase old one.
    MI.getParent()->insert(MI, NewMI);
    MI.eraseFromParent();
  }
};

class StatepointProcessor {
private:
  MachineFunction &MF;
  const TargetRegisterInfo &TRI;
  FrameIndexesCache CacheFI;

public:
  StatepointProcessor(MachineFunction &MF)
      : MF(MF), TRI(*MF.getSubtarget().getRegisterInfo()),
        CacheFI(MF.getFrameInfo(), TRI) {}

  bool process(MachineInstr &MI) {
    StatepointOpers SO(&MI);
    uint64_t Flags = SO.getFlags();
    // Do nothing for LiveIn, it supports all registers.
    if (Flags & (uint64_t)StatepointFlags::DeoptLiveIn)
      return false;
    CallingConv::ID CC = SO.getCallingConv();
    const uint32_t *Mask = TRI.getCallPreservedMask(MF, CC);
    CacheFI.reset();
    StatepointState SS(MI, Mask, CacheFI);

    if (!SS.findRegistersToSpill())
      return false;

    SS.spillRegisters();
    SS.rewriteStatepoint();
    return true;
  }
};
} // namespace

bool FixupStatepointCallerSaved::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  const Function &F = MF.getFunction();
  if (!F.hasGC())
    return false;

  SmallVector<MachineInstr *, 16> Statepoints;
  for (MachineBasicBlock &BB : MF)
    for (MachineInstr &I : BB)
      if (I.getOpcode() == TargetOpcode::STATEPOINT)
        Statepoints.push_back(&I);

  if (Statepoints.empty())
    return false;

  bool Changed = false;
  StatepointProcessor SPP(MF);
  for (MachineInstr *I : Statepoints)
    Changed |= SPP.process(*I);
  return Changed;
}
