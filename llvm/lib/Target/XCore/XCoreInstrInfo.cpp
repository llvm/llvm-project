//===- XCoreInstrInfo.cpp - XCore Instruction Information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the XCore implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "XCoreMachineFunctionInfo.h"
#include "XCoreInstrInfo.h"
#include "XCore.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "XCoreGenInstrInfo.inc"
#include "llvm/Support/Debug.h"

namespace llvm {
namespace XCore {

  // XCore Condition Codes
  enum CondCode {
    COND_TRUE,
    COND_FALSE,
    COND_INVALID
  };
}
}

using namespace llvm;

XCoreInstrInfo::XCoreInstrInfo(void)
  : TargetInstrInfoImpl(XCoreInsts, array_lengthof(XCoreInsts)),
    RI(*this) {
}

static bool isZeroImm(const MachineOperand &op) {
  return op.isImm() && op.getImm() == 0;
}

/// Return true if the instruction is a register to register move and
/// leave the source and dest operands in the passed parameters.
///
bool XCoreInstrInfo::isMoveInstr(const MachineInstr &MI,
                                 unsigned &SrcReg, unsigned &DstReg,
                                 unsigned &SrcSR, unsigned &DstSR) const {
  SrcSR = DstSR = 0; // No sub-registers.

  // We look for 4 kinds of patterns here:
  // add dst, src, 0
  // sub dst, src, 0
  // or dst, src, src
  // and dst, src, src
  if ((MI.getOpcode() == XCore::ADD_2rus || MI.getOpcode() == XCore::SUB_2rus)
      && isZeroImm(MI.getOperand(2))) {
    DstReg = MI.getOperand(0).getReg();
    SrcReg = MI.getOperand(1).getReg();
    return true;
  } else if ((MI.getOpcode() == XCore::OR_3r || MI.getOpcode() == XCore::AND_3r)
      && MI.getOperand(1).getReg() == MI.getOperand(2).getReg()) {
    DstReg = MI.getOperand(0).getReg();
    SrcReg = MI.getOperand(1).getReg();
    return true;
  }
  return false;
}

/// isLoadFromStackSlot - If the specified machine instruction is a direct
/// load from a stack slot, return the virtual or physical register number of
/// the destination along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than loading from the stack slot.
unsigned
XCoreInstrInfo::isLoadFromStackSlot(const MachineInstr *MI, int &FrameIndex) const{
  int Opcode = MI->getOpcode();
  if (Opcode == XCore::LDWFI) 
  {
    if ((MI->getOperand(1).isFI()) && // is a stack slot
        (MI->getOperand(2).isImm()) &&  // the imm is zero
        (isZeroImm(MI->getOperand(2)))) 
    {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
  }
  return 0;
}
  
  /// isStoreToStackSlot - If the specified machine instruction is a direct
  /// store to a stack slot, return the virtual or physical register number of
  /// the source reg along with the FrameIndex of the loaded stack slot.  If
  /// not, return 0.  This predicate must return 0 if the instruction has
  /// any side effects other than storing to the stack slot.
unsigned
XCoreInstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                   int &FrameIndex) const {
  int Opcode = MI->getOpcode();
  if (Opcode == XCore::STWFI)
  {
    if ((MI->getOperand(1).isFI()) && // is a stack slot
        (MI->getOperand(2).isImm()) &&  // the imm is zero
        (isZeroImm(MI->getOperand(2))))
    {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
  }
  return 0;
}

/// isInvariantLoad - Return true if the specified instruction (which is marked
/// mayLoad) is loading from a location whose value is invariant across the
/// function.  For example, loading a value from the constant pool or from
/// from the argument area of a function if it does not change.  This should
/// only return true of *all* loads the instruction does are invariant (if it
/// does multiple loads).
bool
XCoreInstrInfo::isInvariantLoad(const MachineInstr *MI) const {
  // Loads from constants pools and loads from invariant argument slots are
  // invariant
  int Opcode = MI->getOpcode();
  if (Opcode == XCore::LDWCP_ru6 || Opcode == XCore::LDWCP_lru6) {
    return MI->getOperand(1).isCPI();
  }
  int FrameIndex;
  if (isLoadFromStackSlot(MI, FrameIndex)) {
    const MachineFrameInfo &MFI =
      *MI->getParent()->getParent()->getFrameInfo();
    return MFI.isFixedObjectIndex(FrameIndex) &&
           MFI.isImmutableObjectIndex(FrameIndex);
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Branch Analysis
//===----------------------------------------------------------------------===//

static inline bool IsBRU(unsigned BrOpc) {
  return BrOpc == XCore::BRFU_u6
      || BrOpc == XCore::BRFU_lu6
      || BrOpc == XCore::BRBU_u6
      || BrOpc == XCore::BRBU_lu6;
}

static inline bool IsBRT(unsigned BrOpc) {
  return BrOpc == XCore::BRFT_ru6
      || BrOpc == XCore::BRFT_lru6
      || BrOpc == XCore::BRBT_ru6
      || BrOpc == XCore::BRBT_lru6;
}

static inline bool IsBRF(unsigned BrOpc) {
  return BrOpc == XCore::BRFF_ru6
      || BrOpc == XCore::BRFF_lru6
      || BrOpc == XCore::BRBF_ru6
      || BrOpc == XCore::BRBF_lru6;
}

static inline bool IsCondBranch(unsigned BrOpc) {
  return IsBRF(BrOpc) || IsBRT(BrOpc);
}

/// GetCondFromBranchOpc - Return the XCore CC that matches 
/// the correspondent Branch instruction opcode.
static XCore::CondCode GetCondFromBranchOpc(unsigned BrOpc) 
{
  if (IsBRT(BrOpc)) {
    return XCore::COND_TRUE;
  } else if (IsBRF(BrOpc)) {
    return XCore::COND_FALSE;
  } else {
    return XCore::COND_INVALID;
  }
}

/// GetCondBranchFromCond - Return the Branch instruction
/// opcode that matches the cc.
static inline unsigned GetCondBranchFromCond(XCore::CondCode CC) 
{
  switch (CC) {
  default: assert(0 && "Illegal condition code!");
  case XCore::COND_TRUE   : return XCore::BRFT_lru6;
  case XCore::COND_FALSE  : return XCore::BRFF_lru6;
  }
}

/// GetOppositeBranchCondition - Return the inverse of the specified 
/// condition, e.g. turning COND_E to COND_NE.
static inline XCore::CondCode GetOppositeBranchCondition(XCore::CondCode CC)
{
  switch (CC) {
  default: assert(0 && "Illegal condition code!");
  case XCore::COND_TRUE   : return XCore::COND_FALSE;
  case XCore::COND_FALSE  : return XCore::COND_TRUE;
  }
}

/// AnalyzeBranch - Analyze the branching code at the end of MBB, returning
/// true if it cannot be understood (e.g. it's a switch dispatch or isn't
/// implemented for a target).  Upon success, this returns false and returns
/// with the following information in various cases:
///
/// 1. If this block ends with no branches (it just falls through to its succ)
///    just return false, leaving TBB/FBB null.
/// 2. If this block ends with only an unconditional branch, it sets TBB to be
///    the destination block.
/// 3. If this block ends with an conditional branch and it falls through to
///    an successor block, it sets TBB to be the branch destination block and a
///    list of operands that evaluate the condition. These
///    operands can be passed to other TargetInstrInfo methods to create new
///    branches.
/// 4. If this block ends with an conditional branch and an unconditional
///    block, it returns the 'true' destination in TBB, the 'false' destination
///    in FBB, and a list of operands that evaluate the condition. These
///    operands can be passed to other TargetInstrInfo methods to create new
///    branches.
///
/// Note that RemoveBranch and InsertBranch must be implemented to support
/// cases where this method returns success.
///
bool
XCoreInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                              MachineBasicBlock *&FBB,
                              SmallVectorImpl<MachineOperand> &Cond,
                              bool AllowModify) const {
  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I))
    return false;

  // Get the last instruction in the block.
  MachineInstr *LastInst = I;
  
  // If there is only one terminator instruction, process it.
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I)) {
    if (IsBRU(LastInst->getOpcode())) {
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    }
    
    XCore::CondCode BranchCode = GetCondFromBranchOpc(LastInst->getOpcode());
    if (BranchCode == XCore::COND_INVALID)
      return true;  // Can't handle indirect branch.
    
    // Conditional branch
    // Block ends with fall-through condbranch.

    TBB = LastInst->getOperand(1).getMBB();
    Cond.push_back(MachineOperand::CreateImm(BranchCode));
    Cond.push_back(LastInst->getOperand(0));
    return false;
  }
  
  // Get the instruction before it if it's a terminator.
  MachineInstr *SecondLastInst = I;

  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() &&
      isUnpredicatedTerminator(--I))
    return true;
  
  unsigned SecondLastOpc    = SecondLastInst->getOpcode();
  XCore::CondCode BranchCode = GetCondFromBranchOpc(SecondLastOpc);
  
  // If the block ends with conditional branch followed by unconditional,
  // handle it.
  if (BranchCode != XCore::COND_INVALID
    && IsBRU(LastInst->getOpcode())) {

    TBB = SecondLastInst->getOperand(1).getMBB();
    Cond.push_back(MachineOperand::CreateImm(BranchCode));
    Cond.push_back(SecondLastInst->getOperand(0));

    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }
  
  // If the block ends with two unconditional branches, handle it.  The second
  // one is not executed, so remove it.
  if (IsBRU(SecondLastInst->getOpcode()) && 
      IsBRU(LastInst->getOpcode())) {
    TBB = SecondLastInst->getOperand(0).getMBB();
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return false;
  }

  // Otherwise, can't handle this.
  return true;
}

unsigned
XCoreInstrInfo::InsertBranch(MachineBasicBlock &MBB,MachineBasicBlock *TBB,
                             MachineBasicBlock *FBB,
                             const SmallVectorImpl<MachineOperand> &Cond)const{
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 2 || Cond.size() == 0) &&
         "Unexpected number of components!");
  
  if (FBB == 0) { // One way branch.
    if (Cond.empty()) {
      // Unconditional branch
      BuildMI(&MBB, get(XCore::BRFU_lu6)).addMBB(TBB);
    } else {
      // Conditional branch.
      unsigned Opc = GetCondBranchFromCond((XCore::CondCode)Cond[0].getImm());
      BuildMI(&MBB, get(Opc)).addReg(Cond[1].getReg())
                             .addMBB(TBB);
    }
    return 1;
  }
  
  // Two-way Conditional branch.
  assert(Cond.size() == 2 && "Unexpected number of components!");
  unsigned Opc = GetCondBranchFromCond((XCore::CondCode)Cond[0].getImm());
  BuildMI(&MBB, get(Opc)).addReg(Cond[1].getReg())
                         .addMBB(TBB);
  BuildMI(&MBB, get(XCore::BRFU_lu6)).addMBB(FBB);
  return 2;
}

unsigned
XCoreInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin()) return 0;
  --I;
  if (!IsBRU(I->getOpcode()) && !IsCondBranch(I->getOpcode()))
    return 0;
  
  // Remove the branch.
  I->eraseFromParent();
  
  I = MBB.end();

  if (I == MBB.begin()) return 1;
  --I;
  if (!IsCondBranch(I->getOpcode()))
    return 1;
  
  // Remove the branch.
  I->eraseFromParent();
  return 2;
}

bool XCoreInstrInfo::copyRegToReg(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I,
                                     unsigned DestReg, unsigned SrcReg,
                                     const TargetRegisterClass *DestRC,
                                     const TargetRegisterClass *SrcRC) const {
  if (DestRC == SrcRC) {
    if (DestRC == XCore::GRRegsRegisterClass) {
      BuildMI(MBB, I, get(XCore::ADD_2rus), DestReg).addReg(SrcReg).addImm(0);
      return true;
    } else {
      return false;
    }
  }
  
  if (SrcRC == XCore::RRegsRegisterClass && SrcReg == XCore::SP &&
    DestRC == XCore::GRRegsRegisterClass) {
    BuildMI(MBB, I, get(XCore::LDAWSP_ru6), DestReg).addImm(0);
    return true;
  }
  if (DestRC == XCore::RRegsRegisterClass && DestReg == XCore::SP &&
    SrcRC == XCore::GRRegsRegisterClass) {
    BuildMI(MBB, I, get(XCore::SETSP_1r)).addReg(SrcReg);
    return true;
  }
  return false;
}

void XCoreInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator I,
                                  unsigned SrcReg, bool isKill, int FrameIndex,
                                  const TargetRegisterClass *RC) const
{
  BuildMI(MBB, I, get(XCore::STWFI)).addReg(SrcReg, false, false, isKill)
                                    .addFrameIndex(FrameIndex).addImm(0);
}

void XCoreInstrInfo::storeRegToAddr(MachineFunction &MF, unsigned SrcReg,
                            bool isKill, SmallVectorImpl<MachineOperand> &Addr,
                            const TargetRegisterClass *RC,
                            SmallVectorImpl<MachineInstr*> &NewMIs) const
{
  assert(0 && "unimplemented\n");
}

void XCoreInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator I,
                                  unsigned DestReg, int FrameIndex,
                                  const TargetRegisterClass *RC) const
{
  BuildMI(MBB, I, get(XCore::LDWFI), DestReg).addFrameIndex(FrameIndex)
                                             .addImm(0);
}

void XCoreInstrInfo::loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                              SmallVectorImpl<MachineOperand> &Addr,
                              const TargetRegisterClass *RC,
                              SmallVectorImpl<MachineInstr*> &NewMIs) const
{
  assert(0 && "unimplemented\n");
}

bool XCoreInstrInfo::spillCalleeSavedRegisters(MachineBasicBlock &MBB,
				MachineBasicBlock::iterator MI,
			const std::vector<CalleeSavedInfo> &CSI) const
{
  if (CSI.empty()) {
    return true;
  }
  MachineFunction *MF = MBB.getParent();
  const MachineFrameInfo *MFI = MF->getFrameInfo();
  MachineModuleInfo *MMI = MFI->getMachineModuleInfo();
  XCoreFunctionInfo *XFI = MF->getInfo<XCoreFunctionInfo>();
  
  bool emitFrameMoves = XCoreRegisterInfo::needsFrameMoves(*MF);
  
  for (std::vector<CalleeSavedInfo>::const_iterator it = CSI.begin();
                                                    it != CSI.end(); ++it) {
    // Add the callee-saved register as live-in. It's killed at the spill.
    MBB.addLiveIn(it->getReg());

    storeRegToStackSlot(MBB, MI, it->getReg(), true,
                                   it->getFrameIdx(), it->getRegClass());
    if (emitFrameMoves) {
      unsigned SaveLabelId = MMI->NextLabelID();
      BuildMI(MBB, MI, get(XCore::DBG_LABEL)).addImm(SaveLabelId);
      XFI->getSpillLabels().push_back(
          std::pair<unsigned, CalleeSavedInfo>(SaveLabelId, *it));
    }
  }
  return true;
}

bool XCoreInstrInfo::restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MI,
                               const std::vector<CalleeSavedInfo> &CSI) const
{
  bool AtStart = MI == MBB.begin();
  MachineBasicBlock::iterator BeforeI = MI;
  if (!AtStart)
    --BeforeI;
  for (std::vector<CalleeSavedInfo>::const_iterator it = CSI.begin();
                                                    it != CSI.end(); ++it) {
    
    loadRegFromStackSlot(MBB, MI, it->getReg(),
                                  it->getFrameIdx(),
                                  it->getRegClass());
    assert(MI != MBB.begin() &&
           "loadRegFromStackSlot didn't insert any code!");
    // Insert in reverse order.  loadRegFromStackSlot can insert multiple
    // instructions.
    if (AtStart)
      MI = MBB.begin();
    else {
      MI = BeforeI;
      ++MI;
    }
  }
  return true;
}

/// BlockHasNoFallThrough - Analyse if MachineBasicBlock does not
/// fall-through into its successor block.
bool XCoreInstrInfo::
BlockHasNoFallThrough(const MachineBasicBlock &MBB) const 
{
  if (MBB.empty()) return false;
  
  switch (MBB.back().getOpcode()) {
  case XCore::RETSP_u6:     // Return.
  case XCore::RETSP_lu6:
  case XCore::BAU_1r:       // Indirect branch.
  case XCore::BRFU_u6:      // Uncond branch.
  case XCore::BRFU_lu6:
  case XCore::BRBU_u6:
  case XCore::BRBU_lu6:
    return true;
  default: return false;
  }
}

/// ReverseBranchCondition - Return the inverse opcode of the 
/// specified Branch instruction.
bool XCoreInstrInfo::
ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const 
{
  assert((Cond.size() == 2) && 
          "Invalid XCore branch condition!");
  Cond[0].setImm(GetOppositeBranchCondition((XCore::CondCode)Cond[0].getImm()));
  return false;
}
