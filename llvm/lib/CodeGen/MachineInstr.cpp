//===-- lib/CodeGen/MachineInstr.cpp --------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Methods common to all machine instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Constants.h"
#include "llvm/DebugInfo.h"
#include "llvm/Function.h"
#include "llvm/InlineAsm.h"
#include "llvm/LLVMContext.h"
#include "llvm/Metadata.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Value.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LeakDetector.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/Hashing.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// MachineOperand Implementation
//===----------------------------------------------------------------------===//

void MachineOperand::setReg(unsigned Reg) {
  if (getReg() == Reg) return; // No change.

  // Otherwise, we have to change the register.  If this operand is embedded
  // into a machine function, we need to update the old and new register's
  // use/def lists.
  if (MachineInstr *MI = getParent())
    if (MachineBasicBlock *MBB = MI->getParent())
      if (MachineFunction *MF = MBB->getParent()) {
        MachineRegisterInfo &MRI = MF->getRegInfo();
        MRI.removeRegOperandFromUseList(this);
        SmallContents.RegNo = Reg;
        MRI.addRegOperandToUseList(this);
        return;
      }

  // Otherwise, just change the register, no problem.  :)
  SmallContents.RegNo = Reg;
}

void MachineOperand::substVirtReg(unsigned Reg, unsigned SubIdx,
                                  const TargetRegisterInfo &TRI) {
  assert(TargetRegisterInfo::isVirtualRegister(Reg));
  if (SubIdx && getSubReg())
    SubIdx = TRI.composeSubRegIndices(SubIdx, getSubReg());
  setReg(Reg);
  if (SubIdx)
    setSubReg(SubIdx);
}

void MachineOperand::substPhysReg(unsigned Reg, const TargetRegisterInfo &TRI) {
  assert(TargetRegisterInfo::isPhysicalRegister(Reg));
  if (getSubReg()) {
    Reg = TRI.getSubReg(Reg, getSubReg());
    // Note that getSubReg() may return 0 if the sub-register doesn't exist.
    // That won't happen in legal code.
    setSubReg(0);
  }
  setReg(Reg);
}

/// Change a def to a use, or a use to a def.
void MachineOperand::setIsDef(bool Val) {
  assert(isReg() && "Wrong MachineOperand accessor");
  assert((!Val || !isDebug()) && "Marking a debug operation as def");
  if (IsDef == Val)
    return;
  // MRI may keep uses and defs in different list positions.
  if (MachineInstr *MI = getParent())
    if (MachineBasicBlock *MBB = MI->getParent())
      if (MachineFunction *MF = MBB->getParent()) {
        MachineRegisterInfo &MRI = MF->getRegInfo();
        MRI.removeRegOperandFromUseList(this);
        IsDef = Val;
        MRI.addRegOperandToUseList(this);
        return;
      }
  IsDef = Val;
}

/// ChangeToImmediate - Replace this operand with a new immediate operand of
/// the specified value.  If an operand is known to be an immediate already,
/// the setImm method should be used.
void MachineOperand::ChangeToImmediate(int64_t ImmVal) {
  assert((!isReg() || !isTied()) && "Cannot change a tied operand into an imm");
  // If this operand is currently a register operand, and if this is in a
  // function, deregister the operand from the register's use/def list.
  if (isReg() && isOnRegUseList())
    if (MachineInstr *MI = getParent())
      if (MachineBasicBlock *MBB = MI->getParent())
        if (MachineFunction *MF = MBB->getParent())
          MF->getRegInfo().removeRegOperandFromUseList(this);

  OpKind = MO_Immediate;
  Contents.ImmVal = ImmVal;
}

/// ChangeToRegister - Replace this operand with a new register operand of
/// the specified value.  If an operand is known to be an register already,
/// the setReg method should be used.
void MachineOperand::ChangeToRegister(unsigned Reg, bool isDef, bool isImp,
                                      bool isKill, bool isDead, bool isUndef,
                                      bool isDebug) {
  MachineRegisterInfo *RegInfo = 0;
  if (MachineInstr *MI = getParent())
    if (MachineBasicBlock *MBB = MI->getParent())
      if (MachineFunction *MF = MBB->getParent())
        RegInfo = &MF->getRegInfo();
  // If this operand is already a register operand, remove it from the
  // register's use/def lists.
  bool WasReg = isReg();
  if (RegInfo && WasReg)
    RegInfo->removeRegOperandFromUseList(this);

  // Change this to a register and set the reg#.
  OpKind = MO_Register;
  SmallContents.RegNo = Reg;
  SubReg = 0;
  IsDef = isDef;
  IsImp = isImp;
  IsKill = isKill;
  IsDead = isDead;
  IsUndef = isUndef;
  IsInternalRead = false;
  IsEarlyClobber = false;
  IsDebug = isDebug;
  // Ensure isOnRegUseList() returns false.
  Contents.Reg.Prev = 0;
  // Preserve the tie when the operand was already a register.
  if (!WasReg)
    TiedTo = 0;

  // If this operand is embedded in a function, add the operand to the
  // register's use/def list.
  if (RegInfo)
    RegInfo->addRegOperandToUseList(this);
}

/// isIdenticalTo - Return true if this operand is identical to the specified
/// operand. Note that this should stay in sync with the hash_value overload
/// below.
bool MachineOperand::isIdenticalTo(const MachineOperand &Other) const {
  if (getType() != Other.getType() ||
      getTargetFlags() != Other.getTargetFlags())
    return false;

  switch (getType()) {
  case MachineOperand::MO_Register:
    return getReg() == Other.getReg() && isDef() == Other.isDef() &&
           getSubReg() == Other.getSubReg();
  case MachineOperand::MO_Immediate:
    return getImm() == Other.getImm();
  case MachineOperand::MO_CImmediate:
    return getCImm() == Other.getCImm();
  case MachineOperand::MO_FPImmediate:
    return getFPImm() == Other.getFPImm();
  case MachineOperand::MO_MachineBasicBlock:
    return getMBB() == Other.getMBB();
  case MachineOperand::MO_FrameIndex:
    return getIndex() == Other.getIndex();
  case MachineOperand::MO_ConstantPoolIndex:
  case MachineOperand::MO_TargetIndex:
    return getIndex() == Other.getIndex() && getOffset() == Other.getOffset();
  case MachineOperand::MO_JumpTableIndex:
    return getIndex() == Other.getIndex();
  case MachineOperand::MO_GlobalAddress:
    return getGlobal() == Other.getGlobal() && getOffset() == Other.getOffset();
  case MachineOperand::MO_ExternalSymbol:
    return !strcmp(getSymbolName(), Other.getSymbolName()) &&
           getOffset() == Other.getOffset();
  case MachineOperand::MO_BlockAddress:
    return getBlockAddress() == Other.getBlockAddress() &&
           getOffset() == Other.getOffset();
  case MO_RegisterMask:
    return getRegMask() == Other.getRegMask();
  case MachineOperand::MO_MCSymbol:
    return getMCSymbol() == Other.getMCSymbol();
  case MachineOperand::MO_Metadata:
    return getMetadata() == Other.getMetadata();
  }
  llvm_unreachable("Invalid machine operand type");
}

// Note: this must stay exactly in sync with isIdenticalTo above.
hash_code llvm::hash_value(const MachineOperand &MO) {
  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    // Register operands don't have target flags.
    return hash_combine(MO.getType(), MO.getReg(), MO.getSubReg(), MO.isDef());
  case MachineOperand::MO_Immediate:
    return hash_combine(MO.getType(), MO.getTargetFlags(), MO.getImm());
  case MachineOperand::MO_CImmediate:
    return hash_combine(MO.getType(), MO.getTargetFlags(), MO.getCImm());
  case MachineOperand::MO_FPImmediate:
    return hash_combine(MO.getType(), MO.getTargetFlags(), MO.getFPImm());
  case MachineOperand::MO_MachineBasicBlock:
    return hash_combine(MO.getType(), MO.getTargetFlags(), MO.getMBB());
  case MachineOperand::MO_FrameIndex:
    return hash_combine(MO.getType(), MO.getTargetFlags(), MO.getIndex());
  case MachineOperand::MO_ConstantPoolIndex:
  case MachineOperand::MO_TargetIndex:
    return hash_combine(MO.getType(), MO.getTargetFlags(), MO.getIndex(),
                        MO.getOffset());
  case MachineOperand::MO_JumpTableIndex:
    return hash_combine(MO.getType(), MO.getTargetFlags(), MO.getIndex());
  case MachineOperand::MO_ExternalSymbol:
    return hash_combine(MO.getType(), MO.getTargetFlags(), MO.getOffset(),
                        MO.getSymbolName());
  case MachineOperand::MO_GlobalAddress:
    return hash_combine(MO.getType(), MO.getTargetFlags(), MO.getGlobal(),
                        MO.getOffset());
  case MachineOperand::MO_BlockAddress:
    return hash_combine(MO.getType(), MO.getTargetFlags(),
                        MO.getBlockAddress(), MO.getOffset());
  case MachineOperand::MO_RegisterMask:
    return hash_combine(MO.getType(), MO.getTargetFlags(), MO.getRegMask());
  case MachineOperand::MO_Metadata:
    return hash_combine(MO.getType(), MO.getTargetFlags(), MO.getMetadata());
  case MachineOperand::MO_MCSymbol:
    return hash_combine(MO.getType(), MO.getTargetFlags(), MO.getMCSymbol());
  }
  llvm_unreachable("Invalid machine operand type");
}

/// print - Print the specified machine operand.
///
void MachineOperand::print(raw_ostream &OS, const TargetMachine *TM) const {
  // If the instruction is embedded into a basic block, we can find the
  // target info for the instruction.
  if (!TM)
    if (const MachineInstr *MI = getParent())
      if (const MachineBasicBlock *MBB = MI->getParent())
        if (const MachineFunction *MF = MBB->getParent())
          TM = &MF->getTarget();
  const TargetRegisterInfo *TRI = TM ? TM->getRegisterInfo() : 0;

  switch (getType()) {
  case MachineOperand::MO_Register:
    OS << PrintReg(getReg(), TRI, getSubReg());

    if (isDef() || isKill() || isDead() || isImplicit() || isUndef() ||
        isInternalRead() || isEarlyClobber() || isTied()) {
      OS << '<';
      bool NeedComma = false;
      if (isDef()) {
        if (NeedComma) OS << ',';
        if (isEarlyClobber())
          OS << "earlyclobber,";
        if (isImplicit())
          OS << "imp-";
        OS << "def";
        NeedComma = true;
        // <def,read-undef> only makes sense when getSubReg() is set.
        // Don't clutter the output otherwise.
        if (isUndef() && getSubReg())
          OS << ",read-undef";
      } else if (isImplicit()) {
          OS << "imp-use";
          NeedComma = true;
      }

      if (isKill()) {
        if (NeedComma) OS << ',';
        OS << "kill";
        NeedComma = true;
      }
      if (isDead()) {
        if (NeedComma) OS << ',';
        OS << "dead";
        NeedComma = true;
      }
      if (isUndef() && isUse()) {
        if (NeedComma) OS << ',';
        OS << "undef";
        NeedComma = true;
      }
      if (isInternalRead()) {
        if (NeedComma) OS << ',';
        OS << "internal";
        NeedComma = true;
      }
      if (isTied()) {
        if (NeedComma) OS << ',';
        OS << "tied";
        if (TiedTo != 15)
          OS << unsigned(TiedTo - 1);
        NeedComma = true;
      }
      OS << '>';
    }
    break;
  case MachineOperand::MO_Immediate:
    OS << getImm();
    break;
  case MachineOperand::MO_CImmediate:
    getCImm()->getValue().print(OS, false);
    break;
  case MachineOperand::MO_FPImmediate:
    if (getFPImm()->getType()->isFloatTy())
      OS << getFPImm()->getValueAPF().convertToFloat();
    else
      OS << getFPImm()->getValueAPF().convertToDouble();
    break;
  case MachineOperand::MO_MachineBasicBlock:
    OS << "<BB#" << getMBB()->getNumber() << ">";
    break;
  case MachineOperand::MO_FrameIndex:
    OS << "<fi#" << getIndex() << '>';
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    OS << "<cp#" << getIndex();
    if (getOffset()) OS << "+" << getOffset();
    OS << '>';
    break;
  case MachineOperand::MO_TargetIndex:
    OS << "<ti#" << getIndex();
    if (getOffset()) OS << "+" << getOffset();
    OS << '>';
    break;
  case MachineOperand::MO_JumpTableIndex:
    OS << "<jt#" << getIndex() << '>';
    break;
  case MachineOperand::MO_GlobalAddress:
    OS << "<ga:";
    WriteAsOperand(OS, getGlobal(), /*PrintType=*/false);
    if (getOffset()) OS << "+" << getOffset();
    OS << '>';
    break;
  case MachineOperand::MO_ExternalSymbol:
    OS << "<es:" << getSymbolName();
    if (getOffset()) OS << "+" << getOffset();
    OS << '>';
    break;
  case MachineOperand::MO_BlockAddress:
    OS << '<';
    WriteAsOperand(OS, getBlockAddress(), /*PrintType=*/false);
    if (getOffset()) OS << "+" << getOffset();
    OS << '>';
    break;
  case MachineOperand::MO_RegisterMask:
    OS << "<regmask>";
    break;
  case MachineOperand::MO_Metadata:
    OS << '<';
    WriteAsOperand(OS, getMetadata(), /*PrintType=*/false);
    OS << '>';
    break;
  case MachineOperand::MO_MCSymbol:
    OS << "<MCSym=" << *getMCSymbol() << '>';
    break;
  }

  if (unsigned TF = getTargetFlags())
    OS << "[TF=" << TF << ']';
}

//===----------------------------------------------------------------------===//
// MachineMemOperand Implementation
//===----------------------------------------------------------------------===//

/// getAddrSpace - Return the LLVM IR address space number that this pointer
/// points into.
unsigned MachinePointerInfo::getAddrSpace() const {
  if (V == 0) return 0;
  return cast<PointerType>(V->getType())->getAddressSpace();
}

/// getConstantPool - Return a MachinePointerInfo record that refers to the
/// constant pool.
MachinePointerInfo MachinePointerInfo::getConstantPool() {
  return MachinePointerInfo(PseudoSourceValue::getConstantPool());
}

/// getFixedStack - Return a MachinePointerInfo record that refers to the
/// the specified FrameIndex.
MachinePointerInfo MachinePointerInfo::getFixedStack(int FI, int64_t offset) {
  return MachinePointerInfo(PseudoSourceValue::getFixedStack(FI), offset);
}

MachinePointerInfo MachinePointerInfo::getJumpTable() {
  return MachinePointerInfo(PseudoSourceValue::getJumpTable());
}

MachinePointerInfo MachinePointerInfo::getGOT() {
  return MachinePointerInfo(PseudoSourceValue::getGOT());
}

MachinePointerInfo MachinePointerInfo::getStack(int64_t Offset) {
  return MachinePointerInfo(PseudoSourceValue::getStack(), Offset);
}

MachineMemOperand::MachineMemOperand(MachinePointerInfo ptrinfo, unsigned f,
                                     uint64_t s, unsigned int a,
                                     const MDNode *TBAAInfo,
                                     const MDNode *Ranges)
  : PtrInfo(ptrinfo), Size(s),
    Flags((f & ((1 << MOMaxBits) - 1)) | ((Log2_32(a) + 1) << MOMaxBits)),
    TBAAInfo(TBAAInfo), Ranges(Ranges) {
  assert((PtrInfo.V == 0 || isa<PointerType>(PtrInfo.V->getType())) &&
         "invalid pointer value");
  assert(getBaseAlignment() == a && "Alignment is not a power of 2!");
  assert((isLoad() || isStore()) && "Not a load/store!");
}

/// Profile - Gather unique data for the object.
///
void MachineMemOperand::Profile(FoldingSetNodeID &ID) const {
  ID.AddInteger(getOffset());
  ID.AddInteger(Size);
  ID.AddPointer(getValue());
  ID.AddInteger(Flags);
}

void MachineMemOperand::refineAlignment(const MachineMemOperand *MMO) {
  // The Value and Offset may differ due to CSE. But the flags and size
  // should be the same.
  assert(MMO->getFlags() == getFlags() && "Flags mismatch!");
  assert(MMO->getSize() == getSize() && "Size mismatch!");

  if (MMO->getBaseAlignment() >= getBaseAlignment()) {
    // Update the alignment value.
    Flags = (Flags & ((1 << MOMaxBits) - 1)) |
      ((Log2_32(MMO->getBaseAlignment()) + 1) << MOMaxBits);
    // Also update the base and offset, because the new alignment may
    // not be applicable with the old ones.
    PtrInfo = MMO->PtrInfo;
  }
}

/// getAlignment - Return the minimum known alignment in bytes of the
/// actual memory reference.
uint64_t MachineMemOperand::getAlignment() const {
  return MinAlign(getBaseAlignment(), getOffset());
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const MachineMemOperand &MMO) {
  assert((MMO.isLoad() || MMO.isStore()) &&
         "SV has to be a load, store or both.");

  if (MMO.isVolatile())
    OS << "Volatile ";

  if (MMO.isLoad())
    OS << "LD";
  if (MMO.isStore())
    OS << "ST";
  OS << MMO.getSize();

  // Print the address information.
  OS << "[";
  if (!MMO.getValue())
    OS << "<unknown>";
  else
    WriteAsOperand(OS, MMO.getValue(), /*PrintType=*/false);

  // If the alignment of the memory reference itself differs from the alignment
  // of the base pointer, print the base alignment explicitly, next to the base
  // pointer.
  if (MMO.getBaseAlignment() != MMO.getAlignment())
    OS << "(align=" << MMO.getBaseAlignment() << ")";

  if (MMO.getOffset() != 0)
    OS << "+" << MMO.getOffset();
  OS << "]";

  // Print the alignment of the reference.
  if (MMO.getBaseAlignment() != MMO.getAlignment() ||
      MMO.getBaseAlignment() != MMO.getSize())
    OS << "(align=" << MMO.getAlignment() << ")";

  // Print TBAA info.
  if (const MDNode *TBAAInfo = MMO.getTBAAInfo()) {
    OS << "(tbaa=";
    if (TBAAInfo->getNumOperands() > 0)
      WriteAsOperand(OS, TBAAInfo->getOperand(0), /*PrintType=*/false);
    else
      OS << "<unknown>";
    OS << ")";
  }

  // Print nontemporal info.
  if (MMO.isNonTemporal())
    OS << "(nontemporal)";

  return OS;
}

//===----------------------------------------------------------------------===//
// MachineInstr Implementation
//===----------------------------------------------------------------------===//

/// MachineInstr ctor - This constructor creates a dummy MachineInstr with
/// MCID NULL and no operands.
MachineInstr::MachineInstr()
  : MCID(0), Flags(0), AsmPrinterFlags(0),
    NumMemRefs(0), MemRefs(0),
    Parent(0) {
  // Make sure that we get added to a machine basicblock
  LeakDetector::addGarbageObject(this);
}

void MachineInstr::addImplicitDefUseOperands() {
  if (MCID->ImplicitDefs)
    for (const uint16_t *ImpDefs = MCID->getImplicitDefs(); *ImpDefs; ++ImpDefs)
      addOperand(MachineOperand::CreateReg(*ImpDefs, true, true));
  if (MCID->ImplicitUses)
    for (const uint16_t *ImpUses = MCID->getImplicitUses(); *ImpUses; ++ImpUses)
      addOperand(MachineOperand::CreateReg(*ImpUses, false, true));
}

/// MachineInstr ctor - This constructor creates a MachineInstr and adds the
/// implicit operands. It reserves space for the number of operands specified by
/// the MCInstrDesc.
MachineInstr::MachineInstr(const MCInstrDesc &tid, const DebugLoc dl,
                           bool NoImp)
  : MCID(&tid), Flags(0), AsmPrinterFlags(0),
    NumMemRefs(0), MemRefs(0), Parent(0), debugLoc(dl) {
  unsigned NumImplicitOps = 0;
  if (!NoImp)
    NumImplicitOps = MCID->getNumImplicitDefs() + MCID->getNumImplicitUses();
  Operands.reserve(NumImplicitOps + MCID->getNumOperands());
  if (!NoImp)
    addImplicitDefUseOperands();
  // Make sure that we get added to a machine basicblock
  LeakDetector::addGarbageObject(this);
}

/// MachineInstr ctor - Work exactly the same as the ctor two above, except
/// that the MachineInstr is created and added to the end of the specified
/// basic block.
MachineInstr::MachineInstr(MachineBasicBlock *MBB, const DebugLoc dl,
                           const MCInstrDesc &tid)
  : MCID(&tid), Flags(0), AsmPrinterFlags(0),
    NumMemRefs(0), MemRefs(0), Parent(0), debugLoc(dl) {
  assert(MBB && "Cannot use inserting ctor with null basic block!");
  unsigned NumImplicitOps =
    MCID->getNumImplicitDefs() + MCID->getNumImplicitUses();
  Operands.reserve(NumImplicitOps + MCID->getNumOperands());
  addImplicitDefUseOperands();
  // Make sure that we get added to a machine basicblock
  LeakDetector::addGarbageObject(this);
  MBB->push_back(this);  // Add instruction to end of basic block!
}

/// MachineInstr ctor - Copies MachineInstr arg exactly
///
MachineInstr::MachineInstr(MachineFunction &MF, const MachineInstr &MI)
  : MCID(&MI.getDesc()), Flags(0), AsmPrinterFlags(0),
    NumMemRefs(MI.NumMemRefs), MemRefs(MI.MemRefs),
    Parent(0), debugLoc(MI.getDebugLoc()) {
  Operands.reserve(MI.getNumOperands());

  // Add operands
  for (unsigned i = 0; i != MI.getNumOperands(); ++i)
    addOperand(MI.getOperand(i));

  // Copy all the flags.
  Flags = MI.Flags;

  // Set parent to null.
  Parent = 0;

  LeakDetector::addGarbageObject(this);
}

MachineInstr::~MachineInstr() {
  LeakDetector::removeGarbageObject(this);
#ifndef NDEBUG
  for (unsigned i = 0, e = Operands.size(); i != e; ++i) {
    assert(Operands[i].ParentMI == this && "ParentMI mismatch!");
    assert((!Operands[i].isReg() || !Operands[i].isOnRegUseList()) &&
           "Reg operand def/use list corrupted");
  }
#endif
}

/// getRegInfo - If this instruction is embedded into a MachineFunction,
/// return the MachineRegisterInfo object for the current function, otherwise
/// return null.
MachineRegisterInfo *MachineInstr::getRegInfo() {
  if (MachineBasicBlock *MBB = getParent())
    return &MBB->getParent()->getRegInfo();
  return 0;
}

/// RemoveRegOperandsFromUseLists - Unlink all of the register operands in
/// this instruction from their respective use lists.  This requires that the
/// operands already be on their use lists.
void MachineInstr::RemoveRegOperandsFromUseLists(MachineRegisterInfo &MRI) {
  for (unsigned i = 0, e = Operands.size(); i != e; ++i)
    if (Operands[i].isReg())
      MRI.removeRegOperandFromUseList(&Operands[i]);
}

/// AddRegOperandsToUseLists - Add all of the register operands in
/// this instruction from their respective use lists.  This requires that the
/// operands not be on their use lists yet.
void MachineInstr::AddRegOperandsToUseLists(MachineRegisterInfo &MRI) {
  for (unsigned i = 0, e = Operands.size(); i != e; ++i)
    if (Operands[i].isReg())
      MRI.addRegOperandToUseList(&Operands[i]);
}

/// addOperand - Add the specified operand to the instruction.  If it is an
/// implicit operand, it is added to the end of the operand list.  If it is
/// an explicit operand it is added at the end of the explicit operand list
/// (before the first implicit operand).
void MachineInstr::addOperand(const MachineOperand &Op) {
  assert(MCID && "Cannot add operands before providing an instr descriptor");
  bool isImpReg = Op.isReg() && Op.isImplicit();
  MachineRegisterInfo *RegInfo = getRegInfo();

  // If the Operands backing store is reallocated, all register operands must
  // be removed and re-added to RegInfo.  It is storing pointers to operands.
  bool Reallocate = RegInfo &&
    !Operands.empty() && Operands.size() == Operands.capacity();

  // Find the insert location for the new operand.  Implicit registers go at
  // the end, everything goes before the implicit regs.
  unsigned OpNo = Operands.size();

  // Remove all the implicit operands from RegInfo if they need to be shifted.
  // FIXME: Allow mixed explicit and implicit operands on inline asm.
  // InstrEmitter::EmitSpecialNode() is marking inline asm clobbers as
  // implicit-defs, but they must not be moved around.  See the FIXME in
  // InstrEmitter.cpp.
  if (!isImpReg && !isInlineAsm()) {
    while (OpNo && Operands[OpNo-1].isReg() && Operands[OpNo-1].isImplicit()) {
      --OpNo;
      assert(!Operands[OpNo].isTied() && "Cannot move tied operands");
      if (RegInfo)
        RegInfo->removeRegOperandFromUseList(&Operands[OpNo]);
    }
  }

  // OpNo now points as the desired insertion point.  Unless this is a variadic
  // instruction, only implicit regs are allowed beyond MCID->getNumOperands().
  // RegMask operands go between the explicit and implicit operands.
  assert((isImpReg || Op.isRegMask() || MCID->isVariadic() ||
          OpNo < MCID->getNumOperands()) &&
         "Trying to add an operand to a machine instr that is already done!");

  // All operands from OpNo have been removed from RegInfo.  If the Operands
  // backing store needs to be reallocated, we also need to remove any other
  // register operands.
  if (Reallocate)
    for (unsigned i = 0; i != OpNo; ++i)
      if (Operands[i].isReg())
        RegInfo->removeRegOperandFromUseList(&Operands[i]);

  // Insert the new operand at OpNo.
  Operands.insert(Operands.begin() + OpNo, Op);
  Operands[OpNo].ParentMI = this;

  // The Operands backing store has now been reallocated, so we can re-add the
  // operands before OpNo.
  if (Reallocate)
    for (unsigned i = 0; i != OpNo; ++i)
      if (Operands[i].isReg())
        RegInfo->addRegOperandToUseList(&Operands[i]);

  // When adding a register operand, tell RegInfo about it.
  if (Operands[OpNo].isReg()) {
    // Ensure isOnRegUseList() returns false, regardless of Op's status.
    Operands[OpNo].Contents.Reg.Prev = 0;
    // Ignore existing ties. This is not a property that can be copied.
    Operands[OpNo].TiedTo = 0;
    // Add the new operand to RegInfo.
    if (RegInfo)
      RegInfo->addRegOperandToUseList(&Operands[OpNo]);
    // The MCID operand information isn't accurate until we start adding
    // explicit operands. The implicit operands are added first, then the
    // explicits are inserted before them.
    if (!isImpReg) {
      // Tie uses to defs as indicated in MCInstrDesc.
      if (Operands[OpNo].isUse()) {
        int DefIdx = MCID->getOperandConstraint(OpNo, MCOI::TIED_TO);
        if (DefIdx != -1)
          tieOperands(DefIdx, OpNo);
      }
      // If the register operand is flagged as early, mark the operand as such.
      if (MCID->getOperandConstraint(OpNo, MCOI::EARLY_CLOBBER) != -1)
        Operands[OpNo].setIsEarlyClobber(true);
    }
  }

  // Re-add all the implicit ops.
  if (RegInfo) {
    for (unsigned i = OpNo + 1, e = Operands.size(); i != e; ++i) {
      assert(Operands[i].isReg() && "Should only be an implicit reg!");
      RegInfo->addRegOperandToUseList(&Operands[i]);
    }
  }
}

/// RemoveOperand - Erase an operand  from an instruction, leaving it with one
/// fewer operand than it started with.
///
void MachineInstr::RemoveOperand(unsigned OpNo) {
  assert(OpNo < Operands.size() && "Invalid operand number");
  untieRegOperand(OpNo);
  MachineRegisterInfo *RegInfo = getRegInfo();

  // Special case removing the last one.
  if (OpNo == Operands.size()-1) {
    // If needed, remove from the reg def/use list.
    if (RegInfo && Operands.back().isReg() && Operands.back().isOnRegUseList())
      RegInfo->removeRegOperandFromUseList(&Operands.back());

    Operands.pop_back();
    return;
  }

  // Otherwise, we are removing an interior operand.  If we have reginfo to
  // update, remove all operands that will be shifted down from their reg lists,
  // move everything down, then re-add them.
  if (RegInfo) {
    for (unsigned i = OpNo, e = Operands.size(); i != e; ++i) {
      if (Operands[i].isReg())
        RegInfo->removeRegOperandFromUseList(&Operands[i]);
    }
  }

#ifndef NDEBUG
  // Moving tied operands would break the ties.
  for (unsigned i = OpNo + 1, e = Operands.size(); i != e; ++i)
    if (Operands[i].isReg())
      assert(!Operands[i].isTied() && "Cannot move tied operands");
#endif

  Operands.erase(Operands.begin()+OpNo);

  if (RegInfo) {
    for (unsigned i = OpNo, e = Operands.size(); i != e; ++i) {
      if (Operands[i].isReg())
        RegInfo->addRegOperandToUseList(&Operands[i]);
    }
  }
}

/// addMemOperand - Add a MachineMemOperand to the machine instruction.
/// This function should be used only occasionally. The setMemRefs function
/// is the primary method for setting up a MachineInstr's MemRefs list.
void MachineInstr::addMemOperand(MachineFunction &MF,
                                 MachineMemOperand *MO) {
  mmo_iterator OldMemRefs = MemRefs;
  uint16_t OldNumMemRefs = NumMemRefs;

  uint16_t NewNum = NumMemRefs + 1;
  mmo_iterator NewMemRefs = MF.allocateMemRefsArray(NewNum);

  std::copy(OldMemRefs, OldMemRefs + OldNumMemRefs, NewMemRefs);
  NewMemRefs[NewNum - 1] = MO;

  MemRefs = NewMemRefs;
  NumMemRefs = NewNum;
}

bool MachineInstr::hasPropertyInBundle(unsigned Mask, QueryType Type) const {
  const MachineBasicBlock *MBB = getParent();
  MachineBasicBlock::const_instr_iterator MII = *this; ++MII;
  while (MII != MBB->end() && MII->isInsideBundle()) {
    if (MII->getDesc().getFlags() & Mask) {
      if (Type == AnyInBundle)
        return true;
    } else {
      if (Type == AllInBundle)
        return false;
    }
    ++MII;
  }

  return Type == AllInBundle;
}

bool MachineInstr::isIdenticalTo(const MachineInstr *Other,
                                 MICheckType Check) const {
  // If opcodes or number of operands are not the same then the two
  // instructions are obviously not identical.
  if (Other->getOpcode() != getOpcode() ||
      Other->getNumOperands() != getNumOperands())
    return false;

  if (isBundle()) {
    // Both instructions are bundles, compare MIs inside the bundle.
    MachineBasicBlock::const_instr_iterator I1 = *this;
    MachineBasicBlock::const_instr_iterator E1 = getParent()->instr_end();
    MachineBasicBlock::const_instr_iterator I2 = *Other;
    MachineBasicBlock::const_instr_iterator E2= Other->getParent()->instr_end();
    while (++I1 != E1 && I1->isInsideBundle()) {
      ++I2;
      if (I2 == E2 || !I2->isInsideBundle() || !I1->isIdenticalTo(I2, Check))
        return false;
    }
  }

  // Check operands to make sure they match.
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = getOperand(i);
    const MachineOperand &OMO = Other->getOperand(i);
    if (!MO.isReg()) {
      if (!MO.isIdenticalTo(OMO))
        return false;
      continue;
    }

    // Clients may or may not want to ignore defs when testing for equality.
    // For example, machine CSE pass only cares about finding common
    // subexpressions, so it's safe to ignore virtual register defs.
    if (MO.isDef()) {
      if (Check == IgnoreDefs)
        continue;
      else if (Check == IgnoreVRegDefs) {
        if (TargetRegisterInfo::isPhysicalRegister(MO.getReg()) ||
            TargetRegisterInfo::isPhysicalRegister(OMO.getReg()))
          if (MO.getReg() != OMO.getReg())
            return false;
      } else {
        if (!MO.isIdenticalTo(OMO))
          return false;
        if (Check == CheckKillDead && MO.isDead() != OMO.isDead())
          return false;
      }
    } else {
      if (!MO.isIdenticalTo(OMO))
        return false;
      if (Check == CheckKillDead && MO.isKill() != OMO.isKill())
        return false;
    }
  }
  // If DebugLoc does not match then two dbg.values are not identical.
  if (isDebugValue())
    if (!getDebugLoc().isUnknown() && !Other->getDebugLoc().isUnknown()
        && getDebugLoc() != Other->getDebugLoc())
      return false;
  return true;
}

/// removeFromParent - This method unlinks 'this' from the containing basic
/// block, and returns it, but does not delete it.
MachineInstr *MachineInstr::removeFromParent() {
  assert(getParent() && "Not embedded in a basic block!");

  // If it's a bundle then remove the MIs inside the bundle as well.
  if (isBundle()) {
    MachineBasicBlock *MBB = getParent();
    MachineBasicBlock::instr_iterator MII = *this; ++MII;
    MachineBasicBlock::instr_iterator E = MBB->instr_end();
    while (MII != E && MII->isInsideBundle()) {
      MachineInstr *MI = &*MII;
      ++MII;
      MBB->remove(MI);
    }
  }
  getParent()->remove(this);
  return this;
}


/// eraseFromParent - This method unlinks 'this' from the containing basic
/// block, and deletes it.
void MachineInstr::eraseFromParent() {
  assert(getParent() && "Not embedded in a basic block!");
  // If it's a bundle then remove the MIs inside the bundle as well.
  if (isBundle()) {
    MachineBasicBlock *MBB = getParent();
    MachineBasicBlock::instr_iterator MII = *this; ++MII;
    MachineBasicBlock::instr_iterator E = MBB->instr_end();
    while (MII != E && MII->isInsideBundle()) {
      MachineInstr *MI = &*MII;
      ++MII;
      MBB->erase(MI);
    }
  }
  // Erase the individual instruction, which may itself be inside a bundle.
  getParent()->erase_instr(this);
}


/// getNumExplicitOperands - Returns the number of non-implicit operands.
///
unsigned MachineInstr::getNumExplicitOperands() const {
  unsigned NumOperands = MCID->getNumOperands();
  if (!MCID->isVariadic())
    return NumOperands;

  for (unsigned i = NumOperands, e = getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = getOperand(i);
    if (!MO.isReg() || !MO.isImplicit())
      NumOperands++;
  }
  return NumOperands;
}

/// isBundled - Return true if this instruction part of a bundle. This is true
/// if either itself or its following instruction is marked "InsideBundle".
bool MachineInstr::isBundled() const {
  if (isInsideBundle())
    return true;
  MachineBasicBlock::const_instr_iterator nextMI = this;
  ++nextMI;
  return nextMI != Parent->instr_end() && nextMI->isInsideBundle();
}

bool MachineInstr::isStackAligningInlineAsm() const {
  if (isInlineAsm()) {
    unsigned ExtraInfo = getOperand(InlineAsm::MIOp_ExtraInfo).getImm();
    if (ExtraInfo & InlineAsm::Extra_IsAlignStack)
      return true;
  }
  return false;
}

InlineAsm::AsmDialect MachineInstr::getInlineAsmDialect() const {
  assert(isInlineAsm() && "getInlineAsmDialect() only works for inline asms!");
  unsigned ExtraInfo = getOperand(InlineAsm::MIOp_ExtraInfo).getImm();
  return InlineAsm::AsmDialect((ExtraInfo & InlineAsm::Extra_AsmDialect) != 0);
}

int MachineInstr::findInlineAsmFlagIdx(unsigned OpIdx,
                                       unsigned *GroupNo) const {
  assert(isInlineAsm() && "Expected an inline asm instruction");
  assert(OpIdx < getNumOperands() && "OpIdx out of range");

  // Ignore queries about the initial operands.
  if (OpIdx < InlineAsm::MIOp_FirstOperand)
    return -1;

  unsigned Group = 0;
  unsigned NumOps;
  for (unsigned i = InlineAsm::MIOp_FirstOperand, e = getNumOperands(); i < e;
       i += NumOps) {
    const MachineOperand &FlagMO = getOperand(i);
    // If we reach the implicit register operands, stop looking.
    if (!FlagMO.isImm())
      return -1;
    NumOps = 1 + InlineAsm::getNumOperandRegisters(FlagMO.getImm());
    if (i + NumOps > OpIdx) {
      if (GroupNo)
        *GroupNo = Group;
      return i;
    }
    ++Group;
  }
  return -1;
}

const TargetRegisterClass*
MachineInstr::getRegClassConstraint(unsigned OpIdx,
                                    const TargetInstrInfo *TII,
                                    const TargetRegisterInfo *TRI) const {
  assert(getParent() && "Can't have an MBB reference here!");
  assert(getParent()->getParent() && "Can't have an MF reference here!");
  const MachineFunction &MF = *getParent()->getParent();

  // Most opcodes have fixed constraints in their MCInstrDesc.
  if (!isInlineAsm())
    return TII->getRegClass(getDesc(), OpIdx, TRI, MF);

  if (!getOperand(OpIdx).isReg())
    return NULL;

  // For tied uses on inline asm, get the constraint from the def.
  unsigned DefIdx;
  if (getOperand(OpIdx).isUse() && isRegTiedToDefOperand(OpIdx, &DefIdx))
    OpIdx = DefIdx;

  // Inline asm stores register class constraints in the flag word.
  int FlagIdx = findInlineAsmFlagIdx(OpIdx);
  if (FlagIdx < 0)
    return NULL;

  unsigned Flag = getOperand(FlagIdx).getImm();
  unsigned RCID;
  if (InlineAsm::hasRegClassConstraint(Flag, RCID))
    return TRI->getRegClass(RCID);

  // Assume that all registers in a memory operand are pointers.
  if (InlineAsm::getKind(Flag) == InlineAsm::Kind_Mem)
    return TRI->getPointerRegClass(MF);

  return NULL;
}

/// getBundleSize - Return the number of instructions inside the MI bundle.
unsigned MachineInstr::getBundleSize() const {
  assert(isBundle() && "Expecting a bundle");

  const MachineBasicBlock *MBB = getParent();
  MachineBasicBlock::const_instr_iterator I = *this, E = MBB->instr_end();
  unsigned Size = 0;
  while ((++I != E) && I->isInsideBundle()) {
    ++Size;
  }
  assert(Size > 1 && "Malformed bundle");

  return Size;
}

/// findRegisterUseOperandIdx() - Returns the MachineOperand that is a use of
/// the specific register or -1 if it is not found. It further tightens
/// the search criteria to a use that kills the register if isKill is true.
int MachineInstr::findRegisterUseOperandIdx(unsigned Reg, bool isKill,
                                          const TargetRegisterInfo *TRI) const {
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = getOperand(i);
    if (!MO.isReg() || !MO.isUse())
      continue;
    unsigned MOReg = MO.getReg();
    if (!MOReg)
      continue;
    if (MOReg == Reg ||
        (TRI &&
         TargetRegisterInfo::isPhysicalRegister(MOReg) &&
         TargetRegisterInfo::isPhysicalRegister(Reg) &&
         TRI->isSubRegister(MOReg, Reg)))
      if (!isKill || MO.isKill())
        return i;
  }
  return -1;
}

/// readsWritesVirtualRegister - Return a pair of bools (reads, writes)
/// indicating if this instruction reads or writes Reg. This also considers
/// partial defines.
std::pair<bool,bool>
MachineInstr::readsWritesVirtualRegister(unsigned Reg,
                                         SmallVectorImpl<unsigned> *Ops) const {
  bool PartDef = false; // Partial redefine.
  bool FullDef = false; // Full define.
  bool Use = false;

  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = getOperand(i);
    if (!MO.isReg() || MO.getReg() != Reg)
      continue;
    if (Ops)
      Ops->push_back(i);
    if (MO.isUse())
      Use |= !MO.isUndef();
    else if (MO.getSubReg() && !MO.isUndef())
      // A partial <def,undef> doesn't count as reading the register.
      PartDef = true;
    else
      FullDef = true;
  }
  // A partial redefine uses Reg unless there is also a full define.
  return std::make_pair(Use || (PartDef && !FullDef), PartDef || FullDef);
}

/// findRegisterDefOperandIdx() - Returns the operand index that is a def of
/// the specified register or -1 if it is not found. If isDead is true, defs
/// that are not dead are skipped. If TargetRegisterInfo is non-null, then it
/// also checks if there is a def of a super-register.
int
MachineInstr::findRegisterDefOperandIdx(unsigned Reg, bool isDead, bool Overlap,
                                        const TargetRegisterInfo *TRI) const {
  bool isPhys = TargetRegisterInfo::isPhysicalRegister(Reg);
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = getOperand(i);
    // Accept regmask operands when Overlap is set.
    // Ignore them when looking for a specific def operand (Overlap == false).
    if (isPhys && Overlap && MO.isRegMask() && MO.clobbersPhysReg(Reg))
      return i;
    if (!MO.isReg() || !MO.isDef())
      continue;
    unsigned MOReg = MO.getReg();
    bool Found = (MOReg == Reg);
    if (!Found && TRI && isPhys &&
        TargetRegisterInfo::isPhysicalRegister(MOReg)) {
      if (Overlap)
        Found = TRI->regsOverlap(MOReg, Reg);
      else
        Found = TRI->isSubRegister(MOReg, Reg);
    }
    if (Found && (!isDead || MO.isDead()))
      return i;
  }
  return -1;
}

/// findFirstPredOperandIdx() - Find the index of the first operand in the
/// operand list that is used to represent the predicate. It returns -1 if
/// none is found.
int MachineInstr::findFirstPredOperandIdx() const {
  // Don't call MCID.findFirstPredOperandIdx() because this variant
  // is sometimes called on an instruction that's not yet complete, and
  // so the number of operands is less than the MCID indicates. In
  // particular, the PTX target does this.
  const MCInstrDesc &MCID = getDesc();
  if (MCID.isPredicable()) {
    for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
      if (MCID.OpInfo[i].isPredicate())
        return i;
  }

  return -1;
}

// MachineOperand::TiedTo is 4 bits wide.
const unsigned TiedMax = 15;

/// tieOperands - Mark operands at DefIdx and UseIdx as tied to each other.
///
/// Use and def operands can be tied together, indicated by a non-zero TiedTo
/// field. TiedTo can have these values:
///
/// 0:              Operand is not tied to anything.
/// 1 to TiedMax-1: Tied to getOperand(TiedTo-1).
/// TiedMax:        Tied to an operand >= TiedMax-1.
///
/// The tied def must be one of the first TiedMax operands on a normal
/// instruction. INLINEASM instructions allow more tied defs.
///
void MachineInstr::tieOperands(unsigned DefIdx, unsigned UseIdx) {
  MachineOperand &DefMO = getOperand(DefIdx);
  MachineOperand &UseMO = getOperand(UseIdx);
  assert(DefMO.isDef() && "DefIdx must be a def operand");
  assert(UseMO.isUse() && "UseIdx must be a use operand");
  assert(!DefMO.isTied() && "Def is already tied to another use");
  assert(!UseMO.isTied() && "Use is already tied to another def");

  if (DefIdx < TiedMax)
    UseMO.TiedTo = DefIdx + 1;
  else {
    // Inline asm can use the group descriptors to find tied operands, but on
    // normal instruction, the tied def must be within the first TiedMax
    // operands.
    assert(isInlineAsm() && "DefIdx out of range");
    UseMO.TiedTo = TiedMax;
  }

  // UseIdx can be out of range, we'll search for it in findTiedOperandIdx().
  DefMO.TiedTo = std::min(UseIdx + 1, TiedMax);
}

/// Given the index of a tied register operand, find the operand it is tied to.
/// Defs are tied to uses and vice versa. Returns the index of the tied operand
/// which must exist.
unsigned MachineInstr::findTiedOperandIdx(unsigned OpIdx) const {
  const MachineOperand &MO = getOperand(OpIdx);
  assert(MO.isTied() && "Operand isn't tied");

  // Normally TiedTo is in range.
  if (MO.TiedTo < TiedMax)
    return MO.TiedTo - 1;

  // Uses on normal instructions can be out of range.
  if (!isInlineAsm()) {
    // Normal tied defs must be in the 0..TiedMax-1 range.
    if (MO.isUse())
      return TiedMax - 1;
    // MO is a def. Search for the tied use.
    for (unsigned i = TiedMax - 1, e = getNumOperands(); i != e; ++i) {
      const MachineOperand &UseMO = getOperand(i);
      if (UseMO.isReg() && UseMO.isUse() && UseMO.TiedTo == OpIdx + 1)
        return i;
    }
    llvm_unreachable("Can't find tied use");
  }

  // Now deal with inline asm by parsing the operand group descriptor flags.
  // Find the beginning of each operand group.
  SmallVector<unsigned, 8> GroupIdx;
  unsigned OpIdxGroup = ~0u;
  unsigned NumOps;
  for (unsigned i = InlineAsm::MIOp_FirstOperand, e = getNumOperands(); i < e;
       i += NumOps) {
    const MachineOperand &FlagMO = getOperand(i);
    assert(FlagMO.isImm() && "Invalid tied operand on inline asm");
    unsigned CurGroup = GroupIdx.size();
    GroupIdx.push_back(i);
    NumOps = 1 + InlineAsm::getNumOperandRegisters(FlagMO.getImm());
    // OpIdx belongs to this operand group.
    if (OpIdx > i && OpIdx < i + NumOps)
      OpIdxGroup = CurGroup;
    unsigned TiedGroup;
    if (!InlineAsm::isUseOperandTiedToDef(FlagMO.getImm(), TiedGroup))
      continue;
    // Operands in this group are tied to operands in TiedGroup which must be
    // earlier. Find the number of operands between the two groups.
    unsigned Delta = i - GroupIdx[TiedGroup];

    // OpIdx is a use tied to TiedGroup.
    if (OpIdxGroup == CurGroup)
      return OpIdx - Delta;

    // OpIdx is a def tied to this use group.
    if (OpIdxGroup == TiedGroup)
      return OpIdx + Delta;
  }
  llvm_unreachable("Invalid tied operand on inline asm");
}

/// clearKillInfo - Clears kill flags on all operands.
///
void MachineInstr::clearKillInfo() {
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    MachineOperand &MO = getOperand(i);
    if (MO.isReg() && MO.isUse())
      MO.setIsKill(false);
  }
}

/// copyKillDeadInfo - Copies kill / dead operand properties from MI.
///
void MachineInstr::copyKillDeadInfo(const MachineInstr *MI) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || (!MO.isKill() && !MO.isDead()))
      continue;
    for (unsigned j = 0, ee = getNumOperands(); j != ee; ++j) {
      MachineOperand &MOp = getOperand(j);
      if (!MOp.isIdenticalTo(MO))
        continue;
      if (MO.isKill())
        MOp.setIsKill();
      else
        MOp.setIsDead();
      break;
    }
  }
}

/// copyPredicates - Copies predicate operand(s) from MI.
void MachineInstr::copyPredicates(const MachineInstr *MI) {
  assert(!isBundle() && "MachineInstr::copyPredicates() can't handle bundles");

  const MCInstrDesc &MCID = MI->getDesc();
  if (!MCID.isPredicable())
    return;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    if (MCID.OpInfo[i].isPredicate()) {
      // Predicated operands must be last operands.
      addOperand(MI->getOperand(i));
    }
  }
}

void MachineInstr::substituteRegister(unsigned FromReg,
                                      unsigned ToReg,
                                      unsigned SubIdx,
                                      const TargetRegisterInfo &RegInfo) {
  if (TargetRegisterInfo::isPhysicalRegister(ToReg)) {
    if (SubIdx)
      ToReg = RegInfo.getSubReg(ToReg, SubIdx);
    for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
      MachineOperand &MO = getOperand(i);
      if (!MO.isReg() || MO.getReg() != FromReg)
        continue;
      MO.substPhysReg(ToReg, RegInfo);
    }
  } else {
    for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
      MachineOperand &MO = getOperand(i);
      if (!MO.isReg() || MO.getReg() != FromReg)
        continue;
      MO.substVirtReg(ToReg, SubIdx, RegInfo);
    }
  }
}

/// isSafeToMove - Return true if it is safe to move this instruction. If
/// SawStore is set to true, it means that there is a store (or call) between
/// the instruction's location and its intended destination.
bool MachineInstr::isSafeToMove(const TargetInstrInfo *TII,
                                AliasAnalysis *AA,
                                bool &SawStore) const {
  // Ignore stuff that we obviously can't move.
  //
  // Treat volatile loads as stores. This is not strictly necessary for
  // volatiles, but it is required for atomic loads. It is not allowed to move
  // a load across an atomic load with Ordering > Monotonic.
  if (mayStore() || isCall() ||
      (mayLoad() && hasOrderedMemoryRef())) {
    SawStore = true;
    return false;
  }

  if (isLabel() || isDebugValue() ||
      isTerminator() || hasUnmodeledSideEffects())
    return false;

  // See if this instruction does a load.  If so, we have to guarantee that the
  // loaded value doesn't change between the load and the its intended
  // destination. The check for isInvariantLoad gives the targe the chance to
  // classify the load as always returning a constant, e.g. a constant pool
  // load.
  if (mayLoad() && !isInvariantLoad(AA))
    // Otherwise, this is a real load.  If there is a store between the load and
    // end of block, we can't move it.
    return !SawStore;

  return true;
}

/// isSafeToReMat - Return true if it's safe to rematerialize the specified
/// instruction which defined the specified register instead of copying it.
bool MachineInstr::isSafeToReMat(const TargetInstrInfo *TII,
                                 AliasAnalysis *AA,
                                 unsigned DstReg) const {
  bool SawStore = false;
  if (!TII->isTriviallyReMaterializable(this, AA) ||
      !isSafeToMove(TII, AA, SawStore))
    return false;
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = getOperand(i);
    if (!MO.isReg())
      continue;
    // FIXME: For now, do not remat any instruction with register operands.
    // Later on, we can loosen the restriction is the register operands have
    // not been modified between the def and use. Note, this is different from
    // MachineSink because the code is no longer in two-address form (at least
    // partially).
    if (MO.isUse())
      return false;
    else if (!MO.isDead() && MO.getReg() != DstReg)
      return false;
  }
  return true;
}

/// hasOrderedMemoryRef - Return true if this instruction may have an ordered
/// or volatile memory reference, or if the information describing the memory
/// reference is not available. Return false if it is known to have no ordered
/// memory references.
bool MachineInstr::hasOrderedMemoryRef() const {
  // An instruction known never to access memory won't have a volatile access.
  if (!mayStore() &&
      !mayLoad() &&
      !isCall() &&
      !hasUnmodeledSideEffects())
    return false;

  // Otherwise, if the instruction has no memory reference information,
  // conservatively assume it wasn't preserved.
  if (memoperands_empty())
    return true;

  // Check the memory reference information for ordered references.
  for (mmo_iterator I = memoperands_begin(), E = memoperands_end(); I != E; ++I)
    if (!(*I)->isUnordered())
      return true;

  return false;
}

/// isInvariantLoad - Return true if this instruction is loading from a
/// location whose value is invariant across the function.  For example,
/// loading a value from the constant pool or from the argument area
/// of a function if it does not change.  This should only return true of
/// *all* loads the instruction does are invariant (if it does multiple loads).
bool MachineInstr::isInvariantLoad(AliasAnalysis *AA) const {
  // If the instruction doesn't load at all, it isn't an invariant load.
  if (!mayLoad())
    return false;

  // If the instruction has lost its memoperands, conservatively assume that
  // it may not be an invariant load.
  if (memoperands_empty())
    return false;

  const MachineFrameInfo *MFI = getParent()->getParent()->getFrameInfo();

  for (mmo_iterator I = memoperands_begin(),
       E = memoperands_end(); I != E; ++I) {
    if ((*I)->isVolatile()) return false;
    if ((*I)->isStore()) return false;
    if ((*I)->isInvariant()) return true;

    if (const Value *V = (*I)->getValue()) {
      // A load from a constant PseudoSourceValue is invariant.
      if (const PseudoSourceValue *PSV = dyn_cast<PseudoSourceValue>(V))
        if (PSV->isConstant(MFI))
          continue;
      // If we have an AliasAnalysis, ask it whether the memory is constant.
      if (AA && AA->pointsToConstantMemory(
                      AliasAnalysis::Location(V, (*I)->getSize(),
                                              (*I)->getTBAAInfo())))
        continue;
    }

    // Otherwise assume conservatively.
    return false;
  }

  // Everything checks out.
  return true;
}

/// isConstantValuePHI - If the specified instruction is a PHI that always
/// merges together the same virtual register, return the register, otherwise
/// return 0.
unsigned MachineInstr::isConstantValuePHI() const {
  if (!isPHI())
    return 0;
  assert(getNumOperands() >= 3 &&
         "It's illegal to have a PHI without source operands");

  unsigned Reg = getOperand(1).getReg();
  for (unsigned i = 3, e = getNumOperands(); i < e; i += 2)
    if (getOperand(i).getReg() != Reg)
      return 0;
  return Reg;
}

bool MachineInstr::hasUnmodeledSideEffects() const {
  if (hasProperty(MCID::UnmodeledSideEffects))
    return true;
  if (isInlineAsm()) {
    unsigned ExtraInfo = getOperand(InlineAsm::MIOp_ExtraInfo).getImm();
    if (ExtraInfo & InlineAsm::Extra_HasSideEffects)
      return true;
  }

  return false;
}

/// allDefsAreDead - Return true if all the defs of this instruction are dead.
///
bool MachineInstr::allDefsAreDead() const {
  for (unsigned i = 0, e = getNumOperands(); i < e; ++i) {
    const MachineOperand &MO = getOperand(i);
    if (!MO.isReg() || MO.isUse())
      continue;
    if (!MO.isDead())
      return false;
  }
  return true;
}

/// copyImplicitOps - Copy implicit register operands from specified
/// instruction to this instruction.
void MachineInstr::copyImplicitOps(const MachineInstr *MI) {
  for (unsigned i = MI->getDesc().getNumOperands(), e = MI->getNumOperands();
       i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.isImplicit())
      addOperand(MO);
  }
}

void MachineInstr::dump() const {
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  dbgs() << "  " << *this;
#endif
}

static void printDebugLoc(DebugLoc DL, const MachineFunction *MF,
                         raw_ostream &CommentOS) {
  const LLVMContext &Ctx = MF->getFunction()->getContext();
  if (!DL.isUnknown()) {          // Print source line info.
    DIScope Scope(DL.getScope(Ctx));
    // Omit the directory, because it's likely to be long and uninteresting.
    if (Scope.Verify())
      CommentOS << Scope.getFilename();
    else
      CommentOS << "<unknown>";
    CommentOS << ':' << DL.getLine();
    if (DL.getCol() != 0)
      CommentOS << ':' << DL.getCol();
    DebugLoc InlinedAtDL = DebugLoc::getFromDILocation(DL.getInlinedAt(Ctx));
    if (!InlinedAtDL.isUnknown()) {
      CommentOS << " @[ ";
      printDebugLoc(InlinedAtDL, MF, CommentOS);
      CommentOS << " ]";
    }
  }
}

void MachineInstr::print(raw_ostream &OS, const TargetMachine *TM) const {
  // We can be a bit tidier if we know the TargetMachine and/or MachineFunction.
  const MachineFunction *MF = 0;
  const MachineRegisterInfo *MRI = 0;
  if (const MachineBasicBlock *MBB = getParent()) {
    MF = MBB->getParent();
    if (!TM && MF)
      TM = &MF->getTarget();
    if (MF)
      MRI = &MF->getRegInfo();
  }

  // Save a list of virtual registers.
  SmallVector<unsigned, 8> VirtRegs;

  // Print explicitly defined operands on the left of an assignment syntax.
  unsigned StartOp = 0, e = getNumOperands();
  for (; StartOp < e && getOperand(StartOp).isReg() &&
         getOperand(StartOp).isDef() &&
         !getOperand(StartOp).isImplicit();
       ++StartOp) {
    if (StartOp != 0) OS << ", ";
    getOperand(StartOp).print(OS, TM);
    unsigned Reg = getOperand(StartOp).getReg();
    if (TargetRegisterInfo::isVirtualRegister(Reg))
      VirtRegs.push_back(Reg);
  }

  if (StartOp != 0)
    OS << " = ";

  // Print the opcode name.
  if (TM && TM->getInstrInfo())
    OS << TM->getInstrInfo()->getName(getOpcode());
  else
    OS << "UNKNOWN";

  // Print the rest of the operands.
  bool OmittedAnyCallClobbers = false;
  bool FirstOp = true;
  unsigned AsmDescOp = ~0u;
  unsigned AsmOpCount = 0;

  if (isInlineAsm() && e >= InlineAsm::MIOp_FirstOperand) {
    // Print asm string.
    OS << " ";
    getOperand(InlineAsm::MIOp_AsmString).print(OS, TM);

    // Print HasSideEffects, IsAlignStack
    unsigned ExtraInfo = getOperand(InlineAsm::MIOp_ExtraInfo).getImm();
    if (ExtraInfo & InlineAsm::Extra_HasSideEffects)
      OS << " [sideeffect]";
    if (ExtraInfo & InlineAsm::Extra_IsAlignStack)
      OS << " [alignstack]";
    if (getInlineAsmDialect() == InlineAsm::AD_ATT)
      OS << " [attdialect]";
    if (getInlineAsmDialect() == InlineAsm::AD_Intel)
      OS << " [inteldialect]";

    StartOp = AsmDescOp = InlineAsm::MIOp_FirstOperand;
    FirstOp = false;
  }


  for (unsigned i = StartOp, e = getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = getOperand(i);

    if (MO.isReg() && TargetRegisterInfo::isVirtualRegister(MO.getReg()))
      VirtRegs.push_back(MO.getReg());

    // Omit call-clobbered registers which aren't used anywhere. This makes
    // call instructions much less noisy on targets where calls clobber lots
    // of registers. Don't rely on MO.isDead() because we may be called before
    // LiveVariables is run, or we may be looking at a non-allocatable reg.
    if (MF && isCall() &&
        MO.isReg() && MO.isImplicit() && MO.isDef()) {
      unsigned Reg = MO.getReg();
      if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
        const MachineRegisterInfo &MRI = MF->getRegInfo();
        if (MRI.use_empty(Reg) && !MRI.isLiveOut(Reg)) {
          bool HasAliasLive = false;
          for (MCRegAliasIterator AI(Reg, TM->getRegisterInfo(), true);
               AI.isValid(); ++AI) {
            unsigned AliasReg = *AI;
            if (!MRI.use_empty(AliasReg) || MRI.isLiveOut(AliasReg)) {
              HasAliasLive = true;
              break;
            }
          }
          if (!HasAliasLive) {
            OmittedAnyCallClobbers = true;
            continue;
          }
        }
      }
    }

    if (FirstOp) FirstOp = false; else OS << ",";
    OS << " ";
    if (i < getDesc().NumOperands) {
      const MCOperandInfo &MCOI = getDesc().OpInfo[i];
      if (MCOI.isPredicate())
        OS << "pred:";
      if (MCOI.isOptionalDef())
        OS << "opt:";
    }
    if (isDebugValue() && MO.isMetadata()) {
      // Pretty print DBG_VALUE instructions.
      const MDNode *MD = MO.getMetadata();
      if (const MDString *MDS = dyn_cast<MDString>(MD->getOperand(2)))
        OS << "!\"" << MDS->getString() << '\"';
      else
        MO.print(OS, TM);
    } else if (TM && (isInsertSubreg() || isRegSequence()) && MO.isImm()) {
      OS << TM->getRegisterInfo()->getSubRegIndexName(MO.getImm());
    } else if (i == AsmDescOp && MO.isImm()) {
      // Pretty print the inline asm operand descriptor.
      OS << '$' << AsmOpCount++;
      unsigned Flag = MO.getImm();
      switch (InlineAsm::getKind(Flag)) {
      case InlineAsm::Kind_RegUse:             OS << ":[reguse"; break;
      case InlineAsm::Kind_RegDef:             OS << ":[regdef"; break;
      case InlineAsm::Kind_RegDefEarlyClobber: OS << ":[regdef-ec"; break;
      case InlineAsm::Kind_Clobber:            OS << ":[clobber"; break;
      case InlineAsm::Kind_Imm:                OS << ":[imm"; break;
      case InlineAsm::Kind_Mem:                OS << ":[mem"; break;
      default: OS << ":[??" << InlineAsm::getKind(Flag); break;
      }

      unsigned RCID = 0;
      if (InlineAsm::hasRegClassConstraint(Flag, RCID)) {
        if (TM)
          OS << ':' << TM->getRegisterInfo()->getRegClass(RCID)->getName();
        else
          OS << ":RC" << RCID;
      }

      unsigned TiedTo = 0;
      if (InlineAsm::isUseOperandTiedToDef(Flag, TiedTo))
        OS << " tiedto:$" << TiedTo;

      OS << ']';

      // Compute the index of the next operand descriptor.
      AsmDescOp += 1 + InlineAsm::getNumOperandRegisters(Flag);
    } else
      MO.print(OS, TM);
  }

  // Briefly indicate whether any call clobbers were omitted.
  if (OmittedAnyCallClobbers) {
    if (!FirstOp) OS << ",";
    OS << " ...";
  }

  bool HaveSemi = false;
  if (Flags) {
    if (!HaveSemi) OS << ";"; HaveSemi = true;
    OS << " flags: ";

    if (Flags & FrameSetup)
      OS << "FrameSetup";
  }

  if (!memoperands_empty()) {
    if (!HaveSemi) OS << ";"; HaveSemi = true;

    OS << " mem:";
    for (mmo_iterator i = memoperands_begin(), e = memoperands_end();
         i != e; ++i) {
      OS << **i;
      if (llvm::next(i) != e)
        OS << " ";
    }
  }

  // Print the regclass of any virtual registers encountered.
  if (MRI && !VirtRegs.empty()) {
    if (!HaveSemi) OS << ";"; HaveSemi = true;
    for (unsigned i = 0; i != VirtRegs.size(); ++i) {
      const TargetRegisterClass *RC = MRI->getRegClass(VirtRegs[i]);
      OS << " " << RC->getName() << ':' << PrintReg(VirtRegs[i]);
      for (unsigned j = i+1; j != VirtRegs.size();) {
        if (MRI->getRegClass(VirtRegs[j]) != RC) {
          ++j;
          continue;
        }
        if (VirtRegs[i] != VirtRegs[j])
          OS << "," << PrintReg(VirtRegs[j]);
        VirtRegs.erase(VirtRegs.begin()+j);
      }
    }
  }

  // Print debug location information.
  if (isDebugValue() && getOperand(e - 1).isMetadata()) {
    if (!HaveSemi) OS << ";"; HaveSemi = true;
    DIVariable DV(getOperand(e - 1).getMetadata());
    OS << " line no:" <<  DV.getLineNumber();
    if (MDNode *InlinedAt = DV.getInlinedAt()) {
      DebugLoc InlinedAtDL = DebugLoc::getFromDILocation(InlinedAt);
      if (!InlinedAtDL.isUnknown()) {
        OS << " inlined @[ ";
        printDebugLoc(InlinedAtDL, MF, OS);
        OS << " ]";
      }
    }
  } else if (!debugLoc.isUnknown() && MF) {
    if (!HaveSemi) OS << ";"; HaveSemi = true;
    OS << " dbg:";
    printDebugLoc(debugLoc, MF, OS);
  }

  OS << '\n';
}

bool MachineInstr::addRegisterKilled(unsigned IncomingReg,
                                     const TargetRegisterInfo *RegInfo,
                                     bool AddIfNotFound) {
  bool isPhysReg = TargetRegisterInfo::isPhysicalRegister(IncomingReg);
  bool hasAliases = isPhysReg &&
    MCRegAliasIterator(IncomingReg, RegInfo, false).isValid();
  bool Found = false;
  SmallVector<unsigned,4> DeadOps;
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    MachineOperand &MO = getOperand(i);
    if (!MO.isReg() || !MO.isUse() || MO.isUndef())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;

    if (Reg == IncomingReg) {
      if (!Found) {
        if (MO.isKill())
          // The register is already marked kill.
          return true;
        if (isPhysReg && isRegTiedToDefOperand(i))
          // Two-address uses of physregs must not be marked kill.
          return true;
        MO.setIsKill();
        Found = true;
      }
    } else if (hasAliases && MO.isKill() &&
               TargetRegisterInfo::isPhysicalRegister(Reg)) {
      // A super-register kill already exists.
      if (RegInfo->isSuperRegister(IncomingReg, Reg))
        return true;
      if (RegInfo->isSubRegister(IncomingReg, Reg))
        DeadOps.push_back(i);
    }
  }

  // Trim unneeded kill operands.
  while (!DeadOps.empty()) {
    unsigned OpIdx = DeadOps.back();
    if (getOperand(OpIdx).isImplicit())
      RemoveOperand(OpIdx);
    else
      getOperand(OpIdx).setIsKill(false);
    DeadOps.pop_back();
  }

  // If not found, this means an alias of one of the operands is killed. Add a
  // new implicit operand if required.
  if (!Found && AddIfNotFound) {
    addOperand(MachineOperand::CreateReg(IncomingReg,
                                         false /*IsDef*/,
                                         true  /*IsImp*/,
                                         true  /*IsKill*/));
    return true;
  }
  return Found;
}

void MachineInstr::clearRegisterKills(unsigned Reg,
                                      const TargetRegisterInfo *RegInfo) {
  if (!TargetRegisterInfo::isPhysicalRegister(Reg))
    RegInfo = 0;
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    MachineOperand &MO = getOperand(i);
    if (!MO.isReg() || !MO.isUse() || !MO.isKill())
      continue;
    unsigned OpReg = MO.getReg();
    if (OpReg == Reg || (RegInfo && RegInfo->isSuperRegister(Reg, OpReg)))
      MO.setIsKill(false);
  }
}

bool MachineInstr::addRegisterDead(unsigned IncomingReg,
                                   const TargetRegisterInfo *RegInfo,
                                   bool AddIfNotFound) {
  bool isPhysReg = TargetRegisterInfo::isPhysicalRegister(IncomingReg);
  bool hasAliases = isPhysReg &&
    MCRegAliasIterator(IncomingReg, RegInfo, false).isValid();
  bool Found = false;
  SmallVector<unsigned,4> DeadOps;
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    MachineOperand &MO = getOperand(i);
    if (!MO.isReg() || !MO.isDef())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;

    if (Reg == IncomingReg) {
      MO.setIsDead();
      Found = true;
    } else if (hasAliases && MO.isDead() &&
               TargetRegisterInfo::isPhysicalRegister(Reg)) {
      // There exists a super-register that's marked dead.
      if (RegInfo->isSuperRegister(IncomingReg, Reg))
        return true;
      if (RegInfo->isSubRegister(IncomingReg, Reg))
        DeadOps.push_back(i);
    }
  }

  // Trim unneeded dead operands.
  while (!DeadOps.empty()) {
    unsigned OpIdx = DeadOps.back();
    if (getOperand(OpIdx).isImplicit())
      RemoveOperand(OpIdx);
    else
      getOperand(OpIdx).setIsDead(false);
    DeadOps.pop_back();
  }

  // If not found, this means an alias of one of the operands is dead. Add a
  // new implicit operand if required.
  if (Found || !AddIfNotFound)
    return Found;

  addOperand(MachineOperand::CreateReg(IncomingReg,
                                       true  /*IsDef*/,
                                       true  /*IsImp*/,
                                       false /*IsKill*/,
                                       true  /*IsDead*/));
  return true;
}

void MachineInstr::addRegisterDefined(unsigned IncomingReg,
                                      const TargetRegisterInfo *RegInfo) {
  if (TargetRegisterInfo::isPhysicalRegister(IncomingReg)) {
    MachineOperand *MO = findRegisterDefOperand(IncomingReg, false, RegInfo);
    if (MO)
      return;
  } else {
    for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = getOperand(i);
      if (MO.isReg() && MO.getReg() == IncomingReg && MO.isDef() &&
          MO.getSubReg() == 0)
        return;
    }
  }
  addOperand(MachineOperand::CreateReg(IncomingReg,
                                       true  /*IsDef*/,
                                       true  /*IsImp*/));
}

void MachineInstr::setPhysRegsDeadExcept(ArrayRef<unsigned> UsedRegs,
                                         const TargetRegisterInfo &TRI) {
  bool HasRegMask = false;
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    MachineOperand &MO = getOperand(i);
    if (MO.isRegMask()) {
      HasRegMask = true;
      continue;
    }
    if (!MO.isReg() || !MO.isDef()) continue;
    unsigned Reg = MO.getReg();
    if (!TargetRegisterInfo::isPhysicalRegister(Reg)) continue;
    bool Dead = true;
    for (ArrayRef<unsigned>::iterator I = UsedRegs.begin(), E = UsedRegs.end();
         I != E; ++I)
      if (TRI.regsOverlap(*I, Reg)) {
        Dead = false;
        break;
      }
    // If there are no uses, including partial uses, the def is dead.
    if (Dead) MO.setIsDead();
  }

  // This is a call with a register mask operand.
  // Mask clobbers are always dead, so add defs for the non-dead defines.
  if (HasRegMask)
    for (ArrayRef<unsigned>::iterator I = UsedRegs.begin(), E = UsedRegs.end();
         I != E; ++I)
      addRegisterDefined(*I, &TRI);
}

unsigned
MachineInstrExpressionTrait::getHashValue(const MachineInstr* const &MI) {
  // Build up a buffer of hash code components.
  SmallVector<size_t, 8> HashComponents;
  HashComponents.reserve(MI->getNumOperands() + 1);
  HashComponents.push_back(MI->getOpcode());
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.isDef() &&
        TargetRegisterInfo::isVirtualRegister(MO.getReg()))
      continue;  // Skip virtual register defs.

    HashComponents.push_back(hash_value(MO));
  }
  return hash_combine_range(HashComponents.begin(), HashComponents.end());
}

void MachineInstr::emitError(StringRef Msg) const {
  // Find the source location cookie.
  unsigned LocCookie = 0;
  const MDNode *LocMD = 0;
  for (unsigned i = getNumOperands(); i != 0; --i) {
    if (getOperand(i-1).isMetadata() &&
        (LocMD = getOperand(i-1).getMetadata()) &&
        LocMD->getNumOperands() != 0) {
      if (const ConstantInt *CI = dyn_cast<ConstantInt>(LocMD->getOperand(0))) {
        LocCookie = CI->getZExtValue();
        break;
      }
    }
  }

  if (const MachineBasicBlock *MBB = getParent())
    if (const MachineFunction *MF = MBB->getParent())
      return MF->getMMI().getModule()->getContext().emitError(LocCookie, Msg);
  report_fatal_error(Msg);
}
