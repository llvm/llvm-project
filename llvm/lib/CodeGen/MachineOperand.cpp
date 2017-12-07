//===- lib/CodeGen/MachineOperand.cpp -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file Methods common to all machine operands.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/CodeGen/MIRPrinter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/Target/TargetIntrinsicInfo.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

static cl::opt<int>
    PrintRegMaskNumRegs("print-regmask-num-regs",
                        cl::desc("Number of registers to limit to when "
                                 "printing regmask operands in IR dumps. "
                                 "unlimited = -1"),
                        cl::init(32), cl::Hidden);

static const MachineFunction *getMFIfAvailable(const MachineOperand &MO) {
  if (const MachineInstr *MI = MO.getParent())
    if (const MachineBasicBlock *MBB = MI->getParent())
      if (const MachineFunction *MF = MBB->getParent())
        return MF;
  return nullptr;
}
static MachineFunction *getMFIfAvailable(MachineOperand &MO) {
  return const_cast<MachineFunction *>(
      getMFIfAvailable(const_cast<const MachineOperand &>(MO)));
}

void MachineOperand::setReg(unsigned Reg) {
  if (getReg() == Reg)
    return; // No change.

  // Otherwise, we have to change the register.  If this operand is embedded
  // into a machine function, we need to update the old and new register's
  // use/def lists.
  if (MachineFunction *MF = getMFIfAvailable(*this)) {
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
    if (isDef())
      setIsUndef(false);
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
  if (MachineFunction *MF = getMFIfAvailable(*this)) {
    MachineRegisterInfo &MRI = MF->getRegInfo();
    MRI.removeRegOperandFromUseList(this);
    IsDef = Val;
    MRI.addRegOperandToUseList(this);
    return;
  }
  IsDef = Val;
}

// If this operand is currently a register operand, and if this is in a
// function, deregister the operand from the register's use/def list.
void MachineOperand::removeRegFromUses() {
  if (!isReg() || !isOnRegUseList())
    return;

  if (MachineFunction *MF = getMFIfAvailable(*this))
    MF->getRegInfo().removeRegOperandFromUseList(this);
}

/// ChangeToImmediate - Replace this operand with a new immediate operand of
/// the specified value.  If an operand is known to be an immediate already,
/// the setImm method should be used.
void MachineOperand::ChangeToImmediate(int64_t ImmVal) {
  assert((!isReg() || !isTied()) && "Cannot change a tied operand into an imm");

  removeRegFromUses();

  OpKind = MO_Immediate;
  Contents.ImmVal = ImmVal;
}

void MachineOperand::ChangeToFPImmediate(const ConstantFP *FPImm) {
  assert((!isReg() || !isTied()) && "Cannot change a tied operand into an imm");

  removeRegFromUses();

  OpKind = MO_FPImmediate;
  Contents.CFP = FPImm;
}

void MachineOperand::ChangeToES(const char *SymName,
                                unsigned char TargetFlags) {
  assert((!isReg() || !isTied()) &&
         "Cannot change a tied operand into an external symbol");

  removeRegFromUses();

  OpKind = MO_ExternalSymbol;
  Contents.OffsetedInfo.Val.SymbolName = SymName;
  setOffset(0); // Offset is always 0.
  setTargetFlags(TargetFlags);
}

void MachineOperand::ChangeToMCSymbol(MCSymbol *Sym) {
  assert((!isReg() || !isTied()) &&
         "Cannot change a tied operand into an MCSymbol");

  removeRegFromUses();

  OpKind = MO_MCSymbol;
  Contents.Sym = Sym;
}

void MachineOperand::ChangeToFrameIndex(int Idx) {
  assert((!isReg() || !isTied()) &&
         "Cannot change a tied operand into a FrameIndex");

  removeRegFromUses();

  OpKind = MO_FrameIndex;
  setIndex(Idx);
}

void MachineOperand::ChangeToTargetIndex(unsigned Idx, int64_t Offset,
                                         unsigned char TargetFlags) {
  assert((!isReg() || !isTied()) &&
         "Cannot change a tied operand into a FrameIndex");

  removeRegFromUses();

  OpKind = MO_TargetIndex;
  setIndex(Idx);
  setOffset(Offset);
  setTargetFlags(TargetFlags);
}

/// ChangeToRegister - Replace this operand with a new register operand of
/// the specified value.  If an operand is known to be an register already,
/// the setReg method should be used.
void MachineOperand::ChangeToRegister(unsigned Reg, bool isDef, bool isImp,
                                      bool isKill, bool isDead, bool isUndef,
                                      bool isDebug) {
  MachineRegisterInfo *RegInfo = nullptr;
  if (MachineFunction *MF = getMFIfAvailable(*this))
    RegInfo = &MF->getRegInfo();
  // If this operand is already a register operand, remove it from the
  // register's use/def lists.
  bool WasReg = isReg();
  if (RegInfo && WasReg)
    RegInfo->removeRegOperandFromUseList(this);

  // Change this to a register and set the reg#.
  OpKind = MO_Register;
  SmallContents.RegNo = Reg;
  SubReg_TargetFlags = 0;
  IsDef = isDef;
  IsImp = isImp;
  IsKill = isKill;
  IsDead = isDead;
  IsUndef = isUndef;
  IsInternalRead = false;
  IsEarlyClobber = false;
  IsDebug = isDebug;
  // Ensure isOnRegUseList() returns false.
  Contents.Reg.Prev = nullptr;
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
    return strcmp(getSymbolName(), Other.getSymbolName()) == 0 &&
           getOffset() == Other.getOffset();
  case MachineOperand::MO_BlockAddress:
    return getBlockAddress() == Other.getBlockAddress() &&
           getOffset() == Other.getOffset();
  case MachineOperand::MO_RegisterMask:
  case MachineOperand::MO_RegisterLiveOut: {
    // Shallow compare of the two RegMasks
    const uint32_t *RegMask = getRegMask();
    const uint32_t *OtherRegMask = Other.getRegMask();
    if (RegMask == OtherRegMask)
      return true;

    if (const MachineFunction *MF = getMFIfAvailable(*this)) {
      // Calculate the size of the RegMask
      const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
      unsigned RegMaskSize = (TRI->getNumRegs() + 31) / 32;

      // Deep compare of the two RegMasks
      return std::equal(RegMask, RegMask + RegMaskSize, OtherRegMask);
    }
    // We don't know the size of the RegMask, so we can't deep compare the two
    // reg masks.
    return false;
  }
  case MachineOperand::MO_MCSymbol:
    return getMCSymbol() == Other.getMCSymbol();
  case MachineOperand::MO_CFIIndex:
    return getCFIIndex() == Other.getCFIIndex();
  case MachineOperand::MO_Metadata:
    return getMetadata() == Other.getMetadata();
  case MachineOperand::MO_IntrinsicID:
    return getIntrinsicID() == Other.getIntrinsicID();
  case MachineOperand::MO_Predicate:
    return getPredicate() == Other.getPredicate();
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
    return hash_combine(MO.getType(), MO.getTargetFlags(), MO.getBlockAddress(),
                        MO.getOffset());
  case MachineOperand::MO_RegisterMask:
  case MachineOperand::MO_RegisterLiveOut:
    return hash_combine(MO.getType(), MO.getTargetFlags(), MO.getRegMask());
  case MachineOperand::MO_Metadata:
    return hash_combine(MO.getType(), MO.getTargetFlags(), MO.getMetadata());
  case MachineOperand::MO_MCSymbol:
    return hash_combine(MO.getType(), MO.getTargetFlags(), MO.getMCSymbol());
  case MachineOperand::MO_CFIIndex:
    return hash_combine(MO.getType(), MO.getTargetFlags(), MO.getCFIIndex());
  case MachineOperand::MO_IntrinsicID:
    return hash_combine(MO.getType(), MO.getTargetFlags(), MO.getIntrinsicID());
  case MachineOperand::MO_Predicate:
    return hash_combine(MO.getType(), MO.getTargetFlags(), MO.getPredicate());
  }
  llvm_unreachable("Invalid machine operand type");
}

// Try to crawl up to the machine function and get TRI and IntrinsicInfo from
// it.
static void tryToGetTargetInfo(const MachineOperand &MO,
                               const TargetRegisterInfo *&TRI,
                               const TargetIntrinsicInfo *&IntrinsicInfo) {
  if (const MachineFunction *MF = getMFIfAvailable(MO)) {
    TRI = MF->getSubtarget().getRegisterInfo();
    IntrinsicInfo = MF->getTarget().getIntrinsicInfo();
  }
}

void MachineOperand::print(raw_ostream &OS, const TargetRegisterInfo *TRI,
                           const TargetIntrinsicInfo *IntrinsicInfo) const {
  tryToGetTargetInfo(*this, TRI, IntrinsicInfo);
  ModuleSlotTracker DummyMST(nullptr);
  print(OS, DummyMST, LLT{}, /*PrintDef=*/false,
        /*ShouldPrintRegisterTies=*/true,
        /*TiedOperandIdx=*/0, TRI, IntrinsicInfo);
}

void MachineOperand::print(raw_ostream &OS, ModuleSlotTracker &MST,
                           LLT TypeToPrint, bool PrintDef,
                           bool ShouldPrintRegisterTies,
                           unsigned TiedOperandIdx,
                           const TargetRegisterInfo *TRI,
                           const TargetIntrinsicInfo *IntrinsicInfo) const {
  switch (getType()) {
  case MachineOperand::MO_Register: {
    unsigned Reg = getReg();
    if (isImplicit())
      OS << (isDef() ? "implicit-def " : "implicit ");
    else if (PrintDef && isDef())
      // Print the 'def' flag only when the operand is defined after '='.
      OS << "def ";
    if (isInternalRead())
      OS << "internal ";
    if (isDead())
      OS << "dead ";
    if (isKill())
      OS << "killed ";
    if (isUndef())
      OS << "undef ";
    if (isEarlyClobber())
      OS << "early-clobber ";
    if (isDebug())
      OS << "debug-use ";
    OS << printReg(Reg, TRI);
    // Print the sub register.
    if (unsigned SubReg = getSubReg()) {
      if (TRI)
        OS << '.' << TRI->getSubRegIndexName(SubReg);
      else
        OS << ".subreg" << SubReg;
    }
    // Print the register class / bank.
    if (TargetRegisterInfo::isVirtualRegister(Reg)) {
      if (const MachineFunction *MF = getMFIfAvailable(*this)) {
        const MachineRegisterInfo &MRI = MF->getRegInfo();
        if (!PrintDef || MRI.def_empty(Reg)) {
          OS << ':';
          OS << printRegClassOrBank(Reg, MRI, TRI);
        }
      }
    }
    // Print ties.
    if (ShouldPrintRegisterTies && isTied() && !isDef())
      OS << "(tied-def " << TiedOperandIdx << ")";
    // Print types.
    if (TypeToPrint.isValid())
      OS << '(' << TypeToPrint << ')';
    break;
  }
  case MachineOperand::MO_Immediate:
    OS << getImm();
    break;
  case MachineOperand::MO_CImmediate:
    getCImm()->getValue().print(OS, false);
    break;
  case MachineOperand::MO_FPImmediate:
    if (getFPImm()->getType()->isFloatTy()) {
      OS << getFPImm()->getValueAPF().convertToFloat();
    } else if (getFPImm()->getType()->isHalfTy()) {
      APFloat APF = getFPImm()->getValueAPF();
      bool Unused;
      APF.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &Unused);
      OS << "half " << APF.convertToFloat();
    } else if (getFPImm()->getType()->isFP128Ty()) {
      APFloat APF = getFPImm()->getValueAPF();
      SmallString<16> Str;
      getFPImm()->getValueAPF().toString(Str);
      OS << "quad " << Str;
    } else if (getFPImm()->getType()->isX86_FP80Ty()) {
      APFloat APF = getFPImm()->getValueAPF();
      OS << "x86_fp80 0xK";
      APInt API = APF.bitcastToAPInt();
      OS << format_hex_no_prefix(API.getHiBits(16).getZExtValue(), 4,
                                 /*Upper=*/true);
      OS << format_hex_no_prefix(API.getLoBits(64).getZExtValue(), 16,
                                 /*Upper=*/true);
    } else {
      OS << getFPImm()->getValueAPF().convertToDouble();
    }
    break;
  case MachineOperand::MO_MachineBasicBlock:
    OS << printMBBReference(*getMBB());
    break;
  case MachineOperand::MO_FrameIndex:
    OS << "<fi#" << getIndex() << '>';
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    OS << "<cp#" << getIndex();
    if (getOffset())
      OS << "+" << getOffset();
    OS << '>';
    break;
  case MachineOperand::MO_TargetIndex:
    OS << "<ti#" << getIndex();
    if (getOffset())
      OS << "+" << getOffset();
    OS << '>';
    break;
  case MachineOperand::MO_JumpTableIndex:
    OS << "<jt#" << getIndex() << '>';
    break;
  case MachineOperand::MO_GlobalAddress:
    OS << "<ga:";
    getGlobal()->printAsOperand(OS, /*PrintType=*/false, MST);
    if (getOffset())
      OS << "+" << getOffset();
    OS << '>';
    break;
  case MachineOperand::MO_ExternalSymbol:
    OS << "<es:" << getSymbolName();
    if (getOffset())
      OS << "+" << getOffset();
    OS << '>';
    break;
  case MachineOperand::MO_BlockAddress:
    OS << '<';
    getBlockAddress()->printAsOperand(OS, /*PrintType=*/false, MST);
    if (getOffset())
      OS << "+" << getOffset();
    OS << '>';
    break;
  case MachineOperand::MO_RegisterMask: {
    OS << "<regmask";
    if (TRI) {
      unsigned NumRegsInMask = 0;
      unsigned NumRegsEmitted = 0;
      for (unsigned i = 0; i < TRI->getNumRegs(); ++i) {
        unsigned MaskWord = i / 32;
        unsigned MaskBit = i % 32;
        if (getRegMask()[MaskWord] & (1 << MaskBit)) {
          if (PrintRegMaskNumRegs < 0 ||
              NumRegsEmitted <= static_cast<unsigned>(PrintRegMaskNumRegs)) {
            OS << " " << printReg(i, TRI);
            NumRegsEmitted++;
          }
          NumRegsInMask++;
        }
      }
      if (NumRegsEmitted != NumRegsInMask)
        OS << " and " << (NumRegsInMask - NumRegsEmitted) << " more...";
    } else {
      OS << " ...";
    }
    OS << ">";
    break;
  }
  case MachineOperand::MO_RegisterLiveOut:
    OS << "<regliveout>";
    break;
  case MachineOperand::MO_Metadata:
    OS << '<';
    getMetadata()->printAsOperand(OS, MST);
    OS << '>';
    break;
  case MachineOperand::MO_MCSymbol:
    OS << "<MCSym=" << *getMCSymbol() << '>';
    break;
  case MachineOperand::MO_CFIIndex:
    OS << "<call frame instruction>";
    break;
  case MachineOperand::MO_IntrinsicID: {
    Intrinsic::ID ID = getIntrinsicID();
    if (ID < Intrinsic::num_intrinsics)
      OS << "<intrinsic:@" << Intrinsic::getName(ID, None) << '>';
    else if (IntrinsicInfo)
      OS << "<intrinsic:@" << IntrinsicInfo->getName(ID) << '>';
    else
      OS << "<intrinsic:" << ID << '>';
    break;
  }
  case MachineOperand::MO_Predicate: {
    auto Pred = static_cast<CmpInst::Predicate>(getPredicate());
    OS << '<' << (CmpInst::isIntPredicate(Pred) ? "intpred" : "floatpred")
       << CmpInst::getPredicateName(Pred) << '>';
    break;
  }
  }
  if (unsigned TF = getTargetFlags())
    OS << "[TF=" << TF << ']';
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void MachineOperand::dump() const { dbgs() << *this << '\n'; }
#endif

//===----------------------------------------------------------------------===//
// MachineMemOperand Implementation
//===----------------------------------------------------------------------===//

/// getAddrSpace - Return the LLVM IR address space number that this pointer
/// points into.
unsigned MachinePointerInfo::getAddrSpace() const { return AddrSpace; }

/// isDereferenceable - Return true if V is always dereferenceable for
/// Offset + Size byte.
bool MachinePointerInfo::isDereferenceable(unsigned Size, LLVMContext &C,
                                           const DataLayout &DL) const {
  if (!V.is<const Value *>())
    return false;

  const Value *BasePtr = V.get<const Value *>();
  if (BasePtr == nullptr)
    return false;

  return isDereferenceableAndAlignedPointer(
      BasePtr, 1, APInt(DL.getPointerSizeInBits(), Offset + Size), DL);
}

/// getConstantPool - Return a MachinePointerInfo record that refers to the
/// constant pool.
MachinePointerInfo MachinePointerInfo::getConstantPool(MachineFunction &MF) {
  return MachinePointerInfo(MF.getPSVManager().getConstantPool());
}

/// getFixedStack - Return a MachinePointerInfo record that refers to the
/// the specified FrameIndex.
MachinePointerInfo MachinePointerInfo::getFixedStack(MachineFunction &MF,
                                                     int FI, int64_t Offset) {
  return MachinePointerInfo(MF.getPSVManager().getFixedStack(FI), Offset);
}

MachinePointerInfo MachinePointerInfo::getJumpTable(MachineFunction &MF) {
  return MachinePointerInfo(MF.getPSVManager().getJumpTable());
}

MachinePointerInfo MachinePointerInfo::getGOT(MachineFunction &MF) {
  return MachinePointerInfo(MF.getPSVManager().getGOT());
}

MachinePointerInfo MachinePointerInfo::getStack(MachineFunction &MF,
                                                int64_t Offset, uint8_t ID) {
  return MachinePointerInfo(MF.getPSVManager().getStack(), Offset, ID);
}

MachinePointerInfo MachinePointerInfo::getUnknownStack(MachineFunction &MF) {
  return MachinePointerInfo(MF.getDataLayout().getAllocaAddrSpace());
}

MachineMemOperand::MachineMemOperand(MachinePointerInfo ptrinfo, Flags f,
                                     uint64_t s, unsigned int a,
                                     const AAMDNodes &AAInfo,
                                     const MDNode *Ranges, SyncScope::ID SSID,
                                     AtomicOrdering Ordering,
                                     AtomicOrdering FailureOrdering)
    : PtrInfo(ptrinfo), Size(s), FlagVals(f), BaseAlignLog2(Log2_32(a) + 1),
      AAInfo(AAInfo), Ranges(Ranges) {
  assert((PtrInfo.V.isNull() || PtrInfo.V.is<const PseudoSourceValue *>() ||
          isa<PointerType>(PtrInfo.V.get<const Value *>()->getType())) &&
         "invalid pointer value");
  assert(getBaseAlignment() == a && "Alignment is not a power of 2!");
  assert((isLoad() || isStore()) && "Not a load/store!");

  AtomicInfo.SSID = static_cast<unsigned>(SSID);
  assert(getSyncScopeID() == SSID && "Value truncated");
  AtomicInfo.Ordering = static_cast<unsigned>(Ordering);
  assert(getOrdering() == Ordering && "Value truncated");
  AtomicInfo.FailureOrdering = static_cast<unsigned>(FailureOrdering);
  assert(getFailureOrdering() == FailureOrdering && "Value truncated");
}

/// Profile - Gather unique data for the object.
///
void MachineMemOperand::Profile(FoldingSetNodeID &ID) const {
  ID.AddInteger(getOffset());
  ID.AddInteger(Size);
  ID.AddPointer(getOpaqueValue());
  ID.AddInteger(getFlags());
  ID.AddInteger(getBaseAlignment());
}

void MachineMemOperand::refineAlignment(const MachineMemOperand *MMO) {
  // The Value and Offset may differ due to CSE. But the flags and size
  // should be the same.
  assert(MMO->getFlags() == getFlags() && "Flags mismatch!");
  assert(MMO->getSize() == getSize() && "Size mismatch!");

  if (MMO->getBaseAlignment() >= getBaseAlignment()) {
    // Update the alignment value.
    BaseAlignLog2 = Log2_32(MMO->getBaseAlignment()) + 1;
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

void MachineMemOperand::print(raw_ostream &OS) const {
  ModuleSlotTracker DummyMST(nullptr);
  print(OS, DummyMST);
}
void MachineMemOperand::print(raw_ostream &OS, ModuleSlotTracker &MST) const {
  assert((isLoad() || isStore()) && "SV has to be a load, store or both.");

  if (isVolatile())
    OS << "Volatile ";

  if (isLoad())
    OS << "LD";
  if (isStore())
    OS << "ST";
  OS << getSize();

  // Print the address information.
  OS << "[";
  if (const Value *V = getValue())
    V->printAsOperand(OS, /*PrintType=*/false, MST);
  else if (const PseudoSourceValue *PSV = getPseudoValue())
    PSV->printCustom(OS);
  else
    OS << "<unknown>";

  unsigned AS = getAddrSpace();
  if (AS != 0)
    OS << "(addrspace=" << AS << ')';

  // If the alignment of the memory reference itself differs from the alignment
  // of the base pointer, print the base alignment explicitly, next to the base
  // pointer.
  if (getBaseAlignment() != getAlignment())
    OS << "(align=" << getBaseAlignment() << ")";

  if (getOffset() != 0)
    OS << "+" << getOffset();
  OS << "]";

  // Print the alignment of the reference.
  if (getBaseAlignment() != getAlignment() || getBaseAlignment() != getSize())
    OS << "(align=" << getAlignment() << ")";

  // Print TBAA info.
  if (const MDNode *TBAAInfo = getAAInfo().TBAA) {
    OS << "(tbaa=";
    if (TBAAInfo->getNumOperands() > 0)
      TBAAInfo->getOperand(0)->printAsOperand(OS, MST);
    else
      OS << "<unknown>";
    OS << ")";
  }

  // Print AA scope info.
  if (const MDNode *ScopeInfo = getAAInfo().Scope) {
    OS << "(alias.scope=";
    if (ScopeInfo->getNumOperands() > 0)
      for (unsigned i = 0, ie = ScopeInfo->getNumOperands(); i != ie; ++i) {
        ScopeInfo->getOperand(i)->printAsOperand(OS, MST);
        if (i != ie - 1)
          OS << ",";
      }
    else
      OS << "<unknown>";
    OS << ")";
  }

  // Print AA noalias scope info.
  if (const MDNode *NoAliasInfo = getAAInfo().NoAlias) {
    OS << "(noalias=";
    if (NoAliasInfo->getNumOperands() > 0)
      for (unsigned i = 0, ie = NoAliasInfo->getNumOperands(); i != ie; ++i) {
        NoAliasInfo->getOperand(i)->printAsOperand(OS, MST);
        if (i != ie - 1)
          OS << ",";
      }
    else
      OS << "<unknown>";
    OS << ")";
  }

  if (const MDNode *Ranges = getRanges()) {
    unsigned NumRanges = Ranges->getNumOperands();
    if (NumRanges != 0) {
      OS << "(ranges=";

      for (unsigned I = 0; I != NumRanges; ++I) {
        Ranges->getOperand(I)->printAsOperand(OS, MST);
        if (I != NumRanges - 1)
          OS << ',';
      }

      OS << ')';
    }
  }

  if (isNonTemporal())
    OS << "(nontemporal)";
  if (isDereferenceable())
    OS << "(dereferenceable)";
  if (isInvariant())
    OS << "(invariant)";
  if (getFlags() & MOTargetFlag1)
    OS << "(flag1)";
  if (getFlags() & MOTargetFlag2)
    OS << "(flag2)";
  if (getFlags() & MOTargetFlag3)
    OS << "(flag3)";
}
