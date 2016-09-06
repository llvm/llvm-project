//===- llvm/CodeGen/GlobalISel/RegisterBankInfo.cpp --------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the RegisterBankInfo class.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/RegisterBankInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/GlobalISel/RegisterBank.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetOpcodes.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"

#include <algorithm> // For std::max.

#define DEBUG_TYPE "registerbankinfo"

using namespace llvm;

const unsigned RegisterBankInfo::DefaultMappingID = UINT_MAX;
const unsigned RegisterBankInfo::InvalidMappingID = UINT_MAX - 1;

//------------------------------------------------------------------------------
// RegisterBankInfo implementation.
//------------------------------------------------------------------------------
RegisterBankInfo::RegisterBankInfo(unsigned NumRegBanks)
    : NumRegBanks(NumRegBanks) {
  RegBanks.reset(new RegisterBank[NumRegBanks]);
}

bool RegisterBankInfo::verify(const TargetRegisterInfo &TRI) const {
  DEBUG(for (unsigned Idx = 0, End = getNumRegBanks(); Idx != End; ++Idx) {
    const RegisterBank &RegBank = getRegBank(Idx);
    assert(Idx == RegBank.getID() &&
           "ID does not match the index in the array");
    dbgs() << "Verify " << RegBank << '\n';
    assert(RegBank.verify(TRI) && "RegBank is invalid");
  });
  return true;
}

void RegisterBankInfo::createRegisterBank(unsigned ID, const char *Name) {
  DEBUG(dbgs() << "Create register bank: " << ID << " with name \"" << Name
               << "\"\n");
  RegisterBank &RegBank = getRegBank(ID);
  assert(RegBank.getID() == RegisterBank::InvalidID &&
         "A register bank should be created only once");
  RegBank.ID = ID;
  RegBank.Name = Name;
}

void RegisterBankInfo::addRegBankCoverage(unsigned ID, unsigned RCId,
                                          const TargetRegisterInfo &TRI) {
  RegisterBank &RB = getRegBank(ID);
  unsigned NbOfRegClasses = TRI.getNumRegClasses();

  DEBUG(dbgs() << "Add coverage for: " << RB << '\n');

  // Check if RB is underconstruction.
  if (!RB.isValid())
    RB.ContainedRegClasses.resize(NbOfRegClasses);
  else if (RB.covers(*TRI.getRegClass(RCId)))
    // If RB already covers this register class, there is nothing
    // to do.
    return;

  BitVector &Covered = RB.ContainedRegClasses;
  SmallVector<unsigned, 8> WorkList;

  WorkList.push_back(RCId);
  Covered.set(RCId);

  unsigned &MaxSize = RB.Size;
  do {
    unsigned RCId = WorkList.pop_back_val();

    const TargetRegisterClass &CurRC = *TRI.getRegClass(RCId);

    DEBUG(dbgs() << "Examine: " << TRI.getRegClassName(&CurRC)
                 << "(Size*8: " << (CurRC.getSize() * 8) << ")\n");

    // Remember the biggest size in bits.
    MaxSize = std::max(MaxSize, CurRC.getSize() * 8);

    // Walk through all sub register classes and push them into the worklist.
    bool First = true;
    for (BitMaskClassIterator It(CurRC.getSubClassMask(), TRI); It.isValid();
         ++It) {
      unsigned SubRCId = It.getID();
      if (!Covered.test(SubRCId)) {
        if (First)
          DEBUG(dbgs() << "  Enqueue sub-class: ");
        DEBUG(dbgs() << TRI.getRegClassName(TRI.getRegClass(SubRCId)) << ", ");
        WorkList.push_back(SubRCId);
        // Remember that we saw the sub class.
        Covered.set(SubRCId);
        First = false;
      }
    }
    if (!First)
      DEBUG(dbgs() << '\n');

    // Push also all the register classes that can be accessed via a
    // subreg index, i.e., its subreg-class (which is different than
    // its subclass).
    //
    // Note: It would probably be faster to go the other way around
    // and have this method add only super classes, since this
    // information is available in a more efficient way. However, it
    // feels less natural for the client of this APIs plus we will
    // TableGen the whole bitset at some point, so compile time for
    // the initialization is not very important.
    First = true;
    for (unsigned SubRCId = 0; SubRCId < NbOfRegClasses; ++SubRCId) {
      if (Covered.test(SubRCId))
        continue;
      bool Pushed = false;
      const TargetRegisterClass *SubRC = TRI.getRegClass(SubRCId);
      for (SuperRegClassIterator SuperRCIt(SubRC, &TRI); SuperRCIt.isValid();
           ++SuperRCIt) {
        if (Pushed)
          break;
        for (BitMaskClassIterator It(SuperRCIt.getMask(), TRI); It.isValid();
             ++It) {
          unsigned SuperRCId = It.getID();
          if (SuperRCId == RCId) {
            if (First)
              DEBUG(dbgs() << "  Enqueue subreg-class: ");
            DEBUG(dbgs() << TRI.getRegClassName(SubRC) << ", ");
            WorkList.push_back(SubRCId);
            // Remember that we saw the sub class.
            Covered.set(SubRCId);
            Pushed = true;
            First = false;
            break;
          }
        }
      }
    }
    if (!First)
      DEBUG(dbgs() << '\n');
  } while (!WorkList.empty());
}

const RegisterBank *
RegisterBankInfo::getRegBank(unsigned Reg, const MachineRegisterInfo &MRI,
                             const TargetRegisterInfo &TRI) const {
  if (TargetRegisterInfo::isPhysicalRegister(Reg))
    return &getRegBankFromRegClass(*TRI.getMinimalPhysRegClass(Reg));

  assert(Reg && "NoRegister does not have a register bank");
  const RegClassOrRegBank &RegClassOrBank = MRI.getRegClassOrRegBank(Reg);
  if (auto *RB = RegClassOrBank.dyn_cast<const RegisterBank *>())
    return RB;
  if (auto *RC = RegClassOrBank.dyn_cast<const TargetRegisterClass *>())
    return &getRegBankFromRegClass(*RC);
  return nullptr;
}

const RegisterBank *RegisterBankInfo::getRegBankFromConstraints(
    const MachineInstr &MI, unsigned OpIdx, const TargetInstrInfo &TII,
    const TargetRegisterInfo &TRI) const {
  // The mapping of the registers may be available via the
  // register class constraints.
  const TargetRegisterClass *RC = MI.getRegClassConstraint(OpIdx, &TII, &TRI);

  if (!RC)
    return nullptr;

  const RegisterBank &RegBank = getRegBankFromRegClass(*RC);
  // Sanity check that the target properly implemented getRegBankFromRegClass.
  assert(RegBank.covers(*RC) &&
         "The mapping of the register bank does not make sense");
  return &RegBank;
}

const TargetRegisterClass *RegisterBankInfo::constrainGenericRegister(
    unsigned Reg, const TargetRegisterClass &RC, MachineRegisterInfo &MRI) {

  // If the register already has a class, fallback to MRI::constrainRegClass.
  auto &RegClassOrBank = MRI.getRegClassOrRegBank(Reg);
  if (RegClassOrBank.is<const TargetRegisterClass *>())
    return MRI.constrainRegClass(Reg, &RC);

  const RegisterBank *RB = RegClassOrBank.get<const RegisterBank *>();
  assert(RB && "Generic register does not have a register bank");

  // Otherwise, all we can do is ensure the bank covers the class, and set it.
  if (!RB->covers(RC))
    return nullptr;

  MRI.setRegClass(Reg, &RC);
  return &RC;
}

RegisterBankInfo::InstructionMapping
RegisterBankInfo::getInstrMappingImpl(const MachineInstr &MI) const {
  RegisterBankInfo::InstructionMapping Mapping(DefaultMappingID, /*Cost*/ 1,
                                               MI.getNumOperands());
  const MachineFunction &MF = *MI.getParent()->getParent();
  const TargetSubtargetInfo &STI = MF.getSubtarget();
  const TargetRegisterInfo &TRI = *STI.getRegisterInfo();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  // We may need to query the instruction encoding to guess the mapping.
  const TargetInstrInfo &TII = *STI.getInstrInfo();

  // Before doing anything complicated check if the mapping is not
  // directly available.
  bool CompleteMapping = true;
  // For copies we want to walk over the operands and try to find one
  // that has a register bank.
  bool isCopyLike = MI.isCopy() || MI.isPHI();
  // Remember the register bank for reuse for copy-like instructions.
  const RegisterBank *RegBank = nullptr;
  // Remember the size of the register for reuse for copy-like instructions.
  unsigned RegSize = 0;
  for (unsigned OpIdx = 0, End = MI.getNumOperands(); OpIdx != End; ++OpIdx) {
    const MachineOperand &MO = MI.getOperand(OpIdx);
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;
    // The register bank of Reg is just a side effect of the current
    // excution and in particular, there is no reason to believe this
    // is the best default mapping for the current instruction.  Keep
    // it as an alternative register bank if we cannot figure out
    // something.
    const RegisterBank *AltRegBank = getRegBank(Reg, MRI, TRI);
    // For copy-like instruction, we want to reuse the register bank
    // that is already set on Reg, if any, since those instructions do
    // not have any constraints.
    const RegisterBank *CurRegBank = isCopyLike ? AltRegBank : nullptr;
    if (!CurRegBank) {
      // If this is a target specific instruction, we can deduce
      // the register bank from the encoding constraints.
      CurRegBank = getRegBankFromConstraints(MI, OpIdx, TII, TRI);
      if (!CurRegBank) {
        // All our attempts failed, give up.
        CompleteMapping = false;

        if (!isCopyLike)
          // MI does not carry enough information to guess the mapping.
          return InstructionMapping();

        // For copies, we want to keep interating to find a register
        // bank for the other operands if we did not find one yet.
        if (RegBank)
          break;
        continue;
      }
    }
    RegBank = CurRegBank;
    RegSize = getSizeInBits(Reg, MRI, TRI);
    Mapping.setOperandMapping(OpIdx, RegSize, *CurRegBank);
  }

  if (CompleteMapping)
    return Mapping;

  assert(isCopyLike && "We should have bailed on non-copies at this point");
  // For copy like instruction, if none of the operand has a register
  // bank avialable, there is nothing we can propagate.
  if (!RegBank)
    return InstructionMapping();

  // This is a copy-like instruction.
  // Propagate RegBank to all operands that do not have a
  // mapping yet.
  for (unsigned OpIdx = 0, End = MI.getNumOperands(); OpIdx != End; ++OpIdx) {
    const MachineOperand &MO = MI.getOperand(OpIdx);
    // Don't assign a mapping for non-reg operands.
    if (!MO.isReg())
      continue;

    // If a mapping already exists, do not touch it.
    if (!static_cast<const InstructionMapping *>(&Mapping)
             ->getOperandMapping(OpIdx)
             .BreakDown.empty())
      continue;

    Mapping.setOperandMapping(OpIdx, RegSize, *RegBank);
  }
  return Mapping;
}

RegisterBankInfo::InstructionMapping
RegisterBankInfo::getInstrMapping(const MachineInstr &MI) const {
    RegisterBankInfo::InstructionMapping Mapping = getInstrMappingImpl(MI);
    if (Mapping.isValid())
      return Mapping;
  llvm_unreachable("The target must implement this");
}

RegisterBankInfo::InstructionMappings
RegisterBankInfo::getInstrPossibleMappings(const MachineInstr &MI) const {
  InstructionMappings PossibleMappings;
  // Put the default mapping first.
  PossibleMappings.push_back(getInstrMapping(MI));
  // Then the alternative mapping, if any.
  InstructionMappings AltMappings = getInstrAlternativeMappings(MI);
  for (InstructionMapping &AltMapping : AltMappings)
    PossibleMappings.emplace_back(std::move(AltMapping));
#ifndef NDEBUG
  for (const InstructionMapping &Mapping : PossibleMappings)
    assert(Mapping.verify(MI) && "Mapping is invalid");
#endif
  return PossibleMappings;
}

RegisterBankInfo::InstructionMappings
RegisterBankInfo::getInstrAlternativeMappings(const MachineInstr &MI) const {
  // No alternative for MI.
  return InstructionMappings();
}

void RegisterBankInfo::applyDefaultMapping(const OperandsMapper &OpdMapper) {
  MachineInstr &MI = OpdMapper.getMI();
  DEBUG(dbgs() << "Applying default-like mapping\n");
  for (unsigned OpIdx = 0, EndIdx = MI.getNumOperands(); OpIdx != EndIdx;
       ++OpIdx) {
    DEBUG(dbgs() << "OpIdx " << OpIdx);
    MachineOperand &MO = MI.getOperand(OpIdx);
    if (!MO.isReg()) {
      DEBUG(dbgs() << " is not a register, nothing to be done\n");
      continue;
    }
    assert(
        OpdMapper.getInstrMapping().getOperandMapping(OpIdx).BreakDown.size() ==
            1 &&
        "This mapping is too complex for this function");
    iterator_range<SmallVectorImpl<unsigned>::const_iterator> NewRegs =
        OpdMapper.getVRegs(OpIdx);
    if (NewRegs.begin() == NewRegs.end()) {
      DEBUG(dbgs() << " has not been repaired, nothing to be done\n");
      continue;
    }
    DEBUG(dbgs() << " changed, replace " << MO.getReg());
    MO.setReg(*NewRegs.begin());
    DEBUG(dbgs() << " with " << MO.getReg());
  }
}

unsigned RegisterBankInfo::getSizeInBits(unsigned Reg,
                                         const MachineRegisterInfo &MRI,
                                         const TargetRegisterInfo &TRI) {
  const TargetRegisterClass *RC = nullptr;
  if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
    // The size is not directly available for physical registers.
    // Instead, we need to access a register class that contains Reg and
    // get the size of that register class.
    RC = TRI.getMinimalPhysRegClass(Reg);
  } else {
    unsigned RegSize = MRI.getSize(Reg);
    // If Reg is not a generic register, query the register class to
    // get its size.
    if (RegSize)
      return RegSize;
    // Since Reg is not a generic register, it must have a register class.
    RC = MRI.getRegClass(Reg);
  }
  assert(RC && "Unable to deduce the register class");
  return RC->getSize() * 8;
}

//------------------------------------------------------------------------------
// Helper classes implementation.
//------------------------------------------------------------------------------
void RegisterBankInfo::PartialMapping::dump() const {
  print(dbgs());
  dbgs() << '\n';
}

bool RegisterBankInfo::PartialMapping::verify() const {
  assert(RegBank && "Register bank not set");
  assert(Length && "Empty mapping");
  assert((StartIdx < getHighBitIdx()) && "Overflow, switch to APInt?");
  // Check if the minimum width fits into RegBank.
  assert(RegBank->getSize() >= Length && "Register bank too small for Mask");
  return true;
}

void RegisterBankInfo::PartialMapping::print(raw_ostream &OS) const {
  OS << "[" << StartIdx << ", " << getHighBitIdx() << "], RegBank = ";
  if (RegBank)
    OS << *RegBank;
  else
    OS << "nullptr";
}

bool RegisterBankInfo::ValueMapping::verify(unsigned ExpectedBitWidth) const {
  assert(!BreakDown.empty() && "Value mapped nowhere?!");
  unsigned OrigValueBitWidth = 0;
  for (const RegisterBankInfo::PartialMapping &PartMap : BreakDown) {
    // Check that each register bank is big enough to hold the partial value:
    // this check is done by PartialMapping::verify
    assert(PartMap.verify() && "Partial mapping is invalid");
    // The original value should completely be mapped.
    // Thus the maximum accessed index + 1 is the size of the original value.
    OrigValueBitWidth =
        std::max(OrigValueBitWidth, PartMap.getHighBitIdx() + 1);
  }
  assert(OrigValueBitWidth == ExpectedBitWidth && "BitWidth does not match");
  APInt ValueMask(OrigValueBitWidth, 0);
  for (const RegisterBankInfo::PartialMapping &PartMap : BreakDown) {
    // Check that the union of the partial mappings covers the whole value,
    // without overlaps.
    // The high bit is exclusive in the APInt API, thus getHighBitIdx + 1.
    APInt PartMapMask = APInt::getBitsSet(OrigValueBitWidth, PartMap.StartIdx,
                                          PartMap.getHighBitIdx() + 1);
    ValueMask ^= PartMapMask;
    assert((ValueMask & PartMapMask) == PartMapMask &&
           "Some partial mappings overlap");
  }
  assert(ValueMask.isAllOnesValue() && "Value is not fully mapped");
  return true;
}

void RegisterBankInfo::ValueMapping::dump() const {
  print(dbgs());
  dbgs() << '\n';
}

void RegisterBankInfo::ValueMapping::print(raw_ostream &OS) const {
  OS << "#BreakDown: " << BreakDown.size() << " ";
  bool IsFirst = true;
  for (const PartialMapping &PartMap : BreakDown) {
    if (!IsFirst)
      OS << ", ";
    OS << '[' << PartMap << ']';
    IsFirst = false;
  }
}

void RegisterBankInfo::InstructionMapping::setOperandMapping(
    unsigned OpIdx, unsigned MaskSize, const RegisterBank &RegBank) {
  // Build the value mapping.
  assert(MaskSize <= RegBank.getSize() && "Register bank is too small");

  // Create the mapping object.
  getOperandMapping(OpIdx).BreakDown.push_back(
      PartialMapping(0, MaskSize, RegBank));
}

bool RegisterBankInfo::InstructionMapping::verify(
    const MachineInstr &MI) const {
  // Check that all the register operands are properly mapped.
  // Check the constructor invariant.
  assert(NumOperands == MI.getNumOperands() &&
         "NumOperands must match, see constructor");
  assert(MI.getParent() && MI.getParent()->getParent() &&
         "MI must be connected to a MachineFunction");
  const MachineFunction &MF = *MI.getParent()->getParent();
  (void)MF;

  for (unsigned Idx = 0; Idx < NumOperands; ++Idx) {
    const MachineOperand &MO = MI.getOperand(Idx);
    const RegisterBankInfo::ValueMapping &MOMapping = getOperandMapping(Idx);
    (void)MOMapping;
    if (!MO.isReg()) {
      assert(MOMapping.BreakDown.empty() &&
             "We should not care about non-reg mapping");
      continue;
    }
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;
    // Register size in bits.
    // This size must match what the mapping expects.
    assert(MOMapping.verify(getSizeInBits(
               Reg, MF.getRegInfo(), *MF.getSubtarget().getRegisterInfo())) &&
           "Value mapping is invalid");
  }
  return true;
}

void RegisterBankInfo::InstructionMapping::dump() const {
  print(dbgs());
  dbgs() << '\n';
}

void RegisterBankInfo::InstructionMapping::print(raw_ostream &OS) const {
  OS << "ID: " << getID() << " Cost: " << getCost() << " Mapping: ";

  for (unsigned OpIdx = 0; OpIdx != NumOperands; ++OpIdx) {
    const ValueMapping &ValMapping = getOperandMapping(OpIdx);
    if (OpIdx)
      OS << ", ";
    OS << "{ Idx: " << OpIdx << " Map: " << ValMapping << '}';
  }
}

const int RegisterBankInfo::OperandsMapper::DontKnowIdx = -1;

RegisterBankInfo::OperandsMapper::OperandsMapper(
    MachineInstr &MI, const InstructionMapping &InstrMapping,
    MachineRegisterInfo &MRI)
    : MRI(MRI), MI(MI), InstrMapping(InstrMapping) {
  unsigned NumOpds = MI.getNumOperands();
  OpToNewVRegIdx.reset(new int[NumOpds]);
  std::fill(&OpToNewVRegIdx[0], &OpToNewVRegIdx[NumOpds],
            OperandsMapper::DontKnowIdx);
  assert(InstrMapping.verify(MI) && "Invalid mapping for MI");
}

iterator_range<SmallVectorImpl<unsigned>::iterator>
RegisterBankInfo::OperandsMapper::getVRegsMem(unsigned OpIdx) {
  assert(OpIdx < getMI().getNumOperands() && "Out-of-bound access");
  unsigned NumPartialVal =
      getInstrMapping().getOperandMapping(OpIdx).BreakDown.size();
  int StartIdx = OpToNewVRegIdx[OpIdx];

  if (StartIdx == OperandsMapper::DontKnowIdx) {
    // This is the first time we try to access OpIdx.
    // Create the cells that will hold all the partial values at the
    // end of the list of NewVReg.
    StartIdx = NewVRegs.size();
    OpToNewVRegIdx[OpIdx] = StartIdx;
    for (unsigned i = 0; i < NumPartialVal; ++i)
      NewVRegs.push_back(0);
  }
  SmallVectorImpl<unsigned>::iterator End =
      getNewVRegsEnd(StartIdx, NumPartialVal);

  return make_range(&NewVRegs[StartIdx], End);
}

SmallVectorImpl<unsigned>::const_iterator
RegisterBankInfo::OperandsMapper::getNewVRegsEnd(unsigned StartIdx,
                                                 unsigned NumVal) const {
  return const_cast<OperandsMapper *>(this)->getNewVRegsEnd(StartIdx, NumVal);
}
SmallVectorImpl<unsigned>::iterator
RegisterBankInfo::OperandsMapper::getNewVRegsEnd(unsigned StartIdx,
                                                 unsigned NumVal) {
  assert((NewVRegs.size() == StartIdx + NumVal ||
          NewVRegs.size() > StartIdx + NumVal) &&
         "NewVRegs too small to contain all the partial mapping");
  return NewVRegs.size() <= StartIdx + NumVal ? NewVRegs.end()
                                              : &NewVRegs[StartIdx + NumVal];
}

void RegisterBankInfo::OperandsMapper::createVRegs(unsigned OpIdx) {
  assert(OpIdx < getMI().getNumOperands() && "Out-of-bound access");
  iterator_range<SmallVectorImpl<unsigned>::iterator> NewVRegsForOpIdx =
      getVRegsMem(OpIdx);
  const SmallVectorImpl<PartialMapping> &PartMapList =
      getInstrMapping().getOperandMapping(OpIdx).BreakDown;
  SmallVectorImpl<PartialMapping>::const_iterator PartMap = PartMapList.begin();
  for (unsigned &NewVReg : NewVRegsForOpIdx) {
    assert(PartMap != PartMapList.end() && "Out-of-bound access");
    assert(NewVReg == 0 && "Register has already been created");
    NewVReg = MRI.createGenericVirtualRegister(PartMap->Length);
    MRI.setRegBank(NewVReg, *PartMap->RegBank);
    ++PartMap;
  }
}

void RegisterBankInfo::OperandsMapper::setVRegs(unsigned OpIdx,
                                                unsigned PartialMapIdx,
                                                unsigned NewVReg) {
  assert(OpIdx < getMI().getNumOperands() && "Out-of-bound access");
  assert(getInstrMapping().getOperandMapping(OpIdx).BreakDown.size() >
             PartialMapIdx &&
         "Out-of-bound access for partial mapping");
  // Make sure the memory is initialized for that operand.
  (void)getVRegsMem(OpIdx);
  assert(NewVRegs[OpToNewVRegIdx[OpIdx] + PartialMapIdx] == 0 &&
         "This value is already set");
  NewVRegs[OpToNewVRegIdx[OpIdx] + PartialMapIdx] = NewVReg;
}

iterator_range<SmallVectorImpl<unsigned>::const_iterator>
RegisterBankInfo::OperandsMapper::getVRegs(unsigned OpIdx,
                                           bool ForDebug) const {
  (void)ForDebug;
  assert(OpIdx < getMI().getNumOperands() && "Out-of-bound access");
  int StartIdx = OpToNewVRegIdx[OpIdx];

  if (StartIdx == OperandsMapper::DontKnowIdx)
    return make_range(NewVRegs.end(), NewVRegs.end());

  unsigned PartMapSize =
      getInstrMapping().getOperandMapping(OpIdx).BreakDown.size();
  SmallVectorImpl<unsigned>::const_iterator End =
      getNewVRegsEnd(StartIdx, PartMapSize);
  iterator_range<SmallVectorImpl<unsigned>::const_iterator> Res =
      make_range(&NewVRegs[StartIdx], End);
#ifndef NDEBUG
  for (unsigned VReg : Res)
    assert((VReg || ForDebug) && "Some registers are uninitialized");
#endif
  return Res;
}

void RegisterBankInfo::OperandsMapper::dump() const {
  print(dbgs(), true);
  dbgs() << '\n';
}

void RegisterBankInfo::OperandsMapper::print(raw_ostream &OS,
                                             bool ForDebug) const {
  unsigned NumOpds = getMI().getNumOperands();
  if (ForDebug) {
    OS << "Mapping for " << getMI() << "\nwith " << getInstrMapping() << '\n';
    // Print out the internal state of the index table.
    OS << "Populated indices (CellNumber, IndexInNewVRegs): ";
    bool IsFirst = true;
    for (unsigned Idx = 0; Idx != NumOpds; ++Idx) {
      if (OpToNewVRegIdx[Idx] != DontKnowIdx) {
        if (!IsFirst)
          OS << ", ";
        OS << '(' << Idx << ", " << OpToNewVRegIdx[Idx] << ')';
        IsFirst = false;
      }
    }
    OS << '\n';
  } else
    OS << "Mapping ID: " << getInstrMapping().getID() << ' ';

  OS << "Operand Mapping: ";
  // If we have a function, we can pretty print the name of the registers.
  // Otherwise we will print the raw numbers.
  const TargetRegisterInfo *TRI =
      getMI().getParent() && getMI().getParent()->getParent()
          ? getMI().getParent()->getParent()->getSubtarget().getRegisterInfo()
          : nullptr;
  bool IsFirst = true;
  for (unsigned Idx = 0; Idx != NumOpds; ++Idx) {
    if (OpToNewVRegIdx[Idx] == DontKnowIdx)
      continue;
    if (!IsFirst)
      OS << ", ";
    IsFirst = false;
    OS << '(' << PrintReg(getMI().getOperand(Idx).getReg(), TRI) << ", [";
    bool IsFirstNewVReg = true;
    for (unsigned VReg : getVRegs(Idx)) {
      if (!IsFirstNewVReg)
        OS << ", ";
      IsFirstNewVReg = false;
      OS << PrintReg(VReg, TRI);
    }
    OS << "])";
  }
}
