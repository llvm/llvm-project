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

#include "llvm/CodeGen/GlobalISel/RegisterBank.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/GlobalISel/RegisterBankInfo.h"
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

/// Get the size in bits of the \p Reg.
///
/// \pre \p Reg != 0 (NoRegister).
static unsigned getSizeInBits(unsigned Reg, const MachineRegisterInfo &MRI,
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
// RegisterBankInfo implementation.
//------------------------------------------------------------------------------
RegisterBankInfo::RegisterBankInfo(unsigned NumRegBanks)
    : NumRegBanks(NumRegBanks) {
  RegBanks.reset(new RegisterBank[NumRegBanks]);
}

void RegisterBankInfo::verify(const TargetRegisterInfo &TRI) const {
  for (unsigned Idx = 0, End = getNumRegBanks(); Idx != End; ++Idx) {
    const RegisterBank &RegBank = getRegBank(Idx);
    assert(Idx == RegBank.getID() &&
           "ID does not match the index in the array");
    DEBUG(dbgs() << "Verify " << RegBank << '\n');
    RegBank.verify(TRI);
  }
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
                                          const TargetRegisterInfo &TRI,
                                          bool AddTypeMapping) {
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

    // If we have been asked to record the type supported by this
    // register bank, do it now.
    if (AddTypeMapping)
      for (MVT::SimpleValueType SVT :
           make_range(CurRC.vt_begin(), CurRC.vt_end()))
        recordRegBankForType(getRegBank(ID), SVT);

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
  if (RegClassOrBank.is<const RegisterBank *>())
    return RegClassOrBank.get<const RegisterBank *>();
  const TargetRegisterClass *RC =
      RegClassOrBank.get<const TargetRegisterClass *>();
  if (RC)
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
        // Check if we can deduce the register bank from the type of
        // the instruction.
        Type *MITy = MI.getType();
        if (MITy)
          CurRegBank = getRegBankForType(
              MVT::getVT(MITy, /*HandleUnknown*/ true).SimpleTy);
        if (!CurRegBank)
          // Use the current assigned register bank.
          // That may not make much sense though.
          CurRegBank = AltRegBank;
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
    Mapping.verify(MI);
#endif
  return PossibleMappings;
}

RegisterBankInfo::InstructionMappings
RegisterBankInfo::getInstrAlternativeMappings(const MachineInstr &MI) const {
  // No alternative for MI.
  return InstructionMappings();
}

//------------------------------------------------------------------------------
// Helper classes implementation.
//------------------------------------------------------------------------------
void RegisterBankInfo::PartialMapping::dump() const {
  print(dbgs());
  dbgs() << '\n';
}

void RegisterBankInfo::PartialMapping::verify() const {
  assert(RegBank && "Register bank not set");
  // Check what is the minimum width that will live into RegBank.
  // RegBank will have to, at least, accomodate all the bits between the first
  // and last bits active in Mask.
  // If Mask is zero, then ActiveWidth is 0.
  unsigned ActiveWidth = 0;
  // Otherwise, remove the trailing and leading zeros from the bitwidth.
  // 0..0 ActiveWidth 0..0.
  if (Mask.getBoolValue())
    ActiveWidth = Mask.getBitWidth() - Mask.countLeadingZeros() -
                  Mask.countTrailingZeros();
  (void)ActiveWidth;
  assert(ActiveWidth <= Mask.getBitWidth() &&
         "Wrong computation of ActiveWidth, overflow?");
  assert(RegBank->getSize() >= ActiveWidth &&
         "Register bank too small for Mask");
}

void RegisterBankInfo::PartialMapping::print(raw_ostream &OS) const {
  SmallString<128> MaskStr;
  Mask.toString(MaskStr, /*Radix*/ 16, /*Signed*/ 0, /*formatAsCLiteral*/ true);
  OS << "Mask(" << Mask.getBitWidth() << ") = " << MaskStr << ", RegBank = ";
  if (RegBank)
    OS << *RegBank;
  else
    OS << "nullptr";
}

void RegisterBankInfo::ValueMapping::verify(unsigned ExpectedBitWidth) const {
  assert(!BreakDown.empty() && "Value mapped nowhere?!");
  unsigned ValueBitWidth = BreakDown.back().Mask.getBitWidth();
  assert(ValueBitWidth == ExpectedBitWidth && "BitWidth does not match");
  APInt ValueMask(ValueBitWidth, 0);
  for (const RegisterBankInfo::PartialMapping &PartMap : BreakDown) {
    // Check that all the partial mapping have the same bitwidth.
    assert(PartMap.Mask.getBitWidth() == ValueBitWidth &&
           "Value does not have the same size accross the partial mappings");
    // Check that the union of the partial mappings covers the whole value.
    ValueMask |= PartMap.Mask;
    // Check that each register bank is big enough to hold the partial value:
    // this check is done by PartialMapping::verify
    PartMap.verify();
  }
  assert(ValueMask.isAllOnesValue() && "Value is not fully mapped");
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
  APInt Mask(MaskSize, 0);
  // The value is represented by all the bits.
  Mask.flipAllBits();

  // Create the mapping object.
  getOperandMapping(OpIdx).BreakDown.push_back(PartialMapping(Mask, RegBank));
}

void RegisterBankInfo::InstructionMapping::verify(
    const MachineInstr &MI) const {
  // Check that all the register operands are properly mapped.
  // Check the constructor invariant.
  assert(NumOperands == MI.getNumOperands() &&
         "NumOperands must match, see constructor");
  assert(MI.getParent() && MI.getParent()->getParent() &&
         "MI must be connected to a MachineFunction");
  const MachineFunction &MF = *MI.getParent()->getParent();
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  for (unsigned Idx = 0; Idx < NumOperands; ++Idx) {
    const MachineOperand &MO = MI.getOperand(Idx);
    const RegisterBankInfo::ValueMapping &MOMapping = getOperandMapping(Idx);
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
    unsigned RegSize = getSizeInBits(Reg, MRI, TRI);
    MOMapping.verify(RegSize);
  }
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
