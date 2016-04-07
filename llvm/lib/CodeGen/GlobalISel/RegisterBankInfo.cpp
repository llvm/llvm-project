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
#include "llvm/CodeGen/GlobalISel/RegisterBankInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
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

/// Get the size in bits of the \p OpIdx-th operand of \p MI.
///
/// \pre \p MI is part of a basic block and this basic block is part
/// of a function.
static unsigned getSizeInBits(const MachineInstr &MI, unsigned OpIdx) {
  unsigned Reg = MI.getOperand(OpIdx).getReg();
  const TargetRegisterClass *RC = nullptr;
  if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
    const TargetSubtargetInfo &STI =
        MI.getParent()->getParent()->getSubtarget();
    const TargetRegisterInfo &TRI = *STI.getRegisterInfo();
    // The size is not directly available for physical registers.
    // Instead, we need to access a register class that contains Reg and
    // get the size of that register class.
    RC = TRI.getMinimalPhysRegClass(Reg);
  } else {
    const MachineRegisterInfo &MRI = MI.getParent()->getParent()->getRegInfo();
    unsigned RegSize = MRI.getSize(Reg);
    // If Reg is not a generic register, query the register class to
    // get its size.
    if (RegSize)
      return RegSize;
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
                                          const TargetRegisterInfo &TRI) {
  RegisterBank &RB = getRegBank(ID);
  unsigned NbOfRegClasses = TRI.getNumRegClasses();

  DEBUG(dbgs() << "Add coverage for: " << RB << '\n');

  // Check if RB is underconstruction.
  if (!RB.isValid())
    RB.ContainedRegClasses.resize(NbOfRegClasses);
  else if (RB.contains(*TRI.getRegClass(RCId)))
    // If RB already contains this register class, there is nothing
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
    const uint32_t *SubClassMask = CurRC.getSubClassMask();
    // The subclasses mask is broken down into chunks of uint32_t, but it still
    // represents all register classes.
    bool First = true;
    for (unsigned Base = 0; Base < NbOfRegClasses; Base += 32) {
      unsigned Idx = Base;
      for (uint32_t Mask = *SubClassMask++; Mask; Mask >>= 1, ++Idx) {
        unsigned Offset = countTrailingZeros(Mask);
        unsigned SubRCId = Idx + Offset;
        if (!Covered.test(SubRCId)) {
          if (First)
            DEBUG(dbgs() << "  Enqueue sub-class: ");
          DEBUG(dbgs() << TRI.getRegClassName(TRI.getRegClass(SubRCId))
                       << ", ");
          WorkList.push_back(SubRCId);
          // Remember that we saw the sub class.
          Covered.set(SubRCId);
          First = false;
        }

        // Move the cursor to the next sub class.
        // I.e., eat up the zeros then move to the next bit.
        // This last part is done as part of the loop increment.

        // By construction, Offset must be less than 32.
        // Otherwise, than means Mask was zero. I.e., no UB.
        Mask >>= Offset;
        // Remember that we shifted the base offset.
        Idx += Offset;
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
        const uint32_t *SuperRCMask = SuperRCIt.getMask();
        for (unsigned Base = 0; Base < NbOfRegClasses; Base += 32) {
          unsigned Idx = Base;
          for (uint32_t Mask = *SuperRCMask++; Mask; Mask >>= 1, ++Idx) {
            unsigned Offset = countTrailingZeros(Mask);
            unsigned SuperRCId = Idx + Offset;
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

            // Move the cursor to the next sub class.
            // I.e., eat up the zeros then move to the next bit.
            // This last part is done as part of the loop increment.

            // By construction, Offset must be less than 32.
            // Otherwise, than means Mask was zero. I.e., no UB.
            Mask >>= Offset;
            // Remember that we shifted the base offset.
            Idx += Offset;
          }
        }
      }
    }
    if (!First)
      DEBUG(dbgs() << '\n');
  } while (!WorkList.empty());
}

RegisterBankInfo::InstructionMapping
RegisterBankInfo::getInstrMapping(const MachineInstr &MI) const {
  if (MI.getOpcode() > TargetOpcode::GENERIC_OP_END) {
    // This is a target specific opcode:
    // The mapping of the registers is already available via the
    // register class.
    // Just map the register class to a register bank.
    RegisterBankInfo::InstructionMapping Mapping(DefaultMappingID, /*Cost*/ 1,
                                                 MI.getNumOperands());
    const TargetSubtargetInfo &STI =
        MI.getParent()->getParent()->getSubtarget();
    const TargetRegisterInfo &TRI = *STI.getRegisterInfo();
    const TargetInstrInfo &TII = *STI.getInstrInfo();

    for (unsigned OpIdx = 0, End = MI.getNumOperands(); OpIdx != End; ++OpIdx) {
      const MachineOperand &MO = MI.getOperand(OpIdx);
      if (!MO.getReg())
        continue;
      // Since this is a target instruction, the operand must have a register
      // class constraint.
      const TargetRegisterClass *RC =
          MI.getRegClassConstraint(OpIdx, &TII, &TRI);
      // Note: This cannot be a "dynamic" constraint like inline asm,
      //       since inlineasm opcode is a generic opcode.
      assert(RC && "Invalid encoding constraints for target instruction?");

      // Build the value mapping.
      const RegisterBank &RegBank = getRegBankFromRegClass(*RC);
      unsigned RegSize = getSizeInBits(MI, OpIdx);
      assert(RegSize <= RegBank.getSize() && "Register bank too small");
      // Assume the value is mapped in one register that lives in the
      // register bank that covers RC.
      APInt Mask(RegSize, 0);
      // The value is represented by all the bits.
      Mask.flipAllBits();

      // Create the mapping object.
      ValueMapping ValMapping;
      ValMapping.BreakDown.push_back(PartialMapping(Mask, RegBank));
      Mapping.setOperandMapping(OpIdx, ValMapping);
    }
    return Mapping;
  }
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
  Mask.toString(MaskStr, /*Radix*/ 2, /*Signed*/ 0, /*formatAsCLiteral*/ true);
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

void RegisterBankInfo::InstructionMapping::verify(
    const MachineInstr &MI) const {
  // Check that all the register operands are properly mapped.
  // Check the constructor invariant.
  assert(NumOperands == MI.getNumOperands() &&
         "NumOperands must match, see constructor");
  for (unsigned Idx = 0; Idx < NumOperands; ++Idx) {
    const MachineOperand &MO = MI.getOperand(Idx);
    const RegisterBankInfo::ValueMapping &MOMapping = getOperandMapping(Idx);
    if (!MO.isReg()) {
      assert(MOMapping.BreakDown.empty() &&
             "We should not care about non-reg mapping");
      continue;
    }
    // Register size in bits.
    // This size must match what the mapping expects.
    unsigned RegSize = getSizeInBits(MI, Idx);
    MOMapping.verify(RegSize);
  }
}
