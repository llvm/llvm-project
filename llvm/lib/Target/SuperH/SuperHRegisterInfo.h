//===-- SuperHRegisterInfo.h - SuperH Register Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SuperH implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SUPERH_SUPERHREGISTERINFO_H
#define LLVM_LIB_TARGET_SUPERH_SUPERHREGISTERINFO_H

#include "llvm/CodeGen/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "SuperHGenRegisterInfo.inc"

namespace llvm {
class SuperHSubtarget;

struct AddressRange {
  
  /// Start - Start of the address range.
  int32_t Start;

  /// End - End of the address range.
  int32_t End;

  /// getStride - Gets the stride (distance between start and end)
  int32_t getStride() { return End - Start; }
};

/// RelativeAddr - type which represents a SuperH relative address.
/// Relative memory access on SuperH is positive-only, as such
/// memory accesses are decomposed into a Base and an Offset.
struct RelativeAddr {

  /// Base - The base of the relative address, this base may
  /// be negative if the address is lower than the reference
  /// register's stored address.
  int32_t Base;

  /// Offset - The offset from the Base to read from, should
  /// generally always be a positive value.
  int32_t Offset;

  /// toAbsolute - Consolidates the address into an absolute
  /// offset from the reference point/register.
  int32_t toAbsolute() { return Base + Offset; }
};

/// AddressingMode - The addressing mode algorithm to use to
/// calculate the relative address of a memory access.
enum AddressingMode {

  /// ADDR_BUCKETED - Address is split up into buckets,
  /// addresses above the reference point are indexed by
  /// setting the offset to the enclosing bucket's end address
  /// and the offset is set to the distance from the end address
  /// to the real memory offset.
  ADDR_BUCKETED
};

class SuperHRegisterInfo : public SuperHGenRegisterInfo {
protected:
  const SuperHSubtarget &STI;

public:
  SuperHRegisterInfo(const SuperHSubtarget &STI);

  /// getStackBaseOffset - Gets the offset from the stack pointer
  /// to the base of the function's stack frame.
  int32_t getStackBaseOffset(const MachineFunction &MF) const;

  /// getStackAddress - Gets stack relative address for a given 
  /// machine function.
  RelativeAddr getStackAddress(const MachineFunction &MF, int32_t Offset, 
                               uint8_t Bits, uint8_t Scale) const;

  /// getRelativeAddress - Calculates a relative address from an absolute
  /// offset from the reference point/register.
  RelativeAddr getRelativeAddress(int32_t Base, int32_t Offset, uint8_t Bits, 
                                  uint8_t Scale, AddressingMode Mode = ADDR_BUCKETED) const;

  const MCPhysReg *getCalleeSavedRegs(const MachineFunction *MF) const override;
  const uint32_t *getCallPreservedMask(const MachineFunction &MF,
                                       CallingConv::ID CC) const override;
  BitVector getReservedRegs(const MachineFunction &MF) const override;
  Register getFrameRegister(const MachineFunction &MF) const override;

  bool eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS = nullptr) const override;

  bool requiresRegisterScavenging(const MachineFunction &MF) const override { return true; }

  // Helpers
  Register getFrameRegister() const;
  Register getStackRegister() const;
  Register getGOTRegister() const;
};

} // end namespace llvm

#endif // end LLVM_LIB_TARGET_SUPERH_SUPERHREGISTERINFO_H