//===-- RegisterClassInfo.h - Dynamic Register Class Info -*- C++ -*-------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the RegisterClassInfo class which provides dynamic
// information about target register classes. Callee saved and reserved
// registers depends on calling conventions and other dynamic information, so
// some things cannot be determined statically.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGISTERCLASSINFO_H
#define LLVM_CODEGEN_REGISTERCLASSINFO_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Target/TargetRegisterInfo.h"

namespace llvm {

class RegisterClassInfo {
  struct RCInfo {
    unsigned Tag;
    unsigned NumRegs;
    bool ProperSubClass;
    OwningArrayPtr<unsigned> Order;

    RCInfo() : Tag(0), NumRegs(0), ProperSubClass(false) {}
    operator ArrayRef<unsigned>() const {
      return makeArrayRef(Order.get(), NumRegs);
    }
  };

  // Brief cached information for each register class.
  OwningArrayPtr<RCInfo> RegClass;

  // Tag changes whenever cached information needs to be recomputed. An RCInfo
  // entry is valid when its tag matches.
  unsigned Tag;

  const MachineFunction *MF;
  const TargetRegisterInfo *TRI;

  // Callee saved registers of last MF. Assumed to be valid until the next
  // runOnFunction() call.
  const uint16_t *CalleeSaved;

  // Map register number to CalleeSaved index + 1;
  SmallVector<uint8_t, 4> CSRNum;

  // Reserved registers in the current MF.
  BitVector Reserved;

  // Compute all information about RC.
  void compute(const TargetRegisterClass *RC) const;

  // Return an up-to-date RCInfo for RC.
  const RCInfo &get(const TargetRegisterClass *RC) const {
    const RCInfo &RCI = RegClass[RC->getID()];
    if (Tag != RCI.Tag)
      compute(RC);
    return RCI;
  }

public:
  RegisterClassInfo();

  /// runOnFunction - Prepare to answer questions about MF. This must be called
  /// before any other methods are used.
  void runOnMachineFunction(const MachineFunction &MF);

  /// getNumAllocatableRegs - Returns the number of actually allocatable
  /// registers in RC in the current function.
  unsigned getNumAllocatableRegs(const TargetRegisterClass *RC) const {
    return get(RC).NumRegs;
  }

  /// getOrder - Returns the preferred allocation order for RC. The order
  /// contains no reserved registers, and registers that alias callee saved
  /// registers come last.
  ArrayRef<unsigned> getOrder(const TargetRegisterClass *RC) const {
    return get(RC);
  }

  /// isProperSubClass - Returns true if RC has a legal super-class with more
  /// allocatable registers.
  ///
  /// Register classes like GR32_NOSP are not proper sub-classes because %esp
  /// is not allocatable.  Similarly, tGPR is not a proper sub-class in Thumb
  /// mode because the GPR super-class is not legal.
  bool isProperSubClass(const TargetRegisterClass *RC) const {
    return get(RC).ProperSubClass;
  }

  /// getLastCalleeSavedAlias - Returns the last callee saved register that
  /// overlaps PhysReg, or 0 if Reg doesn't overlap a CSR.
  unsigned getLastCalleeSavedAlias(unsigned PhysReg) const {
    assert(TargetRegisterInfo::isPhysicalRegister(PhysReg));
    if (unsigned N = CSRNum[PhysReg])
      return CalleeSaved[N-1];
    return 0;
  }
};
} // end namespace llvm

#endif

