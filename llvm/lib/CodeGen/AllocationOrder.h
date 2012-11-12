//===-- llvm/CodeGen/AllocationOrder.h - Allocation Order -*- C++ -*-------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements an allocation order for virtual registers.
//
// The preferred allocation order for a virtual register depends on allocation
// hints and target hooks. The AllocationOrder class encapsulates all of that.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_ALLOCATIONORDER_H
#define LLVM_CODEGEN_ALLOCATIONORDER_H

namespace llvm {

class RegisterClassInfo;
class VirtRegMap;

class AllocationOrder {
  const unsigned *Begin;
  const unsigned *End;
  const unsigned *Pos;
  const RegisterClassInfo &RCI;
  unsigned Hint;
  bool OwnedBegin;
public:

  /// AllocationOrder - Create a new AllocationOrder for VirtReg.
  /// @param VirtReg      Virtual register to allocate for.
  /// @param VRM          Virtual register map for function.
  /// @param RegClassInfo Information about reserved and allocatable registers.
  AllocationOrder(unsigned VirtReg,
                  const VirtRegMap &VRM,
                  const RegisterClassInfo &RegClassInfo);

  ~AllocationOrder();

  /// next - Return the next physical register in the allocation order, or 0.
  /// It is safe to call next again after it returned 0.
  /// It will keep returning 0 until rewind() is called.
  unsigned next() {
    // First take the hint.
    if (!Pos) {
      Pos = Begin;
      if (Hint)
        return Hint;
    }
    // Then look at the order from TRI.
    while (Pos != End) {
      unsigned Reg = *Pos++;
      if (Reg != Hint)
        return Reg;
    }
    return 0;
  }

  /// rewind - Start over from the beginning.
  void rewind() { Pos = 0; }

  /// isHint - Return true if PhysReg is a preferred register.
  bool isHint(unsigned PhysReg) const { return PhysReg == Hint; }
};

} // end namespace llvm

#endif
