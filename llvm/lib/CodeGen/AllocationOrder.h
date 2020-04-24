//===-- llvm/CodeGen/AllocationOrder.h - Allocation Order -*- C++ -*-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an allocation order for virtual registers.
//
// The preferred allocation order for a virtual register depends on allocation
// hints and target hooks. The AllocationOrder class encapsulates all of that.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ALLOCATIONORDER_H
#define LLVM_LIB_CODEGEN_ALLOCATIONORDER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCRegister.h"

namespace llvm {

class RegisterClassInfo;
class VirtRegMap;
class LiveRegMatrix;

class LLVM_LIBRARY_VISIBILITY AllocationOrder {
  SmallVector<MCPhysReg, 16> Hints;
  ArrayRef<MCPhysReg> Order;
  int Pos;

  // If HardHints is true, *only* Hints will be returned.
  bool HardHints;

public:

  /// Create a new AllocationOrder for VirtReg.
  /// @param VirtReg      Virtual register to allocate for.
  /// @param VRM          Virtual register map for function.
  /// @param RegClassInfo Information about reserved and allocatable registers.
  AllocationOrder(unsigned VirtReg,
                  const VirtRegMap &VRM,
                  const RegisterClassInfo &RegClassInfo,
                  const LiveRegMatrix *Matrix);

  /// Get the allocation order without reordered hints.
  ArrayRef<MCPhysReg> getOrder() const { return Order; }

  /// Return the next physical register in the allocation order, or 0.
  /// It is safe to call next() again after it returned 0, it will keep
  /// returning 0 until rewind() is called.
  unsigned next(unsigned Limit = 0) {
    if (Pos < 0)
      return Hints.end()[Pos++];
    if (HardHints)
      return 0;
    if (!Limit)
      Limit = Order.size();
    while (Pos < int(Limit)) {
      unsigned Reg = Order[Pos++];
      if (!isHint(Reg))
        return Reg;
    }
    return 0;
  }

  /// As next(), but allow duplicates to be returned, and stop before the
  /// Limit'th register in the RegisterClassInfo allocation order.
  ///
  /// This can produce more than Limit registers if there are hints.
  unsigned nextWithDups(unsigned Limit) {
    if (Pos < 0)
      return Hints.end()[Pos++];
    if (HardHints)
      return 0;
    if (Pos < int(Limit))
      return Order[Pos++];
    return 0;
  }

  /// Start over from the beginning.
  void rewind() { Pos = -int(Hints.size()); }

  /// Return true if the last register returned from next() was a preferred register.
  bool isHint() const { return Pos <= 0; }

  /// Return true if PhysReg is a preferred register.
  bool isHint(unsigned PhysReg) const { return is_contained(Hints, PhysReg); }
};

} // end namespace llvm

#endif
