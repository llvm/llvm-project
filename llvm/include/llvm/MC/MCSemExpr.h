//===- MCSemExpr.h - Semantic Level Expressions -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSEMEXPR_H
#define LLVM_MC_MCSEMEXPR_H

#include "llvm/MC/MCRegister.h"
#include <cassert>
#include <cstdint>

namespace llvm {

/// An address of the form Scale * Reg + Offset.
/// A Scale of 0 gives the constant Offset.
class MCSemAddrExpr {
  int64_t Scale;
  MCRegister Reg;
  int64_t Offset;

public:
  MCSemAddrExpr(int64_t Scale, MCRegister Reg, int64_t Offset)
      : Scale(Scale), Reg(Reg), Offset(Offset) {}

  /// \name Accessors
  /// @{

  int64_t getScale() const { return Scale; }
  MCRegister getReg() const { return Reg; }
  int64_t getOffset() const { return Offset; }

  /// @}
};

/// The value of a register, or the value in memory at an address.
class MCSemLeaf {
public:
  enum MCSemLeafType : uint8_t {
    Register, ///< The value of a register.
    Memory    ///< The value in memory at an address.
  };

private:
  MCSemLeafType Kind;
  union {
    MCRegister Reg;
    MCSemAddrExpr Addr;
  };

public:
  explicit MCSemLeaf(MCRegister Reg) : Kind(Register), Reg(Reg) {}
  explicit MCSemLeaf(MCSemAddrExpr Addr) : Kind(Memory), Addr(Addr) {}

  /// \name Accessors
  /// @{

  MCSemLeafType getKind() const { return Kind; }

  MCRegister getReg() const {
    assert(Kind == Register && "Not a register leaf");
    return Reg;
  }

  const MCSemAddrExpr &getAddr() const {
    assert(Kind == Memory && "Not a memory leaf");
    return Addr;
  }

  /// @}
};

/// A value of the form Scale * Leaf + Offset.
/// A Scale of 0 gives the constant Offset.
class MCSemExpr {
  int64_t Scale;
  MCSemLeaf Leaf;
  int64_t Offset;

public:
  MCSemExpr(int64_t Scale, MCSemLeaf Leaf, int64_t Offset)
      : Scale(Scale), Leaf(Leaf), Offset(Offset) {}

  /// \name Accessors
  /// @{

  int64_t getScale() const { return Scale; }
  const MCSemLeaf &getLeaf() const { return Leaf; }
  int64_t getOffset() const { return Offset; }

  /// @}
};

/// A store of Value to the memory at Addr.
struct MCSemStore {
  MCSemAddrExpr Addr;
  MCSemExpr Value;
};

} // end namespace llvm

#endif // LLVM_MC_MCSEMEXPR_H
