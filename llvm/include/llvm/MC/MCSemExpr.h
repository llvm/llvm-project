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
#include "llvm/Support/Compiler.h"
#include <cassert>
#include <cstdint>

namespace llvm {
class MCRegisterInfo;
class raw_ostream;

/// Affine address expression: A * Reg + B
/// Canonical form for unambiguous constant addresses:
/// A == 0 <=> Reg == MCRegister()
class MCSemAddrExpr {
  int64_t A;
  MCRegister Reg;
  int64_t B;

  MCSemAddrExpr(int64_t A, MCRegister Reg, int64_t B) : A(A), Reg(Reg), B(B) {}

public:
  static MCSemAddrExpr createConst(int64_t B) { return {0, MCRegister(), B}; }
  static MCSemAddrExpr createReg(int64_t A, MCRegister Reg, int64_t B) {
    assert(Reg.isValid() && A != 0 && "non-canonical MCSemAddrExpr");
    return {A, Reg, B};
  }

  int64_t getScale() const { return A; }
  /// Returns base register or MCRegister() if constant.
  MCRegister getReg() const { return Reg; }
  int64_t getOffset() const { return B; }
  bool isConstant() const { return A == 0; }

  bool operator==(const MCSemAddrExpr &RHS) const {
    return A == RHS.A && Reg == RHS.Reg && B == RHS.B;
  }
  bool operator!=(const MCSemAddrExpr &RHS) const { return !(*this == RHS); }

  LLVM_ABI void print(raw_ostream &OS,
                      const MCRegisterInfo *MRI = nullptr) const;
  LLVM_ABI void dump() const;
};

/// Leaf of a semantic expression must be either a register value or a memory
/// dereference.
class MCSemLeaf {
  enum MCSemLeafType : uint8_t { Register, Memory };

  MCSemLeafType Kind;
  /// Register or memory dereference type
  MCRegister Reg;
  MCSemAddrExpr Addr;

  MCSemLeaf(MCSemLeafType Kind, MCRegister Reg, MCSemAddrExpr Addr)
      : Kind(Kind), Reg(Reg), Addr(Addr) {}

public:
  static MCSemLeaf createReg(MCRegister Reg) {
    return {Register, Reg, MCSemAddrExpr::createConst(0)};
  }
  static MCSemLeaf createMem(MCSemAddrExpr Addr) {
    return {Memory, MCRegister(), Addr};
  }

  bool isReg() const { return Kind == Register; }
  bool isMem() const { return Kind == Memory; }

  MCRegister getReg() const {
    assert(isReg() && "not a register leaf");
    return Reg;
  }
  const MCSemAddrExpr &getAddr() const {
    assert(isMem() && "not a memory leaf");
    return Addr;
  }

  bool operator==(const MCSemLeaf &RHS) const {
    if (Kind != RHS.Kind)
      return false;
    return isReg() ? Reg == RHS.Reg : Addr == RHS.Addr;
  }
  bool operator!=(const MCSemLeaf &RHS) const { return !(*this == RHS); }

  LLVM_ABI void print(raw_ostream &OS,
                      const MCRegisterInfo *MRI = nullptr) const;
  LLVM_ABI void dump() const;
};

/// Affine semantic value A * Leaf + B
/// Canonical form: A == 0 <=> Leaf == MCSemLeaf::createReg(MCRegister())
/// To prevent ambiguous expressions for the same constant.
class MCSemExpr {
  int64_t A;
  MCSemLeaf Leaf;
  int64_t B;

  MCSemExpr(int64_t A, MCSemLeaf Leaf, int64_t B) : A(A), Leaf(Leaf), B(B) {}

public:
  static MCSemExpr createConst(int64_t B) {
    return {0, MCSemLeaf::createReg(MCRegister()), B};
  }
  static MCSemExpr createReg(int64_t A, MCRegister Reg, int64_t B) {
    assert(Reg.isValid() && A != 0 && "non-canonical MCSemExpr");
    return {A, MCSemLeaf::createReg(Reg), B};
  }
  static MCSemExpr createMem(int64_t A, MCSemAddrExpr Addr, int64_t B) {
    assert(A != 0 && "non-canonical MCSemExpr");
    return {A, MCSemLeaf::createMem(Addr), B};
  }

  int64_t getScale() const { return A; }
  const MCSemLeaf &getLeaf() const { return Leaf; }
  int64_t getOffset() const { return B; }
  bool isConstant() const { return A == 0; }

  bool operator==(const MCSemExpr &RHS) const {
    return A == RHS.A && B == RHS.B && Leaf == RHS.Leaf;
  }
  bool operator!=(const MCSemExpr &RHS) const { return !(*this == RHS); }

  LLVM_ABI void print(raw_ostream &OS,
                      const MCRegisterInfo *MRI = nullptr) const;
  LLVM_ABI void dump() const;
};

} // namespace llvm
#endif