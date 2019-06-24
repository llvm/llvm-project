//===-- llvm/CodeGen/Register.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGISTER_H
#define LLVM_CODEGEN_REGISTER_H

#include <cassert>

namespace llvm {

/// Wrapper class representing virtual and physical registers. Should be passed
/// by value.
class Register {
  unsigned Reg;

public:
  Register(unsigned Val = 0): Reg(Val) {}

  /// Return true if the specified register number is in the virtual register
  /// namespace.
  bool isVirtual() const {
    return int(Reg) < 0;
  }

  /// Return true if the specified register number is in the physical register
  /// namespace.
  bool isPhysical() const {
    return int(Reg) > 0;
  }

  /// Convert a virtual register number to a 0-based index. The first virtual
  /// register in a function will get the index 0.
  unsigned virtRegIndex() const {
    assert(isVirtual() && "Not a virtual register");
    return Reg & ~(1u << 31);
  }

  /// Convert a 0-based index to a virtual register number.
  /// This is the inverse operation of VirtReg2IndexFunctor below.
  static Register index2VirtReg(unsigned Index) {
    return Register(Index | (1u << 31));
  }

  operator unsigned() const {
    return Reg;
  }

  bool isValid() const {
    return Reg != 0;
  }
};

}

#endif
