//===- llvm/CodeGen/GlobalISel/GenericMachineInstrs.h -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Declares convenience wrapper classes for interpreting MachineInstr instances
/// as specific generic operations.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_GENERICMACHINEINSTRS_H
#define LLVM_CODEGEN_GLOBALISEL_GENERICMACHINEINSTRS_H

#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/Support/Casting.h"

namespace llvm {

/// A base class for all GenericMachineInstrs.
class GenericMachineInstr : public MachineInstr {
public:
  GenericMachineInstr() = delete;

  /// Access the Idx'th operand as a register and return it.
  /// This assumes that the Idx'th operand is a Register type.
  Register getReg(unsigned Idx) { return getOperand(Idx).getReg(); }

  static bool classof(const MachineInstr *MI) {
    return isPreISelGenericOpcode(MI->getOpcode());
  }
};

/// Represents any type of generic load or store.
/// G_LOAD, G_STORE, G_ZEXTLOAD, G_SEXTLOAD.
class GLoadStore : public GenericMachineInstr {
public:
  /// Get the source register of the pointer value.
  Register getPointerReg() const { return getOperand(1).getReg(); }

  /// Get the MachineMemOperand on this instruction.
  MachineMemOperand &getMMO() const { return **memoperands_begin(); }

  /// Returns true if the attached MachineMemOperand  has the atomic flag set.
  bool isAtomic() const { return getMMO().isAtomic(); }
  /// Returns true if the attached MachineMemOpeand as the volatile flag set.
  bool isVolatile() const { return getMMO().isVolatile(); }
  /// Returns true if the memory operation is neither atomic or volatile.
  bool isSimple() const { return !isAtomic() && !isVolatile(); }
  /// Returns true if this memory operation doesn't have any ordering
  /// constraints other than normal aliasing. Volatile and (ordered) atomic
  /// memory operations can't be reordered.
  bool isUnordered() const { return getMMO().isUnordered(); }

  /// Returns the size in bytes of the memory access.
  uint64_t getMemSize() { return getMMO().getSize();
  } /// Returns the size in bits of the memory access.
  uint64_t getMemSizeInBits() { return getMMO().getSizeInBits(); }

  static bool classof(const MachineInstr *MI) {
    switch (MI->getOpcode()) {
    case TargetOpcode::G_LOAD:
    case TargetOpcode::G_STORE:
    case TargetOpcode::G_ZEXTLOAD:
    case TargetOpcode::G_SEXTLOAD:
      return true;
    default:
      return false;
    }
  }
};

/// Represents any generic load, including sign/zero extending variants.
class GAnyLoad : public GLoadStore {
public:
  /// Get the definition register of the loaded value.
  Register getDstReg() const { return getOperand(0).getReg(); }

  static bool classof(const MachineInstr *MI) {
    switch (MI->getOpcode()) {
    case TargetOpcode::G_LOAD:
    case TargetOpcode::G_ZEXTLOAD:
    case TargetOpcode::G_SEXTLOAD:
      return true;
    default:
      return false;
    }
  }
};

/// Represents a G_LOAD.
class GLoad : public GAnyLoad {
public:
  static bool classof(const MachineInstr *MI) {
    return MI->getOpcode() == TargetOpcode::G_LOAD;
  }
};

/// Represents either a G_SEXTLOAD or G_ZEXTLOAD.
class GExtLoad : public GAnyLoad {
public:
  static bool classof(const MachineInstr *MI) {
    return MI->getOpcode() == TargetOpcode::G_SEXTLOAD ||
           MI->getOpcode() == TargetOpcode::G_ZEXTLOAD;
  }
};

/// Represents a G_SEXTLOAD.
class GSExtLoad : public GExtLoad {
public:
  static bool classof(const MachineInstr *MI) {
    return MI->getOpcode() == TargetOpcode::G_SEXTLOAD;
  }
};

/// Represents a G_ZEXTLOAD.
class GZExtLoad : public GExtLoad {
public:
  static bool classof(const MachineInstr *MI) {
    return MI->getOpcode() == TargetOpcode::G_ZEXTLOAD;
  }
};

/// Represents a G_STORE.
class GStore : public GLoadStore {
public:
  /// Get the stored value register.
  Register getValueReg() const { return getOperand(0).getReg(); }

  static bool classof(const MachineInstr *MI) {
    return MI->getOpcode() == TargetOpcode::G_STORE;
  }
};

} // namespace llvm

#endif // LLVM_CODEGEN_GLOBALISEL_GENERICMACHINEINSTRS_H