//===- X86InstrFMA3Info.h - X86 FMA3 Instruction Information ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the classes providing information
// about existing X86 FMA3 opcodes, classifying and grouping them.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X86_UTILS_X86INSTRFMA3INFO_H
#define LLVM_LIB_TARGET_X86_UTILS_X86INSTRFMA3INFO_H

#include "X86.h"
#include "llvm/ADT/DenseMap.h"
#include <cassert>
#include <cstdint>
#include <set>

namespace llvm {

/// This class is used to group {132, 213, 231} forms of FMA opcodes together.
/// Each of the groups has either 3 register opcodes, 3 memory opcodes,
/// or 6 register and memory opcodes. Also, each group has an attrubutes field
/// describing it.
struct X86InstrFMA3Group {
  /// An array holding 3 forms of register FMA opcodes.
  /// Entries will be 0 if there are no register opcodes in the group.
  uint16_t RegOpcodes[3];

  /// An array holding 3 forms of memory FMA opcodes.
  /// Entries will be 0 if there are no register opcodes in the group.
  uint16_t MemOpcodes[3];

  /// This bitfield specifies the attributes associated with the created
  /// FMA groups of opcodes.
  uint16_t Attributes;

  enum {
    Form132,
    Form213,
    Form231,
  };

  enum : uint16_t {
    /// This bit must be set in the 'Attributes' field of FMA group if such
    /// group of FMA opcodes consists of FMA intrinsic opcodes.
    X86FMA3Intrinsic = 0x1,

    /// This bit must be set in the 'Attributes' field of FMA group if such
    /// group of FMA opcodes consists of AVX512 opcodes accepting a k-mask and
    /// passing the elements from the 1st operand to the result of the operation
    /// when the correpondings bits in the k-mask are unset.
    X86FMA3KMergeMasked = 0x2,

    /// This bit must be set in the 'Attributes' field of FMA group if such
    /// group of FMA opcodes consists of AVX512 opcodes accepting a k-zeromask.
    X86FMA3KZeroMasked = 0x4,
  };

  /// Returns a memory form opcode that is the equivalent of the given register
  /// form opcode \p RegOpcode. 0 is returned if the group does not have
  /// either register of memory opcodes.
  unsigned getMemOpcode(unsigned RegOpcode) const {
    for (unsigned Form = 0; Form < 3; Form++)
      if (RegOpcodes[Form] == RegOpcode)
        return MemOpcodes[Form];
    return 0;
  }

  /// Returns the 132 form of FMA register opcode.
  unsigned getReg132Opcode() const {
    assert(RegOpcodes[Form132] && "The group does not have register opcodes.");
    return RegOpcodes[Form132];
  }

  /// Returns the 213 form of FMA register opcode.
  unsigned getReg213Opcode() const {
    assert(RegOpcodes[Form213] && "The group does not have register opcodes.");
    return RegOpcodes[Form213];
  }

  /// Returns the 231 form of FMA register opcode.
  unsigned getReg231Opcode() const {
    assert(RegOpcodes[Form231] && "The group does not have register opcodes.");
    return RegOpcodes[Form231];
  }

  /// Returns the 132 form of FMA memory opcode.
  unsigned getMem132Opcode() const {
    assert(MemOpcodes[Form132] && "The group does not have memory opcodes.");
    return MemOpcodes[Form132];
  }

  /// Returns the 213 form of FMA memory opcode.
  unsigned getMem213Opcode() const {
    assert(MemOpcodes[Form213] && "The group does not have memory opcodes.");
    return MemOpcodes[Form213];
  }

  /// Returns the 231 form of FMA memory opcode.
  unsigned getMem231Opcode() const {
    assert(MemOpcodes[Form231] && "The group does not have memory opcodes.");
    return MemOpcodes[Form231];
  }

  /// Returns true iff the group of FMA opcodes holds intrinsic opcodes.
  bool isIntrinsic() const { return (Attributes & X86FMA3Intrinsic) != 0; }

  /// Returns true iff the group of FMA opcodes holds k-merge-masked opcodes.
  bool isKMergeMasked() const {
    return (Attributes & X86FMA3KMergeMasked) != 0;
  }

  /// Returns true iff the group of FMA opcodes holds k-zero-masked opcodes.
  bool isKZeroMasked() const { return (Attributes & X86FMA3KZeroMasked) != 0; }

  /// Returns true iff the group of FMA opcodes holds any of k-masked opcodes.
  bool isKMasked() const {
    return (Attributes & (X86FMA3KMergeMasked | X86FMA3KZeroMasked)) != 0;
  }

  /// Returns true iff the given \p Opcode is a register opcode from the
  /// groups of FMA opcodes.
  bool isRegOpcodeFromGroup(unsigned Opcode) const {
    for (unsigned Form = 0; Form < 3; Form++)
      if (Opcode == RegOpcodes[Form])
        return true;
    return false;
  }

  /// Returns true iff the given \p Opcode is a memory opcode from the
  /// groups of FMA opcodes.
  bool isMemOpcodeFromGroup(unsigned Opcode) const {
    for (unsigned Form = 0; Form < 3; Form++)
      if (Opcode == MemOpcodes[Form])
        return true;
    return false;
  }
};

/// This class provides information about all existing FMA3 opcodes
///
class X86InstrFMA3Info final {
private:
  /// A map that is used to find the group of FMA opcodes using any FMA opcode
  /// from the group.
  DenseMap<unsigned, const X86InstrFMA3Group *> OpcodeToGroup;

public:
  /// Returns the reference to an object of this class. It is assumed that
  /// only one object may exist.
  static X86InstrFMA3Info *getX86InstrFMA3Info();

  /// Constructor. Just creates an object of the class.
  X86InstrFMA3Info();

  /// Returns a reference to a group of FMA3 opcodes to where the given
  /// \p Opcode is included. If the given \p Opcode is not recognized as FMA3
  /// and not included into any FMA3 group, then nullptr is returned.
  static const X86InstrFMA3Group *getFMA3Group(unsigned Opcode) {
    // Find the group including the given opcode.
    const X86InstrFMA3Info *FMA3Info = getX86InstrFMA3Info();
    auto I = FMA3Info->OpcodeToGroup.find(Opcode);
    if (I == FMA3Info->OpcodeToGroup.end())
      return nullptr;

    return I->second;
  }

  /// Returns true iff the given \p Opcode is recognized as FMA3 by this class.
  static bool isFMA3(unsigned Opcode) {
    return getFMA3Group(Opcode) != nullptr;
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_X86_UTILS_X86INSTRFMA3INFO_H
