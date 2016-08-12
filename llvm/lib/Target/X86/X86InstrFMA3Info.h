//===-- X86InstrFMA3Info.h - X86 FMA3 Instruction Information -------------===//
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
#include <set>

using namespace llvm;

/// This class is used to group {132, 213, 231} forms of FMA opcodes together.
/// Each of the groups has either 3 register opcodes, 3 memory opcodes,
/// or 6 register and memory opcodes. Also, each group has an attrubutes field
/// describing it.
class X86InstrFMA3Group {
private:
  /// Reference to an array holding 3 forms of register FMA opcodes.
  /// It may be set to nullptr if the group of FMA opcodes does not have
  /// any register form opcodes.
  const uint16_t *RegOpcodes;

  /// Reference to an array holding 3 forms of memory FMA opcodes.
  /// It may be set to nullptr if the group of FMA opcodes does not have
  /// any register form opcodes.
  const uint16_t *MemOpcodes;

  /// This bitfield specifies the attributes associated with the created
  /// FMA groups of opcodes.
  unsigned Attributes;

  static const unsigned Form132 = 0;
  static const unsigned Form213 = 1;
  static const unsigned Form231 = 2;

public:
  /// This bit must be set in the 'Attributes' field of FMA group if such
  /// group of FMA opcodes consists of FMA intrinsic opcodes.
  static const unsigned X86FMA3Intrinsic = 0x1;

  /// This bit must be set in the 'Attributes' field of FMA group if such
  /// group of FMA opcodes consists of AVX512 opcodes accepting a k-mask and
  /// passing the elements from the 1st operand to the result of the operation
  /// when the correpondings bits in the k-mask are unset.
  static const unsigned X86FMA3KMergeMasked = 0x2;

  /// This bit must be set in the 'Attributes' field of FMA group if such
  /// group of FMA opcodes consists of AVX512 opcodes accepting a k-zeromask.
  static const unsigned X86FMA3KZeroMasked = 0x4;

  /// Constructor. Creates a new group of FMA opcodes with three register form
  /// FMA opcodes \p RegOpcodes and three memory form FMA opcodes \p MemOpcodes.
  /// The parameters \p RegOpcodes and \p MemOpcodes may be set to nullptr,
  /// which means that the created group of FMA opcodes does not have the
  /// corresponding (register or memory) opcodes.
  /// The parameter \p Attr specifies the attributes describing the created
  /// group.
  X86InstrFMA3Group(const uint16_t *RegOpcodes, const uint16_t *MemOpcodes,
                    unsigned Attr)
      : RegOpcodes(RegOpcodes), MemOpcodes(MemOpcodes), Attributes(Attr) {
    assert((RegOpcodes || MemOpcodes) &&
           "Cannot create a group not having any opcodes.");
  }

  /// Returns a memory form opcode that is the equivalent of the given register
  /// form opcode \p RegOpcode. 0 is returned if the group does not have
  /// either register of memory opcodes.
  unsigned getMemOpcode(unsigned RegOpcode) const {
    if (!RegOpcodes || !MemOpcodes)
      return 0;
    for (unsigned Form = 0; Form < 3; Form++)
      if (RegOpcodes[Form] == RegOpcode)
        return MemOpcodes[Form];
    return 0;
  }

  /// Returns the 132 form of FMA register opcode.
  unsigned getReg132Opcode() const {
    assert(RegOpcodes && "The group does not have register opcodes.");
    return RegOpcodes[Form132];
  }

  /// Returns the 213 form of FMA register opcode.
  unsigned getReg213Opcode() const {
    assert(RegOpcodes && "The group does not have register opcodes.");
    return RegOpcodes[Form213];
  }

  /// Returns the 231 form of FMA register opcode.
  unsigned getReg231Opcode() const {
    assert(RegOpcodes && "The group does not have register opcodes.");
    return RegOpcodes[Form231];
  }

  /// Returns the 132 form of FMA memory opcode.
  unsigned getMem132Opcode() const {
    assert(MemOpcodes && "The group does not have memory opcodes.");
    return MemOpcodes[Form132];
  }

  /// Returns the 213 form of FMA memory opcode.
  unsigned getMem213Opcode() const {
    assert(MemOpcodes && "The group does not have memory opcodes.");
    return MemOpcodes[Form213];
  }

  /// Returns the 231 form of FMA memory opcode.
  unsigned getMem231Opcode() const {
    assert(MemOpcodes && "The group does not have memory opcodes.");
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
    if (!RegOpcodes)
      return false;
    for (unsigned Form = 0; Form < 3; Form++)
      if (Opcode == RegOpcodes[Form])
        return true;
    return false;
  }

  /// Returns true iff the given \p Opcode is a memory opcode from the
  /// groups of FMA opcodes.
  bool isMemOpcodeFromGroup(unsigned Opcode) const {
    if (!MemOpcodes)
      return false;
    for (unsigned Form = 0; Form < 3; Form++)
      if (Opcode == MemOpcodes[Form])
        return true;
    return false;
  }
};

/// This class provides information about all existing FMA3 opcodes
///
class X86InstrFMA3Info {
private:
  /// A map that is used to find the group of FMA opcodes using any FMA opcode
  /// from the group.
  DenseMap<unsigned, const X86InstrFMA3Group *> OpcodeToGroup;

  /// Creates groups of FMA opcodes and initializes Opcode-to-Group map.
  /// This method can be called many times, but the actual initialization is
  /// called only once.
  static void initGroupsOnce();

  /// Creates groups of FMA opcodes and initializes Opcode-to-Group map.
  /// This method must be called ONLY from initGroupsOnce(). Otherwise, such
  /// call is not thread safe.
  void initGroupsOnceImpl();

  /// Creates one group of FMA opcodes having the register opcodes
  /// \p RegOpcodes and memory opcodes \p MemOpcodes. The parameter \p Attr
  /// specifies the attributes describing the created group.
  void initRMGroup(const uint16_t *RegOpcodes,
                   const uint16_t *MemOpcodes, unsigned Attr = 0);

  /// Creates one group of FMA opcodes having only the register opcodes
  /// \p RegOpcodes. The parameter \p Attr specifies the attributes describing
  /// the created group.
  void initRGroup(const uint16_t *RegOpcodes, unsigned Attr = 0);

  /// Creates one group of FMA opcodes having only the memory opcodes
  /// \p MemOpcodes. The parameter \p Attr specifies the attributes describing
  /// the created group.
  void initMGroup(const uint16_t *MemOpcodes, unsigned Attr = 0);

public:
  /// Returns the reference to an object of this class. It is assumed that
  /// only one object may exist.
  static X86InstrFMA3Info *getX86InstrFMA3Info();

  /// Constructor. Just creates an object of the class.
  X86InstrFMA3Info() {}

  /// Destructor. Deallocates the memory used for FMA3 Groups.
  ~X86InstrFMA3Info() {
    std::set<const X86InstrFMA3Group *> DeletedGroups;
    auto E = OpcodeToGroup.end();
    for (auto I = OpcodeToGroup.begin(); I != E; I++) {
      const X86InstrFMA3Group *G = I->second;
      if (DeletedGroups.find(G) == DeletedGroups.end()) {
        DeletedGroups.insert(G);
        delete G;
      }
    }
  }

  /// Returns a reference to a group of FMA3 opcodes to where the given
  /// \p Opcode is included. If the given \p Opcode is not recognized as FMA3
  /// and not included into any FMA3 group, then nullptr is returned.
  static const X86InstrFMA3Group *getFMA3Group(unsigned Opcode) {
    // Ensure that the groups of opcodes are initialized.
    initGroupsOnce();

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

  /// Iterator that is used to walk on FMA register opcodes having memory
  /// form equivalents.
  class rm_iterator {
  private:
    /// Iterator associated with the OpcodeToGroup map. It must always be
    /// initialized with an entry from OpcodeToGroup for which I->first
    /// points to a register FMA opcode and I->second points to a group of
    /// FMA opcodes having memory form equivalent of I->first.
    DenseMap<unsigned, const X86InstrFMA3Group *>::const_iterator I;

  public:
    /// Constructor. Creates rm_iterator. The parameter \p I must be an
    /// iterator to OpcodeToGroup map entry having I->first pointing to
    /// register form FMA opcode and I->second pointing to a group of FMA
    /// opcodes holding memory form equivalent for I->fist.
    rm_iterator(DenseMap<unsigned, const X86InstrFMA3Group *>::const_iterator I)
        : I(I) {}

    /// Returns the register form FMA opcode.
    unsigned getRegOpcode() const { return I->first; };

    /// Returns the memory form equivalent opcode for FMA register opcode
    /// referenced by I->first.
    unsigned getMemOpcode() const {
      unsigned Opcode = I->first;
      const X86InstrFMA3Group *Group = I->second;
      return Group->getMemOpcode(Opcode);
    }

    /// Returns a reference to a group of FMA opcodes.
    const X86InstrFMA3Group *getGroup() const { return I->second; }

    bool operator==(const rm_iterator &OtherIt) const { return I == OtherIt.I; }
    bool operator!=(const rm_iterator &OtherIt) const { return I != OtherIt.I; }

    /// Increment. Advances the 'I' iterator to the next OpcodeToGroup entry
    /// having I->first pointing to register form FMA and I->second pointing
    /// to a group of FMA opcodes holding memory form equivalen for I->first.
    rm_iterator &operator++() {
      auto E = getX86InstrFMA3Info()->OpcodeToGroup.end();
      for (++I; I != E; ++I) {
        unsigned RegOpcode = I->first;
        const X86InstrFMA3Group *Group = I->second;
        if (Group->getMemOpcode(RegOpcode) != 0)
          break;
      }
      return *this;
    }
  };

  /// Returns rm_iterator pointing to the first entry of OpcodeToGroup map
  /// with a register FMA opcode having memory form opcode equivalent.
  static rm_iterator rm_begin() {
    initGroupsOnce();
    const X86InstrFMA3Info *FMA3Info = getX86InstrFMA3Info();
    auto I = FMA3Info->OpcodeToGroup.begin();
    auto E = FMA3Info->OpcodeToGroup.end();
    while (I != E) {
      unsigned Opcode = I->first;
      const X86InstrFMA3Group *G = I->second;
      if (G->getMemOpcode(Opcode) != 0)
        break;
      I++;
    }
    return rm_iterator(I);
  }

  /// Returns the last rm_iterator.
  static rm_iterator rm_end() {
    initGroupsOnce();
    return rm_iterator(getX86InstrFMA3Info()->OpcodeToGroup.end());
  }
};

#endif
