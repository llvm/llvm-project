//==- llvm/CodeGen/MachineMemOperand.h - MachineMemOperand class -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MachineMemOperand class, which is a
// description of a memory reference. It is used to help track dependencies
// in the backend.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEMEMOPERAND_H
#define LLVM_CODEGEN_MACHINEMEMOPERAND_H

#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Value.h"  // PointerLikeTypeTraits<Value*>
#include "llvm/Support/DataTypes.h"

namespace llvm {

class FoldingSetNodeID;
class MDNode;
class raw_ostream;
class MachineFunction;
class ModuleSlotTracker;

/// This class contains a discriminated union of information about pointers in
/// memory operands, relating them back to LLVM IR or to virtual locations (such
/// as frame indices) that are exposed during codegen.
struct MachinePointerInfo {
  /// This is the IR pointer value for the access, or it is null if unknown.
  /// If this is null, then the access is to a pointer in the default address
  /// space.
  PointerUnion<const Value *, const PseudoSourceValue *> V;

  /// Offset - This is an offset from the base Value*.
  int64_t Offset;

  explicit MachinePointerInfo(const Value *v = nullptr, int64_t offset = 0)
    : V(v), Offset(offset) {}

  explicit MachinePointerInfo(const PseudoSourceValue *v,
                              int64_t offset = 0)
    : V(v), Offset(offset) {}

  MachinePointerInfo getWithOffset(int64_t O) const {
    if (V.isNull()) return MachinePointerInfo();
    if (V.is<const Value*>())
      return MachinePointerInfo(V.get<const Value*>(), Offset+O);
    return MachinePointerInfo(V.get<const PseudoSourceValue*>(), Offset+O);
  }

  /// Return the LLVM IR address space number that this pointer points into.
  unsigned getAddrSpace() const;

  /// Return a MachinePointerInfo record that refers to the constant pool.
  static MachinePointerInfo getConstantPool(MachineFunction &MF);

  /// Return a MachinePointerInfo record that refers to the specified
  /// FrameIndex.
  static MachinePointerInfo getFixedStack(MachineFunction &MF, int FI,
                                          int64_t Offset = 0);

  /// Return a MachinePointerInfo record that refers to a jump table entry.
  static MachinePointerInfo getJumpTable(MachineFunction &MF);

  /// Return a MachinePointerInfo record that refers to a GOT entry.
  static MachinePointerInfo getGOT(MachineFunction &MF);

  /// Stack pointer relative access.
  static MachinePointerInfo getStack(MachineFunction &MF, int64_t Offset);
};


//===----------------------------------------------------------------------===//
/// A description of a memory reference used in the backend.
/// Instead of holding a StoreInst or LoadInst, this class holds the address
/// Value of the reference along with a byte size and offset. This allows it
/// to describe lowered loads and stores. Also, the special PseudoSourceValue
/// objects can be used to represent loads and stores to memory locations
/// that aren't explicit in the regular LLVM IR.
///
class MachineMemOperand {
public:
  /// Flags values. These may be or'd together.
  enum Flags : uint16_t {
    // No flags set.
    MONone = 0,
    /// The memory access reads data.
    MOLoad = 1u << 0,
    /// The memory access writes data.
    MOStore = 1u << 1,
    /// The memory access is volatile.
    MOVolatile = 1u << 2,
    /// The memory access is non-temporal.
    MONonTemporal = 1u << 3,
    /// The memory access is dereferenceable (i.e., doesn't trap).
    MODereferenceable = 1u << 4,
    /// The memory access always returns the same value (or traps).
    MOInvariant = 1u << 5,

    // Reserved for use by target-specific passes.
    MOTargetFlag1 = 1u << 6,
    MOTargetFlag2 = 1u << 7,
    MOTargetFlag3 = 1u << 8,

    LLVM_MARK_AS_BITMASK_ENUM(/* LargestFlag = */ MOTargetFlag3)
  };

private:
  MachinePointerInfo PtrInfo;
  uint64_t Size;
  Flags FlagVals;
  uint16_t BaseAlignLog2; // log_2(base_alignment) + 1
  AAMDNodes AAInfo;
  const MDNode *Ranges;

public:
  /// Construct a MachineMemOperand object with the specified PtrInfo, flags,
  /// size, and base alignment.
  MachineMemOperand(MachinePointerInfo PtrInfo, Flags flags, uint64_t s,
                    unsigned base_alignment,
                    const AAMDNodes &AAInfo = AAMDNodes(),
                    const MDNode *Ranges = nullptr);

  const MachinePointerInfo &getPointerInfo() const { return PtrInfo; }

  /// Return the base address of the memory access. This may either be a normal
  /// LLVM IR Value, or one of the special values used in CodeGen.
  /// Special values are those obtained via
  /// PseudoSourceValue::getFixedStack(int), PseudoSourceValue::getStack, and
  /// other PseudoSourceValue member functions which return objects which stand
  /// for frame/stack pointer relative references and other special references
  /// which are not representable in the high-level IR.
  const Value *getValue() const { return PtrInfo.V.dyn_cast<const Value*>(); }

  const PseudoSourceValue *getPseudoValue() const {
    return PtrInfo.V.dyn_cast<const PseudoSourceValue*>();
  }

  const void *getOpaqueValue() const { return PtrInfo.V.getOpaqueValue(); }

  /// Return the raw flags of the source value, \see Flags.
  Flags getFlags() const { return FlagVals; }

  /// Bitwise OR the current flags with the given flags.
  void setFlags(Flags f) { FlagVals |= f; }

  /// For normal values, this is a byte offset added to the base address.
  /// For PseudoSourceValue::FPRel values, this is the FrameIndex number.
  int64_t getOffset() const { return PtrInfo.Offset; }

  unsigned getAddrSpace() const { return PtrInfo.getAddrSpace(); }

  /// Return the size in bytes of the memory reference.
  uint64_t getSize() const { return Size; }

  /// Return the minimum known alignment in bytes of the actual memory
  /// reference.
  uint64_t getAlignment() const;

  /// Return the minimum known alignment in bytes of the base address, without
  /// the offset.
  uint64_t getBaseAlignment() const { return (1u << BaseAlignLog2) >> 1; }

  /// Return the AA tags for the memory reference.
  AAMDNodes getAAInfo() const { return AAInfo; }

  /// Return the range tag for the memory reference.
  const MDNode *getRanges() const { return Ranges; }

  bool isLoad() const { return FlagVals & MOLoad; }
  bool isStore() const { return FlagVals & MOStore; }
  bool isVolatile() const { return FlagVals & MOVolatile; }
  bool isNonTemporal() const { return FlagVals & MONonTemporal; }
  bool isDereferenceable() const { return FlagVals & MODereferenceable; }
  bool isInvariant() const { return FlagVals & MOInvariant; }

  /// Returns true if this memory operation doesn't have any ordering
  /// constraints other than normal aliasing. Volatile and atomic memory
  /// operations can't be reordered.
  ///
  /// Currently, we don't model the difference between volatile and atomic
  /// operations. They should retain their ordering relative to all memory
  /// operations.
  bool isUnordered() const { return !isVolatile(); }

  /// Update this MachineMemOperand to reflect the alignment of MMO, if it has a
  /// greater alignment. This must only be used when the new alignment applies
  /// to all users of this MachineMemOperand.
  void refineAlignment(const MachineMemOperand *MMO);

  /// Change the SourceValue for this MachineMemOperand. This should only be
  /// used when an object is being relocated and all references to it are being
  /// updated.
  void setValue(const Value *NewSV) { PtrInfo.V = NewSV; }
  void setValue(const PseudoSourceValue *NewSV) { PtrInfo.V = NewSV; }
  void setOffset(int64_t NewOffset) { PtrInfo.Offset = NewOffset; }

  /// Profile - Gather unique data for the object.
  ///
  void Profile(FoldingSetNodeID &ID) const;

  /// Support for operator<<.
  /// @{
  void print(raw_ostream &OS) const;
  void print(raw_ostream &OS, ModuleSlotTracker &MST) const;
  /// @}

  friend bool operator==(const MachineMemOperand &LHS,
                         const MachineMemOperand &RHS) {
    return LHS.getValue() == RHS.getValue() &&
           LHS.getPseudoValue() == RHS.getPseudoValue() &&
           LHS.getSize() == RHS.getSize() &&
           LHS.getOffset() == RHS.getOffset() &&
           LHS.getFlags() == RHS.getFlags() &&
           LHS.getAAInfo() == RHS.getAAInfo() &&
           LHS.getRanges() == RHS.getRanges() &&
           LHS.getAlignment() == RHS.getAlignment() &&
           LHS.getAddrSpace() == RHS.getAddrSpace();
  }

  friend bool operator!=(const MachineMemOperand &LHS,
                         const MachineMemOperand &RHS) {
    return !(LHS == RHS);
  }
};

inline raw_ostream &operator<<(raw_ostream &OS, const MachineMemOperand &MRO) {
  MRO.print(OS);
  return OS;
}

} // End llvm namespace

#endif
