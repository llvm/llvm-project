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

#include "llvm/Support/DataTypes.h"

namespace llvm {

class Value;
class FoldingSetNodeID;
class MDNode;
class raw_ostream;

/// MachinePointerInfo - This class contains a discriminated union of
/// information about pointers in memory operands, relating them back to LLVM IR
/// or to virtual locations (such as frame indices) that are exposed during
/// codegen.
struct MachinePointerInfo {
  /// V - This is the IR pointer value for the access, or it is null if unknown.
  /// If this is null, then the access is to a pointer in the default address
  /// space.
  const Value *V;
  
  /// Offset - This is an offset from the base Value*.
  int64_t Offset;
  
  explicit MachinePointerInfo(const Value *v = 0, int64_t offset = 0)
    : V(v), Offset(offset) {}
  
  MachinePointerInfo getWithOffset(int64_t O) const {
    if (V == 0) return MachinePointerInfo(0, 0);
    return MachinePointerInfo(V, Offset+O);
  }
  
  /// getAddrSpace - Return the LLVM IR address space number that this pointer
  /// points into.
  unsigned getAddrSpace() const;
  
  /// getConstantPool - Return a MachinePointerInfo record that refers to the
  /// constant pool.
  static MachinePointerInfo getConstantPool();

  /// getFixedStack - Return a MachinePointerInfo record that refers to the
  /// the specified FrameIndex.
  static MachinePointerInfo getFixedStack(int FI, int64_t offset = 0);
  
  /// getJumpTable - Return a MachinePointerInfo record that refers to a
  /// jump table entry.
  static MachinePointerInfo getJumpTable();
  
  /// getGOT - Return a MachinePointerInfo record that refers to a
  /// GOT entry.
  static MachinePointerInfo getGOT();
  
  /// getStack - stack pointer relative access.
  static MachinePointerInfo getStack(int64_t Offset);
};
  
  
//===----------------------------------------------------------------------===//
/// MachineMemOperand - A description of a memory reference used in the backend.
/// Instead of holding a StoreInst or LoadInst, this class holds the address
/// Value of the reference along with a byte size and offset. This allows it
/// to describe lowered loads and stores. Also, the special PseudoSourceValue
/// objects can be used to represent loads and stores to memory locations
/// that aren't explicit in the regular LLVM IR.
///
class MachineMemOperand {
  MachinePointerInfo PtrInfo;
  uint64_t Size;
  unsigned Flags;
  const MDNode *TBAAInfo;
  const MDNode *Ranges;

public:
  /// Flags values. These may be or'd together.
  enum MemOperandFlags {
    /// The memory access reads data.
    MOLoad = 1,
    /// The memory access writes data.
    MOStore = 2,
    /// The memory access is volatile.
    MOVolatile = 4,
    /// The memory access is non-temporal.
    MONonTemporal = 8,
    /// The memory access is invariant.
    MOInvariant = 16,
    // This is the number of bits we need to represent flags.
    MOMaxBits = 5
  };

  /// MachineMemOperand - Construct an MachineMemOperand object with the
  /// specified PtrInfo, flags, size, and base alignment.
  MachineMemOperand(MachinePointerInfo PtrInfo, unsigned flags, uint64_t s,
                    unsigned base_alignment, const MDNode *TBAAInfo = 0,
                    const MDNode *Ranges = 0);

  const MachinePointerInfo &getPointerInfo() const { return PtrInfo; }
  
  /// getValue - Return the base address of the memory access. This may either
  /// be a normal LLVM IR Value, or one of the special values used in CodeGen.
  /// Special values are those obtained via
  /// PseudoSourceValue::getFixedStack(int), PseudoSourceValue::getStack, and
  /// other PseudoSourceValue member functions which return objects which stand
  /// for frame/stack pointer relative references and other special references
  /// which are not representable in the high-level IR.
  const Value *getValue() const { return PtrInfo.V; }

  /// getFlags - Return the raw flags of the source value, \see MemOperandFlags.
  unsigned int getFlags() const { return Flags & ((1 << MOMaxBits) - 1); }

  /// getOffset - For normal values, this is a byte offset added to the base
  /// address. For PseudoSourceValue::FPRel values, this is the FrameIndex
  /// number.
  int64_t getOffset() const { return PtrInfo.Offset; }

  /// getSize - Return the size in bytes of the memory reference.
  uint64_t getSize() const { return Size; }

  /// getAlignment - Return the minimum known alignment in bytes of the
  /// actual memory reference.
  uint64_t getAlignment() const;

  /// getBaseAlignment - Return the minimum known alignment in bytes of the
  /// base address, without the offset.
  uint64_t getBaseAlignment() const { return (1u << (Flags >> MOMaxBits)) >> 1; }

  /// getTBAAInfo - Return the TBAA tag for the memory reference.
  const MDNode *getTBAAInfo() const { return TBAAInfo; }

  /// getRanges - Return the range tag for the memory reference.
  const MDNode *getRanges() const { return Ranges; }

  bool isLoad() const { return Flags & MOLoad; }
  bool isStore() const { return Flags & MOStore; }
  bool isVolatile() const { return Flags & MOVolatile; }
  bool isNonTemporal() const { return Flags & MONonTemporal; }
  bool isInvariant() const { return Flags & MOInvariant; }

  /// isUnordered - Returns true if this memory operation doesn't have any
  /// ordering constraints other than normal aliasing. Volatile and atomic
  /// memory operations can't be reordered.
  ///
  /// Currently, we don't model the difference between volatile and atomic
  /// operations. They should retain their ordering relative to all memory
  /// operations.
  bool isUnordered() const { return !isVolatile(); }

  /// refineAlignment - Update this MachineMemOperand to reflect the alignment
  /// of MMO, if it has a greater alignment. This must only be used when the
  /// new alignment applies to all users of this MachineMemOperand.
  void refineAlignment(const MachineMemOperand *MMO);

  /// setValue - Change the SourceValue for this MachineMemOperand. This
  /// should only be used when an object is being relocated and all references
  /// to it are being updated.
  void setValue(const Value *NewSV) { PtrInfo.V = NewSV; }
  void setOffset(int64_t NewOffset) { PtrInfo.Offset = NewOffset; }

  /// Profile - Gather unique data for the object.
  ///
  void Profile(FoldingSetNodeID &ID) const;
};

raw_ostream &operator<<(raw_ostream &OS, const MachineMemOperand &MRO);

} // End llvm namespace

#endif
