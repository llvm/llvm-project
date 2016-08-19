//== llvm/CodeGen/GlobalISel/LowLevelType.h -------------------- -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// Implement a low-level type suitable for MachineInstr level instruction
/// selection.
///
/// For a type attached to a MachineInstr, we only care about 2 details: total
/// size and the number of vector lanes (if any). Accordingly, there are 4
/// possible valid type-kinds:
///
///    * `unsized` for labels etc
///    * `sN` for scalars and aggregates
///    * `<N x sM>` for vectors, which must have at least 2 elements.
///    * `pN` for pointers
///
/// Other information required for correct selection is expected to be carried
/// by the opcode, or non-type flags. For example the distinction between G_ADD
/// and G_FADD for int/float or fast-math flags.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_LOWLEVELTYPE_H
#define LLVM_CODEGEN_GLOBALISEL_LOWLEVELTYPE_H

#include <cassert>
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/CodeGen/ValueTypes.h"

namespace llvm {

class DataLayout;
class LLVMContext;
class Type;
class raw_ostream;

class LLT {
public:
  enum TypeKind : uint16_t {
    Invalid,
    Scalar,
    Pointer,
    Vector,
    Unsized,
  };

  /// Get a low-level scalar or aggregate "bag of bits".
  static LLT scalar(unsigned SizeInBits) {
    assert(SizeInBits > 0 && "invalid scalar size");
    return LLT{Scalar, 1, SizeInBits};
  }

  /// Get a low-level pointer in the given address space (defaulting to 0).
  static LLT pointer(unsigned AddressSpace) {
    return LLT{Pointer, 1, AddressSpace};
  }

  /// Get a low-level vector of some number of elements and element width.
  /// \p NumElements must be at least 2.
  static LLT vector(uint16_t NumElements, unsigned ScalarSizeInBits) {
    assert(NumElements > 1 && "invalid number of vector elements");
    return LLT{Vector, NumElements, ScalarSizeInBits};
  }

  /// Get a low-level vector of some number of elements and element type.
  static LLT vector(uint16_t NumElements, LLT ScalarTy) {
    assert(NumElements > 1 && "invalid number of vector elements");
    assert(ScalarTy.isScalar() && "invalid vector element type");
    return LLT{Vector, NumElements, ScalarTy.getSizeInBits()};
  }

  /// Get an unsized but valid low-level type (e.g. for a label).
  static LLT unsized() {
    return LLT{Unsized, 0, 0};
  }

  explicit LLT(TypeKind Kind, uint16_t NumElements, unsigned SizeOrAddrSpace)
    : SizeOrAddrSpace(SizeOrAddrSpace), NumElements(NumElements), Kind(Kind) {
    assert((Kind != Vector || NumElements > 1) &&
           "invalid number of vector elements");
  }

  explicit LLT() : SizeOrAddrSpace(0), NumElements(0), Kind(Invalid) {}

  /// Construct a low-level type based on an LLVM type.
  explicit LLT(Type &Ty, const DataLayout *DL = nullptr);

  bool isValid() const { return Kind != Invalid; }

  bool isScalar() const { return Kind == Scalar; }

  bool isPointer() const { return Kind == Pointer; }

  bool isVector() const { return Kind == Vector; }

  bool isSized() const { return Kind == Scalar || Kind == Vector; }

  /// Returns the number of elements in a vector LLT. Must only be called on
  /// vector types.
  uint16_t getNumElements() const {
    assert(isVector() && "cannot get number of elements on scalar/aggregate");
    return NumElements;
  }

  /// Returns the total size of the type. Must only be called on sized types.
  unsigned getSizeInBits() const {
    assert(isSized() && "attempt to get size of unsized type");
    return SizeOrAddrSpace * NumElements;
  }

  unsigned getScalarSizeInBits() const {
    assert(isSized() && "cannot get size of this type");
    return SizeOrAddrSpace;
  }

  unsigned getAddressSpace() const {
    assert(isPointer() && "cannot get address space of non-pointer type");
    return SizeOrAddrSpace;
  }

  /// Returns the vector's element type. Only valid for vector types.
  LLT getElementType() const {
    assert(isVector() && "cannot get element type of scalar/aggregate");
    return scalar(SizeOrAddrSpace);
  }

  /// Get a low-level type with half the size of the original, by halving the
  /// size of the scalar type involved. For example `s32` will become `s16`,
  /// `<2 x s32>` will become `<2 x s16>`.
  LLT halfScalarSize() const {
    assert(isSized() && getScalarSizeInBits() > 1 &&
           getScalarSizeInBits() % 2 == 0 && "cannot half size of this type");
    return LLT{Kind, NumElements, SizeOrAddrSpace / 2};
  }

  /// Get a low-level type with twice the size of the original, by doubling the
  /// size of the scalar type involved. For example `s32` will become `s64`,
  /// `<2 x s32>` will become `<2 x s64>`.
  LLT doubleScalarSize() const {
    assert(isSized() && "cannot change size of this type");
    return LLT{Kind, NumElements, SizeOrAddrSpace * 2};
  }

  /// Get a low-level type with half the size of the original, by halving the
  /// number of vector elements of the scalar type involved. The source must be
  /// a vector type with an even number of elements. For example `<4 x s32>`
  /// will become `<2 x s32>`, `<2 x s32>` will become `s32`.
  LLT halfElements() const {
    assert(isVector() && NumElements % 2 == 0 && "cannot half odd vector");
    if (NumElements == 2)
      return scalar(SizeOrAddrSpace);

    return LLT{Vector, static_cast<uint16_t>(NumElements / 2), SizeOrAddrSpace};
  }

  /// Get a low-level type with twice the size of the original, by doubling the
  /// number of vector elements of the scalar type involved. The source must be
  /// a vector type. For example `<2 x s32>` will become `<4 x s32>`. Doubling
  /// the number of elements in sN produces <2 x sN>.
  LLT doubleElements() const {
    return LLT{Vector, static_cast<uint16_t>(NumElements * 2), SizeOrAddrSpace};
  }

  void print(raw_ostream &OS) const;

  bool operator==(const LLT &RHS) const {
    return Kind == RHS.Kind && SizeOrAddrSpace == RHS.SizeOrAddrSpace &&
           NumElements == RHS.NumElements;
  }

  bool operator!=(const LLT &RHS) const { return !(*this == RHS); }

  friend struct DenseMapInfo<LLT>;
private:
  unsigned SizeOrAddrSpace;
  uint16_t NumElements;
  TypeKind Kind;
};

inline raw_ostream& operator<<(raw_ostream &OS, const LLT &Ty) {
  Ty.print(OS);
  return OS;
}

template<> struct DenseMapInfo<LLT> {
  static inline LLT getEmptyKey() {
    return LLT{LLT::Invalid, 0, -1u};
  }
  static inline LLT getTombstoneKey() {
    return LLT{LLT::Invalid, 0, -2u};
  }
  static inline unsigned getHashValue(const LLT &Ty) {
    uint64_t Val = ((uint64_t)Ty.SizeOrAddrSpace << 32) |
                   ((uint64_t)Ty.NumElements << 16) | (uint64_t)Ty.Kind;
    return DenseMapInfo<uint64_t>::getHashValue(Val);
  }
  static bool isEqual(const LLT &LHS, const LLT &RHS) {
    return LHS == RHS;
  }
};

}

#endif
