//===- llvm/DataLayout.h - Data size & alignment info -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines layout properties related to datatype size/offset/alignment
// information.  It uses lazy annotations to cache information about how
// structure types are laid out and used.
//
// This structure should be created once, filled in if the defaults are not
// correct and then passed around by const&.  None of the members functions
// require modification to the object.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_DATALAYOUT_H
#define LLVM_IR_DATALAYOUT_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TrailingObjects.h"
#include "llvm/Support/TypeSize.h"
#include <cassert>
#include <cstdint>
#include <string>

// This needs to be outside of the namespace, to avoid conflict with llvm-c
// decl.
using LLVMTargetDataRef = struct LLVMOpaqueTargetData *;

namespace llvm {

class GlobalVariable;
class LLVMContext;
class StructLayout;
class Triple;
class Value;

// FIXME: Currently the DataLayout string carries a "preferred alignment"
// for types. As the DataLayout is module/global, this should likely be
// sunk down to an FTTI element that is queried rather than a global
// preference.

/// A parsed version of the target data layout string in and methods for
/// querying it.
///
/// The target data layout string is specified *by the target* - a frontend
/// generating LLVM IR is required to generate the right target data for the
/// target being codegen'd to.
class DataLayout {
public:
  /// Primitive type specification.
  struct PrimitiveSpec {
    uint32_t BitWidth;
    Align ABIAlign;
    Align PrefAlign;

    LLVM_ABI bool operator==(const PrimitiveSpec &Other) const;
  };

  /// Pointer type specification.
  struct PointerSpec {
    uint32_t AddrSpace;
    uint32_t BitWidth;
    Align ABIAlign;
    Align PrefAlign;
    uint32_t IndexBitWidth;
    /// Pointers in this address space don't have a well-defined bitwise
    /// representation (e.g. may be relocated by a copying garbage collector).
    /// Additionally, they may also be non-integral (i.e. containing additional
    /// metadata such as bounds information/permissions).
    bool IsNonIntegral;
    LLVM_ABI bool operator==(const PointerSpec &Other) const;
  };

  enum class FunctionPtrAlignType {
    /// The function pointer alignment is independent of the function alignment.
    Independent,
    /// The function pointer alignment is a multiple of the function alignment.
    MultipleOfFunctionAlign,
  };

private:
  bool BigEndian = false;

  unsigned AllocaAddrSpace = 0;
  unsigned ProgramAddrSpace = 0;
  unsigned DefaultGlobalsAddrSpace = 0;

  MaybeAlign StackNaturalAlign;
  MaybeAlign FunctionPtrAlign;
  FunctionPtrAlignType TheFunctionPtrAlignType =
      FunctionPtrAlignType::Independent;

  enum ManglingModeT {
    MM_None,
    MM_ELF,
    MM_MachO,
    MM_WinCOFF,
    MM_WinCOFFX86,
    MM_GOFF,
    MM_Mips,
    MM_XCOFF
  };
  ManglingModeT ManglingMode = MM_None;

  // FIXME: `unsigned char` truncates the value parsed by `parseSpecifier`.
  SmallVector<unsigned char, 8> LegalIntWidths;

  /// Primitive type specifications. Sorted and uniqued by type bit width.
  SmallVector<PrimitiveSpec, 6> IntSpecs;
  SmallVector<PrimitiveSpec, 4> FloatSpecs;
  SmallVector<PrimitiveSpec, 10> VectorSpecs;

  /// Pointer type specifications. Sorted and uniqued by address space number.
  SmallVector<PointerSpec, 8> PointerSpecs;

  /// The string representation used to create this DataLayout
  std::string StringRepresentation;

  /// Struct type ABI and preferred alignments. The default spec is "a:8:64".
  Align StructABIAlignment = Align::Constant<1>();
  Align StructPrefAlignment = Align::Constant<8>();

  // The StructType -> StructLayout map.
  mutable void *LayoutMap = nullptr;

  /// Sets or updates the specification for the given primitive type.
  void setPrimitiveSpec(char Specifier, uint32_t BitWidth, Align ABIAlign,
                        Align PrefAlign);

  /// Searches for a pointer specification that matches the given address space.
  /// Returns the default address space specification if not found.
  LLVM_ABI const PointerSpec &getPointerSpec(uint32_t AddrSpace) const;

  /// Sets or updates the specification for pointer in the given address space.
  void setPointerSpec(uint32_t AddrSpace, uint32_t BitWidth, Align ABIAlign,
                      Align PrefAlign, uint32_t IndexBitWidth,
                      bool IsNonIntegral);

  /// Internal helper to get alignment for integer of given bitwidth.
  LLVM_ABI Align getIntegerAlignment(uint32_t BitWidth, bool abi_or_pref) const;

  /// Internal helper method that returns requested alignment for type.
  Align getAlignment(Type *Ty, bool abi_or_pref) const;

  /// Attempts to parse primitive specification ('i', 'f', or 'v').
  Error parsePrimitiveSpec(StringRef Spec);

  /// Attempts to parse aggregate specification ('a').
  Error parseAggregateSpec(StringRef Spec);

  /// Attempts to parse pointer specification ('p').
  Error parsePointerSpec(StringRef Spec);

  /// Attempts to parse a single specification.
  Error parseSpecification(StringRef Spec,
                           SmallVectorImpl<unsigned> &NonIntegralAddressSpaces);

  /// Attempts to parse a data layout string.
  Error parseLayoutString(StringRef LayoutString);

public:
  /// Constructs a DataLayout with default values.
  LLVM_ABI DataLayout();

  /// Constructs a DataLayout from a specification string.
  /// WARNING: Aborts execution if the string is malformed. Use parse() instead.
  LLVM_ABI explicit DataLayout(StringRef LayoutString);

  DataLayout(const DataLayout &DL) { *this = DL; }

  LLVM_ABI ~DataLayout(); // Not virtual, do not subclass this class

  LLVM_ABI DataLayout &operator=(const DataLayout &Other);

  LLVM_ABI bool operator==(const DataLayout &Other) const;
  bool operator!=(const DataLayout &Other) const { return !(*this == Other); }

  /// Parse a data layout string and return the layout. Return an error
  /// description on failure.
  LLVM_ABI static Expected<DataLayout> parse(StringRef LayoutString);

  /// Layout endianness...
  bool isLittleEndian() const { return !BigEndian; }
  bool isBigEndian() const { return BigEndian; }

  /// Returns the string representation of the DataLayout.
  ///
  /// This representation is in the same format accepted by the string
  /// constructor above. This should not be used to compare two DataLayout as
  /// different string can represent the same layout.
  const std::string &getStringRepresentation() const {
    return StringRepresentation;
  }

  /// Test if the DataLayout was constructed from an empty string.
  bool isDefault() const { return StringRepresentation.empty(); }

  /// Returns true if the specified type is known to be a native integer
  /// type supported by the CPU.
  ///
  /// For example, i64 is not native on most 32-bit CPUs and i37 is not native
  /// on any known one. This returns false if the integer width is not legal.
  ///
  /// The width is specified in bits.
  bool isLegalInteger(uint64_t Width) const {
    return llvm::is_contained(LegalIntWidths, Width);
  }

  bool isIllegalInteger(uint64_t Width) const { return !isLegalInteger(Width); }

  /// Returns the natural stack alignment, or MaybeAlign() if one wasn't
  /// specified.
  MaybeAlign getStackAlignment() const { return StackNaturalAlign; }

  unsigned getAllocaAddrSpace() const { return AllocaAddrSpace; }

  PointerType *getAllocaPtrType(LLVMContext &Ctx) const {
    return PointerType::get(Ctx, AllocaAddrSpace);
  }

  /// Returns the alignment of function pointers, which may or may not be
  /// related to the alignment of functions.
  /// \see getFunctionPtrAlignType
  MaybeAlign getFunctionPtrAlign() const { return FunctionPtrAlign; }

  /// Return the type of function pointer alignment.
  /// \see getFunctionPtrAlign
  FunctionPtrAlignType getFunctionPtrAlignType() const {
    return TheFunctionPtrAlignType;
  }

  unsigned getProgramAddressSpace() const { return ProgramAddrSpace; }
  unsigned getDefaultGlobalsAddressSpace() const {
    return DefaultGlobalsAddrSpace;
  }

  bool hasMicrosoftFastStdCallMangling() const {
    return ManglingMode == MM_WinCOFFX86;
  }

  /// Returns true if symbols with leading question marks should not receive IR
  /// mangling. True for Windows mangling modes.
  bool doNotMangleLeadingQuestionMark() const {
    return ManglingMode == MM_WinCOFF || ManglingMode == MM_WinCOFFX86;
  }

  bool hasLinkerPrivateGlobalPrefix() const { return ManglingMode == MM_MachO; }

  StringRef getLinkerPrivateGlobalPrefix() const {
    if (ManglingMode == MM_MachO)
      return "l";
    return "";
  }

  char getGlobalPrefix() const {
    switch (ManglingMode) {
    case MM_None:
    case MM_ELF:
    case MM_GOFF:
    case MM_Mips:
    case MM_WinCOFF:
    case MM_XCOFF:
      return '\0';
    case MM_MachO:
    case MM_WinCOFFX86:
      return '_';
    }
    llvm_unreachable("invalid mangling mode");
  }

  StringRef getPrivateGlobalPrefix() const {
    switch (ManglingMode) {
    case MM_None:
      return "";
    case MM_ELF:
    case MM_WinCOFF:
      return ".L";
    case MM_GOFF:
      return "L#";
    case MM_Mips:
      return "$";
    case MM_MachO:
    case MM_WinCOFFX86:
      return "L";
    case MM_XCOFF:
      return "L..";
    }
    llvm_unreachable("invalid mangling mode");
  }

  LLVM_ABI static const char *getManglingComponent(const Triple &T);

  /// Returns true if the specified type fits in a native integer type
  /// supported by the CPU.
  ///
  /// For example, if the CPU only supports i32 as a native integer type, then
  /// i27 fits in a legal integer type but i45 does not.
  bool fitsInLegalInteger(unsigned Width) const {
    for (unsigned LegalIntWidth : LegalIntWidths)
      if (Width <= LegalIntWidth)
        return true;
    return false;
  }

  /// Layout pointer alignment
  LLVM_ABI Align getPointerABIAlignment(unsigned AS) const;

  /// Return target's alignment for stack-based pointers
  /// FIXME: The defaults need to be removed once all of
  /// the backends/clients are updated.
  LLVM_ABI Align getPointerPrefAlignment(unsigned AS = 0) const;

  /// The pointer representation size in bytes, rounded up to a whole number of
  /// bytes. The difference between this function and getAddressSize() is that
  /// this one returns the size of the entire pointer representation (including
  /// metadata bits for fat pointers) and the latter only returns the number of
  /// address bits.
  /// \sa DataLayout::getAddressSizeInBits
  /// FIXME: The defaults need to be removed once all of
  /// the backends/clients are updated.
  LLVM_ABI unsigned getPointerSize(unsigned AS = 0) const;

  /// The index size in bytes used for address calculation, rounded up to a
  /// whole number of bytes. This not only defines the size used in
  /// getelementptr operations, but also the size of addresses in this \p AS.
  /// For example, a 64-bit CHERI-enabled target has 128-bit pointers of which
  /// only 64 are used to represent the address and the remaining ones are used
  /// for metadata such as bounds and access permissions. In this case
  /// getPointerSize() returns 16, but getIndexSize() returns 8.
  /// To help with code understanding, the alias getAddressSize() can be used
  /// instead of getIndexSize() to clarify that an address width is needed.
  LLVM_ABI unsigned getIndexSize(unsigned AS) const;

  /// The integral size of a pointer in a given address space in bytes, which
  /// is defined to be the same as getIndexSize(). This exists as a separate
  /// function to make it clearer when reading code that the size of an address
  /// is being requested. While targets exist where index size and the
  /// underlying address width are not identical (e.g. AMDGPU fat pointers with
  /// 48-bit addresses and 32-bit offsets indexing), there is currently no need
  /// to differentiate these properties in LLVM.
  /// \sa DataLayout::getIndexSize
  /// \sa DataLayout::getAddressSizeInBits
  unsigned getAddressSize(unsigned AS) const { return getIndexSize(AS); }

  /// Return the address spaces containing non-integral pointers.  Pointers in
  /// this address space don't have a well-defined bitwise representation.
  SmallVector<unsigned, 8> getNonIntegralAddressSpaces() const {
    SmallVector<unsigned, 8> AddrSpaces;
    for (const PointerSpec &PS : PointerSpecs) {
      if (PS.IsNonIntegral)
        AddrSpaces.push_back(PS.AddrSpace);
    }
    return AddrSpaces;
  }

  bool isNonIntegralAddressSpace(unsigned AddrSpace) const {
    return getPointerSpec(AddrSpace).IsNonIntegral;
  }

  bool isNonIntegralPointerType(PointerType *PT) const {
    return isNonIntegralAddressSpace(PT->getAddressSpace());
  }

  bool isNonIntegralPointerType(Type *Ty) const {
    auto *PTy = dyn_cast<PointerType>(Ty);
    return PTy && isNonIntegralPointerType(PTy);
  }

  /// The size in bits of the pointer representation in a given address space.
  /// This is not necessarily the same as the integer address of a pointer (e.g.
  /// for fat pointers).
  /// \sa DataLayout::getAddressSizeInBits()
  /// FIXME: The defaults need to be removed once all of
  /// the backends/clients are updated.
  unsigned getPointerSizeInBits(unsigned AS = 0) const {
    return getPointerSpec(AS).BitWidth;
  }

  /// The size in bits of indices used for address calculation in getelementptr
  /// and for addresses in the given AS. See getIndexSize() for more
  /// information.
  /// \sa DataLayout::getAddressSizeInBits()
  unsigned getIndexSizeInBits(unsigned AS) const {
    return getPointerSpec(AS).IndexBitWidth;
  }

  /// The size in bits of an address in for the given AS. This is defined to
  /// return the same value as getIndexSizeInBits() since there is currently no
  /// target that requires these two properties to have different values. See
  /// getIndexSize() for more information.
  /// \sa DataLayout::getIndexSizeInBits()
  unsigned getAddressSizeInBits(unsigned AS) const {
    return getIndexSizeInBits(AS);
  }

  /// The pointer representation size in bits for this type. If this function is
  /// called with a pointer type, then the type size of the pointer is returned.
  /// If this function is called with a vector of pointers, then the type size
  /// of the pointer is returned.  This should only be called with a pointer or
  /// vector of pointers.
  LLVM_ABI unsigned getPointerTypeSizeInBits(Type *) const;

  /// The size in bits of the index used in GEP calculation for this type.
  /// The function should be called with pointer or vector of pointers type.
  /// This is defined to return the same value as getAddressSizeInBits(),
  /// but separate functions exist for code clarity.
  LLVM_ABI unsigned getIndexTypeSizeInBits(Type *Ty) const;

  /// The size in bits of an address for this type.
  /// This is defined to return the same value as getIndexTypeSizeInBits(),
  /// but separate functions exist for code clarity.
  unsigned getAddressSizeInBits(Type *Ty) const {
    return getIndexTypeSizeInBits(Ty);
  }

  unsigned getPointerTypeSize(Type *Ty) const {
    return getPointerTypeSizeInBits(Ty) / 8;
  }

  /// Size examples:
  ///
  /// Type        SizeInBits  StoreSizeInBits  AllocSizeInBits[*]
  /// ----        ----------  ---------------  ---------------
  ///  i1            1           8                8
  ///  i8            8           8                8
  ///  i19          19          24               32
  ///  i32          32          32               32
  ///  i100        100         104              128
  ///  i128        128         128              128
  ///  Float        32          32               32
  ///  Double       64          64               64
  ///  X86_FP80     80          80               96
  ///
  /// [*] The alloc size depends on the alignment, and thus on the target.
  ///     These values are for x86-32 linux.

  /// Returns the number of bits necessary to hold the specified type.
  ///
  /// If Ty is a scalable vector type, the scalable property will be set and
  /// the runtime size will be a positive integer multiple of the base size.
  ///
  /// For example, returns 36 for i36 and 80 for x86_fp80. The type passed must
  /// have a size (Type::isSized() must return true).
  TypeSize getTypeSizeInBits(Type *Ty) const;

  /// Returns the maximum number of bytes that may be overwritten by
  /// storing the specified type.
  ///
  /// If Ty is a scalable vector type, the scalable property will be set and
  /// the runtime size will be a positive integer multiple of the base size.
  ///
  /// For example, returns 5 for i36 and 10 for x86_fp80.
  TypeSize getTypeStoreSize(Type *Ty) const {
    TypeSize StoreSizeInBits = getTypeStoreSizeInBits(Ty);
    return {StoreSizeInBits.getKnownMinValue() / 8,
            StoreSizeInBits.isScalable()};
  }

  /// Returns the maximum number of bits that may be overwritten by
  /// storing the specified type; always a multiple of 8.
  ///
  /// If Ty is a scalable vector type, the scalable property will be set and
  /// the runtime size will be a positive integer multiple of the base size.
  ///
  /// For example, returns 40 for i36 and 80 for x86_fp80.
  TypeSize getTypeStoreSizeInBits(Type *Ty) const {
    TypeSize BaseSize = getTypeSizeInBits(Ty);
    uint64_t AlignedSizeInBits =
        alignToPowerOf2(BaseSize.getKnownMinValue(), 8);
    return {AlignedSizeInBits, BaseSize.isScalable()};
  }

  /// Returns true if no extra padding bits are needed when storing the
  /// specified type.
  ///
  /// For example, returns false for i19 that has a 24-bit store size.
  bool typeSizeEqualsStoreSize(Type *Ty) const {
    return getTypeSizeInBits(Ty) == getTypeStoreSizeInBits(Ty);
  }

  /// Returns the offset in bytes between successive objects of the
  /// specified type, including alignment padding.
  ///
  /// If Ty is a scalable vector type, the scalable property will be set and
  /// the runtime size will be a positive integer multiple of the base size.
  ///
  /// This is the amount that alloca reserves for this type. For example,
  /// returns 12 or 16 for x86_fp80, depending on alignment.
  TypeSize getTypeAllocSize(Type *Ty) const;

  /// Returns the offset in bits between successive objects of the
  /// specified type, including alignment padding; always a multiple of 8.
  ///
  /// If Ty is a scalable vector type, the scalable property will be set and
  /// the runtime size will be a positive integer multiple of the base size.
  ///
  /// This is the amount that alloca reserves for this type. For example,
  /// returns 96 or 128 for x86_fp80, depending on alignment.
  TypeSize getTypeAllocSizeInBits(Type *Ty) const {
    return 8 * getTypeAllocSize(Ty);
  }

  /// Returns the minimum ABI-required alignment for the specified type.
  LLVM_ABI Align getABITypeAlign(Type *Ty) const;

  /// Helper function to return `Alignment` if it's set or the result of
  /// `getABITypeAlign(Ty)`, in any case the result is a valid alignment.
  inline Align getValueOrABITypeAlignment(MaybeAlign Alignment,
                                          Type *Ty) const {
    return Alignment ? *Alignment : getABITypeAlign(Ty);
  }

  /// Returns the minimum ABI-required alignment for an integer type of
  /// the specified bitwidth.
  Align getABIIntegerTypeAlignment(unsigned BitWidth) const {
    return getIntegerAlignment(BitWidth, /* abi_or_pref */ true);
  }

  /// Returns the preferred stack/global alignment for the specified
  /// type.
  ///
  /// This is always at least as good as the ABI alignment.
  LLVM_ABI Align getPrefTypeAlign(Type *Ty) const;

  /// Returns an integer type with size at least as big as that of a
  /// pointer in the given address space.
  LLVM_ABI IntegerType *getIntPtrType(LLVMContext &C,
                                      unsigned AddressSpace = 0) const;

  /// Returns an integer (vector of integer) type with size at least as
  /// big as that of a pointer of the given pointer (vector of pointer) type.
  LLVM_ABI Type *getIntPtrType(Type *) const;

  /// Returns the smallest integer type with size at least as big as
  /// Width bits.
  LLVM_ABI Type *getSmallestLegalIntType(LLVMContext &C,
                                         unsigned Width = 0) const;

  /// Returns the largest legal integer type, or null if none are set.
  Type *getLargestLegalIntType(LLVMContext &C) const {
    unsigned LargestSize = getLargestLegalIntTypeSizeInBits();
    return (LargestSize == 0) ? nullptr : Type::getIntNTy(C, LargestSize);
  }

  /// Returns the size of largest legal integer type size, or 0 if none
  /// are set.
  LLVM_ABI unsigned getLargestLegalIntTypeSizeInBits() const;

  /// Returns the type of a GEP index in \p AddressSpace.
  /// If it was not specified explicitly, it will be the integer type of the
  /// pointer width - IntPtrType.
  LLVM_ABI IntegerType *getIndexType(LLVMContext &C,
                                     unsigned AddressSpace) const;
  /// Returns the type of an address in \p AddressSpace
  IntegerType *getAddressType(LLVMContext &C, unsigned AddressSpace) const {
    return getIndexType(C, AddressSpace);
  }

  /// Returns the type of a GEP index.
  /// If it was not specified explicitly, it will be the integer type of the
  /// pointer width - IntPtrType.
  LLVM_ABI Type *getIndexType(Type *PtrTy) const;
  /// Returns the type of an address in \p AddressSpace
  Type *getAddressType(Type *PtrTy) const { return getIndexType(PtrTy); }

  /// Returns the offset from the beginning of the type for the specified
  /// indices.
  ///
  /// Note that this takes the element type, not the pointer type.
  /// This is used to implement getelementptr.
  LLVM_ABI int64_t getIndexedOffsetInType(Type *ElemTy,
                                          ArrayRef<Value *> Indices) const;

  /// Get GEP indices to access Offset inside ElemTy. ElemTy is updated to be
  /// the result element type and Offset to be the residual offset.
  LLVM_ABI SmallVector<APInt> getGEPIndicesForOffset(Type *&ElemTy,
                                                     APInt &Offset) const;

  /// Get single GEP index to access Offset inside ElemTy. Returns std::nullopt
  /// if index cannot be computed, e.g. because the type is not an aggregate.
  /// ElemTy is updated to be the result element type and Offset to be the
  /// residual offset.
  LLVM_ABI std::optional<APInt> getGEPIndexForOffset(Type *&ElemTy,
                                                     APInt &Offset) const;

  /// Returns a StructLayout object, indicating the alignment of the
  /// struct, its size, and the offsets of its fields.
  ///
  /// Note that this information is lazily cached.
  LLVM_ABI const StructLayout *getStructLayout(StructType *Ty) const;

  /// Returns the preferred alignment of the specified global.
  ///
  /// This includes an explicitly requested alignment (if the global has one).
  LLVM_ABI Align getPreferredAlign(const GlobalVariable *GV) const;
};

inline DataLayout *unwrap(LLVMTargetDataRef P) {
  return reinterpret_cast<DataLayout *>(P);
}

inline LLVMTargetDataRef wrap(const DataLayout *P) {
  return reinterpret_cast<LLVMTargetDataRef>(const_cast<DataLayout *>(P));
}

/// Used to lazily calculate structure layout information for a target machine,
/// based on the DataLayout structure.
class StructLayout final : private TrailingObjects<StructLayout, TypeSize> {
  friend TrailingObjects;

  TypeSize StructSize;
  Align StructAlignment;
  unsigned IsPadded : 1;
  unsigned NumElements : 31;

public:
  TypeSize getSizeInBytes() const { return StructSize; }

  TypeSize getSizeInBits() const { return 8 * StructSize; }

  Align getAlignment() const { return StructAlignment; }

  /// Returns whether the struct has padding or not between its fields.
  /// NB: Padding in nested element is not taken into account.
  bool hasPadding() const { return IsPadded; }

  /// Given a valid byte offset into the structure, returns the structure
  /// index that contains it.
  LLVM_ABI unsigned getElementContainingOffset(uint64_t FixedOffset) const;

  MutableArrayRef<TypeSize> getMemberOffsets() {
    return getTrailingObjects(NumElements);
  }

  ArrayRef<TypeSize> getMemberOffsets() const {
    return getTrailingObjects(NumElements);
  }

  TypeSize getElementOffset(unsigned Idx) const {
    assert(Idx < NumElements && "Invalid element idx!");
    return getMemberOffsets()[Idx];
  }

  TypeSize getElementOffsetInBits(unsigned Idx) const {
    return getElementOffset(Idx) * 8;
  }

private:
  friend class DataLayout; // Only DataLayout can create this class

  StructLayout(StructType *ST, const DataLayout &DL);
};

// The implementation of this method is provided inline as it is particularly
// well suited to constant folding when called on a specific Type subclass.
inline TypeSize DataLayout::getTypeSizeInBits(Type *Ty) const {
  assert(Ty->isSized() && "Cannot getTypeInfo() on a type that is unsized!");
  switch (Ty->getTypeID()) {
  case Type::LabelTyID:
    return TypeSize::getFixed(getPointerSizeInBits(0));
  case Type::PointerTyID:
    return TypeSize::getFixed(
        getPointerSizeInBits(Ty->getPointerAddressSpace()));
  case Type::ArrayTyID: {
    ArrayType *ATy = cast<ArrayType>(Ty);
    return ATy->getNumElements() *
           getTypeAllocSizeInBits(ATy->getElementType());
  }
  case Type::StructTyID:
    // Get the layout annotation... which is lazily created on demand.
    return getStructLayout(cast<StructType>(Ty))->getSizeInBits();
  case Type::IntegerTyID:
    return TypeSize::getFixed(Ty->getIntegerBitWidth());
  case Type::HalfTyID:
  case Type::BFloatTyID:
    return TypeSize::getFixed(16);
  case Type::FloatTyID:
    return TypeSize::getFixed(32);
  case Type::DoubleTyID:
    return TypeSize::getFixed(64);
  case Type::PPC_FP128TyID:
  case Type::FP128TyID:
    return TypeSize::getFixed(128);
  case Type::X86_AMXTyID:
    return TypeSize::getFixed(8192);
  // In memory objects this is always aligned to a higher boundary, but
  // only 80 bits contain information.
  case Type::X86_FP80TyID:
    return TypeSize::getFixed(80);
  case Type::FixedVectorTyID:
  case Type::ScalableVectorTyID: {
    VectorType *VTy = cast<VectorType>(Ty);
    auto EltCnt = VTy->getElementCount();
    uint64_t MinBits = EltCnt.getKnownMinValue() *
                       getTypeSizeInBits(VTy->getElementType()).getFixedValue();
    return TypeSize(MinBits, EltCnt.isScalable());
  }
  case Type::TargetExtTyID: {
    Type *LayoutTy = cast<TargetExtType>(Ty)->getLayoutType();
    return getTypeSizeInBits(LayoutTy);
  }
  default:
    llvm_unreachable("DataLayout::getTypeSizeInBits(): Unsupported type");
  }
}

} // end namespace llvm

#endif // LLVM_IR_DATALAYOUT_H
