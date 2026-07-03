//===- X86.cpp ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TypeSize.h"
#include <algorithm>
#include <cassert>
#include <cstdint>

namespace llvm {
namespace abi {

static unsigned getNativeVectorSizeForAVXABI(X86AVXABILevel AVXLevel) {
  switch (AVXLevel) {
  case X86AVXABILevel::AVX512:
    return 512;
  case X86AVXABILevel::AVX:
    return 256;
  case X86AVXABILevel::None:
    return 128;
  }
  llvm_unreachable("Unknown AVXLevel");
}

// The width of an integer's storage container, mirroring Clang's
// ASTContext::getTypeSize. For a plain integer this is its bit width; for a
// _BitInt(N) it is N rounded up to the type's alignment. The x86-64 _BitInt
// max alignment is 64, so this clamp is target-specific and kept file-local.
static uint64_t getClangIntegerWidthInBits(const IntegerType *IT) {
  uint64_t NumBits = IT->getSizeInBits().getFixedValue();
  if (!IT->isBitInt())
    return NumBits;
  uint64_t BitAlign =
      std::max<uint64_t>(8, std::min<uint64_t>(64, llvm::bit_ceil(NumBits)));
  return llvm::alignTo(NumBits, BitAlign);
}

static uint64_t getClangVectorWidthInBits(const VectorType *VT) {
  const Type *EltTy = VT->getElementType();
  uint64_t EltWidth = EltTy->getSizeInBits().getFixedValue();
  if (const auto *IT = dyn_cast<IntegerType>(EltTy))
    EltWidth = getClangIntegerWidthInBits(IT);
  uint64_t Width =
      std::max<uint64_t>(8, EltWidth * VT->getNumElements().getKnownMinValue());
  if (Width & (Width - 1))
    Width = llvm::alignTo(Width, llvm::bit_ceil(Width));
  return Width;
}

// The storage-container width of a type, mirroring Clang's getTypeSize. Used on
// the stack path so a _BitInt or illegal vector coerces to the integer covering
// its storage, not its raw iN width.
static uint64_t getClangTypeWidthInBits(const Type *Ty) {
  if (const auto *VT = dyn_cast<VectorType>(Ty))
    return getClangVectorWidthInBits(VT);
  if (const auto *IT = dyn_cast<IntegerType>(Ty))
    return getClangIntegerWidthInBits(IT);
  return Ty->getSizeInBits().getFixedValue();
}

class X86_64TargetInfo : public TargetInfo {
public:
  enum Class { Integer, Sse, SseUp, X87, X87Up, ComplexX87, NoClass, Memory };

private:
  TypeBuilder &TB;
  X86AVXABILevel AVXLevel;
  bool Has64BitPointers;

  static Class merge(Class Accum, Class Field);

  void postMerge(unsigned AggregateSize, Class &Lo, Class &Hi) const;

  void classify(const Type *T, uint64_t OffsetBase, Class &Lo, Class &Hi,
                bool IsNamedArg, bool IsRegCall = false) const;

  const Type *getIntegerTypeAtOffset(const Type *IRType, unsigned IROffset,
                                     const Type *SourceTy,
                                     unsigned SourceOffset,
                                     bool InMemory = false) const;

  const Type *getSSETypeAtOffset(const Type *ABIType, unsigned ABIOffset,
                                 const Type *SourceTy,
                                 unsigned SourceOffset) const;
  bool isIllegalVectorType(const Type *Ty) const;
  bool containsMatrixField(const RecordType *RT) const;

  void computeInfo(FunctionInfo &FI) const override;
  ArgInfo getIndirectReturnResult(const Type *Ty) const;
  const Type *getFPTypeAtOffset(const Type *Ty, unsigned Offset) const;

  const Type *isSingleElementStruct(const Type *Ty) const;
  const Type *getByteVectorType(const Type *Ty) const;

  const Type *createPairType(const Type *Lo, const Type *Hi) const;
  ArgInfo getIndirectResult(const Type *Ty, unsigned FreeIntRegs) const;

  ArgInfo classifyReturnType(const Type *RetTy) const;

  ArgInfo classifyArgumentType(const Type *Ty, unsigned FreeIntRegs,
                               unsigned &NeededInt, unsigned &NeededSse,
                               bool IsNamedArg, bool IsRegCall = false) const;
  const Type *useFirstFieldIfTransparentUnion(const Type *Ty) const;

public:
  X86_64TargetInfo(TypeBuilder &TypeBuilder, X86AVXABILevel AVXABILevel,
                   bool Has64BitPtrs, const ABICompatInfo &Compat)
      : TargetInfo(Compat), TB(TypeBuilder), AVXLevel(AVXABILevel),
        Has64BitPointers(Has64BitPtrs) {}

  bool has64BitPointers() const { return Has64BitPointers; }
};

// Gets the "best" type to represent the union.
static const Type *reduceUnionForX8664(const RecordType *UnionType,
                                       TypeBuilder &TB) {
  assert(UnionType->isUnion() && "Expected union type");

  ArrayRef<FieldInfo> Fields = UnionType->getFields();
  if (Fields.empty()) {
    return nullptr;
  }

  const Type *StorageType = nullptr;

  for (const auto &Field : Fields) {
    if (Field.IsBitField && Field.IsUnnamedBitfield &&
        Field.BitFieldWidth == 0) {
      continue;
    }

    const Type *FieldType = Field.FieldType;

    if (UnionType->isTransparentUnion() && !StorageType) {
      StorageType = FieldType;
      break;
    }

    if (!StorageType ||
        FieldType->getAlignment() > StorageType->getAlignment() ||
        (FieldType->getAlignment() == StorageType->getAlignment() &&
         TypeSize::isKnownGT(FieldType->getSizeInBits(),
                             StorageType->getSizeInBits()))) {
      StorageType = FieldType;
    }
  }
  return StorageType;
}

void X86_64TargetInfo::postMerge(unsigned AggregateSize, Class &Lo,
                                 Class &Hi) const {
  // AMD64-ABI 3.2.3p2: Rule 5. Then a post merger cleanup is done:
  //
  // (a) If one of the classes is Memory, the whole argument is passed in
  //     memory.
  //
  // (b) If X87Up is not preceded by X87, the whole argument is passed in
  //     memory.
  //
  // (c) If the size of the aggregate exceeds two eightbytes and the first
  //     eightbyte isn't SSE or any other eightbyte isn't SSEUP, the whole
  //     argument is passed in memory. NOTE: This is necessary to keep the
  //     ABI working for processors that don't support the __m256 type.
  //
  // (d) If SSEUP is not preceded by SSE or SSEUP, it is converted to SSE.
  //
  // Some of these are enforced by the merging logic.  Others can arise
  // only with unions; for example:
  //   union { _Complex double; unsigned; }
  //
  // Note that clauses (b) and (c) were added in 0.98.

  if (Hi == Memory)
    Lo = Memory;
  if (Hi == X87Up && Lo != X87 && getABICompatInfo().HonorsRevision98)
    Lo = Memory;
  if (AggregateSize > 128 && (Lo != Sse || Hi != SseUp))
    Lo = Memory;
  if (Hi == SseUp && Lo != Sse)
    Hi = Sse;
}
X86_64TargetInfo::Class X86_64TargetInfo::merge(Class Accum, Class Field) {
  // AMD64-ABI 3.2.3p2: Rule 4. Each field of an object is
  // classified recursively so that always two fields are
  // considered. The resulting class is calculated according to
  // the classes of the fields in the eightbyte:
  //
  // (a) If both classes are equal, this is the resulting class.
  //
  // (b) If one of the classes is NO_CLASS, the resulting class is
  // the other class.
  //
  // (c) If one of the classes is MEMORY, the result is the MEMORY
  // class.
  //
  // (d) If one of the classes is INTEGER, the result is the
  // INTEGER.
  //
  // (e) If one of the classes is X87, X87Up, COMPLEX_X87 class,
  // MEMORY is used as class.
  //
  // (f) Otherwise class SSE is used.

  // Accum should never be memory (we should have returned) or
  // ComplexX87 (because this cannot be passed in a structure).
  assert((Accum != Memory && Accum != ComplexX87) &&
         "Invalid accumulated classification during merge.");

  if (Accum == Field || Field == NoClass)
    return Accum;
  if (Field == Memory)
    return Memory;
  if (Accum == NoClass)
    return Field;
  if (Accum == Integer || Field == Integer)
    return Integer;
  if (Field == X87 || Field == X87Up || Field == ComplexX87 || Accum == X87 ||
      Accum == X87Up)
    return Memory;

  return Sse;
}

// A record with a matrix-extension field is passed in memory.  clang has no
// matrix-specific ABI code: a matrix falls through X86_64ABIInfo::classify to
// the default MEMORY class.  We model matrices as arrays, so this check
// reproduces that record-with-matrix -> MEMORY result.
bool X86_64TargetInfo::containsMatrixField(const RecordType *RT) const {
  for (const auto &Field : RT->getFields()) {
    const Type *FieldType = Field.FieldType;

    if (const auto *AT = dyn_cast<ArrayType>(FieldType)) {
      if (AT->isMatrixType())
        return true;
      continue;
    }

    if (const auto *NestedRT = dyn_cast<RecordType>(FieldType))
      if (containsMatrixField(NestedRT))
        return true;
  }
  return false;
}

void X86_64TargetInfo::classify(const Type *T, uint64_t OffsetBase, Class &Lo,
                                Class &Hi, bool IsNamedArg,
                                bool IsRegCall) const {
  Lo = Hi = NoClass;
  Class &Current = OffsetBase < 64 ? Lo : Hi;
  Current = Memory;

  if (T->isVoid()) {
    Current = NoClass;
    return;
  }

  if (const auto *IT = dyn_cast<IntegerType>(T)) {
    auto BitWidth = IT->getSizeInBits().getFixedValue();

    if (BitWidth == 128 ||
        (IT->isBitInt() && BitWidth > 64 && BitWidth <= 128)) {
      Lo = Integer;
      Hi = Integer;
    } else if (BitWidth <= 64) {
      Current = Integer;
    }

    return;
  }

  if (const auto *FT = dyn_cast<FloatType>(T)) {
    const auto *FltSem = FT->getSemantics();

    if (FltSem == &llvm::APFloat::IEEEsingle() ||
        FltSem == &llvm::APFloat::IEEEdouble() ||
        FltSem == &llvm::APFloat::IEEEhalf() ||
        FltSem == &llvm::APFloat::BFloat()) {
      Current = Sse;
    } else if (FltSem == &llvm::APFloat::IEEEquad()) {
      Lo = Sse;
      Hi = SseUp;
    } else if (FltSem == &llvm::APFloat::x87DoubleExtended()) {
      Lo = X87;
      Hi = X87Up;
    } else {
      Current = Sse;
    }
    return;
  }
  if (T->isPointer()) {
    Current = Integer;
    return;
  }

  if (const auto *MPT = dyn_cast<MemberPointerType>(T)) {
    if (MPT->isFunctionPointer()) {
      if (Has64BitPointers) {
        Lo = Hi = Integer;
      } else {
        uint64_t EbFuncPtr = OffsetBase / 64;
        uint64_t EbThisAdj = (OffsetBase + 64 - 1) / 64;
        if (EbFuncPtr != EbThisAdj) {
          Lo = Hi = Integer;
        } else {
          Current = Integer;
        }
      }
    } else {
      Current = Integer;
    }
    return;
  }

  if (const auto *VT = dyn_cast<VectorType>(T)) {
    auto Size = VT->getSizeInBits().getFixedValue();
    const Type *ElementType = VT->getElementType();

    if (Size == 1 || Size == 8 || Size == 16 || Size == 32) {
      // gcc passes the following as integer:
      // 4 bytes - <4 x char>, <2 x short>, <1 x int>, <1 x float>
      // 2 bytes - <2 x char>, <1 x short>
      // 1 byte  - <1 x char>
      Current = Integer;
      // If this type crosses an eightbyte boundary, it should be
      // split.
      uint64_t EbLo = (OffsetBase) / 64;
      uint64_t EbHi = (OffsetBase + Size - 1) / 64;
      if (EbLo != EbHi)
        Hi = Lo;
    } else if (Size == 64) {
      if (const auto *FT = dyn_cast<FloatType>(ElementType)) {
        // gcc passes <1 x double> in memory. :(
        if (FT->getSemantics() == &llvm::APFloat::IEEEdouble())
          return;
      }

      // gcc passes <1 x long long> as SSE but clang used to unconditionally
      // pass them as integer.  For platforms where clang is the de facto
      // platform compiler, we must continue to use integer.
      if (const auto *IT = dyn_cast<IntegerType>(ElementType)) {
        uint64_t ElemBits = IT->getSizeInBits().getFixedValue();
        if (!getABICompatInfo().ClassifyIntegerMMXAsSSE && ElemBits == 64 &&
            !IT->isBitInt()) {
          Current = Integer;
        } else {
          Current = Sse;
        }
      } else {
        Current = Sse;
      }
      // If this type crosses an eightbyte boundary, it should be
      // split.
      if (OffsetBase && OffsetBase != 64)
        Hi = Lo;
    } else if (Size == 128 ||
               (IsNamedArg && Size <= getNativeVectorSizeForAVXABI(AVXLevel))) {
      if (const auto *IT = dyn_cast<IntegerType>(ElementType)) {
        uint64_t ElemBits = IT->getSizeInBits().getFixedValue();
        // gcc passes 256 and 512 bit <X x __int128> vectors in memory. :(
        if (getABICompatInfo().PassInt128VectorsInMem && Size != 128 &&
            ElemBits == 128 && !IT->isBitInt())
          return;
      }

      // Arguments of 256-bits are split into four eightbyte chunks. The
      // least significant one belongs to class SSE and all the others to class
      // SSEUP. The original Lo and Hi design considers that types can't be
      // greater than 128-bits, so a 64-bit split in Hi and Lo makes sense.
      // This design isn't correct for 256-bits, but since there're no cases
      // where the upper parts would need to be inspected, avoid adding
      // complexity and just consider Hi to match the 64-256 part.
      //
      // Note that per 3.5.7 of AMD64-ABI, 256-bit args are only passed in
      // registers if they are "named", i.e. not part of the "..." of a
      // variadic function.
      //
      // Similarly, per 3.2.3. of the AVX512 draft, 512-bits ("named") args are
      // split into eight eightbyte chunks, one SSE and seven SSEUP.
      Lo = Sse;
      Hi = SseUp;
    }
    return;
  }

  if (const auto *CT = dyn_cast<ComplexType>(T)) {
    const Type *ElementType = CT->getElementType();
    uint64_t Size = T->getSizeInBits().getFixedValue();

    if (isa<IntegerType>(ElementType)) {
      if (Size <= 64)
        Current = Integer;
      else if (Size <= 128)
        Lo = Hi = Integer;
    } else if (const auto *EFT = dyn_cast<FloatType>(ElementType)) {
      const auto *FltSem = EFT->getSemantics();
      if (FltSem == &llvm::APFloat::IEEEhalf() ||
          FltSem == &llvm::APFloat::IEEEsingle() ||
          FltSem == &llvm::APFloat::BFloat())
        Current = Sse;
      else if (FltSem == &llvm::APFloat::IEEEquad())
        Current = Memory;
      else if (FltSem == &llvm::APFloat::x87DoubleExtended())
        Current = ComplexX87;
      else if (FltSem == &llvm::APFloat::IEEEdouble())
        Lo = Hi = Sse;
      else
        llvm_unreachable("Unexpected long double representation!");
    }

    uint64_t ElementSize = ElementType->getSizeInBits().getFixedValue();
    // If this complex type crosses an eightbyte boundary then it
    // should be split.
    uint64_t EbReal = OffsetBase / 64;
    uint64_t EbImag = (OffsetBase + ElementSize) / 64;
    if (Hi == NoClass && EbReal != EbImag)
      Hi = Lo;

    return;
  }

  if (const auto *AT = dyn_cast<ArrayType>(T)) {
    // A matrix type is modeled as an array but, like Clang, is treated as a
    // non-aggregate scalar: it matches no class here and stays in the Memory
    // class, so classify*Type later returns it Direct (coerced to its
    // flattened vector) rather than classifying it field-by-field.
    if (AT->isMatrixType())
      return;

    // Arrays are treated like structures.
    uint64_t Size = AT->getSizeInBits().getFixedValue();

    // AMD64-ABI 3.2.3p2: Rule 1. If the size of an object is larger
    // than eight eightbytes, ..., it has class MEMORY.
    // regcall ABI doesn't have limitation to an object. The only limitation
    // is the free registers, which will be checked in computeInfo.
    if (!IsRegCall && Size > 512)
      return;

    // AMD64-ABI 3.2.3p2: Rule 1. If ..., or it contains unaligned
    // fields, it has class MEMORY.
    //
    // Only need to check alignment of array base.
    const Type *ElementType = AT->getElementType();
    uint64_t ElemAlign = ElementType->getAlignment().value() * 8;
    if (OffsetBase % ElemAlign)
      return;

    // Otherwise implement simplified merge. We could be smarter about
    // this, but it isn't worth it and would be harder to verify.
    Current = NoClass;
    uint64_t EltSize = ElementType->getSizeInBits().getFixedValue();
    uint64_t ArraySize = AT->getNumElements();

    // The only case a 256-bit wide vector could be used is when the array
    // contains a single 256-bit element. Since Lo and Hi logic isn't extended
    // to work for sizes wider than 128, early check and fallback to memory.
    //
    if (Size > 128 &&
        (Size != EltSize || Size > getNativeVectorSizeForAVXABI(AVXLevel)))
      return;

    for (uint64_t I = 0, Offset = OffsetBase; I < ArraySize;
         ++I, Offset += EltSize) {
      Class FieldLo, FieldHi;
      classify(ElementType, Offset, FieldLo, FieldHi, IsNamedArg);
      Lo = merge(Lo, FieldLo);
      Hi = merge(Hi, FieldHi);
      if (Lo == Memory || Hi == Memory)
        break;
    }
    postMerge(Size, Lo, Hi);
    assert((Hi != SseUp || Lo == Sse) && "Invalid SseUp array classification.");
    return;
  }

  if (const auto *RT = dyn_cast<RecordType>(T)) {
    uint64_t Size = RT->getSizeInBits().getFixedValue();

    if (containsMatrixField(RT)) {
      Lo = Memory;
      return;
    }

    // AMD64-ABI 3.2.3p2: Rule 1. If the size of an object is larger
    // than eight eightbytes, ..., it has class MEMORY.
    if (Size > 512)
      return;

    // AMD64-ABI 3.2.3p2: Rule 2. If a C++ object has either a non-trivial
    // copy constructor or a non-trivial destructor, it is passed by invisible
    // reference.
    if (getRecordArgABI(RT))
      return;

    // Assume variable sized types are passed in memory.
    if (RT->hasFlexibleArrayMember())
      return;

    // Reset Lo class, this will be recomputed.
    Current = NoClass;

    // If this is a C++ record, classify the bases first.
    if (RT->isCXXRecord()) {
      for (const auto &Base : RT->getBaseClasses()) {

        // Classify this field.
        //
        // AMD64-ABI 3.2.3p2: Rule 3. If the size of the aggregate exceeds a
        // single eightbyte, each is classified separately. Each eightbyte gets
        // initialized to class NO_CLASS.
        Class FieldLo, FieldHi;
        uint64_t Offset = OffsetBase + Base.OffsetInBits;
        classify(Base.FieldType, Offset, FieldLo, FieldHi, IsNamedArg);
        Lo = merge(Lo, FieldLo);
        Hi = merge(Hi, FieldHi);

        if (getABICompatInfo().ReturnCXXRecordGreaterThan128InMem &&
            (Size > 128 &&
             (Size != Base.FieldType->getSizeInBits().getFixedValue() ||
              Size > getNativeVectorSizeForAVXABI(AVXLevel))))
          Lo = Memory;

        if (Lo == Memory || Hi == Memory) {
          postMerge(Size, Lo, Hi);
          return;
        }
      }
    }

    // Classify the fields one at a time, merging the results.

    bool IsUnion = RT->isUnion() && !getABICompatInfo().Clang11Compat;
    for (const auto &Field : RT->getFields()) {
      uint64_t Offset = OffsetBase + Field.OffsetInBits;
      bool BitField = Field.IsBitField;

      if (BitField && Field.IsUnnamedBitfield)
        continue;

      if (Size > 128 &&
          ((!IsUnion &&
            Size != Field.FieldType->getSizeInBits().getFixedValue()) ||
           Size > getNativeVectorSizeForAVXABI(AVXLevel))) {
        Lo = Memory;
        postMerge(Size, Lo, Hi);
        return;
      }

      bool IsInMemory = Offset % (Field.FieldType->getAlignment().value() * 8);
      if (!BitField && IsInMemory) {
        Lo = Memory;
        postMerge(Size, Lo, Hi);
        return;
      }

      Class FieldLo, FieldHi;

      if (BitField) {
        uint64_t BitFieldSize = Field.BitFieldWidth;
        uint64_t EbLo = Offset / 64;
        uint64_t EbHi = (Offset + BitFieldSize - 1) / 64;

        if (EbLo) {
          assert(EbHi == EbLo && "Invalid classification, type > 16 bytes.");
          FieldLo = NoClass;
          FieldHi = Integer;
        } else {
          FieldLo = Integer;
          FieldHi = EbHi ? Integer : NoClass;
        }
      } else {
        classify(Field.FieldType, Offset, FieldLo, FieldHi, IsNamedArg);
      }

      Lo = merge(Lo, FieldLo);
      Hi = merge(Hi, FieldHi);
      if (Lo == Memory || Hi == Memory)
        break;
    }
    postMerge(Size, Lo, Hi);
    return;
  }

  Lo = Memory;
  Hi = NoClass;
}

const Type *
X86_64TargetInfo::useFirstFieldIfTransparentUnion(const Type *Ty) const {
  if (const auto *RT = dyn_cast<RecordType>(Ty)) {
    if (RT->isUnion() && RT->isTransparentUnion()) {
      auto Fields = RT->getFields();
      assert(!Fields.empty() && "transparent union cannot be empty");
      return Fields.front().FieldType;
    }
  }
  return Ty;
}

ArgInfo
X86_64TargetInfo::classifyArgumentType(const Type *Ty, unsigned FreeIntRegs,
                                       unsigned &NeededInt, unsigned &NeededSSE,
                                       bool IsNamedArg, bool IsRegCall) const {

  Ty = useFirstFieldIfTransparentUnion(Ty);

  X86_64TargetInfo::Class Lo, Hi;
  classify(Ty, 0, Lo, Hi, IsNamedArg, IsRegCall);

  // Check some invariants
  assert((Hi != Memory || Lo == Memory) && "Invalid memory classification.");
  assert((Hi != SseUp || Lo == Sse) && "Invalid SseUp classification.");

  NeededInt = 0;
  NeededSSE = 0;
  const Type *ResType = nullptr;

  switch (Lo) {
  case NoClass:
    if (Hi == NoClass)
      return ArgInfo::getIgnore();
    // If the low part is just padding, it takes no register, leave ResType
    // null.
    assert((Hi == Sse || Hi == Integer || Hi == X87Up) &&
           "Unknown missing lo part");
    break;

    // AMD64-ABI 3.2.3p3: Rule 1. If the class is MEMORY, pass the argument
    // on the stack.
  case Memory:
    // AMD64-ABI 3.2.3p3: Rule 5. If the class is X87, X87Up or
    // COMPLEX_X87, it is passed in memory.
  case X87:
  case ComplexX87:
    if (getRecordArgABI(Ty) == RAA_Indirect)
      ++NeededInt;
    return getIndirectResult(Ty, FreeIntRegs);

  case SseUp:
  case X87Up:
    llvm_unreachable("Invalid classification for lo word.");

    // AMD64-ABI 3.2.3p3: Rule 2. If the class is INTEGER, the next
    // available register of the sequence %rdi, %rsi, %rdx, %rcx, %r8
    // and %r9 is used.
  case Integer:
    ++NeededInt;

    // Pick an 8-byte type based on the preferred type.
    ResType = getIntegerTypeAtOffset(Ty, 0, Ty, 0);

    // If we have a sign or zero extended integer, make sure to return Extend
    // so that the parameter gets the right LLVM IR attributes.
    if (Hi == NoClass && ResType->isInteger()) {
      if (Ty->isInteger() && isPromotableInteger(cast<IntegerType>(Ty)))
        return ArgInfo::getExtend(Ty);
    }

    if (ResType->isInteger() && ResType->getSizeInBits() == 128) {
      assert(Hi == Integer);
      ++NeededInt;
      return ArgInfo::getDirect(ResType);
    }
    break;

    // AMD64-ABI 3.2.3p3: Rule 3. If the class is SSE, the next
    // available SSE register is used, the registers are taken in the
    // order from %xmm0 to %xmm7.
  case Sse:
    ResType = getSSETypeAtOffset(Ty, 0, Ty, 0);
    ++NeededSSE;
    break;
  }

  const Type *HighPart = nullptr;
  switch (Hi) {
    // Memory was handled previously, ComplexX87 and X87 should
    // never occur as hi classes, and X87Up must be preceded by X87,
    // which is passed in memory.
  case Memory:
  case X87:
  case ComplexX87:
    llvm_unreachable("Invalid classification for hi word.");

  case NoClass:
    break;

  case Integer:
    ++NeededInt;
    // Pick an 8-byte type based on the preferred type.
    HighPart = getIntegerTypeAtOffset(Ty, 8, Ty, 8);

    if (Lo == NoClass) // Pass HighPart at offset 8 in memory.
      return ArgInfo::getDirect(HighPart, 8);
    break;

    // X87Up generally doesn't occur here (long double is passed in
    // memory), except in situations involving unions.
  case X87Up:
  case Sse:
    ++NeededSSE;
    HighPart = getSSETypeAtOffset(Ty, 8, Ty, 8);

    if (Lo == NoClass) // Pass HighPart at offset 8 in memory.
      return ArgInfo::getDirect(HighPart, 8);
    break;

    // AMD64-ABI 3.2.3p3: Rule 4. If the class is SSEUP, the
    // eightbyte is passed in the upper half of the last used SSE
    // register. This only happens when 128-bit vectors are passed.
  case SseUp:
    assert(Lo == Sse && "Unexpected SseUp classification");
    ResType = getByteVectorType(Ty);
    break;
  }

  // If a high part was specified, merge it together with the low part. It is
  // known to pass in the high eightbyte of the result. We do this by forming a
  // first class struct aggregate with the high and low part: {low, high}
  if (HighPart)
    ResType = createPairType(ResType, HighPart);

  return ArgInfo::getDirect(ResType);
}

ArgInfo X86_64TargetInfo::classifyReturnType(const Type *RetTy) const {
  // AMD64-ABI 3.2.3p4: Rule 1. Classify the return type with the
  // classification algorithm.

  X86_64TargetInfo::Class Lo, Hi;
  classify(RetTy, 0, Lo, Hi, /*isNamedArg*/ true);

  // Check some invariants
  assert((Hi != Memory || Lo == Memory) && "Invalid memory classification.");
  assert((Hi != SseUp || Lo == Sse) && "Invalid SseUp classification.");

  const Type *ResType = nullptr;
  switch (Lo) {
  case NoClass:
    if (Hi == NoClass)
      return ArgInfo::getIgnore();
    // If the low part is just padding, it takes no register, leave ResType
    // null.
    assert((Hi == Sse || Hi == Integer || Hi == X87Up) &&
           "Unknown missing lo part");
    break;
  case SseUp:
  case X87Up:
    llvm_unreachable("Invalid classification for lo word.");

    // AMD64-ABI 3.2.3p4: Rule 2. Types of class memory are returned via
    // hidden argument.
  case Memory:
    return getIndirectReturnResult(RetTy);

    // AMD64-ABI 3.2.3p4: Rule 3. If the class is INTEGER, the next
    // available register of the sequence %rax, %rdx is used.
  case Integer:
    ResType = getIntegerTypeAtOffset(RetTy, 0, RetTy, 0);
    // If we have a sign or zero extended integer, make sure to return Extend
    // so that the parameter gets the right LLVM IR attributes.
    if (Hi == NoClass && ResType->isInteger()) {
      if (const IntegerType *IntTy = dyn_cast<IntegerType>(RetTy)) {
        if (isPromotableInteger(IntTy))
          return ArgInfo::getExtend(RetTy);
      }
    }
    if (ResType->isInteger() && ResType->getSizeInBits() == 128) {
      assert(Hi == Integer);
      return ArgInfo::getDirect(ResType);
    }
    break;

    // AMD64-ABI 3.2.3p4: Rule 4. If the class is SSE, the next
    // available SSE register of the sequence %xmm0, %xmm1 is used.
  case Sse:
    ResType = getSSETypeAtOffset(RetTy, 0, RetTy, 0);
    break;

    // AMD64-ABI 3.2.3p4: Rule 6. If the class is X87, the value is
    // returned on the X87 stack in %st0 as 80-bit x87 number.
  case X87:
    ResType = TB.getFloatType(APFloat::x87DoubleExtended(), Align(16));
    break;

    // AMD64-ABI 3.2.3p4: Rule 8. If the class is COMPLEX_X87, the real
    // part of the value is returned in %st0 and the imaginary part in
    // %st1.
  case ComplexX87:
    assert(Hi == ComplexX87 && "Unexpected ComplexX87 classification.");
    {
      const Type *X87Type =
          TB.getFloatType(APFloat::x87DoubleExtended(), Align(16));
      FieldInfo Fields[] = {FieldInfo(X87Type, 0), FieldInfo(X87Type, 80)};
      ResType = TB.getRecordType(Fields, TypeSize::getFixed(160), Align(16));
    }
    break;
  }

  const Type *HighPart = nullptr;
  switch (Hi) {
    // Memory was handled previously and X87 should
    // never occur as a hi class.
  case Memory:
  case X87:
    llvm_unreachable("Invalid classification for hi word.");

  case ComplexX87:
  case NoClass:
    break;

  case Integer:
    HighPart = getIntegerTypeAtOffset(RetTy, 8, RetTy, 8);
    if (Lo == NoClass)
      return ArgInfo::getDirect(HighPart, 8);
    break;

  case Sse:
    HighPart = getSSETypeAtOffset(RetTy, 8, RetTy, 8);
    if (Lo == NoClass)
      return ArgInfo::getDirect(HighPart, 8);
    break;

    // AMD64-ABI 3.2.3p4: Rule 5. If the class is SSEUP, the eightbyte
    // is passed in the next available eightbyte chunk if the last used
    // vector register.
    //
    // SSEUP should always be preceded by SSE, just widen.
  case SseUp:
    assert(Lo == Sse && "Unexpected SseUp classification.");
    ResType = getByteVectorType(RetTy);
    break;

    // AMD64-ABI 3.2.3p4: Rule 7. If the class is X87Up, the value is
    // returned together with the previous X87 value in %st0.
  case X87Up:
    // If X87Up is preceded by X87, we don't need to do
    // anything. However, in some cases with unions it may not be
    // preceded by X87. In such situations we follow gcc and pass the
    // extra bits in an SSE reg.
    if (Lo != X87) {
      HighPart = getSSETypeAtOffset(RetTy, 8, RetTy, 8);
      if (Lo == NoClass) // Return HighPart at offset 8 in memory.
        return ArgInfo::getDirect(HighPart, 8);
    }
    break;
  }

  // If a high part was specified, merge it together with the low part.  It is
  // known to pass in the high eightbyte of the result.  We do this by forming a
  // first class struct aggregate with the high and low part: {low, high}
  if (HighPart)
    ResType = createPairType(ResType, HighPart);

  return ArgInfo::getDirect(ResType);
}

///  Given a high and low type that can ideally
/// be used as elements of a two register pair to pass or return, return a
/// first class aggregate to represent them.  For example, if the low part of
/// a by-value argument should be passed as i32* and the high part as float,
/// return {i32*, float}.
const Type *X86_64TargetInfo::createPairType(const Type *Lo,
                                             const Type *Hi) const {
  // In order to correctly satisfy the ABI, we need to the high part to start
  // at offset 8.  If the high and low parts we inferred are both 4-byte types
  // (e.g. i32 and i32) then the resultant struct type ({i32,i32}) won't have
  // the second element at offset 8.  Check for this:
  unsigned LoSize = (unsigned)Lo->getTypeAllocSize();
  llvm::Align HiAlign = Hi->getAlignment();
  unsigned HiStart = alignTo(LoSize, HiAlign);

  assert(HiStart != 0 && HiStart <= 8 && "Invalid x86-64 argument pair!");

  // To handle this, we have to increase the size of the low part so that the
  // second element will start at an 8 byte offset.  We can't increase the size
  // of the second element because it might make us access off the end of the
  // struct.
  const Type *AdjustedLo = Lo;
  if (HiStart != 8) {
    // There are usually two sorts of types the ABI generation code can produce
    // for the low part of a pair that aren't 8 bytes in size: half, float or
    // i8/i16/i32.  This can also include pointers when they are 32-bit (X32 and
    // NaCl).
    // Promote these to a larger type.
    if (Lo->isFloat()) {
      const FloatType *FT = cast<FloatType>(Lo);
      if (FT->getSemantics() == &APFloat::IEEEhalf() ||
          FT->getSemantics() == &APFloat::IEEEsingle() ||
          FT->getSemantics() == &APFloat::BFloat())
        AdjustedLo = TB.getFloatType(APFloat::IEEEdouble(), Align(8));
    }
    // Promote integers and pointers to i64
    else if (Lo->isInteger() || Lo->isPointer())
      AdjustedLo = TB.getIntegerType(64, Align(8), /*Signed=*/false);
    else
      assert((Lo->isInteger() || Lo->isPointer()) &&
             "Invalid/unknown low type in pair");
    unsigned AdjustedLoSize = AdjustedLo->getSizeInBits().getFixedValue() / 8;
    HiStart = alignTo(AdjustedLoSize, HiAlign);
  }

  // Create the pair struct
  FieldInfo Fields[] = {FieldInfo(AdjustedLo, 0), FieldInfo(Hi, HiStart * 8)};

  // Verify the high part is at offset 8
  assert((8 * 8) == Fields[1].OffsetInBits &&
         "High part must be at offset 8 bytes");

  uint64_t PairSizeInBits =
      Fields[1].OffsetInBits + Hi->getSizeInBits().getFixedValue();
  return TB.getRecordType(Fields, TypeSize::getFixed(PairSizeInBits), Align(8),
                          StructPacking::Default);
}

static bool bitsContainNoUserData(const Type *Ty, unsigned StartBit,
                                  unsigned EndBit) {
  // If range is completely beyond type size, it's definitely padding
  unsigned TySize = Ty->getSizeInBits().getFixedValue();
  if (TySize <= StartBit)
    return true;

  // Handle arrays - check each element
  if (const ArrayType *AT = dyn_cast<ArrayType>(Ty)) {
    const Type *EltTy = AT->getElementType();
    unsigned EltSize = EltTy->getSizeInBits().getFixedValue();

    for (unsigned I = 0; I < AT->getNumElements(); ++I) {
      unsigned EltOffset = I * EltSize;
      if (EltOffset >= EndBit)
        break;

      unsigned EltStart = (EltOffset < StartBit) ? StartBit - EltOffset : 0;
      if (!bitsContainNoUserData(EltTy, EltStart, EndBit - EltOffset))
        return false;
    }
    return true;
  }

  // Handle structs - check all fields and base classes
  if (const RecordType *RT = dyn_cast<RecordType>(Ty)) {
    if (RT->isUnion()) {
      for (const auto &Field : RT->getFields()) {
        if (Field.IsUnnamedBitfield)
          continue;

        unsigned FieldStart =
            (Field.OffsetInBits < StartBit) ? StartBit - Field.OffsetInBits : 0;
        unsigned FieldEnd =
            FieldStart + Field.FieldType->getSizeInBits().getFixedValue();

        // Check if field overlaps with the queried range
        if (FieldStart < EndBit && FieldEnd > StartBit) {
          // There's an overlap, so there is user data
          unsigned RelativeStart =
              (StartBit > FieldStart) ? StartBit - FieldStart : 0;
          unsigned RelativeEnd =
              (EndBit < FieldEnd)
                  ? EndBit - FieldStart
                  : Field.FieldType->getSizeInBits().getFixedValue();

          if (!bitsContainNoUserData(Field.FieldType, RelativeStart,
                                     RelativeEnd)) {
            return false;
          }
        }
      }
      return true;
    }
    // Check base classes first (for C++ records)
    if (RT->isCXXRecord()) {
      for (unsigned I = 0; I < RT->getNumBaseClasses(); ++I) {
        const FieldInfo &Base = RT->getBaseClasses()[I];
        if (Base.OffsetInBits >= EndBit)
          continue;

        unsigned BaseStart =
            (Base.OffsetInBits < StartBit) ? StartBit - Base.OffsetInBits : 0;
        if (!bitsContainNoUserData(Base.FieldType, BaseStart,
                                   EndBit - Base.OffsetInBits))
          return false;
      }
    }

    for (unsigned I = 0; I < RT->getNumFields(); ++I) {
      const FieldInfo &Field = RT->getFields()[I];
      if (Field.OffsetInBits >= EndBit)
        break;

      unsigned FieldStart =
          (Field.OffsetInBits < StartBit) ? StartBit - Field.OffsetInBits : 0;
      if (!bitsContainNoUserData(Field.FieldType, FieldStart,
                                 EndBit - Field.OffsetInBits))
        return false;
    }
    return true;
  }

  // For unions, vectors, and primitives - assume all bits are user data
  return false;
}

const Type *X86_64TargetInfo::getIntegerTypeAtOffset(const Type *ABIType,
                                                     unsigned ABIOffset,
                                                     const Type *SourceTy,
                                                     unsigned SourceOffset,
                                                     bool InMemory) const {

  const Type *WorkingType = ABIType;
  if (InMemory && ABIType->isInteger()) {
    const auto *IT = cast<IntegerType>(ABIType);
    unsigned OriginalBitWidth = IT->getSizeInBits().getFixedValue();

    unsigned WidenedBitWidth = OriginalBitWidth;
    if (OriginalBitWidth <= 8) {
      WidenedBitWidth = 8;
    } else {
      WidenedBitWidth = llvm::bit_ceil(OriginalBitWidth);
    }

    if (WidenedBitWidth != OriginalBitWidth) {
      WorkingType = TB.getIntegerType(WidenedBitWidth, ABIType->getAlignment(),
                                      IT->isSigned());
    }
  }
  // If we're dealing with an un-offset ABI type, then it means that we're
  // returning an 8-byte unit starting with it. See if we can safely use it.
  if (ABIOffset == 0) {
    // Pointers and int64's always fill the 8-byte unit.  Return WorkingType,
    // which is the in-memory-widened type (e.g. a _BitInt(37) field widened to
    // i64): returning the raw ABIType here would coerce the eightbyte to the
    // narrow iN instead of the storage integer clang uses.
    if ((WorkingType->isPointer() && Has64BitPointers) ||
        (WorkingType->isInteger() &&
         cast<IntegerType>(WorkingType)->getSizeInBits() == 64))
      return WorkingType;

    // If we have a 1/2/4-byte integer, we can use it only if the rest of the
    // goodness in the source type is just tail padding. This is allowed to
    // kick in for struct {double,int} on the int, but not on
    // struct{double,int,int} because we wouldn't return the second int. We
    // have to do this analysis on the source type because we can't depend on
    // unions being lowered a specific way etc.
    if ((WorkingType->isInteger() &&
         (cast<IntegerType>(WorkingType)->getSizeInBits() == 1 ||
          cast<IntegerType>(WorkingType)->getSizeInBits() == 8 ||
          cast<IntegerType>(WorkingType)->getSizeInBits() == 16 ||
          cast<IntegerType>(WorkingType)->getSizeInBits() == 32)) ||
        (WorkingType->isPointer() && !Has64BitPointers)) {

      unsigned BitWidth = WorkingType->isPointer()
                              ? 32
                              : cast<IntegerType>(WorkingType)->getSizeInBits();

      if (bitsContainNoUserData(SourceTy, SourceOffset * 8 + BitWidth,
                                SourceOffset * 8 + 64))
        return WorkingType;
    }
  }

  if (const auto *RTy = dyn_cast<RecordType>(ABIType)) {
    if (RTy->isUnion()) {
      const Type *ReducedType = reduceUnionForX8664(RTy, TB);
      if (ReducedType)
        return getIntegerTypeAtOffset(ReducedType, ABIOffset, SourceTy,
                                      SourceOffset, true);
    }
    if (const FieldInfo *Element =
            RTy->getElementContainingOffset(ABIOffset * 8)) {

      unsigned ElementOffsetBytes = Element->OffsetInBits / 8;
      return getIntegerTypeAtOffset(Element->FieldType,
                                    ABIOffset - ElementOffsetBytes, SourceTy,
                                    SourceOffset, true);
    }
  }

  if (const auto *ATy = dyn_cast<ArrayType>(ABIType)) {
    const Type *EltTy = ATy->getElementType();
    unsigned EltSize = EltTy->getSizeInBits() / 8;
    if (EltSize > 0) {
      unsigned EltOffset = (ABIOffset / EltSize) * EltSize;
      return getIntegerTypeAtOffset(EltTy, ABIOffset - EltOffset, SourceTy,
                                    SourceOffset, true);
    }
  }

  // If we have a 128-bit integer, we can pass it safely using an i128
  // so we return that
  if (ABIType->isInteger() && ABIType->getSizeInBits() == 128) {
    assert(ABIOffset == 0);
    return ABIType;
  }

  unsigned TySizeInBytes =
      llvm::divideCeil(SourceTy->getSizeInBits().getFixedValue(), 8);
  if (auto *IT = dyn_cast<IntegerType>(SourceTy)) {
    if (IT->isBitInt())
      TySizeInBytes =
          alignTo(SourceTy->getSizeInBits().getFixedValue(), 64) / 8;
  }
  assert(TySizeInBytes != SourceOffset && "Empty field?");
  unsigned AvailableSize = TySizeInBytes - SourceOffset;
  return TB.getIntegerType(std::min(AvailableSize, 8U) * 8, Align(1), false);
}
/// Returns the floating point type at the specified offset within a type, or
/// nullptr if no floating point type is found at that offset.
const Type *X86_64TargetInfo::getFPTypeAtOffset(const Type *Ty,
                                                unsigned Offset) const {
  // Check for direct match at offset 0
  if (Offset == 0 && Ty->isFloat())
    return Ty;

  if (const ComplexType *CT = dyn_cast<ComplexType>(Ty)) {
    const Type *ElementType = CT->getElementType();
    unsigned ElementSize = ElementType->getSizeInBits().getFixedValue() / 8;

    if (Offset == 0 || Offset == ElementSize)
      return ElementType;
    return nullptr;
  }

  // Handle struct types by checking each field
  if (const RecordType *RT = dyn_cast<RecordType>(Ty)) {
    if (const FieldInfo *Element = RT->getElementContainingOffset(Offset * 8)) {
      unsigned ElementOffsetBytes = Element->OffsetInBits / 8;
      return getFPTypeAtOffset(Element->FieldType, Offset - ElementOffsetBytes);
    }
  }

  // Handle array types
  if (const ArrayType *AT = dyn_cast<ArrayType>(Ty)) {
    const Type *EltTy = AT->getElementType();
    unsigned EltSize = EltTy->getSizeInBits() / 8;
    unsigned EltIndex = Offset / EltSize;

    return getFPTypeAtOffset(EltTy, Offset - (EltIndex * EltSize));
  }

  // No floating point type found at this offset
  return nullptr;
}

/// Helper to check if a floating point type matches specific semantics
static bool isFloatTypeWithSemantics(const Type *Ty,
                                     const fltSemantics &Semantics) {
  if (!Ty->isFloat())
    return false;
  const FloatType *FT = cast<FloatType>(Ty);
  return FT->getSemantics() == &Semantics;
}

/// GetSSETypeAtOffset - Return a type that will be passed by the backend in the
/// low 8 bytes of an XMM register, corresponding to the SSE class.
const Type *X86_64TargetInfo::getSSETypeAtOffset(const Type *ABIType,
                                                 unsigned ABIOffset,
                                                 const Type *SourceTy,
                                                 unsigned SourceOffset) const {

  if (const auto *RTy = dyn_cast<RecordType>(ABIType)) {
    if (RTy->isUnion()) {
      const Type *ReducedType = reduceUnionForX8664(RTy, TB);
      if (ReducedType) {
        return getSSETypeAtOffset(ReducedType, ABIOffset, SourceTy,
                                  SourceOffset);
      }
    }
  }

  auto Is16bitFpTy = [](const Type *T) {
    return isFloatTypeWithSemantics(T, APFloat::IEEEhalf()) ||
           isFloatTypeWithSemantics(T, APFloat::BFloat());
  };

  // Get the floating point type at the requested offset
  const Type *T0 = getFPTypeAtOffset(ABIType, ABIOffset);
  if (!T0 || isFloatTypeWithSemantics(T0, APFloat::IEEEdouble()))
    return TB.getFloatType(APFloat::IEEEdouble(), Align(8));

  // Calculate remaining source size in bytes
  unsigned SourceSize =
      (SourceTy->getSizeInBits().getFixedValue() / 8) - SourceOffset;

  // Try to get adjacent FP type
  const Type *T1 = nullptr;
  unsigned T0Size =
      alignTo(T0->getSizeInBits().getFixedValue(), T0->getAlignment().value()) /
      8;
  if (SourceSize > T0Size)
    T1 = getFPTypeAtOffset(ABIType, ABIOffset + T0Size);

  if (T1 == nullptr) {
    if (Is16bitFpTy(T0) && SourceSize > 4)
      T1 = getFPTypeAtOffset(ABIType, ABIOffset + 4);

    if (T1 == nullptr)
      return T0;
  }
  // Handle vector cases
  if (isFloatTypeWithSemantics(T0, APFloat::IEEEsingle()) &&
      isFloatTypeWithSemantics(T1, APFloat::IEEEsingle()))
    return TB.getVectorType(T0, ElementCount::getFixed(2), Align(8));

  if (Is16bitFpTy(T0) && Is16bitFpTy(T1)) {
    const Type *T2 = nullptr;
    if (SourceSize > 4)
      T2 = getFPTypeAtOffset(ABIType, ABIOffset + 4);
    if (!T2)
      return TB.getVectorType(T0, ElementCount::getFixed(2), Align(8));
    return TB.getVectorType(T0, ElementCount::getFixed(4), Align(8));
  }

  // Mixed half-float cases
  if (Is16bitFpTy(T0) || Is16bitFpTy(T1))
    return TB.getVectorType(TB.getFloatType(APFloat::IEEEhalf(), Align(2)),
                            ElementCount::getFixed(4), Align(8));

  // Default to double
  return TB.getFloatType(APFloat::IEEEdouble(), Align(8));
}

/// The ABI specifies that a value should be passed in a full vector XMM/YMM
/// register. Pick an LLVM IR type that will be passed as a vector register.
const Type *X86_64TargetInfo::getByteVectorType(const Type *Ty) const {
  // Wrapper structs/arrays that only contain vectors are passed just like
  // vectors; strip them off if present.
  if (const Type *InnerTy = isSingleElementStruct(Ty))
    Ty = InnerTy;

  // Handle vector types
  if (const VectorType *VT = dyn_cast<VectorType>(Ty)) {
    // Don't pass vXi128 vectors in their native type, the backend can't
    // legalize them.
    if (getABICompatInfo().PassInt128VectorsInMem &&
        VT->getElementType()->isInteger() &&
        cast<IntegerType>(VT->getElementType())->getSizeInBits() == 128) {
      unsigned Size = VT->getSizeInBits().getFixedValue();
      return TB.getVectorType(TB.getIntegerType(64, Align(8), /*Signed=*/false),
                              ElementCount::getFixed(Size / 64),
                              Align(Size / 8));
    }
    return VT;
  }

  // Handle fp128
  if (isFloatTypeWithSemantics(Ty, APFloat::IEEEquad()))
    return Ty;

  // We couldn't find the preferred IR vector type for 'Ty'.
  unsigned Size = Ty->getSizeInBits().getFixedValue();
  assert((Size == 128 || Size == 256 || Size == 512) && "Invalid vector size");

  return TB.getVectorType(TB.getFloatType(APFloat::IEEEdouble(), Align(8)),
                          ElementCount::getFixed(Size / 64), Align(Size / 8));
}

// Returns the single element if this is a single-element struct wrapper
const Type *X86_64TargetInfo::isSingleElementStruct(const Type *Ty) const {
  const auto *RT = dyn_cast<RecordType>(Ty);
  if (!RT)
    return nullptr;

  if (RT->hasFlexibleArrayMember())
    return nullptr;

  const Type *Found = nullptr;

  for (const auto &Base : RT->getBaseClasses()) {
    const Type *BaseTy = Base.FieldType;
    auto *BaseRT = dyn_cast<RecordType>(BaseTy);

    if (!BaseRT || BaseRT->isEmpty())
      continue;

    const Type *Elem = isSingleElementStruct(BaseTy);
    if (!Elem || Found)
      return nullptr;
    Found = Elem;
  }

  for (const auto &FI : RT->getFields()) {
    if (FI.isEmpty())
      continue;

    const Type *FTy = FI.FieldType;

    while (auto *AT = dyn_cast<ArrayType>(FTy)) {
      if (AT->getNumElements() != 1)
        break;
      FTy = AT->getElementType();
    }

    const Type *Elem;
    if (auto *InnerRT = dyn_cast<RecordType>(FTy))
      Elem = isSingleElementStruct(InnerRT);
    else
      Elem = FTy;
    if (!Elem || Found)
      return nullptr;
    Found = Elem;
  }

  if (!Found)
    return nullptr;
  if (Found->getSizeInBits() != Ty->getSizeInBits())
    return nullptr;

  return Found;
}

bool X86_64TargetInfo::isIllegalVectorType(const Type *Ty) const {
  if (const auto *VecTy = dyn_cast<VectorType>(Ty)) {
    uint64_t Size = VecTy->getSizeInBits().getFixedValue();
    unsigned LargestVector = getNativeVectorSizeForAVXABI(AVXLevel);

    // Vectors <= 64 bits or > largest supported vector size are illegal
    if (Size <= 64 || Size > LargestVector)
      return true;

    // Check for 128-bit integer element vectors that should be passed in memory
    const Type *EltTy = VecTy->getElementType();
    if (getABICompatInfo().PassInt128VectorsInMem && EltTy->isInteger()) {
      const auto *IntTy = cast<IntegerType>(EltTy);
      if (IntTy->getSizeInBits().getFixedValue() == 128)
        return true;
    }
  }
  return false;
}

ArgInfo X86_64TargetInfo::getIndirectResult(const Type *Ty,
                                            unsigned FreeIntRegs) const {
  // If this is a scalar LLVM value then assume LLVM will pass it in the right
  // place naturally.
  //
  // This assumption is optimistic, as there could be free registers available
  // when we need to pass this argument in memory, and LLVM could try to pass
  // the argument in the free register. This does not seem to happen currently,
  // but this code would be much safer if we could mark the argument with
  // 'onstack'. See PR12193.
  if (!isAggregateTypeForABI(Ty) && !isIllegalVectorType(Ty) &&
      !(Ty->isInteger() && cast<IntegerType>(Ty)->isBitInt())) {
    return (Ty->isInteger() && isPromotableInteger(cast<IntegerType>(Ty))
                ? ArgInfo::getExtend(Ty)
                : ArgInfo::getDirect());
  }

  // Check if this is a record type that needs special handling
  if (auto RecordRAA = getRecordArgABI(Ty))
    return getNaturalAlignIndirect(Ty, RecordRAA ==
                                           RecordArgABI::RAA_DirectInMemory);

  // Compute the byval alignment. We specify the alignment of the byval in all
  // cases so that the mid-level optimizer knows the alignment of the byval.
  uint64_t AlignVal = std::max<uint64_t>(Ty->getAlignment().value(), 8u);

  // Attempt to avoid passing indirect results using byval when possible. This
  // is important for good codegen.
  //
  // We do this by coercing the value into a scalar type which the backend can
  // handle naturally (i.e., without using byval).
  //
  // For simplicity, we currently only do this when we have exhausted all of the
  // free integer registers. Doing this when there are free integer registers
  // would require more care, as we would have to ensure that the coerced value
  // did not claim the unused register. That would require either reording the
  // arguments to the function (so that any subsequent inreg values came first),
  // or only doing this optimization when there were no following arguments that
  // might be inreg.
  //
  // We currently expect it to be rare (particularly in well written code) for
  // arguments to be passed on the stack when there are still free integer
  // registers available (this would typically imply large structs being passed
  // by value), so this seems like a fair tradeoff for now.
  //
  // We can revisit this if the backend grows support for 'onstack' parameter
  // attributes. See PR12193.
  if (FreeIntRegs == 0) {
    // Use the storage-container width (like Clang's getTypeSize) so a stack
    // _BitInt or illegal vector coerces to the integer covering its storage,
    // not its raw iN width.
    uint64_t Size = getClangTypeWidthInBits(Ty);

    // If this type fits in an eightbyte, coerce it into the matching integral
    // type, which will end up on the stack (with alignment 8).
    if (AlignVal == 8 && Size <= 64) {
      const Type *IntTy =
          TB.getIntegerType(Size, llvm::Align(8), /*Signed=*/false);
      return ArgInfo::getDirect(IntTy);
    }
  }

  return ArgInfo::getIndirect(llvm::Align(AlignVal), /*ByVal=*/true);
}

ArgInfo X86_64TargetInfo::getIndirectReturnResult(const Type *Ty) const {
  if (!isAggregateTypeForABI(Ty)) {
    // Bit-precise integers are returned indirectly regardless of size.
    if (const auto *IntTy = dyn_cast<IntegerType>(Ty)) {
      if (IntTy->isBitInt())
        return getNaturalAlignIndirect(IntTy, /*ByVal=*/true);
      if (isPromotableInteger(IntTy))
        return ArgInfo::getExtend(Ty);
    }
    return ArgInfo::getDirect();
  }

  return getNaturalAlignIndirect(Ty, /*ByVal=*/true);
}

static bool classifyCXXReturnType(FunctionInfo &FI) {
  const abi::Type *Ty = FI.getReturnType();

  if (const auto *RT = llvm::dyn_cast<abi::RecordType>(Ty)) {
    if (!RT->canPassInRegisters()) {
      // A C++ record that cannot pass in registers (non-trivial copy/dtor)
      // is returned indirectly with ByVal=false, matching
      // ItaniumCXXABI::classifyReturnType. This is the RAA path and is distinct
      // from getIndirectReturnResult (plain aggregates), which uses ByVal=true.
      FI.getReturnInfo() =
          ArgInfo::getIndirect(RT->getAlignment(), /*ByVal=*/false);
      return true;
    }
  }

  return false;
}

void X86_64TargetInfo::computeInfo(FunctionInfo &FI) const {
  CallingConv::ID CallingConv = FI.getCallingConvention();

  // Only the standard SysV (C) calling convention is classified here. Any other
  // convention must be added explicitly once it has been verified against this
  // classifier rather than silently taking the SysV path.
  switch (CallingConv) {
  case CallingConv::C:
    break;
  default:
    llvm_unreachable(
        "calling convention not supported by the LLVMABI X86_64 classifier");
  }

  unsigned FreeIntRegs = 6;
  unsigned FreeSSERegs = 8;
  unsigned NeededInt = 0, NeededSSE = 0;

  if (!classifyCXXReturnType(FI)) {
    const Type *RetTy = FI.getReturnType();
    FI.getReturnInfo() = classifyReturnType(RetTy);
  }

  if (FI.getReturnInfo().isIndirect())
    --FreeIntRegs;

  unsigned NumRequiredArgs = FI.getNumRequiredArgs();

  unsigned ArgNo = 0;
  for (auto IT = FI.arg_begin(), IE = FI.arg_end(); IT != IE; ++IT, ++ArgNo) {
    bool IsNamedArg = ArgNo < NumRequiredArgs;
    const Type *ArgTy = IT->ABIType;
    NeededInt = 0;
    NeededSSE = 0;

    ArgInfo AI = classifyArgumentType(ArgTy, FreeIntRegs, NeededInt, NeededSSE,
                                      IsNamedArg);

    // AMD64-ABI 3.2.3p3: If there are no registers available for any
    // eightbyte of an argument, the whole argument is passed on the
    // stack. If registers have already been assigned for some
    // eightbytes of such an argument, the assignments get reverted.
    if (FreeIntRegs >= NeededInt && FreeSSERegs >= NeededSSE) {
      FreeIntRegs -= NeededInt;
      FreeSSERegs -= NeededSSE;
      IT->Info = AI;
    } else {
      // Not enough registers, pass on stack
      IT->Info = getIndirectResult(ArgTy, FreeIntRegs);
    }
  }
}

std::unique_ptr<TargetInfo>
createX86_64TargetInfo(TypeBuilder &TB, X86AVXABILevel AVXLevel,
                       bool Has64BitPointers, const ABICompatInfo &Compat) {
  return std::make_unique<X86_64TargetInfo>(TB, AVXLevel, Has64BitPointers,
                                            Compat);
}

} // namespace abi
} // namespace llvm
