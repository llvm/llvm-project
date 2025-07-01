//===- X86.cpp ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/ABIFunctionInfo.h"
#include "llvm/ABI/ABIInfo.h"
#include "llvm/ABI/TargetCodegenInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/TargetParser/Triple.h"
#include <cstdint>

namespace llvm {
namespace abi {

enum class AVXABILevel { None, AVX, AVX512 };

static unsigned getNativeVectorSizeForAVXABI(AVXABILevel AVXLevel) {
  switch (AVXLevel) {
  case AVXABILevel::AVX512:
    return 512;
  case AVXABILevel::AVX:
    return 256;
  case AVXABILevel::None:
    return 128;
  }
  llvm_unreachable("Unknown AVXLevel");
}

class X86_64ABIInfo : public ABIInfo {
public:
  enum Class {
    Integer = 0,
    SSE,
    SSEUp,
    X87,
    X87UP,
    Complex_X87,
    NoClass,
    Memory
  };

private:
  AVXABILevel AVXLevel;
  bool Has64BitPointers;
  const llvm::Triple &TargetTriple;

  static Class merge(Class Accum, Class Field);

  void postMerge(unsigned AggregateSize, Class &Lo, Class &Hi) const;

  void classify(const Type *T, uint64_t OffsetBase, Class &Lo, Class &Hi,
                bool IsNamedArg, bool IsRegCall = false) const;

  llvm::Type *getByteVectorType(const Type *Ty) const;
  llvm::Type *getSseTypeAtOffset(llvm::Type *IRType, unsigned IROffset,
                                 const Type *SourceTy,
                                 unsigned SourceOffset) const;

  llvm::Type *getIntegerTypeAtOffset(llvm::Type *IRType, unsigned IROffset,
                                     const Type *SourceTy,
                                     unsigned SourceOffset) const;

  ABIArgInfo getIndirectReturnResult(const Type *Ty) const;

  ABIArgInfo getIndirectResult(const Type *Ty, unsigned FreeIntRegs) const;

  ABIArgInfo classifyReturnType(const Type *RetTy) const override;

  ABIArgInfo classifyArgumentType(const Type *Ty, unsigned FreeIntRegs,
                                  unsigned &NeededInt, unsigned &NeededSse,
                                  bool IsNamedArg,
                                  bool IsRegCall = false) const;

  ABIArgInfo classifyRegCallStructType(const Type *Ty, unsigned &NeededInt,
                                       unsigned &NeededSSE,
                                       unsigned &MaxVectorWidth) const;

  ABIArgInfo classifyRegCallStructTypeImpl(const Type *Ty, unsigned &NeededInt,
                                           unsigned &NeededSSE,
                                           unsigned &MaxVectorWidth) const;

  bool isIllegalVectorType(const Type *Ty) const;

  // The Functionality of these methods will be moved to
  // llvm::abi::ABICompatInfo

  bool honorsRevision98() const { return !TargetTriple.isOSDarwin(); }

  bool classifyIntegerMMXAsSSE() const {
    if (TargetTriple.isOSDarwin() || TargetTriple.isPS() ||
        TargetTriple.isOSFreeBSD())
      return false;
    return true;
  }

  bool passInt128VectorsInMem() const {
    // TODO: accept ABICompat info from the frontends
    return TargetTriple.isOSLinux() || TargetTriple.isOSNetBSD();
  }

  bool returnCXXRecordGreaterThan128InMem() const {
    // TODO: accept ABICompat info from the frontends
    return true;
  }

public:
  X86_64ABIInfo(const Triple &Triple, AVXABILevel AVXABILevel,
                bool Has64BitPtrs, const ABICompatInfo &Compat)
      : ABIInfo(Compat), AVXLevel(AVXABILevel), Has64BitPointers(Has64BitPtrs),
        TargetTriple(Triple) {}

  bool isPassedUsingAVXType(const Type *Type) const {
    unsigned NeededInt, NeededSse;
    ABIArgInfo Info = classifyArgumentType(Type, 0, NeededInt, NeededSse, true);

    if (Info.isDirect()) {
      auto *Ty = Info.getCoerceToType();
      if (auto *VectorTy = dyn_cast_or_null<VectorType>(Ty))
        return VectorTy->getSizeInBits().getFixedValue();
    }
    return false;
  }

  void computeInfo(ABIFunctionInfo &FI) const override;

  bool has64BitPointers() const { return Has64BitPointers; }
};

void X86_64ABIInfo::postMerge(unsigned AggregateSize, Class &Lo,
                              Class &Hi) const {
  // AMD64-ABI 3.2.3p2: Rule 5. Then a post merger cleanup is done:
  //
  // (a) If one of the classes is Memory, the whole argument is passed in
  //     memory.
  //
  // (b) If X87UP is not preceded by X87, the whole argument is passed in
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
  if (Hi == X87UP && Lo != X87 && getABICompatInfo().Flags.HonorsRevision98)
    Lo = Memory;
  if (AggregateSize > 128 && (Lo != SSE && Hi != SSEUp))
    Lo = Memory;
  if (Hi == SSEUp && Lo != SSE)
    Hi = SSE;
}
X86_64ABIInfo::Class X86_64ABIInfo::merge(Class Accum, Class Field) {
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
  // (e) If one of the classes is X87, X87UP, COMPLEX_X87 class,
  // MEMORY is used as class.
  //
  // (f) Otherwise class SSE is used.

  // Accum should never be memory (we should have returned) or
  // ComplexX87 (because this cannot be passed in a structure).
  assert((Accum != Memory && Accum != Complex_X87) &&
         "Invalid accumulated classification during merge.");

  if (Accum == Field || Field == NoClass)
    return Accum;
  if (Accum == NoClass)
    return Field;
  if (Field == Memory)
    return Memory;
  if (Accum == Integer || Field == Integer)
    return Integer;
  if (Field == X87 || Field == X87UP || Field == Complex_X87 || Accum == X87 ||
      Accum == X87UP)
    return Memory;

  return SSE;
}
void X86_64ABIInfo::classify(const Type *T, uint64_t OffsetBase, Class &Lo,
                             Class &Hi, bool IsNamedArg, bool IsRegCall) const {
  Lo = Hi = NoClass;
  Class &Current = OffsetBase < 64 ? Lo : Hi;
  Current = Memory;

  if (T->isVoid()) {
    Current = NoClass;
    return;
  }

  if (const auto *IT = dyn_cast<IntegerType>(T)) {
    auto BitWidth = IT->getSizeInBits().getFixedValue();

    if (BitWidth == 128) {
      Lo = Integer;
      Hi = Integer;
    } else if (BitWidth <= 64)
      Current = Integer;

    return;
  }

  if (const auto *FT = dyn_cast<FloatType>(T)) {
    const auto *FltSem = FT->getSemantics();

    if (FltSem == &llvm::APFloat::IEEEsingle() ||
        FltSem == &llvm::APFloat::IEEEdouble() ||
        FltSem == &llvm::APFloat::IEEEhalf() ||
        FltSem == &llvm::APFloat::BFloat()) {
      Current = SSE;
    } else if (FltSem == &llvm::APFloat::IEEEquad()) {
      Lo = SSE;
      Hi = SSEUp;
    } else if (FltSem == &llvm::APFloat::x87DoubleExtended()) {
      Lo = X87;
      Hi = X87UP;
    } else {
      Current = SSE;
    }
    return;
  }

  if (T->isPointer()) {
    Current = Integer;
    return;
  }

  if (const auto *VT = dyn_cast<VectorType>(T)) {
    auto Size = VT->getSizeInBits().getFixedValue();
    const Type *ElementType = VT->getElementType();

    if (Size == 1 || Size == 8 || Size == 16 || Size == 32) {
      Current = Integer;
      uint64_t EB_Lo = (OffsetBase) / 64;
      uint64_t EB_Hi = (OffsetBase + Size - 1) / 64;
      if (EB_Lo != EB_Hi)
        Hi = Lo;
    } else if (Size == 64) {
      if (const auto *FT = dyn_cast<FloatType>(ElementType)) {
        if (FT->getSemantics() == &llvm::APFloat::IEEEdouble())
          return;
      }

      if (const auto *IT = dyn_cast<IntegerType>(ElementType)) {
        uint64_t ElemBits = IT->getSizeInBits().getFixedValue();
        if (!getABICompatInfo().Flags.ClassifyIntegerMMXAsSSE &&
            (ElemBits == 64 || ElemBits == 32)) {
          Current = Integer;
        } else {
          Current = SSE;
        }
      } else {
        Current = SSE;
      }
      if (OffsetBase && OffsetBase != 64)
        Hi = Lo;
    } else if (Size == 128 ||
               (IsNamedArg && Size <= getNativeVectorSizeForAVXABI(AVXLevel))) {
      if (const auto *IT = dyn_cast<IntegerType>(ElementType)) {
        uint64_t ElemBits = IT->getSizeInBits().getFixedValue();
        if (getABICompatInfo().Flags.PassInt128VectorsInMem && Size != 128 &&
            ElemBits == 128)
          return;
      }

      Lo = SSE;
      Hi = SSEUp;
    }
    return;
  }

  if (const auto *AT = dyn_cast<ArrayType>(T)) {
    uint64_t Size = AT->getSizeInBits().getFixedValue();

    if (!IsRegCall && Size > 512)
      return;

    const Type *ElementType = AT->getElementType();
    uint64_t ElemAlign = ElementType->getAlignment().value() * 8;
    if (OffsetBase % ElemAlign)
      return;

    Current = NoClass;
    uint64_t EltSize = ElementType->getSizeInBits().getFixedValue();
    uint64_t ArraySize = AT->getNumElements();

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
    assert((Hi != SSEUp || Lo == SSE) && "Invalid SSEUp array classification.");
    return;
  }
  if (const auto *ST = dyn_cast<StructType>(T)) {
    uint64_t Size = ST->getSizeInBits().getFixedValue();

    if (Size > 512)
      return;

    Current = NoClass;

    const FieldInfo *Fields = ST->getFields();
    uint32_t NumFields = ST->getNumFields();

    for (uint32_t I = 0; I < NumFields; ++I) {
      const FieldInfo &Field = Fields[I];
      uint64_t Offset = OffsetBase + Field.OffsetInBits;
      bool BitField = Field.IsBitField;

      if (Size > 128 &&
          Size != Field.FieldType->getSizeInBits().getFixedValue() &&
          Size > getNativeVectorSizeForAVXABI(AVXLevel)) {
        Lo = Memory;
        postMerge(Size, Lo, Hi);
        return;
      }
      if (!BitField) {
        uint64_t FieldAlign = Field.FieldType->getAlignment().value() * 8;
        if (Offset % FieldAlign) {
          Lo = Memory;
          postMerge(Size, Lo, Hi);
          return;
        }
      }

      Class FieldLo, FieldHi;

      if (BitField) {
        uint64_t BitFieldSize = Field.BitFieldWidth;
        uint64_t EB_Lo = Offset / 64;
        uint64_t EB_Hi = (Offset + BitFieldSize - 1) / 64;

        if (EB_Lo) {
          assert(EB_Hi == EB_Lo && "Invalid classification, type > 16 bytes.");
          FieldLo = NoClass;
          FieldHi = Integer;
        } else {
          FieldLo = Integer;
          FieldHi = EB_Hi ? Integer : NoClass;
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
  if (const auto *UT = dyn_cast<UnionType>(T)) {
    uint64_t Size = UT->getSizeInBits().getFixedValue();

    if (Size > 512)
      return;

    Current = NoClass;

    const FieldInfo *Fields = UT->getFields();
    uint32_t NumFields = UT->getNumFields();

    for (uint32_t I = 0; I < NumFields; ++I) {
      const FieldInfo &Field = Fields[I];
      uint64_t Offset = OffsetBase + Field.OffsetInBits;

      Class FieldLo, FieldHi;
      classify(Field.FieldType, Offset, FieldLo, FieldHi, IsNamedArg);
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

} // namespace abi
} // namespace llvm
