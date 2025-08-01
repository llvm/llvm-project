//===- X86.cpp ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/ABIFunctionInfo.h"
#include "llvm/ABI/ABIInfo.h"
#include "llvm/ABI/ABITypeMapper.h"
#include "llvm/ABI/TargetCodegenInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
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
  TypeBuilder &TB;
  X86AVXABILevel AVXLevel;
  bool Has64BitPointers;
  const llvm::Triple &TargetTriple;

  static Class merge(Class Accum, Class Field);

  void postMerge(unsigned AggregateSize, Class &Lo, Class &Hi) const;

  void classify(const Type *T, uint64_t OffsetBase, Class &Lo, Class &Hi,
                bool IsNamedArg, bool IsRegCall = false) const;

  // llvm::Type *getSseTypeAtOffset(llvm::Type *IRType, unsigned IROffset,
  //                                const Type *SourceTy,
  //                                unsigned SourceOffset) const;

  const Type *getIntegerTypeAtOffset(const Type *IRType, unsigned IROffset,
                                     const Type *SourceTy,
                                     unsigned SourceOffset) const;
  // llvm::Type *getIntegerTypeAtOffset(llvm::Type *IRType, unsigned IROffset,
  // 		const Type *SourceTy,
  // 		unsigned SourceOffset) const;
  // Type *getIntegerTypeForClass(const Type *OriginalType,
  //                                 uint64_t OffsetInBytes) const;

  const Type *getSSETypeAtOffset(const Type *ABIType, unsigned ABIOffset,
                                 const Type *SourceTy,
                                 unsigned SourceOffset) const;

  void computeInfo(ABIFunctionInfo &FI) const override;
  ABIArgInfo getIndirectReturnResult(const Type *Ty) const;
  const Type *getFPTypeAtOffset(const Type *Ty, unsigned Offset) const;

  const Type *isSingleElementStruct(const Type *Ty) const;
  const Type *getByteVectorType(const Type *Ty) const;

  const Type *createPairType(const Type *Lo, const Type *Hi) const;
  ABIArgInfo getIndirectResult(const Type *Ty, unsigned FreeIntRegs) const;

  ABIArgInfo classifyReturnType(const Type *RetTy) const override;
  const char *getClassName(Class C) const;

  // ABIArgInfo classifyArgumentType(const Type *Ty, unsigned FreeIntRegs,
  //                                 unsigned &NeededInt, unsigned &NeededSse,
  //                                 bool IsNamedArg,
  //                                 bool IsRegCall = false) const;

  // ABIArgInfo classifyRegCallStructType(const Type *Ty, unsigned &NeededInt,
  //                                      unsigned &NeededSSE,
  //                                      unsigned &MaxVectorWidth) const;
  //
  // ABIArgInfo classifyRegCallStructTypeImpl(const Type *Ty, unsigned
  // &NeededInt,
  //                                          unsigned &NeededSSE,
  //                                          unsigned &MaxVectorWidth) const;
  //
  // bool isIllegalVectorType(const Type *Ty) const;

  // The Functionality of these methods will be moved to
  // llvm::abi::ABICompatInfo

  // bool honorsRevision98() const { return !TargetTriple.isOSDarwin(); }
  //
  // bool classifyIntegerMMXAsSSE() const {
  //   if (TargetTriple.isOSDarwin() || TargetTriple.isPS() ||
  //       TargetTriple.isOSFreeBSD())
  //     return false;
  //   return true;
  // }
  //
  // bool passInt128VectorsInMem() const {
  //   // TODO: accept ABICompat info from the frontends
  //   return TargetTriple.isOSLinux() || TargetTriple.isOSNetBSD();
  // }
  //
  // bool returnCXXRecordGreaterThan128InMem() const {
  //   // TODO: accept ABICompat info from the frontends
  //   return true;
  // }

public:
  X86_64ABIInfo(TypeBuilder &TypeBuilder, const Triple &Triple,
                X86AVXABILevel AVXABILevel, bool Has64BitPtrs,
                const ABICompatInfo &Compat)
      : ABIInfo(Compat), TB(TypeBuilder), AVXLevel(AVXABILevel),
        Has64BitPointers(Has64BitPtrs), TargetTriple(Triple) {}

  // bool isPassedUsingAVXType(const Type *Type) const {
  //   unsigned NeededInt, NeededSse;
  //   ABIArgInfo Info = classifyArgumentType(Type, 0, NeededInt, NeededSse,
  //                                          /*IsNamedArg=*/true);
  //
  //   if (Info.isDirect()) {
  //     auto *Ty = Info.getCoerceToType();
  //     if (auto *VectorTy = dyn_cast_or_null<VectorType>(Ty))
  //       return VectorTy->getSizeInBits().getFixedValue() > 128;
  //   }
  //   return false;
  // }

  // void computeInfo(ABIFunctionInfo &FI) const override;

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

    if (BitWidth == 128 ||
        (IT->isBitInt() && BitWidth > 64 && BitWidth <= 128)) {
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

  if (const auto *MPT = dyn_cast<MemberPointerType>(T)) {
    if (MPT->isFunctionPointer()) {
      if (MPT->has64BitPointers()) {
        Lo = Hi = Integer;
      } else {
        uint64_t EB_FuncPtr = OffsetBase / 64;
        uint64_t EB_ThisAdj = (OffsetBase + 64 - 1) / 64;
        if (EB_FuncPtr != EB_ThisAdj) {
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
      uint64_t EB_Lo = (OffsetBase) / 64;
      uint64_t EB_Hi = (OffsetBase + Size - 1) / 64;
      if (EB_Lo != EB_Hi)
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
        if (!getABICompatInfo().Flags.ClassifyIntegerMMXAsSSE &&
            (ElemBits == 64 || ElemBits == 32)) {
          Current = Integer;
        } else {
          Current = SSE;
        }
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
        if (getABICompatInfo().Flags.PassInt128VectorsInMem && Size != 128 &&
            ElemBits == 128)
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
      Lo = SSE;
      Hi = SSEUp;
    }
    return;
  }

  if (const auto *CT = dyn_cast<ComplexType>(T)) {
    const Type *ElementType = CT->getElementType();
    uint64_t Size = T->getSizeInBits().getFixedValue();

    if (const auto *EIT = dyn_cast<IntegerType>(ElementType)) {
      if (Size <= 64)
        Current = Integer;
      else if (Size <= 128)
        Lo = Hi = Integer;
    } else if (const auto *EFT = dyn_cast<FloatType>(ElementType)) {
      const auto *FltSem = EFT->getSemantics();
      if (FltSem == &llvm::APFloat::IEEEhalf() ||
          FltSem == &llvm::APFloat::IEEEsingle() ||
          FltSem == &llvm::APFloat::BFloat())
        Current = SSE;
      else if (FltSem == &llvm::APFloat::IEEEquad())
        Current = Memory;
      else if (FltSem == &llvm::APFloat::x87DoubleExtended())
        Current = Complex_X87;
      else if (FltSem == &llvm::APFloat::IEEEdouble())
        Lo = Hi = SSE;
      else
        llvm_unreachable("Unexpected long double representation!");
    }

    uint64_t ElementSize = ElementType->getSizeInBits().getFixedValue();
    // If this complex type crosses an eightbyte boundary then it
    // should be split.
    uint64_t EB_Real = OffsetBase / 64;
    uint64_t EB_Imag = (OffsetBase + ElementSize) / 64;
    if (Hi == NoClass && EB_Real != EB_Imag)
      Hi = Lo;

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


    // AMD64-ABI 3.2.3p2: Rule 1. If the size of an object is larger
    // than eight eightbytes, ..., it has class MEMORY.
    if (Size > 512)
      return;

    // AMD64-ABI 3.2.3p2: Rule 2. If a C++ object has either a non-trivial
    // copy constructor or a non-trivial destructor, it is passed by invisible
    // reference.
    if (ST->isCXXRecord() && (getRecordArgABI(ST))) {
      return;
    }

    // Assume variable sized types are passed in memory.
    if (ST->hasFlexibleArrayMember()) {
      return;
    }
    // Reset Lo class, this will be recomputed.
    Current = NoClass;

    // If this is a C++ record, classify the bases first.
    if (ST->isCXXRecord()) {
      const FieldInfo *BaseClasses = ST->getBaseClasses();
      for (uint32_t I = 0; I < ST->getNumBaseClasses(); ++I) {
        const FieldInfo &Base = BaseClasses[I];

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

        if (getABICompatInfo().Flags.ReturnCXXRecordGreaterThan128InMem &&
            (Size > 128 &&
             (Size != Base.FieldType->getSizeInBits().getFixedValue() ||
              Size > getNativeVectorSizeForAVXABI(AVXLevel)))) {
          Lo = Memory;
          postMerge(Size, Lo, Hi);
          return;
        }

        if (Lo == Memory || Hi == Memory) {
          postMerge(Size, Lo, Hi);
          return;
        }
      }
    }

    // Classify the fields one at a time, merging the results.
    const FieldInfo *Fields = ST->getFields();
    uint32_t NumFields = ST->getNumFields();

    for (uint32_t I = 0; I < NumFields; ++I) {
      const FieldInfo &Field = Fields[I];
      uint64_t Offset = OffsetBase + Field.OffsetInBits;
      bool BitField = Field.IsBitField;

	  if (BitField && Field.IsUnnamedBitfield)
		  continue;

      if (Size > 128 && (Size > getNativeVectorSizeForAVXABI(AVXLevel))) {
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
      uint64_t Offset = OffsetBase;

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

ABIArgInfo X86_64ABIInfo::classifyReturnType(const Type *RetTy) const {
  // AMD64-ABI 3.2.3p4: Rule 1. Classify the return type with the
  // classification algorithm.

  X86_64ABIInfo::Class Lo, Hi;
  classify(RetTy, 0, Lo, Hi, /*isNamedArg*/ true);

  // Check some invariants
  assert((Hi != Memory || Lo == Memory) && "Invalid memory classification.");
  assert((Hi != SSEUp || Lo == SSE) && "Invalid SSEUp classification.");

  const Type *ResType = nullptr;
  switch (Lo) {
  case NoClass:
    if (Hi == NoClass)
      return ABIArgInfo::getIgnore();
    // If the low part is just padding, it takes no register, leave ResType
    // null.
    assert((Hi == SSE || Hi == Integer || Hi == X87UP) &&
           "Unknown missing lo part");
    break;
  case SSEUp:
  case X87UP:
    llvm_unreachable("Invalid classification for lo word.");

    // AMD64-ABI 3.2.3p4: Rule 2. Types of class memory are returned via
    // hidden argument.
  case Memory:
    return getIndirectReturnResult(RetTy);

    // AMD64-ABI 3.2.3p4: Rule 3. If the class is INTEGER, the next
    // available register of the sequence %rax, %rdx is used.
  case Integer:
    ResType = getIntegerTypeAtOffset(RetTy, 0, RetTy, 0);
    if (const auto *IT = dyn_cast<IntegerType>(ResType)) {
      if (IT->isBool() && RetTy->isInteger() &&
          cast<IntegerType>(RetTy)->isBool()) {
        // Convert boolean to i1 for LLVM IR
        ResType = TB.getIntegerType(1, IT->getAlignment(), false, true);
        return ABIArgInfo::getExtend(ResType);
      }
    }
    // If we have a sign or zero extended integer, make sure to return Extend
    // so that the parameter gets the right LLVM IR attributes.
    if (Hi == NoClass && ResType->isInteger()) {
      if (const IntegerType *IntTy = dyn_cast<IntegerType>(RetTy)) {
        if (IntTy && isPromotableIntegerType(IntTy)) {
          ABIArgInfo Info = ABIArgInfo::getExtend(RetTy);
          return Info;
        }
      }
    }
    if (ResType->isInteger() && ResType->getSizeInBits() == 128) {
      assert(Hi == Integer);
      return ABIArgInfo::getDirect(ResType);
    }
    break;

    // AMD64-ABI 3.2.3p4: Rule 4. If the class is SSE, the next
    // available SSE register of the sequence %xmm0, %xmm1 is used.
  case SSE:
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
  case Complex_X87:
    assert(Hi == Complex_X87 && "Unexpected ComplexX87 classification.");
    {
      const Type *X87Type =
          TB.getFloatType(APFloat::x87DoubleExtended(), Align(16));
      FieldInfo Fields[] = {
          FieldInfo(X87Type, 0), FieldInfo(X87Type, 128) // 128 bits offset
      };
      ResType =
          TB.getCoercedStructType(Fields, TypeSize::getFixed(256), Align(16));
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

  case Complex_X87:
  case NoClass:
    break;

  case Integer:
    HighPart = getIntegerTypeAtOffset(RetTy, 8, RetTy, 8);
    if (Lo == NoClass)
      return ABIArgInfo::getDirect(HighPart, 8);
    break;

  case SSE:
    HighPart = getSSETypeAtOffset(RetTy, 8, RetTy, 8);
    if (Lo == NoClass)
      return ABIArgInfo::getDirect(HighPart, 8);
    break;

    // AMD64-ABI 3.2.3p4: Rule 5. If the class is SSEUP, the eightbyte
    // is passed in the next available eightbyte chunk if the last used
    // vector register.
    //
    // SSEUP should always be preceded by SSE, just widen.
  case SSEUp:
    assert(Lo == SSE && "Unexpected SSEUp classification.");
    ResType = getByteVectorType(RetTy);
    break;

    // AMD64-ABI 3.2.3p4: Rule 7. If the class is X87UP, the value is
    // returned together with the previous X87 value in %st0.
  case X87UP:
    // If X87Up is preceded by X87, we don't need to do
    // anything. However, in some cases with unions it may not be
    // preceded by X87. In such situations we follow gcc and pass the
    // extra bits in an SSE reg.
    if (Lo != X87) {
      HighPart = getSSETypeAtOffset(RetTy, 8, RetTy, 8);
      if (Lo == NoClass) // Return HighPart at offset 8 in memory.
        return ABIArgInfo::getDirect(HighPart, 8);
    }
    break;
  }

  // If a high part was specified, merge it together with the low part.  It is
  // known to pass in the high eightbyte of the result.  We do this by forming a
  // first class struct aggregate with the high and low part: {low, high}
  if (HighPart)
    ResType = createPairType(ResType, HighPart);

  return ABIArgInfo::getDirect(ResType);
}

/// GetX86_64ByValArgumentPair - Given a high and low type that can ideally
/// be used as elements of a two register pair to pass or return, return a
/// first class aggregate to represent them.  For example, if the low part of
/// a by-value argument should be passed as i32* and the high part as float,
/// return {i32*, float}.
const Type *X86_64ABIInfo::createPairType(const Type *Lo,
                                          const Type *Hi) const {
  // In order to correctly satisfy the ABI, we need to the high part to start
  // at offset 8.  If the high and low parts we inferred are both 4-byte types
  // (e.g. i32 and i32) then the resultant struct type ({i32,i32}) won't have
  // the second element at offset 8.  Check for this:
  unsigned LoSize = (unsigned)Lo->getTypeAllocSize();
  Align HiAlign = Hi->getAlignment();
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
          FT->getSemantics() == &APFloat::BFloat()) {
        AdjustedLo = TB.getFloatType(APFloat::IEEEdouble(), Align(8));
      }
    }
    // Promote integers and pointers to i64
    else if (Lo->isInteger() || Lo->isPointer()) {
      AdjustedLo = TB.getIntegerType(64, Align(8), /*Signed=*/false);
    } else {
      assert((Lo->isInteger() || Lo->isPointer()) &&
             "Invalid/unknown low type in pair");
    }
    unsigned AdjustedLoSize = AdjustedLo->getSizeInBits().getFixedValue() / 8;
    HiStart = alignTo(AdjustedLoSize, HiAlign);
  }

  // Create the pair struct
  FieldInfo Fields[] = {
      FieldInfo(AdjustedLo, 0),  // Low part at offset 0
      FieldInfo(Hi, HiStart * 8) // High part at offset 8 bytes (64 bits)
  };

  // Verify the high part is at offset 8
  assert((8 * 8) == Fields[1].OffsetInBits &&
         "High part must be at offset 8 bytes");

  return TB.getCoercedStructType(Fields,
                                 TypeSize::getFixed(128), // Total size 16 bytes
                                 Align(8),                // Natural alignment
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
        break; // Elements are sorted by offset

      unsigned EltStart = (EltOffset < StartBit) ? StartBit - EltOffset : 0;
      if (!bitsContainNoUserData(EltTy, EltStart, EndBit - EltOffset))
        return false;
    }
    return true;
  }

  // Handle structs - check all fields and base classes
  if (const StructType *ST = dyn_cast<StructType>(Ty)) {
    // Check base classes first (for C++ records)
    if (ST->isCXXRecord()) {
      for (unsigned I = 0; I < ST->getNumBaseClasses(); ++I) {
        const FieldInfo &Base = ST->getBaseClasses()[I];
        if (Base.OffsetInBits >= EndBit)
          continue;

        unsigned BaseStart =
            (Base.OffsetInBits < StartBit) ? StartBit - Base.OffsetInBits : 0;
        if (!bitsContainNoUserData(Base.FieldType, BaseStart,
                                   EndBit - Base.OffsetInBits))
          return false;
      }
    }

    // Check all fields
    for (unsigned I = 0; I < ST->getNumFields(); ++I) {
      const FieldInfo &Field = ST->getFields()[I];
      if (Field.OffsetInBits >= EndBit)
        break; // Fields are sorted by offset

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

const Type *X86_64ABIInfo::getIntegerTypeAtOffset(const Type *ABIType,
                                                  unsigned ABIOffset,
                                                  const Type *SourceTy,
                                                  unsigned SourceOffset) const {
  // If we're dealing with an un-offset ABI type, then it means that we're
  // returning an 8-byte unit starting with it.  See if we can safely use it.
  if (ABIOffset == 0) {
    // Pointers and 64-bit integers fill the 8-byte unit
    if ((ABIType->isPointer() && Has64BitPointers) ||
        (ABIType->isInteger() &&
         cast<IntegerType>(ABIType)->getSizeInBits() == 64)){
      return ABIType;
	}

    // If we have a 1/2/4-byte integer, we can use it only if the rest of the
    // goodness in the source type is just tail padding.  This is allowed to
    // kick in for struct {double,int} on the int, but not on
    // struct{double,int,int} because we wouldn't return the second int.  We
    // have to do this analysis on the source type because we can't depend on
    // unions being lowered a specific way etc.
    if (ABIType->isInteger()) {
      unsigned BitWidth = cast<IntegerType>(ABIType)->getSizeInBits();
      if (BitWidth == 8 || BitWidth == 16 || BitWidth == 32) {
        // Check if the rest is just padding
        if (bitsContainNoUserData(SourceTy, SourceOffset * 8 + BitWidth,
                                  SourceOffset * 8 + 64))
          return ABIType;
      }
    } else if (ABIType->isPointer() && !Has64BitPointers) {
      // Check if the rest is just padding
      if (bitsContainNoUserData(SourceTy, SourceOffset * 8 + 32,
                                SourceOffset * 8 + 64))
        return ABIType;
    }
  }

  // Handle structs by recursing into fields
  if (auto *STy = dyn_cast<StructType>(ABIType)) {
    const FieldInfo *Fields = STy->getFields();

    // Find field containing the IROffset
    for (unsigned I = 0; I < STy->getNumFields(); ++I) {
      const FieldInfo &Field = Fields[I];
      unsigned FieldOffsetBytes = Field.OffsetInBits / 8;
      unsigned FieldSizeBytes = Field.FieldType->getSizeInBits() / 8;

      // Check if IROffset falls within this field
      if (ABIOffset >= FieldOffsetBytes &&
          ABIOffset < FieldOffsetBytes + FieldSizeBytes) {
        return getIntegerTypeAtOffset(Field.FieldType,
                                      ABIOffset - FieldOffsetBytes, SourceTy,
                                      SourceOffset);
      }
    }
  }

  // Handle arrays
  if (auto *ATy = dyn_cast<ArrayType>(ABIType)) {
    const Type *EltTy = ATy->getElementType();
    unsigned EltSize = EltTy->getSizeInBits() / 8;
    if (EltSize > 0) { // Avoid division by zero
      unsigned EltOffset = (ABIOffset / EltSize) * EltSize;
      return getIntegerTypeAtOffset(EltTy, ABIOffset - EltOffset, SourceTy,
                                    SourceOffset);
    }
  }

  if (ABIType->isInteger() && ABIType->getSizeInBits().getFixedValue() == 128) {
    assert(ABIOffset == 0);
    return ABIType;
  }

  // Default case - use integer type that fits
  unsigned TySizeInBytes =
      SourceTy->getSizeInBits().getFixedValue() / 8;
  if (auto *IT = dyn_cast<IntegerType>(SourceTy)){
	  if (IT->isBitInt())
		  TySizeInBytes = alignTo(SourceTy->getSizeInBits().getFixedValue(),64)/8;
  }
  assert(TySizeInBytes != SourceOffset && "Empty field?");
  unsigned AvailableSize = TySizeInBytes - SourceOffset;
  return TB.getIntegerType(std::min(AvailableSize, 8U) * 8, Align(1),
                           /*Signed=*/false);
}
/// Returns the floating point type at the specified offset within a type, or
/// nullptr if no floating point type is found at that offset.
const Type *X86_64ABIInfo::getFPTypeAtOffset(const Type *Ty,
                                             unsigned Offset) const {
  // Check for direct match at offset 0
  if (Offset == 0 && Ty->isFloat()) {
    return Ty;
  }

  if (const ComplexType *CT = dyn_cast<ComplexType>(Ty)) {
    const Type *ElementType = CT->getElementType();
    unsigned ElementSize = ElementType->getSizeInBits().getFixedValue() / 8;

    if (Offset == 0 || Offset == ElementSize) {
      return ElementType;
    }
    return nullptr;
  }

  // Handle struct types by checking each field
  if (const StructType *ST = dyn_cast<StructType>(Ty)) {
    if (!ST->getNumFields())
      return nullptr;
    const FieldInfo *Fields = ST->getFields();

    // Find the field containing the requested offset
    for (unsigned i = 0; i < ST->getNumFields(); ++i) {
      unsigned FieldOffset = Fields[i].OffsetInBits / 8;
      unsigned FieldSize =
          Fields[i].FieldType->getSizeInBits().getFixedValue() / 8;

      // Check if offset falls within this field
      if (Offset >= FieldOffset && Offset < FieldOffset + FieldSize) {
        return getFPTypeAtOffset(Fields[i].FieldType, Offset - FieldOffset);
      }
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
const Type *X86_64ABIInfo::getSSETypeAtOffset(const Type *ABIType,
                                              unsigned ABIOffset,
                                              const Type *SourceTy,
                                              unsigned SourceOffset) const {

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
  if (SourceSize > T0Size) {
    T1 = getFPTypeAtOffset(ABIType, ABIOffset + T0Size);
  }

  if (T1 == nullptr) {
    if (Is16bitFpTy(T0) && SourceSize > 4)
      T1 = getFPTypeAtOffset(ABIType, ABIOffset + 4);

    if (T1 == nullptr)
      return T0;
  }
  // Handle vector cases
  if (isFloatTypeWithSemantics(T0, APFloat::IEEEsingle()) &&
      isFloatTypeWithSemantics(T1, APFloat::IEEEsingle())) {
    return TB.getVectorType(T0, ElementCount::getFixed(2), Align(8));
  }
  if (Is16bitFpTy(T0) && Is16bitFpTy(T1)) {
    const Type *T2 = nullptr;
    if (SourceSize > 4)
      T2 = getFPTypeAtOffset(ABIType, ABIOffset + 4);
    if (!T2)
      return TB.getVectorType(T0, ElementCount::getFixed(2), Align(8));
    return TB.getVectorType(T0, ElementCount::getFixed(4), Align(8));
  }

  // Mixed half-float cases
  if (Is16bitFpTy(T0) || Is16bitFpTy(T1)) {
    return TB.getVectorType(TB.getFloatType(APFloat::IEEEhalf(), Align(2)),
                            ElementCount::getFixed(4), Align(8));
  }

  // Default to double
  return TB.getFloatType(APFloat::IEEEdouble(), Align(8));
}

/// The ABI specifies that a value should be passed in a full vector XMM/YMM
/// register. Pick an LLVM IR type that will be passed as a vector register.
const Type *X86_64ABIInfo::getByteVectorType(const Type *Ty) const {
  // Wrapper structs/arrays that only contain vectors are passed just like
  // vectors; strip them off if present.
  if (const Type *InnerTy = isSingleElementStruct(Ty))
    Ty = InnerTy;

  // Handle vector types
  if (const VectorType *VT = dyn_cast<VectorType>(Ty)) {
    // Don't pass vXi128 vectors in their native type, the backend can't
    // legalize them.
    if (getABICompatInfo().Flags.PassInt128VectorsInMem &&
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
const Type *X86_64ABIInfo::isSingleElementStruct(const Type *Ty) const {
  const auto *ST = dyn_cast<StructType>(Ty);
  if (!ST)
    return nullptr;

  if (ST->isPolymorphic()                       ||
      ST->hasNonTrivialCopyConstructor()        ||
      ST->hasNonTrivialDestructor()             ||
      ST->hasFlexibleArrayMember()              ||
      ST->getNumVirtualBaseClasses() != 0)        
    return nullptr;

  const Type *Found = nullptr;

  for (unsigned I = 0; I < ST->getNumBaseClasses(); ++I) {
    const Type *BaseTy = ST->getBaseClasses()[I].FieldType;
    auto *BaseST = dyn_cast<StructType>(BaseTy);

    if (!BaseST || isEmptyRecord(BaseST))
      continue;                               // ignore empty bases

    const Type *Elem = isSingleElementStruct(BaseTy);
    if (!Elem || Found)
      return nullptr;
    Found = Elem;
  }

  for (unsigned I = 0; I < ST->getNumFields(); ++I) {
    const FieldInfo &FI = ST->getFields()[I];
    if (isEmptyField(FI))
      continue;

    const Type *FTy = FI.FieldType;

    while (auto *AT = dyn_cast<ArrayType>(FTy)) {
      if (AT->getNumElements() != 1)
        break;
      FTy = AT->getElementType();
    }

    const Type *Elem;
    if (auto *InnerST = dyn_cast<StructType>(FTy))
      Elem = isSingleElementStruct(InnerST);
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


ABIArgInfo X86_64ABIInfo::getIndirectReturnResult(const Type *Ty) const {
  // If this is a scalar value, handle it specially
  if (!isAggregateTypeForABI(Ty)) {
    // Handle integer types that need extension
    if (Ty->isInteger()) {
      const IntegerType *IntTy = cast<IntegerType>(Ty);
      if (isPromotableIntegerType(IntTy)) {
        ABIArgInfo Info = ABIArgInfo::getExtend(Ty);
        return Info;
      }
	  if (IntTy->isBitInt())
		  return getNaturalAlignIndirect(IntTy);
    }
    return ABIArgInfo::getDirect();
  }

  // For aggregate types or other cases, return as indirect
  return getNaturalAlignIndirect(Ty);
}

void X86_64ABIInfo::computeInfo(ABIFunctionInfo &FI) const {
  // TODO: Implement proper function info computation
}
class X8664TargetCodeGenInfo : public TargetCodeGenInfo {
public:
  X8664TargetCodeGenInfo(TypeBuilder &TB, const Triple &Triple,
                         X86AVXABILevel AVXLevel, bool Has64BitPointers,
                         const ABICompatInfo &Compat)
      : TargetCodeGenInfo(std::make_unique<X86_64ABIInfo>(
            TB, Triple, AVXLevel, Has64BitPointers, Compat)) {}
};

std::unique_ptr<TargetCodeGenInfo>
createX8664TargetCodeGenInfo(TypeBuilder &TB, const Triple &Triple,
                             X86AVXABILevel AVXLevel, bool Has64BitPointers,
                             const ABICompatInfo &Compat) {
  return std::make_unique<X8664TargetCodeGenInfo>(TB, Triple, AVXLevel,
                                                  Has64BitPointers, Compat);
}
} // namespace abi
} // namespace llvm
