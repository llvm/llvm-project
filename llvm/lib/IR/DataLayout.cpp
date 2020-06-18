//===- DataLayout.cpp - Data size & alignment routines ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines layout properties related to datatype size/offset/alignment
// information.
//
// This structure should be created once, filled in if the defaults are not
// correct and then passed around by const&.  None of the members functions
// require modification to the object.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DataLayout.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TypeSize.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <utility>

using namespace llvm;

//===----------------------------------------------------------------------===//
// Support for StructLayout
//===----------------------------------------------------------------------===//

StructLayout::StructLayout(StructType *ST, const DataLayout &DL) {
  assert(!ST->isOpaque() && "Cannot get layout of opaque structs");
  StructSize = 0;
  IsPadded = false;
  NumElements = ST->getNumElements();

  // Loop over each of the elements, placing them in memory.
  for (unsigned i = 0, e = NumElements; i != e; ++i) {
    Type *Ty = ST->getElementType(i);
    const Align TyAlign(ST->isPacked() ? 1 : DL.getABITypeAlignment(Ty));

    // Add padding if necessary to align the data element properly.
    if (!isAligned(TyAlign, StructSize)) {
      IsPadded = true;
      StructSize = alignTo(StructSize, TyAlign);
    }

    // Keep track of maximum alignment constraint.
    StructAlignment = std::max(TyAlign, StructAlignment);

    MemberOffsets[i] = StructSize;
    StructSize += DL.getTypeAllocSize(Ty); // Consume space for this data item
  }

  // Add padding to the end of the struct so that it could be put in an array
  // and all array elements would be aligned correctly.
  if (!isAligned(StructAlignment, StructSize)) {
    IsPadded = true;
    StructSize = alignTo(StructSize, StructAlignment);
  }
}

/// getElementContainingOffset - Given a valid offset into the structure,
/// return the structure index that contains it.
unsigned StructLayout::getElementContainingOffset(uint64_t Offset) const {
  const uint64_t *SI =
    std::upper_bound(&MemberOffsets[0], &MemberOffsets[NumElements], Offset);
  assert(SI != &MemberOffsets[0] && "Offset not in structure type!");
  --SI;
  assert(*SI <= Offset && "upper_bound didn't work");
  assert((SI == &MemberOffsets[0] || *(SI-1) <= Offset) &&
         (SI+1 == &MemberOffsets[NumElements] || *(SI+1) > Offset) &&
         "Upper bound didn't work!");

  // Multiple fields can have the same offset if any of them are zero sized.
  // For example, in { i32, [0 x i32], i32 }, searching for offset 4 will stop
  // at the i32 element, because it is the last element at that offset.  This is
  // the right one to return, because anything after it will have a higher
  // offset, implying that this element is non-empty.
  return SI-&MemberOffsets[0];
}

//===----------------------------------------------------------------------===//
// LayoutAlignElem, LayoutAlign support
//===----------------------------------------------------------------------===//

LayoutAlignElem LayoutAlignElem::get(AlignTypeEnum align_type, Align abi_align,
                                     Align pref_align, uint32_t bit_width) {
  assert(abi_align <= pref_align && "Preferred alignment worse than ABI!");
  LayoutAlignElem retval;
  retval.AlignType = align_type;
  retval.ABIAlign = abi_align;
  retval.PrefAlign = pref_align;
  retval.TypeBitWidth = bit_width;
  return retval;
}

bool
LayoutAlignElem::operator==(const LayoutAlignElem &rhs) const {
  return (AlignType == rhs.AlignType
          && ABIAlign == rhs.ABIAlign
          && PrefAlign == rhs.PrefAlign
          && TypeBitWidth == rhs.TypeBitWidth);
}

//===----------------------------------------------------------------------===//
// PointerAlignElem, PointerAlign support
//===----------------------------------------------------------------------===//

PointerAlignElem PointerAlignElem::get(uint32_t AddressSpace, Align ABIAlign,
                                       Align PrefAlign, uint32_t TypeByteWidth,
                                       uint32_t IndexWidth) {
  assert(ABIAlign <= PrefAlign && "Preferred alignment worse than ABI!");
  PointerAlignElem retval;
  retval.AddressSpace = AddressSpace;
  retval.ABIAlign = ABIAlign;
  retval.PrefAlign = PrefAlign;
  retval.TypeByteWidth = TypeByteWidth;
  retval.IndexWidth = IndexWidth;
  return retval;
}

bool
PointerAlignElem::operator==(const PointerAlignElem &rhs) const {
  return (ABIAlign == rhs.ABIAlign
          && AddressSpace == rhs.AddressSpace
          && PrefAlign == rhs.PrefAlign
          && TypeByteWidth == rhs.TypeByteWidth
          && IndexWidth == rhs.IndexWidth);
}

//===----------------------------------------------------------------------===//
//                       DataLayout Class Implementation
//===----------------------------------------------------------------------===//

const char *DataLayout::getManglingComponent(const Triple &T) {
  if (T.isOSBinFormatMachO())
    return "-m:o";
  if (T.isOSWindows() && T.isOSBinFormatCOFF())
    return T.getArch() == Triple::x86 ? "-m:x" : "-m:w";
  return "-m:e";
}

static const LayoutAlignElem DefaultAlignments[] = {
    {INTEGER_ALIGN, 1, Align(1), Align(1)},    // i1
    {INTEGER_ALIGN, 8, Align(1), Align(1)},    // i8
    {INTEGER_ALIGN, 16, Align(2), Align(2)},   // i16
    {INTEGER_ALIGN, 32, Align(4), Align(4)},   // i32
    {INTEGER_ALIGN, 64, Align(4), Align(8)},   // i64
    {FLOAT_ALIGN, 16, Align(2), Align(2)},     // half, bfloat
    {FLOAT_ALIGN, 32, Align(4), Align(4)},     // float
    {FLOAT_ALIGN, 64, Align(8), Align(8)},     // double
    {FLOAT_ALIGN, 128, Align(16), Align(16)},  // ppcf128, quad, ...
    {VECTOR_ALIGN, 64, Align(8), Align(8)},    // v2i32, v1i64, ...
    {VECTOR_ALIGN, 128, Align(16), Align(16)}, // v16i8, v8i16, v4i32, ...
    {AGGREGATE_ALIGN, 0, Align(1), Align(8)}   // struct
};

void DataLayout::reset(StringRef Desc) {
  clear();

  LayoutMap = nullptr;
  BigEndian = false;
  AllocaAddrSpace = 0;
  StackNaturalAlign.reset();
  ProgramAddrSpace = 0;
  FunctionPtrAlign.reset();
  TheFunctionPtrAlignType = FunctionPtrAlignType::Independent;
  ManglingMode = MM_None;
  NonIntegralAddressSpaces.clear();

  // Default alignments
  for (const LayoutAlignElem &E : DefaultAlignments) {
    setAlignment((AlignTypeEnum)E.AlignType, E.ABIAlign, E.PrefAlign,
                 E.TypeBitWidth);
  }
  setPointerAlignment(0, Align(8), Align(8), 8, 8);

  parseSpecifier(Desc);
}

/// Checked version of split, to ensure mandatory subparts.
static std::pair<StringRef, StringRef> split(StringRef Str, char Separator) {
  assert(!Str.empty() && "parse error, string can't be empty here");
  std::pair<StringRef, StringRef> Split = Str.split(Separator);
  if (Split.second.empty() && Split.first != Str)
    report_fatal_error("Trailing separator in datalayout string");
  if (!Split.second.empty() && Split.first.empty())
    report_fatal_error("Expected token before separator in datalayout string");
  return Split;
}

/// Get an unsigned integer, including error checks.
static unsigned getInt(StringRef R) {
  unsigned Result;
  bool error = R.getAsInteger(10, Result); (void)error;
  if (error)
    report_fatal_error("not a number, or does not fit in an unsigned int");
  return Result;
}

/// Convert bits into bytes. Assert if not a byte width multiple.
static unsigned inBytes(unsigned Bits) {
  if (Bits % 8)
    report_fatal_error("number of bits must be a byte width multiple");
  return Bits / 8;
}

static unsigned getAddrSpace(StringRef R) {
  unsigned AddrSpace = getInt(R);
  if (!isUInt<24>(AddrSpace))
    report_fatal_error("Invalid address space, must be a 24-bit integer");
  return AddrSpace;
}

void DataLayout::parseSpecifier(StringRef Desc) {
  StringRepresentation = std::string(Desc);
  while (!Desc.empty()) {
    // Split at '-'.
    std::pair<StringRef, StringRef> Split = split(Desc, '-');
    Desc = Split.second;

    // Split at ':'.
    Split = split(Split.first, ':');

    // Aliases used below.
    StringRef &Tok  = Split.first;  // Current token.
    StringRef &Rest = Split.second; // The rest of the string.

    if (Tok == "ni") {
      do {
        Split = split(Rest, ':');
        Rest = Split.second;
        unsigned AS = getInt(Split.first);
        if (AS == 0)
          report_fatal_error("Address space 0 can never be non-integral");
        NonIntegralAddressSpaces.push_back(AS);
      } while (!Rest.empty());

      continue;
    }

    char Specifier = Tok.front();
    Tok = Tok.substr(1);

    switch (Specifier) {
    case 's':
      // Ignored for backward compatibility.
      // FIXME: remove this on LLVM 4.0.
      break;
    case 'E':
      BigEndian = true;
      break;
    case 'e':
      BigEndian = false;
      break;
    case 'p': {
      // Address space.
      unsigned AddrSpace = Tok.empty() ? 0 : getInt(Tok);
      if (!isUInt<24>(AddrSpace))
        report_fatal_error("Invalid address space, must be a 24bit integer");

      // Size.
      if (Rest.empty())
        report_fatal_error(
            "Missing size specification for pointer in datalayout string");
      Split = split(Rest, ':');
      unsigned PointerMemSize = inBytes(getInt(Tok));
      if (!PointerMemSize)
        report_fatal_error("Invalid pointer size of 0 bytes");

      // ABI alignment.
      if (Rest.empty())
        report_fatal_error(
            "Missing alignment specification for pointer in datalayout string");
      Split = split(Rest, ':');
      unsigned PointerABIAlign = inBytes(getInt(Tok));
      if (!isPowerOf2_64(PointerABIAlign))
        report_fatal_error(
            "Pointer ABI alignment must be a power of 2");

      // Size of index used in GEP for address calculation.
      // The parameter is optional. By default it is equal to size of pointer.
      unsigned IndexSize = PointerMemSize;

      // Preferred alignment.
      unsigned PointerPrefAlign = PointerABIAlign;
      if (!Rest.empty()) {
        Split = split(Rest, ':');
        PointerPrefAlign = inBytes(getInt(Tok));
        if (!isPowerOf2_64(PointerPrefAlign))
          report_fatal_error(
            "Pointer preferred alignment must be a power of 2");

        // Now read the index. It is the second optional parameter here.
        if (!Rest.empty()) {
          Split = split(Rest, ':');
          IndexSize = inBytes(getInt(Tok));
          if (!IndexSize)
            report_fatal_error("Invalid index size of 0 bytes");
        }
      }
      setPointerAlignment(AddrSpace, assumeAligned(PointerABIAlign),
                          assumeAligned(PointerPrefAlign), PointerMemSize,
                          IndexSize);
      break;
    }
    case 'i':
    case 'v':
    case 'f':
    case 'a': {
      AlignTypeEnum AlignType;
      switch (Specifier) {
      default: llvm_unreachable("Unexpected specifier!");
      case 'i': AlignType = INTEGER_ALIGN; break;
      case 'v': AlignType = VECTOR_ALIGN; break;
      case 'f': AlignType = FLOAT_ALIGN; break;
      case 'a': AlignType = AGGREGATE_ALIGN; break;
      }

      // Bit size.
      unsigned Size = Tok.empty() ? 0 : getInt(Tok);

      if (AlignType == AGGREGATE_ALIGN && Size != 0)
        report_fatal_error(
            "Sized aggregate specification in datalayout string");

      // ABI alignment.
      if (Rest.empty())
        report_fatal_error(
            "Missing alignment specification in datalayout string");
      Split = split(Rest, ':');
      const unsigned ABIAlign = inBytes(getInt(Tok));
      if (AlignType != AGGREGATE_ALIGN && !ABIAlign)
        report_fatal_error(
            "ABI alignment specification must be >0 for non-aggregate types");

      if (!isUInt<16>(ABIAlign))
        report_fatal_error("Invalid ABI alignment, must be a 16bit integer");
      if (ABIAlign != 0 && !isPowerOf2_64(ABIAlign))
        report_fatal_error("Invalid ABI alignment, must be a power of 2");

      // Preferred alignment.
      unsigned PrefAlign = ABIAlign;
      if (!Rest.empty()) {
        Split = split(Rest, ':');
        PrefAlign = inBytes(getInt(Tok));
      }

      if (!isUInt<16>(PrefAlign))
        report_fatal_error(
            "Invalid preferred alignment, must be a 16bit integer");
      if (PrefAlign != 0 && !isPowerOf2_64(PrefAlign))
        report_fatal_error("Invalid preferred alignment, must be a power of 2");

      setAlignment(AlignType, assumeAligned(ABIAlign), assumeAligned(PrefAlign),
                   Size);

      break;
    }
    case 'n':  // Native integer types.
      while (true) {
        unsigned Width = getInt(Tok);
        if (Width == 0)
          report_fatal_error(
              "Zero width native integer type in datalayout string");
        LegalIntWidths.push_back(Width);
        if (Rest.empty())
          break;
        Split = split(Rest, ':');
      }
      break;
    case 'S': { // Stack natural alignment.
      uint64_t Alignment = inBytes(getInt(Tok));
      if (Alignment != 0 && !llvm::isPowerOf2_64(Alignment))
        report_fatal_error("Alignment is neither 0 nor a power of 2");
      StackNaturalAlign = MaybeAlign(Alignment);
      break;
    }
    case 'F': {
      switch (Tok.front()) {
      case 'i':
        TheFunctionPtrAlignType = FunctionPtrAlignType::Independent;
        break;
      case 'n':
        TheFunctionPtrAlignType = FunctionPtrAlignType::MultipleOfFunctionAlign;
        break;
      default:
        report_fatal_error("Unknown function pointer alignment type in "
                           "datalayout string");
      }
      Tok = Tok.substr(1);
      uint64_t Alignment = inBytes(getInt(Tok));
      if (Alignment != 0 && !llvm::isPowerOf2_64(Alignment))
        report_fatal_error("Alignment is neither 0 nor a power of 2");
      FunctionPtrAlign = MaybeAlign(Alignment);
      break;
    }
    case 'P': { // Function address space.
      ProgramAddrSpace = getAddrSpace(Tok);
      break;
    }
    case 'A': { // Default stack/alloca address space.
      AllocaAddrSpace = getAddrSpace(Tok);
      break;
    }
    case 'm':
      if (!Tok.empty())
        report_fatal_error("Unexpected trailing characters after mangling specifier in datalayout string");
      if (Rest.empty())
        report_fatal_error("Expected mangling specifier in datalayout string");
      if (Rest.size() > 1)
        report_fatal_error("Unknown mangling specifier in datalayout string");
      switch(Rest[0]) {
      default:
        report_fatal_error("Unknown mangling in datalayout string");
      case 'e':
        ManglingMode = MM_ELF;
        break;
      case 'o':
        ManglingMode = MM_MachO;
        break;
      case 'm':
        ManglingMode = MM_Mips;
        break;
      case 'w':
        ManglingMode = MM_WinCOFF;
        break;
      case 'x':
        ManglingMode = MM_WinCOFFX86;
        break;
      }
      break;
    default:
      report_fatal_error("Unknown specifier in datalayout string");
      break;
    }
  }
}

DataLayout::DataLayout(const Module *M) {
  init(M);
}

void DataLayout::init(const Module *M) { *this = M->getDataLayout(); }

bool DataLayout::operator==(const DataLayout &Other) const {
  bool Ret = BigEndian == Other.BigEndian &&
             AllocaAddrSpace == Other.AllocaAddrSpace &&
             StackNaturalAlign == Other.StackNaturalAlign &&
             ProgramAddrSpace == Other.ProgramAddrSpace &&
             FunctionPtrAlign == Other.FunctionPtrAlign &&
             TheFunctionPtrAlignType == Other.TheFunctionPtrAlignType &&
             ManglingMode == Other.ManglingMode &&
             LegalIntWidths == Other.LegalIntWidths &&
             Alignments == Other.Alignments && Pointers == Other.Pointers;
  // Note: getStringRepresentation() might differs, it is not canonicalized
  return Ret;
}

DataLayout::AlignmentsTy::iterator
DataLayout::findAlignmentLowerBound(AlignTypeEnum AlignType,
                                    uint32_t BitWidth) {
  auto Pair = std::make_pair((unsigned)AlignType, BitWidth);
  return partition_point(Alignments, [=](const LayoutAlignElem &E) {
    return std::make_pair(E.AlignType, E.TypeBitWidth) < Pair;
  });
}

void DataLayout::setAlignment(AlignTypeEnum align_type, Align abi_align,
                              Align pref_align, uint32_t bit_width) {
  // AlignmentsTy::ABIAlign and AlignmentsTy::PrefAlign were once stored as
  // uint16_t, it is unclear if there are requirements for alignment to be less
  // than 2^16 other than storage. In the meantime we leave the restriction as
  // an assert. See D67400 for context.
  assert(Log2(abi_align) < 16 && Log2(pref_align) < 16 && "Alignment too big");
  if (!isUInt<24>(bit_width))
    report_fatal_error("Invalid bit width, must be a 24bit integer");
  if (pref_align < abi_align)
    report_fatal_error(
        "Preferred alignment cannot be less than the ABI alignment");

  AlignmentsTy::iterator I = findAlignmentLowerBound(align_type, bit_width);
  if (I != Alignments.end() &&
      I->AlignType == (unsigned)align_type && I->TypeBitWidth == bit_width) {
    // Update the abi, preferred alignments.
    I->ABIAlign = abi_align;
    I->PrefAlign = pref_align;
  } else {
    // Insert before I to keep the vector sorted.
    Alignments.insert(I, LayoutAlignElem::get(align_type, abi_align,
                                              pref_align, bit_width));
  }
}

DataLayout::PointersTy::iterator
DataLayout::findPointerLowerBound(uint32_t AddressSpace) {
  return std::lower_bound(Pointers.begin(), Pointers.end(), AddressSpace,
                          [](const PointerAlignElem &A, uint32_t AddressSpace) {
    return A.AddressSpace < AddressSpace;
  });
}

void DataLayout::setPointerAlignment(uint32_t AddrSpace, Align ABIAlign,
                                     Align PrefAlign, uint32_t TypeByteWidth,
                                     uint32_t IndexWidth) {
  if (PrefAlign < ABIAlign)
    report_fatal_error(
        "Preferred alignment cannot be less than the ABI alignment");

  PointersTy::iterator I = findPointerLowerBound(AddrSpace);
  if (I == Pointers.end() || I->AddressSpace != AddrSpace) {
    Pointers.insert(I, PointerAlignElem::get(AddrSpace, ABIAlign, PrefAlign,
                                             TypeByteWidth, IndexWidth));
  } else {
    I->ABIAlign = ABIAlign;
    I->PrefAlign = PrefAlign;
    I->TypeByteWidth = TypeByteWidth;
    I->IndexWidth = IndexWidth;
  }
}

/// getAlignmentInfo - Return the alignment (either ABI if ABIInfo = true or
/// preferred if ABIInfo = false) the layout wants for the specified datatype.
Align DataLayout::getAlignmentInfo(AlignTypeEnum AlignType, uint32_t BitWidth,
                                   bool ABIInfo, Type *Ty) const {
  AlignmentsTy::const_iterator I = findAlignmentLowerBound(AlignType, BitWidth);
  // See if we found an exact match. Of if we are looking for an integer type,
  // but don't have an exact match take the next largest integer. This is where
  // the lower_bound will point to when it fails an exact match.
  if (I != Alignments.end() && I->AlignType == (unsigned)AlignType &&
      (I->TypeBitWidth == BitWidth || AlignType == INTEGER_ALIGN))
    return ABIInfo ? I->ABIAlign : I->PrefAlign;

  if (AlignType == INTEGER_ALIGN) {
    // If we didn't have a larger value try the largest value we have.
    if (I != Alignments.begin()) {
      --I; // Go to the previous entry and see if its an integer.
      if (I->AlignType == INTEGER_ALIGN)
        return ABIInfo ? I->ABIAlign : I->PrefAlign;
    }
  } else if (AlignType == VECTOR_ALIGN) {
    // By default, use natural alignment for vector types. This is consistent
    // with what clang and llvm-gcc do.
    unsigned Alignment =
        getTypeAllocSize(cast<VectorType>(Ty)->getElementType());
    // We're only calculating a natural alignment, so it doesn't have to be
    // based on the full size for scalable vectors. Using the minimum element
    // count should be enough here.
    Alignment *= cast<VectorType>(Ty)->getElementCount().Min;
    Alignment = PowerOf2Ceil(Alignment);
    return Align(Alignment);
   }

  // If we still couldn't find a reasonable default alignment, fall back
  // to a simple heuristic that the alignment is the first power of two
  // greater-or-equal to the store size of the type.  This is a reasonable
  // approximation of reality, and if the user wanted something less
  // less conservative, they should have specified it explicitly in the data
  // layout.
   unsigned Alignment = getTypeStoreSize(Ty);
   Alignment = PowerOf2Ceil(Alignment);
   return Align(Alignment);
}

namespace {

class StructLayoutMap {
  using LayoutInfoTy = DenseMap<StructType*, StructLayout*>;
  LayoutInfoTy LayoutInfo;

public:
  ~StructLayoutMap() {
    // Remove any layouts.
    for (const auto &I : LayoutInfo) {
      StructLayout *Value = I.second;
      Value->~StructLayout();
      free(Value);
    }
  }

  StructLayout *&operator[](StructType *STy) {
    return LayoutInfo[STy];
  }
};

} // end anonymous namespace

void DataLayout::clear() {
  LegalIntWidths.clear();
  Alignments.clear();
  Pointers.clear();
  delete static_cast<StructLayoutMap *>(LayoutMap);
  LayoutMap = nullptr;
}

DataLayout::~DataLayout() {
  clear();
}

const StructLayout *DataLayout::getStructLayout(StructType *Ty) const {
  if (!LayoutMap)
    LayoutMap = new StructLayoutMap();

  StructLayoutMap *STM = static_cast<StructLayoutMap*>(LayoutMap);
  StructLayout *&SL = (*STM)[Ty];
  if (SL) return SL;

  // Otherwise, create the struct layout.  Because it is variable length, we
  // malloc it, then use placement new.
  int NumElts = Ty->getNumElements();
  StructLayout *L = (StructLayout *)
      safe_malloc(sizeof(StructLayout)+(NumElts-1) * sizeof(uint64_t));

  // Set SL before calling StructLayout's ctor.  The ctor could cause other
  // entries to be added to TheMap, invalidating our reference.
  SL = L;

  new (L) StructLayout(Ty, *this);

  return L;
}

Align DataLayout::getPointerABIAlignment(unsigned AS) const {
  PointersTy::const_iterator I = findPointerLowerBound(AS);
  if (I == Pointers.end() || I->AddressSpace != AS) {
    I = findPointerLowerBound(0);
    assert(I->AddressSpace == 0);
  }
  return I->ABIAlign;
}

Align DataLayout::getPointerPrefAlignment(unsigned AS) const {
  PointersTy::const_iterator I = findPointerLowerBound(AS);
  if (I == Pointers.end() || I->AddressSpace != AS) {
    I = findPointerLowerBound(0);
    assert(I->AddressSpace == 0);
  }
  return I->PrefAlign;
}

unsigned DataLayout::getPointerSize(unsigned AS) const {
  PointersTy::const_iterator I = findPointerLowerBound(AS);
  if (I == Pointers.end() || I->AddressSpace != AS) {
    I = findPointerLowerBound(0);
    assert(I->AddressSpace == 0);
  }
  return I->TypeByteWidth;
}

unsigned DataLayout::getMaxPointerSize() const {
  unsigned MaxPointerSize = 0;
  for (auto &P : Pointers)
    MaxPointerSize = std::max(MaxPointerSize, P.TypeByteWidth);

  return MaxPointerSize;
}

unsigned DataLayout::getPointerTypeSizeInBits(Type *Ty) const {
  assert(Ty->isPtrOrPtrVectorTy() &&
         "This should only be called with a pointer or pointer vector type");
  Ty = Ty->getScalarType();
  return getPointerSizeInBits(cast<PointerType>(Ty)->getAddressSpace());
}

unsigned DataLayout::getIndexSize(unsigned AS) const {
  PointersTy::const_iterator I = findPointerLowerBound(AS);
  if (I == Pointers.end() || I->AddressSpace != AS) {
    I = findPointerLowerBound(0);
    assert(I->AddressSpace == 0);
  }
  return I->IndexWidth;
}

unsigned DataLayout::getIndexTypeSizeInBits(Type *Ty) const {
  assert(Ty->isPtrOrPtrVectorTy() &&
         "This should only be called with a pointer or pointer vector type");
  Ty = Ty->getScalarType();
  return getIndexSizeInBits(cast<PointerType>(Ty)->getAddressSpace());
}

/*!
  \param abi_or_pref Flag that determines which alignment is returned. true
  returns the ABI alignment, false returns the preferred alignment.
  \param Ty The underlying type for which alignment is determined.

  Get the ABI (\a abi_or_pref == true) or preferred alignment (\a abi_or_pref
  == false) for the requested type \a Ty.
 */
Align DataLayout::getAlignment(Type *Ty, bool abi_or_pref) const {
  AlignTypeEnum AlignType;

  assert(Ty->isSized() && "Cannot getTypeInfo() on a type that is unsized!");
  switch (Ty->getTypeID()) {
  // Early escape for the non-numeric types.
  case Type::LabelTyID:
    return abi_or_pref ? getPointerABIAlignment(0) : getPointerPrefAlignment(0);
  case Type::PointerTyID: {
    unsigned AS = cast<PointerType>(Ty)->getAddressSpace();
    return abi_or_pref ? getPointerABIAlignment(AS)
                       : getPointerPrefAlignment(AS);
    }
  case Type::ArrayTyID:
    return getAlignment(cast<ArrayType>(Ty)->getElementType(), abi_or_pref);

  case Type::StructTyID: {
    // Packed structure types always have an ABI alignment of one.
    if (cast<StructType>(Ty)->isPacked() && abi_or_pref)
      return Align(1);

    // Get the layout annotation... which is lazily created on demand.
    const StructLayout *Layout = getStructLayout(cast<StructType>(Ty));
    const Align Align = getAlignmentInfo(AGGREGATE_ALIGN, 0, abi_or_pref, Ty);
    return std::max(Align, Layout->getAlignment());
  }
  case Type::IntegerTyID:
    AlignType = INTEGER_ALIGN;
    break;
  case Type::HalfTyID:
  case Type::BFloatTyID:
  case Type::FloatTyID:
  case Type::DoubleTyID:
  // PPC_FP128TyID and FP128TyID have different data contents, but the
  // same size and alignment, so they look the same here.
  case Type::PPC_FP128TyID:
  case Type::FP128TyID:
  case Type::X86_FP80TyID:
    AlignType = FLOAT_ALIGN;
    break;
  case Type::X86_MMXTyID:
  case Type::FixedVectorTyID:
  case Type::ScalableVectorTyID:
    AlignType = VECTOR_ALIGN;
    break;
  default:
    llvm_unreachable("Bad type for getAlignment!!!");
  }

  // If we're dealing with a scalable vector, we just need the known minimum
  // size for determining alignment. If not, we'll get the exact size.
  return getAlignmentInfo(AlignType, getTypeSizeInBits(Ty).getKnownMinSize(),
                          abi_or_pref, Ty);
}

/// TODO: Remove this function once the transition to Align is over.
unsigned DataLayout::getABITypeAlignment(Type *Ty) const {
  return getABITypeAlign(Ty).value();
}

Align DataLayout::getABITypeAlign(Type *Ty) const {
  return getAlignment(Ty, true);
}

/// getABIIntegerTypeAlignment - Return the minimum ABI-required alignment for
/// an integer type of the specified bitwidth.
Align DataLayout::getABIIntegerTypeAlignment(unsigned BitWidth) const {
  return getAlignmentInfo(INTEGER_ALIGN, BitWidth, true, nullptr);
}

/// TODO: Remove this function once the transition to Align is over.
unsigned DataLayout::getPrefTypeAlignment(Type *Ty) const {
  return getPrefTypeAlign(Ty).value();
}

Align DataLayout::getPrefTypeAlign(Type *Ty) const {
  return getAlignment(Ty, false);
}

IntegerType *DataLayout::getIntPtrType(LLVMContext &C,
                                       unsigned AddressSpace) const {
  return IntegerType::get(C, getPointerSizeInBits(AddressSpace));
}

Type *DataLayout::getIntPtrType(Type *Ty) const {
  assert(Ty->isPtrOrPtrVectorTy() &&
         "Expected a pointer or pointer vector type.");
  unsigned NumBits = getPointerTypeSizeInBits(Ty);
  IntegerType *IntTy = IntegerType::get(Ty->getContext(), NumBits);
  if (VectorType *VecTy = dyn_cast<VectorType>(Ty))
    return FixedVectorType::get(IntTy, VecTy->getNumElements());
  return IntTy;
}

Type *DataLayout::getSmallestLegalIntType(LLVMContext &C, unsigned Width) const {
  for (unsigned LegalIntWidth : LegalIntWidths)
    if (Width <= LegalIntWidth)
      return Type::getIntNTy(C, LegalIntWidth);
  return nullptr;
}

unsigned DataLayout::getLargestLegalIntTypeSizeInBits() const {
  auto Max = std::max_element(LegalIntWidths.begin(), LegalIntWidths.end());
  return Max != LegalIntWidths.end() ? *Max : 0;
}

Type *DataLayout::getIndexType(Type *Ty) const {
  assert(Ty->isPtrOrPtrVectorTy() &&
         "Expected a pointer or pointer vector type.");
  unsigned NumBits = getIndexTypeSizeInBits(Ty);
  IntegerType *IntTy = IntegerType::get(Ty->getContext(), NumBits);
  if (VectorType *VecTy = dyn_cast<VectorType>(Ty))
    return FixedVectorType::get(IntTy, VecTy->getNumElements());
  return IntTy;
}

int64_t DataLayout::getIndexedOffsetInType(Type *ElemTy,
                                           ArrayRef<Value *> Indices) const {
  int64_t Result = 0;

  generic_gep_type_iterator<Value* const*>
    GTI = gep_type_begin(ElemTy, Indices),
    GTE = gep_type_end(ElemTy, Indices);
  for (; GTI != GTE; ++GTI) {
    Value *Idx = GTI.getOperand();
    if (StructType *STy = GTI.getStructTypeOrNull()) {
      assert(Idx->getType()->isIntegerTy(32) && "Illegal struct idx");
      unsigned FieldNo = cast<ConstantInt>(Idx)->getZExtValue();

      // Get structure layout information...
      const StructLayout *Layout = getStructLayout(STy);

      // Add in the offset, as calculated by the structure layout info...
      Result += Layout->getElementOffset(FieldNo);
    } else {
      // Get the array index and the size of each array element.
      if (int64_t arrayIdx = cast<ConstantInt>(Idx)->getSExtValue())
        Result += arrayIdx * getTypeAllocSize(GTI.getIndexedType());
    }
  }

  return Result;
}

/// getPreferredAlignment - Return the preferred alignment of the specified
/// global.  This includes an explicitly requested alignment (if the global
/// has one).
unsigned DataLayout::getPreferredAlignment(const GlobalVariable *GV) const {
  unsigned GVAlignment = GV->getAlignment();
  // If a section is specified, always precisely honor explicit alignment,
  // so we don't insert padding into a section we don't control.
  if (GVAlignment && GV->hasSection())
    return GVAlignment;

  // If no explicit alignment is specified, compute the alignment based on
  // the IR type. If an alignment is specified, increase it to match the ABI
  // alignment of the IR type.
  //
  // FIXME: Not sure it makes sense to use the alignment of the type if
  // there's already an explicit alignment specification.
  Type *ElemType = GV->getValueType();
  unsigned Alignment = getPrefTypeAlignment(ElemType);
  if (GVAlignment >= Alignment) {
    Alignment = GVAlignment;
  } else if (GVAlignment != 0) {
    Alignment = std::max(GVAlignment, getABITypeAlignment(ElemType));
  }

  // If no explicit alignment is specified, and the global is large, increase
  // the alignment to 16.
  // FIXME: Why 16, specifically?
  if (GV->hasInitializer() && GVAlignment == 0) {
    if (Alignment < 16) {
      // If the global is not external, see if it is large.  If so, give it a
      // larger alignment.
      if (getTypeSizeInBits(ElemType) > 128)
        Alignment = 16;    // 16-byte alignment.
    }
  }
  return Alignment;
}

/// getPreferredAlignmentLog - Return the preferred alignment of the
/// specified global, returned in log form.  This includes an explicitly
/// requested alignment (if the global has one).
unsigned DataLayout::getPreferredAlignmentLog(const GlobalVariable *GV) const {
  return Log2_32(getPreferredAlignment(GV));
}
