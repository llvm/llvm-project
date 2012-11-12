//===-- DataLayout.cpp - Data size & alignment routines --------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

#include "llvm/DataLayout.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Mutex.h"
#include "llvm/ADT/DenseMap.h"
#include <algorithm>
#include <cstdlib>
using namespace llvm;

// Handle the Pass registration stuff necessary to use DataLayout's.

// Register the default SparcV9 implementation...
INITIALIZE_PASS(DataLayout, "datalayout", "Data Layout", false, true)
char DataLayout::ID = 0;

//===----------------------------------------------------------------------===//
// Support for StructLayout
//===----------------------------------------------------------------------===//

StructLayout::StructLayout(StructType *ST, const DataLayout &TD) {
  assert(!ST->isOpaque() && "Cannot get layout of opaque structs");
  StructAlignment = 0;
  StructSize = 0;
  NumElements = ST->getNumElements();

  // Loop over each of the elements, placing them in memory.
  for (unsigned i = 0, e = NumElements; i != e; ++i) {
    Type *Ty = ST->getElementType(i);
    unsigned TyAlign = ST->isPacked() ? 1 : TD.getABITypeAlignment(Ty);

    // Add padding if necessary to align the data element properly.
    if ((StructSize & (TyAlign-1)) != 0)
      StructSize = DataLayout::RoundUpAlignment(StructSize, TyAlign);

    // Keep track of maximum alignment constraint.
    StructAlignment = std::max(TyAlign, StructAlignment);

    MemberOffsets[i] = StructSize;
    StructSize += TD.getTypeAllocSize(Ty); // Consume space for this data item
  }

  // Empty structures have alignment of 1 byte.
  if (StructAlignment == 0) StructAlignment = 1;

  // Add padding to the end of the struct so that it could be put in an array
  // and all array elements would be aligned correctly.
  if ((StructSize & (StructAlignment-1)) != 0)
    StructSize = DataLayout::RoundUpAlignment(StructSize, StructAlignment);
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

LayoutAlignElem
LayoutAlignElem::get(AlignTypeEnum align_type, unsigned abi_align,
                     unsigned pref_align, uint32_t bit_width) {
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

const LayoutAlignElem
DataLayout::InvalidAlignmentElem =
            LayoutAlignElem::get((AlignTypeEnum) -1, 0, 0, 0);

//===----------------------------------------------------------------------===//
// PointerAlignElem, PointerAlign support
//===----------------------------------------------------------------------===//

PointerAlignElem
PointerAlignElem::get(uint32_t addr_space, unsigned abi_align,
                     unsigned pref_align, uint32_t bit_width) {
  assert(abi_align <= pref_align && "Preferred alignment worse than ABI!");
  PointerAlignElem retval;
  retval.AddressSpace = addr_space;
  retval.ABIAlign = abi_align;
  retval.PrefAlign = pref_align;
  retval.TypeBitWidth = bit_width;
  return retval;
}

bool
PointerAlignElem::operator==(const PointerAlignElem &rhs) const {
  return (ABIAlign == rhs.ABIAlign
          && AddressSpace == rhs.AddressSpace
          && PrefAlign == rhs.PrefAlign
          && TypeBitWidth == rhs.TypeBitWidth);
}

const PointerAlignElem
DataLayout::InvalidPointerElem = PointerAlignElem::get(~0U, 0U, 0U, 0U);

//===----------------------------------------------------------------------===//
//                       DataLayout Class Implementation
//===----------------------------------------------------------------------===//

/// getInt - Get an integer ignoring errors.
static int getInt(StringRef R) {
  int Result = 0;
  R.getAsInteger(10, Result);
  return Result;
}

void DataLayout::init() {
  initializeDataLayoutPass(*PassRegistry::getPassRegistry());

  LayoutMap = 0;
  LittleEndian = false;
  StackNaturalAlign = 0;

  // Default alignments
  setAlignment(INTEGER_ALIGN,   1,  1, 1);   // i1
  setAlignment(INTEGER_ALIGN,   1,  1, 8);   // i8
  setAlignment(INTEGER_ALIGN,   2,  2, 16);  // i16
  setAlignment(INTEGER_ALIGN,   4,  4, 32);  // i32
  setAlignment(INTEGER_ALIGN,   4,  8, 64);  // i64
  setAlignment(FLOAT_ALIGN,     2,  2, 16);  // half
  setAlignment(FLOAT_ALIGN,     4,  4, 32);  // float
  setAlignment(FLOAT_ALIGN,     8,  8, 64);  // double
  setAlignment(FLOAT_ALIGN,    16, 16, 128); // ppcf128, quad, ...
  setAlignment(VECTOR_ALIGN,    8,  8, 64);  // v2i32, v1i64, ...
  setAlignment(VECTOR_ALIGN,   16, 16, 128); // v16i8, v8i16, v4i32, ...
  setAlignment(AGGREGATE_ALIGN, 0,  8,  0);  // struct
  setPointerAlignment(0, 8, 8, 8);
}

std::string DataLayout::parseSpecifier(StringRef Desc, DataLayout *td) {

  if (td)
    td->init();

  while (!Desc.empty()) {
    std::pair<StringRef, StringRef> Split = Desc.split('-');
    StringRef Token = Split.first;
    Desc = Split.second;

    if (Token.empty())
      continue;

    Split = Token.split(':');
    StringRef Specifier = Split.first;
    Token = Split.second;

    assert(!Specifier.empty() && "Can't be empty here");

    switch (Specifier[0]) {
    case 'E':
      if (td)
        td->LittleEndian = false;
      break;
    case 'e':
      if (td)
        td->LittleEndian = true;
      break;
    case 'p': {
      int AddrSpace = 0;
      if (Specifier.size() > 1) {
        AddrSpace = getInt(Specifier.substr(1));
        if (AddrSpace < 0 || AddrSpace > (1 << 24))
          return "Invalid address space, must be a positive 24bit integer";
      }
      Split = Token.split(':');
      int PointerMemSizeBits = getInt(Split.first);
      if (PointerMemSizeBits < 0 || PointerMemSizeBits % 8 != 0)
        return "invalid pointer size, must be a positive 8-bit multiple";

      // Pointer ABI alignment.
      Split = Split.second.split(':');
      int PointerABIAlignBits = getInt(Split.first);
      if (PointerABIAlignBits < 0 || PointerABIAlignBits % 8 != 0) {
        return "invalid pointer ABI alignment, "
               "must be a positive 8-bit multiple";
      }

      // Pointer preferred alignment.
      Split = Split.second.split(':');
      int PointerPrefAlignBits = getInt(Split.first);
      if (PointerPrefAlignBits < 0 || PointerPrefAlignBits % 8 != 0) {
        return "invalid pointer preferred alignment, "
               "must be a positive 8-bit multiple";
      }

      if (PointerPrefAlignBits == 0)
        PointerPrefAlignBits = PointerABIAlignBits;
      if (td)
        td->setPointerAlignment(AddrSpace, PointerABIAlignBits/8,
            PointerPrefAlignBits/8, PointerMemSizeBits/8);
      break;
    }
    case 'i':
    case 'v':
    case 'f':
    case 'a':
    case 's': {
      AlignTypeEnum AlignType;
      char field = Specifier[0];
      switch (field) {
      default:
      case 'i': AlignType = INTEGER_ALIGN; break;
      case 'v': AlignType = VECTOR_ALIGN; break;
      case 'f': AlignType = FLOAT_ALIGN; break;
      case 'a': AlignType = AGGREGATE_ALIGN; break;
      case 's': AlignType = STACK_ALIGN; break;
      }
      int Size = getInt(Specifier.substr(1));
      if (Size < 0) {
        return std::string("invalid ") + field + "-size field, "
               "must be positive";
      }

      Split = Token.split(':');
      int ABIAlignBits = getInt(Split.first);
      if (ABIAlignBits < 0 || ABIAlignBits % 8 != 0) {
        return std::string("invalid ") + field +"-abi-alignment field, "
               "must be a positive 8-bit multiple";
      }
      unsigned ABIAlign = ABIAlignBits / 8;

      Split = Split.second.split(':');

      int PrefAlignBits = getInt(Split.first);
      if (PrefAlignBits < 0 || PrefAlignBits % 8 != 0) {
        return std::string("invalid ") + field +"-preferred-alignment field, "
               "must be a positive 8-bit multiple";
      }
      unsigned PrefAlign = PrefAlignBits / 8;
      if (PrefAlign == 0)
        PrefAlign = ABIAlign;

      if (td)
        td->setAlignment(AlignType, ABIAlign, PrefAlign, Size);
      break;
    }
    case 'n':  // Native integer types.
      Specifier = Specifier.substr(1);
      do {
        int Width = getInt(Specifier);
        if (Width <= 0) {
          return std::string("invalid native integer size \'") +
            Specifier.str() + "\', must be a positive integer.";
        }
        if (td && Width != 0)
          td->LegalIntWidths.push_back(Width);
        Split = Token.split(':');
        Specifier = Split.first;
        Token = Split.second;
      } while (!Specifier.empty() || !Token.empty());
      break;
    case 'S': { // Stack natural alignment.
      int StackNaturalAlignBits = getInt(Specifier.substr(1));
      if (StackNaturalAlignBits < 0 || StackNaturalAlignBits % 8 != 0) {
        return "invalid natural stack alignment (S-field), "
               "must be a positive 8-bit multiple";
      }
      if (td)
        td->StackNaturalAlign = StackNaturalAlignBits / 8;
      break;
    }
    default:
      break;
    }
  }

  return "";
}

/// Default ctor.
///
/// @note This has to exist, because this is a pass, but it should never be
/// used.
DataLayout::DataLayout() : ImmutablePass(ID) {
  report_fatal_error("Bad DataLayout ctor used.  "
                    "Tool did not specify a DataLayout to use?");
}

DataLayout::DataLayout(const Module *M)
  : ImmutablePass(ID) {
  std::string errMsg = parseSpecifier(M->getDataLayout(), this);
  assert(errMsg == "" && "Module M has malformed data layout string.");
  (void)errMsg;
}

void
DataLayout::setAlignment(AlignTypeEnum align_type, unsigned abi_align,
                         unsigned pref_align, uint32_t bit_width) {
  assert(abi_align <= pref_align && "Preferred alignment worse than ABI!");
  assert(pref_align < (1 << 16) && "Alignment doesn't fit in bitfield");
  assert(bit_width < (1 << 24) && "Bit width doesn't fit in bitfield");
  for (unsigned i = 0, e = Alignments.size(); i != e; ++i) {
    if (Alignments[i].AlignType == (unsigned)align_type &&
        Alignments[i].TypeBitWidth == bit_width) {
      // Update the abi, preferred alignments.
      Alignments[i].ABIAlign = abi_align;
      Alignments[i].PrefAlign = pref_align;
      return;
    }
  }

  Alignments.push_back(LayoutAlignElem::get(align_type, abi_align,
                                            pref_align, bit_width));
}

void
DataLayout::setPointerAlignment(uint32_t addr_space, unsigned abi_align,
                         unsigned pref_align, uint32_t bit_width) {
  assert(abi_align <= pref_align && "Preferred alignment worse than ABI!");
  DenseMap<unsigned,PointerAlignElem>::iterator val = Pointers.find(addr_space);
  if (val == Pointers.end()) {
    Pointers[addr_space] = PointerAlignElem::get(addr_space,
          abi_align, pref_align, bit_width);
  } else {
    val->second.ABIAlign = abi_align;
    val->second.PrefAlign = pref_align;
    val->second.TypeBitWidth = bit_width;
  }
}

/// getAlignmentInfo - Return the alignment (either ABI if ABIInfo = true or
/// preferred if ABIInfo = false) the layout wants for the specified datatype.
unsigned DataLayout::getAlignmentInfo(AlignTypeEnum AlignType,
                                      uint32_t BitWidth, bool ABIInfo,
                                      Type *Ty) const {
  // Check to see if we have an exact match and remember the best match we see.
  int BestMatchIdx = -1;
  int LargestInt = -1;
  for (unsigned i = 0, e = Alignments.size(); i != e; ++i) {
    if (Alignments[i].AlignType == (unsigned)AlignType &&
        Alignments[i].TypeBitWidth == BitWidth)
      return ABIInfo ? Alignments[i].ABIAlign : Alignments[i].PrefAlign;

    // The best match so far depends on what we're looking for.
     if (AlignType == INTEGER_ALIGN &&
         Alignments[i].AlignType == INTEGER_ALIGN) {
      // The "best match" for integers is the smallest size that is larger than
      // the BitWidth requested.
      if (Alignments[i].TypeBitWidth > BitWidth && (BestMatchIdx == -1 ||
           Alignments[i].TypeBitWidth < Alignments[BestMatchIdx].TypeBitWidth))
        BestMatchIdx = i;
      // However, if there isn't one that's larger, then we must use the
      // largest one we have (see below)
      if (LargestInt == -1 ||
          Alignments[i].TypeBitWidth > Alignments[LargestInt].TypeBitWidth)
        LargestInt = i;
    }
  }

  // Okay, we didn't find an exact solution.  Fall back here depending on what
  // is being looked for.
  if (BestMatchIdx == -1) {
    // If we didn't find an integer alignment, fall back on most conservative.
    if (AlignType == INTEGER_ALIGN) {
      BestMatchIdx = LargestInt;
    } else {
      assert(AlignType == VECTOR_ALIGN && "Unknown alignment type!");

      // By default, use natural alignment for vector types. This is consistent
      // with what clang and llvm-gcc do.
      unsigned Align = getTypeAllocSize(cast<VectorType>(Ty)->getElementType());
      Align *= cast<VectorType>(Ty)->getNumElements();
      // If the alignment is not a power of 2, round up to the next power of 2.
      // This happens for non-power-of-2 length vectors.
      if (Align & (Align-1))
        Align = NextPowerOf2(Align);
      return Align;
    }
  }

  // Since we got a "best match" index, just return it.
  return ABIInfo ? Alignments[BestMatchIdx].ABIAlign
                 : Alignments[BestMatchIdx].PrefAlign;
}

namespace {

class StructLayoutMap {
  typedef DenseMap<StructType*, StructLayout*> LayoutInfoTy;
  LayoutInfoTy LayoutInfo;

public:
  virtual ~StructLayoutMap() {
    // Remove any layouts.
    for (LayoutInfoTy::iterator I = LayoutInfo.begin(), E = LayoutInfo.end();
         I != E; ++I) {
      StructLayout *Value = I->second;
      Value->~StructLayout();
      free(Value);
    }
  }

  StructLayout *&operator[](StructType *STy) {
    return LayoutInfo[STy];
  }

  // for debugging...
  virtual void dump() const {}
};

} // end anonymous namespace

DataLayout::~DataLayout() {
  delete static_cast<StructLayoutMap*>(LayoutMap);
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
  StructLayout *L =
    (StructLayout *)malloc(sizeof(StructLayout)+(NumElts-1) * sizeof(uint64_t));

  // Set SL before calling StructLayout's ctor.  The ctor could cause other
  // entries to be added to TheMap, invalidating our reference.
  SL = L;

  new (L) StructLayout(Ty, *this);

  return L;
}

std::string DataLayout::getStringRepresentation() const {
  std::string Result;
  raw_string_ostream OS(Result);

  OS << (LittleEndian ? "e" : "E");
  SmallVector<unsigned, 8> addrSpaces;
  // Lets get all of the known address spaces and sort them
  // into increasing order so that we can emit the string
  // in a cleaner format.
  for (DenseMap<unsigned, PointerAlignElem>::const_iterator
      pib = Pointers.begin(), pie = Pointers.end();
      pib != pie; ++pib) {
    addrSpaces.push_back(pib->first);
  }
  std::sort(addrSpaces.begin(), addrSpaces.end());
  for (SmallVector<unsigned, 8>::iterator asb = addrSpaces.begin(),
      ase = addrSpaces.end(); asb != ase; ++asb) {
    const PointerAlignElem &PI = Pointers.find(*asb)->second;
    OS << "-p";
    if (PI.AddressSpace) {
      OS << PI.AddressSpace;
    }
     OS << ":" << PI.TypeBitWidth*8 << ':' << PI.ABIAlign*8
        << ':' << PI.PrefAlign*8;
  }
  OS << "-S" << StackNaturalAlign*8;

  for (unsigned i = 0, e = Alignments.size(); i != e; ++i) {
    const LayoutAlignElem &AI = Alignments[i];
    OS << '-' << (char)AI.AlignType << AI.TypeBitWidth << ':'
       << AI.ABIAlign*8 << ':' << AI.PrefAlign*8;
  }

  if (!LegalIntWidths.empty()) {
    OS << "-n" << (unsigned)LegalIntWidths[0];

    for (unsigned i = 1, e = LegalIntWidths.size(); i != e; ++i)
      OS << ':' << (unsigned)LegalIntWidths[i];
  }
  return OS.str();
}


uint64_t DataLayout::getTypeSizeInBits(Type *Ty) const {
  assert(Ty->isSized() && "Cannot getTypeInfo() on a type that is unsized!");
  switch (Ty->getTypeID()) {
  case Type::LabelTyID:
    return getPointerSizeInBits(0);
  case Type::PointerTyID: {
    unsigned AS = dyn_cast<PointerType>(Ty)->getAddressSpace();
    return getPointerSizeInBits(AS);
    }
  case Type::ArrayTyID: {
    ArrayType *ATy = cast<ArrayType>(Ty);
    return getTypeAllocSizeInBits(ATy->getElementType())*ATy->getNumElements();
  }
  case Type::StructTyID:
    // Get the layout annotation... which is lazily created on demand.
    return getStructLayout(cast<StructType>(Ty))->getSizeInBits();
  case Type::IntegerTyID:
    return cast<IntegerType>(Ty)->getBitWidth();
  case Type::VoidTyID:
    return 8;
  case Type::HalfTyID:
    return 16;
  case Type::FloatTyID:
    return 32;
  case Type::DoubleTyID:
  case Type::X86_MMXTyID:
    return 64;
  case Type::PPC_FP128TyID:
  case Type::FP128TyID:
    return 128;
  // In memory objects this is always aligned to a higher boundary, but
  // only 80 bits contain information.
  case Type::X86_FP80TyID:
    return 80;
  case Type::VectorTyID: {
    VectorType *VTy = cast<VectorType>(Ty);
    return VTy->getNumElements()*getTypeSizeInBits(VTy->getElementType());
  }
  default:
    llvm_unreachable("DataLayout::getTypeSizeInBits(): Unsupported type");
  }
}

/*!
  \param abi_or_pref Flag that determines which alignment is returned. true
  returns the ABI alignment, false returns the preferred alignment.
  \param Ty The underlying type for which alignment is determined.

  Get the ABI (\a abi_or_pref == true) or preferred alignment (\a abi_or_pref
  == false) for the requested type \a Ty.
 */
unsigned DataLayout::getAlignment(Type *Ty, bool abi_or_pref) const {
  int AlignType = -1;

  assert(Ty->isSized() && "Cannot getTypeInfo() on a type that is unsized!");
  switch (Ty->getTypeID()) {
  // Early escape for the non-numeric types.
  case Type::LabelTyID:
    return (abi_or_pref
            ? getPointerABIAlignment(0)
            : getPointerPrefAlignment(0));
  case Type::PointerTyID: {
    unsigned AS = dyn_cast<PointerType>(Ty)->getAddressSpace();
    return (abi_or_pref
            ? getPointerABIAlignment(AS)
            : getPointerPrefAlignment(AS));
    }
  case Type::ArrayTyID:
    return getAlignment(cast<ArrayType>(Ty)->getElementType(), abi_or_pref);

  case Type::StructTyID: {
    // Packed structure types always have an ABI alignment of one.
    if (cast<StructType>(Ty)->isPacked() && abi_or_pref)
      return 1;

    // Get the layout annotation... which is lazily created on demand.
    const StructLayout *Layout = getStructLayout(cast<StructType>(Ty));
    unsigned Align = getAlignmentInfo(AGGREGATE_ALIGN, 0, abi_or_pref, Ty);
    return std::max(Align, Layout->getAlignment());
  }
  case Type::IntegerTyID:
  case Type::VoidTyID:
    AlignType = INTEGER_ALIGN;
    break;
  case Type::HalfTyID:
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
  case Type::VectorTyID:
    AlignType = VECTOR_ALIGN;
    break;
  default:
    llvm_unreachable("Bad type for getAlignment!!!");
  }

  return getAlignmentInfo((AlignTypeEnum)AlignType, getTypeSizeInBits(Ty),
                          abi_or_pref, Ty);
}

unsigned DataLayout::getABITypeAlignment(Type *Ty) const {
  return getAlignment(Ty, true);
}

/// getABIIntegerTypeAlignment - Return the minimum ABI-required alignment for
/// an integer type of the specified bitwidth.
unsigned DataLayout::getABIIntegerTypeAlignment(unsigned BitWidth) const {
  return getAlignmentInfo(INTEGER_ALIGN, BitWidth, true, 0);
}


unsigned DataLayout::getCallFrameTypeAlignment(Type *Ty) const {
  for (unsigned i = 0, e = Alignments.size(); i != e; ++i)
    if (Alignments[i].AlignType == STACK_ALIGN)
      return Alignments[i].ABIAlign;

  return getABITypeAlignment(Ty);
}

unsigned DataLayout::getPrefTypeAlignment(Type *Ty) const {
  return getAlignment(Ty, false);
}

unsigned DataLayout::getPreferredTypeAlignmentShift(Type *Ty) const {
  unsigned Align = getPrefTypeAlignment(Ty);
  assert(!(Align & (Align-1)) && "Alignment is not a power of two!");
  return Log2_32(Align);
}

/// getIntPtrType - Return an integer type with size at least as big as that
/// of a pointer in the given address space.
IntegerType *DataLayout::getIntPtrType(LLVMContext &C,
                                       unsigned AddressSpace) const {
  return IntegerType::get(C, getPointerSizeInBits(AddressSpace));
}

/// getIntPtrType - Return an integer (vector of integer) type with size at
/// least as big as that of a pointer of the given pointer (vector of pointer)
/// type.
Type *DataLayout::getIntPtrType(Type *Ty) const {
  assert(Ty->isPtrOrPtrVectorTy() &&
         "Expected a pointer or pointer vector type.");
  unsigned NumBits = getTypeSizeInBits(Ty->getScalarType());
  IntegerType *IntTy = IntegerType::get(Ty->getContext(), NumBits);
  if (VectorType *VecTy = dyn_cast<VectorType>(Ty))
    return VectorType::get(IntTy, VecTy->getNumElements());
  return IntTy;
}

uint64_t DataLayout::getIndexedOffset(Type *ptrTy,
                                      ArrayRef<Value *> Indices) const {
  Type *Ty = ptrTy;
  assert(Ty->isPointerTy() && "Illegal argument for getIndexedOffset()");
  uint64_t Result = 0;

  generic_gep_type_iterator<Value* const*>
    TI = gep_type_begin(ptrTy, Indices);
  for (unsigned CurIDX = 0, EndIDX = Indices.size(); CurIDX != EndIDX;
       ++CurIDX, ++TI) {
    if (StructType *STy = dyn_cast<StructType>(*TI)) {
      assert(Indices[CurIDX]->getType() ==
             Type::getInt32Ty(ptrTy->getContext()) &&
             "Illegal struct idx");
      unsigned FieldNo = cast<ConstantInt>(Indices[CurIDX])->getZExtValue();

      // Get structure layout information...
      const StructLayout *Layout = getStructLayout(STy);

      // Add in the offset, as calculated by the structure layout info...
      Result += Layout->getElementOffset(FieldNo);

      // Update Ty to refer to current element
      Ty = STy->getElementType(FieldNo);
    } else {
      // Update Ty to refer to current element
      Ty = cast<SequentialType>(Ty)->getElementType();

      // Get the array index and the size of each array element.
      if (int64_t arrayIdx = cast<ConstantInt>(Indices[CurIDX])->getSExtValue())
        Result += (uint64_t)arrayIdx * getTypeAllocSize(Ty);
    }
  }

  return Result;
}

/// getPreferredAlignment - Return the preferred alignment of the specified
/// global.  This includes an explicitly requested alignment (if the global
/// has one).
unsigned DataLayout::getPreferredAlignment(const GlobalVariable *GV) const {
  Type *ElemType = GV->getType()->getElementType();
  unsigned Alignment = getPrefTypeAlignment(ElemType);
  unsigned GVAlignment = GV->getAlignment();
  if (GVAlignment >= Alignment) {
    Alignment = GVAlignment;
  } else if (GVAlignment != 0) {
    Alignment = std::max(GVAlignment, getABITypeAlignment(ElemType));
  }

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
