//===-- TargetData.cpp - Data size & alignment routines --------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target properties related to datatype size/offset/alignment
// information.
//
// This structure should be created once, filled in if the defaults are not
// correct and then passed around by const&.  None of the members functions
// require modification to the object.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetData.h"
#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include <algorithm>
#include <cstdlib>
using namespace llvm;

// Handle the Pass registration stuff necessary to use TargetData's.

// Register the default SparcV9 implementation...
static RegisterPass<TargetData> X("targetdata", "Target Data Layout", false, 
                                  true);
char TargetData::ID = 0;

//===----------------------------------------------------------------------===//
// Support for StructLayout
//===----------------------------------------------------------------------===//

StructLayout::StructLayout(const StructType *ST, const TargetData &TD) {
  StructAlignment = 0;
  StructSize = 0;
  NumElements = ST->getNumElements();

  // Loop over each of the elements, placing them in memory.
  for (unsigned i = 0, e = NumElements; i != e; ++i) {
    const Type *Ty = ST->getElementType(i);
    unsigned TyAlign = ST->isPacked() ? 1 : TD.getABITypeAlignment(Ty);

    // Add padding if necessary to align the data element properly.
    if ((StructSize & (TyAlign-1)) != 0)
      StructSize = TargetData::RoundUpAlignment(StructSize, TyAlign);

    // Keep track of maximum alignment constraint.
    StructAlignment = std::max(TyAlign, StructAlignment);

    MemberOffsets[i] = StructSize;
    StructSize += TD.getTypePaddedSize(Ty); // Consume space for this data item
  }

  // Empty structures have alignment of 1 byte.
  if (StructAlignment == 0) StructAlignment = 1;

  // Add padding to the end of the struct so that it could be put in an array
  // and all array elements would be aligned correctly.
  if ((StructSize & (StructAlignment-1)) != 0)
    StructSize = TargetData::RoundUpAlignment(StructSize, StructAlignment);
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
// TargetAlignElem, TargetAlign support
//===----------------------------------------------------------------------===//

TargetAlignElem
TargetAlignElem::get(AlignTypeEnum align_type, unsigned char abi_align,
                     unsigned char pref_align, uint32_t bit_width) {
  assert(abi_align <= pref_align && "Preferred alignment worse than ABI!");
  TargetAlignElem retval;
  retval.AlignType = align_type;
  retval.ABIAlign = abi_align;
  retval.PrefAlign = pref_align;
  retval.TypeBitWidth = bit_width;
  return retval;
}

bool
TargetAlignElem::operator==(const TargetAlignElem &rhs) const {
  return (AlignType == rhs.AlignType
          && ABIAlign == rhs.ABIAlign
          && PrefAlign == rhs.PrefAlign
          && TypeBitWidth == rhs.TypeBitWidth);
}

std::ostream &
TargetAlignElem::dump(std::ostream &os) const {
  return os << AlignType
            << TypeBitWidth
            << ":" << (int) (ABIAlign * 8)
            << ":" << (int) (PrefAlign * 8);
}

const TargetAlignElem TargetData::InvalidAlignmentElem =
                TargetAlignElem::get((AlignTypeEnum) -1, 0, 0, 0);

//===----------------------------------------------------------------------===//
//                       TargetData Class Implementation
//===----------------------------------------------------------------------===//

/*!
 A TargetDescription string consists of a sequence of hyphen-delimited
 specifiers for target endianness, pointer size and alignments, and various
 primitive type sizes and alignments. A typical string looks something like:
 <br><br>
 "E-p:32:32:32-i1:8:8-i8:8:8-i32:32:32-i64:32:64-f32:32:32-f64:32:64"
 <br><br>
 (note: this string is not fully specified and is only an example.)
 \p
 Alignments come in two flavors: ABI and preferred. ABI alignment (abi_align,
 below) dictates how a type will be aligned within an aggregate and when used
 as an argument.  Preferred alignment (pref_align, below) determines a type's
 alignment when emitted as a global.
 \p
 Specifier string details:
 <br><br>
 <i>[E|e]</i>: Endianness. "E" specifies a big-endian target data model, "e"
 specifies a little-endian target data model.
 <br><br>
 <i>p:@verbatim<size>:<abi_align>:<pref_align>@endverbatim</i>: Pointer size, 
 ABI and preferred alignment.
 <br><br>
 <i>@verbatim<type><size>:<abi_align>:<pref_align>@endverbatim</i>: Numeric type
 alignment. Type is
 one of <i>i|f|v|a</i>, corresponding to integer, floating point, vector (aka
 packed) or aggregate.  Size indicates the size, e.g., 32 or 64 bits.
 \p
 The default string, fully specified is:
 <br><br>
 "E-p:64:64:64-a0:0:0-f32:32:32-f64:0:64"
 "-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:0:64"
 "-v64:64:64-v128:128:128"
 <br><br>
 Note that in the case of aggregates, 0 is the default ABI and preferred
 alignment. This is a special case, where the aggregate's computed worst-case
 alignment will be used.
 */ 
void TargetData::init(const std::string &TargetDescription) {
  std::string temp = TargetDescription;
  
  LittleEndian = false;
  PointerMemSize = 8;
  PointerABIAlign   = 8;
  PointerPrefAlign = PointerABIAlign;

  // Default alignments
  setAlignment(INTEGER_ALIGN,   1,  1, 1);   // Bool
  setAlignment(INTEGER_ALIGN,   1,  1, 8);   // Byte
  setAlignment(INTEGER_ALIGN,   2,  2, 16);  // short
  setAlignment(INTEGER_ALIGN,   4,  4, 32);  // int
  setAlignment(INTEGER_ALIGN,   4,  8, 64);  // long
  setAlignment(FLOAT_ALIGN,     4,  4, 32);  // float
  setAlignment(FLOAT_ALIGN,     8,  8, 64);  // double
  setAlignment(VECTOR_ALIGN,    8,  8, 64);  // v2i32
  setAlignment(VECTOR_ALIGN,   16, 16, 128); // v16i8, v8i16, v4i32, ...
  setAlignment(AGGREGATE_ALIGN, 0,  8,  0);  // struct, union, class, ...

  while (!temp.empty()) {
    std::string token = getToken(temp, "-");
    std::string arg0 = getToken(token, ":");
    const char *p = arg0.c_str();
    switch(*p) {
    case 'E':
      LittleEndian = false;
      break;
    case 'e':
      LittleEndian = true;
      break;
    case 'p':
      PointerMemSize = atoi(getToken(token,":").c_str()) / 8;
      PointerABIAlign = atoi(getToken(token,":").c_str()) / 8;
      PointerPrefAlign = atoi(getToken(token,":").c_str()) / 8;
      if (PointerPrefAlign == 0)
        PointerPrefAlign = PointerABIAlign;
      break;
    case 'i':
    case 'v':
    case 'f':
    case 'a':
    case 's': {
      AlignTypeEnum align_type = STACK_ALIGN; // Dummy init, silence warning
      switch(*p) {
        case 'i': align_type = INTEGER_ALIGN; break;
        case 'v': align_type = VECTOR_ALIGN; break;
        case 'f': align_type = FLOAT_ALIGN; break;
        case 'a': align_type = AGGREGATE_ALIGN; break;
        case 's': align_type = STACK_ALIGN; break;
      }
      uint32_t size = (uint32_t) atoi(++p);
      unsigned char abi_align = atoi(getToken(token, ":").c_str()) / 8;
      unsigned char pref_align = atoi(getToken(token, ":").c_str()) / 8;
      if (pref_align == 0)
        pref_align = abi_align;
      setAlignment(align_type, abi_align, pref_align, size);
      break;
    }
    default:
      break;
    }
  }
}

TargetData::TargetData(const Module *M) 
  : ImmutablePass(&ID) {
  init(M->getDataLayout());
}

void
TargetData::setAlignment(AlignTypeEnum align_type, unsigned char abi_align,
                         unsigned char pref_align, uint32_t bit_width) {
  assert(abi_align <= pref_align && "Preferred alignment worse than ABI!");
  for (unsigned i = 0, e = Alignments.size(); i != e; ++i) {
    if (Alignments[i].AlignType == align_type &&
        Alignments[i].TypeBitWidth == bit_width) {
      // Update the abi, preferred alignments.
      Alignments[i].ABIAlign = abi_align;
      Alignments[i].PrefAlign = pref_align;
      return;
    }
  }
  
  Alignments.push_back(TargetAlignElem::get(align_type, abi_align,
                                            pref_align, bit_width));
}

/// getAlignmentInfo - Return the alignment (either ABI if ABIInfo = true or 
/// preferred if ABIInfo = false) the target wants for the specified datatype.
unsigned TargetData::getAlignmentInfo(AlignTypeEnum AlignType, 
                                      uint32_t BitWidth, bool ABIInfo,
                                      const Type *Ty) const {
  // Check to see if we have an exact match and remember the best match we see.
  int BestMatchIdx = -1;
  int LargestInt = -1;
  for (unsigned i = 0, e = Alignments.size(); i != e; ++i) {
    if (Alignments[i].AlignType == AlignType &&
        Alignments[i].TypeBitWidth == BitWidth)
      return ABIInfo ? Alignments[i].ABIAlign : Alignments[i].PrefAlign;
    
    // The best match so far depends on what we're looking for.
    if (AlignType == VECTOR_ALIGN && Alignments[i].AlignType == VECTOR_ALIGN) {
      // If this is a specification for a smaller vector type, we will fall back
      // to it.  This happens because <128 x double> can be implemented in terms
      // of 64 <2 x double>.
      if (Alignments[i].TypeBitWidth < BitWidth) {
        // Verify that we pick the biggest of the fallbacks.
        if (BestMatchIdx == -1 ||
            Alignments[BestMatchIdx].TypeBitWidth < Alignments[i].TypeBitWidth)
          BestMatchIdx = i;
      }
    } else if (AlignType == INTEGER_ALIGN && 
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
      
      // If we didn't find a vector size that is smaller or equal to this type,
      // then we will end up scalarizing this to its element type.  Just return
      // the alignment of the element.
      return getAlignment(cast<VectorType>(Ty)->getElementType(), ABIInfo);
    }    
  }
    
  // Since we got a "best match" index, just return it.
  return ABIInfo ? Alignments[BestMatchIdx].ABIAlign
                 : Alignments[BestMatchIdx].PrefAlign;
}

namespace {

/// LayoutInfo - The lazy cache of structure layout information maintained by
/// TargetData.  Note that the struct types must have been free'd before
/// llvm_shutdown is called (and thus this is deallocated) because all the
/// targets with cached elements should have been destroyed.
///
typedef std::pair<const TargetData*,const StructType*> LayoutKey;

struct DenseMapLayoutKeyInfo {
  static inline LayoutKey getEmptyKey() { return LayoutKey(0, 0); }
  static inline LayoutKey getTombstoneKey() {
    return LayoutKey((TargetData*)(intptr_t)-1, 0);
  }
  static unsigned getHashValue(const LayoutKey &Val) {
    return DenseMapInfo<void*>::getHashValue(Val.first) ^
           DenseMapInfo<void*>::getHashValue(Val.second);
  }
  static bool isEqual(const LayoutKey &LHS, const LayoutKey &RHS) {
    return LHS == RHS;
  }

  static bool isPod() { return true; }
};

typedef DenseMap<LayoutKey, StructLayout*, DenseMapLayoutKeyInfo> LayoutInfoTy;

}

static ManagedStatic<LayoutInfoTy> LayoutInfo;

TargetData::~TargetData() {
  if (!LayoutInfo.isConstructed())
    return;
  
  // Remove any layouts for this TD.
  LayoutInfoTy &TheMap = *LayoutInfo;
  for (LayoutInfoTy::iterator I = TheMap.begin(), E = TheMap.end(); I != E; ) {
    if (I->first.first == this) {
      I->second->~StructLayout();
      free(I->second);
      TheMap.erase(I++);
    } else {
      ++I;
    }
  }
}

const StructLayout *TargetData::getStructLayout(const StructType *Ty) const {
  LayoutInfoTy &TheMap = *LayoutInfo;
  
  StructLayout *&SL = TheMap[LayoutKey(this, Ty)];
  if (SL) return SL;

  // Otherwise, create the struct layout.  Because it is variable length, we 
  // malloc it, then use placement new.
  int NumElts = Ty->getNumElements();
  StructLayout *L =
    (StructLayout *)malloc(sizeof(StructLayout)+(NumElts-1)*sizeof(uint64_t));
  
  // Set SL before calling StructLayout's ctor.  The ctor could cause other
  // entries to be added to TheMap, invalidating our reference.
  SL = L;
  
  new (L) StructLayout(Ty, *this);
  return L;
}

/// InvalidateStructLayoutInfo - TargetData speculatively caches StructLayout
/// objects.  If a TargetData object is alive when types are being refined and
/// removed, this method must be called whenever a StructType is removed to
/// avoid a dangling pointer in this cache.
void TargetData::InvalidateStructLayoutInfo(const StructType *Ty) const {
  if (!LayoutInfo.isConstructed()) return;  // No cache.
  
  LayoutInfoTy::iterator I = LayoutInfo->find(LayoutKey(this, Ty));
  if (I == LayoutInfo->end()) return;
  
  I->second->~StructLayout();
  free(I->second);
  LayoutInfo->erase(I);
}


std::string TargetData::getStringRepresentation() const {
  std::string repr;
  repr.append(LittleEndian ? "e" : "E");
  repr.append("-p:").append(itostr((int64_t) (PointerMemSize * 8))).
      append(":").append(itostr((int64_t) (PointerABIAlign * 8))).
      append(":").append(itostr((int64_t) (PointerPrefAlign * 8)));
  for (align_const_iterator I = Alignments.begin();
       I != Alignments.end();
       ++I) {
    repr.append("-").append(1, (char) I->AlignType).
      append(utostr((int64_t) I->TypeBitWidth)).
      append(":").append(utostr((uint64_t) (I->ABIAlign * 8))).
      append(":").append(utostr((uint64_t) (I->PrefAlign * 8)));
  }
  return repr;
}


uint64_t TargetData::getTypeSizeInBits(const Type *Ty) const {
  assert(Ty->isSized() && "Cannot getTypeInfo() on a type that is unsized!");
  switch (Ty->getTypeID()) {
  case Type::LabelTyID:
  case Type::PointerTyID:
    return getPointerSizeInBits();
  case Type::ArrayTyID: {
    const ArrayType *ATy = cast<ArrayType>(Ty);
    return getTypePaddedSizeInBits(ATy->getElementType())*ATy->getNumElements();
  }
  case Type::StructTyID:
    // Get the layout annotation... which is lazily created on demand.
    return getStructLayout(cast<StructType>(Ty))->getSizeInBits();
  case Type::IntegerTyID:
    return cast<IntegerType>(Ty)->getBitWidth();
  case Type::VoidTyID:
    return 8;
  case Type::FloatTyID:
    return 32;
  case Type::DoubleTyID:
    return 64;
  case Type::PPC_FP128TyID:
  case Type::FP128TyID:
    return 128;
  // In memory objects this is always aligned to a higher boundary, but
  // only 80 bits contain information.
  case Type::X86_FP80TyID:
    return 80;
  case Type::VectorTyID:
    return cast<VectorType>(Ty)->getBitWidth();
  default:
    assert(0 && "TargetData::getTypeSizeInBits(): Unsupported type");
    break;
  }
  return 0;
}

/*!
  \param abi_or_pref Flag that determines which alignment is returned. true
  returns the ABI alignment, false returns the preferred alignment.
  \param Ty The underlying type for which alignment is determined.

  Get the ABI (\a abi_or_pref == true) or preferred alignment (\a abi_or_pref
  == false) for the requested type \a Ty.
 */
unsigned char TargetData::getAlignment(const Type *Ty, bool abi_or_pref) const {
  int AlignType = -1;

  assert(Ty->isSized() && "Cannot getTypeInfo() on a type that is unsized!");
  switch (Ty->getTypeID()) {
  // Early escape for the non-numeric types.
  case Type::LabelTyID:
  case Type::PointerTyID:
    return (abi_or_pref
            ? getPointerABIAlignment()
            : getPointerPrefAlignment());
  case Type::ArrayTyID:
    return getAlignment(cast<ArrayType>(Ty)->getElementType(), abi_or_pref);
    
  case Type::StructTyID: {
    // Packed structure types always have an ABI alignment of one.
    if (cast<StructType>(Ty)->isPacked() && abi_or_pref)
      return 1;
    
    // Get the layout annotation... which is lazily created on demand.
    const StructLayout *Layout = getStructLayout(cast<StructType>(Ty));
    unsigned Align = getAlignmentInfo(AGGREGATE_ALIGN, 0, abi_or_pref, Ty);
    return std::max(Align, (unsigned)Layout->getAlignment());
  }
  case Type::IntegerTyID:
  case Type::VoidTyID:
    AlignType = INTEGER_ALIGN;
    break;
  case Type::FloatTyID:
  case Type::DoubleTyID:
  // PPC_FP128TyID and FP128TyID have different data contents, but the
  // same size and alignment, so they look the same here.
  case Type::PPC_FP128TyID:
  case Type::FP128TyID:
  case Type::X86_FP80TyID:
    AlignType = FLOAT_ALIGN;
    break;
  case Type::VectorTyID:
    AlignType = VECTOR_ALIGN;
    break;
  default:
    assert(0 && "Bad type for getAlignment!!!");
    break;
  }

  return getAlignmentInfo((AlignTypeEnum)AlignType, getTypeSizeInBits(Ty),
                          abi_or_pref, Ty);
}

unsigned char TargetData::getABITypeAlignment(const Type *Ty) const {
  return getAlignment(Ty, true);
}

unsigned char TargetData::getCallFrameTypeAlignment(const Type *Ty) const {
  for (unsigned i = 0, e = Alignments.size(); i != e; ++i)
    if (Alignments[i].AlignType == STACK_ALIGN)
      return Alignments[i].ABIAlign;

  return getABITypeAlignment(Ty);
}

unsigned char TargetData::getPrefTypeAlignment(const Type *Ty) const {
  return getAlignment(Ty, false);
}

unsigned char TargetData::getPreferredTypeAlignmentShift(const Type *Ty) const {
  unsigned Align = (unsigned) getPrefTypeAlignment(Ty);
  assert(!(Align & (Align-1)) && "Alignment is not a power of two!");
  return Log2_32(Align);
}

/// getIntPtrType - Return an unsigned integer type that is the same size or
/// greater to the host pointer size.
const Type *TargetData::getIntPtrType() const {
  return IntegerType::get(getPointerSizeInBits());
}


uint64_t TargetData::getIndexedOffset(const Type *ptrTy, Value* const* Indices,
                                      unsigned NumIndices) const {
  const Type *Ty = ptrTy;
  assert(isa<PointerType>(Ty) && "Illegal argument for getIndexedOffset()");
  uint64_t Result = 0;

  generic_gep_type_iterator<Value* const*>
    TI = gep_type_begin(ptrTy, Indices, Indices+NumIndices);
  for (unsigned CurIDX = 0; CurIDX != NumIndices; ++CurIDX, ++TI) {
    if (const StructType *STy = dyn_cast<StructType>(*TI)) {
      assert(Indices[CurIDX]->getType() == Type::Int32Ty &&
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
      int64_t arrayIdx = cast<ConstantInt>(Indices[CurIDX])->getSExtValue();
      Result += arrayIdx * (int64_t)getTypePaddedSize(Ty);
    }
  }

  return Result;
}

/// getPreferredAlignment - Return the preferred alignment of the specified
/// global.  This includes an explicitly requested alignment (if the global
/// has one).
unsigned TargetData::getPreferredAlignment(const GlobalVariable *GV) const {
  const Type *ElemType = GV->getType()->getElementType();
  unsigned Alignment = getPrefTypeAlignment(ElemType);
  if (GV->getAlignment() > Alignment)
    Alignment = GV->getAlignment();

  if (GV->hasInitializer()) {
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
unsigned TargetData::getPreferredAlignmentLog(const GlobalVariable *GV) const {
  return Log2_32(getPreferredAlignment(GV));
}
