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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemAlloc.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/TargetParser/Triple.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <utility>

using namespace llvm;

//===----------------------------------------------------------------------===//
// Support for StructLayout
//===----------------------------------------------------------------------===//

StructLayout::StructLayout(StructType *ST, const DataLayout &DL)
    : StructSize(TypeSize::getFixed(0)) {
  assert(!ST->isOpaque() && "Cannot get layout of opaque structs");
  IsPadded = false;
  NumElements = ST->getNumElements();

  // Loop over each of the elements, placing them in memory.
  for (unsigned i = 0, e = NumElements; i != e; ++i) {
    Type *Ty = ST->getElementType(i);
    if (i == 0 && Ty->isScalableTy())
      StructSize = TypeSize::getScalable(0);

    const Align TyAlign = ST->isPacked() ? Align(1) : DL.getABITypeAlign(Ty);

    // Add padding if necessary to align the data element properly.
    // Currently the only structure with scalable size will be the homogeneous
    // scalable vector types. Homogeneous scalable vector types have members of
    // the same data type so no alignment issue will happen. The condition here
    // assumes so and needs to be adjusted if this assumption changes (e.g. we
    // support structures with arbitrary scalable data type, or structure that
    // contains both fixed size and scalable size data type members).
    if (!StructSize.isScalable() && !isAligned(TyAlign, StructSize)) {
      IsPadded = true;
      StructSize = TypeSize::getFixed(alignTo(StructSize, TyAlign));
    }

    // Keep track of maximum alignment constraint.
    StructAlignment = std::max(TyAlign, StructAlignment);

    getMemberOffsets()[i] = StructSize;
    // Consume space for this data item
    StructSize += DL.getTypeAllocSize(Ty);
  }

  // Add padding to the end of the struct so that it could be put in an array
  // and all array elements would be aligned correctly.
  if (!StructSize.isScalable() && !isAligned(StructAlignment, StructSize)) {
    IsPadded = true;
    StructSize = TypeSize::getFixed(alignTo(StructSize, StructAlignment));
  }
}

/// getElementContainingOffset - Given a valid offset into the structure,
/// return the structure index that contains it.
unsigned StructLayout::getElementContainingOffset(uint64_t FixedOffset) const {
  assert(!StructSize.isScalable() &&
         "Cannot get element at offset for structure containing scalable "
         "vector types");
  TypeSize Offset = TypeSize::getFixed(FixedOffset);
  ArrayRef<TypeSize> MemberOffsets = getMemberOffsets();

  const auto *SI =
      std::upper_bound(MemberOffsets.begin(), MemberOffsets.end(), Offset,
                       [](TypeSize LHS, TypeSize RHS) -> bool {
                         return TypeSize::isKnownLT(LHS, RHS);
                       });
  assert(SI != MemberOffsets.begin() && "Offset not in structure type!");
  --SI;
  assert(TypeSize::isKnownLE(*SI, Offset) && "upper_bound didn't work");
  assert(
      (SI == MemberOffsets.begin() || TypeSize::isKnownLE(*(SI - 1), Offset)) &&
      (SI + 1 == MemberOffsets.end() ||
       TypeSize::isKnownGT(*(SI + 1), Offset)) &&
      "Upper bound didn't work!");

  // Multiple fields can have the same offset if any of them are zero sized.
  // For example, in { i32, [0 x i32], i32 }, searching for offset 4 will stop
  // at the i32 element, because it is the last element at that offset.  This is
  // the right one to return, because anything after it will have a higher
  // offset, implying that this element is non-empty.
  return SI - MemberOffsets.begin();
}

namespace {

class StructLayoutMap {
  using LayoutInfoTy = DenseMap<StructType *, StructLayout *>;
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

  StructLayout *&operator[](StructType *STy) { return LayoutInfo[STy]; }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
//                       DataLayout Class Implementation
//===----------------------------------------------------------------------===//

bool DataLayout::PrimitiveSpec::operator==(const PrimitiveSpec &Other) const {
  return BitWidth == Other.BitWidth && ABIAlign == Other.ABIAlign &&
         PrefAlign == Other.PrefAlign;
}

bool DataLayout::PointerSpec::operator==(const PointerSpec &Other) const {
  return AddrSpace == Other.AddrSpace && BitWidth == Other.BitWidth &&
         ABIAlign == Other.ABIAlign && PrefAlign == Other.PrefAlign &&
         IndexBitWidth == Other.IndexBitWidth;
}

namespace {
/// Predicate to sort primitive specs by bit width.
struct LessPrimitiveBitWidth {
  bool operator()(const DataLayout::PrimitiveSpec &LHS,
                  unsigned RHSBitWidth) const {
    return LHS.BitWidth < RHSBitWidth;
  }
};

/// Predicate to sort pointer specs by address space number.
struct LessPointerAddrSpace {
  bool operator()(const DataLayout::PointerSpec &LHS,
                  unsigned RHSAddrSpace) const {
    return LHS.AddrSpace < RHSAddrSpace;
  }
};
} // namespace

const char *DataLayout::getManglingComponent(const Triple &T) {
  if (T.isOSBinFormatGOFF())
    return "-m:l";
  if (T.isOSBinFormatMachO())
    return "-m:o";
  if ((T.isOSWindows() || T.isUEFI()) && T.isOSBinFormatCOFF())
    return T.getArch() == Triple::x86 ? "-m:x" : "-m:w";
  if (T.isOSBinFormatXCOFF())
    return "-m:a";
  return "-m:e";
}

// Default primitive type specifications.
// NOTE: These arrays must be sorted by type bit width.
constexpr DataLayout::PrimitiveSpec DefaultIntSpecs[] = {
    {1, Align::Constant<1>(), Align::Constant<1>()},  // i1:8:8
    {8, Align::Constant<1>(), Align::Constant<1>()},  // i8:8:8
    {16, Align::Constant<2>(), Align::Constant<2>()}, // i16:16:16
    {32, Align::Constant<4>(), Align::Constant<4>()}, // i32:32:32
    {64, Align::Constant<4>(), Align::Constant<8>()}, // i64:32:64
};
constexpr DataLayout::PrimitiveSpec DefaultFloatSpecs[] = {
    {16, Align::Constant<2>(), Align::Constant<2>()},    // f16:16:16
    {32, Align::Constant<4>(), Align::Constant<4>()},    // f32:32:32
    {64, Align::Constant<8>(), Align::Constant<8>()},    // f64:64:64
    {128, Align::Constant<16>(), Align::Constant<16>()}, // f128:128:128
};
constexpr DataLayout::PrimitiveSpec DefaultVectorSpecs[] = {
    {64, Align::Constant<8>(), Align::Constant<8>()},    // v64:64:64
    {128, Align::Constant<16>(), Align::Constant<16>()}, // v128:128:128
};

// Default pointer type specifications.
constexpr DataLayout::PointerSpec DefaultPointerSpecs[] = {
    {0, 64, Align::Constant<8>(), Align::Constant<8>(), 64} // p0:64:64:64:64
};

DataLayout::DataLayout()
    : IntSpecs(ArrayRef(DefaultIntSpecs)),
      FloatSpecs(ArrayRef(DefaultFloatSpecs)),
      VectorSpecs(ArrayRef(DefaultVectorSpecs)),
      PointerSpecs(ArrayRef(DefaultPointerSpecs)) {}

DataLayout::DataLayout(StringRef LayoutString) : DataLayout() {
  if (Error Err = parseLayoutString(LayoutString))
    report_fatal_error(std::move(Err));
}

DataLayout &DataLayout::operator=(const DataLayout &Other) {
  delete static_cast<StructLayoutMap *>(LayoutMap);
  LayoutMap = nullptr;
  StringRepresentation = Other.StringRepresentation;
  BigEndian = Other.BigEndian;
  AllocaAddrSpace = Other.AllocaAddrSpace;
  ProgramAddrSpace = Other.ProgramAddrSpace;
  DefaultGlobalsAddrSpace = Other.DefaultGlobalsAddrSpace;
  StackNaturalAlign = Other.StackNaturalAlign;
  FunctionPtrAlign = Other.FunctionPtrAlign;
  TheFunctionPtrAlignType = Other.TheFunctionPtrAlignType;
  ManglingMode = Other.ManglingMode;
  LegalIntWidths = Other.LegalIntWidths;
  IntSpecs = Other.IntSpecs;
  FloatSpecs = Other.FloatSpecs;
  VectorSpecs = Other.VectorSpecs;
  PointerSpecs = Other.PointerSpecs;
  StructABIAlignment = Other.StructABIAlignment;
  StructPrefAlignment = Other.StructPrefAlignment;
  NonIntegralAddressSpaces = Other.NonIntegralAddressSpaces;
  return *this;
}

bool DataLayout::operator==(const DataLayout &Other) const {
  // NOTE: StringRepresentation might differ, it is not canonicalized.
  // FIXME: NonIntegralAddressSpaces isn't compared.
  return BigEndian == Other.BigEndian &&
         AllocaAddrSpace == Other.AllocaAddrSpace &&
         ProgramAddrSpace == Other.ProgramAddrSpace &&
         DefaultGlobalsAddrSpace == Other.DefaultGlobalsAddrSpace &&
         StackNaturalAlign == Other.StackNaturalAlign &&
         FunctionPtrAlign == Other.FunctionPtrAlign &&
         TheFunctionPtrAlignType == Other.TheFunctionPtrAlignType &&
         ManglingMode == Other.ManglingMode &&
         LegalIntWidths == Other.LegalIntWidths && IntSpecs == Other.IntSpecs &&
         FloatSpecs == Other.FloatSpecs && VectorSpecs == Other.VectorSpecs &&
         PointerSpecs == Other.PointerSpecs &&
         StructABIAlignment == Other.StructABIAlignment &&
         StructPrefAlignment == Other.StructPrefAlignment;
}

Expected<DataLayout> DataLayout::parse(StringRef LayoutString) {
  DataLayout Layout;
  if (Error Err = Layout.parseLayoutString(LayoutString))
    return std::move(Err);
  return Layout;
}

static Error createSpecFormatError(Twine Format) {
  return createStringError("malformed specification, must be of the form \"" +
                           Format + "\"");
}

/// Attempts to parse an address space component of a specification.
static Error parseAddrSpace(StringRef Str, unsigned &AddrSpace) {
  if (Str.empty())
    return createStringError("address space component cannot be empty");

  if (!to_integer(Str, AddrSpace, 10) || !isUInt<24>(AddrSpace))
    return createStringError("address space must be a 24-bit integer");

  return Error::success();
}

/// Attempts to parse a size component of a specification.
static Error parseSize(StringRef Str, unsigned &BitWidth,
                       StringRef Name = "size") {
  if (Str.empty())
    return createStringError(Name + " component cannot be empty");

  if (!to_integer(Str, BitWidth, 10) || BitWidth == 0 || !isUInt<24>(BitWidth))
    return createStringError(Name + " must be a non-zero 24-bit integer");

  return Error::success();
}

/// Attempts to parse an alignment component of a specification.
///
/// On success, returns the value converted to byte amount in \p Alignment.
/// If the value is zero and \p AllowZero is true, \p Alignment is set to one.
///
/// Return an error in a number of cases:
/// - \p Str is empty or contains characters other than decimal digits;
/// - the value is zero and \p AllowZero is false;
/// - the value is too large;
/// - the value is not a multiple of the byte width;
/// - the value converted to byte amount is not not a power of two.
static Error parseAlignment(StringRef Str, Align &Alignment, StringRef Name,
                            bool AllowZero = false) {
  if (Str.empty())
    return createStringError(Name + " alignment component cannot be empty");

  unsigned Value;
  if (!to_integer(Str, Value, 10) || !isUInt<16>(Value))
    return createStringError(Name + " alignment must be a 16-bit integer");

  if (Value == 0) {
    if (!AllowZero)
      return createStringError(Name + " alignment must be non-zero");
    Alignment = Align(1);
    return Error::success();
  }

  constexpr unsigned ByteWidth = 8;
  if (Value % ByteWidth || !isPowerOf2_32(Value / ByteWidth))
    return createStringError(
        Name + " alignment must be a power of two times the byte width");

  Alignment = Align(Value / ByteWidth);
  return Error::success();
}

Error DataLayout::parsePrimitiveSpec(StringRef Spec) {
  // [ifv]<size>:<abi>[:<pref>]
  SmallVector<StringRef, 3> Components;
  char Specifier = Spec.front();
  assert(Specifier == 'i' || Specifier == 'f' || Specifier == 'v');
  Spec.drop_front().split(Components, ':');

  if (Components.size() < 2 || Components.size() > 3)
    return createSpecFormatError(Twine(Specifier) + "<size>:<abi>[:<pref>]");

  // Size. Required, cannot be zero.
  unsigned BitWidth;
  if (Error Err = parseSize(Components[0], BitWidth))
    return Err;

  // ABI alignment.
  Align ABIAlign;
  if (Error Err = parseAlignment(Components[1], ABIAlign, "ABI"))
    return Err;

  if (Specifier == 'i' && BitWidth == 8 && ABIAlign != 1)
    return createStringError("i8 must be 8-bit aligned");

  // Preferred alignment. Optional, defaults to the ABI alignment.
  Align PrefAlign = ABIAlign;
  if (Components.size() > 2)
    if (Error Err = parseAlignment(Components[2], PrefAlign, "preferred"))
      return Err;

  if (PrefAlign < ABIAlign)
    return createStringError(
        "preferred alignment cannot be less than the ABI alignment");

  setPrimitiveSpec(Specifier, BitWidth, ABIAlign, PrefAlign);
  return Error::success();
}

Error DataLayout::parseAggregateSpec(StringRef Spec) {
  // a<size>:<abi>[:<pref>]
  SmallVector<StringRef, 3> Components;
  assert(Spec.front() == 'a');
  Spec.drop_front().split(Components, ':');

  if (Components.size() < 2 || Components.size() > 3)
    return createSpecFormatError("a:<abi>[:<pref>]");

  // According to LangRef, <size> component must be absent altogether.
  // For backward compatibility, allow it to be specified, but require
  // it to be zero.
  if (!Components[0].empty()) {
    unsigned BitWidth;
    if (!to_integer(Components[0], BitWidth, 10) || BitWidth != 0)
      return createStringError("size must be zero");
  }

  // ABI alignment. Required. Can be zero, meaning use one byte alignment.
  Align ABIAlign;
  if (Error Err =
          parseAlignment(Components[1], ABIAlign, "ABI", /*AllowZero=*/true))
    return Err;

  // Preferred alignment. Optional, defaults to the ABI alignment.
  Align PrefAlign = ABIAlign;
  if (Components.size() > 2)
    if (Error Err = parseAlignment(Components[2], PrefAlign, "preferred"))
      return Err;

  if (PrefAlign < ABIAlign)
    return createStringError(
        "preferred alignment cannot be less than the ABI alignment");

  StructABIAlignment = ABIAlign;
  StructPrefAlignment = PrefAlign;
  return Error::success();
}

Error DataLayout::parsePointerSpec(StringRef Spec) {
  // p[<n>]:<size>:<abi>[:<pref>[:<idx>]]
  SmallVector<StringRef, 5> Components;
  assert(Spec.front() == 'p');
  Spec.drop_front().split(Components, ':');

  if (Components.size() < 3 || Components.size() > 5)
    return createSpecFormatError("p[<n>]:<size>:<abi>[:<pref>[:<idx>]]");

  // Address space. Optional, defaults to 0.
  unsigned AddrSpace = 0;
  if (!Components[0].empty())
    if (Error Err = parseAddrSpace(Components[0], AddrSpace))
      return Err;

  // Size. Required, cannot be zero.
  unsigned BitWidth;
  if (Error Err = parseSize(Components[1], BitWidth, "pointer size"))
    return Err;

  // ABI alignment. Required, cannot be zero.
  Align ABIAlign;
  if (Error Err = parseAlignment(Components[2], ABIAlign, "ABI"))
    return Err;

  // Preferred alignment. Optional, defaults to the ABI alignment.
  // Cannot be zero.
  Align PrefAlign = ABIAlign;
  if (Components.size() > 3)
    if (Error Err = parseAlignment(Components[3], PrefAlign, "preferred"))
      return Err;

  if (PrefAlign < ABIAlign)
    return createStringError(
        "preferred alignment cannot be less than the ABI alignment");

  // Index size. Optional, defaults to pointer size. Cannot be zero.
  unsigned IndexBitWidth = BitWidth;
  if (Components.size() > 4)
    if (Error Err = parseSize(Components[4], IndexBitWidth, "index size"))
      return Err;

  if (IndexBitWidth > BitWidth)
    return createStringError(
        "index size cannot be larger than the pointer size");

  setPointerSpec(AddrSpace, BitWidth, ABIAlign, PrefAlign, IndexBitWidth);
  return Error::success();
}

Error DataLayout::parseSpecification(StringRef Spec) {
  // The "ni" specifier is the only two-character specifier. Handle it first.
  if (Spec.starts_with("ni")) {
    // ni:<address space>[:<address space>]...
    StringRef Rest = Spec.drop_front(2);

    // Drop the first ':', then split the rest of the string the usual way.
    if (!Rest.consume_front(":"))
      return createSpecFormatError("ni:<address space>[:<address space>]...");

    for (StringRef Str : split(Rest, ':')) {
      unsigned AddrSpace;
      if (Error Err = parseAddrSpace(Str, AddrSpace))
        return Err;
      if (AddrSpace == 0)
        return createStringError("address space 0 cannot be non-integral");
      NonIntegralAddressSpaces.push_back(AddrSpace);
    }
    return Error::success();
  }

  // The rest of the specifiers are single-character.
  assert(!Spec.empty() && "Empty specification is handled by the caller");
  char Specifier = Spec.front();

  if (Specifier == 'i' || Specifier == 'f' || Specifier == 'v')
    return parsePrimitiveSpec(Spec);

  if (Specifier == 'a')
    return parseAggregateSpec(Spec);

  if (Specifier == 'p')
    return parsePointerSpec(Spec);

  StringRef Rest = Spec.drop_front();
  switch (Specifier) {
  case 's':
    // Deprecated, but ignoring here to preserve loading older textual llvm
    // ASM file
    break;
  case 'e':
  case 'E':
    if (!Rest.empty())
      return createStringError(
          "malformed specification, must be just 'e' or 'E'");
    BigEndian = Specifier == 'E';
    break;
  case 'n': // Native integer types.
    // n<size>[:<size>]...
    for (StringRef Str : split(Rest, ':')) {
      unsigned BitWidth;
      if (Error Err = parseSize(Str, BitWidth))
        return Err;
      LegalIntWidths.push_back(BitWidth);
    }
    break;
  case 'S': { // Stack natural alignment.
    // S<size>
    if (Rest.empty())
      return createSpecFormatError("S<size>");
    Align Alignment;
    if (Error Err = parseAlignment(Rest, Alignment, "stack natural"))
      return Err;
    StackNaturalAlign = Alignment;
    break;
  }
  case 'F': {
    // F<type><abi>
    if (Rest.empty())
      return createSpecFormatError("F<type><abi>");
    char Type = Rest.front();
    Rest = Rest.drop_front();
    switch (Type) {
    case 'i':
      TheFunctionPtrAlignType = FunctionPtrAlignType::Independent;
      break;
    case 'n':
      TheFunctionPtrAlignType = FunctionPtrAlignType::MultipleOfFunctionAlign;
      break;
    default:
      return createStringError("unknown function pointer alignment type '" +
                               Twine(Type) + "'");
    }
    Align Alignment;
    if (Error Err = parseAlignment(Rest, Alignment, "ABI"))
      return Err;
    FunctionPtrAlign = Alignment;
    break;
  }
  case 'P': { // Function address space.
    if (Rest.empty())
      return createSpecFormatError("P<address space>");
    if (Error Err = parseAddrSpace(Rest, ProgramAddrSpace))
      return Err;
    break;
  }
  case 'A': { // Default stack/alloca address space.
    if (Rest.empty())
      return createSpecFormatError("A<address space>");
    if (Error Err = parseAddrSpace(Rest, AllocaAddrSpace))
      return Err;
    break;
  }
  case 'G': { // Default address space for global variables.
    if (Rest.empty())
      return createSpecFormatError("G<address space>");
    if (Error Err = parseAddrSpace(Rest, DefaultGlobalsAddrSpace))
      return Err;
    break;
  }
  case 'm':
    if (!Rest.consume_front(":") || Rest.empty())
      return createSpecFormatError("m:<mangling>");
    if (Rest.size() > 1)
      return createStringError("unknown mangling mode");
    switch (Rest[0]) {
    default:
      return createStringError("unknown mangling mode");
    case 'e':
      ManglingMode = MM_ELF;
      break;
    case 'l':
      ManglingMode = MM_GOFF;
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
    case 'a':
      ManglingMode = MM_XCOFF;
      break;
    }
    break;
  default:
    return createStringError("unknown specifier '" + Twine(Specifier) + "'");
  }

  return Error::success();
}

Error DataLayout::parseLayoutString(StringRef LayoutString) {
  StringRepresentation = std::string(LayoutString);

  if (LayoutString.empty())
    return Error::success();

  // Split the data layout string into specifications separated by '-' and
  // parse each specification individually, updating internal data structures.
  for (StringRef Spec : split(LayoutString, '-')) {
    if (Spec.empty())
      return createStringError("empty specification is not allowed");
    if (Error Err = parseSpecification(Spec))
      return Err;
  }

  return Error::success();
}

void DataLayout::setPrimitiveSpec(char Specifier, uint32_t BitWidth,
                                  Align ABIAlign, Align PrefAlign) {
  SmallVectorImpl<PrimitiveSpec> *Specs;
  switch (Specifier) {
  default:
    llvm_unreachable("Unexpected specifier");
  case 'i':
    Specs = &IntSpecs;
    break;
  case 'f':
    Specs = &FloatSpecs;
    break;
  case 'v':
    Specs = &VectorSpecs;
    break;
  }

  auto I = lower_bound(*Specs, BitWidth, LessPrimitiveBitWidth());
  if (I != Specs->end() && I->BitWidth == BitWidth) {
    // Update the abi, preferred alignments.
    I->ABIAlign = ABIAlign;
    I->PrefAlign = PrefAlign;
  } else {
    // Insert before I to keep the vector sorted.
    Specs->insert(I, PrimitiveSpec{BitWidth, ABIAlign, PrefAlign});
  }
}

const DataLayout::PointerSpec &
DataLayout::getPointerSpec(uint32_t AddrSpace) const {
  if (AddrSpace != 0) {
    auto I = lower_bound(PointerSpecs, AddrSpace, LessPointerAddrSpace());
    if (I != PointerSpecs.end() && I->AddrSpace == AddrSpace)
      return *I;
  }

  assert(PointerSpecs[0].AddrSpace == 0);
  return PointerSpecs[0];
}

void DataLayout::setPointerSpec(uint32_t AddrSpace, uint32_t BitWidth,
                                Align ABIAlign, Align PrefAlign,
                                uint32_t IndexBitWidth) {
  auto I = lower_bound(PointerSpecs, AddrSpace, LessPointerAddrSpace());
  if (I == PointerSpecs.end() || I->AddrSpace != AddrSpace) {
    PointerSpecs.insert(I, PointerSpec{AddrSpace, BitWidth, ABIAlign, PrefAlign,
                                       IndexBitWidth});
  } else {
    I->BitWidth = BitWidth;
    I->ABIAlign = ABIAlign;
    I->PrefAlign = PrefAlign;
    I->IndexBitWidth = IndexBitWidth;
  }
}

Align DataLayout::getIntegerAlignment(uint32_t BitWidth,
                                      bool abi_or_pref) const {
  auto I = lower_bound(IntSpecs, BitWidth, LessPrimitiveBitWidth());
  // If we don't have an exact match, use alignment of next larger integer
  // type. If there is none, use alignment of largest integer type by going
  // back one element.
  if (I == IntSpecs.end())
    --I;
  return abi_or_pref ? I->ABIAlign : I->PrefAlign;
}

DataLayout::~DataLayout() { delete static_cast<StructLayoutMap *>(LayoutMap); }

const StructLayout *DataLayout::getStructLayout(StructType *Ty) const {
  if (!LayoutMap)
    LayoutMap = new StructLayoutMap();

  StructLayoutMap *STM = static_cast<StructLayoutMap*>(LayoutMap);
  StructLayout *&SL = (*STM)[Ty];
  if (SL) return SL;

  // Otherwise, create the struct layout.  Because it is variable length, we
  // malloc it, then use placement new.
  StructLayout *L = (StructLayout *)safe_malloc(
      StructLayout::totalSizeToAlloc<TypeSize>(Ty->getNumElements()));

  // Set SL before calling StructLayout's ctor.  The ctor could cause other
  // entries to be added to TheMap, invalidating our reference.
  SL = L;

  new (L) StructLayout(Ty, *this);

  return L;
}

Align DataLayout::getPointerABIAlignment(unsigned AS) const {
  return getPointerSpec(AS).ABIAlign;
}

Align DataLayout::getPointerPrefAlignment(unsigned AS) const {
  return getPointerSpec(AS).PrefAlign;
}

unsigned DataLayout::getPointerSize(unsigned AS) const {
  return divideCeil(getPointerSpec(AS).BitWidth, 8);
}

unsigned DataLayout::getMaxIndexSize() const {
  unsigned MaxIndexSize = 0;
  for (const PointerSpec &Spec : PointerSpecs)
    MaxIndexSize =
        std::max(MaxIndexSize, (unsigned)divideCeil(Spec.BitWidth, 8));

  return MaxIndexSize;
}

unsigned DataLayout::getPointerTypeSizeInBits(Type *Ty) const {
  assert(Ty->isPtrOrPtrVectorTy() &&
         "This should only be called with a pointer or pointer vector type");
  Ty = Ty->getScalarType();
  return getPointerSizeInBits(cast<PointerType>(Ty)->getAddressSpace());
}

unsigned DataLayout::getIndexSize(unsigned AS) const {
  return divideCeil(getPointerSpec(AS).IndexBitWidth, 8);
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
    const Align Align = abi_or_pref ? StructABIAlignment : StructPrefAlignment;
    return std::max(Align, Layout->getAlignment());
  }
  case Type::IntegerTyID:
    return getIntegerAlignment(Ty->getIntegerBitWidth(), abi_or_pref);
  case Type::HalfTyID:
  case Type::BFloatTyID:
  case Type::FloatTyID:
  case Type::DoubleTyID:
  // PPC_FP128TyID and FP128TyID have different data contents, but the
  // same size and alignment, so they look the same here.
  case Type::PPC_FP128TyID:
  case Type::FP128TyID:
  case Type::X86_FP80TyID: {
    unsigned BitWidth = getTypeSizeInBits(Ty).getFixedValue();
    auto I = lower_bound(FloatSpecs, BitWidth, LessPrimitiveBitWidth());
    if (I != FloatSpecs.end() && I->BitWidth == BitWidth)
      return abi_or_pref ? I->ABIAlign : I->PrefAlign;

    // If we still couldn't find a reasonable default alignment, fall back
    // to a simple heuristic that the alignment is the first power of two
    // greater-or-equal to the store size of the type.  This is a reasonable
    // approximation of reality, and if the user wanted something less
    // less conservative, they should have specified it explicitly in the data
    // layout.
    return Align(PowerOf2Ceil(BitWidth / 8));
  }
  case Type::FixedVectorTyID:
  case Type::ScalableVectorTyID: {
    unsigned BitWidth = getTypeSizeInBits(Ty).getKnownMinValue();
    auto I = lower_bound(VectorSpecs, BitWidth, LessPrimitiveBitWidth());
    if (I != VectorSpecs.end() && I->BitWidth == BitWidth)
      return abi_or_pref ? I->ABIAlign : I->PrefAlign;

    // By default, use natural alignment for vector types. This is consistent
    // with what clang and llvm-gcc do.
    //
    // We're only calculating a natural alignment, so it doesn't have to be
    // based on the full size for scalable vectors. Using the minimum element
    // count should be enough here.
    return Align(PowerOf2Ceil(getTypeStoreSize(Ty).getKnownMinValue()));
  }
  case Type::X86_AMXTyID:
    return Align(64);
  case Type::TargetExtTyID: {
    Type *LayoutTy = cast<TargetExtType>(Ty)->getLayoutType();
    return getAlignment(LayoutTy, abi_or_pref);
  }
  default:
    llvm_unreachable("Bad type for getAlignment!!!");
  }
}

Align DataLayout::getABITypeAlign(Type *Ty) const {
  return getAlignment(Ty, true);
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
    return VectorType::get(IntTy, VecTy);
  return IntTy;
}

Type *DataLayout::getSmallestLegalIntType(LLVMContext &C, unsigned Width) const {
  for (unsigned LegalIntWidth : LegalIntWidths)
    if (Width <= LegalIntWidth)
      return Type::getIntNTy(C, LegalIntWidth);
  return nullptr;
}

unsigned DataLayout::getLargestLegalIntTypeSizeInBits() const {
  auto Max = llvm::max_element(LegalIntWidths);
  return Max != LegalIntWidths.end() ? *Max : 0;
}

IntegerType *DataLayout::getIndexType(LLVMContext &C,
                                      unsigned AddressSpace) const {
  return IntegerType::get(C, getIndexSizeInBits(AddressSpace));
}

Type *DataLayout::getIndexType(Type *Ty) const {
  assert(Ty->isPtrOrPtrVectorTy() &&
         "Expected a pointer or pointer vector type.");
  unsigned NumBits = getIndexTypeSizeInBits(Ty);
  IntegerType *IntTy = IntegerType::get(Ty->getContext(), NumBits);
  if (VectorType *VecTy = dyn_cast<VectorType>(Ty))
    return VectorType::get(IntTy, VecTy);
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
      if (int64_t ArrayIdx = cast<ConstantInt>(Idx)->getSExtValue())
        Result += ArrayIdx * GTI.getSequentialElementStride(*this);
    }
  }

  return Result;
}

static APInt getElementIndex(TypeSize ElemSize, APInt &Offset) {
  // Skip over scalable or zero size elements. Also skip element sizes larger
  // than the positive index space, because the arithmetic below may not be
  // correct in that case.
  unsigned BitWidth = Offset.getBitWidth();
  if (ElemSize.isScalable() || ElemSize == 0 ||
      !isUIntN(BitWidth - 1, ElemSize)) {
    return APInt::getZero(BitWidth);
  }

  APInt Index = Offset.sdiv(ElemSize);
  Offset -= Index * ElemSize;
  if (Offset.isNegative()) {
    // Prefer a positive remaining offset to allow struct indexing.
    --Index;
    Offset += ElemSize;
    assert(Offset.isNonNegative() && "Remaining offset shouldn't be negative");
  }
  return Index;
}

std::optional<APInt> DataLayout::getGEPIndexForOffset(Type *&ElemTy,
                                                      APInt &Offset) const {
  if (auto *ArrTy = dyn_cast<ArrayType>(ElemTy)) {
    ElemTy = ArrTy->getElementType();
    return getElementIndex(getTypeAllocSize(ElemTy), Offset);
  }

  if (isa<VectorType>(ElemTy)) {
    // Vector GEPs are partially broken (e.g. for overaligned element types),
    // and may be forbidden in the future, so avoid generating GEPs into
    // vectors. See https://discourse.llvm.org/t/67497
    return std::nullopt;
  }

  if (auto *STy = dyn_cast<StructType>(ElemTy)) {
    const StructLayout *SL = getStructLayout(STy);
    uint64_t IntOffset = Offset.getZExtValue();
    if (IntOffset >= SL->getSizeInBytes())
      return std::nullopt;

    unsigned Index = SL->getElementContainingOffset(IntOffset);
    Offset -= SL->getElementOffset(Index);
    ElemTy = STy->getElementType(Index);
    return APInt(32, Index);
  }

  // Non-aggregate type.
  return std::nullopt;
}

SmallVector<APInt> DataLayout::getGEPIndicesForOffset(Type *&ElemTy,
                                                      APInt &Offset) const {
  assert(ElemTy->isSized() && "Element type must be sized");
  SmallVector<APInt> Indices;
  Indices.push_back(getElementIndex(getTypeAllocSize(ElemTy), Offset));
  while (Offset != 0) {
    std::optional<APInt> Index = getGEPIndexForOffset(ElemTy, Offset);
    if (!Index)
      break;
    Indices.push_back(*Index);
  }

  return Indices;
}

/// getPreferredAlign - Return the preferred alignment of the specified global.
/// This includes an explicitly requested alignment (if the global has one).
Align DataLayout::getPreferredAlign(const GlobalVariable *GV) const {
  MaybeAlign GVAlignment = GV->getAlign();
  // If a section is specified, always precisely honor explicit alignment,
  // so we don't insert padding into a section we don't control.
  if (GVAlignment && GV->hasSection())
    return *GVAlignment;

  // If no explicit alignment is specified, compute the alignment based on
  // the IR type. If an alignment is specified, increase it to match the ABI
  // alignment of the IR type.
  //
  // FIXME: Not sure it makes sense to use the alignment of the type if
  // there's already an explicit alignment specification.
  Type *ElemType = GV->getValueType();
  Align Alignment = getPrefTypeAlign(ElemType);
  if (GVAlignment) {
    if (*GVAlignment >= Alignment)
      Alignment = *GVAlignment;
    else
      Alignment = std::max(*GVAlignment, getABITypeAlign(ElemType));
  }

  // If no explicit alignment is specified, and the global is large, increase
  // the alignment to 16.
  // FIXME: Why 16, specifically?
  if (GV->hasInitializer() && !GVAlignment) {
    if (Alignment < Align(16)) {
      // If the global is not external, see if it is large.  If so, give it a
      // larger alignment.
      if (getTypeSizeInBits(ElemType) > 128)
        Alignment = Align(16); // 16-byte alignment.
    }
  }
  return Alignment;
}
