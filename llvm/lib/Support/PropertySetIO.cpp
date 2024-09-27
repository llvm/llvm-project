//==- PropertySetIO.cpp - models a sequence of property sets and their I/O -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/PropertySetIO.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LineIterator.h"

#include <memory>
#include <string>

using namespace llvm::util;
using namespace llvm;

namespace {

using byte = Base64::byte;

::llvm::Error makeError(const Twine &Msg) {
  return createStringError(std::error_code{}, Msg);
}

} // anonymous namespace

Expected<std::unique_ptr<PropertySetRegistry>>
PropertySetRegistry::read(const MemoryBuffer *Buf) {
  auto Res = std::make_unique<PropertySetRegistry>();
  PropertySet *CurPropSet = nullptr;

  for (line_iterator LI(*Buf); !LI.is_at_end(); LI++) {
    // see if this line starts a new property set
    if (LI->starts_with("[")) {
      // yes - parse the category (property name)
      auto EndPos = LI->rfind(']');
      if (EndPos == StringRef::npos)
        return makeError("invalid line: " + *LI);
      StringRef Category = LI->substr(1, EndPos - 1);
      CurPropSet = &(*Res)[Category];
      continue;
    }
    if (!CurPropSet)
      return makeError("property category missing");
    // parse name and type+value
    auto Parts = LI->split('=');

    if (Parts.first.empty() || Parts.second.empty())
      return makeError("invalid property line: " + *LI);
    auto TypeVal = Parts.second.split('|');

    if (TypeVal.first.empty() || TypeVal.second.empty())
      return makeError("invalid property value: " + Parts.second);
    APInt Tint;

    // parse type
    if (TypeVal.first.getAsInteger(10, Tint))
      return makeError("invalid property type: " + TypeVal.first);
    Expected<PropertyValue::Type> Ttag =
        PropertyValue::getTypeTag(static_cast<int>(Tint.getSExtValue()));
    StringRef Val = TypeVal.second;

    if (!Ttag)
      return Ttag.takeError();
    PropertyValue Prop(Ttag.get());

    // parse value depending on its type
    switch (Ttag.get()) {
    case PropertyValue::Type::UINT32: {
      APInt ValV;
      if (Val.getAsInteger(10, ValV))
        return createStringError(std::error_code{},
                                 "invalid property value: ", Val.data());
      Prop.set(static_cast<uint32_t>(ValV.getZExtValue()));
      break;
    }
    case PropertyValue::Type::BYTE_ARRAY: {
      Expected<std::unique_ptr<byte[]>> DecArr =
          Base64::decode(Val.data(), Val.size());
      if (!DecArr)
        return DecArr.takeError();
      Prop.set(DecArr.get().release());
      break;
    }
    default:
      return createStringError(std::error_code{},
                               "unsupported property type: ", Ttag.get());
    }
    (*CurPropSet)[Parts.first] = std::move(Prop);
  }
  if (!CurPropSet)
    return makeError("invalid property set registry");

  return Expected<std::unique_ptr<PropertySetRegistry>>(std::move(Res));
}

namespace llvm {
// output a property to a stream
raw_ostream &operator<<(raw_ostream &Out, const PropertyValue &Prop) {
  Out << static_cast<int>(Prop.getType()) << "|";
  switch (Prop.getType()) {
  case PropertyValue::Type::UINT32:
    Out << Prop.asUint32();
    break;
  case PropertyValue::Type::BYTE_ARRAY: {
    util::PropertyValue::SizeTy Size = Prop.getRawByteArraySize();
    Base64::encode(Prop.asRawByteArray(), Out, (size_t)Size);
    break;
  }
  default:
    llvm_unreachable(
        ("unsupported property type: " + utostr(Prop.getType())).c_str());
  }
  return Out;
}
} // namespace llvm

void PropertySetRegistry::write(raw_ostream &Out) const {
  for (const auto &PropSet : PropSetMap) {
    Out << "[" << PropSet.first << "]\n";

    for (const auto &Props : PropSet.second) {
      Out << Props.first << "=" << Props.second << "\n";
    }
  }
}

namespace llvm {
namespace util {

template <> uint32_t &PropertyValue::getValueRef<uint32_t>() {
  return Val.UInt32Val;
}

template <> byte *&PropertyValue::getValueRef<byte *>() {
  return Val.ByteArrayVal;
}

template <> PropertyValue::Type PropertyValue::getTypeTag<uint32_t>() {
  return UINT32;
}

template <> PropertyValue::Type PropertyValue::getTypeTag<byte *>() {
  return BYTE_ARRAY;
}

PropertyValue::PropertyValue(const byte *Data, SizeTy DataBitSize) {
  constexpr int ByteSizeInBits = 8;
  Ty = BYTE_ARRAY;
  SizeTy DataSize = (DataBitSize + (ByteSizeInBits - 1)) / ByteSizeInBits;
  constexpr size_t SizeFieldSize = sizeof(SizeTy);

  // Allocate space for size and data.
  Val.ByteArrayVal = new byte[SizeFieldSize + DataSize];

  // Write the size into first bytes.
  for (size_t I = 0; I < SizeFieldSize; ++I) {
    Val.ByteArrayVal[I] = (byte)DataBitSize;
    DataBitSize >>= ByteSizeInBits;
  }
  // Append data.
  std::memcpy(Val.ByteArrayVal + SizeFieldSize, Data, DataSize);
}

PropertyValue::PropertyValue(const PropertyValue &P) { *this = P; }

PropertyValue::PropertyValue(PropertyValue &&P) { *this = std::move(P); }

PropertyValue &PropertyValue::operator=(PropertyValue &&P) {
  copy(P);

  if (P.getType() == BYTE_ARRAY)
    P.Val.ByteArrayVal = nullptr;
  P.Ty = NONE;
  return *this;
}

PropertyValue &PropertyValue::operator=(const PropertyValue &P) {
  if (P.getType() == BYTE_ARRAY)
    *this = PropertyValue(P.asByteArray(), P.getByteArraySizeInBits());
  else
    copy(P);
  return *this;
}

void PropertyValue::copy(const PropertyValue &P) {
  Ty = P.Ty;
  Val = P.Val;
}

constexpr char PropertySetRegistry::SYCL_SPECIALIZATION_CONSTANTS[];
constexpr char PropertySetRegistry::SYCL_DEVICELIB_REQ_MASK[];
constexpr char PropertySetRegistry::SYCL_SPEC_CONSTANTS_DEFAULT_VALUES[];
constexpr char PropertySetRegistry::SYCL_KERNEL_PARAM_OPT_INFO[];
constexpr char PropertySetRegistry::SYCL_PROGRAM_METADATA[];
constexpr char PropertySetRegistry::SYCL_MISC_PROP[];
constexpr char PropertySetRegistry::SYCL_ASSERT_USED[];
constexpr char PropertySetRegistry::SYCL_EXPORTED_SYMBOLS[];
constexpr char PropertySetRegistry::SYCL_IMPORTED_SYMBOLS[];
constexpr char PropertySetRegistry::SYCL_DEVICE_GLOBALS[];
constexpr char PropertySetRegistry::SYCL_DEVICE_REQUIREMENTS[];
constexpr char PropertySetRegistry::SYCL_HOST_PIPES[];
constexpr char PropertySetRegistry::SYCL_VIRTUAL_FUNCTIONS[];

} // namespace util
} // namespace llvm
