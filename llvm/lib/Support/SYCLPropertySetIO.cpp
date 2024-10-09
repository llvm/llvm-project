//= SYCLPropertySetIO.cpp - models a sequence of property sets and their I/O =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SYCLPropertySetIO.h"

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

::llvm::Error makeError(const Twine &Msg) {
  return createStringError(std::error_code{}, Msg);
}

} // anonymous namespace

Expected<std::unique_ptr<SYCLPropertySetRegistry>>
SYCLPropertySetRegistry::read(const MemoryBuffer *Buf) {
  auto Res = std::make_unique<SYCLPropertySetRegistry>();
  SYCLPropertySet *CurPropSet = nullptr;

  for (line_iterator LI(*Buf); !LI.is_at_end(); LI++) {
    // See if this line starts a new property set
    if (LI->starts_with("[")) {
      // Parse the category (property name)
      auto EndPos = LI->rfind(']');
      if (EndPos == StringRef::npos)
        return makeError("invalid line: " + *LI);
      StringRef Category = LI->substr(1, EndPos - 1);
      CurPropSet = &(*Res)[Category];
      continue;
    }
    if (!CurPropSet)
      return makeError("property category missing");
    // Parse name and type+value
    auto Parts = LI->split('=');

    if (Parts.first.empty() || Parts.second.empty())
      return makeError("invalid property line: " + *LI);
    auto TypeVal = Parts.second.split('|');

    if (TypeVal.first.empty() || TypeVal.second.empty())
      return makeError("invalid property value: " + Parts.second);
    APInt Tint;

    // Parse type
    if (TypeVal.first.getAsInteger(10, Tint))
      return makeError("invalid property type: " + TypeVal.first);
    Expected<SYCLPropertyValue::Type> Ttag =
        SYCLPropertyValue::getTypeTag(static_cast<int>(Tint.getSExtValue()));
    StringRef Val = TypeVal.second;

    if (!Ttag)
      return Ttag.takeError();
    SYCLPropertyValue Prop(Ttag.get());

    // Parse value depending on its type
    switch (Ttag.get()) {
    case SYCLPropertyValue::Type::UINT32: {
      APInt ValV;
      if (Val.getAsInteger(10, ValV))
        return createStringError(std::error_code{},
                                 "invalid property value: ", Val.data());
      Prop.set(static_cast<uint32_t>(ValV.getZExtValue()));
      llvm::errs() << "ARV: read int done\n";
      break;
    }
    case SYCLPropertyValue::Type::BYTE_ARRAY: {
      std::vector<char> Output;
      if (Error Err = decodeBase64(Val, Output))
        return std::move(Err);
      Prop.set(reinterpret_cast<std::byte *>(Output.data()), Output.size());
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

  return Expected<std::unique_ptr<SYCLPropertySetRegistry>>(std::move(Res));
}

namespace llvm {
// Output a property to a stream
raw_ostream &operator<<(raw_ostream &Out, const SYCLPropertyValue &Prop) {
  Out << static_cast<int>(Prop.getType()) << '|';
  switch (Prop.getType()) {
  case SYCLPropertyValue::Type::UINT32:
    Out << Prop.asUint32();
    break;
  case SYCLPropertyValue::Type::BYTE_ARRAY: {
    auto PropArr = Prop.asByteArray();
    std::vector<std::byte> V(PropArr, PropArr + Prop.getByteArraySize() /
                                                    sizeof(std::byte));
    Out << encodeBase64(V);
    break;
  }
  default:
    llvm_unreachable(
        ("unsupported property type: " + utostr(Prop.getType())).c_str());
  }
  return Out;
}
} // namespace llvm

void SYCLPropertySetRegistry::write(raw_ostream &Out) const {
  for (const auto &PropSet : PropSetMap) {
    Out << '[' << PropSet.first << "]\n";

    for (const auto &Props : PropSet.second)
      Out << Props.first << '=' << Props.second << '\n';
  }
}

namespace llvm {
namespace util {

template <> SYCLPropertyValue::Type SYCLPropertyValue::getTypeTag<uint32_t>() {
  return UINT32;
}

template <>
SYCLPropertyValue::Type SYCLPropertyValue::getTypeTag<std::byte *>() {
  return BYTE_ARRAY;
}

SYCLPropertyValue::SYCLPropertyValue(const std::byte *Data, SizeTy DataBitSize)
    : Ty(BYTE_ARRAY) {
  constexpr int ByteSizeInBits = 8;
  SizeTy DataSize = (DataBitSize + (ByteSizeInBits - 1)) / ByteSizeInBits;
  constexpr size_t SizeFieldSize = sizeof(SizeTy);

  // Allocate space for size and data.
  Val = new std::byte[SizeFieldSize + DataSize];

  // Write the size into first bytes.
  for (size_t I = 0; I < SizeFieldSize; ++I) {
    auto ByteArrayVal = std::get<std::byte *>(Val);
    ByteArrayVal[I] = (std::byte)DataBitSize;
    DataBitSize >>= ByteSizeInBits;
  }
  // Append data.
  auto ByteArrayVal = std::get<std::byte *>(Val);
  std::memcpy(ByteArrayVal + SizeFieldSize, Data, DataSize);
}

SYCLPropertyValue::SYCLPropertyValue(const SYCLPropertyValue &P) { *this = P; }

SYCLPropertyValue::SYCLPropertyValue(SYCLPropertyValue &&P) {
  *this = std::move(P);
}

SYCLPropertyValue &SYCLPropertyValue::operator=(SYCLPropertyValue &&P) {
  copy(P);

  if (P.getType() == BYTE_ARRAY)
    P.Val = nullptr;
  P.Ty = NONE;
  return *this;
}

SYCLPropertyValue &SYCLPropertyValue::operator=(const SYCLPropertyValue &P) {
  if (P.getType() == BYTE_ARRAY)
    *this = SYCLPropertyValue(P.asByteArray(), P.getByteArraySizeInBits());
  else
    copy(P);
  return *this;
}

void SYCLPropertyValue::copy(const SYCLPropertyValue &P) {
  Ty = P.Ty;
  Val = P.Val;
}

constexpr char SYCLPropertySetRegistry::SYCL_SPECIALIZATION_CONSTANTS[];
constexpr char SYCLPropertySetRegistry::SYCL_DEVICELIB_REQ_MASK[];
constexpr char SYCLPropertySetRegistry::SYCL_SPEC_CONSTANTS_DEFAULT_VALUES[];
constexpr char SYCLPropertySetRegistry::SYCL_KERNEL_PARAM_OPT_INFO[];
constexpr char SYCLPropertySetRegistry::SYCL_PROGRAM_METADATA[];
constexpr char SYCLPropertySetRegistry::SYCL_MISC_PROP[];
constexpr char SYCLPropertySetRegistry::SYCL_ASSERT_USED[];
constexpr char SYCLPropertySetRegistry::SYCL_EXPORTED_SYMBOLS[];
constexpr char SYCLPropertySetRegistry::SYCL_IMPORTED_SYMBOLS[];
constexpr char SYCLPropertySetRegistry::SYCL_DEVICE_GLOBALS[];
constexpr char SYCLPropertySetRegistry::SYCL_DEVICE_REQUIREMENTS[];
constexpr char SYCLPropertySetRegistry::SYCL_HOST_PIPES[];
constexpr char SYCLPropertySetRegistry::SYCL_VIRTUAL_FUNCTIONS[];

} // namespace util
} // namespace llvm
