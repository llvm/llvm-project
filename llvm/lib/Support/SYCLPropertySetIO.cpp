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

static llvm::Error makeError(const Twine &Msg) {
  return createStringError(std::error_code{}, Msg);
}

Expected<std::unique_ptr<SYCLPropertySetRegistry>>
SYCLPropertySetRegistry::read(const MemoryBuffer *Buf) {
  auto Res = std::make_unique<SYCLPropertySetRegistry>();
  SYCLPropertySet *CurPropSet = nullptr;

  for (line_iterator LI(*Buf); !LI.is_at_end(); LI++) {
    // See if this line starts a new property set
    if (LI->starts_with("[")) {
      // Parse the category (property name)
      size_t EndPos = LI->rfind(']');
      if (EndPos == StringRef::npos)
        return makeError("invalid line: " + *LI);
      StringRef Category = LI->substr(1, EndPos - 1);
      CurPropSet = &(*Res)[Category];
      continue;
    }
    if (!CurPropSet)
      return makeError("property category missing");
    // Parse name and type+value
    auto [PropName, PropTypeAndValue] = LI->split('=');

    if (PropName.empty() || PropTypeAndValue.empty())
      return makeError("invalid property line: " + *LI);
    auto [PropType, PropVal] = PropTypeAndValue.split('|');

    if (PropType.empty() || PropVal.empty())
      return makeError("invalid property value: " + PropTypeAndValue);
    APInt Tint;

    // Parse type
    if (PropType.getAsInteger(10, Tint))
      return makeError("invalid property type: " + PropType);
    Expected<SYCLPropertyValue::Type> Ttag =
        SYCLPropertyValue::getTypeTag(static_cast<int>(Tint.getSExtValue()));
    StringRef Val = PropVal;

    if (!Ttag)
      return Ttag.takeError();
    SYCLPropertyValue Prop(Ttag.get());

    // Parse value depending on its type
    if (Prop.getType() == SYCLPropertyValue::Type::UInt32) {
      APInt ValV;
      if (Val.getAsInteger(10, ValV))
        return createStringError(std::error_code{},
                                 "invalid property value: ", Val.data());
      Prop.set(static_cast<uint32_t>(ValV.getZExtValue()));
    } else if (Prop.getType() == SYCLPropertyValue::Type::ByteArray) {
      std::vector<char> Output;
      // Output resized to maximum output size for base64 decoding
      Output.resize(((Val.size() + 3) / 4) * 3);
      if (Error Err = decodeBase64(Val, Output))
        return std::move(Err);
      Prop.set(reinterpret_cast<std::byte *>(Output.data()), Output.size());
    } else {
      return createStringError(std::error_code{},
                               "unsupported property type\n");
    }
    (*CurPropSet)[PropName] = std::move(Prop);
  }
  if (!CurPropSet)
    return makeError("invalid property set registry");

  return Expected<std::unique_ptr<SYCLPropertySetRegistry>>(std::move(Res));
}

namespace llvm {
// Output a property to a stream
raw_ostream &operator<<(raw_ostream &Out, const SYCLPropertyValue &Prop) {
  Out << static_cast<int>(Prop.getType()) << '|';
  if (Prop.getType() == SYCLPropertyValue::Type::UInt32) {
    Out << Prop.asUint32();
    return Out;
  }
  if (Prop.getType() == SYCLPropertyValue::Type::ByteArray) {
    const std::byte *PropArr = Prop.asByteArray();
    std::vector<std::byte> V(PropArr, PropArr + Prop.getByteArraySize() /
                                                    sizeof(std::byte));
    Out << encodeBase64(V);
    return Out;
  }
  llvm_unreachable("unsupported property type");
}
} // namespace llvm

void SYCLPropertySetRegistry::write(raw_ostream &Out) const {
  for (const auto &[PropCategory, Props] : PropSetMap) {
    Out << '[' << PropCategory << "]\n";

    for (const auto &[PropName, PropVal] : Props)
      Out << PropName << '=' << PropVal << '\n';
  }
}

namespace llvm {
namespace util {

SYCLPropertyValue::SYCLPropertyValue(const std::byte *Data,
                                     SizeTy DataBitSize) {
  SizeTy DataSize = (DataBitSize + (CHAR_BIT - 1)) / CHAR_BIT;
  constexpr size_t SizeFieldSize = sizeof(SizeTy);

  // Allocate space for size and data.
  Val = std::unique_ptr<std::byte, Deleter>(
      new std::byte[SizeFieldSize + DataSize], Deleter{});

  // Write the size into first bytes.
  if (auto ByteArrayVal =
          std::get_if<std::unique_ptr<std::byte, Deleter>>(&Val)) {
    for (size_t I = 0; I < SizeFieldSize; ++I) {
      (*ByteArrayVal).get()[I] = (std::byte)DataBitSize;
      DataBitSize >>= CHAR_BIT;
    }
    // Append data.
    std::memcpy((*ByteArrayVal).get() + SizeFieldSize, Data, DataSize);
  } else
    llvm_unreachable("must be a byte array value");
}

SYCLPropertyValue::SYCLPropertyValue(const SYCLPropertyValue &P) { *this = P; }

SYCLPropertyValue::SYCLPropertyValue(SYCLPropertyValue &&P) {
  *this = std::move(P);
}

SYCLPropertyValue &SYCLPropertyValue::operator=(SYCLPropertyValue &&P) {
  copy(P);

  if (std::holds_alternative<std::unique_ptr<std::byte, Deleter>>(Val))
    P.Val = nullptr;
  return *this;
}

SYCLPropertyValue &SYCLPropertyValue::operator=(const SYCLPropertyValue &P) {
  copy(P);
  return *this;
}

void SYCLPropertyValue::copy(const SYCLPropertyValue &P) {
  if (std::holds_alternative<std::unique_ptr<std::byte, Deleter>>(P.Val)) {
    // Allocate space for size and data.
    Val = std::unique_ptr<std::byte, Deleter>(
        new std::byte[P.getRawByteArraySize()], Deleter{});
    if (auto ByteArrayVal =
            std::get_if<std::unique_ptr<std::byte, Deleter>>(&Val))
      std::memcpy((*ByteArrayVal).get(), P.asRawByteArray(),
                  P.getRawByteArraySize());
  } else
    Val = P.asUint32();
}

} // namespace util
} // namespace llvm
