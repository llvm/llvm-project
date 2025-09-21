///===- llvm/Frontend/Offloading/PropertySet.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/Offloading/PropertySet.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBufferRef.h"

using namespace llvm;
using namespace llvm::offloading;

void llvm::offloading::writePropertiesToJSON(
    const PropertySetRegistry &PSRegistry, raw_ostream &Out) {
  json::OStream J(Out);
  J.object([&] {
    for (const auto &[CategoryName, PropSet] : PSRegistry) {
      auto PropSetCapture = PropSet;
      J.attributeObject(CategoryName, [&] {
        for (const auto &[PropName, PropVal] : PropSetCapture) {
          switch (PropVal.index()) {
          case 0:
            J.attribute(PropName, std::get<uint32_t>(PropVal));
            break;
          case 1:
            J.attribute(PropName, encodeBase64(std::get<ByteArray>(PropVal)));
            break;
          default:
            llvm_unreachable("unsupported property type");
          }
        }
      });
    }
  });
}

// note: createStringError has an overload that takes a format string,
// but it uses llvm::format instead of llvm::formatv, which does
// not work with json::Value. This is a helper function to use
// llvm::formatv with createStringError.
template <typename... Ts> auto createStringErrorV(Ts &&...Args) {
  return createStringError(formatv(std::forward<Ts>(Args)...));
}

Expected<PropertyValue>
readPropertyValueFromJSON(const json::Value &PropValueVal) {
  if (std::optional<uint64_t> Val = PropValueVal.getAsUINT64())
    return PropertyValue(static_cast<uint32_t>(*Val));

  if (std::optional<StringRef> Val = PropValueVal.getAsString()) {
    std::vector<char> Decoded;
    if (Error E = decodeBase64(*Val, Decoded))
      return createStringErrorV("unable to base64 decode the string {0}: {1}",
                                Val, toString(std::move(E)));
    return PropertyValue(ByteArray(Decoded.begin(), Decoded.end()));
  }

  return createStringErrorV("expected a uint64 or a string, got {0}",
                            PropValueVal);
}

Expected<PropertySetRegistry>
llvm::offloading::readPropertiesFromJSON(MemoryBufferRef Buf) {
  PropertySetRegistry Res;
  Expected<json::Value> V = json::parse(Buf.getBuffer());
  if (Error E = V.takeError())
    return E;

  const json::Object *O = V->getAsObject();
  if (!O)
    return createStringErrorV(
        "error while deserializing property set registry: "
        "expected JSON object, got {0}",
        *V);

  for (const auto &[CategoryName, Value] : *O) {
    const json::Object *PropSetVal = Value.getAsObject();
    if (!PropSetVal)
      return createStringErrorV("error while deserializing property set {0}: "
                                "expected JSON array, got {1}",
                                CategoryName.str(), Value);

    PropertySet &PropSet = Res[CategoryName.str()];
    for (const auto &[PropName, PropValueVal] : *PropSetVal) {
      Expected<PropertyValue> Prop = readPropertyValueFromJSON(PropValueVal);
      if (Error E = Prop.takeError())
        return createStringErrorV(
            "error while deserializing property {0} in property set {1}: {2}",
            PropName.str(), CategoryName.str(), toString(std::move(E)));

      auto [It, Inserted] =
          PropSet.try_emplace(PropName.str(), std::move(*Prop));
      assert(Inserted && "Property already exists in PropertySet");
      (void)Inserted;
    }
  }
  return Res;
}
