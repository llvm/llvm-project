//===- JSONConfigUtils.h - Shared LTO JSON helpers --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_LTO_JSONCONFIGUTILS_H
#define LLVM_LIB_LTO_JSONCONFIGUTILS_H

#include "BitcodeMetadataUtils.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"

namespace llvm {
class TargetOptions;
}

namespace llvm::lto::bitcodemeta {

// Paths and profile buffers may contain bytes that are invalid UTF-8.
inline json::Value encodeString(StringRef S) {
  if (json::isUTF8(S))
    return S.str();
  return json::Object{{"base64", encodeBase64(S)}};
}

inline Expected<std::string> decodeString(const json::Value &V) {
  if (auto S = V.getAsString())
    return S->str();
  if (auto *Object = V.getAsObject())
    if (auto Base64 = Object->getString("base64")) {
      std::vector<char> Bytes;
      if (Error E = decodeBase64(*Base64, Bytes))
        return std::move(E);
      return std::string(Bytes.begin(), Bytes.end());
    }
  return metadataError("expected a string or base64-encoded bytes");
}

class JSONWriter {
  json::Object &Out;

public:
  explicit JSONWriter(json::Object &Out) : Out(Out) {}

  void putI32(StringRef Key, int32_t V) { Out[Key] = V; }
  void putU32(StringRef Key, uint32_t V) { Out[Key] = V; }
  void putBool(StringRef Key, bool V) { Out[Key] = V; }
  void putString(StringRef Key, StringRef V) { Out[Key] = encodeString(V); }
  void putStringList(StringRef Key, ArrayRef<std::string> Values) {
    json::Array List;
    for (StringRef S : Values)
      List.push_back(encodeString(S));
    Out[Key] = std::move(List);
  }
  void putValue(StringRef Key, json::Value V) { Out[Key] = std::move(V); }
};

struct JSONEntryApplier {
  const json::Value &Entry;
  StringRef Key;

  Error applyI32(function_ref<void(int32_t)> Setter) {
    auto V = Entry.getAsInteger();
    if (!V || !isInt<32>(*V))
      return metadataError(Key + ": expected a signed 32-bit integer");
    Setter(*V);
    return Error::success();
  }

  Error applyU32(function_ref<void(uint32_t)> Setter) {
    auto V = Entry.getAsUINT64();
    if (!V || !isUInt<32>(*V))
      return metadataError(Key + ": expected an unsigned 32-bit integer");
    Setter(*V);
    return Error::success();
  }

  Error applyBool(function_ref<void(bool)> Setter) {
    auto V = Entry.getAsBoolean();
    if (!V)
      return metadataError(Key + ": expected a boolean");
    Setter(*V);
    return Error::success();
  }

  Error applyString(function_ref<void(StringRef)> Setter) {
    auto V = decodeString(Entry);
    if (!V)
      return V.takeError();
    Setter(*V);
    return Error::success();
  }

  Error applyStringList(function_ref<void(std::vector<std::string>)> Setter) {
    auto *Array = Entry.getAsArray();
    if (!Array)
      return metadataError(Key + ": expected a string array");
    std::vector<std::string> Values;
    Values.reserve(Array->size());
    for (const auto &Element : *Array) {
      auto V = decodeString(Element);
      if (!V)
        return V.takeError();
      Values.push_back(std::move(*V));
    }
    Setter(std::move(Values));
    return Error::success();
  }
};

template <typename T, typename ApplyEntryFn>
Expected<T> decodeVersionedJSON(const json::Value &Value, unsigned Version,
                                StringRef Kind, ApplyEntryFn ApplyEntry) {
  auto *Object = Value.getAsObject();
  if (!Object)
    return metadataError(Kind + ": expected a JSON object");
  if (Object->getInteger("Version") != Version)
    return metadataError("unsupported " + Kind + " JSON version");
  T Result{};
  for (const auto &[Key, Entry] : *Object)
    if (Key != "Version")
      if (Error E = ApplyEntry(Result, Key, Entry))
        return std::move(E);
  return Result;
}

json::Object encodeTargetOptionsJSON(const TargetOptions &Options);
Expected<TargetOptions> decodeTargetOptionsJSON(const json::Value &Value);

} // namespace llvm::lto::bitcodemeta

#endif
