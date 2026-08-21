//===------------------- JSON.cpp - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Typed JSON helpers on top of LLVM's JSON support.
// Provides convenient wrappers for reading and writing JSON files.
//
//===----------------------------------------------------------------------===//
#include "Utils/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

#include <chrono>

using namespace llvm;
using namespace llvm::advisor;

Expected<json::Value> llvm::advisor::parseJSONFile(StringRef Path) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer = MemoryBuffer::getFile(Path);
  if (!Buffer)
    return createStringError(Buffer.getError(), "cannot read '%s'",
                             Path.str().c_str());

  Expected<json::Value> Parsed = json::parse((*Buffer)->getBuffer());
  if (!Parsed)
    return Parsed.takeError();
  return std::move(*Parsed);
}

Error llvm::advisor::writeJSONFile(StringRef Path, const json::Value &Value) {
  std::error_code EC;
  ToolOutputFile Out(Path, EC, sys::fs::OF_Text);
  if (EC)
    return createStringError(EC, "cannot write '%s'", Path.str().c_str());
  Out.os() << Value << '\n';
  Out.keep();
  return Error::success();
}

Expected<std::string> llvm::advisor::getString(const json::Object &Object,
                                               StringRef Key) {
  std::optional<StringRef> Value = Object.getString(Key);
  if (!Value)
    return createStringError(inconvertibleErrorCode(), "missing string '%s'",
                             Key.str().c_str());
  return Value->str();
}

SmallVector<std::string, 8>
llvm::advisor::getStringArray(const json::Object &Object, StringRef Key) {
  SmallVector<std::string, 8> Out;
  const json::Array *Array = Object.getArray(Key);
  if (!Array)
    return Out;

  for (const json::Value &Value : *Array) {
    std::optional<StringRef> String = Value.getAsString();
    if (String)
      Out.push_back(String->str());
  }
  return Out;
}

std::string llvm::advisor::stringifyJSON(const json::Value &Value) {
  std::string Storage;
  raw_string_ostream OS(Storage);
  OS << Value;
  return OS.str();
}

static int64_t unixNow() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

static std::string uniqueRequestID() {
  static uint64_t Counter = 0;
  int64_t Now = unixNow();
  return ("req_" + Twine(Now) + "_" + Twine(Counter++)).str();
}

json::Value llvm::advisor::successEnvelope(json::Value Data) {
  return json::Object{{"request_id", uniqueRequestID()},
                      {"timestamp_unix", unixNow()},
                      {"status", "success"},
                      {"data", std::move(Data)}};
}

json::Value llvm::advisor::errorEnvelope(StringRef Code, StringRef Message) {
  return json::Object{{"request_id", uniqueRequestID()},
                      {"timestamp_unix", unixNow()},
                      {"status", "error"},
                      {"error", json::Object{{"code", Code}, {"message", Message}}}};
}
