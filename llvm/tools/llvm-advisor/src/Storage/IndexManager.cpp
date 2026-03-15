//===------------------- IndexManager.cpp - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of IndexManager in Storage
//
//===----------------------------------------------------------------------===//

#include "Storage/IndexManager.h"
#include "Utils/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

#include <system_error>

using namespace llvm;
using namespace llvm::advisor;

static Error writeAnchor(StringRef Path, StringRef ID) {
  std::error_code EC;
  ToolOutputFile Out(Path, EC, sys::fs::OF_Text);
  if (EC)
    return createStringError(EC, "cannot write index anchor '%s'",
                             Path.str().c_str());
  Out.os() << ID << '\n';
  Out.keep();
  return Error::success();
}

static Expected<std::string> readAnchor(StringRef Path) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer = MemoryBuffer::getFile(Path);
  if (!Buffer) {
    if (Buffer.getError() == std::errc::no_such_file_or_directory)
      return std::string();
    return createStringError(Buffer.getError(), "cannot read index anchor '%s'",
                             Path.str().c_str());
  }
  return (*Buffer)->getBuffer().trim().str();
}

Error IndexManager::load() {
  Expected<std::string> RootID = readAnchor(AnchorPath);
  if (!RootID)
    return RootID.takeError();
  if (RootID->empty())
    return Error::success();

  Expected<std::string> Data = Blobs.get(*RootID);
  if (!Data)
    return Data.takeError();
  Expected<json::Value> Value = json::parse(*Data);
  if (!Value)
    return Value.takeError();

  const json::Object *Root = Value->getAsObject();
  if (!Root)
    return createStringError(inconvertibleErrorCode(),
                             "index root is not an object");

  for (const auto &IndexEntry : *Root) {
    const json::Object *KeyMap = IndexEntry.second.getAsObject();
    if (!KeyMap)
      continue;
    for (const auto &KeyEntry : *KeyMap) {
      const json::Array *Values = KeyEntry.second.getAsArray();
      if (!Values)
        continue;
      for (const json::Value &V : *Values) {
        if (std::optional<StringRef> S = V.getAsString())
          Indexes[IndexEntry.first][KeyEntry.first].push_back(S->str());
      }
    }
  }
  return Error::success();
}

Error IndexManager::flush() {
  json::Object Root;
  for (const auto &IndexEntry : Indexes) {
    json::Object KeyMap;
    for (const auto &KeyEntry : IndexEntry.second) {
      json::Array Values;
      for (const std::string &V : KeyEntry.second)
        Values.push_back(V);
      KeyMap[KeyEntry.first()] = std::move(Values);
    }
    Root[IndexEntry.first()] = std::move(KeyMap);
  }

  Expected<std::string> RootID =
      Blobs.put(stringifyJSON(json::Value(std::move(Root))));
  if (!RootID)
    return RootID.takeError();
  return writeAnchor(AnchorPath, *RootID);
}

Error IndexManager::add(StringRef Index, StringRef Key, StringRef Value) {
  Indexes[Index][Key].push_back(Value.str());
  return flush();
}

SmallVector<std::string, 16> IndexManager::lookup(StringRef Index,
                                                  StringRef Key) const {
  SmallVector<std::string, 16> Out;
  StringMap<StringMap<SmallVector<std::string, 4>>>::const_iterator I =
      Indexes.find(Index);
  if (I == Indexes.end())
    return Out;
  StringMap<SmallVector<std::string, 4>>::const_iterator J =
      I->second.find(Key);
  if (J == I->second.end())
    return Out;
  for (const std::string &Value : J->second)
    Out.push_back(Value);
  return Out;
}

void IndexManager::clear(StringRef Index) {
  Indexes.erase(Index);
  consumeError(flush());
}
