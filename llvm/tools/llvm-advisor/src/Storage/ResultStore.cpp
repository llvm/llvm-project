//===------------------- ResultStore.cpp - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of ResultStore in Storage
//
//===----------------------------------------------------------------------===//

#include "Storage/ResultStore.h"
#include "Utils/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

#include <optional>
#include <system_error>

using namespace llvm;
using namespace llvm::advisor;

static Error writeAnchor(StringRef Path, StringRef ID) {
  std::error_code EC;
  ToolOutputFile Out(Path, EC, sys::fs::OF_Text);
  if (EC)
    return createStringError(EC, "cannot write result anchor '%s'",
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
    return createStringError(Buffer.getError(),
                             "cannot read result anchor '%s'",
                             Path.str().c_str());
  }
  return (*Buffer)->getBuffer().trim().str();
}

Error ResultStore::load() {
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
                             "result root is not an object");

  if (const json::Object *SchemaMap = Root->getObject("schemas")) {
    for (const auto &Entry : *SchemaMap) {
      if (std::optional<StringRef> Version = Entry.second.getAsString())
        Schemas[Entry.first] = Version->str();
    }
  }
  if (const json::Object *ResultMap = Root->getObject("results")) {
    for (const auto &Entry : *ResultMap) {
      if (std::optional<StringRef> ID = Entry.second.getAsString())
        Results[Entry.first] = ID->str();
    }
  }
  return Error::success();
}

Error ResultStore::flush() {
  json::Object SchemaMap;
  json::Object ResultMap;
  for (const StringMapEntry<std::string> &Entry : Schemas)
    SchemaMap[Entry.first()] = Entry.second;
  for (const StringMapEntry<std::string> &Entry : Results)
    ResultMap[Entry.first()] = Entry.second;

  Expected<std::string> RootID = Blobs.put(stringifyJSON(json::Object{
      {"schemas", std::move(SchemaMap)}, {"results", std::move(ResultMap)}}));
  if (!RootID)
    return RootID.takeError();
  return writeAnchor(AnchorPath, *RootID);
}

Error ResultStore::registerSchema(StringRef CapabilityID, StringRef Version) {
  Schemas[CapabilityID] = Version.str();
  return flush();
}

Expected<std::string> ResultStore::put(StringRef RunKey,
                                       const json::Value &Result) {
  Expected<std::string> ID = Blobs.put(stringifyJSON(Result));
  if (!ID)
    return ID.takeError();
  Results[RunKey] = *ID;
  if (Error Err = flush())
    return std::move(Err);
  return *ID;
}

Expected<std::string> ResultStore::get(StringRef RunKey) const {
  StringMap<std::string>::const_iterator I = Results.find(RunKey);
  if (I == Results.end())
    return createStringError(inconvertibleErrorCode(), "unknown result: %s",
                             RunKey.str().c_str());
  return I->second;
}
