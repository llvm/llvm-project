//===------------------- SchemaManager.cpp - LLVM Advisor ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of SchemaManager in Storage
//
//===----------------------------------------------------------------------===//
#include "Storage/SchemaManager.h"
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
    return createStringError(EC, "cannot write schema anchor '%s'",
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
                             "cannot read schema anchor '%s'",
                             Path.str().c_str());
  }
  return (*Buffer)->getBuffer().trim().str();
}

Error SchemaManager::load() {
  Expected<std::string> RootID = readAnchor(AnchorPath);
  if (!RootID)
    return RootID.takeError();
  if (RootID->empty()) {
    StoredVersion = 0;
    return Error::success();
  }

  Expected<std::string> Data = Blobs.get(*RootID);
  if (!Data)
    return Data.takeError();
  Expected<json::Value> Value = json::parse(*Data);
  if (!Value)
    return Value.takeError();

  const json::Object *Root = Value->getAsObject();
  if (!Root)
    return createStringError(inconvertibleErrorCode(),
                             "schema root is not an object");

  if (std::optional<int64_t> V = Root->getInteger("version"))
    StoredVersion = static_cast<unsigned>(*V);
  else
    StoredVersion = 0;

  return Error::success();
}

Error SchemaManager::flush() {
  Expected<std::string> RootID = Blobs.put(stringifyJSON(
      json::Object{{"version", static_cast<int64_t>(StoredVersion)}}));
  if (!RootID)
    return RootID.takeError();
  return writeAnchor(AnchorPath, *RootID);
}

Error SchemaManager::migrate(unsigned From, unsigned To) {
  if (From == To)
    return Error::success();
  if (From > To)
    return createStringError(inconvertibleErrorCode(),
                             "schema downgrade from %u to %u is not supported",
                             From, To);

  for (unsigned V = From; V < To; ++V) {
    switch (V) {
    case 0:
      // 0 -> 1: initial schema creation
      break;
    default:
      return createStringError(inconvertibleErrorCode(),
                               "unknown schema migration step %u -> %u", V,
                               V + 1);
    }
  }

  StoredVersion = To;
  return flush();
}

Error SchemaManager::validateOnStartup() {
  if (Error Err = load())
    return Err;

  if (StoredVersion == 0) {
    StoredVersion = CurrentVersion;
    return flush();
  }

  if (StoredVersion < CurrentVersion)
    return migrate(StoredVersion, CurrentVersion);

  if (StoredVersion > CurrentVersion)
    return createStringError(inconvertibleErrorCode(),
                             "schema version %u is newer than supported %u",
                             StoredVersion, CurrentVersion);

  return Error::success();
}
