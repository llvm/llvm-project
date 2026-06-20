//===------------------- BlobStore.cpp - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of BlobStore in Storage
//
//===----------------------------------------------------------------------===//

#include "Storage/BlobStore.h"
#include "Utils/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>

using namespace llvm;
using namespace llvm::advisor;

static constexpr size_t ChunkThreshold = 256ULL * 1024 * 1024;
static constexpr size_t ChunkSize = 64ULL * 1024 * 1024;

Expected<std::string> BlobStore::put(StringRef Data) {
  if (Data.size() <= ChunkThreshold) {
    Expected<cas::ObjectRef> Ref =
        CAS.store({}, ArrayRef<char>(Data.data(), Data.size()));
    if (!Ref)
      return Ref.takeError();
    return CAS.getID(*Ref).toString();
  }

  json::Array Chunks;
  for (size_t Offset = 0, Size = Data.size(); Offset < Size;
       Offset += ChunkSize) {
    Expected<std::string> ChunkID =
        put(Data.slice(Offset, std::min(Offset + ChunkSize, Size)));
    if (!ChunkID)
      return ChunkID.takeError();
    Chunks.push_back(*ChunkID);
  }

  return put(stringifyJSON(
      json::Object{{"llvm_advisor_blob_manifest", 1},
                   {"chunk_size", static_cast<int64_t>(ChunkSize)},
                   {"total_size", static_cast<int64_t>(Data.size())},
                   {"chunks", std::move(Chunks)}}));
}

Expected<std::string> BlobStore::putFile(StringRef Path) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer = MemoryBuffer::getFile(Path);
  if (!Buffer)
    return createStringError(Buffer.getError(), "cannot read blob file '%s'",
                             Path.str().c_str());
  return put((*Buffer)->getBuffer());
}

Expected<std::string> BlobStore::get(StringRef ID) {
  Expected<cas::CASID> Parsed = CAS.parseID(ID);
  if (!Parsed)
    return Parsed.takeError();
  Expected<cas::ObjectProxy> Proxy = CAS.getProxy(*Parsed);
  if (!Proxy)
    return Proxy.takeError();
  StringRef Data = Proxy->getData();

  Expected<json::Value> Value = json::parse(Data);
  if (!Value) {
    consumeError(Value.takeError());
    return Data.str();
  }
  const json::Object *Object = Value->getAsObject();
  if (!Object || !Object->getInteger("llvm_advisor_blob_manifest"))
    return Data.str();

  const json::Array *Chunks = Object->getArray("chunks");
  if (!Chunks)
    return createStringError(inconvertibleErrorCode(), "invalid blob manifest");

  std::string Out;
  if (std::optional<int64_t> Total = Object->getInteger("total_size"))
    Out.reserve(static_cast<size_t>(*Total));
  for (const json::Value &Chunk : *Chunks) {
    std::optional<StringRef> ChunkID = Chunk.getAsString();
    if (!ChunkID)
      return createStringError(inconvertibleErrorCode(),
                               "invalid blob manifest chunk");
    Expected<std::string> ChunkData = get(*ChunkID);
    if (!ChunkData)
      return ChunkData.takeError();
    Out += *ChunkData;
  }
  return Out;
}

Expected<bool> BlobStore::exists(StringRef ID) {
  Expected<cas::CASID> Parsed = CAS.parseID(ID);
  if (!Parsed)
    return Parsed.takeError();
  return CAS.getReference(*Parsed).has_value();
}
