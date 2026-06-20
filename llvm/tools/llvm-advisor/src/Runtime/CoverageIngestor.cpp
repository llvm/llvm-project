//===------------------- CoverageIngestor.cpp - LLVM Advisor ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of CoverageIngestor in Runtime
//
//===----------------------------------------------------------------------===//

#include "Runtime/CoverageIngestor.h"
#include "Runtime/RuntimeUtils.h"
#include "llvm/Support/MemoryBuffer.h"

#include <optional>

using namespace llvm;
using namespace llvm::advisor;

Expected<json::Value> CoverageIngestor::load(StringRef Path) {
  if (Path.empty())
    return createStringError(inconvertibleErrorCode(), "empty coverage path");
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer = MemoryBuffer::getFile(Path);
  if (!Buffer)
    return createStringError(Buffer.getError(), "cannot read coverage '%s'",
                             Path.data());

  Expected<json::Value> Parsed = json::parse((*Buffer)->getBuffer());
  if (!Parsed)
    return Parsed.takeError();
  const json::Object *Root = Parsed->getAsObject();
  if (!Root)
    return createStringError(inconvertibleErrorCode(),
                             "coverage payload is not a JSON object");

  int64_t FileCount = 0;
  int64_t SegmentCount = 0;
  int64_t RegionCount = 0;
  json::Array Files;

  if (const json::Array *Data = Root->getArray("data")) {
    for (const json::Value &Export : *Data) {
      const json::Object *ExportObject = Export.getAsObject();
      if (!ExportObject)
        continue;
      const json::Array *ExportFiles = ExportObject->getArray("files");
      if (!ExportFiles)
        continue;
      for (const json::Value &FileValue : *ExportFiles) {
        const json::Object *File = FileValue.getAsObject();
        if (!File)
          continue;
        ++FileCount;
        const json::Array *Segments = File->getArray("segments");
        const json::Array *Regions = File->getArray("regions");
        SegmentCount += Segments ? Segments->size() : 0;
        RegionCount += Regions ? Regions->size() : 0;
        if (Files.size() < 256) {
          std::string Name;
          if (std::optional<StringRef> Filename = File->getString("filename"))
            Name = Filename->str();
          Files.push_back(json::Object{
              {"file", Name},
              {"segments",
               static_cast<int64_t>(Segments ? Segments->size() : 0)},
              {"regions",
               static_cast<int64_t>(Regions ? Regions->size() : 0)}});
        }
      }
    }
  }

  json::Object Summary;
  if (const json::Object *Totals = Root->getObject("totals"))
    if (const json::Object *Lines = Totals->getObject("lines"))
      Summary["covered_lines"] = getInteger(*Lines, "covered");

  Summary["kind"] = "coverage-mapping";
  Summary["format"] = "llvm-cov-export-json";
  Summary["path"] = Path;
  Summary["file_count"] = FileCount;
  Summary["segment_count"] = SegmentCount;
  Summary["region_count"] = RegionCount;
  Summary["files"] = std::move(Files);
  return Summary;
}
