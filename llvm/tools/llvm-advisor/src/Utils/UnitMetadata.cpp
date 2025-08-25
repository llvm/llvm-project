//===-- UnitMetadata.cpp - Compilation Unit Metadata Management -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the UnitMetadata class for tracking compilation unit
// metadata including timestamps, file counts, and processing status.
//
//===----------------------------------------------------------------------===//

#include "UnitMetadata.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>

using namespace llvm;

namespace llvm {
namespace advisor {
namespace utils {

UnitMetadata::UnitMetadata(StringRef OutputDirectory)
    : OutputDirectory(OutputDirectory.str()) {
  MetadataFilePath = getMetadataPath();
}

Error UnitMetadata::loadMetadata() {
  if (!fileExists(MetadataFilePath)) {
    return Error::success(); // No existing metadata, start fresh
  }

  auto BufferOrError = MemoryBuffer::getFile(MetadataFilePath);
  if (!BufferOrError) {
    return createStringError(BufferOrError.getError(),
                             "Failed to open metadata file: " +
                                 MetadataFilePath);
  }

  return fromJson(BufferOrError->get()->getBuffer());
}

Error UnitMetadata::saveMetadata() {
  // Ensure output directory exists
  if (auto Err = createDirectoryIfNeeded(OutputDirectory)) {
    return Err;
  }

  auto JsonStr = toJson();
  if (!JsonStr) {
    return JsonStr.takeError();
  }

  std::error_code EC;
  raw_fd_ostream File(MetadataFilePath, EC);
  if (EC) {
    return createStringError(EC, "Failed to create metadata file: " +
                                     MetadataFilePath);
  }

  File << *JsonStr;
  return Error::success();
}

void UnitMetadata::clear() { Units.clear(); }

void UnitMetadata::registerUnit(StringRef UnitName) {
  CompilationUnitInfo Info;
  Info.Name = UnitName.str();
  Info.Timestamp = std::chrono::system_clock::now();
  Info.TotalFiles = 0;
  Info.Status = "in_progress";
  Info.OutputPath = OutputDirectory + "/" + UnitName.str();

  Units[UnitName.str()] = Info;
}

void UnitMetadata::updateUnitStatus(StringRef UnitName, StringRef Status) {
  auto it = Units.find(UnitName.str());
  if (it != Units.end()) {
    it->second.Status = Status.str();
    // Update timestamp when status changes
    it->second.Timestamp = std::chrono::system_clock::now();
  }
}

void UnitMetadata::updateUnitFileCount(StringRef UnitName, size_t FileCount) {
  auto it = Units.find(UnitName.str());
  if (it != Units.end()) {
    it->second.TotalFiles = FileCount;
  }
}

void UnitMetadata::addArtifactType(StringRef UnitName, StringRef Type) {
  auto it = Units.find(UnitName.str());
  if (it != Units.end()) {
    auto &Types = it->second.ArtifactTypes;
    if (std::find(Types.begin(), Types.end(), Type.str()) == Types.end()) {
      Types.push_back(Type.str());
    }
  }
}

void UnitMetadata::setUnitProperty(StringRef UnitName, StringRef Key,
                                   StringRef Value) {
  auto it = Units.find(UnitName.str());
  if (it != Units.end()) {
    it->second.Properties[Key.str()] = Value.str();
  }
}

bool UnitMetadata::hasUnit(StringRef UnitName) const {
  return Units.find(UnitName.str()) != Units.end();
}

Expected<CompilationUnitInfo>
UnitMetadata::getUnitInfo(StringRef UnitName) const {
  auto it = Units.find(UnitName.str());
  if (it != Units.end()) {
    return it->second;
  }
  return createStringError(
      std::make_error_code(std::errc::no_such_file_or_directory),
      "Unit not found: " + UnitName);
}

std::vector<CompilationUnitInfo> UnitMetadata::getAllUnits() const {
  std::vector<CompilationUnitInfo> Result;
  for (const auto &Pair : Units) {
    Result.push_back(Pair.second);
  }

  // Sort by timestamp (most recent first)
  std::sort(Result.begin(), Result.end(),
            [](const CompilationUnitInfo &A, const CompilationUnitInfo &B) {
              return A.Timestamp > B.Timestamp;
            });

  return Result;
}

std::vector<std::string> UnitMetadata::getRecentUnits(size_t Count) const {
  auto AllUnits = getAllUnits();
  std::vector<std::string> Result;

  size_t MaxCount = std::min(Count, AllUnits.size());
  for (size_t i = 0; i < MaxCount; ++i) {
    Result.push_back(AllUnits[i].Name);
  }

  return Result;
}

Expected<std::string> UnitMetadata::getMostRecentUnit() const {
  auto RecentUnits = getRecentUnits(1);
  if (RecentUnits.empty()) {
    return createStringError(
        std::make_error_code(std::errc::no_such_file_or_directory),
        "No units found");
  }
  return RecentUnits[0];
}

void UnitMetadata::removeUnit(StringRef UnitName) {
  Units.erase(UnitName.str());
}

void UnitMetadata::cleanupOldUnits(int MaxAgeInDays) {
  auto Now = std::chrono::system_clock::now();
  auto MaxAge = std::chrono::hours(24 * MaxAgeInDays);

  std::vector<std::string> UnitsToRemove;
  for (const auto &Pair : Units) {
    if (Now - Pair.second.Timestamp > MaxAge) {
      UnitsToRemove.push_back(Pair.first);
    }
  }

  for (const auto &UnitName : UnitsToRemove) {
    removeUnit(UnitName);
  }
}

size_t UnitMetadata::getUnitCount() const { return Units.size(); }

Expected<std::string> UnitMetadata::toJson() const {
  json::Object Root;
  json::Array UnitsArray;

  for (const auto &Pair : Units) {
    const auto &Info = Pair.second;
    json::Object UnitObj;

    UnitObj["name"] = Info.Name;
    UnitObj["timestamp"] = formatTimestamp(Info.Timestamp);
    UnitObj["total_files"] = static_cast<int64_t>(Info.TotalFiles);
    UnitObj["status"] = Info.Status;
    UnitObj["output_path"] = Info.OutputPath;

    // Artifact types
    json::Array ArtifactTypesArray;
    for (const auto &Type : Info.ArtifactTypes) {
      ArtifactTypesArray.push_back(Type);
    }
    UnitObj["artifact_types"] = std::move(ArtifactTypesArray);

    // Properties
    json::Object PropertiesObj;
    for (const auto &Prop : Info.Properties) {
      PropertiesObj[Prop.first] = Prop.second;
    }
    UnitObj["properties"] = std::move(PropertiesObj);

    UnitsArray.push_back(std::move(UnitObj));
  }

  Root["units"] = std::move(UnitsArray);

  std::string Result;
  raw_string_ostream OS(Result);
  OS << json::Value(std::move(Root));
  return Result;
}

Error UnitMetadata::fromJson(StringRef JsonStr) {
  Units.clear();

  Expected<json::Value> JsonOrError = json::parse(JsonStr);
  if (!JsonOrError) {
    return JsonOrError.takeError();
  }

  auto *Root = JsonOrError->getAsObject();
  if (!Root) {
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "Invalid JSON: root must be an object");
  }

  auto *UnitsArray = Root->getArray("units");
  if (!UnitsArray) {
    return Error::success(); // No units section found, return success with
                             // empty units
  }

  for (const auto &UnitValue : *UnitsArray) {
    auto *UnitObj = UnitValue.getAsObject();
    if (!UnitObj) {
      continue; // Skip invalid unit objects
    }

    CompilationUnitInfo Info;

    // Parse name
    if (auto NameOpt = UnitObj->getString("name")) {
      Info.Name = NameOpt->str();
    } else {
      continue; // Skip units without name
    }

    // Parse timestamp
    if (auto TimestampOpt = UnitObj->getString("timestamp")) {
      auto TimestampOrError = parseTimestamp(*TimestampOpt);
      if (TimestampOrError) {
        Info.Timestamp = *TimestampOrError;
      } else {
        // Use current time if parsing fails
        Info.Timestamp = std::chrono::system_clock::now();
      }
    } else {
      Info.Timestamp = std::chrono::system_clock::now();
    }

    // Parse total_files
    if (auto FilesOpt = UnitObj->getInteger("total_files")) {
      Info.TotalFiles = static_cast<size_t>(*FilesOpt);
    } else {
      Info.TotalFiles = 0;
    }

    // Parse status
    if (auto StatusOpt = UnitObj->getString("status")) {
      Info.Status = StatusOpt->str();
    } else {
      Info.Status = "unknown";
    }

    // Parse output_path
    if (auto PathOpt = UnitObj->getString("output_path")) {
      Info.OutputPath = PathOpt->str();
    } else {
      Info.OutputPath = OutputDirectory + "/" + Info.Name;
    }

    // Parse artifact_types
    if (auto TypesArray = UnitObj->getArray("artifact_types")) {
      for (const auto &TypeValue : *TypesArray) {
        if (auto TypeStr = TypeValue.getAsString()) {
          Info.ArtifactTypes.push_back(TypeStr->str());
        }
      }
    }

    // Parse properties
    if (auto PropertiesObj = UnitObj->getObject("properties")) {
      for (const auto &Prop : *PropertiesObj) {
        if (auto ValueStr = Prop.second.getAsString()) {
          Info.Properties[Prop.first.str()] = ValueStr->str();
        }
      }
    }

    Units[Info.Name] = Info;
  }

  return Error::success();
}

std::string UnitMetadata::getMetadataPath() const {
  return OutputDirectory + "/.llvm-advisor-metadata.json";
}

std::string UnitMetadata::formatTimestamp(
    const std::chrono::system_clock::time_point &TimePoint) const {
  auto TimeT = std::chrono::system_clock::to_time_t(TimePoint);
  std::stringstream ss;
  ss << std::put_time(std::gmtime(&TimeT), "%Y-%m-%dT%H:%M:%SZ");
  return ss.str();
}

Expected<std::chrono::system_clock::time_point>
UnitMetadata::parseTimestamp(StringRef TimestampStr) const {
  std::tm tm = {};
  std::istringstream ss(TimestampStr.str());
  ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%SZ");

  if (ss.fail()) {
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "Invalid timestamp format: " + TimestampStr);
  }

  return std::chrono::system_clock::from_time_t(std::mktime(&tm));
}

bool UnitMetadata::fileExists(StringRef Path) const {
  return sys::fs::exists(Path);
}

Error UnitMetadata::createDirectoryIfNeeded(StringRef Path) const {
  if (!sys::fs::exists(Path)) {
    std::error_code EC = sys::fs::create_directories(Path);
    if (EC) {
      return createStringError(EC, "Error creating directory: " + Path);
    }
  }
  return Error::success();
}

} // namespace utils
} // namespace advisor
} // namespace llvm
