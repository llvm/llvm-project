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
    : outputDirectory(OutputDirectory.str()),
      metadataFilePath(getMetadataPath()) {}

Error UnitMetadata::loadMetadata() {
  if (!fileExists(metadataFilePath))
    return Error::success();

  auto BufferOrError = MemoryBuffer::getFile(metadataFilePath);
  if (!BufferOrError)
    return createStringError(BufferOrError.getError(),
                             "Failed to open metadata file: " +
                                 metadataFilePath);

  return fromJson(BufferOrError->get()->getBuffer());
}

Error UnitMetadata::saveMetadata() {
  if (auto Err = createDirectoryIfNeeded(outputDirectory))
    return Err;

  auto JsonStr = toJson();
  if (!JsonStr)
    return JsonStr.takeError();

  std::error_code Ec;
  raw_fd_ostream File(metadataFilePath, Ec);
  if (Ec)
    return createStringError(Ec, "Failed to create metadata file: " +
                                     metadataFilePath);

  File << *JsonStr;
  return Error::success();
}

void UnitMetadata::clear() { units.clear(); }

void UnitMetadata::registerUnit(StringRef UnitName) {
  CompilationUnitInfo Info;
  Info.name = UnitName.str();
  Info.timestamp = std::chrono::system_clock::now();
  Info.totalFiles = 0;
  Info.status = "in_progress";
  Info.outputPath = outputDirectory + "/" + UnitName.str();

  units[UnitName.str()] = std::move(Info);
}

void UnitMetadata::updateUnitStatus(StringRef UnitName, StringRef Status) {
  auto It = units.find(UnitName.str());
  if (It == units.end())
    return;

  It->second.status = Status.str();
  It->second.timestamp = std::chrono::system_clock::now();
}

void UnitMetadata::updateUnitFileCount(StringRef UnitName, size_t FileCount) {
  auto It = units.find(UnitName.str());
  if (It == units.end())
    return;
  It->second.totalFiles = FileCount;
}

void UnitMetadata::addArtifactType(StringRef UnitName, StringRef Type) {
  auto It = units.find(UnitName.str());
  if (It == units.end())
    return;

  auto &Types = It->second.artifactTypes;
  if (std::find(Types.begin(), Types.end(), Type.str()) == Types.end())
    Types.push_back(Type.str());
}

void UnitMetadata::setUnitProperty(StringRef UnitName, StringRef Key,
                                   StringRef Value) {
  auto It = units.find(UnitName.str());
  if (It == units.end())
    return;
  It->second.properties[Key.str()] = Value.str();
}

bool UnitMetadata::hasUnit(StringRef UnitName) const {
  return units.find(UnitName.str()) != units.end();
}

Expected<CompilationUnitInfo>
UnitMetadata::getUnitInfo(StringRef UnitName) const {
  auto It = units.find(UnitName.str());
  if (It != units.end())
    return It->second;

  return createStringError(
      std::make_error_code(std::errc::no_such_file_or_directory),
      "Unit not found: " + UnitName);
}

std::vector<CompilationUnitInfo> UnitMetadata::getAllUnits() const {
  std::vector<CompilationUnitInfo> Result;
  Result.reserve(units.size());
  for (const auto &Pair : units)
    Result.push_back(Pair.second);

  std::sort(Result.begin(), Result.end(),
            [](const CompilationUnitInfo &A, const CompilationUnitInfo &B) {
              return A.timestamp > B.timestamp;
            });
  return Result;
}

std::vector<std::string> UnitMetadata::getRecentUnits(size_t Count) const {
  auto AllUnits = getAllUnits();
  std::vector<std::string> Result;

  size_t MaxCount = std::min(Count, AllUnits.size());
  Result.reserve(MaxCount);
  for (size_t I = 0; I < MaxCount; ++I)
    Result.push_back(AllUnits[I].name);

  return Result;
}

Expected<std::string> UnitMetadata::getMostRecentUnit() const {
  auto RecentUnits = getRecentUnits(1);
  if (RecentUnits.empty())
    return createStringError(
        std::make_error_code(std::errc::no_such_file_or_directory),
        "No units found");
  return RecentUnits.front();
}

void UnitMetadata::removeUnit(StringRef UnitName) {
  units.erase(UnitName.str());
}

void UnitMetadata::cleanupOldUnits(int MaxAgeInDays) {
  auto Now = std::chrono::system_clock::now();
  auto MaxAge = std::chrono::hours(24 * MaxAgeInDays);

  std::vector<std::string> UnitsToRemove;
  for (const auto &Pair : units)
    if (Now - Pair.second.timestamp > MaxAge)
      UnitsToRemove.push_back(Pair.first);

  for (const auto &UnitName : UnitsToRemove)
    removeUnit(UnitName);
}

size_t UnitMetadata::getUnitCount() const { return units.size(); }

Expected<std::string> UnitMetadata::toJson() const {
  json::Object Root;
  json::Array UnitsArray;

  for (const auto &Pair : units) {
    const auto &Info = Pair.second;
    json::Object UnitObj;

    UnitObj["name"] = Info.name;
    UnitObj["timestamp"] = formatTimestamp(Info.timestamp);
    UnitObj["total_files"] = static_cast<int64_t>(Info.totalFiles);
    UnitObj["status"] = Info.status;
    UnitObj["output_path"] = Info.outputPath;

    json::Array ArtifactTypesArray;
    for (const auto &Type : Info.artifactTypes)
      ArtifactTypesArray.push_back(Type);
    UnitObj["artifact_types"] = std::move(ArtifactTypesArray);

    json::Object PropertiesObj;
    for (const auto &Prop : Info.properties)
      PropertiesObj[Prop.first] = Prop.second;
    UnitObj["properties"] = std::move(PropertiesObj);

    UnitsArray.push_back(std::move(UnitObj));
  }

  Root["units"] = std::move(UnitsArray);

  std::string Result;
  raw_string_ostream Os(Result);
  Os << json::Value(std::move(Root));
  return Result;
}

Error UnitMetadata::fromJson(StringRef JsonStr) {
  units.clear();

  auto JsonOrError = json::parse(JsonStr);
  if (!JsonOrError)
    return JsonOrError.takeError();

  auto *Root = JsonOrError->getAsObject();
  if (!Root)
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "Config file must contain JSON object");

  auto *UnitsArray = Root->getArray("units");
  if (!UnitsArray)
    return Error::success();

  for (const auto &UnitValue : *UnitsArray) {
    auto *UnitObj = UnitValue.getAsObject();
    if (!UnitObj)
      continue;

    CompilationUnitInfo Info;

    if (auto NameOpt = UnitObj->getString("name")) {
      Info.name = NameOpt->str();
    } else {
      continue;
    }

    if (auto TimestampOpt = UnitObj->getString("timestamp")) {
      auto TimestampOrError = parseTimestamp(*TimestampOpt);
      Info.timestamp = TimestampOrError ? *TimestampOrError
                                        : std::chrono::system_clock::now();
    } else {
      Info.timestamp = std::chrono::system_clock::now();
    }

    if (auto FilesOpt = UnitObj->getInteger("total_files")) {
      Info.totalFiles = static_cast<size_t>(*FilesOpt);
    } else {
      Info.totalFiles = 0;
    }

    if (auto StatusOpt = UnitObj->getString("status")) {
      Info.status = StatusOpt->str();
    } else {
      Info.status = "unknown";
    }

    if (auto PathOpt = UnitObj->getString("output_path")) {
      Info.outputPath = PathOpt->str();
    } else {
      Info.outputPath = outputDirectory + "/" + Info.name;
    }

    if (const auto *TypesArray = UnitObj->getArray("artifact_types")) {
      for (const auto &TypeValue : *TypesArray)
        if (auto TypeStr = TypeValue.getAsString())
          Info.artifactTypes.push_back(TypeStr->str());
    }

    if (const auto *PropertiesObj = UnitObj->getObject("properties")) {
      for (const auto &Prop : *PropertiesObj)
        if (auto ValueStr = Prop.second.getAsString())
          Info.properties[Prop.first.str()] = ValueStr->str();
    }

    units[Info.name] = std::move(Info);
  }

  return Error::success();
}

std::string UnitMetadata::getMetadataPath() const {
  return outputDirectory + "/.llvm-advisor-metadata.json";
}

std::string UnitMetadata::formatTimestamp(
    const std::chrono::system_clock::time_point &TimePoint) const {
  auto TimeT = std::chrono::system_clock::to_time_t(TimePoint);
  std::stringstream Ss;
  Ss << std::put_time(std::gmtime(&TimeT), "%Y-%m-%dT%H:%M:%SZ");
  return Ss.str();
}

Expected<std::chrono::system_clock::time_point>
UnitMetadata::parseTimestamp(StringRef TimestampStr) const {
  std::tm Tm = {};
  std::istringstream Ss(TimestampStr.str());
  Ss >> std::get_time(&Tm, "%Y-%m-%dT%H:%M:%SZ");

  if (Ss.fail())
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "Invalid timestamp format: " + TimestampStr);

  return std::chrono::system_clock::from_time_t(std::mktime(&Tm));
}

bool UnitMetadata::fileExists(StringRef Path) const {
  return sys::fs::exists(Path);
}

Error UnitMetadata::createDirectoryIfNeeded(StringRef Path) const {
  if (!sys::fs::exists(Path)) {
    std::error_code Ec = sys::fs::create_directories(Path);
    if (Ec)
      return createStringError(Ec, "Error creating directory: " + Path);
  }
  return Error::success();
}

} // namespace utils
} // namespace advisor
} // namespace llvm
