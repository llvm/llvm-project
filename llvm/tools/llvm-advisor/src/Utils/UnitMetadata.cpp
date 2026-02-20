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

UnitMetadata::UnitMetadata(StringRef outputDirectory)
    : outputDirectory(outputDirectory.str()),
      metadataFilePath(getMetadataPath()) {}

Error UnitMetadata::loadMetadata() {
  if (!fileExists(metadataFilePath))
    return Error::success();

  auto bufferOrError = MemoryBuffer::getFile(metadataFilePath);
  if (!bufferOrError)
    return createStringError(bufferOrError.getError(),
                             "Failed to open metadata file: " +
                                 metadataFilePath);

  return fromJson(bufferOrError->get()->getBuffer());
}

Error UnitMetadata::saveMetadata() {
  if (auto err = createDirectoryIfNeeded(outputDirectory))
    return err;

  auto jsonStr = toJson();
  if (!jsonStr)
    return jsonStr.takeError();

  std::error_code ec;
  raw_fd_ostream file(metadataFilePath, ec);
  if (ec)
    return createStringError(ec, "Failed to create metadata file: " +
                                     metadataFilePath);

  file << *jsonStr;
  return Error::success();
}

void UnitMetadata::clear() { units.clear(); }

void UnitMetadata::registerUnit(StringRef unitName) {
  CompilationUnitInfo info;
  info.name = unitName.str();
  info.timestamp = std::chrono::system_clock::now();
  info.totalFiles = 0;
  info.status = "in_progress";
  info.outputPath = outputDirectory + "/" + unitName.str();

  units[unitName.str()] = std::move(info);
}

void UnitMetadata::updateUnitStatus(StringRef unitName, StringRef status) {
  auto it = units.find(unitName.str());
  if (it == units.end())
    return;

  it->second.status = status.str();
  it->second.timestamp = std::chrono::system_clock::now();
}

void UnitMetadata::updateUnitFileCount(StringRef unitName, size_t fileCount) {
  auto it = units.find(unitName.str());
  if (it == units.end())
    return;
  it->second.totalFiles = fileCount;
}

void UnitMetadata::addArtifactType(StringRef unitName, StringRef type) {
  auto it = units.find(unitName.str());
  if (it == units.end())
    return;

  auto &types = it->second.artifactTypes;
  if (std::find(types.begin(), types.end(), type.str()) == types.end())
    types.push_back(type.str());
}

void UnitMetadata::setUnitProperty(StringRef unitName, StringRef key,
                                   StringRef value) {
  auto it = units.find(unitName.str());
  if (it == units.end())
    return;
  it->second.properties[key.str()] = value.str();
}

bool UnitMetadata::hasUnit(StringRef unitName) const {
  return units.find(unitName.str()) != units.end();
}

Expected<CompilationUnitInfo> UnitMetadata::getUnitInfo(StringRef unitName) const {
  auto it = units.find(unitName.str());
  if (it != units.end())
    return it->second;

  return createStringError(
      std::make_error_code(std::errc::no_such_file_or_directory),
      "Unit not found: " + unitName);
}

std::vector<CompilationUnitInfo> UnitMetadata::getAllUnits() const {
  std::vector<CompilationUnitInfo> result;
  result.reserve(units.size());
  for (const auto &pair : units)
    result.push_back(pair.second);

  std::sort(result.begin(), result.end(),
            [](const CompilationUnitInfo &a, const CompilationUnitInfo &b) {
              return a.timestamp > b.timestamp;
            });
  return result;
}

std::vector<std::string> UnitMetadata::getRecentUnits(size_t count) const {
  auto allUnits = getAllUnits();
  std::vector<std::string> result;

  size_t maxCount = std::min(count, allUnits.size());
  result.reserve(maxCount);
  for (size_t i = 0; i < maxCount; ++i)
    result.push_back(allUnits[i].name);

  return result;
}

Expected<std::string> UnitMetadata::getMostRecentUnit() const {
  auto recentUnits = getRecentUnits(1);
  if (recentUnits.empty())
    return createStringError(
        std::make_error_code(std::errc::no_such_file_or_directory),
        "No units found");
  return recentUnits.front();
}

void UnitMetadata::removeUnit(StringRef unitName) { units.erase(unitName.str()); }

void UnitMetadata::cleanupOldUnits(int maxAgeInDays) {
  auto now = std::chrono::system_clock::now();
  auto maxAge = std::chrono::hours(24 * maxAgeInDays);

  std::vector<std::string> unitsToRemove;
  for (const auto &pair : units)
    if (now - pair.second.timestamp > maxAge)
      unitsToRemove.push_back(pair.first);

  for (const auto &unitName : unitsToRemove)
    removeUnit(unitName);
}

size_t UnitMetadata::getUnitCount() const { return units.size(); }

Expected<std::string> UnitMetadata::toJson() const {
  json::Object root;
  json::Array unitsArray;

  for (const auto &pair : units) {
    const auto &info = pair.second;
    json::Object unitObj;

    unitObj["name"] = info.name;
    unitObj["timestamp"] = formatTimestamp(info.timestamp);
    unitObj["total_files"] = static_cast<int64_t>(info.totalFiles);
    unitObj["status"] = info.status;
    unitObj["output_path"] = info.outputPath;

    json::Array artifactTypesArray;
    for (const auto &type : info.artifactTypes)
      artifactTypesArray.push_back(type);
    unitObj["artifact_types"] = std::move(artifactTypesArray);

    json::Object propertiesObj;
    for (const auto &prop : info.properties)
      propertiesObj[prop.first] = prop.second;
    unitObj["properties"] = std::move(propertiesObj);

    unitsArray.push_back(std::move(unitObj));
  }

  root["units"] = std::move(unitsArray);

  std::string result;
  raw_string_ostream os(result);
  os << json::Value(std::move(root));
  return result;
}

Error UnitMetadata::fromJson(StringRef jsonStr) {
  units.clear();

  auto jsonOrError = json::parse(jsonStr);
  if (!jsonOrError)
    return jsonOrError.takeError();

  auto *root = jsonOrError->getAsObject();
  if (!root)
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "Config file must contain JSON object");

  auto *unitsArray = root->getArray("units");
  if (!unitsArray)
    return Error::success();

  for (const auto &unitValue : *unitsArray) {
    auto *unitObj = unitValue.getAsObject();
    if (!unitObj)
      continue;

    CompilationUnitInfo info;

    if (auto nameOpt = unitObj->getString("name")) {
      info.name = nameOpt->str();
    } else {
      continue;
    }

    if (auto timestampOpt = unitObj->getString("timestamp")) {
      auto timestampOrError = parseTimestamp(*timestampOpt);
      info.timestamp =
          timestampOrError ? *timestampOrError : std::chrono::system_clock::now();
    } else {
      info.timestamp = std::chrono::system_clock::now();
    }

    if (auto filesOpt = unitObj->getInteger("total_files")) {
      info.totalFiles = static_cast<size_t>(*filesOpt);
    } else {
      info.totalFiles = 0;
    }

    if (auto statusOpt = unitObj->getString("status")) {
      info.status = statusOpt->str();
    } else {
      info.status = "unknown";
    }

    if (auto pathOpt = unitObj->getString("output_path")) {
      info.outputPath = pathOpt->str();
    } else {
      info.outputPath = outputDirectory + "/" + info.name;
    }

    if (auto typesArray = unitObj->getArray("artifact_types")) {
      for (const auto &typeValue : *typesArray)
        if (auto typeStr = typeValue.getAsString())
          info.artifactTypes.push_back(typeStr->str());
    }

    if (auto propertiesObj = unitObj->getObject("properties")) {
      for (const auto &prop : *propertiesObj)
        if (auto valueStr = prop.second.getAsString())
          info.properties[prop.first.str()] = valueStr->str();
    }

    units[info.name] = std::move(info);
  }

  return Error::success();
}

std::string UnitMetadata::getMetadataPath() const {
  return outputDirectory + "/.llvm-advisor-metadata.json";
}

std::string UnitMetadata::formatTimestamp(
    const std::chrono::system_clock::time_point &timePoint) const {
  auto timeT = std::chrono::system_clock::to_time_t(timePoint);
  std::stringstream ss;
  ss << std::put_time(std::gmtime(&timeT), "%Y-%m-%dT%H:%M:%SZ");
  return ss.str();
}

Expected<std::chrono::system_clock::time_point>
UnitMetadata::parseTimestamp(StringRef timestampStr) const {
  std::tm tm = {};
  std::istringstream ss(timestampStr.str());
  ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%SZ");

  if (ss.fail())
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "Invalid timestamp format: " + timestampStr);

  return std::chrono::system_clock::from_time_t(std::mktime(&tm));
}

bool UnitMetadata::fileExists(StringRef path) const {
  return sys::fs::exists(path);
}

Error UnitMetadata::createDirectoryIfNeeded(StringRef path) const {
  if (!sys::fs::exists(path)) {
    std::error_code ec = sys::fs::create_directories(path);
    if (ec)
      return createStringError(ec, "Error creating directory: " + path);
  }
  return Error::success();
}

} // namespace utils
} // namespace advisor
} // namespace llvm
