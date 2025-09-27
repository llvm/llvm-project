//===-- UnitMetadata.h - Compilation Unit Metadata Management --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the UnitMetadata class for tracking compilation unit
// metadata including timestamps, file counts, and processing status.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADVISOR_UTILS_UNITMETADATA_H
#define LLVM_ADVISOR_UTILS_UNITMETADATA_H

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <map>
#include <string>
#include <vector>

namespace llvm {
namespace advisor {
namespace utils {

struct CompilationUnitInfo {
  std::string Name;
  std::chrono::system_clock::time_point Timestamp;
  size_t TotalFiles;
  std::vector<std::string> ArtifactTypes;
  std::string Status; // "in_progress", "completed", "failed"
  std::string OutputPath;
  std::map<std::string, std::string> Properties;
};

class UnitMetadata {
public:
  UnitMetadata(StringRef OutputDirectory);
  ~UnitMetadata() = default;

  // Main operations
  Error loadMetadata();
  Error saveMetadata();
  void clear();

  // Unit management
  void registerUnit(StringRef UnitName);
  void updateUnitStatus(StringRef UnitName, StringRef Status);
  void updateUnitFileCount(StringRef UnitName, size_t FileCount);
  void addArtifactType(StringRef UnitName, StringRef Type);
  void setUnitProperty(StringRef UnitName, StringRef Key, StringRef Value);

  // Query operations
  bool hasUnit(StringRef UnitName) const;
  Expected<CompilationUnitInfo> getUnitInfo(StringRef UnitName) const;
  std::vector<CompilationUnitInfo> getAllUnits() const;
  std::vector<std::string> getRecentUnits(size_t Count = 5) const;
  Expected<std::string> getMostRecentUnit() const;

  // Utility operations
  void removeUnit(StringRef UnitName);
  void cleanupOldUnits(int MaxAgeInDays = 30);
  size_t getUnitCount() const;

  // JSON operations
  Expected<std::string> toJson() const;
  Error fromJson(StringRef JsonStr);

private:
  std::string OutputDirectory;
  std::string MetadataFilePath;
  std::map<std::string, CompilationUnitInfo> Units;

  // Helper methods
  std::string getMetadataPath() const;
  std::string
  formatTimestamp(const std::chrono::system_clock::time_point &TimePoint) const;
  Expected<std::chrono::system_clock::time_point>
  parseTimestamp(StringRef TimestampStr) const;
  bool fileExists(StringRef Path) const;
  Error createDirectoryIfNeeded(StringRef Path) const;
};

} // namespace utils
} // namespace advisor
} // namespace llvm

#endif // LLVM_ADVISOR_UTILS_UNITMETADATA_H
