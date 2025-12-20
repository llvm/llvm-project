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

#ifndef LLVM_TOOLS_LLVM_ADVISOR_SRC_UTILS_UNITMETADATA_H
#define LLVM_TOOLS_LLVM_ADVISOR_SRC_UTILS_UNITMETADATA_H

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <map>
#include <string>
#include <vector>



namespace llvm::advisor::utils {

struct CompilationUnitInfo {
  std::string name;
  std::chrono::system_clock::time_point timestamp;
  size_t totalFiles;
  std::vector<std::string> artifactTypes;
  std::string status; // "in_progress", "completed", "failed"
  std::string outputPath;
  std::map<std::string, std::string> properties;
};

class UnitMetadata {
public:
  UnitMetadata(StringRef outputDirectory);
  ~UnitMetadata() = default;

  // Main operations
  auto loadMetadata() -> Error;
  auto saveMetadata() -> Error;
  void clear();

  // Unit management
  void registerUnit(StringRef unitName);
  void updateUnitStatus(StringRef unitName, StringRef status);
  void updateUnitFileCount(StringRef unitName, size_t fileCount);
  void addArtifactType(StringRef unitName, StringRef type);
  void setUnitProperty(StringRef unitName, StringRef key, StringRef value);

  // Query operations
  [[nodiscard]] auto hasUnit(StringRef unitName) const -> bool;
  [[nodiscard]] auto getUnitInfo(StringRef unitName) const
      -> Expected<CompilationUnitInfo>;
  [[nodiscard]] auto getAllUnits() const -> std::vector<CompilationUnitInfo>;
  [[nodiscard]] auto getRecentUnits(size_t count = 5) const
      -> std::vector<std::string>;
  [[nodiscard]] auto getMostRecentUnit() const -> Expected<std::string>;

  // Utility operations
  void removeUnit(StringRef unitName);
  void cleanupOldUnits(int maxAgeInDays = 30);
  [[nodiscard]] auto getUnitCount() const -> size_t;

  // JSON operations
  [[nodiscard]] auto toJson() const -> Expected<std::string>;
  auto fromJson(StringRef jsonStr) -> Error;

private:
  std::string outputDirectory;
  std::string metadataFilePath;
  std::map<std::string, CompilationUnitInfo> units;

  // Helper methods
  [[nodiscard]] auto getMetadataPath() const -> std::string;
  [[nodiscard]] auto
  formatTimestamp(const std::chrono::system_clock::time_point &timePoint) const
      -> std::string;
  [[nodiscard]] auto
  parseTimestamp(StringRef timestampStr) const
      -> Expected<std::chrono::system_clock::time_point>;
  [[nodiscard]] auto fileExists(StringRef path) const -> bool;
  auto createDirectoryIfNeeded(StringRef path) const -> Error;
};

} // namespace llvm::advisor::utils



#endif // LLVM_TOOLS_LLVM_ADVISOR_SRC_UTILS_UNITMETADATA_H
