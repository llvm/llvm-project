//===- CASOptions.h - Options for configuring the CAS -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the llvm::cas::CASConfiguration interface.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_CASCONFIGURATION_H
#define LLVM_CAS_CASCONFIGURATION_H

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <string>
#include <vector>

namespace llvm {
namespace cas {

class ActionCache;
class ObjectStore;

/// Base class for options configuring which CAS to use.
class CASConfiguration {
public:
  /// Path to a persistent backing store on-disk.
  ///
  /// - "" means there is none; falls back to in-memory.
  /// - "auto" is an alias for an automatically chosen location in the user's
  ///   system cache.
  std::string CASPath;
  /// Path to the CAS plugin library.
  std::string PluginPath;
  /// Each entry is a (<option-name>, <value>) pair.
  std::vector<std::pair<std::string, std::string>> PluginOptions;

  friend bool operator==(const CASConfiguration &LHS,
                         const CASConfiguration &RHS) {
    return LHS.CASPath == RHS.CASPath && LHS.PluginPath == RHS.PluginPath &&
           LHS.PluginOptions == RHS.PluginOptions;
  }
  friend bool operator!=(const CASConfiguration &LHS,
                         const CASConfiguration &RHS) {
    return !(LHS == RHS);
  }

  // Create CASDatabase from the CASConfiguration.
  llvm::Expected<std::pair<std::shared_ptr<llvm::cas::ObjectStore>,
                           std::shared_ptr<llvm::cas::ActionCache>>>
  createDatabases() const;

  /// Write CAS configuration file.
  void writeConfigurationFile(raw_ostream &OS) const;

  /// Create CASConfiguration from config file content.
  static llvm::Expected<CASConfiguration>
  createFromConfig(llvm::StringRef Content);

  /// Create CASConfiguration from recurively search config file from a path.
  ///
  /// Returns the path to configuration file and its corresponding
  /// CASConfiguration.
  static std::optional<std::pair<std::string, CASConfiguration>>
  createFromSearchConfigFile(
      StringRef Path,
      llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS = nullptr);

  /// Get resolved CASPath.
  Error getResolvedCASPath(llvm::SmallVectorImpl<char> &Result) const;
 
  /// DenseMap support \{
  static cas::CASConfiguration getDenseMapEmptyKey() { return {}; }

  static cas::CASConfiguration getDenseMapTombstoneKey() {
    return {"$", "$", {}};
  }
  unsigned getHashValue() const {
    return detail::combineHashValue(
        DenseMapInfo<StringRef>::getHashValue(StringRef(CASPath)),
        DenseMapInfo<StringRef>::getHashValue(StringRef(PluginPath)));
  }
  /// \}
  operator bool() { return !CASPath.empty() || !PluginPath.empty(); }
};

} // namespace cas

template <> struct DenseMapInfo<cas::CASConfiguration> {
  static cas::CASConfiguration getEmptyKey() {
    return cas::CASConfiguration::getDenseMapEmptyKey();
  }

  static cas::CASConfiguration getTombstoneKey() {
    return cas::CASConfiguration::getDenseMapTombstoneKey();
  }

  static unsigned getHashValue(const cas::CASConfiguration &config) {
    return config.getHashValue();
  }

  static bool isEqual(const cas::CASConfiguration &LHS,
                      const cas::CASConfiguration &RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm
#endif
