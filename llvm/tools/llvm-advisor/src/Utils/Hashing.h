//===------------------- Hashing.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides hashing functions for UnitID, CapabilityRunKey, and content hashing.
// Uses BLAKE3 for fast, secure hashing.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "Core/AdvisorTypes.h"

namespace llvm::advisor {

/// Return the BLAKE3 hash of a string as a hex string.
std::string hashString(StringRef Data);

/// Return the BLAKE3 hash of a file's contents as a hex string.
Expected<std::string> hashFile(StringRef Path);

/// Return the BLAKE3 hash of a JSON value as a hex string.
std::string hashJSON(const json::Value &Value);

/// Compute a deterministic hash-based ID for a compilation unit.
std::string computeUnitID(const UnitRecord &Unit);

/// Compute a deterministic snapshot ID from root paths and a timestamp.
std::string computeSnapshotID(StringRef SourceRoot, StringRef BuildRoot,
                              uint64_t CreatedUnix);

/// Compute a deterministic run key from a unit, capability, and input digest.
std::string computeCapabilityRunKey(const UnitRecord &Unit,
                                    StringRef CapabilityID,
                                    StringRef CapabilityVersion,
                                    StringRef InputDigest);

} // namespace llvm::advisor
