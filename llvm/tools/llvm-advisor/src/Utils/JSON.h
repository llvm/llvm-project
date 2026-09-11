//===------------------- JSON.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Typed JSON helpers on top of LLVM's JSON support.
// Provides convenient wrappers for reading and writing JSON files.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"

namespace llvm::advisor {

/// Parse a JSON file from disk.
Expected<json::Value> parseJSONFile(StringRef Path);

/// Write a JSON value to disk atomically.
Error writeJSONFile(StringRef Path, const json::Value &Value);

/// Extract a string value from a JSON object.
Expected<std::string> getString(const json::Object &Object, StringRef Key);

/// Extract an array of strings from a JSON object.
/// Non-string elements are silently skipped.
SmallVector<std::string, 8> getStringArray(const json::Object &Object,
                                           StringRef Key);

/// Serialize a JSON value to a string.
std::string stringifyJSON(const json::Value &Value);

/// Wrap data in a standard success envelope with request metadata.
json::Value successEnvelope(json::Value Data);

/// Wrap an error in a standard error envelope with request metadata.
json::Value errorEnvelope(StringRef Code, StringRef Message);

} // namespace llvm::advisor
