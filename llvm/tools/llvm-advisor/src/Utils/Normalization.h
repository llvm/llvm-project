//===------------------- Normalization.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Command line normalization and path canonicalization.
// Normalizes paths and commands for consistent comparison.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace llvm::advisor {

/// Resolve a path relative to Base and normalize separators/dots.
std::string normalizePath(StringRef Path, StringRef Base = {});

/// Resolve a path to its real path and verify it is within AllowedRoots.
Expected<std::string> canonicalizePath(StringRef Path,
                                       ArrayRef<StringRef> AllowedRoots);

/// Remove non-deterministic flags from a command line.
SmallVector<std::string, 16> normalizeCommand(ArrayRef<std::string> Arguments);

/// Detect language from file extension.
std::string inferLanguage(StringRef Path);

/// Extract the target triple from compiler arguments.
std::string inferTargetTriple(ArrayRef<std::string> Arguments);

} // namespace llvm::advisor
