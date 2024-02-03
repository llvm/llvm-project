//===- llvm/TextAPI/Utils.h - TAPI Utils -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper functionality used for Darwin specific operations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TEXTAPI_UTILS_H
#define LLVM_TEXTAPI_UTILS_H

#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#if !defined(PATH_MAX)
#define PATH_MAX 1024
#endif

#define MACCATALYST_PREFIX_PATH "/System/iOSSupport"
#define DRIVERKIT_PREFIX_PATH "/System/DriverKit"

namespace llvm::MachO {

using PathSeq = std::vector<std::string>;

// Defines simple struct for storing symbolic links.
struct SymLink {
  std::string SrcPath;
  std::string LinkContent;

  SymLink(std::string Path, std::string Link)
      : SrcPath(std::move(Path)), LinkContent(std::move(Link)) {}

  SymLink(StringRef Path, StringRef Link)
      : SrcPath(std::string(Path)), LinkContent(std::string(Link)) {}
};

/// Replace extension considering frameworks.
///
/// \param Path Location of file.
/// \param Extension File extension to update with.
void replace_extension(SmallVectorImpl<char> &Path, const Twine &Extension);

/// Determine whether to skip over symlink due to either too many symlink levels
/// or is cyclic.
///
/// \param Path Location to symlink.
/// \param Result Holds whether to skip over Path.
std::error_code shouldSkipSymLink(const Twine &Path, bool &Result);

/// Turn absolute symlink into relative.
///
/// \param From The symlink.
/// \param To What the symlink points to.
/// \param RelativePath Path location to update what the symlink points to.
std::error_code make_relative(StringRef From, StringRef To,
                              SmallVectorImpl<char> &RelativePath);

/// Determine if library is private by parsing file path.
/// It does not touch the file system.
///
/// \param Path File path for library.
/// \param IsSymLink Whether path points to a symlink.
bool isPrivateLibrary(StringRef Path, bool IsSymLink = false);

} // namespace llvm::MachO
#endif // LLVM_TEXTAPI_UTILS_H
