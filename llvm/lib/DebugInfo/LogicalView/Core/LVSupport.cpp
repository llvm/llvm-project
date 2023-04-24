//===-- LVSupport.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the supporting functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/LogicalView/Core/LVSupport.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include <iomanip>

using namespace llvm;
using namespace llvm::logicalview;

#define DEBUG_TYPE "Support"

namespace {
// Unique string pool instance used by all logical readers.
LVStringPool StringPool;
} // namespace
LVStringPool &llvm::logicalview::getStringPool() { return StringPool; }

// Perform the following transformations to the given 'Path':
// - all characters to lowercase.
// - '\\' into '/' (Platform independent).
// - '//' into '/'
std::string llvm::logicalview::transformPath(StringRef Path) {
  std::string Name(Path);
  std::transform(Name.begin(), Name.end(), Name.begin(), tolower);
  std::replace(Name.begin(), Name.end(), '\\', '/');

  // Remove all duplicate slashes.
  size_t Pos = 0;
  while ((Pos = Name.find("//", Pos)) != std::string::npos)
    Name.erase(Pos, 1);

  return Name;
}

// Convert the given 'Path' to lowercase and change any matching character
// from 'CharSet' into '_'.
// The characters in 'CharSet' are:
//   '/', '\', '<', '>', '.', ':', '%', '*', '?', '|', '"', ' '.
std::string llvm::logicalview::flattenedFilePath(StringRef Path) {
  std::string Name(Path);
  std::transform(Name.begin(), Name.end(), Name.begin(), tolower);

  const char *CharSet = "/\\<>.:%*?|\" ";
  char *Input = Name.data();
  while (Input && *Input) {
    Input = strpbrk(Input, CharSet);
    if (Input)
      *Input++ = '_';
  };
  return Name;
}
