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

namespace llvm::MachO {

using PathSeq = std::vector<std::string>;

/// Replace extension considering frameworks.
///
/// \param Path Location of file.
/// \param Extension File extension to update with.
void replace_extension(SmallVectorImpl<char> &Path, const Twine &Extension);
} // namespace llvm::MachO
#endif // LLVM_TEXTAPI_UTILS_H
