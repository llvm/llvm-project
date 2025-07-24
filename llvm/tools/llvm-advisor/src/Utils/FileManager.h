//===-------------------- FileManager.h - LLVM Advisor --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the FileManager code generator driver. It provides a convenient
// command-line interface for generating an assembly file or a relocatable file,
// given LLVM bitcode.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_ADVISOR_FILE_MANAGER_H
#define LLVM_ADVISOR_FILE_MANAGER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <string>

namespace llvm {
namespace advisor {

class FileManager {
public:
  /// Create unique temporary directory with pattern llvm-advisor-xxxxx
  static Expected<std::string>
  createTempDir(llvm::StringRef prefix = "llvm-advisor");

  /// Recursively copy directory
  static Error copyDirectory(llvm::StringRef source, llvm::StringRef dest);

  /// Remove directory and contents
  static Error removeDirectory(llvm::StringRef path);

  /// Find files matching pattern
  static llvm::SmallVector<std::string, 8> findFiles(llvm::StringRef directory,
                                                     llvm::StringRef pattern);

  /// Find files by extension
  static llvm::SmallVector<std::string, 8>
  findFilesByExtension(llvm::StringRef directory,
                       const llvm::SmallVector<std::string, 8> &extensions);

  /// Move file from source to destination
  static Error moveFile(llvm::StringRef source, llvm::StringRef dest);

  /// Copy file from source to destination
  static Error copyFile(llvm::StringRef source, llvm::StringRef dest);

  /// Get file size
  static Expected<size_t> getFileSize(llvm::StringRef path);
};

} // namespace advisor
} // namespace llvm

#endif
