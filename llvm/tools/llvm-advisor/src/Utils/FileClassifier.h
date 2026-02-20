//===----------------- FileClassifier.h - LLVM Advisor --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the FileClassifier code generator driver. It provides a convenient
// command-line interface for generating an assembly file or a relocatable file,
// given LLVM bitcode.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_LLVM_ADVISOR_SRC_UTILS_FILECLASSIFIER_H
#define LLVM_TOOLS_LLVM_ADVISOR_SRC_UTILS_FILECLASSIFIER_H

#include "llvm/ADT/StringRef.h"
#include <string>


namespace llvm::advisor {

struct FileClassification {
  std::string category;
  std::string description;
  bool isTemporary = false;
  bool isGenerated = true;
};

class FileClassifier {
public:
  [[nodiscard]] auto classifyFile(llvm::StringRef filePath) const -> FileClassification;
  [[nodiscard]] auto shouldCollect(llvm::StringRef filePath) const -> bool;
  [[nodiscard]] auto getLanguage(llvm::StringRef filePath) const -> std::string;
};

} // namespace llvm::advisor


#endif
