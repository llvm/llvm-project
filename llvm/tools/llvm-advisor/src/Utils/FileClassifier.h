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
#ifndef LLVM_ADVISOR_FILE_CLASSIFIER_H
#define LLVM_ADVISOR_FILE_CLASSIFIER_H

#include "llvm/ADT/StringRef.h"
#include <string>

namespace llvm {
namespace advisor {

struct FileClassification {
  std::string category;
  std::string description;
  bool isTemporary = false;
  bool isGenerated = true;
};

class FileClassifier {
public:
  FileClassification classifyFile(llvm::StringRef filePath) const;
  bool shouldCollect(llvm::StringRef filePath) const;
  std::string getLanguage(llvm::StringRef filePath) const;
};

} // namespace advisor
} // namespace llvm

#endif
