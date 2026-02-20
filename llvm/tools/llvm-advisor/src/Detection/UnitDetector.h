//===------------------- UnitDetector.h - LLVM Advisor --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the UnitDetector code generator driver. It provides a convenient
// command-line interface for generating an assembly file or a relocatable file,
// given LLVM bitcode.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_ADVISOR_DETECTION_UNITDETECTOR_H
#define LLVM_ADVISOR_DETECTION_UNITDETECTOR_H

#include "../Config/AdvisorConfig.h"
#include "../Core/CompilationUnit.h"
#include "../Utils/FileClassifier.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include <string>

namespace llvm {
namespace advisor {

class UnitDetector {
public:
  explicit UnitDetector(const AdvisorConfig &config);

  llvm::Expected<llvm::SmallVector<CompilationUnitInfo, 4>>
  detectUnits(llvm::StringRef compiler,
              const llvm::SmallVectorImpl<std::string> &args);

private:
  llvm::SmallVector<SourceFile, 4>
  findSourceFiles(const llvm::SmallVectorImpl<std::string> &args) const;
  void extractBuildInfo(const llvm::SmallVectorImpl<std::string> &args,
                        CompilationUnitInfo &unit);
  std::string
  generateUnitName(const llvm::SmallVectorImpl<SourceFile> &sources) const;

  const AdvisorConfig &config;
  FileClassifier classifier;
};

} // namespace advisor
} // namespace llvm

#endif // LLVM_ADVISOR_DETECTION_UNITDETECTOR_H
