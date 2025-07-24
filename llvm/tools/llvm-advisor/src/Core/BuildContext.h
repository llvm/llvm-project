//===------------------- BuildContext.h - LLVM Advisor --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the BuildContext code generator driver. It provides a convenient
// command-line interface for generating an assembly file or a relocatable file,
// given LLVM bitcode.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADVISOR_CORE_BUILDCONTEXT_H
#define LLVM_ADVISOR_CORE_BUILDCONTEXT_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <unordered_map>

namespace llvm {
namespace advisor {

enum class BuildPhase {
  Unknown,
  Preprocessing,
  Compilation,
  Assembly,
  Linking,
  Archiving,
  CMakeConfigure,
  CMakeBuild,
  MakefileBuild
};

enum class BuildTool {
  Unknown,
  Clang,
  GCC,
  LLVM_Tools,
  CMake,
  Make,
  Ninja,
  Linker,
  Archiver
};

struct BuildContext {
  BuildPhase phase;
  BuildTool tool;
  std::string workingDirectory;
  std::string outputDirectory;
  llvm::SmallVector<std::string, 8> inputFiles;
  llvm::SmallVector<std::string, 8> outputFiles;
  llvm::SmallVector<std::string, 8> expectedGeneratedFiles;
  std::unordered_map<std::string, std::string> metadata;
  bool hasOffloading = false;
  bool hasDebugInfo = false;
  bool hasOptimization = false;
};

} // namespace advisor
} // namespace llvm

#endif // LLVM_ADVISOR_CORE_BUILDCONTEXT_H
