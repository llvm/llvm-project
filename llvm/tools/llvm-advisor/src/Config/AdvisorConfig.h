//===------------------- AdvisorConfig.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the AdvisorConfig code generator driver. It provides a convenient
// command-line interface for generating an assembly file or a relocatable file,
// given LLVM bitcode.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_ADVISOR_SRC_CONFIG_ADVISORCONFIG_H
#define LLVM_TOOLS_LLVM_ADVISOR_SRC_CONFIG_ADVISORCONFIG_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <string>
#include <unordered_map>

namespace llvm::advisor {

class AdvisorConfig {
public:
  AdvisorConfig();

  auto loadFromFile(llvm::StringRef path) -> Expected<bool>;

  void setOutputDir(const std::string &dir) { outputDir = dir; }
  void setVerbose(bool verbose) { isVerbose = verbose; }
  void setKeepTemps(bool keep) { keepTemps = keep; }
  void setRunProfiler(bool run) { runProfiler = run; }
  void setTimeout(int seconds) { timeoutSeconds = seconds; }

  auto getOutputDir() const -> const std::string & { return outputDir; }
  auto getVerbose() const -> bool { return isVerbose; }
  auto getKeepTemps() const -> bool { return keepTemps; }
  auto getRunProfiler() const -> bool { return runProfiler; }
  auto getTimeout() const -> int { return timeoutSeconds; }

  auto getToolPath(llvm::StringRef tool) const -> std::string;
  void setToolPath(llvm::StringRef tool, llvm::StringRef path) {
    toolPaths[tool.str()] = path.str();
  }

private:
  std::string outputDir;
  bool isVerbose = false;
  bool keepTemps = false;
  bool runProfiler = true;
  int timeoutSeconds = 60;
  // Optional per-tool path overrides loaded from the configuration file.
  std::unordered_map<std::string, std::string> toolPaths;
};

} // namespace llvm::advisor

#endif
