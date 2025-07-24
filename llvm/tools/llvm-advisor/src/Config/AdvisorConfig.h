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

#ifndef LLVM_ADVISOR_CONFIG_H
#define LLVM_ADVISOR_CONFIG_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <string>

namespace llvm {
namespace advisor {

class AdvisorConfig {
public:
  AdvisorConfig();

  Expected<bool> loadFromFile(llvm::StringRef path);

  void setOutputDir(const std::string &dir) { OutputDir_ = dir; }
  void setVerbose(bool verbose) { Verbose_ = verbose; }
  void setKeepTemps(bool keep) { KeepTemps_ = keep; }
  void setRunProfiler(bool run) { RunProfiler_ = run; }
  void setTimeout(int seconds) { TimeoutSeconds_ = seconds; }

  const std::string &getOutputDir() const { return OutputDir_; }
  bool getVerbose() const { return Verbose_; }
  bool getKeepTemps() const { return KeepTemps_; }
  bool getRunProfiler() const { return RunProfiler_; }
  int getTimeout() const { return TimeoutSeconds_; }

  std::string getToolPath(llvm::StringRef tool) const;

private:
  std::string OutputDir_;
  bool Verbose_ = false;
  bool KeepTemps_ = false;
  bool RunProfiler_ = true;
  int TimeoutSeconds_ = 60;
};

} // namespace advisor
} // namespace llvm

#endif
