#ifndef LLVM_ADVISOR_CONFIG_H
#define LLVM_ADVISOR_CONFIG_H

#include "llvm/Support/Error.h"
#include <string>

namespace llvm {
namespace advisor {

class AdvisorConfig {
public:
  AdvisorConfig();

  Expected<bool> loadFromFile(const std::string &path);

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

  std::string getToolPath(const std::string &tool) const;

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
