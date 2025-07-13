#ifndef LLVM_ADVISOR_PROCESS_RUNNER_H
#define LLVM_ADVISOR_PROCESS_RUNNER_H

#include "llvm/Support/Error.h"
#include <string>
#include <vector>

namespace llvm {
namespace advisor {

class ProcessRunner {
public:
  struct ProcessResult {
    int exitCode;
    std::string stdout;
    std::string stderr;
    double executionTime;
  };

  static Expected<ProcessResult> run(const std::string &program,
                                     const std::vector<std::string> &args,
                                     int timeoutSeconds = 60);

  static Expected<ProcessResult>
  runWithEnv(const std::string &program, const std::vector<std::string> &args,
             const std::vector<std::string> &env, int timeoutSeconds = 60);
};

} // namespace advisor
} // namespace llvm

#endif
