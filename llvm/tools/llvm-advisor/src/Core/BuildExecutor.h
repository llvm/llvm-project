#ifndef LLVM_ADVISOR_BUILD_EXECUTOR_H
#define LLVM_ADVISOR_BUILD_EXECUTOR_H

#include "../Config/AdvisorConfig.h"
#include "BuildContext.h"
#include "llvm/Support/Error.h"
#include <set>
#include <string>
#include <vector>

namespace llvm {
namespace advisor {

class BuildExecutor {
public:
  BuildExecutor(const AdvisorConfig &config);

  Expected<int> execute(const std::string &compiler,
                        const std::vector<std::string> &args,
                        BuildContext &buildContext, const std::string &tempDir);

private:
  std::vector<std::string>
  instrumentCompilerArgs(const std::vector<std::string> &args,
                         BuildContext &buildContext,
                         const std::string &tempDir);

  const AdvisorConfig &config_;
};

} // namespace advisor
} // namespace llvm

#endif
