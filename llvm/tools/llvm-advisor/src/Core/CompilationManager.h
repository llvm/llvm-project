#ifndef LLVM_ADVISOR_COMPILATION_MANAGER_H
#define LLVM_ADVISOR_COMPILATION_MANAGER_H

#include "../Config/AdvisorConfig.h"
#include "../Utils/FileClassifier.h"
#include "BuildExecutor.h"
#include "CompilationUnit.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <set>
#include <vector>

namespace llvm {
namespace advisor {

class CompilationManager {
public:
  explicit CompilationManager(const AdvisorConfig &config);
  ~CompilationManager();

  Expected<int> executeWithDataCollection(const std::string &compiler,
                                          const std::vector<std::string> &args);

private:
  std::set<std::string> scanDirectory(const std::string &dir) const;

  void
  collectGeneratedFiles(const std::set<std::string> &existingFiles,
                        std::vector<std::unique_ptr<CompilationUnit>> &units);

  Error
  organizeOutput(const std::vector<std::unique_ptr<CompilationUnit>> &units);

  void cleanupLeakedFiles();

  const AdvisorConfig &config_;
  BuildExecutor buildExecutor_;
  std::string tempDir_;
  std::string initialWorkingDir_;
};

} // namespace advisor
} // namespace llvm

#endif
