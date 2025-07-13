#ifndef LLVM_ADVISOR_UNIT_DETECTOR_H
#define LLVM_ADVISOR_UNIT_DETECTOR_H

#include "../Config/AdvisorConfig.h"
#include "../Core/CompilationUnit.h"
#include "../Utils/FileClassifier.h"
#include "llvm/Support/Error.h"
#include <vector>

namespace llvm {
namespace advisor {

class UnitDetector {
public:
  explicit UnitDetector(const AdvisorConfig &config);

  Expected<std::vector<CompilationUnitInfo>>
  detectUnits(const std::string &compiler,
              const std::vector<std::string> &args);

private:
  std::vector<SourceFile>
  findSourceFiles(const std::vector<std::string> &args) const;
  void extractBuildInfo(const std::vector<std::string> &args,
                        CompilationUnitInfo &unit);
  std::string generateUnitName(const std::vector<SourceFile> &sources) const;

  const AdvisorConfig &config_;
  FileClassifier classifier_;
};

} // namespace advisor
} // namespace llvm

#endif
