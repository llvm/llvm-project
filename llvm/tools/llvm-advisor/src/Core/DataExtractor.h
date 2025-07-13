#ifndef LLVM_ADVISOR_DATA_EXTRACTOR_H
#define LLVM_ADVISOR_DATA_EXTRACTOR_H

#include "../Config/AdvisorConfig.h"
#include "CompilationUnit.h"
#include "llvm/Support/Error.h"
#include <string>
#include <vector>

namespace llvm {
namespace advisor {

class DataExtractor {
public:
  DataExtractor(const AdvisorConfig &config);

  Error extractAllData(CompilationUnit &unit, const std::string &tempDir);

private:
  std::vector<std::string>
  getBaseCompilerArgs(const CompilationUnitInfo &unitInfo) const;

  Error extractIR(CompilationUnit &unit, const std::string &tempDir);
  Error extractAssembly(CompilationUnit &unit, const std::string &tempDir);
  Error extractAST(CompilationUnit &unit, const std::string &tempDir);
  Error extractPreprocessed(CompilationUnit &unit, const std::string &tempDir);
  Error extractIncludeTree(CompilationUnit &unit, const std::string &tempDir);
  Error extractDebugInfo(CompilationUnit &unit, const std::string &tempDir);
  Error extractStaticAnalysis(CompilationUnit &unit,
                              const std::string &tempDir);
  Error extractMacroExpansion(CompilationUnit &unit,
                              const std::string &tempDir);
  Error extractCompilationPhases(CompilationUnit &unit,
                                 const std::string &tempDir);

  Error runCompilerWithFlags(const std::vector<std::string> &args);

  const AdvisorConfig &config_;
};

} // namespace advisor
} // namespace llvm

#endif
