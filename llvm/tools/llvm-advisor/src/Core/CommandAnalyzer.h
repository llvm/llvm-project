#ifndef LLVM_ADVISOR_COMMAND_ANALYZER_H
#define LLVM_ADVISOR_COMMAND_ANALYZER_H

#include "BuildContext.h"
#include <string>
#include <vector>

namespace llvm {
namespace advisor {

class CommandAnalyzer {
public:
  CommandAnalyzer(const std::string &command,
                  const std::vector<std::string> &args);

  BuildContext analyze() const;

private:
  BuildTool detectBuildTool() const;
  BuildPhase detectBuildPhase(BuildTool tool) const;
  void detectBuildFeatures(BuildContext &context) const;
  std::vector<std::string> extractInputFiles() const;
  std::vector<std::string> extractOutputFiles() const;

  std::string command_;
  std::vector<std::string> args_;
};

} // namespace advisor
} // namespace llvm

#endif
