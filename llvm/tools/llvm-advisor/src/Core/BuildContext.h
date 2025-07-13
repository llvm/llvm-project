#ifndef LLVM_ADVISOR_BUILD_CONTEXT_H
#define LLVM_ADVISOR_BUILD_CONTEXT_H

#include <map>
#include <string>
#include <vector>

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
  std::vector<std::string> inputFiles;
  std::vector<std::string> outputFiles;
  std::vector<std::string> expectedGeneratedFiles;
  std::map<std::string, std::string> metadata;
  bool hasOffloading = false;
  bool hasDebugInfo = false;
  bool hasOptimization = false;
};

} // namespace advisor
} // namespace llvm

#endif
