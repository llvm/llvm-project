#include "UnitDetector.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Path.h"

namespace llvm {
namespace advisor {

UnitDetector::UnitDetector(const AdvisorConfig &config) : config_(config) {}

Expected<std::vector<CompilationUnitInfo>>
UnitDetector::detectUnits(const std::string &compiler,
                          const std::vector<std::string> &args) {

  auto sources = findSourceFiles(args);
  if (sources.empty()) {
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "No source files found");
  }

  CompilationUnitInfo unit;
  unit.name = generateUnitName(sources);
  unit.sources = sources;

  // Store original args but filter out source files for the compile flags
  for (const auto &arg : args) {
    // Skip source files when adding to compile flags
    StringRef extension = sys::path::extension(arg);
    if (!arg.empty() && arg[0] != '-' &&
        (extension == ".c" || extension == ".cpp" || extension == ".cc" ||
         extension == ".cxx" || extension == ".C")) {
      continue;
    }
    unit.compileFlags.push_back(arg);
  }

  // Extract output files and features
  extractBuildInfo(args, unit);

  return std::vector<CompilationUnitInfo>{unit};
}

std::vector<SourceFile>
UnitDetector::findSourceFiles(const std::vector<std::string> &args) const {
  std::vector<SourceFile> sources;

  for (const auto &arg : args) {
    if (arg.empty() || arg[0] == '-')
      continue;

    StringRef extension = sys::path::extension(arg);
    if (extension == ".c" || extension == ".cpp" || extension == ".cc" ||
        extension == ".cxx" || extension == ".C") {

      SourceFile source;
      source.path = arg;
      source.language = classifier_.getLanguage(arg);
      source.isHeader = false;
      sources.push_back(source);
    }
  }

  return sources;
}

void UnitDetector::extractBuildInfo(const std::vector<std::string> &args,
                                    CompilationUnitInfo &unit) {
  for (size_t i = 0; i < args.size(); ++i) {
    const auto &arg = args[i];

    if (arg == "-o" && i + 1 < args.size()) {
      StringRef output = args[i + 1];
      StringRef ext = sys::path::extension(output);
      if (ext == ".o") {
        unit.outputObject = args[i + 1];
      } else {
        unit.outputExecutable = args[i + 1];
      }
    }

    if (arg.find("openmp") != std::string::npos ||
        arg.find("offload") != std::string::npos ||
        arg.find("cuda") != std::string::npos) {
      unit.hasOffloading = true;
    }

    if (StringRef(arg).starts_with("-march=")) {
      unit.targetArch = arg.substr(7);
    }
  }
}

std::string
UnitDetector::generateUnitName(const std::vector<SourceFile> &sources) const {
  if (sources.empty())
    return "unknown";

  // Use first source file name as base
  std::string baseName = sys::path::stem(sources[0].path).str();

  // Add hash for uniqueness when multiple sources
  if (sources.size() > 1) {
    std::string combined;
    for (const auto &source : sources) {
      combined += source.path;
    }
    auto hash = hash_value(combined);
    baseName += "_" + std::to_string(static_cast<size_t>(hash) % 10000);
  }

  return baseName;
}

} // namespace advisor
} // namespace llvm
