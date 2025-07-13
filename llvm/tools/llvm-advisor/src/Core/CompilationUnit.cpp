#include "CompilationUnit.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace llvm {
namespace advisor {

CompilationUnit::CompilationUnit(const CompilationUnitInfo &info,
                                 const std::string &workDir)
    : info_(info), workDir_(workDir) {
  // Create unit-specific data directory
  SmallString<128> dataDir;
  sys::path::append(dataDir, workDir, "units", info.name);
  sys::fs::create_directories(dataDir);
}

std::string CompilationUnit::getPrimarySource() const {
  if (info_.sources.empty()) {
    return "";
  }
  return info_.sources[0].path;
}

std::string CompilationUnit::getDataDir() const {
  SmallString<128> dataDir;
  sys::path::append(dataDir, workDir_, "units", info_.name);
  return dataDir.str().str();
}

std::string CompilationUnit::getExecutablePath() const {
  return info_.outputExecutable;
}

void CompilationUnit::addGeneratedFile(const std::string &type,
                                       const std::string &path) {
  generatedFiles_[type].push_back(path);
}

bool CompilationUnit::hasGeneratedFiles(const std::string &type) const {
  if (type.empty()) {
    return !generatedFiles_.empty();
  }
  auto it = generatedFiles_.find(type);
  return it != generatedFiles_.end() && !it->second.empty();
}

std::vector<std::string>
CompilationUnit::getGeneratedFiles(const std::string &type) const {
  if (type.empty()) {
    std::vector<std::string> allFiles;
    for (const auto &pair : generatedFiles_) {
      allFiles.insert(allFiles.end(), pair.second.begin(), pair.second.end());
    }
    return allFiles;
  }
  auto it = generatedFiles_.find(type);
  return it != generatedFiles_.end() ? it->second : std::vector<std::string>();
}

const std::unordered_map<std::string, std::vector<std::string>> &
CompilationUnit::getAllGeneratedFiles() const {
  return generatedFiles_;
}

} // namespace advisor
} // namespace llvm