#ifndef LLVM_ADVISOR_COMPILATION_UNIT_H
#define LLVM_ADVISOR_COMPILATION_UNIT_H

#include "llvm/Support/Error.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace llvm {
namespace advisor {

struct SourceFile {
  std::string path;
  std::string language;
  bool isHeader = false;
  std::vector<std::string> dependencies;
};

struct CompilationUnitInfo {
  std::string name;
  std::vector<SourceFile> sources;
  std::vector<std::string> compileFlags;
  std::string targetArch;
  bool hasOffloading = false;
  std::string outputObject;
  std::string outputExecutable;
};

class CompilationUnit {
public:
  CompilationUnit(const CompilationUnitInfo &info, const std::string &workDir);

  const std::string &getName() const { return info_.name; }
  const CompilationUnitInfo &getInfo() const { return info_; }
  const std::string &getWorkDir() const { return workDir_; }
  std::string getPrimarySource() const;

  std::string getDataDir() const;
  std::string getExecutablePath() const;

  void addGeneratedFile(const std::string &type, const std::string &path);

  bool hasGeneratedFiles(const std::string &type) const;
  std::vector<std::string>
  getGeneratedFiles(const std::string &type = "") const;
  const std::unordered_map<std::string, std::vector<std::string>> &
  getAllGeneratedFiles() const;

private:
  CompilationUnitInfo info_;
  std::string workDir_;
  std::unordered_map<std::string, std::vector<std::string>> generatedFiles_;
};

} // namespace advisor
} // namespace llvm

#endif