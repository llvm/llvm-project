#ifndef LLVM_ADVISOR_FILE_MANAGER_H
#define LLVM_ADVISOR_FILE_MANAGER_H

#include "llvm/Support/Error.h"
#include <string>
#include <vector>

namespace llvm {
namespace advisor {

class FileManager {
public:
  /// Create unique temporary directory with pattern llvm-advisor-xxxxx
  static Expected<std::string>
  createTempDir(const std::string &prefix = "llvm-advisor");

  /// Recursively copy directory
  static Error copyDirectory(const std::string &source,
                             const std::string &dest);

  /// Remove directory and contents
  static Error removeDirectory(const std::string &path);

  /// Find files matching pattern
  static std::vector<std::string> findFiles(const std::string &directory,
                                            const std::string &pattern);

  /// Find files by extension
  static std::vector<std::string>
  findFilesByExtension(const std::string &directory,
                       const std::vector<std::string> &extensions);

  /// Move file from source to destination
  static Error moveFile(const std::string &source, const std::string &dest);

  /// Copy file from source to destination
  static Error copyFile(const std::string &source, const std::string &dest);

  /// Get file size
  static Expected<size_t> getFileSize(const std::string &path);
};

} // namespace advisor
} // namespace llvm

#endif
