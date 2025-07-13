#ifndef LLVM_ADVISOR_FILE_CLASSIFIER_H
#define LLVM_ADVISOR_FILE_CLASSIFIER_H

#include <string>

namespace llvm {
namespace advisor {

struct FileClassification {
  std::string category;
  std::string description;
  bool isTemporary = false;
  bool isGenerated = true;
};

class FileClassifier {
public:
  FileClassification classifyFile(const std::string &filePath) const;
  bool shouldCollect(const std::string &filePath) const;
  std::string getLanguage(const std::string &filePath) const;
};

} // namespace advisor
} // namespace llvm

#endif
