//===-- EJitModuleLoader.cpp - Bitcode Lookup -----------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitModuleLoader.h"
#include "llvm/Support/Error.h"

using namespace llvm;
using namespace llvm::ejit;

void EJitModuleLoader::registerBitcode(const std::string &funcName,
                                       const uint8_t *data, size_t size) {
  entries_[funcName] = {data, size};
  totalSize_ += size;
}

Expected<StringRef>
EJitModuleLoader::getBitcode(const std::string &funcName) const {
  auto it = entries_.find(funcName);
  if (it == entries_.end())
    return make_error<StringError>(
        "No bitcode registered for function: " + funcName,
        inconvertibleErrorCode());
  return StringRef(reinterpret_cast<const char *>(it->second.data),
                   it->second.size);
}

uint32_t EJitModuleLoader::getFuncIndex(const std::string &funcName) {
  auto it = funcToIndex_.find(funcName);
  if (it != funcToIndex_.end())
    return it->second;
  uint32_t idx = static_cast<uint32_t>(indexToFunc_.size());
  funcToIndex_[funcName] = idx;
  indexToFunc_.push_back(funcName);
  return idx;
}

const std::string &EJitModuleLoader::getFuncName(uint32_t index) const {
  static const std::string empty;
  if (index >= indexToFunc_.size())
    return empty;
  return indexToFunc_[index];
}

size_t EJitModuleLoader::getEntryCount() const {
  return entries_.size();
}

size_t EJitModuleLoader::getTotalBitcodeSize() const {
  return totalSize_;
}
