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

size_t EJitModuleLoader::getEntryCount() const {
  return entries_.size();
}

size_t EJitModuleLoader::getTotalBitcodeSize() const {
  return totalSize_;
}
