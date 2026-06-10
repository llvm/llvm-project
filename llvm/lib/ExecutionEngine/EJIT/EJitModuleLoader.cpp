//===-- EJitModuleLoader.cpp - Bitcode Lookup -----------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitModuleLoader.h"
#include "llvm/ExecutionEngine/EJIT/EJitCommon.h"
#include "llvm/Support/Error.h"

using namespace llvm;
using namespace llvm::ejit;

void EJitModuleLoader::registerBitcode(const std::string &funcName,
                                       const uint8_t *data, size_t size) {
  entries_[funcName] = {funcName, data, size};
  totalSize_ += size;

  // Populate funcIdx index for O(1) lookup in the hot path.
  // Uses deterministic FNV-1a hash — AOT passes compute the same hash.
  uint32_t idx = hashFuncName(funcName);
  auto &entry = entriesByFuncIdx_[idx];
  if (entry.data && entry.funcName != funcName) {
    errs() << "EJIT PANIC: funcIdx collision — '" << funcName
           << "' and '" << entry.funcName
           << "' both hash to " << idx << "\n";
    report_fatal_error("EJIT: hash collision on funcIdx, rename functions");
  }
  entry = {funcName, data, size};
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

Expected<StringRef>
EJitModuleLoader::getBitcodeByFuncIdx(uint32_t funcIdx) const {
  auto it = entriesByFuncIdx_.find(funcIdx);
  if (it == entriesByFuncIdx_.end())
    return make_error<StringError>(
        "No bitcode registered for funcIdx: " + std::to_string(funcIdx),
        inconvertibleErrorCode());
  return StringRef(reinterpret_cast<const char *>(it->second.data),
                   it->second.size);
}

uint32_t EJitModuleLoader::getFuncIndex(const std::string &funcName) {
  auto it = funcToIndex_.find(funcName);
  if (it != funcToIndex_.end())
    return it->second;
  // For the new wrapper path (v2), the funcIdx equals the deterministic hash.
  // getFuncIndex remains for backward compat; new code uses hashFuncName.
  uint32_t idx = hashFuncName(funcName);
  funcToIndex_[funcName] = idx;
  return idx;
}

const std::string &EJitModuleLoader::getFuncName(uint32_t index) const {
  auto it = entriesByFuncIdx_.find(index);
  if (it != entriesByFuncIdx_.end())
    return it->second.funcName;
  static const std::string empty;
  return empty;
}

const std::string &EJitModuleLoader::getFuncNameByFuncIdx(uint32_t funcIdx) const {
  return getFuncName(funcIdx);
}

bool EJitModuleLoader::detectHashCollisions() const {
  // Collisions are detected and fatal at registerBitcode time.
  // This method exists as a documented checkpoint for ejit_init.
  return false;
}

size_t EJitModuleLoader::getEntryCount() const {
  return entries_.size();
}

size_t EJitModuleLoader::getTotalBitcodeSize() const {
  return totalSize_;
}
