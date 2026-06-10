//===-- EJitModuleLoader.cpp - Bitcode Lookup -----------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitModuleLoader.h"
#include "llvm/ExecutionEngine/EJIT/EJitCommon.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace llvm::ejit;

void EJitModuleLoader::registerBitcode(const std::string &funcName,
                                       const uint8_t *data, size_t size) {
  // Hash collision detection: fatal error if two different names map to
  // the same funcIdx. 32-bit FNV-1a makes this vanishingly unlikely.
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
EJitModuleLoader::getBitcodeByFuncIdx(uint32_t funcIdx) const {
  auto it = entriesByFuncIdx_.find(funcIdx);
  if (it == entriesByFuncIdx_.end())
    return make_error<StringError>(
        "No bitcode registered for funcIdx: " + std::to_string(funcIdx),
        inconvertibleErrorCode());
  return StringRef(reinterpret_cast<const char *>(it->second.data),
                   it->second.size);
}

const std::string &
EJitModuleLoader::getFuncNameByFuncIdx(uint32_t funcIdx) const {
  auto it = entriesByFuncIdx_.find(funcIdx);
  if (it != entriesByFuncIdx_.end())
    return it->second.funcName;
  static const std::string empty;
  return empty;
}

const EJitModuleLoader::FuncMeta &
EJitModuleLoader::getOrCacheFuncMeta(uint32_t funcIdx) {
  auto it = funcMetaCache_.find(funcIdx);
  if (it != funcMetaCache_.end())
    return it->second;

  FuncMeta &meta = funcMetaCache_[funcIdx];
  auto bitcode = getBitcodeByFuncIdx(funcIdx);
  if (!bitcode) {
    consumeError(bitcode.takeError());
    return meta;
  }

  auto Ctx = std::make_unique<LLVMContext>();
  auto Buf = MemoryBuffer::getMemBuffer(*bitcode,
      "meta_" + std::to_string(funcIdx) + ".bc");
  auto MOrErr = parseBitcodeFile(Buf->getMemBufferRef(), *Ctx);
  if (!MOrErr) {
    consumeError(MOrErr.takeError());
    return meta;
  }

  for (Function &F : (*MOrErr)->functions()) {
    if (F.isDeclaration())
      continue;
    // Precondition: detectHashCollisions() passed at init, so
    // hashFuncName uniquely identifies this function.
    if (hashFuncName(F.getName()) != funcIdx)
      continue;

    MDNode *MD = F.getMetadata(MD_EJIT_METADATA);
    if (!MD)
      break;

    for (const MDOperand &Op : MD->operands()) {
      auto *Sub = dyn_cast<MDNode>(Op.get());
      if (!Sub || Sub->getNumOperands() < 3)
        continue;
      auto *Tag = dyn_cast<MDString>(Sub->getOperand(0));
      if (!Tag || Tag->getString() != TAG_EJIT_PERIOD_ARR_IND)
        continue;
      auto *PN = dyn_cast<MDString>(Sub->getOperand(1));
      if (PN && meta.dimCount < 4)
        meta.periodNames[meta.dimCount++] = PN->getString().str();
    }
    break;
  }
  return meta;
}
