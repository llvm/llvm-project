//===-- EJitModuleLoader.cpp - Bitcode Lookup -----------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitModuleLoader.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/ExecutionEngine/EJIT/EJitCommon.h"
#include "llvm/ExecutionEngine/EJIT/EJitFuncRegistry.h"
#include "llvm/ExecutionEngine/EJIT/EJitLifecycleRegistry.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace llvm::ejit;

bool EJitModuleLoader::registerBitcode(const std::string &funcName,
                                       const uint8_t *data, size_t size) {
  // Reject malformed payloads up front; the caller propagates the failure to
  // ejit_init (never builds a half-registered taskpool).
  if (!data || size == 0)
    return false;
  // Dense, order-independent funcIndex assigned ONCE by name in the process-
  // global registry. The wrapper backfills the SAME value into its per-function
  // global, so the index it requests always selects this bitcode. Distinct
  // names get distinct dense indices (a monotonic counter), so two functions
  // can never alias one slot; capacity exhaustion is a clean rejection.
  uint32_t idx = EJitFuncRegistry::instance().resolveAssign(funcName);
  if (idx == kEJitInvalidFuncIndex)
    return false; // funcIndex capacity exhausted.
  auto It = entriesByFuncIdx_.find(idx);
  if (It != entriesByFuncIdx_.end()) {
    // An occupied slot is always the same name re-registering (the counter
    // registry never hands two names the same index).
    if (It->second.funcName != funcName)
      return false; // defensive: unreachable with the counter registry.
    // Same name, same payload (data ptr + size): idempotent success. A
    // different payload is rejected (the original bitcode, funcIndex and any
    // live cache are kept) rather than silently swapped.
    if (It->second.data == data && It->second.size == size)
      return true;
    return false;
  }
  Entry E;
  E.funcName = funcName;
  E.data = data;
  E.size = size;
  entriesByFuncIdx_.emplace(idx, std::move(E));
  return true;
}

Expected<StringRef>
EJitModuleLoader::getBitcodeByFuncIdx(uint32_t funcIdx) const {
  auto It = entriesByFuncIdx_.find(funcIdx);
  if (It == entriesByFuncIdx_.end())
    return make_error<StringError>("No bitcode registered for funcIdx: " +
                                       std::to_string(funcIdx),
                                   inconvertibleErrorCode());
  const Entry &E = It->second;
  return StringRef(reinterpret_cast<const char *>(E.data), E.size);
}

const std::string &
EJitModuleLoader::getFuncNameByFuncIdx(uint32_t funcIdx) const {
  auto It = entriesByFuncIdx_.find(funcIdx);
  if (It != entriesByFuncIdx_.end())
    return It->second.funcName;
  static const std::string empty;
  return empty;
}

const EJitModuleLoader::FuncMeta &
EJitModuleLoader::getOrCacheFuncMeta(uint32_t funcIdx) {
  auto it = funcMetaCache_.find(funcIdx);
  if (it != funcMetaCache_.end())
    return it->second;

  FuncMeta &meta = funcMetaCache_[funcIdx];
  auto Eit = entriesByFuncIdx_.find(funcIdx);
  if (Eit == entriesByFuncIdx_.end())
    return meta;
  const std::string FuncName = Eit->second.funcName;

  auto bitcode = getBitcodeByFuncIdx(funcIdx);
  if (!bitcode) {
    consumeError(bitcode.takeError());
    return meta;
  }

  auto Ctx = std::make_unique<LLVMContext>();
  auto Buf = MemoryBuffer::getMemBuffer(
      *bitcode, "meta_" + std::to_string(funcIdx) + ".bc");
  auto MOrErr = parseBitcodeFile(Buf->getMemBufferRef(), *Ctx);
  if (!MOrErr) {
    consumeError(MOrErr.takeError());
    return meta;
  }

  for (Function &F : (*MOrErr)->functions()) {
    if (F.isDeclaration() || F.getName() != FuncName)
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
      if (PN && meta.dimCount < 4) {
        // Read the explicit dimType slot the wrapper used, BY NAME, from the
        // process-global lifecycle registry — the loader never re-derives or
        // re-sorts it. Unregistered lifecycle → kEJitInvalidDimType (the
        // compile driver then rejects the request).
        meta.periodNames[meta.dimCount] = PN->getString().str();
        meta.dimTypes[meta.dimCount] =
            EJitLifecycleRegistry::instance().lookup(PN->getString());
        ++meta.dimCount;
      }
    }
    break;
  }
  return meta;
}
