//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/BuiltinUnifiedCASDatabases.h"
#include "BuiltinCAS.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/UnifiedOnDiskCache.h"

using namespace llvm;
using namespace llvm::cas;

Expected<std::pair<std::unique_ptr<ObjectStore>, std::unique_ptr<ActionCache>>>
cas::createOnDiskUnifiedCASDatabases(StringRef Path) {
  std::shared_ptr<ondisk::UnifiedOnDiskCache> UniDB;
  if (Error E = builtin::createBuiltinUnifiedOnDiskCache(Path).moveInto(UniDB))
    return std::move(E);
  auto CAS = builtin::createObjectStoreFromUnifiedOnDiskCache(UniDB);
  auto AC = builtin::createActionCacheFromUnifiedOnDiskCache(std::move(UniDB));
  return std::make_pair(std::move(CAS), std::move(AC));
}

Expected<ValidationResult> cas::validateOnDiskUnifiedCASDatabasesIfNeeded(
    StringRef Path, bool CheckHash, bool AllowRecovery, bool ForceValidation,
    std::optional<StringRef> LLVMCasBinary) {
#if LLVM_ENABLE_ONDISK_CAS
  return ondisk::UnifiedOnDiskCache::validateIfNeeded(
      Path, builtin::BuiltinCASContext::getHashName(),
      sizeof(builtin::HashType), CheckHash, builtin::hashingFunc, AllowRecovery,
      ForceValidation, LLVMCasBinary);
#else
  return createStringError(inconvertibleErrorCode(), "OnDiskCache is disabled");
#endif
}
