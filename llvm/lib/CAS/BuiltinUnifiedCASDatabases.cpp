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
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::cas;

/// \returns the directory within \p Path for the builtin on-disk CAS, named
/// after the builtin hashing algorithm, so that it does not conflict with
/// on-disk CAS data of other hashing algorithms in \p Path.
static std::string getBuiltinCASPath(StringRef Path) {
  SmallString<256> CASPath(Path);
  sys::path::append(CASPath, builtin::BuiltinCASContext::getHashName());
  return std::string(CASPath);
}

Expected<std::pair<std::unique_ptr<ObjectStore>, std::unique_ptr<ActionCache>>>
cas::createOnDiskUnifiedCASDatabases(StringRef Path) {
  std::shared_ptr<ondisk::UnifiedOnDiskCache> UniDB;
  if (Error E =
          builtin::createBuiltinUnifiedOnDiskCache(getBuiltinCASPath(Path))
              .moveInto(UniDB))
    return std::move(E);
  auto CAS = builtin::createObjectStoreFromUnifiedOnDiskCache(UniDB);
  auto AC = builtin::createActionCacheFromUnifiedOnDiskCache(std::move(UniDB));
  return std::make_pair(std::move(CAS), std::move(AC));
}

Expected<ValidationResult>
cas::validateOnDiskUnifiedCASDatabasesIfNeeded(StringRef Path, bool CheckHash,
                                               bool ForceValidation) {
#if LLVM_ENABLE_ONDISK_CAS
  return ondisk::UnifiedOnDiskCache::validateIfNeeded(
      getBuiltinCASPath(Path), builtin::BuiltinCASContext::getHashName(),
      sizeof(builtin::HashType), CheckHash, builtin::hashingFunc,
      ForceValidation);
#else
  return createStringError(inconvertibleErrorCode(), "OnDiskCache is disabled");
#endif
}

Expected<ValidationResult>
cas::recoverOnDiskUnifiedCASDatabases(StringRef Path) {
#if LLVM_ENABLE_ONDISK_CAS
  return ondisk::UnifiedOnDiskCache::recover(getBuiltinCASPath(Path));
#else
  return createStringError(inconvertibleErrorCode(), "OnDiskCache is disabled");
#endif
}
