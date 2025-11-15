//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CAS_BUILTINCAS_H
#define LLVM_LIB_CAS_BUILTINCAS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/CAS/BuiltinCASContext.h"
#include "llvm/CAS/ObjectStore.h"

namespace llvm::cas {
class ActionCache;
namespace ondisk {
class UnifiedOnDiskCache;
} // namespace ondisk
namespace builtin {

/// Common base class for builtin CAS implementations using the same CASContext.
class BuiltinCAS : public ObjectStore {
public:
  BuiltinCAS() : ObjectStore(BuiltinCASContext::getDefaultContext()) {}

  Expected<CASID> parseID(StringRef Reference) final;

  Expected<ObjectRef> store(ArrayRef<ObjectRef> Refs,
                            ArrayRef<char> Data) final;
  virtual Expected<ObjectRef> storeImpl(ArrayRef<uint8_t> ComputedHash,
                                        ArrayRef<ObjectRef> Refs,
                                        ArrayRef<char> Data) = 0;

  virtual Expected<ObjectRef>
  storeFromNullTerminatedRegion(ArrayRef<uint8_t> ComputedHash,
                                sys::fs::mapped_file_region Map) {
    return storeImpl(ComputedHash, {}, ArrayRef(Map.data(), Map.size()));
  }

  /// Both builtin CAS implementations provide lifetime for free, so this can
  /// be const, and readData() and getDataSize() can be implemented on top of
  /// it.
  virtual ArrayRef<char> getDataConst(ObjectHandle Node) const = 0;

  ArrayRef<char> getData(ObjectHandle Node,
                         bool RequiresNullTerminator) const final {
    // BuiltinCAS Objects are always null terminated.
    return getDataConst(Node);
  }
  uint64_t getDataSize(ObjectHandle Node) const final {
    return getDataConst(Node).size();
  }

  Error createUnknownObjectError(const CASID &ID) const {
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "unknown object '" + ID.toString() + "'");
  }

  Error createCorruptObjectError(const CASID &ID) const {
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "corrupt object '" + ID.toString() + "'");
  }

  Error createCorruptStorageError() const {
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "corrupt storage");
  }

  Error validateObject(const CASID &ID) final;
};

/// Create a \p UnifiedOnDiskCache instance that uses \p BLAKE3 hashing.
Expected<std::unique_ptr<ondisk::UnifiedOnDiskCache>>
createBuiltinUnifiedOnDiskCache(StringRef Path);

/// \param UniDB A \p UnifiedOnDiskCache instance from \p
/// createBuiltinUnifiedOnDiskCache.
std::unique_ptr<ObjectStore> createObjectStoreFromUnifiedOnDiskCache(
    std::shared_ptr<ondisk::UnifiedOnDiskCache> UniDB);

/// \param UniDB A \p UnifiedOnDiskCache instance from \p
/// createBuiltinUnifiedOnDiskCache.
std::unique_ptr<ActionCache> createActionCacheFromUnifiedOnDiskCache(
    std::shared_ptr<ondisk::UnifiedOnDiskCache> UniDB);

// FIXME: Proxy not portable. Maybe also error-prone?
constexpr StringLiteral DefaultDirProxy = "/^llvm::cas::builtin::default";
constexpr StringLiteral DefaultDir = "llvm.cas.builtin.default";

} // end namespace builtin
} // end namespace llvm::cas

#endif // LLVM_LIB_CAS_BUILTINCAS_H
