//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_UNIFIEDONDISKCACHE_H
#define LLVM_CAS_UNIFIEDONDISKCACHE_H

#include "llvm/CAS/BuiltinUnifiedCASDatabases.h"
#include "llvm/CAS/OnDiskGraphDB.h"
#include <atomic>

namespace llvm::cas::ondisk {

class OnDiskKeyValueDB;

/// A unified CAS nodes and key-value database, using on-disk storage for both.
/// It manages storage growth and provides APIs for garbage collection.
///
/// High-level properties:
/// * While \p UnifiedOnDiskCache is open on a directory, by any process, the
///   storage size in that directory will keep growing unrestricted. For data to
///   become eligible for garbage-collection there should be no open instances
///   of \p UnifiedOnDiskCache for that directory, by any process.
/// * Garbage-collection needs to be triggered explicitly by the client. It can
///   be triggered on a directory concurrently, at any time and by any process,
///   without affecting any active readers/writers, in the same process or other
///   processes.
///
/// Usage patterns should be that an instance of \p UnifiedOnDiskCache is open
/// for a limited period of time, e.g. for the duration of a build operation.
/// For long-living processes that need periodic access to a
/// \p UnifiedOnDiskCache, the client should devise a scheme where access is
/// performed within some defined period. For example, if a service is designed
/// to continuously wait for requests that access a \p UnifiedOnDiskCache, it
/// could keep the instance alive while new requests are coming in but close it
/// after a time period in which there are no new requests.
class UnifiedOnDiskCache {
public:
  /// The \p OnDiskGraphDB instance for the open directory.
  OnDiskGraphDB &getGraphDB() { return *PrimaryGraphDB; }

  /// The \p OnDiskGraphDB instance for the open directory.
  OnDiskKeyValueDB &getKeyValueDB() { return *PrimaryKVDB; }

  /// Open a \p UnifiedOnDiskCache instance for a directory.
  ///
  /// \param Path directory for the on-disk database. The directory will be
  /// created if it doesn't exist.
  /// \param SizeLimit Optional size for limiting growth. This has an effect for
  /// when the instance is closed.
  /// \param HashName Identifier name for the hashing algorithm that is going to
  /// be used.
  /// \param HashByteSize Size for the object digest hash bytes.
  /// \param FaultInPolicy Controls how nodes are copied to primary store. This
  /// is recorded at creation time and subsequent opens need to pass the same
  /// policy otherwise the \p open will fail.
  LLVM_ABI_FOR_TEST static Expected<std::unique_ptr<UnifiedOnDiskCache>>
  open(StringRef Path, std::optional<uint64_t> SizeLimit, StringRef HashName,
       unsigned HashByteSize,
       OnDiskGraphDB::FaultInPolicy FaultInPolicy =
           OnDiskGraphDB::FaultInPolicy::FullTree);

  /// Validate the data in \p Path, if needed to ensure correctness.
  ///
  /// Note: if invalid data is detected and \p AllowRecovery is true, then
  /// recovery requires exclusive access to the CAS and it is an error to
  /// attempt recovery if there is concurrent use of the CAS.
  ///
  /// \param Path directory for the on-disk database.
  /// \param HashName Identifier name for the hashing algorithm that is going to
  /// be used.
  /// \param HashByteSize Size for the object digest hash bytes.
  /// \param CheckHash Whether to validate hashes match the data.
  /// \param AllowRecovery Whether to automatically recover from invalid data by
  /// marking the files for garbage collection.
  /// \param ForceValidation Whether to force validation to occur even if it
  /// should not be necessary.
  /// \param LLVMCasBinary If provided, validation is performed out-of-process
  /// using the given \c llvm-cas executable which protects against crashes
  /// during validation. Otherwise validation is performed in-process.
  ///
  /// \returns \c Valid if the data is already valid, \c Recovered if data
  /// was invalid but has been cleared, \c Skipped if validation is not needed,
  /// or an \c Error if validation cannot be performed or if the data is left
  /// in an invalid state because \p AllowRecovery is false.
  static Expected<ValidationResult>
  validateIfNeeded(StringRef Path, StringRef HashName, unsigned HashByteSize,
                   bool CheckHash, bool AllowRecovery, bool ForceValidation,
                   std::optional<StringRef> LLVMCasBinary);

  /// This is called implicitly at destruction time, so it is not required for a
  /// client to call this. After calling \p close the only method that is valid
  /// to call is \p needsGarbageCollection.
  ///
  /// \param CheckSizeLimit if true it will check whether the primary store has
  /// exceeded its intended size limit. If false the check is skipped even if a
  /// \p SizeLimit was passed to the \p open call.
  LLVM_ABI_FOR_TEST Error close(bool CheckSizeLimit = true);

  /// Set the size for limiting growth. This has an effect for when the instance
  /// is closed.
  LLVM_ABI_FOR_TEST void setSizeLimit(std::optional<uint64_t> SizeLimit);

  /// \returns the storage size of the cache data.
  LLVM_ABI_FOR_TEST uint64_t getStorageSize() const;

  /// \returns whether the primary store has exceeded the intended size limit.
  /// This can return false even if the overall size of the opened directory is
  /// over the \p SizeLimit passed to \p open. To know whether garbage
  /// collection needs to be triggered or not, call \p needsGarbaseCollection.
  LLVM_ABI_FOR_TEST bool hasExceededSizeLimit() const;

  /// \returns whether there are unused data that can be deleted using a
  /// \p collectGarbage call.
  bool needsGarbageCollection() const { return NeedsGarbageCollection; }

  /// Remove any unused data from the directory at \p Path. If there are no such
  /// data the operation is a no-op.
  ///
  /// This can be called concurrently, regardless of whether there is an open
  /// \p UnifiedOnDiskCache instance or not; it has no effect on readers/writers
  /// in the same process or other processes.
  ///
  /// It is recommended that garbage-collection is triggered concurrently in the
  /// background, so that it has minimal effect on the workload of the process.
  LLVM_ABI_FOR_TEST static Error collectGarbage(StringRef Path);

  /// Remove unused data from the current UnifiedOnDiskCache.
  Error collectGarbage();

  /// Helper function to convert the value stored in KeyValueDB and ObjectID.
  LLVM_ABI_FOR_TEST static ObjectID getObjectIDFromValue(ArrayRef<char> Value);

  using ValueBytes = std::array<char, sizeof(uint64_t)>;
  LLVM_ABI_FOR_TEST static ValueBytes getValueFromObjectID(ObjectID ID);

  LLVM_ABI_FOR_TEST ~UnifiedOnDiskCache();

private:
  friend class OnDiskGraphDB;
  friend class OnDiskKeyValueDB;

  UnifiedOnDiskCache();

  Expected<std::optional<ArrayRef<char>>>
  faultInFromUpstreamKV(ArrayRef<uint8_t> Key);

  /// \returns the storage size of the primary directory.
  uint64_t getPrimaryStorageSize() const;

  std::string RootPath;
  std::atomic<uint64_t> SizeLimit;

  int LockFD = -1;

  std::atomic<bool> NeedsGarbageCollection;
  std::string PrimaryDBDir;

  std::unique_ptr<OnDiskGraphDB> UpstreamGraphDB;
  std::unique_ptr<OnDiskGraphDB> PrimaryGraphDB;

  std::unique_ptr<OnDiskKeyValueDB> UpstreamKVDB;
  std::unique_ptr<OnDiskKeyValueDB> PrimaryKVDB;
};

} // namespace llvm::cas::ondisk

#endif // LLVM_CAS_UNIFIEDONDISKCACHE_H
