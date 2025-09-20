//===- ConcurrentHashtable.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_CONCURRENTHASHTABLE_H
#define LLVM_ADT_CONCURRENTHASHTABLE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Config/llvm-config.h" // for LLVM_ENABLE_THREADS
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/RWMutex.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/xxhash.h"
#include <atomic>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <type_traits>

namespace llvm {

/// This file contains an implementation of resizeable concurrent hashtable.
/// The hashtable allows only concurrent insertions:
///
/// std::pair<DataTy, bool> = insert ( const KeyTy& );
///
/// Data structure:
///
/// Inserted value KeyTy is mapped to 64-bit hash value ->
///
///          [------- 64-bit Hash value --------]
///          [  StartEntryIndex ][ Bucket Index ]
///                    |                |
///              points to the     points to
///              first probe       the bucket.
///              position inside
///              bucket entries
///
/// After initialization, all buckets have an initial size. During insertions,
/// buckets might be extended to contain more entries. Each bucket can be
/// independently resized and rehashed(no need to lock the whole table).
/// Different buckets may have different sizes. If the single bucket is full
/// then the bucket is resized.
///
/// ConcurrentHashTableBase is a base implementation which encapsulates
/// common operations. The implementation assumes that stored data are
/// POD. It uses special markers for uninitialised values. Uninintialised
/// value is either zero, either 0xff(depending on the ZeroIsUndefValue
/// parameter).
///
/// ConcurrentHashTableBase has a MutexTy parameter which should satisfy
/// std::shared_mutex interface to lock buckets:
///   - lock_shared/unlock_shared if rehashing is not neccessary.
///   - lock/unlock when the bucket should be exclusively locked and resized.
/// To get single-threaded version of ConcurrentHashTableBase set MutexTy
/// to void.

template <typename KeyTy> struct ConcurrentHashTableBaseInfo {
  /// \returns Hash value for the specified \p Key.
  static inline uint64_t getHashValue(const KeyTy &Key) {
    return std::hash<KeyTy>{}(Key);
  }
};

template <typename KeyTy, typename DataTy, typename DerivedImplTy,
          typename Info = ConcurrentHashTableBaseInfo<KeyTy>,
          typename MutexTy = sys::RWMutex, bool ZeroIsUndefValue = true>
class ConcurrentHashTableBase {
public:
  /// ReservedSize - Specify the number of items for which space
  ///                will be allocated in advance.
  /// ThreadsNum - Specify the number of threads that will work
  ///                with the table.
  /// InitialNumberOfBuckets - Specify number of buckets. Small
  ///                number of buckets may lead to high thread
  ///                competitions and slow execution due to cache
  ///                synchronization.
  ConcurrentHashTableBase(
      uint64_t ReservedSize = 0,
      size_t ThreadsNum = parallel::strategy.compute_thread_count(),
      uint64_t InitialNumberOfBuckets = 0) {
    assert((ThreadsNum > 0) && "ThreadsNum must be greater than 0");

    this->ThreadsNum = ThreadsNum;

    // Calculate number of buckets.
    if constexpr (std::is_void<MutexTy>::value)
      NumberOfBuckets = 1;
    else {
      if (InitialNumberOfBuckets)
        NumberOfBuckets = InitialNumberOfBuckets;
      else
        NumberOfBuckets = (ThreadsNum == 1) ? 1 : ThreadsNum * 256;
      NumberOfBuckets = PowerOf2Ceil(NumberOfBuckets);
    }

    // Allocate buckets.
    BucketsArray = std::make_unique<Bucket[]>(NumberOfBuckets);

    uint64_t InitialBucketSize =
        calculateBucketSizeFromOverallSize(ReservedSize);

    // Initialize each bucket.
    for (uint64_t CurIdx = 0; CurIdx < NumberOfBuckets; ++CurIdx) {
      BucketsArray[CurIdx].Size = InitialBucketSize;
      BucketsArray[CurIdx].Data =
          static_cast<DerivedImplTy *>(this)->allocateData(InitialBucketSize);
    }

    // Calculate mask.
    BucketsHashMask = NumberOfBuckets - 1;

    size_t LeadingZerosNumber = countl_zero(BucketsHashMask);
    BucketsHashBitsNum = 64 - LeadingZerosNumber;
  }

  /// Erase content of table.
  void clear(uint64_t ReservedSize = 0) {
    if (ReservedSize) {
      for (uint64_t CurIdx = 0; CurIdx < NumberOfBuckets; ++CurIdx) {
        Bucket &CurBucket = BucketsArray[CurIdx];
        static_cast<DerivedImplTy *>(this)->deallocateData(CurBucket.Data,
                                                           CurBucket.Size);
        uint64_t BucketSize = calculateBucketSizeFromOverallSize(ReservedSize);
        CurBucket.Size = BucketSize;
        CurBucket.Data =
            static_cast<DerivedImplTy *>(this)->allocateData(BucketSize);
      }
    } else {
      for (uint64_t CurIdx = 0; CurIdx < NumberOfBuckets; ++CurIdx) {
        Bucket &CurBucket = BucketsArray[CurIdx];
        uint64_t BufferSize =
            static_cast<DerivedImplTy *>(this)->getBufferSize(CurBucket.Size);
        fillBufferWithUndefValue(CurBucket.Data, BufferSize);
      }
    }
  }

  ~ConcurrentHashTableBase() {
    // Deallocate buckets.
    for (uint64_t Idx = 0; Idx < NumberOfBuckets; Idx++)
      static_cast<DerivedImplTy *>(this)->deallocateData(
          BucketsArray[Idx].Data, BucketsArray[Idx].Size);
  }

  /// Print information about current state of hash table structures.
  void printStatistic(raw_ostream &OS) {
    OS << "\n--- HashTable statistic:\n";
    OS << "\nNumber of buckets = " << NumberOfBuckets;

    uint64_t OverallNumberOfAllocatedEntries = 0;
    uint64_t OverallNumberOfEntries = 0;
    uint64_t OverallSize = sizeof(*this) + NumberOfBuckets * sizeof(Bucket);

    DenseMap<uint64_t, uint64_t> BucketSizesMap;

    // For each bucket...
    for (uint64_t Idx = 0; Idx < NumberOfBuckets; Idx++) {
      Bucket &CurBucket = BucketsArray[Idx];

      BucketSizesMap[CurBucket.Size]++;

      OverallNumberOfAllocatedEntries += CurBucket.Size;
      OverallNumberOfEntries +=
          calculateNumberOfEntries(CurBucket.Data, CurBucket.Size);
      OverallSize +=
          static_cast<DerivedImplTy *>(this)->getBufferSize(CurBucket.Size);
    }

    OS << "\nOverall number of entries = " << OverallNumberOfEntries;
    OS << "\nOverall allocated size = " << OverallSize;

    std::stringstream stream;
    stream << std::fixed << std::setprecision(2)
           << ((float)OverallNumberOfEntries /
               (float)OverallNumberOfAllocatedEntries);
    std::string str = stream.str();

    OS << "\nLoad factor = " << str;
    for (auto &BucketSize : BucketSizesMap)
      OS << "\n Number of buckets with size " << BucketSize.first << ": "
         << BucketSize.second;
  }

protected:
  struct VoidMutex {
    inline void lock_shared() {}
    inline void unlock_shared() {}
    inline void lock() {}
    inline void unlock() {}
  };

  /// Bucket structure. Keeps bucket data.
  struct Bucket
      : public
#if LLVM_ENABLE_THREADS
        std::conditional_t<std::is_void<MutexTy>::value, VoidMutex, MutexTy>
#else
        VoidMutex
#endif
  {
    Bucket() = default;

    /// Size of bucket.
    uint64_t Size;

    /// Buffer keeping bucket data.
    uint8_t *Data;
  };

  void fillBufferWithUndefValue(uint8_t *Data, uint64_t BufferSize) const {
    if constexpr (ZeroIsUndefValue)
      memset(Data, 0, BufferSize);
    else
      memset(Data, 0xff, BufferSize);
  }

  uint64_t calculateNumberOfEntries(uint8_t *Data, uint64_t Size) {
    uint64_t Result = 0;
    for (uint64_t CurIdx = 0; CurIdx < Size; CurIdx++) {
      auto &AtomicData =
          static_cast<DerivedImplTy *>(this)->getDataEntry(Data, CurIdx, Size);
      if (!isNull(AtomicData.load()))
        Result++;
    }

    return Result;
  }

  uint64_t calculateBucketSizeFromOverallSize(uint64_t OverallSize) const {
    uint64_t BucketSize = OverallSize / NumberOfBuckets;
    BucketSize = std::max((uint64_t)1, BucketSize);
    BucketSize = PowerOf2Ceil(BucketSize);
    return BucketSize;
  }

  template <typename T> static inline bool isNull(T Data) {
    if constexpr (ZeroIsUndefValue)
      return Data == 0;
    else if constexpr (sizeof(Data) == 1)
      return static_cast<uint8_t>(Data) == 0xff;
    else if constexpr (sizeof(Data) == 2)
      return static_cast<uint16_t>(Data) == 0xffff;
    else if constexpr (sizeof(Data) == 4)
      return static_cast<uint32_t>(Data) == 0xffffffff;
    else if constexpr (sizeof(Data) == 8)
      return static_cast<uint64_t>(Data) == 0xffffffffffffffff;

    llvm_unreachable("Unsupported data size");
  }

  /// Common implementation of insert method. This implementation selects
  /// bucket, locks bucket and calls to child implementation which does final
  /// insertion.
  template <typename... Args>
  std::pair<DataTy, bool> insert(const KeyTy &NewKey, Args... args) {
    // Calculate hash.
    uint64_t Hash = Info::getHashValue(NewKey);
    // Get bucket.
    Bucket &CurBucket = BucketsArray[getBucketIdx(Hash)];

    // Calculate extendend hash bits.
    uint64_t ExtHashBits = Hash >> BucketsHashBitsNum;
    std::pair<DataTy, bool> Result;

    while (true) {
      uint64_t BucketSizeForRehashing = 0;
      CurBucket.lock_shared();
      // Call child implementation.
      if (static_cast<DerivedImplTy *>(this)->insertImpl(
              CurBucket, ExtHashBits, NewKey, Result, args...)) {
        CurBucket.unlock_shared();
        return Result;
      }

      BucketSizeForRehashing = CurBucket.Size;
      CurBucket.unlock_shared();

      // Rehash bucket.
      rehashBucket(CurBucket, BucketSizeForRehashing);
    }

    llvm_unreachable("Unhandled path of insert() method");
    return {};
  }

  /// Rehash bucket data.
  void rehashBucket(Bucket &CurBucket, uint64_t BucketSizeForRehashing) {
    CurBucket.lock();
    uint64_t OldSize = CurBucket.Size;
    if (BucketSizeForRehashing != OldSize) {
      CurBucket.unlock();
      return;
    }

    uint8_t *OldData = CurBucket.Data;
    uint64_t NewSize = OldSize << 1;
    uint8_t *NewData =
        static_cast<DerivedImplTy *>(this)->allocateData(NewSize);

    // Iterate through old data.
    for (uint64_t CurIdx = 0; CurIdx < OldSize; ++CurIdx) {
      auto &AtomicData = static_cast<DerivedImplTy *>(this)->getDataEntry(
          OldData, CurIdx, OldSize);
      auto CurData = AtomicData.load(std::memory_order_acquire);

      // Check data entry for null value.
      if (!isNull(CurData)) {
        auto &AtomicKey = static_cast<DerivedImplTy *>(this)->getKeyEntry(
            OldData, CurIdx, OldSize);
        auto CurKey = AtomicKey.load(std::memory_order_acquire);

        // Get index for position in the new bucket.
        uint64_t ExtHashBits =
            static_cast<DerivedImplTy *>(this)->getExtHashBits(CurKey);
        uint64_t NewIdx = getStartIdx(ExtHashBits, NewSize);
        while (true) {
          auto &NewAtomicData =
              static_cast<DerivedImplTy *>(this)->getDataEntry(NewData, NewIdx,
                                                               NewSize);
          auto NewCurData = NewAtomicData.load(std::memory_order_acquire);

          if (isNull(NewCurData)) {
            // Store data entry and key into the new bucket data.
            NewAtomicData.store(CurData, std::memory_order_release);
            auto &NewAtomicKey =
                static_cast<DerivedImplTy *>(this)->getKeyEntry(NewData, NewIdx,
                                                                NewSize);
            NewAtomicKey.store(CurKey, std::memory_order_release);
            break;
          }

          ++NewIdx;
          NewIdx &= (NewSize - 1);
        }
      }
    }

    CurBucket.Size = NewSize;
    CurBucket.Data = NewData;
    CurBucket.unlock();

    static_cast<DerivedImplTy *>(this)->deallocateData(OldData, OldSize);
  }

  uint64_t getBucketIdx(hash_code Hash) { return Hash & BucketsHashMask; }

  uint64_t getStartIdx(uint64_t ExtHashBits, uint64_t BucketSize) {
    assert((BucketSize > 0) && "Empty bucket");

    return ExtHashBits & (BucketSize - 1);
  }

  /// Number of bits in hash mask.
  uint8_t BucketsHashBitsNum = 0;

  /// Hash mask.
  uint64_t BucketsHashMask = 0;

  /// Array of buckets.
  std::unique_ptr<Bucket[]> BucketsArray;

  /// The number of buckets.
  uint64_t NumberOfBuckets = 0;

  /// Number of available threads.
  size_t ThreadsNum = 0;
};

/// ConcurrentHashTable: This class is optimized for small data like
/// uint32_t or uint64_t. It keeps keys and data in the internal table.
/// Keys and data should have equal alignment and size. They also should
/// satisfy requirements for atomic operations.
///
/// Bucket.Data contains an array of pairs [ DataTy, KeyTy ]:
///
/// [Bucket].Data -> [DataTy0][KeyTy0]...[DataTyN][KeyTyN]

template <typename KeyTy> class ConcurrentHashTableInfo {
public:
  /// \returns Hash value for the specified \p Key.
  static inline uint64_t getHashValue(KeyTy Key) {
    return std::hash<KeyTy>{}(Key);
  }

  /// \returns true if both \p LHS and \p RHS are equal.
  static inline bool isEqual(KeyTy LHS, KeyTy RHS) { return LHS == RHS; }
};

template <typename KeyTy, typename DataTy,
          typename Info = ConcurrentHashTableInfo<KeyTy>,
          typename MutexTy = sys::RWMutex, bool ZeroIsUndefValue = false,
          uint64_t MaxProbeCount = 512>
class ConcurrentHashTable
    : public ConcurrentHashTableBase<
          KeyTy, DataTy,
          ConcurrentHashTable<KeyTy, DataTy, Info, MutexTy, ZeroIsUndefValue,
                              MaxProbeCount>,
          Info, MutexTy, ZeroIsUndefValue> {
  using SuperClass = ConcurrentHashTableBase<
      KeyTy, DataTy,
      ConcurrentHashTable<KeyTy, DataTy, Info, MutexTy, ZeroIsUndefValue>, Info,
      MutexTy, ZeroIsUndefValue>;
  friend SuperClass;

  using Bucket = typename SuperClass::Bucket;
  using AtomicEntryTy = std::atomic<DataTy>;
  using AtomicKeyTy = std::atomic<KeyTy>;

  static_assert(sizeof(KeyTy) == sizeof(DataTy));
  static_assert(alignof(KeyTy) == alignof(DataTy));

public:
  ConcurrentHashTable(
      uint64_t ReservedSize = 0,
      size_t ThreadsNum = parallel::strategy.compute_thread_count(),
      uint64_t InitialNumberOfBuckets = 0)
      : ConcurrentHashTableBase<
            KeyTy, DataTy,
            ConcurrentHashTable<KeyTy, DataTy, Info, MutexTy, ZeroIsUndefValue,
                                MaxProbeCount>,
            Info, MutexTy, ZeroIsUndefValue>(ReservedSize, ThreadsNum,
                                             InitialNumberOfBuckets) {}

  std::pair<DataTy, bool> insert(const KeyTy &Key,
                                 function_ref<DataTy(KeyTy)> onInsert) {
    return SuperClass::insert(Key, onInsert);
  }

protected:
  /// Returns size of the buffer required to keep bucket data of \p Size.
  uint64_t getBufferSize(uint64_t Size) const {
    return (sizeof(DataTy) + sizeof(KeyTy)) * Size;
  }

  /// Allocates bucket data.
  uint8_t *allocateData(uint64_t Size) const {
    uint64_t BufferSize = getBufferSize(Size);
    uint8_t *Data = static_cast<uint8_t *>(
        llvm::allocate_buffer(BufferSize, alignof(DataTy)));
    SuperClass::fillBufferWithUndefValue(Data, BufferSize);
    return Data;
  }

  /// Deallocate bucket data.
  void deallocateData(uint8_t *Data, uint64_t Size) const {
    llvm::deallocate_buffer(Data, getBufferSize(Size), alignof(DataTy));
  }

  /// Returns reference to data entry with index /p CurIdx.
  LLVM_ATTRIBUTE_ALWAYS_INLINE AtomicEntryTy &
  getDataEntry(uint8_t *Data, uint64_t CurIdx, uint64_t Size) {
    return *(reinterpret_cast<AtomicEntryTy *>(
        Data + (sizeof(DataTy) + sizeof(KeyTy)) * CurIdx));
  }

  /// Returns reference to key entry with index /p CurIdx.
  LLVM_ATTRIBUTE_ALWAYS_INLINE AtomicKeyTy &
  getKeyEntry(uint8_t *Data, uint64_t CurIdx, uint64_t Size) {
    return *(reinterpret_cast<AtomicKeyTy *>(
        Data + (sizeof(KeyTy) + sizeof(DataTy)) * CurIdx + sizeof(DataTy)));
  }

  /// Returns extended hash bits value for specified key.
  LLVM_ATTRIBUTE_ALWAYS_INLINE uint64_t getExtHashBits(KeyTy Key) const {
    return (Info::getHashValue(Key)) >> SuperClass::BucketsHashBitsNum;
  }

  /// Inserts data returned by \p onInsert into the hashtable.
  ///   a) If data was inserted returns true and set \p Result.second = true
  ///   and \p Result.first = Data.
  ///   b) If data was found returns true and set \p Result.second = false
  ///   and \p Result.first = Data.
  ///   c) If the table is full returns false.
  LLVM_ATTRIBUTE_ALWAYS_INLINE bool
  insertImpl(Bucket &CurBucket, uint64_t ExtHashBits, const KeyTy &NewKey,
             std::pair<DataTy, bool> &Result,
             function_ref<DataTy(KeyTy)> onInsert) {
    assert(!SuperClass::isNull(NewKey) && "Null key value");

    uint64_t BucketSize = CurBucket.Size;
    uint8_t *Data = CurBucket.Data;
    uint64_t BucketMaxProbeCount = std::min(BucketSize, MaxProbeCount);
    uint64_t CurProbeCount = 0;
    uint64_t CurEntryIdx = SuperClass::getStartIdx(ExtHashBits, BucketSize);

    while (CurProbeCount < BucketMaxProbeCount) {
      AtomicKeyTy &AtomicKey = getKeyEntry(Data, CurEntryIdx, BucketSize);
      KeyTy CurKey = AtomicKey.load(std::memory_order_acquire);

      AtomicEntryTy &AtomicEntry = getDataEntry(Data, CurEntryIdx, BucketSize);

      if (SuperClass::isNull(CurKey)) {
        // Found empty slot. Insert data.
        if (AtomicKey.compare_exchange_strong(CurKey, NewKey)) {
          DataTy NewData = onInsert(NewKey);
          assert(!SuperClass::isNull(NewData) && "Null data value");

          AtomicEntry.store(NewData, std::memory_order_release);
          Result.first = NewData;
          Result.second = true;
          return true;
        }

        // The slot is overwritten from another thread. Retry slot probing.
        continue;
      } else if (Info::isEqual(CurKey, NewKey)) {
        // Already existed entry matched with inserted data is found.

        DataTy CurData = AtomicEntry.load(std::memory_order_acquire);
        while (SuperClass::isNull(CurData))
          CurData = AtomicEntry.load(std::memory_order_acquire);

        Result.first = CurData;
        Result.second = false;
        return true;
      }

      CurProbeCount++;
      CurEntryIdx++;
      CurEntryIdx &= (BucketSize - 1);
    }

    return false;
  }
};

/// ConcurrentHashTableByPtr: This class is optimized for the case when key
/// and/or data is an aggregate type. It keeps hash instead of the key and
/// pointer to the data allocated in external thread-safe allocator.
/// This hashtable is useful to have efficient access to aggregate data(like
/// strings, type descriptors...) and to keep only single copy of such an
/// aggregate.
///
/// To save space it keeps only 32-bits of the hash value. Which limits number
/// of resizings for single bucket up to x2^31.
///
/// Bucket.Data contains an array of EntryDataTy first and then array of
/// ExtHashBitsTy:
///
/// [Bucket].Data ->
///        [EntryDataTy0]...[EntryDataTyN][ExtHashBitsTy0]...[ExtHashBitsTyN]

template <typename KeyTy, typename KeyDataTy, typename AllocatorTy>
class ConcurrentHashTableInfoByPtr {
public:
  /// \returns Hash value for the specified \p Key.
  static inline uint64_t getHashValue(const KeyTy &Key) {
    return xxh3_64bits(Key);
  }

  /// \returns true if both \p LHS and \p RHS are equal.
  static inline bool isEqual(const KeyTy &LHS, const KeyTy &RHS) {
    return LHS == RHS;
  }

  /// \returns key for the specified \p KeyData.
  static inline const KeyTy &getKey(const KeyDataTy &KeyData) {
    return KeyData.getKey();
  }

  /// \returns newly created object of KeyDataTy type.
  static inline KeyDataTy *create(const KeyTy &Key, AllocatorTy &Allocator) {
    return KeyDataTy::create(Key, Allocator);
  }
};

template <typename KeyTy, typename KeyDataTy, typename AllocatorTy,
          typename Info =
              ConcurrentHashTableInfoByPtr<KeyTy, KeyDataTy, AllocatorTy>,
          typename MutexTy = sys::RWMutex, bool ZeroIsUndefValue = true,
          uint64_t MaxProbeCount = 512>
class ConcurrentHashTableByPtr
    : public ConcurrentHashTableBase<
          KeyTy, KeyDataTy *,
          ConcurrentHashTableByPtr<KeyTy, KeyDataTy, AllocatorTy, Info, MutexTy,
                                   ZeroIsUndefValue, MaxProbeCount>,
          Info, MutexTy, ZeroIsUndefValue> {
  using SuperClass = ConcurrentHashTableBase<
      KeyTy, KeyDataTy *,
      ConcurrentHashTableByPtr<KeyTy, KeyDataTy, AllocatorTy, Info, MutexTy,
                               ZeroIsUndefValue, MaxProbeCount>,
      Info, MutexTy, ZeroIsUndefValue>;
  friend SuperClass;

  using Bucket = typename SuperClass::Bucket;

  using EntryDataTy = KeyDataTy *;
  using ExtHashBitsTy = uint32_t;

  using AtomicEntryDataTy = std::atomic<EntryDataTy>;
  using AtomicExtHashBitsTy = std::atomic<ExtHashBitsTy>;

  static constexpr uint64_t MaxBucketSize = 1Ull << 31;
  static constexpr uint64_t MaxNumberOfBuckets = 0xFFFFFFFFUll;

  static_assert(alignof(EntryDataTy) >= alignof(ExtHashBitsTy),
                "EntryDataTy alignment must be greater or equal to "
                "ExtHashBitsTy alignment");
  static_assert(
      (alignof(EntryDataTy) % alignof(ExtHashBitsTy)) == 0,
      "EntryDataTy alignment must be a multiple of ExtHashBitsTy alignment");

public:
  ConcurrentHashTableByPtr(
      AllocatorTy &Allocator, uint64_t ReservedSize = 0,
      size_t ThreadsNum = parallel::strategy.compute_thread_count(),
      uint64_t InitialNumberOfBuckets = 0)
      : ConcurrentHashTableBase<
            KeyTy, KeyDataTy *,
            ConcurrentHashTableByPtr<KeyTy, KeyDataTy, AllocatorTy, Info,
                                     MutexTy, ZeroIsUndefValue, MaxProbeCount>,
            Info, MutexTy, ZeroIsUndefValue>(ReservedSize, ThreadsNum,
                                             InitialNumberOfBuckets),
        MultiThreadAllocator(Allocator) {
    assert((SuperClass::NumberOfBuckets <= MaxNumberOfBuckets) &&
           "NumberOfBuckets is too big");
  }

  std::pair<KeyDataTy *, bool> insert(const KeyTy &Key) {
    KeyDataTy *NewData = nullptr;
    return SuperClass::insert(Key, NewData);
  }

protected:
  /// Returns size of the buffer required to keep bucket data of \p Size.
  uint64_t getBufferSize(uint64_t Size) const {
    return (sizeof(EntryDataTy) + sizeof(ExtHashBitsTy)) * Size;
  }

  /// Allocates bucket data.
  uint8_t *allocateData(uint64_t Size) const {
    uint64_t BufferSize = getBufferSize(Size);
    uint8_t *Data = static_cast<uint8_t *>(
        llvm::allocate_buffer(BufferSize, alignof(EntryDataTy)));
    SuperClass::fillBufferWithUndefValue(Data, BufferSize);
    return Data;
  }

  /// Deallocate bucket data.
  void deallocateData(uint8_t *Data, uint64_t Size) const {
    llvm::deallocate_buffer(Data, getBufferSize(Size), alignof(EntryDataTy));
  }

  /// Returns reference to data entry with index /p CurIdx.
  LLVM_ATTRIBUTE_ALWAYS_INLINE AtomicEntryDataTy &
  getDataEntry(uint8_t *Data, uint64_t CurIdx, uint64_t) {
    return *(reinterpret_cast<AtomicEntryDataTy *>(
        Data + sizeof(AtomicEntryDataTy) * CurIdx));
  }

  /// Returns reference to key entry with index /p CurIdx.
  LLVM_ATTRIBUTE_ALWAYS_INLINE AtomicExtHashBitsTy &
  getKeyEntry(uint8_t *Data, uint64_t CurIdx, uint64_t Size) {
    return *(reinterpret_cast<AtomicExtHashBitsTy *>(
        Data + sizeof(AtomicEntryDataTy) * Size +
        sizeof(AtomicExtHashBitsTy) * CurIdx));
  }

  /// Returns extended hash bits value for specified key. We keep hash value
  /// instead of the key. So we can use kept value instead of calculating hash
  /// again.
  LLVM_ATTRIBUTE_ALWAYS_INLINE ExtHashBitsTy
  getExtHashBits(ExtHashBitsTy Key) const {
    return Key;
  }

  /// Inserts data created from \p NewKey into the hashtable.
  ///   a) If data was inserted then returns true and set \p Result.second =
  ///      true and \p Result.first = KeyDataTy*.
  ///   b) If data was found returns true and set \p Result.second = false
  ///      and \p Result.first = KeyDataTy*.
  ///   c) If the table is full returns false.
  LLVM_ATTRIBUTE_ALWAYS_INLINE bool
  insertImpl(Bucket &CurBucket, uint64_t ExtHashBits, const KeyTy &NewKey,
             std::pair<KeyDataTy *, bool> &Result, KeyDataTy *&NewData) {
    uint64_t BucketSize = CurBucket.Size;
    uint8_t *Data = CurBucket.Data;
    uint64_t BucketMaxProbeCount = std::min(BucketSize, MaxProbeCount);
    uint64_t CurProbeCount = 0;
    uint64_t CurEntryIdx = SuperClass::getStartIdx(ExtHashBits, BucketSize);

    while (CurProbeCount < BucketMaxProbeCount) {
      AtomicExtHashBitsTy &AtomicKey =
          getKeyEntry(Data, CurEntryIdx, BucketSize);
      ExtHashBitsTy CurHashBits = AtomicKey.load(std::memory_order_acquire);

      if (CurHashBits == static_cast<ExtHashBitsTy>(ExtHashBits) ||
          SuperClass::isNull(CurHashBits)) {
        AtomicEntryDataTy &AtomicData =
            getDataEntry(Data, CurEntryIdx, BucketSize);
        EntryDataTy EntryData = AtomicData.load(std::memory_order_acquire);
        if (SuperClass::isNull(EntryData)) {
          // Found empty slot. Insert data.
          if (!NewData)
            NewData = Info::create(NewKey, MultiThreadAllocator);

          if (AtomicData.compare_exchange_strong(EntryData, NewData)) {

            AtomicKey.store(ExtHashBits, std::memory_order_release);
            Result.first = NewData;
            Result.second = true;
            return true;
          }

          // The slot is overwritten from another thread. Retry slot probing.
          continue;
        } else if (Info::isEqual(Info::getKey(*EntryData), NewKey)) {
          // Hash matched. Check value for equality.
          if (NewData)
            MultiThreadAllocator.Deallocate(NewData);

          // Already existed entry matched with inserted data is found.
          Result.first = EntryData;
          Result.second = false;
          return true;
        }
      }

      CurProbeCount++;
      CurEntryIdx++;
      CurEntryIdx &= (BucketSize - 1);
    }

    if (BucketSize == MaxBucketSize)
      report_fatal_error("ConcurrentHashTableByPtr is full");

    return false;
  }

  // Used for allocating KeyDataTy values.
  AllocatorTy &MultiThreadAllocator;
};

} // end namespace llvm

#endif // LLVM_ADT_CONCURRENTHASHTABLE_H
