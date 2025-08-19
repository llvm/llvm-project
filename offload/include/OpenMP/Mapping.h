//===-- OpenMP/Mapping.h - OpenMP/OpenACC pointer mapping -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarations for managing host-to-device pointer mappings.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_OPENMP_MAPPING_H
#define OMPTARGET_OPENMP_MAPPING_H

#include "ExclusiveAccess.h"
#include "Shared/EnvironmentVar.h"
#include "omptarget.h"

#include <cstdint>
#include <mutex>
#include <string>

#include "llvm/ADT/SmallSet.h"

struct DeviceTy;
class AsyncInfoTy;

using map_var_info_t = void *;

class MappingConfig {

  MappingConfig() {
    BoolEnvar ForceAtomic = BoolEnvar("LIBOMPTARGET_MAP_FORCE_ATOMIC", true);
    UseEventsForAtomicTransfers = ForceAtomic;
  }

public:
  static const MappingConfig &get() {
    static MappingConfig MP;
    return MP;
  };

  /// Flag to indicate if we use events to ensure the atomicity of
  /// map clauses or not. Can be modified with an environment variable.
  bool UseEventsForAtomicTransfers = true;
};

/// Information about shadow pointers.
struct ShadowPtrInfoTy {
  void **HstPtrAddr = nullptr;
  void *HstPtrVal = nullptr;
  void **TgtPtrAddr = nullptr;
  void *TgtPtrVal = nullptr;

  bool operator==(const ShadowPtrInfoTy &Other) const {
    return HstPtrAddr == Other.HstPtrAddr;
  }
};

inline bool operator<(const ShadowPtrInfoTy &lhs, const ShadowPtrInfoTy &rhs) {
  return lhs.HstPtrAddr < rhs.HstPtrAddr;
}

/// Map between host data and target data.
struct HostDataToTargetTy {
  const uintptr_t HstPtrBase; // host info.
  const uintptr_t HstPtrBegin;
  const uintptr_t HstPtrEnd;       // non-inclusive.
  const map_var_info_t HstPtrName; // Optional source name of mapped variable.

  const uintptr_t TgtAllocBegin; // allocated target memory
  const uintptr_t TgtPtrBegin; // mapped target memory = TgtAllocBegin + padding

private:
  static const uint64_t INFRefCount = ~(uint64_t)0;
  static std::string refCountToStr(uint64_t RefCount) {
    return RefCount == INFRefCount ? "INF" : std::to_string(RefCount);
  }

  struct StatesTy {
    StatesTy(uint64_t DRC, uint64_t HRC)
        : DynRefCount(DRC), HoldRefCount(HRC) {}
    /// The dynamic reference count is the standard reference count as of OpenMP
    /// 4.5.  The hold reference count is an OpenMP extension for the sake of
    /// OpenACC support.
    ///
    /// The 'ompx_hold' map type modifier is permitted only on "omp target" and
    /// "omp target data", and "delete" is permitted only on "omp target exit
    /// data" and associated runtime library routines.  As a result, we really
    /// need to implement "reset" functionality only for the dynamic reference
    /// counter.  Likewise, only the dynamic reference count can be infinite
    /// because, for example, omp_target_associate_ptr and "omp declare target
    /// link" operate only on it.  Nevertheless, it's actually easier to follow
    /// the code (and requires less assertions for special cases) when we just
    /// implement these features generally across both reference counters here.
    /// Thus, it's the users of this class that impose those restrictions.
    ///
    uint64_t DynRefCount;
    uint64_t HoldRefCount;

    /// A map of shadow pointers associated with this entry, the keys are host
    /// pointer addresses to identify stale entries.
    llvm::SmallSet<ShadowPtrInfoTy, 2> ShadowPtrInfos;

    /// Pointer to the event corresponding to the data update of this map.
    /// Note: At present this event is created when the first data transfer from
    /// host to device is issued, and only being used for H2D. It is not used
    /// for data transfer in another direction (device to host). It is still
    /// unclear whether we need it for D2H. If in the future we need similar
    /// mechanism for D2H, and if the event cannot be shared between them, Event
    /// should be written as <tt>void *Event[2]</tt>.
    void *Event = nullptr;

    /// Number of threads currently holding a reference to the entry at a
    /// targetDataEnd. This is used to ensure that only the last thread that
    /// references this entry will actually delete it.
    int32_t DataEndThreadCount = 0;
  };
  // When HostDataToTargetTy is used by std::set, std::set::iterator is const
  // use unique_ptr to make States mutable.
  const std::unique_ptr<StatesTy> States;

public:
  HostDataToTargetTy(uintptr_t BP, uintptr_t B, uintptr_t E,
                     uintptr_t TgtAllocBegin, uintptr_t TgtPtrBegin,
                     bool UseHoldRefCount, map_var_info_t Name = nullptr,
                     bool IsINF = false)
      : HstPtrBase(BP), HstPtrBegin(B), HstPtrEnd(E), HstPtrName(Name),
        TgtAllocBegin(TgtAllocBegin), TgtPtrBegin(TgtPtrBegin),
        States(std::make_unique<StatesTy>(UseHoldRefCount ? 0
                                          : IsINF         ? INFRefCount
                                                          : 1,
                                          !UseHoldRefCount ? 0
                                          : IsINF          ? INFRefCount
                                                           : 1)) {}

  /// Get the total reference count.  This is smarter than just getDynRefCount()
  /// + getHoldRefCount() because it handles the case where at least one is
  /// infinity and the other is non-zero.
  uint64_t getTotalRefCount() const {
    if (States->DynRefCount == INFRefCount ||
        States->HoldRefCount == INFRefCount)
      return INFRefCount;
    return States->DynRefCount + States->HoldRefCount;
  }

  /// Get the dynamic reference count.
  uint64_t getDynRefCount() const { return States->DynRefCount; }

  /// Get the hold reference count.
  uint64_t getHoldRefCount() const { return States->HoldRefCount; }

  /// Get the event bound to this data map.
  void *getEvent() const { return States->Event; }

  /// Add a new event, if necessary.
  /// Returns OFFLOAD_FAIL if something went wrong, OFFLOAD_SUCCESS otherwise.
  int addEventIfNecessary(DeviceTy &Device, AsyncInfoTy &AsyncInfo) const;

  /// Functions that manages the number of threads referencing the entry in a
  /// targetDataEnd.
  void incDataEndThreadCount() { ++States->DataEndThreadCount; }

  [[nodiscard]] int32_t decDataEndThreadCount() {
    return --States->DataEndThreadCount;
  }

  [[nodiscard]] int32_t getDataEndThreadCount() const {
    return States->DataEndThreadCount;
  }

  /// Set the event bound to this data map.
  void setEvent(void *Event) const { States->Event = Event; }

  /// Reset the specified reference count unless it's infinity.  Reset to 1
  /// (even if currently 0) so it can be followed by a decrement.
  void resetRefCount(bool UseHoldRefCount) const {
    uint64_t &ThisRefCount =
        UseHoldRefCount ? States->HoldRefCount : States->DynRefCount;
    if (ThisRefCount != INFRefCount)
      ThisRefCount = 1;
  }

  /// Increment the specified reference count unless it's infinity.
  void incRefCount(bool UseHoldRefCount) const {
    uint64_t &ThisRefCount =
        UseHoldRefCount ? States->HoldRefCount : States->DynRefCount;
    if (ThisRefCount != INFRefCount) {
      ++ThisRefCount;
      assert(ThisRefCount < INFRefCount && "refcount overflow");
    }
  }

  /// Decrement the specified reference count unless it's infinity or zero, and
  /// return the total reference count.
  uint64_t decRefCount(bool UseHoldRefCount) const {
    uint64_t &ThisRefCount =
        UseHoldRefCount ? States->HoldRefCount : States->DynRefCount;
    uint64_t OtherRefCount =
        UseHoldRefCount ? States->DynRefCount : States->HoldRefCount;
    (void)OtherRefCount;
    if (ThisRefCount != INFRefCount) {
      if (ThisRefCount > 0)
        --ThisRefCount;
      else
        assert(OtherRefCount >= 0 && "total refcount underflow");
    }
    return getTotalRefCount();
  }

  /// Is the dynamic (and thus the total) reference count infinite?
  bool isDynRefCountInf() const { return States->DynRefCount == INFRefCount; }

  /// Convert the dynamic reference count to a debug string.
  std::string dynRefCountToStr() const {
    return refCountToStr(States->DynRefCount);
  }

  /// Convert the hold reference count to a debug string.
  std::string holdRefCountToStr() const {
    return refCountToStr(States->HoldRefCount);
  }

  /// Should one decrement of the specified reference count (after resetting it
  /// if \c AfterReset) remove this mapping?
  bool decShouldRemove(bool UseHoldRefCount, bool AfterReset = false) const {
    uint64_t ThisRefCount =
        UseHoldRefCount ? States->HoldRefCount : States->DynRefCount;
    uint64_t OtherRefCount =
        UseHoldRefCount ? States->DynRefCount : States->HoldRefCount;
    if (OtherRefCount > 0)
      return false;
    if (AfterReset)
      return ThisRefCount != INFRefCount;
    return ThisRefCount == 1;
  }

  /// Add the shadow pointer info \p ShadowPtrInfo to this entry but only if the
  /// the target ptr value was not already present in the existing set of shadow
  /// pointers. Return true if something was added.
  bool addShadowPointer(const ShadowPtrInfoTy &ShadowPtrInfo) const {
    auto Pair = States->ShadowPtrInfos.insert(ShadowPtrInfo);
    if (Pair.second)
      return true;
    // Check for a stale entry, if found, replace the old one.
    if ((*Pair.first).TgtPtrVal == ShadowPtrInfo.TgtPtrVal)
      return false;
    States->ShadowPtrInfos.erase(ShadowPtrInfo);
    return addShadowPointer(ShadowPtrInfo);
  }

  /// Apply \p CB to all shadow pointers of this entry. Returns OFFLOAD_FAIL if
  /// \p CB returned OFFLOAD_FAIL for any of them, otherwise this returns
  /// OFFLOAD_SUCCESS. The entry is locked for this operation.
  template <typename CBTy> int foreachShadowPointerInfo(CBTy CB) const {
    for (auto &It : States->ShadowPtrInfos)
      if (CB(const_cast<ShadowPtrInfoTy &>(It)) == OFFLOAD_FAIL)
        return OFFLOAD_FAIL;
    return OFFLOAD_SUCCESS;
  }

  /// Lock this entry for exclusive access. Ensure to get exclusive access to
  /// HDTTMap first!
  void lock() const { Mtx.lock(); }

  /// Unlock this entry to allow other threads inspecting it.
  void unlock() const { Mtx.unlock(); }

private:
  // Mutex that needs to be held before the entry is inspected or modified. The
  // HDTTMap mutex needs to be held before trying to lock any HDTT Entry.
  mutable std::mutex Mtx;
};

/// Wrapper around the HostDataToTargetTy to be used in the HDTT map. In
/// addition to the HDTT pointer we store the key value explicitly. This
/// allows the set to inspect (sort/search/...) this entry without an additional
/// load of HDTT. HDTT is a pointer to allow the modification of the set without
/// invalidating HDTT entries which can now be inspected at the same time.
struct HostDataToTargetMapKeyTy {
  uintptr_t KeyValue;

  HostDataToTargetMapKeyTy(void *Key) : KeyValue(uintptr_t(Key)) {}
  HostDataToTargetMapKeyTy(uintptr_t Key) : KeyValue(Key) {}
  HostDataToTargetMapKeyTy(HostDataToTargetTy *HDTT)
      : KeyValue(HDTT->HstPtrBegin), HDTT(HDTT) {}
  HostDataToTargetTy *HDTT;
};
inline bool operator<(const HostDataToTargetMapKeyTy &LHS,
                      const uintptr_t &RHS) {
  return LHS.KeyValue < RHS;
}
inline bool operator<(const uintptr_t &LHS,
                      const HostDataToTargetMapKeyTy &RHS) {
  return LHS < RHS.KeyValue;
}
inline bool operator<(const HostDataToTargetMapKeyTy &LHS,
                      const HostDataToTargetMapKeyTy &RHS) {
  return LHS.KeyValue < RHS.KeyValue;
}

/// This struct will be returned by \p DeviceTy::getTargetPointer which provides
/// more data than just a target pointer. A TargetPointerResultTy that has a non
/// null Entry owns the entry. As long as the TargetPointerResultTy (TPR) exists
/// the entry is locked. To give up ownership without destroying the TPR use the
/// reset() function.
struct TargetPointerResultTy {
  struct FlagTy {
    /// If the map table entry is just created
    unsigned IsNewEntry : 1;
    /// If the pointer is actually a host pointer (when unified memory enabled)
    unsigned IsHostPointer : 1;
    /// If the pointer is present in the mapping table.
    unsigned IsPresent : 1;
    /// Flag indicating that this was the last user of the entry and the ref
    /// count is now 0.
    unsigned IsLast : 1;
    /// If the pointer is contained.
    unsigned IsContained : 1;
  } Flags = {0, 0, 0, 0, 0};

  TargetPointerResultTy(const TargetPointerResultTy &) = delete;
  TargetPointerResultTy &operator=(const TargetPointerResultTy &TPR) = delete;
  TargetPointerResultTy() {}

  TargetPointerResultTy(FlagTy Flags, HostDataToTargetTy *Entry,
                        void *TargetPointer)
      : Flags(Flags), TargetPointer(TargetPointer), Entry(Entry) {
    if (Entry)
      Entry->lock();
  }

  TargetPointerResultTy(TargetPointerResultTy &&TPR)
      : Flags(TPR.Flags), TargetPointer(TPR.TargetPointer), Entry(TPR.Entry) {
    TPR.Entry = nullptr;
  }

  TargetPointerResultTy &operator=(TargetPointerResultTy &&TPR) {
    if (&TPR != this) {
      std::swap(Flags, TPR.Flags);
      std::swap(Entry, TPR.Entry);
      std::swap(TargetPointer, TPR.TargetPointer);
    }
    return *this;
  }

  ~TargetPointerResultTy() {
    if (Entry)
      Entry->unlock();
  }

  bool isPresent() const { return Flags.IsPresent; }

  bool isHostPointer() const { return Flags.IsHostPointer; }

  bool isContained() const { return Flags.IsContained; }

  /// The corresponding target pointer
  void *TargetPointer = nullptr;

  HostDataToTargetTy *getEntry() const { return Entry; }
  void setEntry(HostDataToTargetTy *HDTTT,
                HostDataToTargetTy *OwnedTPR = nullptr) {
    if (Entry)
      Entry->unlock();
    Entry = HDTTT;
    if (Entry && Entry != OwnedTPR)
      Entry->lock();
  }

  void reset() { *this = TargetPointerResultTy(); }

private:
  /// The corresponding map table entry which is stable.
  HostDataToTargetTy *Entry = nullptr;
};

struct LookupResult {
  struct {
    unsigned IsContained : 1;
    unsigned ExtendsBefore : 1;
    unsigned ExtendsAfter : 1;
  } Flags;

  LookupResult() : Flags({0, 0, 0}), TPR() {}

  TargetPointerResultTy TPR;
};

// This structure stores information of a mapped memory region.
struct MapComponentInfoTy {
  void *Base;
  void *Begin;
  int64_t Size;
  int64_t Type;
  void *Name;
  MapComponentInfoTy() = default;
  MapComponentInfoTy(void *Base, void *Begin, int64_t Size, int64_t Type,
                     void *Name)
      : Base(Base), Begin(Begin), Size(Size), Type(Type), Name(Name) {}
};

// This structure stores all components of a user-defined mapper. The number of
// components are dynamically decided, so we utilize C++ STL vector
// implementation here.
struct MapperComponentsTy {
  llvm::SmallVector<MapComponentInfoTy> Components;
  int32_t size() { return Components.size(); }
};

// The mapper function pointer type. It follows the signature below:
// void .omp_mapper.<type_name>.<mapper_id>.(void *rt_mapper_handle,
//                                           void *base, void *begin,
//                                           size_t size, int64_t type,
//                                           void * name);
typedef void (*MapperFuncPtrTy)(void *, void *, void *, int64_t, int64_t,
                                void *);

/// Structure to store information about a single ATTACH map entry.
struct AttachMapInfo {
  void *PointerBase;
  void *PointeeBegin;
  int64_t PointerSize;
  int64_t MapType;
  map_var_info_t Pointername;

  AttachMapInfo(void *PointerBase, void *PointeeBegin, int64_t Size,
                int64_t Type, map_var_info_t Name)
      : PointerBase(PointerBase), PointeeBegin(PointeeBegin), PointerSize(Size),
        MapType(Type), Pointername(Name) {}
};

/// Structure to track ATTACH entries and new allocations across recursive calls
/// (for handling mappers) to targetDataBegin for a given construct.
struct AttachInfoTy {
  /// ATTACH map entries for deferred processing.
  llvm::SmallVector<AttachMapInfo> AttachEntries;

  /// Key: host pointer, Value: allocation size.
  llvm::DenseMap<void *, int64_t> NewAllocations;

  AttachInfoTy() = default;

  // Delete copy constructor and copy assignment operator to prevent copying
  AttachInfoTy(const AttachInfoTy &) = delete;
  AttachInfoTy &operator=(const AttachInfoTy &) = delete;
};

// Function pointer type for targetData* functions (targetDataBegin,
// targetDataEnd and targetDataUpdate).
typedef int (*TargetDataFuncPtrTy)(ident_t *, DeviceTy &, int32_t, void **,
                                   void **, int64_t *, int64_t *,
                                   map_var_info_t *, void **, AsyncInfoTy &,
                                   AttachInfoTy *, bool);

void dumpTargetPointerMappings(const ident_t *Loc, DeviceTy &Device,
                               bool toStdOut = false);

int targetDataBegin(ident_t *Loc, DeviceTy &Device, int32_t ArgNum,
                    void **ArgsBase, void **Args, int64_t *ArgSizes,
                    int64_t *ArgTypes, map_var_info_t *ArgNames,
                    void **ArgMappers, AsyncInfoTy &AsyncInfo,
                    AttachInfoTy *AttachInfo = nullptr,
                    bool FromMapper = false);

int targetDataEnd(ident_t *Loc, DeviceTy &Device, int32_t ArgNum,
                  void **ArgBases, void **Args, int64_t *ArgSizes,
                  int64_t *ArgTypes, map_var_info_t *ArgNames,
                  void **ArgMappers, AsyncInfoTy &AsyncInfo,
                  AttachInfoTy *AttachInfo = nullptr, bool FromMapper = false);

int targetDataUpdate(ident_t *Loc, DeviceTy &Device, int32_t ArgNum,
                     void **ArgsBase, void **Args, int64_t *ArgSizes,
                     int64_t *ArgTypes, map_var_info_t *ArgNames,
                     void **ArgMappers, AsyncInfoTy &AsyncInfo,
                     AttachInfoTy *AttachInfo = nullptr,
                     bool FromMapper = false);

// Process deferred ATTACH map entries collected during targetDataBegin.
int processAttachEntries(DeviceTy &Device, AttachInfoTy &AttachInfo,
                         AsyncInfoTy &AsyncInfo);

struct MappingInfoTy {
  MappingInfoTy(DeviceTy &Device) : Device(Device) {}

  /// Host data to device map type with a wrapper key indirection that allows
  /// concurrent modification of the entries without invalidating the underlying
  /// entries.
  using HostDataToTargetListTy =
      std::set<HostDataToTargetMapKeyTy, std::less<>>;

  /// The HDTTMap is a protected object that can only be accessed by one thread
  /// at a time.
  ProtectedObj<HostDataToTargetListTy> HostDataToTargetMap;

  /// The type used to access the HDTT map.
  using HDTTMapAccessorTy = decltype(HostDataToTargetMap)::AccessorTy;

  /// Lookup the mapping of \p HstPtrBegin in \p HDTTMap. The accessor ensures
  /// exclusive access to the HDTT map.
  LookupResult lookupMapping(HDTTMapAccessorTy &HDTTMap, void *HstPtrBegin,
                             int64_t Size,
                             HostDataToTargetTy *OwnedTPR = nullptr);

  /// Get the target pointer based on host pointer begin and base. If the
  /// mapping already exists, the target pointer will be returned directly. In
  /// addition, if required, the memory region pointed by \p HstPtrBegin of size
  /// \p Size will also be transferred to the device. If the mapping doesn't
  /// exist, and if unified shared memory is not enabled, a new mapping will be
  /// created and the data will also be transferred accordingly. nullptr will be
  /// returned because of any of following reasons:
  /// - Data allocation failed;
  /// - The user tried to do an illegal mapping;
  /// - Data transfer issue fails.
  TargetPointerResultTy getTargetPointer(
      HDTTMapAccessorTy &HDTTMap, void *HstPtrBegin, void *HstPtrBase,
      int64_t TgtPadding, int64_t Size, map_var_info_t HstPtrName,
      bool HasFlagTo, bool HasFlagAlways, bool IsImplicit, bool UpdateRefCount,
      bool HasCloseModifier, bool HasPresentModifier, bool HasHoldModifier,
      AsyncInfoTy &AsyncInfo, HostDataToTargetTy *OwnedTPR = nullptr,
      bool ReleaseHDTTMap = true);

  /// Return the target pointer for \p HstPtrBegin in \p HDTTMap. The accessor
  /// ensures exclusive access to the HDTT map.
  void *getTgtPtrBegin(HDTTMapAccessorTy &HDTTMap, void *HstPtrBegin,
                       int64_t Size);

  /// Return the target pointer begin (where the data will be moved).
  /// Used by targetDataBegin, targetDataEnd, targetDataUpdate and target.
  /// - \p UpdateRefCount and \p UseHoldRefCount controls which and if the entry
  /// reference counters will be decremented.
  /// - \p MustContain enforces that the query must not extend beyond an already
  /// mapped entry to be valid.
  /// - \p ForceDelete deletes the entry regardless of its reference counting
  /// (unless it is infinite).
  /// - \p FromDataEnd tracks the number of threads referencing the entry at
  /// targetDataEnd for delayed deletion purpose.
  [[nodiscard]] TargetPointerResultTy
  getTgtPtrBegin(void *HstPtrBegin, int64_t Size, bool UpdateRefCount,
                 bool UseHoldRefCount, bool MustContain = false,
                 bool ForceDelete = false, bool FromDataEnd = false);

  /// Remove the \p Entry from the data map. Expect the entry's total reference
  /// count to be zero and the caller thread to be the last one using it. \p
  /// HDTTMap ensure the caller holds exclusive access and can modify the map.
  /// Return \c OFFLOAD_SUCCESS if the map entry existed, and return \c
  /// OFFLOAD_FAIL if not. It is the caller's responsibility to skip calling
  /// this function if the map entry is not expected to exist because \p
  /// HstPtrBegin uses shared memory.
  [[nodiscard]] int eraseMapEntry(HDTTMapAccessorTy &HDTTMap,
                                  HostDataToTargetTy *Entry, int64_t Size);

  /// Deallocate the \p Entry from the device memory and delete it. Return \c
  /// OFFLOAD_SUCCESS if the deallocation operations executed successfully, and
  /// return \c OFFLOAD_FAIL otherwise.
  [[nodiscard]] int deallocTgtPtrAndEntry(HostDataToTargetTy *Entry,
                                          int64_t Size);

  int associatePtr(void *HstPtrBegin, void *TgtPtrBegin, int64_t Size);
  int disassociatePtr(void *HstPtrBegin);

  /// Print information about the transfer from \p HstPtr to \p TgtPtr (or vice
  /// versa if \p H2D is false). If there is an existing mapping, or if \p Entry
  /// is set, the associated metadata will be printed as well.
  void printCopyInfo(void *TgtPtr, void *HstPtr, int64_t Size, bool H2D,
                     HostDataToTargetTy *Entry,
                     MappingInfoTy::HDTTMapAccessorTy *HDTTMapPtr);

private:
  DeviceTy &Device;
};

#endif // OMPTARGET_OPENMP_MAPPING_H
