//===----------- device.h - Target independent OpenMP target RTL ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarations for managing devices that are handled by RTL plugins.
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_DEVICE_H
#define _OMPTARGET_DEVICE_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>

#include "ExclusiveAccess.h"
#include "omptarget.h"
#include "rtl.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

// Forward declarations.
struct RTLInfoTy;
struct __tgt_bin_desc;
struct __tgt_target_table;

using map_var_info_t = void *;

// enum for OMP_TARGET_OFFLOAD; keep in sync with kmp.h definition
enum kmp_target_offload_kind {
  tgt_disabled = 0,
  tgt_default = 1,
  tgt_mandatory = 2
};
typedef enum kmp_target_offload_kind kmp_target_offload_kind_t;

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

///
struct PendingCtorDtorListsTy {
  std::list<void *> PendingCtors;
  std::list<void *> PendingDtors;
};
typedef std::map<__tgt_bin_desc *, PendingCtorDtorListsTy>
    PendingCtorsDtorsPerLibrary;

struct DeviceTy {
  int32_t DeviceID;
  RTLInfoTy *RTL;
  int32_t RTLDeviceID;

  bool IsInit;
  std::once_flag InitFlag;
  bool HasPendingGlobals;

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

  PendingCtorsDtorsPerLibrary PendingCtorsDtors;

  std::mutex PendingGlobalsMtx;

  DeviceTy(RTLInfoTy *RTL);
  // DeviceTy is not copyable
  DeviceTy(const DeviceTy &D) = delete;
  DeviceTy &operator=(const DeviceTy &D) = delete;

  ~DeviceTy();

  // Return true if data can be copied to DstDevice directly
  bool isDataExchangable(const DeviceTy &DstDevice);

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

  // calls to RTL
  int32_t initOnce();
  __tgt_target_table *loadBinary(void *Img);

  // device memory allocation/deallocation routines
  /// Allocates \p Size bytes on the device, host or shared memory space
  /// (depending on \p Kind) and returns the address/nullptr when
  /// succeeds/fails. \p HstPtr is an address of the host data which the
  /// allocated target data will be associated with. If it is unknown, the
  /// default value of \p HstPtr is nullptr. Note: this function doesn't do
  /// pointer association. Actually, all the __tgt_rtl_data_alloc
  /// implementations ignore \p HstPtr. \p Kind dictates what allocator should
  /// be used (host, shared, device).
  void *allocData(int64_t Size, void *HstPtr = nullptr,
                  int32_t Kind = TARGET_ALLOC_DEFAULT);
  /// Deallocates memory which \p TgtPtrBegin points at and returns
  /// OFFLOAD_SUCCESS/OFFLOAD_FAIL when succeeds/fails. p Kind dictates what
  /// allocator should be used (host, shared, device).
  int32_t deleteData(void *TgtPtrBegin, int32_t Kind = TARGET_ALLOC_DEFAULT);

  // Data transfer. When AsyncInfo is nullptr, the transfer will be
  // synchronous.
  // Copy data from host to device
  int32_t submitData(void *TgtPtrBegin, void *HstPtrBegin, int64_t Size,
                     AsyncInfoTy &AsyncInfo,
                     HostDataToTargetTy *Entry = nullptr);
  // Copy data from device back to host
  int32_t retrieveData(void *HstPtrBegin, void *TgtPtrBegin, int64_t Size,
                       AsyncInfoTy &AsyncInfo,
                       HostDataToTargetTy *Entry = nullptr);
  // Copy data from current device to destination device directly
  int32_t dataExchange(void *SrcPtr, DeviceTy &DstDev, void *DstPtr,
                       int64_t Size, AsyncInfoTy &AsyncInfo);

  /// Notify the plugin about a new mapping starting at the host address
  /// \p HstPtr and \p Size bytes.
  int32_t notifyDataMapped(void *HstPtr, int64_t Size);

  /// Notify the plugin about an existing mapping being unmapped starting at
  /// the host address \p HstPtr.
  int32_t notifyDataUnmapped(void *HstPtr);

  // Launch the kernel identified by \p TgtEntryPtr with the given arguments.
  int32_t launchKernel(void *TgtEntryPtr, void **TgtVarsPtr,
                       ptrdiff_t *TgtOffsets, const KernelArgsTy &KernelArgs,
                       AsyncInfoTy &AsyncInfo);

  /// Synchronize device/queue/event based on \p AsyncInfo and return
  /// OFFLOAD_SUCCESS/OFFLOAD_FAIL when succeeds/fails.
  int32_t synchronize(AsyncInfoTy &AsyncInfo);

  /// Query for device/queue/event based completion on \p AsyncInfo in a
  /// non-blocking manner and return OFFLOAD_SUCCESS/OFFLOAD_FAIL when
  /// succeeds/fails. Must be called multiple times until AsyncInfo is
  /// completed and AsyncInfo.isDone() returns true.
  int32_t queryAsync(AsyncInfoTy &AsyncInfo);

  /// Calls the corresponding print in the \p RTLDEVID
  /// device RTL to obtain the information of the specific device.
  bool printDeviceInfo(int32_t RTLDevID);

  /// Event related interfaces.
  /// {
  /// Create an event.
  int32_t createEvent(void **Event);

  /// Record the event based on status in AsyncInfo->Queue at the moment the
  /// function is called.
  int32_t recordEvent(void *Event, AsyncInfoTy &AsyncInfo);

  /// Wait for an event. This function can be blocking or non-blocking,
  /// depending on the implmentation. It is expected to set a dependence on the
  /// event such that corresponding operations shall only start once the event
  /// is fulfilled.
  int32_t waitEvent(void *Event, AsyncInfoTy &AsyncInfo);

  /// Synchronize the event. It is expected to block the thread.
  int32_t syncEvent(void *Event);

  /// Destroy the event.
  int32_t destroyEvent(void *Event);
  /// }

private:
  // Call to RTL
  void init(); // To be called only via DeviceTy::initOnce()

  /// Deinitialize the device (and plugin).
  void deinit();
};

extern bool deviceIsReady(int DeviceNum);

/// Struct for the data required to handle plugins
struct PluginManager {
  PluginManager(bool UseEventsForAtomicTransfers)
      : UseEventsForAtomicTransfers(UseEventsForAtomicTransfers) {}

  /// RTLs identified on the host
  RTLsTy RTLs;

  /// Executable images and information extracted from the input images passed
  /// to the runtime.
  std::list<std::pair<__tgt_device_image, __tgt_image_info>> Images;

  /// Devices associated with RTLs
  llvm::SmallVector<std::unique_ptr<DeviceTy>> Devices;
  std::mutex RTLsMtx; ///< For RTLs and Devices

  /// Translation table retreived from the binary
  HostEntriesBeginToTransTableTy HostEntriesBeginToTransTable;
  std::mutex TrlTblMtx; ///< For Translation Table
  /// Host offload entries in order of image registration
  llvm::SmallVector<__tgt_offload_entry *> HostEntriesBeginRegistrationOrder;

  /// Map from ptrs on the host to an entry in the Translation Table
  HostPtrToTableMapTy HostPtrToTableMap;
  std::mutex TblMapMtx; ///< For HostPtrToTableMap

  // Store target policy (disabled, mandatory, default)
  kmp_target_offload_kind_t TargetOffloadPolicy = tgt_default;
  std::mutex TargetOffloadMtx; ///< For TargetOffloadPolicy

  /// Flag to indicate if we use events to ensure the atomicity of
  /// map clauses or not. Can be modified with an environment variable.
  const bool UseEventsForAtomicTransfers;

  // Work around for plugins that call dlopen on shared libraries that call
  // tgt_register_lib during their initialisation. Stash the pointers in a
  // vector until the plugins are all initialised and then register them.
  bool maybeDelayRegisterLib(__tgt_bin_desc *Desc) {
    if (!RTLsLoaded) {
      // Only reachable from libomptarget constructor
      DelayedBinDesc.push_back(Desc);
      return true;
    } else {
      return false;
    }
  }

  void registerDelayedLibraries() {
    // Only called by libomptarget constructor
    RTLsLoaded = true;
    for (auto *Desc : DelayedBinDesc)
      __tgt_register_lib(Desc);
    DelayedBinDesc.clear();
  }

private:
  bool RTLsLoaded = false;
  llvm::SmallVector<__tgt_bin_desc *> DelayedBinDesc;
};

extern PluginManager *PM;

#endif
