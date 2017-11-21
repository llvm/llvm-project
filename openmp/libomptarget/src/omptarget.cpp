//===------ omptarget.cpp - Target independent OpenMP target RTL -- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the interface to be used by Clang during the codegen of a
// target region.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <list>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// Header file global to this project
#include "omptarget.h"

#ifdef OMPTARGET_DEBUG
static int DebugLevel = 0;

#define DP(...) \
  do { \
    if (DebugLevel > 0) { \
      DEBUGP("Libomptarget", __VA_ARGS__); \
    } \
  } while (false)
#else // OMPTARGET_DEBUG
#define DP(...) {}
#endif // OMPTARGET_DEBUG

#define INF_REF_CNT (LONG_MAX>>1) // leave room for additions/subtractions
#define CONSIDERED_INF(x) (x > (INF_REF_CNT>>1))

// List of all plugins that can support offloading.
static const char *RTLNames[] = {
    /* PowerPC target */ "libomptarget.rtl.ppc64.so",
    /* x86_64 target  */ "libomptarget.rtl.x86_64.so",
    /* CUDA target    */ "libomptarget.rtl.cuda.so",
    /* AArch64 target */ "libomptarget.rtl.aarch64.so"};

// forward declarations
struct RTLInfoTy;
static int target(int64_t device_id, void *host_ptr, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    int32_t team_num, int32_t thread_limit, int IsTeamConstruct);

/// Map between host data and target data.
struct HostDataToTargetTy {
  uintptr_t HstPtrBase; // host info.
  uintptr_t HstPtrBegin;
  uintptr_t HstPtrEnd; // non-inclusive.

  uintptr_t TgtPtrBegin; // target info.

  long RefCount;

  HostDataToTargetTy()
      : HstPtrBase(0), HstPtrBegin(0), HstPtrEnd(0),
        TgtPtrBegin(0), RefCount(0) {}
  HostDataToTargetTy(uintptr_t BP, uintptr_t B, uintptr_t E, uintptr_t TB)
      : HstPtrBase(BP), HstPtrBegin(B), HstPtrEnd(E),
        TgtPtrBegin(TB), RefCount(1) {}
  HostDataToTargetTy(uintptr_t BP, uintptr_t B, uintptr_t E, uintptr_t TB,
      long RF)
      : HstPtrBase(BP), HstPtrBegin(B), HstPtrEnd(E),
        TgtPtrBegin(TB), RefCount(RF) {}
};

typedef std::list<HostDataToTargetTy> HostDataToTargetListTy;

struct LookupResult {
  struct {
    unsigned IsContained   : 1;
    unsigned ExtendsBefore : 1;
    unsigned ExtendsAfter  : 1;
  } Flags;

  HostDataToTargetListTy::iterator Entry;

  LookupResult() : Flags({0,0,0}), Entry() {}
};

/// Map for shadow pointers
struct ShadowPtrValTy {
  void *HstPtrVal;
  void *TgtPtrAddr;
  void *TgtPtrVal;
};
typedef std::map<void *, ShadowPtrValTy> ShadowPtrListTy;

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

  HostDataToTargetListTy HostDataToTargetMap;
  PendingCtorsDtorsPerLibrary PendingCtorsDtors;

  ShadowPtrListTy ShadowPtrMap;

  std::mutex DataMapMtx, PendingGlobalsMtx, ShadowMtx;

  uint64_t loopTripCnt;

  DeviceTy(RTLInfoTy *RTL)
      : DeviceID(-1), RTL(RTL), RTLDeviceID(-1), IsInit(false), InitFlag(),
        HasPendingGlobals(false), HostDataToTargetMap(),
        PendingCtorsDtors(), ShadowPtrMap(), DataMapMtx(), PendingGlobalsMtx(),
        ShadowMtx(), loopTripCnt(0) {}

  // The existence of mutexes makes DeviceTy non-copyable. We need to
  // provide a copy constructor and an assignment operator explicitly.
  DeviceTy(const DeviceTy &d)
      : DeviceID(d.DeviceID), RTL(d.RTL), RTLDeviceID(d.RTLDeviceID),
        IsInit(d.IsInit), InitFlag(), HasPendingGlobals(d.HasPendingGlobals),
        HostDataToTargetMap(d.HostDataToTargetMap),
        PendingCtorsDtors(d.PendingCtorsDtors), ShadowPtrMap(d.ShadowPtrMap),
        DataMapMtx(), PendingGlobalsMtx(),
        ShadowMtx(), loopTripCnt(d.loopTripCnt) {}

  DeviceTy& operator=(const DeviceTy &d) {
    DeviceID = d.DeviceID;
    RTL = d.RTL;
    RTLDeviceID = d.RTLDeviceID;
    IsInit = d.IsInit;
    HasPendingGlobals = d.HasPendingGlobals;
    HostDataToTargetMap = d.HostDataToTargetMap;
    PendingCtorsDtors = d.PendingCtorsDtors;
    ShadowPtrMap = d.ShadowPtrMap;
    loopTripCnt = d.loopTripCnt;

    return *this;
  }

  long getMapEntryRefCnt(void *HstPtrBegin);
  LookupResult lookupMapping(void *HstPtrBegin, int64_t Size);
  void *getOrAllocTgtPtr(void *HstPtrBegin, void *HstPtrBase, int64_t Size,
      bool &IsNew, bool IsImplicit, bool UpdateRefCount = true);
  void *getTgtPtrBegin(void *HstPtrBegin, int64_t Size);
  void *getTgtPtrBegin(void *HstPtrBegin, int64_t Size, bool &IsLast,
      bool UpdateRefCount);
  int deallocTgtPtr(void *TgtPtrBegin, int64_t Size, bool ForceDelete);
  int associatePtr(void *HstPtrBegin, void *TgtPtrBegin, int64_t Size);
  int disassociatePtr(void *HstPtrBegin);

  // calls to RTL
  int32_t initOnce();
  __tgt_target_table *load_binary(void *Img);

  int32_t data_submit(void *TgtPtrBegin, void *HstPtrBegin, int64_t Size);
  int32_t data_retrieve(void *HstPtrBegin, void *TgtPtrBegin, int64_t Size);

  int32_t run_region(void *TgtEntryPtr, void **TgtVarsPtr,
      ptrdiff_t *TgtOffsets, int32_t TgtVarsSize);
  int32_t run_team_region(void *TgtEntryPtr, void **TgtVarsPtr,
      ptrdiff_t *TgtOffsets, int32_t TgtVarsSize, int32_t NumTeams,
      int32_t ThreadLimit, uint64_t LoopTripCount);

private:
  // Call to RTL
  void init(); // To be called only via DeviceTy::initOnce()
};

/// Map between Device ID (i.e. openmp device id) and its DeviceTy.
typedef std::vector<DeviceTy> DevicesTy;
static DevicesTy Devices;

struct RTLInfoTy {
  typedef int32_t(is_valid_binary_ty)(void *);
  typedef int32_t(number_of_devices_ty)();
  typedef int32_t(init_device_ty)(int32_t);
  typedef __tgt_target_table *(load_binary_ty)(int32_t, void *);
  typedef void *(data_alloc_ty)(int32_t, int64_t, void *);
  typedef int32_t(data_submit_ty)(int32_t, void *, void *, int64_t);
  typedef int32_t(data_retrieve_ty)(int32_t, void *, void *, int64_t);
  typedef int32_t(data_delete_ty)(int32_t, void *);
  typedef int32_t(run_region_ty)(int32_t, void *, void **, ptrdiff_t *,
                                 int32_t);
  typedef int32_t(run_team_region_ty)(int32_t, void *, void **, ptrdiff_t *,
                                      int32_t, int32_t, int32_t, uint64_t);

  int32_t Idx;                     // RTL index, index is the number of devices
                                   // of other RTLs that were registered before,
                                   // i.e. the OpenMP index of the first device
                                   // to be registered with this RTL.
  int32_t NumberOfDevices;         // Number of devices this RTL deals with.
  std::vector<DeviceTy *> Devices; // one per device (NumberOfDevices in total).

  void *LibraryHandler;

#ifdef OMPTARGET_DEBUG
  std::string RTLName;
#endif

  // Functions implemented in the RTL.
  is_valid_binary_ty *is_valid_binary;
  number_of_devices_ty *number_of_devices;
  init_device_ty *init_device;
  load_binary_ty *load_binary;
  data_alloc_ty *data_alloc;
  data_submit_ty *data_submit;
  data_retrieve_ty *data_retrieve;
  data_delete_ty *data_delete;
  run_region_ty *run_region;
  run_team_region_ty *run_team_region;

  // Are there images associated with this RTL.
  bool isUsed;

  // Mutex for thread-safety when calling RTL interface functions.
  // It is easier to enforce thread-safety at the libomptarget level,
  // so that developers of new RTLs do not have to worry about it.
  std::mutex Mtx;

  // The existence of the mutex above makes RTLInfoTy non-copyable.
  // We need to provide a copy constructor explicitly.
  RTLInfoTy()
      : Idx(-1), NumberOfDevices(-1), Devices(), LibraryHandler(0),
#ifdef OMPTARGET_DEBUG
        RTLName(),
#endif
        is_valid_binary(0), number_of_devices(0), init_device(0),
        load_binary(0), data_alloc(0), data_submit(0), data_retrieve(0),
        data_delete(0), run_region(0), run_team_region(0), isUsed(false),
        Mtx() {}

  RTLInfoTy(const RTLInfoTy &r) : Mtx() {
    Idx = r.Idx;
    NumberOfDevices = r.NumberOfDevices;
    Devices = r.Devices;
    LibraryHandler = r.LibraryHandler;
#ifdef OMPTARGET_DEBUG
    RTLName = r.RTLName;
#endif
    is_valid_binary = r.is_valid_binary;
    number_of_devices = r.number_of_devices;
    init_device = r.init_device;
    load_binary = r.load_binary;
    data_alloc = r.data_alloc;
    data_submit = r.data_submit;
    data_retrieve = r.data_retrieve;
    data_delete = r.data_delete;
    run_region = r.run_region;
    run_team_region = r.run_team_region;
    isUsed = r.isUsed;
  }
};

/// RTLs identified in the system.
class RTLsTy {
private:
  // Mutex-like object to guarantee thread-safety and unique initialization
  // (i.e. the library attempts to load the RTLs (plugins) only once).
  std::once_flag initFlag;
  void LoadRTLs(); // not thread-safe

public:
  // List of the detected runtime libraries.
  std::list<RTLInfoTy> AllRTLs;

  // Array of pointers to the detected runtime libraries that have compatible
  // binaries.
  std::vector<RTLInfoTy *> UsedRTLs;

  explicit RTLsTy() {}

  // Load all the runtime libraries (plugins) if not done before.
  void LoadRTLsOnce();
};

void RTLsTy::LoadRTLs() {
#ifdef OMPTARGET_DEBUG
  if (char *envStr = getenv("LIBOMPTARGET_DEBUG")) {
    DebugLevel = std::stoi(envStr);
  }
#endif // OMPTARGET_DEBUG

  // Parse environment variable OMP_TARGET_OFFLOAD (if set)
  char *envStr = getenv("OMP_TARGET_OFFLOAD");
  if (envStr && !strcmp(envStr, "DISABLED")) {
    DP("Target offloading disabled by environment\n");
    return;
  }

  DP("Loading RTLs...\n");

  // Attempt to open all the plugins and, if they exist, check if the interface
  // is correct and if they are supporting any devices.
  for (auto *Name : RTLNames) {
    DP("Loading library '%s'...\n", Name);
    void *dynlib_handle = dlopen(Name, RTLD_NOW);

    if (!dynlib_handle) {
      // Library does not exist or cannot be found.
      DP("Unable to load library '%s': %s!\n", Name, dlerror());
      continue;
    }

    DP("Successfully loaded library '%s'!\n", Name);

    // Retrieve the RTL information from the runtime library.
    RTLInfoTy R;

    R.LibraryHandler = dynlib_handle;
    R.isUsed = false;

#ifdef OMPTARGET_DEBUG
    R.RTLName = Name;
#endif

    if (!(*((void**) &R.is_valid_binary) = dlsym(
              dynlib_handle, "__tgt_rtl_is_valid_binary")))
      continue;
    if (!(*((void**) &R.number_of_devices) = dlsym(
              dynlib_handle, "__tgt_rtl_number_of_devices")))
      continue;
    if (!(*((void**) &R.init_device) = dlsym(
              dynlib_handle, "__tgt_rtl_init_device")))
      continue;
    if (!(*((void**) &R.load_binary) = dlsym(
              dynlib_handle, "__tgt_rtl_load_binary")))
      continue;
    if (!(*((void**) &R.data_alloc) = dlsym(
              dynlib_handle, "__tgt_rtl_data_alloc")))
      continue;
    if (!(*((void**) &R.data_submit) = dlsym(
              dynlib_handle, "__tgt_rtl_data_submit")))
      continue;
    if (!(*((void**) &R.data_retrieve) = dlsym(
              dynlib_handle, "__tgt_rtl_data_retrieve")))
      continue;
    if (!(*((void**) &R.data_delete) = dlsym(
              dynlib_handle, "__tgt_rtl_data_delete")))
      continue;
    if (!(*((void**) &R.run_region) = dlsym(
              dynlib_handle, "__tgt_rtl_run_target_region")))
      continue;
    if (!(*((void**) &R.run_team_region) = dlsym(
              dynlib_handle, "__tgt_rtl_run_target_team_region")))
      continue;

    // No devices are supported by this RTL?
    if (!(R.NumberOfDevices = R.number_of_devices())) {
      DP("No devices supported in this RTL\n");
      continue;
    }

    DP("Registering RTL %s supporting %d devices!\n",
        R.RTLName.c_str(), R.NumberOfDevices);

    // The RTL is valid! Will save the information in the RTLs list.
    AllRTLs.push_back(R);
  }

  DP("RTLs loaded!\n");

  return;
}

void RTLsTy::LoadRTLsOnce() {
  // RTL.LoadRTLs() is called only once in a thread-safe fashion.
  std::call_once(initFlag, &RTLsTy::LoadRTLs, this);
}

static RTLsTy RTLs;
static std::mutex RTLsMtx;

/// Map between the host entry begin and the translation table. Each
/// registered library gets one TranslationTable. Use the map from
/// __tgt_offload_entry so that we may quickly determine whether we
/// are trying to (re)register an existing lib or really have a new one.
struct TranslationTable {
  __tgt_target_table HostTable;

  // Image assigned to a given device.
  std::vector<__tgt_device_image *> TargetsImages; // One image per device ID.

  // Table of entry points or NULL if it was not already computed.
  std::vector<__tgt_target_table *> TargetsTable; // One table per device ID.
};
typedef std::map<__tgt_offload_entry *, TranslationTable>
    HostEntriesBeginToTransTableTy;
static HostEntriesBeginToTransTableTy HostEntriesBeginToTransTable;
static std::mutex TrlTblMtx;

/// Map between the host ptr and a table index
struct TableMap {
  TranslationTable *Table; // table associated with the host ptr.
  uint32_t Index; // index in which the host ptr translated entry is found.
  TableMap() : Table(0), Index(0) {}
  TableMap(TranslationTable *table, uint32_t index)
      : Table(table), Index(index) {}
};
typedef std::map<void *, TableMap> HostPtrToTableMapTy;
static HostPtrToTableMapTy HostPtrToTableMap;
static std::mutex TblMapMtx;

/// Check whether a device has an associated RTL and initialize it if it's not
/// already initialized.
static bool device_is_ready(int device_num) {
  DP("Checking whether device %d is ready.\n", device_num);
  // Devices.size() can only change while registering a new
  // library, so try to acquire the lock of RTLs' mutex.
  RTLsMtx.lock();
  size_t Devices_size = Devices.size();
  RTLsMtx.unlock();
  if (Devices_size <= (size_t)device_num) {
    DP("Device ID  %d does not have a matching RTL\n", device_num);
    return false;
  }

  // Get device info
  DeviceTy &Device = Devices[device_num];

  DP("Is the device %d (local ID %d) initialized? %d\n", device_num,
       Device.RTLDeviceID, Device.IsInit);

  // Init the device if not done before
  if (!Device.IsInit && Device.initOnce() != OFFLOAD_SUCCESS) {
    DP("Failed to init device %d\n", device_num);
    return false;
  }

  DP("Device %d is ready to use.\n", device_num);

  return true;
}

////////////////////////////////////////////////////////////////////////////////
// Target API functions
//
EXTERN int omp_get_num_devices(void) {
  RTLsMtx.lock();
  size_t Devices_size = Devices.size();
  RTLsMtx.unlock();

  DP("Call to omp_get_num_devices returning %zd\n", Devices_size);

  return Devices_size;
}

EXTERN int omp_get_initial_device(void) {
  DP("Call to omp_get_initial_device returning %d\n", HOST_DEVICE);
  return HOST_DEVICE;
}

EXTERN void *omp_target_alloc(size_t size, int device_num) {
  DP("Call to omp_target_alloc for device %d requesting %zu bytes\n",
      device_num, size);

  if (size <= 0) {
    DP("Call to omp_target_alloc with non-positive length\n");
    return NULL;
  }

  void *rc = NULL;

  if (device_num == omp_get_initial_device()) {
    rc = malloc(size);
    DP("omp_target_alloc returns host ptr " DPxMOD "\n", DPxPTR(rc));
    return rc;
  }

  if (!device_is_ready(device_num)) {
    DP("omp_target_alloc returns NULL ptr\n");
    return NULL;
  }

  DeviceTy &Device = Devices[device_num];
  rc = Device.RTL->data_alloc(Device.RTLDeviceID, size, NULL);
  DP("omp_target_alloc returns device ptr " DPxMOD "\n", DPxPTR(rc));
  return rc;
}

EXTERN void omp_target_free(void *device_ptr, int device_num) {
  DP("Call to omp_target_free for device %d and address " DPxMOD "\n",
      device_num, DPxPTR(device_ptr));

  if (!device_ptr) {
    DP("Call to omp_target_free with NULL ptr\n");
    return;
  }

  if (device_num == omp_get_initial_device()) {
    free(device_ptr);
    DP("omp_target_free deallocated host ptr\n");
    return;
  }

  if (!device_is_ready(device_num)) {
    DP("omp_target_free returns, nothing to do\n");
    return;
  }

  DeviceTy &Device = Devices[device_num];
  Device.RTL->data_delete(Device.RTLDeviceID, (void *)device_ptr);
  DP("omp_target_free deallocated device ptr\n");
}

EXTERN int omp_target_is_present(void *ptr, int device_num) {
  DP("Call to omp_target_is_present for device %d and address " DPxMOD "\n",
      device_num, DPxPTR(ptr));

  if (!ptr) {
    DP("Call to omp_target_is_present with NULL ptr, returning false\n");
    return false;
  }

  if (device_num == omp_get_initial_device()) {
    DP("Call to omp_target_is_present on host, returning true\n");
    return true;
  }

  RTLsMtx.lock();
  size_t Devices_size = Devices.size();
  RTLsMtx.unlock();
  if (Devices_size <= (size_t)device_num) {
    DP("Call to omp_target_is_present with invalid device ID, returning "
        "false\n");
    return false;
  }

  DeviceTy& Device = Devices[device_num];
  bool IsLast; // not used
  int rc = (Device.getTgtPtrBegin(ptr, 0, IsLast, false) != NULL);
  DP("Call to omp_target_is_present returns %d\n", rc);
  return rc;
}

EXTERN int omp_target_memcpy(void *dst, void *src, size_t length,
    size_t dst_offset, size_t src_offset, int dst_device, int src_device) {
  DP("Call to omp_target_memcpy, dst device %d, src device %d, "
      "dst addr " DPxMOD ", src addr " DPxMOD ", dst offset %zu, "
      "src offset %zu, length %zu\n", dst_device, src_device, DPxPTR(dst),
      DPxPTR(src), dst_offset, src_offset, length);

  if (!dst || !src || length <= 0) {
    DP("Call to omp_target_memcpy with invalid arguments\n");
    return OFFLOAD_FAIL;
  }

  if (src_device != omp_get_initial_device() && !device_is_ready(src_device)) {
      DP("omp_target_memcpy returns OFFLOAD_FAIL\n");
      return OFFLOAD_FAIL;
  }

  if (dst_device != omp_get_initial_device() && !device_is_ready(dst_device)) {
      DP("omp_target_memcpy returns OFFLOAD_FAIL\n");
      return OFFLOAD_FAIL;
  }

  int rc = OFFLOAD_SUCCESS;
  void *srcAddr = (char *)src + src_offset;
  void *dstAddr = (char *)dst + dst_offset;

  if (src_device == omp_get_initial_device() &&
      dst_device == omp_get_initial_device()) {
    DP("copy from host to host\n");
    const void *p = memcpy(dstAddr, srcAddr, length);
    if (p == NULL)
      rc = OFFLOAD_FAIL;
  } else if (src_device == omp_get_initial_device()) {
    DP("copy from host to device\n");
    DeviceTy& DstDev = Devices[dst_device];
    rc = DstDev.data_submit(dstAddr, srcAddr, length);
  } else if (dst_device == omp_get_initial_device()) {
    DP("copy from device to host\n");
    DeviceTy& SrcDev = Devices[src_device];
    rc = SrcDev.data_retrieve(dstAddr, srcAddr, length);
  } else {
    DP("copy from device to device\n");
    void *buffer = malloc(length);
    DeviceTy& SrcDev = Devices[src_device];
    DeviceTy& DstDev = Devices[dst_device];
    rc = SrcDev.data_retrieve(buffer, srcAddr, length);
    if (rc == OFFLOAD_SUCCESS)
      rc = DstDev.data_submit(dstAddr, buffer, length);
  }

  DP("omp_target_memcpy returns %d\n", rc);
  return rc;
}

EXTERN int omp_target_memcpy_rect(void *dst, void *src, size_t element_size,
    int num_dims, const size_t *volume, const size_t *dst_offsets,
    const size_t *src_offsets, const size_t *dst_dimensions,
    const size_t *src_dimensions, int dst_device, int src_device) {
  DP("Call to omp_target_memcpy_rect, dst device %d, src device %d, "
      "dst addr " DPxMOD ", src addr " DPxMOD ", dst offsets " DPxMOD ", "
      "src offsets " DPxMOD ", dst dims " DPxMOD ", src dims " DPxMOD ", "
      "volume " DPxMOD ", element size %zu, num_dims %d\n", dst_device,
      src_device, DPxPTR(dst), DPxPTR(src), DPxPTR(dst_offsets),
      DPxPTR(src_offsets), DPxPTR(dst_dimensions), DPxPTR(src_dimensions),
      DPxPTR(volume), element_size, num_dims);

  if (!(dst || src)) {
    DP("Call to omp_target_memcpy_rect returns max supported dimensions %d\n",
        INT_MAX);
    return INT_MAX;
  }

  if (!dst || !src || element_size < 1 || num_dims < 1 || !volume ||
      !dst_offsets || !src_offsets || !dst_dimensions || !src_dimensions) {
    DP("Call to omp_target_memcpy_rect with invalid arguments\n");
    return OFFLOAD_FAIL;
  }

  int rc;
  if (num_dims == 1) {
    rc = omp_target_memcpy(dst, src, element_size * volume[0],
        element_size * dst_offsets[0], element_size * src_offsets[0],
        dst_device, src_device);
  } else {
    size_t dst_slice_size = element_size;
    size_t src_slice_size = element_size;
    for (int i=1; i<num_dims; ++i) {
      dst_slice_size *= dst_dimensions[i];
      src_slice_size *= src_dimensions[i];
    }

    size_t dst_off = dst_offsets[0] * dst_slice_size;
    size_t src_off = src_offsets[0] * src_slice_size;
    for (size_t i=0; i<volume[0]; ++i) {
      rc = omp_target_memcpy_rect((char *) dst + dst_off + dst_slice_size * i,
          (char *) src + src_off + src_slice_size * i, element_size,
          num_dims - 1, volume + 1, dst_offsets + 1, src_offsets + 1,
          dst_dimensions + 1, src_dimensions + 1, dst_device, src_device);

      if (rc) {
        DP("Recursive call to omp_target_memcpy_rect returns unsuccessfully\n");
        return rc;
      }
    }
  }

  DP("omp_target_memcpy_rect returns %d\n", rc);
  return rc;
}

EXTERN int omp_target_associate_ptr(void *host_ptr, void *device_ptr,
    size_t size, size_t device_offset, int device_num) {
  DP("Call to omp_target_associate_ptr with host_ptr " DPxMOD ", "
      "device_ptr " DPxMOD ", size %zu, device_offset %zu, device_num %d\n",
      DPxPTR(host_ptr), DPxPTR(device_ptr), size, device_offset, device_num);

  if (!host_ptr || !device_ptr || size <= 0) {
    DP("Call to omp_target_associate_ptr with invalid arguments\n");
    return OFFLOAD_FAIL;
  }

  if (device_num == omp_get_initial_device()) {
    DP("omp_target_associate_ptr: no association possible on the host\n");
    return OFFLOAD_FAIL;
  }

  if (!device_is_ready(device_num)) {
    DP("omp_target_associate_ptr returns OFFLOAD_FAIL\n");
    return OFFLOAD_FAIL;
  }

  DeviceTy& Device = Devices[device_num];
  void *device_addr = (void *)((uint64_t)device_ptr + (uint64_t)device_offset);
  int rc = Device.associatePtr(host_ptr, device_addr, size);
  DP("omp_target_associate_ptr returns %d\n", rc);
  return rc;
}

EXTERN int omp_target_disassociate_ptr(void *host_ptr, int device_num) {
  DP("Call to omp_target_disassociate_ptr with host_ptr " DPxMOD ", "
      "device_num %d\n", DPxPTR(host_ptr), device_num);

  if (!host_ptr) {
    DP("Call to omp_target_associate_ptr with invalid host_ptr\n");
    return OFFLOAD_FAIL;
  }

  if (device_num == omp_get_initial_device()) {
    DP("omp_target_disassociate_ptr: no association possible on the host\n");
    return OFFLOAD_FAIL;
  }

  if (!device_is_ready(device_num)) {
    DP("omp_target_disassociate_ptr returns OFFLOAD_FAIL\n");
    return OFFLOAD_FAIL;
  }

  DeviceTy& Device = Devices[device_num];
  int rc = Device.disassociatePtr(host_ptr);
  DP("omp_target_disassociate_ptr returns %d\n", rc);
  return rc;
}

////////////////////////////////////////////////////////////////////////////////
// functionality for device

int DeviceTy::associatePtr(void *HstPtrBegin, void *TgtPtrBegin, int64_t Size) {
  DataMapMtx.lock();

  // Check if entry exists
  for (auto &HT : HostDataToTargetMap) {
    if ((uintptr_t)HstPtrBegin == HT.HstPtrBegin) {
      // Mapping already exists
      bool isValid = HT.HstPtrBegin == (uintptr_t) HstPtrBegin &&
                     HT.HstPtrEnd == (uintptr_t) HstPtrBegin + Size &&
                     HT.TgtPtrBegin == (uintptr_t) TgtPtrBegin;
      DataMapMtx.unlock();
      if (isValid) {
        DP("Attempt to re-associate the same device ptr+offset with the same "
            "host ptr, nothing to do\n");
        return OFFLOAD_SUCCESS;
      } else {
        DP("Not allowed to re-associate a different device ptr+offset with the "
            "same host ptr\n");
        return OFFLOAD_FAIL;
      }
    }
  }

  // Mapping does not exist, allocate it
  HostDataToTargetTy newEntry;

  // Set up missing fields
  newEntry.HstPtrBase = (uintptr_t) HstPtrBegin;
  newEntry.HstPtrBegin = (uintptr_t) HstPtrBegin;
  newEntry.HstPtrEnd = (uintptr_t) HstPtrBegin + Size;
  newEntry.TgtPtrBegin = (uintptr_t) TgtPtrBegin;
  // refCount must be infinite
  newEntry.RefCount = INF_REF_CNT;

  DP("Creating new map entry: HstBase=" DPxMOD ", HstBegin=" DPxMOD ", HstEnd="
      DPxMOD ", TgtBegin=" DPxMOD "\n", DPxPTR(newEntry.HstPtrBase),
      DPxPTR(newEntry.HstPtrBegin), DPxPTR(newEntry.HstPtrEnd),
      DPxPTR(newEntry.TgtPtrBegin));
  HostDataToTargetMap.push_front(newEntry);

  DataMapMtx.unlock();

  return OFFLOAD_SUCCESS;
}

int DeviceTy::disassociatePtr(void *HstPtrBegin) {
  DataMapMtx.lock();

  // Check if entry exists
  for (HostDataToTargetListTy::iterator ii = HostDataToTargetMap.begin();
      ii != HostDataToTargetMap.end(); ++ii) {
    if ((uintptr_t)HstPtrBegin == ii->HstPtrBegin) {
      // Mapping exists
      if (CONSIDERED_INF(ii->RefCount)) {
        DP("Association found, removing it\n");
        HostDataToTargetMap.erase(ii);
        DataMapMtx.unlock();
        return OFFLOAD_SUCCESS;
      } else {
        DP("Trying to disassociate a pointer which was not mapped via "
            "omp_target_associate_ptr\n");
        break;
      }
    }
  }

  // Mapping not found
  DataMapMtx.unlock();
  DP("Association not found\n");
  return OFFLOAD_FAIL;
}

// Get ref count of map entry containing HstPtrBegin
long DeviceTy::getMapEntryRefCnt(void *HstPtrBegin) {
  uintptr_t hp = (uintptr_t)HstPtrBegin;
  long RefCnt = -1;

  DataMapMtx.lock();
  for (auto &HT : HostDataToTargetMap) {
    if (hp >= HT.HstPtrBegin && hp < HT.HstPtrEnd) {
      DP("DeviceTy::getMapEntry: requested entry found\n");
      RefCnt = HT.RefCount;
      break;
    }
  }
  DataMapMtx.unlock();

  if (RefCnt < 0) {
    DP("DeviceTy::getMapEntry: requested entry not found\n");
  }

  return RefCnt;
}

LookupResult DeviceTy::lookupMapping(void *HstPtrBegin, int64_t Size) {
  uintptr_t hp = (uintptr_t)HstPtrBegin;
  LookupResult lr;

  DP("Looking up mapping(HstPtrBegin=" DPxMOD ", Size=%ld)...\n", DPxPTR(hp),
      Size);
  for (lr.Entry = HostDataToTargetMap.begin();
      lr.Entry != HostDataToTargetMap.end(); ++lr.Entry) {
    auto &HT = *lr.Entry;
    // Is it contained?
    lr.Flags.IsContained = hp >= HT.HstPtrBegin && hp < HT.HstPtrEnd &&
        (hp+Size) <= HT.HstPtrEnd;
    // Does it extend into an already mapped region?
    lr.Flags.ExtendsBefore = hp < HT.HstPtrBegin && (hp+Size) > HT.HstPtrBegin;
    // Does it extend beyond the mapped region?
    lr.Flags.ExtendsAfter = hp < HT.HstPtrEnd && (hp+Size) > HT.HstPtrEnd;

    if (lr.Flags.IsContained || lr.Flags.ExtendsBefore ||
        lr.Flags.ExtendsAfter) {
      break;
    }
  }

  if (lr.Flags.ExtendsBefore) {
    DP("WARNING: Pointer is not mapped but section extends into already "
        "mapped data\n");
  }
  if (lr.Flags.ExtendsAfter) {
    DP("WARNING: Pointer is already mapped but section extends beyond mapped "
        "region\n");
  }

  return lr;
}

// Used by target_data_begin
// Return the target pointer begin (where the data will be moved).
// Allocate memory if this is the first occurrence if this mapping.
// Increment the reference counter.
// If NULL is returned, then either data allocation failed or the user tried
// to do an illegal mapping.
void *DeviceTy::getOrAllocTgtPtr(void *HstPtrBegin, void *HstPtrBase,
    int64_t Size, bool &IsNew, bool IsImplicit, bool UpdateRefCount) {
  void *rc = NULL;
  DataMapMtx.lock();
  LookupResult lr = lookupMapping(HstPtrBegin, Size);

  // Check if the pointer is contained.
  if (lr.Flags.IsContained ||
      ((lr.Flags.ExtendsBefore || lr.Flags.ExtendsAfter) && IsImplicit)) {
    auto &HT = *lr.Entry;
    IsNew = false;

    if (UpdateRefCount)
      ++HT.RefCount;

    uintptr_t tp = HT.TgtPtrBegin + ((uintptr_t)HstPtrBegin - HT.HstPtrBegin);
    DP("Mapping exists%s with HstPtrBegin=" DPxMOD ", TgtPtrBegin=" DPxMOD ", "
        "Size=%ld,%s RefCount=%s\n", (IsImplicit ? " (implicit)" : ""),
        DPxPTR(HstPtrBegin), DPxPTR(tp), Size,
        (UpdateRefCount ? " updated" : ""),
        (CONSIDERED_INF(HT.RefCount)) ? "INF" :
            std::to_string(HT.RefCount).c_str());
    rc = (void *)tp;
  } else if ((lr.Flags.ExtendsBefore || lr.Flags.ExtendsAfter) && !IsImplicit) {
    // Explicit extension of mapped data - not allowed.
    DP("Explicit extension of mapping is not allowed.\n");
  } else if (Size) {
    // If it is not contained and Size > 0 we should create a new entry for it.
    IsNew = true;
    uintptr_t tp = (uintptr_t)RTL->data_alloc(RTLDeviceID, Size, HstPtrBegin);
    DP("Creating new map entry: HstBase=" DPxMOD ", HstBegin=" DPxMOD ", "
        "HstEnd=" DPxMOD ", TgtBegin=" DPxMOD "\n", DPxPTR(HstPtrBase),
        DPxPTR(HstPtrBegin), DPxPTR((uintptr_t)HstPtrBegin + Size), DPxPTR(tp));
    HostDataToTargetMap.push_front(HostDataToTargetTy((uintptr_t)HstPtrBase,
        (uintptr_t)HstPtrBegin, (uintptr_t)HstPtrBegin + Size, tp));
    rc = (void *)tp;
  }

  DataMapMtx.unlock();
  return rc;
}

// Used by target_data_begin, target_data_end, target_data_update and target.
// Return the target pointer begin (where the data will be moved).
// Decrement the reference counter if called from target_data_end.
void *DeviceTy::getTgtPtrBegin(void *HstPtrBegin, int64_t Size, bool &IsLast,
    bool UpdateRefCount) {
  void *rc = NULL;
  DataMapMtx.lock();
  LookupResult lr = lookupMapping(HstPtrBegin, Size);

  if (lr.Flags.IsContained || lr.Flags.ExtendsBefore || lr.Flags.ExtendsAfter) {
    auto &HT = *lr.Entry;
    IsLast = !(HT.RefCount > 1);

    if (HT.RefCount > 1 && UpdateRefCount)
      --HT.RefCount;

    uintptr_t tp = HT.TgtPtrBegin + ((uintptr_t)HstPtrBegin - HT.HstPtrBegin);
    DP("Mapping exists with HstPtrBegin=" DPxMOD ", TgtPtrBegin=" DPxMOD ", "
        "Size=%ld,%s RefCount=%s\n", DPxPTR(HstPtrBegin), DPxPTR(tp), Size,
        (UpdateRefCount ? " updated" : ""),
        (CONSIDERED_INF(HT.RefCount)) ? "INF" :
            std::to_string(HT.RefCount).c_str());
    rc = (void *)tp;
  } else {
    IsLast = false;
  }

  DataMapMtx.unlock();
  return rc;
}

// Return the target pointer begin (where the data will be moved).
// Lock-free version called when loading global symbols from the fat binary.
void *DeviceTy::getTgtPtrBegin(void *HstPtrBegin, int64_t Size) {
  uintptr_t hp = (uintptr_t)HstPtrBegin;
  LookupResult lr = lookupMapping(HstPtrBegin, Size);
  if (lr.Flags.IsContained || lr.Flags.ExtendsBefore || lr.Flags.ExtendsAfter) {
    auto &HT = *lr.Entry;
    uintptr_t tp = HT.TgtPtrBegin + (hp - HT.HstPtrBegin);
    return (void *)tp;
  }

  return NULL;
}

int DeviceTy::deallocTgtPtr(void *HstPtrBegin, int64_t Size, bool ForceDelete) {
  // Check if the pointer is contained in any sub-nodes.
  int rc;
  DataMapMtx.lock();
  LookupResult lr = lookupMapping(HstPtrBegin, Size);
  if (lr.Flags.IsContained || lr.Flags.ExtendsBefore || lr.Flags.ExtendsAfter) {
    auto &HT = *lr.Entry;
    if (ForceDelete)
      HT.RefCount = 1;
    if (--HT.RefCount <= 0) {
      assert(HT.RefCount == 0 && "did not expect a negative ref count");
      DP("Deleting tgt data " DPxMOD " of size %ld\n",
          DPxPTR(HT.TgtPtrBegin), Size);
      RTL->data_delete(RTLDeviceID, (void *)HT.TgtPtrBegin);
      DP("Removing%s mapping with HstPtrBegin=" DPxMOD ", TgtPtrBegin=" DPxMOD
          ", Size=%ld\n", (ForceDelete ? " (forced)" : ""),
          DPxPTR(HT.HstPtrBegin), DPxPTR(HT.TgtPtrBegin), Size);
      HostDataToTargetMap.erase(lr.Entry);
    }
    rc = OFFLOAD_SUCCESS;
  } else {
    DP("Section to delete (hst addr " DPxMOD ") does not exist in the allocated"
       " memory\n", DPxPTR(HstPtrBegin));
    rc = OFFLOAD_FAIL;
  }

  DataMapMtx.unlock();
  return rc;
}

/// Init device, should not be called directly.
void DeviceTy::init() {
  int32_t rc = RTL->init_device(RTLDeviceID);
  if (rc == OFFLOAD_SUCCESS) {
    IsInit = true;
  }
}

/// Thread-safe method to initialize the device only once.
int32_t DeviceTy::initOnce() {
  std::call_once(InitFlag, &DeviceTy::init, this);

  // At this point, if IsInit is true, then either this thread or some other
  // thread in the past successfully initialized the device, so we can return
  // OFFLOAD_SUCCESS. If this thread executed init() via call_once() and it
  // failed, return OFFLOAD_FAIL. If call_once did not invoke init(), it means
  // that some other thread already attempted to execute init() and if IsInit
  // is still false, return OFFLOAD_FAIL.
  if (IsInit)
    return OFFLOAD_SUCCESS;
  else
    return OFFLOAD_FAIL;
}

// Load binary to device.
__tgt_target_table *DeviceTy::load_binary(void *Img) {
  RTL->Mtx.lock();
  __tgt_target_table *rc = RTL->load_binary(RTLDeviceID, Img);
  RTL->Mtx.unlock();
  return rc;
}

// Submit data to device.
int32_t DeviceTy::data_submit(void *TgtPtrBegin, void *HstPtrBegin,
    int64_t Size) {
  return RTL->data_submit(RTLDeviceID, TgtPtrBegin, HstPtrBegin, Size);
}

// Retrieve data from device.
int32_t DeviceTy::data_retrieve(void *HstPtrBegin, void *TgtPtrBegin,
    int64_t Size) {
  return RTL->data_retrieve(RTLDeviceID, HstPtrBegin, TgtPtrBegin, Size);
}

// Run region on device
int32_t DeviceTy::run_region(void *TgtEntryPtr, void **TgtVarsPtr,
    ptrdiff_t *TgtOffsets, int32_t TgtVarsSize) {
  return RTL->run_region(RTLDeviceID, TgtEntryPtr, TgtVarsPtr, TgtOffsets,
      TgtVarsSize);
}

// Run team region on device.
int32_t DeviceTy::run_team_region(void *TgtEntryPtr, void **TgtVarsPtr,
    ptrdiff_t *TgtOffsets, int32_t TgtVarsSize, int32_t NumTeams,
    int32_t ThreadLimit, uint64_t LoopTripCount) {
  return RTL->run_team_region(RTLDeviceID, TgtEntryPtr, TgtVarsPtr, TgtOffsets,
      TgtVarsSize, NumTeams, ThreadLimit, LoopTripCount);
}

////////////////////////////////////////////////////////////////////////////////
// Functionality for registering libs

static void RegisterImageIntoTranslationTable(TranslationTable &TT,
    RTLInfoTy &RTL, __tgt_device_image *image) {

  // same size, as when we increase one, we also increase the other.
  assert(TT.TargetsTable.size() == TT.TargetsImages.size() &&
         "We should have as many images as we have tables!");

  // Resize the Targets Table and Images to accommodate the new targets if
  // required
  unsigned TargetsTableMinimumSize = RTL.Idx + RTL.NumberOfDevices;

  if (TT.TargetsTable.size() < TargetsTableMinimumSize) {
    TT.TargetsImages.resize(TargetsTableMinimumSize, 0);
    TT.TargetsTable.resize(TargetsTableMinimumSize, 0);
  }

  // Register the image in all devices for this target type.
  for (int32_t i = 0; i < RTL.NumberOfDevices; ++i) {
    // If we are changing the image we are also invalidating the target table.
    if (TT.TargetsImages[RTL.Idx + i] != image) {
      TT.TargetsImages[RTL.Idx + i] = image;
      TT.TargetsTable[RTL.Idx + i] = 0; // lazy initialization of target table.
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Functionality for registering Ctors/Dtors

static void RegisterGlobalCtorsDtorsForImage(__tgt_bin_desc *desc,
    __tgt_device_image *img, RTLInfoTy *RTL) {

  for (int32_t i = 0; i < RTL->NumberOfDevices; ++i) {
    DeviceTy &Device = Devices[RTL->Idx + i];
    Device.PendingGlobalsMtx.lock();
    Device.HasPendingGlobals = true;
    for (__tgt_offload_entry *entry = img->EntriesBegin;
        entry != img->EntriesEnd; ++entry) {
      if (entry->flags & OMP_DECLARE_TARGET_CTOR) {
        DP("Adding ctor " DPxMOD " to the pending list.\n",
            DPxPTR(entry->addr));
        Device.PendingCtorsDtors[desc].PendingCtors.push_back(entry->addr);
      } else if (entry->flags & OMP_DECLARE_TARGET_DTOR) {
        // Dtors are pushed in reverse order so they are executed from end
        // to beginning when unregistering the library!
        DP("Adding dtor " DPxMOD " to the pending list.\n",
            DPxPTR(entry->addr));
        Device.PendingCtorsDtors[desc].PendingDtors.push_front(entry->addr);
      }

      if (entry->flags & OMP_DECLARE_TARGET_LINK) {
        DP("The \"link\" attribute is not yet supported!\n");
      }
    }
    Device.PendingGlobalsMtx.unlock();
  }
}

////////////////////////////////////////////////////////////////////////////////
/// adds a target shared library to the target execution image
EXTERN void __tgt_register_lib(__tgt_bin_desc *desc) {

  // Attempt to load all plugins available in the system.
  RTLs.LoadRTLsOnce();

  RTLsMtx.lock();
  // Register the images with the RTLs that understand them, if any.
  for (int32_t i = 0; i < desc->NumDeviceImages; ++i) {
    // Obtain the image.
    __tgt_device_image *img = &desc->DeviceImages[i];

    RTLInfoTy *FoundRTL = NULL;

    // Scan the RTLs that have associated images until we find one that supports
    // the current image.
    for (auto &R : RTLs.AllRTLs) {
      if (!R.is_valid_binary(img)) {
        DP("Image " DPxMOD " is NOT compatible with RTL %s!\n",
            DPxPTR(img->ImageStart), R.RTLName.c_str());
        continue;
      }

      DP("Image " DPxMOD " is compatible with RTL %s!\n",
          DPxPTR(img->ImageStart), R.RTLName.c_str());

      // If this RTL is not already in use, initialize it.
      if (!R.isUsed) {
        // Initialize the device information for the RTL we are about to use.
        DeviceTy device(&R);

        size_t start = Devices.size();
        Devices.resize(start + R.NumberOfDevices, device);
        for (int32_t device_id = 0; device_id < R.NumberOfDevices;
            device_id++) {
          // global device ID
          Devices[start + device_id].DeviceID = start + device_id;
          // RTL local device ID
          Devices[start + device_id].RTLDeviceID = device_id;

          // Save pointer to device in RTL in case we want to unregister the RTL
          R.Devices.push_back(&Devices[start + device_id]);
        }

        // Initialize the index of this RTL and save it in the used RTLs.
        R.Idx = (RTLs.UsedRTLs.empty())
                    ? 0
                    : RTLs.UsedRTLs.back()->Idx +
                          RTLs.UsedRTLs.back()->NumberOfDevices;
        assert((size_t) R.Idx == start &&
            "RTL index should equal the number of devices used so far.");
        R.isUsed = true;
        RTLs.UsedRTLs.push_back(&R);

        DP("RTL " DPxMOD " has index %d!\n", DPxPTR(R.LibraryHandler), R.Idx);
      }

      // Initialize (if necessary) translation table for this library.
      TrlTblMtx.lock();
      if(!HostEntriesBeginToTransTable.count(desc->HostEntriesBegin)){
        TranslationTable &tt =
            HostEntriesBeginToTransTable[desc->HostEntriesBegin];
        tt.HostTable.EntriesBegin = desc->HostEntriesBegin;
        tt.HostTable.EntriesEnd = desc->HostEntriesEnd;
      }

      // Retrieve translation table for this library.
      TranslationTable &TransTable =
          HostEntriesBeginToTransTable[desc->HostEntriesBegin];

      DP("Registering image " DPxMOD " with RTL %s!\n",
          DPxPTR(img->ImageStart), R.RTLName.c_str());
      RegisterImageIntoTranslationTable(TransTable, R, img);
      TrlTblMtx.unlock();
      FoundRTL = &R;

      // Load ctors/dtors for static objects
      RegisterGlobalCtorsDtorsForImage(desc, img, FoundRTL);

      // if an RTL was found we are done - proceed to register the next image
      break;
    }

    if (!FoundRTL) {
      DP("No RTL found for image " DPxMOD "!\n", DPxPTR(img->ImageStart));
    }
  }
  RTLsMtx.unlock();


  DP("Done registering entries!\n");
}

////////////////////////////////////////////////////////////////////////////////
/// unloads a target shared library
EXTERN void __tgt_unregister_lib(__tgt_bin_desc *desc) {
  DP("Unloading target library!\n");

  RTLsMtx.lock();
  // Find which RTL understands each image, if any.
  for (int32_t i = 0; i < desc->NumDeviceImages; ++i) {
    // Obtain the image.
    __tgt_device_image *img = &desc->DeviceImages[i];

    RTLInfoTy *FoundRTL = NULL;

    // Scan the RTLs that have associated images until we find one that supports
    // the current image. We only need to scan RTLs that are already being used.
    for (auto *R : RTLs.UsedRTLs) {

      assert(R->isUsed && "Expecting used RTLs.");

      if (!R->is_valid_binary(img)) {
        DP("Image " DPxMOD " is NOT compatible with RTL " DPxMOD "!\n",
            DPxPTR(img->ImageStart), DPxPTR(R->LibraryHandler));
        continue;
      }

      DP("Image " DPxMOD " is compatible with RTL " DPxMOD "!\n",
          DPxPTR(img->ImageStart), DPxPTR(R->LibraryHandler));

      FoundRTL = R;

      // Execute dtors for static objects if the device has been used, i.e.
      // if its PendingCtors list has been emptied.
      for (int32_t i = 0; i < FoundRTL->NumberOfDevices; ++i) {
        DeviceTy &Device = Devices[FoundRTL->Idx + i];
        Device.PendingGlobalsMtx.lock();
        if (Device.PendingCtorsDtors[desc].PendingCtors.empty()) {
          for (auto &dtor : Device.PendingCtorsDtors[desc].PendingDtors) {
            int rc = target(Device.DeviceID, dtor, 0, NULL, NULL, NULL, NULL, 1,
                1, true /*team*/);
            if (rc != OFFLOAD_SUCCESS) {
              DP("Running destructor " DPxMOD " failed.\n", DPxPTR(dtor));
            }
          }
          // Remove this library's entry from PendingCtorsDtors
          Device.PendingCtorsDtors.erase(desc);
        }
        Device.PendingGlobalsMtx.unlock();
      }

      DP("Unregistered image " DPxMOD " from RTL " DPxMOD "!\n",
          DPxPTR(img->ImageStart), DPxPTR(R->LibraryHandler));

      break;
    }

    // if no RTL was found proceed to unregister the next image
    if (!FoundRTL){
      DP("No RTLs in use support the image " DPxMOD "!\n",
          DPxPTR(img->ImageStart));
    }
  }
  RTLsMtx.unlock();
  DP("Done unregistering images!\n");

  // Remove entries from HostPtrToTableMap
  TblMapMtx.lock();
  for (__tgt_offload_entry *cur = desc->HostEntriesBegin;
      cur < desc->HostEntriesEnd; ++cur) {
    HostPtrToTableMap.erase(cur->addr);
  }

  // Remove translation table for this descriptor.
  auto tt = HostEntriesBeginToTransTable.find(desc->HostEntriesBegin);
  if (tt != HostEntriesBeginToTransTable.end()) {
    DP("Removing translation table for descriptor " DPxMOD "\n",
        DPxPTR(desc->HostEntriesBegin));
    HostEntriesBeginToTransTable.erase(tt);
  } else {
    DP("Translation table for descriptor " DPxMOD " cannot be found, probably "
        "it has been already removed.\n", DPxPTR(desc->HostEntriesBegin));
  }

  TblMapMtx.unlock();

  // TODO: Remove RTL and the devices it manages if it's not used anymore?
  // TODO: Write some RTL->unload_image(...) function?

  DP("Done unregistering library!\n");
}

/// Map global data and execute pending ctors
static int InitLibrary(DeviceTy& Device) {
  /*
   * Map global data
   */
  int32_t device_id = Device.DeviceID;
  int rc = OFFLOAD_SUCCESS;

  Device.PendingGlobalsMtx.lock();
  TrlTblMtx.lock();
  for (HostEntriesBeginToTransTableTy::iterator
      ii = HostEntriesBeginToTransTable.begin();
      ii != HostEntriesBeginToTransTable.end(); ++ii) {
    TranslationTable *TransTable = &ii->second;
    if (TransTable->TargetsTable[device_id] != 0) {
      // Library entries have already been processed
      continue;
    }

    // 1) get image.
    assert(TransTable->TargetsImages.size() > (size_t)device_id &&
           "Not expecting a device ID outside the table's bounds!");
    __tgt_device_image *img = TransTable->TargetsImages[device_id];
    if (!img) {
      DP("No image loaded for device id %d.\n", device_id);
      rc = OFFLOAD_FAIL;
      break;
    }
    // 2) load image into the target table.
    __tgt_target_table *TargetTable =
        TransTable->TargetsTable[device_id] = Device.load_binary(img);
    // Unable to get table for this image: invalidate image and fail.
    if (!TargetTable) {
      DP("Unable to generate entries table for device id %d.\n", device_id);
      TransTable->TargetsImages[device_id] = 0;
      rc = OFFLOAD_FAIL;
      break;
    }

    // Verify whether the two table sizes match.
    size_t hsize =
        TransTable->HostTable.EntriesEnd - TransTable->HostTable.EntriesBegin;
    size_t tsize = TargetTable->EntriesEnd - TargetTable->EntriesBegin;

    // Invalid image for these host entries!
    if (hsize != tsize) {
      DP("Host and Target tables mismatch for device id %d [%zx != %zx].\n",
         device_id, hsize, tsize);
      TransTable->TargetsImages[device_id] = 0;
      TransTable->TargetsTable[device_id] = 0;
      rc = OFFLOAD_FAIL;
      break;
    }

    // process global data that needs to be mapped.
    Device.DataMapMtx.lock();
    __tgt_target_table *HostTable = &TransTable->HostTable;
    for (__tgt_offload_entry *CurrDeviceEntry = TargetTable->EntriesBegin,
                             *CurrHostEntry = HostTable->EntriesBegin,
                             *EntryDeviceEnd = TargetTable->EntriesEnd;
         CurrDeviceEntry != EntryDeviceEnd;
         CurrDeviceEntry++, CurrHostEntry++) {
      if (CurrDeviceEntry->size != 0) {
        // has data.
        assert(CurrDeviceEntry->size == CurrHostEntry->size &&
               "data size mismatch");

        // Fortran may use multiple weak declarations for the same symbol,
        // therefore we must allow for multiple weak symbols to be loaded from
        // the fat binary. Treat these mappings as any other "regular" mapping.
        // Add entry to map.
        if (Device.getTgtPtrBegin(CurrHostEntry->addr, CurrHostEntry->size))
          continue;
        DP("Add mapping from host " DPxMOD " to device " DPxMOD " with size %zu"
            "\n", DPxPTR(CurrHostEntry->addr), DPxPTR(CurrDeviceEntry->addr),
            CurrDeviceEntry->size);
        Device.HostDataToTargetMap.push_front(HostDataToTargetTy(
            (uintptr_t)CurrHostEntry->addr /*HstPtrBase*/,
            (uintptr_t)CurrHostEntry->addr /*HstPtrBegin*/,
            (uintptr_t)CurrHostEntry->addr + CurrHostEntry->size /*HstPtrEnd*/,
            (uintptr_t)CurrDeviceEntry->addr /*TgtPtrBegin*/,
            INF_REF_CNT /*RefCount*/));
      }
    }
    Device.DataMapMtx.unlock();
  }
  TrlTblMtx.unlock();

  if (rc != OFFLOAD_SUCCESS) {
    Device.PendingGlobalsMtx.unlock();
    return rc;
  }

  /*
   * Run ctors for static objects
   */
  if (!Device.PendingCtorsDtors.empty()) {
    // Call all ctors for all libraries registered so far
    for (auto &lib : Device.PendingCtorsDtors) {
      if (!lib.second.PendingCtors.empty()) {
        DP("Has pending ctors... call now\n");
        for (auto &entry : lib.second.PendingCtors) {
          void *ctor = entry;
          int rc = target(device_id, ctor, 0, NULL, NULL, NULL,
                          NULL, 1, 1, true /*team*/);
          if (rc != OFFLOAD_SUCCESS) {
            DP("Running ctor " DPxMOD " failed.\n", DPxPTR(ctor));
            Device.PendingGlobalsMtx.unlock();
            return OFFLOAD_FAIL;
          }
        }
        // Clear the list to indicate that this device has been used
        lib.second.PendingCtors.clear();
        DP("Done with pending ctors for lib " DPxMOD "\n", DPxPTR(lib.first));
      }
    }
  }
  Device.HasPendingGlobals = false;
  Device.PendingGlobalsMtx.unlock();

  return OFFLOAD_SUCCESS;
}

// Check whether a device has been initialized, global ctors have been
// executed and global data has been mapped; do so if not already done.
static int CheckDevice(int32_t device_id) {
  // Is device ready?
  if (!device_is_ready(device_id)) {
    DP("Device %d is not ready.\n", device_id);
    return OFFLOAD_FAIL;
  }

  // Get device info.
  DeviceTy &Device = Devices[device_id];

  // Check whether global data has been mapped for this device
  Device.PendingGlobalsMtx.lock();
  bool hasPendingGlobals = Device.HasPendingGlobals;
  Device.PendingGlobalsMtx.unlock();
  if (hasPendingGlobals && InitLibrary(Device) != OFFLOAD_SUCCESS) {
    DP("Failed to init globals on device %d\n", device_id);
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

// Following datatypes and functions (tgt_oldmap_type, combined_entry_t,
// translate_map, cleanup_map) will be removed once the compiler starts using
// the new map types.

// Old map types
enum tgt_oldmap_type {
  OMP_TGT_OLDMAPTYPE_TO          = 0x001, // copy data from host to device
  OMP_TGT_OLDMAPTYPE_FROM        = 0x002, // copy data from device to host
  OMP_TGT_OLDMAPTYPE_ALWAYS      = 0x004, // copy regardless of the ref. count
  OMP_TGT_OLDMAPTYPE_DELETE      = 0x008, // force unmapping of data
  OMP_TGT_OLDMAPTYPE_MAP_PTR     = 0x010, // map pointer as well as pointee
  OMP_TGT_OLDMAPTYPE_FIRST_MAP   = 0x020, // first occurrence of mapped variable
  OMP_TGT_OLDMAPTYPE_RETURN_PTR  = 0x040, // return TgtBase addr of mapped data
  OMP_TGT_OLDMAPTYPE_PRIVATE_PTR = 0x080, // private variable - not mapped
  OMP_TGT_OLDMAPTYPE_PRIVATE_VAL = 0x100  // copy by value - not mapped
};

// Temporary functions for map translation and cleanup
struct combined_entry_t {
  int num_members; // number of members in combined entry
  void *base_addr; // base address of combined entry
  void *begin_addr; // begin address of combined entry
  void *end_addr; // size of combined entry
};

static void translate_map(int32_t arg_num, void **args_base, void **args,
    int64_t *arg_sizes, int64_t *arg_types, int32_t &new_arg_num,
    void **&new_args_base, void **&new_args, int64_t *&new_arg_sizes,
    int64_t *&new_arg_types, bool is_target_construct) {
  if (arg_num <= 0) {
    DP("Nothing to translate\n");
    new_arg_num = 0;
    return;
  }

  // array of combined entries
  combined_entry_t *cmb_entries =
      (combined_entry_t *) alloca(arg_num * sizeof(combined_entry_t));
  // number of combined entries
  long num_combined = 0;
  // old entry is MAP_PTR?
  bool *is_ptr_old = (bool *) alloca(arg_num * sizeof(bool));
  // old entry is member of member_of[old] cmb_entry
  int *member_of = (int *) alloca(arg_num * sizeof(int));
  // temporary storage for modifications of the original arg_types
  int64_t *mod_arg_types = (int64_t *) alloca(arg_num  *sizeof(int64_t));

  DP("Translating %d map entries\n", arg_num);
  for (int i = 0; i < arg_num; ++i) {
    member_of[i] = -1;
    is_ptr_old[i] = false;
    mod_arg_types[i] = arg_types[i];
    // Scan previous entries to see whether this entry shares the same base
    for (int j = 0; j < i; ++j) {
      void *new_begin_addr = NULL;
      void *new_end_addr = NULL;

      if (mod_arg_types[i] & OMP_TGT_OLDMAPTYPE_MAP_PTR) {
        if (args_base[i] == args[j]) {
          if (!(mod_arg_types[j] & OMP_TGT_OLDMAPTYPE_MAP_PTR)) {
            DP("Entry %d has the same base as entry %d's begin address\n", i,
                j);
            new_begin_addr = args_base[i];
            new_end_addr = (char *)args_base[i] + sizeof(void *);
            assert(arg_sizes[j] == sizeof(void *));
            is_ptr_old[j] = true;
          } else {
            DP("Entry %d has the same base as entry %d's begin address, but "
                "%d's base was a MAP_PTR too\n", i, j, j);
            int32_t to_from_always_delete =
                OMP_TGT_OLDMAPTYPE_TO | OMP_TGT_OLDMAPTYPE_FROM |
                OMP_TGT_OLDMAPTYPE_ALWAYS | OMP_TGT_OLDMAPTYPE_DELETE;
            if (mod_arg_types[j] & to_from_always_delete) {
              DP("Resetting to/from/always/delete flags for entry %d because "
                  "it is only a pointer to pointer\n", j);
              mod_arg_types[j] &= ~to_from_always_delete;
            }
          }
        }
      } else {
        if (!(mod_arg_types[i] & OMP_TGT_OLDMAPTYPE_FIRST_MAP) &&
            args_base[i] == args_base[j]) {
          DP("Entry %d has the same base address as entry %d\n", i, j);
          new_begin_addr = args[i];
          new_end_addr = (char *)args[i] + arg_sizes[i];
        }
      }

      // If we have combined the entry with a previous one
      if (new_begin_addr) {
        int id;
        if(member_of[j] == -1) {
          // We have a new entry
          id = num_combined++;
          DP("Creating new combined entry %d for old entry %d\n", id, j);
          // Initialize new entry
          cmb_entries[id].num_members = 1;
          cmb_entries[id].base_addr = args_base[j];
          if (mod_arg_types[j] & OMP_TGT_OLDMAPTYPE_MAP_PTR) {
            cmb_entries[id].begin_addr = args_base[j];
            cmb_entries[id].end_addr = (char *)args_base[j] + arg_sizes[j];
          } else {
            cmb_entries[id].begin_addr = args[j];
            cmb_entries[id].end_addr = (char *)args[j] + arg_sizes[j];
          }
          member_of[j] = id;
        } else {
          // Reuse existing combined entry
          DP("Reusing existing combined entry %d\n", member_of[j]);
          id = member_of[j];
        }

        // Update combined entry
        DP("Adding entry %d to combined entry %d\n", i, id);
        cmb_entries[id].num_members++;
        // base_addr stays the same
        cmb_entries[id].begin_addr =
            std::min(cmb_entries[id].begin_addr, new_begin_addr);
        cmb_entries[id].end_addr =
            std::max(cmb_entries[id].end_addr, new_end_addr);
        member_of[i] = id;
        break;
      }
    }
  }

  DP("New entries: %ld combined + %d original\n", num_combined, arg_num);
  new_arg_num = arg_num + num_combined;
  new_args_base = (void **) malloc(new_arg_num * sizeof(void *));
  new_args = (void **) malloc(new_arg_num * sizeof(void *));
  new_arg_sizes = (int64_t *) malloc(new_arg_num * sizeof(int64_t));
  new_arg_types = (int64_t *) malloc(new_arg_num * sizeof(int64_t));

  const int64_t alignment = 8;

  int next_id = 0; // next ID
  int next_cid = 0; // next combined ID
  int *combined_to_new_id = (int *) alloca(num_combined * sizeof(int));
  for (int i = 0; i < arg_num; ++i) {
    // It is member_of
    if (member_of[i] == next_cid) {
      int cid = next_cid++; // ID of this combined entry
      int nid = next_id++; // ID of the new (global) entry
      combined_to_new_id[cid] = nid;
      DP("Combined entry %3d will become new entry %3d\n", cid, nid);

      int64_t padding = (int64_t)cmb_entries[cid].begin_addr % alignment;
      if (padding) {
        DP("Using a padding of %" PRId64 " for begin address " DPxMOD "\n",
            padding, DPxPTR(cmb_entries[cid].begin_addr));
        cmb_entries[cid].begin_addr =
            (char *)cmb_entries[cid].begin_addr - padding;
      }

      new_args_base[nid] = cmb_entries[cid].base_addr;
      new_args[nid] = cmb_entries[cid].begin_addr;
      new_arg_sizes[nid] = (int64_t) ((char *)cmb_entries[cid].end_addr -
          (char *)cmb_entries[cid].begin_addr);
      new_arg_types[nid] = OMP_TGT_MAPTYPE_TARGET_PARAM;
      DP("Entry %3d: base_addr " DPxMOD ", begin_addr " DPxMOD ", "
          "size %" PRId64 ", type 0x%" PRIx64 "\n", nid,
          DPxPTR(new_args_base[nid]), DPxPTR(new_args[nid]), new_arg_sizes[nid],
          new_arg_types[nid]);
    } else if (member_of[i] != -1) {
      DP("Combined entry %3d has been encountered before, do nothing\n",
          member_of[i]);
    }

    // Now that the combined entry (the one the old entry was a member of) has
    // been inserted into the new arguments list, proceed with the old entry.
    int nid = next_id++;
    DP("Old entry %3d will become new entry %3d\n", i, nid);

    new_args_base[nid] = args_base[i];
    new_args[nid] = args[i];
    new_arg_sizes[nid] = arg_sizes[i];
    int64_t old_type = mod_arg_types[i];

    if (is_ptr_old[i]) {
      // Reset TO and FROM flags
      old_type &= ~(OMP_TGT_OLDMAPTYPE_TO | OMP_TGT_OLDMAPTYPE_FROM);
    }

    if (member_of[i] == -1) {
      if (!is_target_construct)
        old_type &= ~OMP_TGT_MAPTYPE_TARGET_PARAM;
      new_arg_types[nid] = old_type;
      DP("Entry %3d: base_addr " DPxMOD ", begin_addr " DPxMOD ", size %" PRId64
          ", type 0x%" PRIx64 " (old entry %d not MEMBER_OF)\n", nid,
          DPxPTR(new_args_base[nid]), DPxPTR(new_args[nid]), new_arg_sizes[nid],
          new_arg_types[nid], i);
    } else {
      // Old entry is not FIRST_MAP
      old_type &= ~OMP_TGT_OLDMAPTYPE_FIRST_MAP;
      // Add MEMBER_OF
      int new_member_of = combined_to_new_id[member_of[i]];
      old_type |= ((int64_t)new_member_of + 1) << 48;
      new_arg_types[nid] = old_type;
      DP("Entry %3d: base_addr " DPxMOD ", begin_addr " DPxMOD ", size %" PRId64
        ", type 0x%" PRIx64 " (old entry %d MEMBER_OF %d)\n", nid,
        DPxPTR(new_args_base[nid]), DPxPTR(new_args[nid]), new_arg_sizes[nid],
        new_arg_types[nid], i, new_member_of);
    }
  }
}

static void cleanup_map(int32_t new_arg_num, void **new_args_base,
    void **new_args, int64_t *new_arg_sizes, int64_t *new_arg_types,
    int32_t arg_num, void **args_base) {
  if (new_arg_num > 0) {
    int offset = new_arg_num - arg_num;
    for (int32_t i = 0; i < arg_num; ++i) {
      // Restore old base address
      args_base[i] = new_args_base[i+offset];
    }
    free(new_args_base);
    free(new_args);
    free(new_arg_sizes);
    free(new_arg_types);
  }
}

static short member_of(int64_t type) {
  return ((type & OMP_TGT_MAPTYPE_MEMBER_OF) >> 48) - 1;
}

/// Internal function to do the mapping and transfer the data to the device
static int target_data_begin(DeviceTy &Device, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  // process each input.
  int rc = OFFLOAD_SUCCESS;
  for (int32_t i = 0; i < arg_num; ++i) {
    // Ignore private variables and arrays - there is no mapping for them.
    if ((arg_types[i] & OMP_TGT_MAPTYPE_LITERAL) ||
        (arg_types[i] & OMP_TGT_MAPTYPE_PRIVATE))
      continue;

    void *HstPtrBegin = args[i];
    void *HstPtrBase = args_base[i];
    // Address of pointer on the host and device, respectively.
    void *Pointer_HstPtrBegin, *Pointer_TgtPtrBegin;
    bool IsNew, Pointer_IsNew;
    bool IsImplicit = arg_types[i] & OMP_TGT_MAPTYPE_IMPLICIT;
    bool UpdateRef = !(arg_types[i] & OMP_TGT_MAPTYPE_MEMBER_OF);
    if (arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ) {
      DP("Has a pointer entry: \n");
      // base is address of pointer.
      Pointer_TgtPtrBegin = Device.getOrAllocTgtPtr(HstPtrBase, HstPtrBase,
          sizeof(void *), Pointer_IsNew, IsImplicit, UpdateRef);
      if (!Pointer_TgtPtrBegin) {
        DP("Call to getOrAllocTgtPtr returned null pointer (device failure or "
            "illegal mapping).\n");
      }
      DP("There are %zu bytes allocated at target address " DPxMOD " - is%s new"
          "\n", sizeof(void *), DPxPTR(Pointer_TgtPtrBegin),
          (Pointer_IsNew ? "" : " not"));
      Pointer_HstPtrBegin = HstPtrBase;
      // modify current entry.
      HstPtrBase = *(void **)HstPtrBase;
      UpdateRef = true; // subsequently update ref count of pointee
    }

    void *TgtPtrBegin = Device.getOrAllocTgtPtr(HstPtrBegin, HstPtrBase,
        arg_sizes[i], IsNew, IsImplicit, UpdateRef);
    if (!TgtPtrBegin && arg_sizes[i]) {
      // If arg_sizes[i]==0, then the argument is a pointer to NULL, so
      // getOrAlloc() returning NULL is not an error.
      DP("Call to getOrAllocTgtPtr returned null pointer (device failure or "
          "illegal mapping).\n");
    }
    DP("There are %" PRId64 " bytes allocated at target address " DPxMOD
        " - is%s new\n", arg_sizes[i], DPxPTR(TgtPtrBegin),
        (IsNew ? "" : " not"));

    if (arg_types[i] & OMP_TGT_MAPTYPE_RETURN_PARAM) {
      void *ret_ptr;
      if (arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ)
        ret_ptr = Pointer_TgtPtrBegin;
      else {
        bool IsLast; // not used
        ret_ptr = Device.getTgtPtrBegin(HstPtrBegin, 0, IsLast, false);
      }

      DP("Returning device pointer " DPxMOD "\n", DPxPTR(ret_ptr));
      args_base[i] = ret_ptr;
    }

    if (arg_types[i] & OMP_TGT_MAPTYPE_TO) {
      bool copy = false;
      if (IsNew || (arg_types[i] & OMP_TGT_MAPTYPE_ALWAYS)) {
        copy = true;
      } else if (arg_types[i] & OMP_TGT_MAPTYPE_MEMBER_OF) {
        // Copy data only if the "parent" struct has RefCount==1.
        short parent_idx = member_of(arg_types[i]);
        long parent_rc = Device.getMapEntryRefCnt(args[parent_idx]);
        assert(parent_rc > 0 && "parent struct not found");
        if (parent_rc == 1) {
          copy = true;
        }
      }

      if (copy) {
        DP("Moving %" PRId64 " bytes (hst:" DPxMOD ") -> (tgt:" DPxMOD ")\n",
            arg_sizes[i], DPxPTR(HstPtrBegin), DPxPTR(TgtPtrBegin));
        int rt = Device.data_submit(TgtPtrBegin, HstPtrBegin, arg_sizes[i]);
        if (rt != OFFLOAD_SUCCESS) {
          DP("Copying data to device failed.\n");
          rc = OFFLOAD_FAIL;
        }
      }
    }

    if (arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ) {
      DP("Update pointer (" DPxMOD ") -> [" DPxMOD "]\n",
          DPxPTR(Pointer_TgtPtrBegin), DPxPTR(TgtPtrBegin));
      uint64_t Delta = (uint64_t)HstPtrBegin - (uint64_t)HstPtrBase;
      void *TgtPtrBase = (void *)((uint64_t)TgtPtrBegin - Delta);
      int rt = Device.data_submit(Pointer_TgtPtrBegin, &TgtPtrBase,
          sizeof(void *));
      if (rt != OFFLOAD_SUCCESS) {
        DP("Copying data to device failed.\n");
        rc = OFFLOAD_FAIL;
      }
      // create shadow pointers for this entry
      Device.ShadowMtx.lock();
      Device.ShadowPtrMap[Pointer_HstPtrBegin] = {HstPtrBase,
          Pointer_TgtPtrBegin, TgtPtrBase};
      Device.ShadowMtx.unlock();
    }
  }

  return rc;
}

EXTERN void __tgt_target_data_begin_nowait(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    int32_t depNum, void *depList, int32_t noAliasDepNum,
    void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, 0);

  __tgt_target_data_begin(device_id, arg_num, args_base, args, arg_sizes,
                          arg_types);
}

/// creates host-to-target data mapping, stores it in the
/// libomptarget.so internal structure (an entry in a stack of data maps)
/// and passes the data to the device.
EXTERN void __tgt_target_data_begin(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  DP("Entering data begin region for device %ld with %d mappings\n", device_id,
     arg_num);

  // No devices available?
  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
    DP("Use default device id %ld\n", device_id);
  }

  if (CheckDevice(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %ld ready\n", device_id);
    return;
  }

  DeviceTy& Device = Devices[device_id];

  // Translate maps
  int32_t new_arg_num;
  void **new_args_base;
  void **new_args;
  int64_t *new_arg_sizes;
  int64_t *new_arg_types;
  translate_map(arg_num, args_base, args, arg_sizes, arg_types, new_arg_num,
      new_args_base, new_args, new_arg_sizes, new_arg_types, false);

  //target_data_begin(Device, arg_num, args_base, args, arg_sizes, arg_types);
  target_data_begin(Device, new_arg_num, new_args_base, new_args, new_arg_sizes,
      new_arg_types);

  // Cleanup translation memory
  cleanup_map(new_arg_num, new_args_base, new_args, new_arg_sizes,
      new_arg_types, arg_num, args_base);
}

/// Internal function to undo the mapping and retrieve the data from the device.
static int target_data_end(DeviceTy &Device, int32_t arg_num, void **args_base,
    void **args, int64_t *arg_sizes, int64_t *arg_types) {
  int rc = OFFLOAD_SUCCESS;
  // process each input.
  for (int32_t i = arg_num - 1; i >= 0; --i) {
    // Ignore private variables and arrays - there is no mapping for them.
    // Also, ignore the use_device_ptr directive, it has no effect here.
    if ((arg_types[i] & OMP_TGT_MAPTYPE_LITERAL) ||
        (arg_types[i] & OMP_TGT_MAPTYPE_PRIVATE))
      continue;

    void *HstPtrBegin = args[i];
    bool IsLast;
    bool UpdateRef = !(arg_types[i] & OMP_TGT_MAPTYPE_MEMBER_OF) ||
        (arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ);
    bool ForceDelete = arg_types[i] & OMP_TGT_MAPTYPE_DELETE;

    // If PTR_AND_OBJ, HstPtrBegin is address of pointee
    void *TgtPtrBegin = Device.getTgtPtrBegin(HstPtrBegin, arg_sizes[i], IsLast,
        UpdateRef);
    DP("There are %" PRId64 " bytes allocated at target address " DPxMOD
        " - is%s last\n", arg_sizes[i], DPxPTR(TgtPtrBegin),
        (IsLast ? "" : " not"));

    bool DelEntry = IsLast || ForceDelete;

    if ((arg_types[i] & OMP_TGT_MAPTYPE_MEMBER_OF) &&
        !(arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ)) {
      DelEntry = false; // protect parent struct from being deallocated
    }

    if ((arg_types[i] & OMP_TGT_MAPTYPE_FROM) || DelEntry) {
      // Move data back to the host
      if (arg_types[i] & OMP_TGT_MAPTYPE_FROM) {
        bool Always = arg_types[i] & OMP_TGT_MAPTYPE_ALWAYS;
        bool CopyMember = false;
        if ((arg_types[i] & OMP_TGT_MAPTYPE_MEMBER_OF) &&
            !(arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ)) {
          // Copy data only if the "parent" struct has RefCount==1.
          short parent_idx = member_of(arg_types[i]);
          long parent_rc = Device.getMapEntryRefCnt(args[parent_idx]);
          assert(parent_rc > 0 && "parent struct not found");
          if (parent_rc == 1) {
            CopyMember = true;
          }
        }

        if (DelEntry || Always || CopyMember) {
          DP("Moving %" PRId64 " bytes (tgt:" DPxMOD ") -> (hst:" DPxMOD ")\n",
              arg_sizes[i], DPxPTR(TgtPtrBegin), DPxPTR(HstPtrBegin));
          int rt = Device.data_retrieve(HstPtrBegin, TgtPtrBegin, arg_sizes[i]);
          if (rt != OFFLOAD_SUCCESS) {
            DP("Copying data from device failed.\n");
            rc = OFFLOAD_FAIL;
          }
        }
      }

      // If we copied back to the host a struct/array containing pointers, we
      // need to restore the original host pointer values from their shadow
      // copies. If the struct is going to be deallocated, remove any remaining
      // shadow pointer entries for this struct.
      uintptr_t lb = (uintptr_t) HstPtrBegin;
      uintptr_t ub = (uintptr_t) HstPtrBegin + arg_sizes[i];
      Device.ShadowMtx.lock();
      for (ShadowPtrListTy::iterator it = Device.ShadowPtrMap.begin();
          it != Device.ShadowPtrMap.end(); ++it) {
        void **ShadowHstPtrAddr = (void**) it->first;

        // An STL map is sorted on its keys; use this property
        // to quickly determine when to break out of the loop.
        if ((uintptr_t) ShadowHstPtrAddr < lb)
          continue;
        if ((uintptr_t) ShadowHstPtrAddr >= ub)
          break;

        // If we copied the struct to the host, we need to restore the pointer.
        if (arg_types[i] & OMP_TGT_MAPTYPE_FROM) {
          DP("Restoring original host pointer value " DPxMOD " for host "
              "pointer " DPxMOD "\n", DPxPTR(it->second.HstPtrVal),
              DPxPTR(ShadowHstPtrAddr));
          *ShadowHstPtrAddr = it->second.HstPtrVal;
        }
        // If the struct is to be deallocated, remove the shadow entry.
        if (DelEntry) {
          DP("Removing shadow pointer " DPxMOD "\n", DPxPTR(ShadowHstPtrAddr));
          Device.ShadowPtrMap.erase(it);
        }
      }
      Device.ShadowMtx.unlock();

      // Deallocate map
      if (DelEntry) {
        int rt = Device.deallocTgtPtr(HstPtrBegin, arg_sizes[i], ForceDelete);
        if (rt != OFFLOAD_SUCCESS) {
          DP("Deallocating data from device failed.\n");
          rc = OFFLOAD_FAIL;
        }
      }
    }
  }

  return rc;
}

/// passes data from the target, releases target memory and destroys
/// the host-target mapping (top entry from the stack of data maps)
/// created by the last __tgt_target_data_begin.
EXTERN void __tgt_target_data_end(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  DP("Entering data end region with %d mappings\n", arg_num);

  // No devices available?
  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  RTLsMtx.lock();
  size_t Devices_size = Devices.size();
  RTLsMtx.unlock();
  if (Devices_size <= (size_t)device_id) {
    DP("Device ID  %ld does not have a matching RTL.\n", device_id);
    return;
  }

  DeviceTy &Device = Devices[device_id];
  if (!Device.IsInit) {
    DP("uninit device: ignore");
    return;
  }

  // Translate maps
  int32_t new_arg_num;
  void **new_args_base;
  void **new_args;
  int64_t *new_arg_sizes;
  int64_t *new_arg_types;
  translate_map(arg_num, args_base, args, arg_sizes, arg_types, new_arg_num,
      new_args_base, new_args, new_arg_sizes, new_arg_types, false);

  //target_data_end(Device, arg_num, args_base, args, arg_sizes, arg_types);
  target_data_end(Device, new_arg_num, new_args_base, new_args, new_arg_sizes,
      new_arg_types);

  // Cleanup translation memory
  cleanup_map(new_arg_num, new_args_base, new_args, new_arg_sizes,
      new_arg_types, arg_num, args_base);
}

EXTERN void __tgt_target_data_end_nowait(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    int32_t depNum, void *depList, int32_t noAliasDepNum,
    void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, 0);

  __tgt_target_data_end(device_id, arg_num, args_base, args, arg_sizes,
                        arg_types);
}

/// passes data to/from the target.
EXTERN void __tgt_target_data_update(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  DP("Entering data update with %d mappings\n", arg_num);

  // No devices available?
  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDevice(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %ld ready\n", device_id);
    return;
  }

  DeviceTy& Device = Devices[device_id];

  // process each input.
  for (int32_t i = 0; i < arg_num; ++i) {
    if ((arg_types[i] & OMP_TGT_MAPTYPE_LITERAL) ||
        (arg_types[i] & OMP_TGT_MAPTYPE_PRIVATE))
      continue;

    void *HstPtrBegin = args[i];
    int64_t MapSize = arg_sizes[i];
    bool IsLast;
    void *TgtPtrBegin = Device.getTgtPtrBegin(HstPtrBegin, MapSize, IsLast,
        false);

    if (arg_types[i] & OMP_TGT_MAPTYPE_FROM) {
      DP("Moving %" PRId64 " bytes (tgt:" DPxMOD ") -> (hst:" DPxMOD ")\n",
          arg_sizes[i], DPxPTR(TgtPtrBegin), DPxPTR(HstPtrBegin));
      Device.data_retrieve(HstPtrBegin, TgtPtrBegin, MapSize);

      uintptr_t lb = (uintptr_t) HstPtrBegin;
      uintptr_t ub = (uintptr_t) HstPtrBegin + MapSize;
      Device.ShadowMtx.lock();
      for (ShadowPtrListTy::iterator it = Device.ShadowPtrMap.begin();
          it != Device.ShadowPtrMap.end(); ++it) {
        void **ShadowHstPtrAddr = (void**) it->first;
        if ((uintptr_t) ShadowHstPtrAddr < lb)
          continue;
        if ((uintptr_t) ShadowHstPtrAddr >= ub)
          break;
        DP("Restoring original host pointer value " DPxMOD " for host pointer "
            DPxMOD "\n", DPxPTR(it->second.HstPtrVal),
            DPxPTR(ShadowHstPtrAddr));
        *ShadowHstPtrAddr = it->second.HstPtrVal;
      }
      Device.ShadowMtx.unlock();
    }

    if (arg_types[i] & OMP_TGT_MAPTYPE_TO) {
      DP("Moving %" PRId64 " bytes (hst:" DPxMOD ") -> (tgt:" DPxMOD ")\n",
          arg_sizes[i], DPxPTR(HstPtrBegin), DPxPTR(TgtPtrBegin));
      Device.data_submit(TgtPtrBegin, HstPtrBegin, MapSize);

      uintptr_t lb = (uintptr_t) HstPtrBegin;
      uintptr_t ub = (uintptr_t) HstPtrBegin + MapSize;
      Device.ShadowMtx.lock();
      for (ShadowPtrListTy::iterator it = Device.ShadowPtrMap.begin();
          it != Device.ShadowPtrMap.end(); ++it) {
        void **ShadowHstPtrAddr = (void**) it->first;
        if ((uintptr_t) ShadowHstPtrAddr < lb)
          continue;
        if ((uintptr_t) ShadowHstPtrAddr >= ub)
          break;
        DP("Restoring original target pointer value " DPxMOD " for target "
            "pointer " DPxMOD "\n", DPxPTR(it->second.TgtPtrVal),
            DPxPTR(it->second.TgtPtrAddr));
        Device.data_submit(it->second.TgtPtrAddr,
            &it->second.TgtPtrVal, sizeof(void *));
      }
      Device.ShadowMtx.unlock();
    }
  }
}

EXTERN void __tgt_target_data_update_nowait(
    int64_t device_id, int32_t arg_num, void **args_base, void **args,
    int64_t *arg_sizes, int64_t *arg_types, int32_t depNum, void *depList,
    int32_t noAliasDepNum, void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, 0);

  __tgt_target_data_update(device_id, arg_num, args_base, args, arg_sizes,
                           arg_types);
}

/// performs the same actions as data_begin in case arg_num is
/// non-zero and initiates run of the offloaded region on the target platform;
/// if arg_num is non-zero after the region execution is done it also
/// performs the same action as data_update and data_end above. This function
/// returns 0 if it was able to transfer the execution to a target and an
/// integer different from zero otherwise.
static int target(int64_t device_id, void *host_ptr, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    int32_t team_num, int32_t thread_limit, int IsTeamConstruct) {
  DeviceTy &Device = Devices[device_id];

  // Find the table information in the map or look it up in the translation
  // tables.
  TableMap *TM = 0;
  TblMapMtx.lock();
  HostPtrToTableMapTy::iterator TableMapIt = HostPtrToTableMap.find(host_ptr);
  if (TableMapIt == HostPtrToTableMap.end()) {
    // We don't have a map. So search all the registered libraries.
    TrlTblMtx.lock();
    for (HostEntriesBeginToTransTableTy::iterator
             ii = HostEntriesBeginToTransTable.begin(),
             ie = HostEntriesBeginToTransTable.end();
         !TM && ii != ie; ++ii) {
      // get the translation table (which contains all the good info).
      TranslationTable *TransTable = &ii->second;
      // iterate over all the host table entries to see if we can locate the
      // host_ptr.
      __tgt_offload_entry *begin = TransTable->HostTable.EntriesBegin;
      __tgt_offload_entry *end = TransTable->HostTable.EntriesEnd;
      __tgt_offload_entry *cur = begin;
      for (uint32_t i = 0; cur < end; ++cur, ++i) {
        if (cur->addr != host_ptr)
          continue;
        // we got a match, now fill the HostPtrToTableMap so that we
        // may avoid this search next time.
        TM = &HostPtrToTableMap[host_ptr];
        TM->Table = TransTable;
        TM->Index = i;
        break;
      }
    }
    TrlTblMtx.unlock();
  } else {
    TM = &TableMapIt->second;
  }
  TblMapMtx.unlock();

  // No map for this host pointer found!
  if (!TM) {
    DP("Host ptr " DPxMOD " does not have a matching target pointer.\n",
       DPxPTR(host_ptr));
    return OFFLOAD_FAIL;
  }

  // get target table.
  TrlTblMtx.lock();
  assert(TM->Table->TargetsTable.size() > (size_t)device_id &&
         "Not expecting a device ID outside the table's bounds!");
  __tgt_target_table *TargetTable = TM->Table->TargetsTable[device_id];
  TrlTblMtx.unlock();
  assert(TargetTable && "Global data has not been mapped\n");

  // Move data to device.
  int rc = target_data_begin(Device, arg_num, args_base, args, arg_sizes,
      arg_types);

  if (rc != OFFLOAD_SUCCESS) {
    DP("Call to target_data_begin failed, skipping target execution.\n");
    // Call target_data_end to dealloc whatever target_data_begin allocated
    // and return OFFLOAD_FAIL.
    target_data_end(Device, arg_num, args_base, args, arg_sizes, arg_types);
    return OFFLOAD_FAIL;
  }

  std::vector<void *> tgt_args;
  std::vector<ptrdiff_t> tgt_offsets;

  // List of (first-)private arrays allocated for this target region
  std::vector<void *> fpArrays;

  for (int32_t i = 0; i < arg_num; ++i) {
    if (!(arg_types[i] & OMP_TGT_MAPTYPE_TARGET_PARAM)) {
      // This is not a target parameter, do not push it into tgt_args.
      continue;
    }
    void *HstPtrBegin = args[i];
    void *HstPtrBase = args_base[i];
    void *TgtPtrBegin;
    ptrdiff_t TgtBaseOffset;
    bool IsLast; // unused.
    if (arg_types[i] & OMP_TGT_MAPTYPE_LITERAL) {
      DP("Forwarding first-private value " DPxMOD " to the target construct\n",
          DPxPTR(HstPtrBase));
      TgtPtrBegin = HstPtrBase;
      TgtBaseOffset = 0;
    } else if (arg_types[i] & OMP_TGT_MAPTYPE_PRIVATE) {
      // Allocate memory for (first-)private array
      TgtPtrBegin = Device.RTL->data_alloc(Device.RTLDeviceID,
          arg_sizes[i], HstPtrBegin);
      if (!TgtPtrBegin) {
        DP ("Data allocation for %sprivate array " DPxMOD " failed\n",
            (arg_types[i] & OMP_TGT_MAPTYPE_TO ? "first-" : ""),
            DPxPTR(HstPtrBegin));
        rc = OFFLOAD_FAIL;
        break;
      } else {
        fpArrays.push_back(TgtPtrBegin);
        TgtBaseOffset = (intptr_t)HstPtrBase - (intptr_t)HstPtrBegin;
#ifdef OMPTARGET_DEBUG
        void *TgtPtrBase = (void *)((intptr_t)TgtPtrBegin + TgtBaseOffset);
        DP("Allocated %" PRId64 " bytes of target memory at " DPxMOD " for "
            "%sprivate array " DPxMOD " - pushing target argument " DPxMOD "\n",
            arg_sizes[i], DPxPTR(TgtPtrBegin),
            (arg_types[i] & OMP_TGT_MAPTYPE_TO ? "first-" : ""),
            DPxPTR(HstPtrBegin), DPxPTR(TgtPtrBase));
#endif
        // If first-private, copy data from host
        if (arg_types[i] & OMP_TGT_MAPTYPE_TO) {
          int rt = Device.data_submit(TgtPtrBegin, HstPtrBegin, arg_sizes[i]);
          if (rt != OFFLOAD_SUCCESS) {
            DP ("Copying data to device failed.\n");
            rc = OFFLOAD_FAIL;
            break;
          }
        }
      }
    } else if (arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ) {
      TgtPtrBegin = Device.getTgtPtrBegin(HstPtrBase, sizeof(void *), IsLast,
          false);
      TgtBaseOffset = 0; // no offset for ptrs.
      DP("Obtained target argument " DPxMOD " from host pointer " DPxMOD " to "
         "object " DPxMOD "\n", DPxPTR(TgtPtrBegin), DPxPTR(HstPtrBase),
         DPxPTR(HstPtrBase));
    } else {
      TgtPtrBegin = Device.getTgtPtrBegin(HstPtrBegin, arg_sizes[i], IsLast,
          false);
      TgtBaseOffset = (intptr_t)HstPtrBase - (intptr_t)HstPtrBegin;
#ifdef OMPTARGET_DEBUG
      void *TgtPtrBase = (void *)((intptr_t)TgtPtrBegin + TgtBaseOffset);
      DP("Obtained target argument " DPxMOD " from host pointer " DPxMOD "\n",
          DPxPTR(TgtPtrBase), DPxPTR(HstPtrBegin));
#endif
    }
    tgt_args.push_back(TgtPtrBegin);
    tgt_offsets.push_back(TgtBaseOffset);
  }

  assert(tgt_args.size() == tgt_offsets.size() &&
      "Size mismatch in arguments and offsets");

  // Pop loop trip count
  uint64_t ltc = Device.loopTripCnt;
  Device.loopTripCnt = 0;

  // Launch device execution.
  if (rc == OFFLOAD_SUCCESS) {
    DP("Launching target execution %s with pointer " DPxMOD " (index=%d).\n",
        TargetTable->EntriesBegin[TM->Index].name,
        DPxPTR(TargetTable->EntriesBegin[TM->Index].addr), TM->Index);
    if (IsTeamConstruct) {
      rc = Device.run_team_region(TargetTable->EntriesBegin[TM->Index].addr,
          &tgt_args[0], &tgt_offsets[0], tgt_args.size(), team_num,
          thread_limit, ltc);
    } else {
      rc = Device.run_region(TargetTable->EntriesBegin[TM->Index].addr,
          &tgt_args[0], &tgt_offsets[0], tgt_args.size());
    }
  } else {
    DP("Errors occurred while obtaining target arguments, skipping kernel "
        "execution\n");
  }

  // Deallocate (first-)private arrays
  for (auto it : fpArrays) {
    int rt = Device.RTL->data_delete(Device.RTLDeviceID, it);
    if (rt != OFFLOAD_SUCCESS) {
      DP("Deallocation of (first-)private arrays failed.\n");
      rc = OFFLOAD_FAIL;
    }
  }

  // Move data from device.
  int rt = target_data_end(Device, arg_num, args_base, args, arg_sizes,
      arg_types);

  if (rt != OFFLOAD_SUCCESS) {
    DP("Call to target_data_end failed.\n");
    rc = OFFLOAD_FAIL;
  }

  return rc;
}

EXTERN int __tgt_target(int64_t device_id, void *host_ptr, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  DP("Entering target region with entry point " DPxMOD " and device Id %ld\n",
     DPxPTR(host_ptr), device_id);

  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDevice(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %ld ready\n", device_id);
    return OFFLOAD_FAIL;
  }

  // Translate maps
  int32_t new_arg_num;
  void **new_args_base;
  void **new_args;
  int64_t *new_arg_sizes;
  int64_t *new_arg_types;
  translate_map(arg_num, args_base, args, arg_sizes, arg_types, new_arg_num,
      new_args_base, new_args, new_arg_sizes, new_arg_types, true);

  //return target(device_id, host_ptr, arg_num, args_base, args, arg_sizes,
  //    arg_types, 0, 0, false /*team*/, false /*recursive*/);
  int rc = target(device_id, host_ptr, new_arg_num, new_args_base, new_args,
      new_arg_sizes, new_arg_types, 0, 0, false /*team*/);

  // Cleanup translation memory
  cleanup_map(new_arg_num, new_args_base, new_args, new_arg_sizes,
      new_arg_types, arg_num, args_base);

  return rc;
}

EXTERN int __tgt_target_nowait(int64_t device_id, void *host_ptr,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, int32_t depNum, void *depList, int32_t noAliasDepNum,
    void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, 0);

  return __tgt_target(device_id, host_ptr, arg_num, args_base, args, arg_sizes,
                      arg_types);
}

EXTERN int __tgt_target_teams(int64_t device_id, void *host_ptr,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, int32_t team_num, int32_t thread_limit) {
  DP("Entering target region with entry point " DPxMOD " and device Id %ld\n",
     DPxPTR(host_ptr), device_id);

  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDevice(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %ld ready\n", device_id);
    return OFFLOAD_FAIL;
  }

  // Translate maps
  int32_t new_arg_num;
  void **new_args_base;
  void **new_args;
  int64_t *new_arg_sizes;
  int64_t *new_arg_types;
  translate_map(arg_num, args_base, args, arg_sizes, arg_types, new_arg_num,
      new_args_base, new_args, new_arg_sizes, new_arg_types, true);

  //return target(device_id, host_ptr, arg_num, args_base, args, arg_sizes,
  //              arg_types, team_num, thread_limit, true /*team*/,
  //              false /*recursive*/);
  int rc = target(device_id, host_ptr, new_arg_num, new_args_base, new_args,
      new_arg_sizes, new_arg_types, team_num, thread_limit, true /*team*/);

  // Cleanup translation memory
  cleanup_map(new_arg_num, new_args_base, new_args, new_arg_sizes,
      new_arg_types, arg_num, args_base);

  return rc;
}

EXTERN int __tgt_target_teams_nowait(int64_t device_id, void *host_ptr,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, int32_t team_num, int32_t thread_limit, int32_t depNum,
    void *depList, int32_t noAliasDepNum, void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, 0);

  return __tgt_target_teams(device_id, host_ptr, arg_num, args_base, args,
                            arg_sizes, arg_types, team_num, thread_limit);
}


// The trip count mechanism will be revised - this scheme is not thread-safe.
EXTERN void __kmpc_push_target_tripcount(int64_t device_id,
    uint64_t loop_tripcount) {
  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDevice(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %ld ready\n", device_id);
    return;
  }

  DP("__kmpc_push_target_tripcount(%ld, %" PRIu64 ")\n", device_id,
      loop_tripcount);
  Devices[device_id].loopTripCnt = loop_tripcount;
}

