//===----RTLs/cuda/src/rtl.cpp - Target RTLs Implementation ------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL for CUDA machine
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstddef>
#include <cuda.h>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "Debug.h"
#include "omptargetplugin.h"

#define TARGET_NAME CUDA
#define DEBUG_PREFIX "Target " GETNAME(TARGET_NAME) " RTL"

// Utility for retrieving and printing CUDA error string.
#ifdef OMPTARGET_DEBUG
#define CUDA_ERR_STRING(err)                                                   \
  do {                                                                         \
    if (getDebugLevel() > 0) {                                                      \
      const char *errStr;                                                      \
      cuGetErrorString(err, &errStr);                                          \
      DP("CUDA error is: %s\n", errStr);                                       \
    }                                                                          \
  } while (false)
#else // OMPTARGET_DEBUG
#define CUDA_ERR_STRING(err) {}
#endif // OMPTARGET_DEBUG

#include "../../common/elf_common.c"

/// Keep entries table per device.
struct FuncOrGblEntryTy {
  __tgt_target_table Table;
  std::vector<__tgt_offload_entry> Entries;
};

enum ExecutionModeType {
  SPMD, // constructors, destructors,
  // combined constructs (`teams distribute parallel for [simd]`)
  GENERIC, // everything else
  NONE
};

/// Use a single entity to encode a kernel and a set of flags.
struct KernelTy {
  CUfunction Func;

  // execution mode of kernel
  // 0 - SPMD mode (without master warp)
  // 1 - Generic mode (with master warp)
  int8_t ExecutionMode;

  /// Maximal number of threads per block for this kernel.
  int MaxThreadsPerBlock = 0;

  KernelTy(CUfunction _Func, int8_t _ExecutionMode)
      : Func(_Func), ExecutionMode(_ExecutionMode) {}
};

/// Device environment data
/// Manually sync with the deviceRTL side for now, move to a dedicated header
/// file later.
struct omptarget_device_environmentTy {
  int32_t debug_level;
};

namespace {
bool checkResult(CUresult Err, const char *ErrMsg) {
  if (Err == CUDA_SUCCESS)
    return true;

  DP("%s", ErrMsg);
  CUDA_ERR_STRING(Err);
  return false;
}

int memcpyDtoD(const void *SrcPtr, void *DstPtr, int64_t Size,
               CUstream Stream) {
  CUresult Err =
      cuMemcpyDtoDAsync((CUdeviceptr)DstPtr, (CUdeviceptr)SrcPtr, Size, Stream);

  if (Err != CUDA_SUCCESS) {
    DP("Error when copying data from device to device. Pointers: src "
       "= " DPxMOD ", dst = " DPxMOD ", size = %" PRId64 "\n",
       DPxPTR(SrcPtr), DPxPTR(DstPtr), Size);
    CUDA_ERR_STRING(Err);
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

// Structure contains per-device data
struct DeviceDataTy {
  /// List that contains all the kernels.
  std::list<KernelTy> KernelsList;

  std::list<FuncOrGblEntryTy> FuncGblEntries;

  CUcontext Context = nullptr;
  // Device properties
  int ThreadsPerBlock = 0;
  int BlocksPerGrid = 0;
  int WarpSize = 0;
  // OpenMP properties
  int NumTeams = 0;
  int NumThreads = 0;
};

class StreamManagerTy {
  int NumberOfDevices;
  // The initial size of stream pool
  int EnvNumInitialStreams;
  // Per-device stream mutex
  std::vector<std::unique_ptr<std::mutex>> StreamMtx;
  // Per-device stream Id indicates the next available stream in the pool
  std::vector<int> NextStreamId;
  // Per-device stream pool
  std::vector<std::vector<CUstream>> StreamPool;
  // Reference to per-device data
  std::vector<DeviceDataTy> &DeviceData;

  // If there is no CUstream left in the pool, we will resize the pool to
  // allocate more CUstream. This function should be called with device mutex,
  // and we do not resize to smaller one.
  void resizeStreamPool(const int DeviceId, const size_t NewSize) {
    std::vector<CUstream> &Pool = StreamPool[DeviceId];
    const size_t CurrentSize = Pool.size();
    assert(NewSize > CurrentSize && "new size is not larger than current size");

    CUresult Err = cuCtxSetCurrent(DeviceData[DeviceId].Context);
    if (!checkResult(Err, "Error returned from cuCtxSetCurrent\n")) {
      // We will return if cannot switch to the right context in case of
      // creating bunch of streams that are not corresponding to the right
      // device. The offloading will fail later because selected CUstream is
      // nullptr.
      return;
    }

    Pool.resize(NewSize, nullptr);

    for (size_t I = CurrentSize; I < NewSize; ++I) {
      checkResult(cuStreamCreate(&Pool[I], CU_STREAM_NON_BLOCKING),
                  "Error returned from cuStreamCreate\n");
    }
  }

public:
  StreamManagerTy(const int NumberOfDevices,
                  std::vector<DeviceDataTy> &DeviceData)
      : NumberOfDevices(NumberOfDevices), EnvNumInitialStreams(32),
        DeviceData(DeviceData) {
    StreamPool.resize(NumberOfDevices);
    NextStreamId.resize(NumberOfDevices);
    StreamMtx.resize(NumberOfDevices);

    if (const char *EnvStr = getenv("LIBOMPTARGET_NUM_INITIAL_STREAMS"))
      EnvNumInitialStreams = std::stoi(EnvStr);

    // Initialize the next stream id
    std::fill(NextStreamId.begin(), NextStreamId.end(), 0);

    // Initialize stream mutex
    for (std::unique_ptr<std::mutex> &Ptr : StreamMtx)
      Ptr = std::make_unique<std::mutex>();
  }

  ~StreamManagerTy() {
    // Destroy streams
    for (int I = 0; I < NumberOfDevices; ++I) {
      checkResult(cuCtxSetCurrent(DeviceData[I].Context),
                  "Error returned from cuCtxSetCurrent\n");

      for (CUstream &S : StreamPool[I]) {
        if (S)
          checkResult(cuStreamDestroy(S),
                      "Error returned from cuStreamDestroy\n");
      }
    }
  }

  // Get a CUstream from pool. Per-device next stream id always points to the
  // next available CUstream. That means, CUstreams [0, id-1] have been
  // assigned, and [id,] are still available. If there is no CUstream left, we
  // will ask more CUstreams from CUDA RT. Each time a CUstream is assigned,
  // the id will increase one.
  // xxxxxs+++++++++
  //      ^
  //      id
  // After assignment, the pool becomes the following and s is assigned.
  // xxxxxs+++++++++
  //       ^
  //       id
  CUstream getStream(const int DeviceId) {
    const std::lock_guard<std::mutex> Lock(*StreamMtx[DeviceId]);
    int &Id = NextStreamId[DeviceId];
    // No CUstream left in the pool, we need to request from CUDA RT
    if (Id == StreamPool[DeviceId].size()) {
      // By default we double the stream pool every time
      resizeStreamPool(DeviceId, Id * 2);
    }
    return StreamPool[DeviceId][Id++];
  }

  // Return a CUstream back to pool. As mentioned above, per-device next
  // stream is always points to the next available CUstream, so when we return
  // a CUstream, we need to first decrease the id, and then copy the CUstream
  // back.
  // It is worth noting that, the order of streams return might be different
  // from that they're assigned, that saying, at some point, there might be
  // two identical CUstreams.
  // xxax+a+++++
  //     ^
  //     id
  // However, it doesn't matter, because they're always on the two sides of
  // id. The left one will in the end be overwritten by another CUstream.
  // Therefore, after several execution, the order of pool might be different
  // from its initial state.
  void returnStream(const int DeviceId, CUstream Stream) {
    const std::lock_guard<std::mutex> Lock(*StreamMtx[DeviceId]);
    int &Id = NextStreamId[DeviceId];
    assert(Id > 0 && "Wrong stream ID");
    StreamPool[DeviceId][--Id] = Stream;
  }

  bool initializeDeviceStreamPool(const int DeviceId) {
    assert(StreamPool[DeviceId].empty() && "stream pool has been initialized");

    resizeStreamPool(DeviceId, EnvNumInitialStreams);

    // Check the size of stream pool
    if (StreamPool[DeviceId].size() != EnvNumInitialStreams)
      return false;

    // Check whether each stream is valid
    for (CUstream &S : StreamPool[DeviceId])
      if (!S)
        return false;

    return true;
  }
};

class DeviceRTLTy {
  int NumberOfDevices;
  // OpenMP environment properties
  int EnvNumTeams;
  int EnvTeamLimit;
  // OpenMP requires flags
  int64_t RequiresFlags;

  static constexpr const int HardTeamLimit = 1U << 16U; // 64k
  static constexpr const int HardThreadLimit = 1024;
  static constexpr const int DefaultNumTeams = 128;
  static constexpr const int DefaultNumThreads = 128;

  std::unique_ptr<StreamManagerTy> StreamManager;
  std::vector<DeviceDataTy> DeviceData;
  std::vector<CUmodule> Modules;

  // Record entry point associated with device
  void addOffloadEntry(const int DeviceId, const __tgt_offload_entry entry) {
    FuncOrGblEntryTy &E = DeviceData[DeviceId].FuncGblEntries.back();
    E.Entries.push_back(entry);
  }

  // Return true if the entry is associated with device
  bool findOffloadEntry(const int DeviceId, const void *Addr) const {
    for (const __tgt_offload_entry &Itr :
         DeviceData[DeviceId].FuncGblEntries.back().Entries)
      if (Itr.addr == Addr)
        return true;

    return false;
  }

  // Return the pointer to the target entries table
  __tgt_target_table *getOffloadEntriesTable(const int DeviceId) {
    FuncOrGblEntryTy &E = DeviceData[DeviceId].FuncGblEntries.back();

    if (E.Entries.empty())
      return nullptr;

    // Update table info according to the entries and return the pointer
    E.Table.EntriesBegin = E.Entries.data();
    E.Table.EntriesEnd = E.Entries.data() + E.Entries.size();

    return &E.Table;
  }

  // Clear entries table for a device
  void clearOffloadEntriesTable(const int DeviceId) {
    DeviceData[DeviceId].FuncGblEntries.emplace_back();
    FuncOrGblEntryTy &E = DeviceData[DeviceId].FuncGblEntries.back();
    E.Entries.clear();
    E.Table.EntriesBegin = E.Table.EntriesEnd = nullptr;
  }

  CUstream getStream(const int DeviceId, __tgt_async_info *AsyncInfoPtr) const {
    assert(AsyncInfoPtr && "AsyncInfoPtr is nullptr");

    if (!AsyncInfoPtr->Queue)
      AsyncInfoPtr->Queue = StreamManager->getStream(DeviceId);

    return reinterpret_cast<CUstream>(AsyncInfoPtr->Queue);
  }

public:
  // This class should not be copied
  DeviceRTLTy(const DeviceRTLTy &) = delete;
  DeviceRTLTy(DeviceRTLTy &&) = delete;

  DeviceRTLTy()
      : NumberOfDevices(0), EnvNumTeams(-1), EnvTeamLimit(-1),
        RequiresFlags(OMP_REQ_UNDEFINED) {

    DP("Start initializing CUDA\n");

    CUresult Err = cuInit(0);
    if (!checkResult(Err, "Error returned from cuInit\n")) {
      return;
    }

    Err = cuDeviceGetCount(&NumberOfDevices);
    if (!checkResult(Err, "Error returned from cuDeviceGetCount\n"))
      return;

    if (NumberOfDevices == 0) {
      DP("There are no devices supporting CUDA.\n");
      return;
    }

    DeviceData.resize(NumberOfDevices);

    // Get environment variables regarding teams
    if (const char *EnvStr = getenv("OMP_TEAM_LIMIT")) {
      // OMP_TEAM_LIMIT has been set
      EnvTeamLimit = std::stoi(EnvStr);
      DP("Parsed OMP_TEAM_LIMIT=%d\n", EnvTeamLimit);
    }
    if (const char *EnvStr = getenv("OMP_NUM_TEAMS")) {
      // OMP_NUM_TEAMS has been set
      EnvNumTeams = std::stoi(EnvStr);
      DP("Parsed OMP_NUM_TEAMS=%d\n", EnvNumTeams);
    }

    StreamManager =
        std::make_unique<StreamManagerTy>(NumberOfDevices, DeviceData);
  }

  ~DeviceRTLTy() {
    // First destruct stream manager in case of Contexts is destructed before it
    StreamManager = nullptr;

    for (CUmodule &M : Modules)
      // Close module
      if (M)
        checkResult(cuModuleUnload(M), "Error returned from cuModuleUnload\n");

    for (DeviceDataTy &D : DeviceData) {
      // Destroy context
      if (D.Context) {
        checkResult(cuCtxSetCurrent(D.Context),
                    "Error returned from cuCtxSetCurrent\n");
        CUdevice Device;
        checkResult(cuCtxGetDevice(&Device),
                    "Error returned from cuCtxGetDevice\n");
        checkResult(cuDevicePrimaryCtxRelease(Device),
                    "Error returned from cuDevicePrimaryCtxRelease\n");
      }
    }
  }

  // Check whether a given DeviceId is valid
  bool isValidDeviceId(const int DeviceId) const {
    return DeviceId >= 0 && DeviceId < NumberOfDevices;
  }

  int getNumOfDevices() const { return NumberOfDevices; }

  void setRequiresFlag(const int64_t Flags) { this->RequiresFlags = Flags; }

  int initDevice(const int DeviceId) {
    CUdevice Device;

    DP("Getting device %d\n", DeviceId);
    CUresult Err = cuDeviceGet(&Device, DeviceId);
    if (!checkResult(Err, "Error returned from cuDeviceGet\n"))
      return OFFLOAD_FAIL;

    // Query the current flags of the primary context and set its flags if
    // it is inactive
    unsigned int FormerPrimaryCtxFlags = 0;
    int FormerPrimaryCtxIsActive = 0;
    Err = cuDevicePrimaryCtxGetState(Device, &FormerPrimaryCtxFlags,
                                     &FormerPrimaryCtxIsActive);
    if (!checkResult(Err, "Error returned from cuDevicePrimaryCtxGetState\n"))
      return OFFLOAD_FAIL;

    if (FormerPrimaryCtxIsActive) {
      DP("The primary context is active, no change to its flags\n");
      if ((FormerPrimaryCtxFlags & CU_CTX_SCHED_MASK) !=
          CU_CTX_SCHED_BLOCKING_SYNC)
        DP("Warning the current flags are not CU_CTX_SCHED_BLOCKING_SYNC\n");
    } else {
      DP("The primary context is inactive, set its flags to "
         "CU_CTX_SCHED_BLOCKING_SYNC\n");
      Err = cuDevicePrimaryCtxSetFlags(Device, CU_CTX_SCHED_BLOCKING_SYNC);
      if (!checkResult(Err, "Error returned from cuDevicePrimaryCtxSetFlags\n"))
        return OFFLOAD_FAIL;
    }

    // Retain the per device primary context and save it to use whenever this
    // device is selected.
    Err = cuDevicePrimaryCtxRetain(&DeviceData[DeviceId].Context, Device);
    if (!checkResult(Err, "Error returned from cuDevicePrimaryCtxRetain\n"))
      return OFFLOAD_FAIL;

    Err = cuCtxSetCurrent(DeviceData[DeviceId].Context);
    if (!checkResult(Err, "Error returned from cuCtxSetCurrent\n"))
      return OFFLOAD_FAIL;

    // Initialize stream pool
    if (!StreamManager->initializeDeviceStreamPool(DeviceId))
      return OFFLOAD_FAIL;

    // Query attributes to determine number of threads/block and blocks/grid.
    int MaxGridDimX;
    Err = cuDeviceGetAttribute(&MaxGridDimX, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                               Device);
    if (Err != CUDA_SUCCESS) {
      DP("Error getting max grid dimension, use default value %d\n",
         DeviceRTLTy::DefaultNumTeams);
      DeviceData[DeviceId].BlocksPerGrid = DeviceRTLTy::DefaultNumTeams;
    } else if (MaxGridDimX <= DeviceRTLTy::HardTeamLimit) {
      DP("Using %d CUDA blocks per grid\n", MaxGridDimX);
      DeviceData[DeviceId].BlocksPerGrid = MaxGridDimX;
    } else {
      DP("Max CUDA blocks per grid %d exceeds the hard team limit %d, capping "
         "at the hard limit\n",
         MaxGridDimX, DeviceRTLTy::HardTeamLimit);
      DeviceData[DeviceId].BlocksPerGrid = DeviceRTLTy::HardTeamLimit;
    }

    // We are only exploiting threads along the x axis.
    int MaxBlockDimX;
    Err = cuDeviceGetAttribute(&MaxBlockDimX,
                               CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, Device);
    if (Err != CUDA_SUCCESS) {
      DP("Error getting max block dimension, use default value %d\n",
         DeviceRTLTy::DefaultNumThreads);
      DeviceData[DeviceId].ThreadsPerBlock = DeviceRTLTy::DefaultNumThreads;
    } else if (MaxBlockDimX <= DeviceRTLTy::HardThreadLimit) {
      DP("Using %d CUDA threads per block\n", MaxBlockDimX);
      DeviceData[DeviceId].ThreadsPerBlock = MaxBlockDimX;
    } else {
      DP("Max CUDA threads per block %d exceeds the hard thread limit %d, "
         "capping at the hard limit\n",
         MaxBlockDimX, DeviceRTLTy::HardThreadLimit);
      DeviceData[DeviceId].ThreadsPerBlock = DeviceRTLTy::HardThreadLimit;
    }

    // Get and set warp size
    int WarpSize;
    Err =
        cuDeviceGetAttribute(&WarpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, Device);
    if (Err != CUDA_SUCCESS) {
      DP("Error getting warp size, assume default value 32\n");
      DeviceData[DeviceId].WarpSize = 32;
    } else {
      DP("Using warp size %d\n", WarpSize);
      DeviceData[DeviceId].WarpSize = WarpSize;
    }

    // Adjust teams to the env variables
    if (EnvTeamLimit > 0 && DeviceData[DeviceId].BlocksPerGrid > EnvTeamLimit) {
      DP("Capping max CUDA blocks per grid to OMP_TEAM_LIMIT=%d\n",
         EnvTeamLimit);
      DeviceData[DeviceId].BlocksPerGrid = EnvTeamLimit;
    }

    DP("Max number of CUDA blocks %d, threads %d & warp size %d\n",
       DeviceData[DeviceId].BlocksPerGrid, DeviceData[DeviceId].ThreadsPerBlock,
       DeviceData[DeviceId].WarpSize);

    // Set default number of teams
    if (EnvNumTeams > 0) {
      DP("Default number of teams set according to environment %d\n",
         EnvNumTeams);
      DeviceData[DeviceId].NumTeams = EnvNumTeams;
    } else {
      DeviceData[DeviceId].NumTeams = DeviceRTLTy::DefaultNumTeams;
      DP("Default number of teams set according to library's default %d\n",
         DeviceRTLTy::DefaultNumTeams);
    }

    if (DeviceData[DeviceId].NumTeams > DeviceData[DeviceId].BlocksPerGrid) {
      DP("Default number of teams exceeds device limit, capping at %d\n",
         DeviceData[DeviceId].BlocksPerGrid);
      DeviceData[DeviceId].NumTeams = DeviceData[DeviceId].BlocksPerGrid;
    }

    // Set default number of threads
    DeviceData[DeviceId].NumThreads = DeviceRTLTy::DefaultNumThreads;
    DP("Default number of threads set according to library's default %d\n",
       DeviceRTLTy::DefaultNumThreads);
    if (DeviceData[DeviceId].NumThreads >
        DeviceData[DeviceId].ThreadsPerBlock) {
      DP("Default number of threads exceeds device limit, capping at %d\n",
         DeviceData[DeviceId].ThreadsPerBlock);
      DeviceData[DeviceId].NumTeams = DeviceData[DeviceId].ThreadsPerBlock;
    }

    return OFFLOAD_SUCCESS;
  }

  __tgt_target_table *loadBinary(const int DeviceId,
                                 const __tgt_device_image *Image) {
    // Set the context we are using
    CUresult Err = cuCtxSetCurrent(DeviceData[DeviceId].Context);
    if (!checkResult(Err, "Error returned from cuCtxSetCurrent\n"))
      return nullptr;

    // Clear the offload table as we are going to create a new one.
    clearOffloadEntriesTable(DeviceId);

    // Create the module and extract the function pointers.
    CUmodule Module;
    DP("Load data from image " DPxMOD "\n", DPxPTR(Image->ImageStart));
    Err = cuModuleLoadDataEx(&Module, Image->ImageStart, 0, nullptr, nullptr);
    if (!checkResult(Err, "Error returned from cuModuleLoadDataEx\n"))
      return nullptr;

    DP("CUDA module successfully loaded!\n");

    Modules.push_back(Module);

    // Find the symbols in the module by name.
    const __tgt_offload_entry *HostBegin = Image->EntriesBegin;
    const __tgt_offload_entry *HostEnd = Image->EntriesEnd;

    std::list<KernelTy> &KernelsList = DeviceData[DeviceId].KernelsList;
    for (const __tgt_offload_entry *E = HostBegin; E != HostEnd; ++E) {
      if (!E->addr) {
        // We return nullptr when something like this happens, the host should
        // have always something in the address to uniquely identify the target
        // region.
        DP("Invalid binary: host entry '<null>' (size = %zd)...\n", E->size);
        return nullptr;
      }

      if (E->size) {
        __tgt_offload_entry Entry = *E;
        CUdeviceptr CUPtr;
        size_t CUSize;
        Err = cuModuleGetGlobal(&CUPtr, &CUSize, Module, E->name);
        // We keep this style here because we need the name
        if (Err != CUDA_SUCCESS) {
          DP("Loading global '%s' (Failed)\n", E->name);
          CUDA_ERR_STRING(Err);
          return nullptr;
        }

        if (CUSize != E->size) {
          DP("Loading global '%s' - size mismatch (%zd != %zd)\n", E->name,
             CUSize, E->size);
          return nullptr;
        }

        DP("Entry point " DPxMOD " maps to global %s (" DPxMOD ")\n",
           DPxPTR(E - HostBegin), E->name, DPxPTR(CUPtr));

        Entry.addr = (void *)(CUPtr);

        // Note: In the current implementation declare target variables
        // can either be link or to. This means that once unified
        // memory is activated via the requires directive, the variable
        // can be used directly from the host in both cases.
        // TODO: when variables types other than to or link are added,
        // the below condition should be changed to explicitly
        // check for to and link variables types:
        // (RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY && (e->flags &
        // OMP_DECLARE_TARGET_LINK || e->flags == OMP_DECLARE_TARGET_TO))
        if (RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY) {
          // If unified memory is present any target link or to variables
          // can access host addresses directly. There is no longer a
          // need for device copies.
          cuMemcpyHtoD(CUPtr, E->addr, sizeof(void *));
          DP("Copy linked variable host address (" DPxMOD
             ") to device address (" DPxMOD ")\n",
             DPxPTR(*((void **)E->addr)), DPxPTR(CUPtr));
        }

        addOffloadEntry(DeviceId, Entry);

        continue;
      }

      CUfunction Func;
      Err = cuModuleGetFunction(&Func, Module, E->name);
      // We keep this style here because we need the name
      if (Err != CUDA_SUCCESS) {
        DP("Loading '%s' (Failed)\n", E->name);
        CUDA_ERR_STRING(Err);
        return nullptr;
      }

      DP("Entry point " DPxMOD " maps to %s (" DPxMOD ")\n",
         DPxPTR(E - HostBegin), E->name, DPxPTR(Func));

      // default value GENERIC (in case symbol is missing from cubin file)
      int8_t ExecModeVal = ExecutionModeType::GENERIC;
      std::string ExecModeNameStr(E->name);
      ExecModeNameStr += "_exec_mode";
      const char *ExecModeName = ExecModeNameStr.c_str();

      CUdeviceptr ExecModePtr;
      size_t CUSize;
      Err = cuModuleGetGlobal(&ExecModePtr, &CUSize, Module, ExecModeName);
      if (Err == CUDA_SUCCESS) {
        if (CUSize != sizeof(int8_t)) {
          DP("Loading global exec_mode '%s' - size mismatch (%zd != %zd)\n",
             ExecModeName, CUSize, sizeof(int8_t));
          return nullptr;
        }

        Err = cuMemcpyDtoH(&ExecModeVal, ExecModePtr, CUSize);
        if (Err != CUDA_SUCCESS) {
          DP("Error when copying data from device to host. Pointers: "
             "host = " DPxMOD ", device = " DPxMOD ", size = %zd\n",
             DPxPTR(&ExecModeVal), DPxPTR(ExecModePtr), CUSize);
          CUDA_ERR_STRING(Err);
          return nullptr;
        }

        if (ExecModeVal < 0 || ExecModeVal > 1) {
          DP("Error wrong exec_mode value specified in cubin file: %d\n",
             ExecModeVal);
          return nullptr;
        }
      } else {
        DP("Loading global exec_mode '%s' - symbol missing, using default "
           "value GENERIC (1)\n",
           ExecModeName);
        CUDA_ERR_STRING(Err);
      }

      KernelsList.emplace_back(Func, ExecModeVal);

      __tgt_offload_entry Entry = *E;
      Entry.addr = &KernelsList.back();
      addOffloadEntry(DeviceId, Entry);
    }

    // send device environment data to the device
    {
      omptarget_device_environmentTy DeviceEnv{0};

#ifdef OMPTARGET_DEBUG
      if (const char *EnvStr = getenv("LIBOMPTARGET_DEVICE_RTL_DEBUG"))
        DeviceEnv.debug_level = std::stoi(EnvStr);
#endif

      const char *DeviceEnvName = "omptarget_device_environment";
      CUdeviceptr DeviceEnvPtr;
      size_t CUSize;

      Err = cuModuleGetGlobal(&DeviceEnvPtr, &CUSize, Module, DeviceEnvName);
      if (Err == CUDA_SUCCESS) {
        if (CUSize != sizeof(DeviceEnv)) {
          DP("Global device_environment '%s' - size mismatch (%zu != %zu)\n",
             DeviceEnvName, CUSize, sizeof(int32_t));
          CUDA_ERR_STRING(Err);
          return nullptr;
        }

        Err = cuMemcpyHtoD(DeviceEnvPtr, &DeviceEnv, CUSize);
        if (Err != CUDA_SUCCESS) {
          DP("Error when copying data from host to device. Pointers: "
             "host = " DPxMOD ", device = " DPxMOD ", size = %zu\n",
             DPxPTR(&DeviceEnv), DPxPTR(DeviceEnvPtr), CUSize);
          CUDA_ERR_STRING(Err);
          return nullptr;
        }

        DP("Sending global device environment data %zu bytes\n", CUSize);
      } else {
        DP("Finding global device environment '%s' - symbol missing.\n",
           DeviceEnvName);
        DP("Continue, considering this is a device RTL which does not accept "
           "environment setting.\n");
      }
    }

    return getOffloadEntriesTable(DeviceId);
  }

  void *dataAlloc(const int DeviceId, const int64_t Size) const {
    if (Size == 0)
      return nullptr;

    CUresult Err = cuCtxSetCurrent(DeviceData[DeviceId].Context);
    if (!checkResult(Err, "Error returned from cuCtxSetCurrent\n"))
      return nullptr;

    CUdeviceptr DevicePtr;
    Err = cuMemAlloc(&DevicePtr, Size);
    if (!checkResult(Err, "Error returned from cuMemAlloc\n"))
      return nullptr;

    return (void *)DevicePtr;
  }

  int dataSubmit(const int DeviceId, const void *TgtPtr, const void *HstPtr,
                 const int64_t Size, __tgt_async_info *AsyncInfoPtr) const {
    assert(AsyncInfoPtr && "AsyncInfoPtr is nullptr");

    CUresult Err = cuCtxSetCurrent(DeviceData[DeviceId].Context);
    if (!checkResult(Err, "Error returned from cuCtxSetCurrent\n"))
      return OFFLOAD_FAIL;

    CUstream Stream = getStream(DeviceId, AsyncInfoPtr);

    Err = cuMemcpyHtoDAsync((CUdeviceptr)TgtPtr, HstPtr, Size, Stream);
    if (Err != CUDA_SUCCESS) {
      DP("Error when copying data from host to device. Pointers: host = " DPxMOD
         ", device = " DPxMOD ", size = %" PRId64 "\n",
         DPxPTR(HstPtr), DPxPTR(TgtPtr), Size);
      CUDA_ERR_STRING(Err);
      return OFFLOAD_FAIL;
    }

    return OFFLOAD_SUCCESS;
  }

  int dataRetrieve(const int DeviceId, void *HstPtr, const void *TgtPtr,
                   const int64_t Size, __tgt_async_info *AsyncInfoPtr) const {
    assert(AsyncInfoPtr && "AsyncInfoPtr is nullptr");

    CUresult Err = cuCtxSetCurrent(DeviceData[DeviceId].Context);
    if (!checkResult(Err, "Error returned from cuCtxSetCurrent\n"))
      return OFFLOAD_FAIL;

    CUstream Stream = getStream(DeviceId, AsyncInfoPtr);

    Err = cuMemcpyDtoHAsync(HstPtr, (CUdeviceptr)TgtPtr, Size, Stream);
    if (Err != CUDA_SUCCESS) {
      DP("Error when copying data from device to host. Pointers: host = " DPxMOD
         ", device = " DPxMOD ", size = %" PRId64 "\n",
         DPxPTR(HstPtr), DPxPTR(TgtPtr), Size);
      CUDA_ERR_STRING(Err);
      return OFFLOAD_FAIL;
    }

    return OFFLOAD_SUCCESS;
  }

  int dataExchange(int SrcDevId, const void *SrcPtr, int DstDevId, void *DstPtr,
                   int64_t Size, __tgt_async_info *AsyncInfoPtr) const {
    assert(AsyncInfoPtr && "AsyncInfoPtr is nullptr");

    CUresult Err = cuCtxSetCurrent(DeviceData[SrcDevId].Context);
    if (!checkResult(Err, "Error returned from cuCtxSetCurrent\n"))
      return OFFLOAD_FAIL;

    CUstream Stream = getStream(SrcDevId, AsyncInfoPtr);

    // If they are two devices, we try peer to peer copy first
    if (SrcDevId != DstDevId) {
      int CanAccessPeer = 0;
      Err = cuDeviceCanAccessPeer(&CanAccessPeer, SrcDevId, DstDevId);
      if (Err != CUDA_SUCCESS) {
        DP("Error returned from cuDeviceCanAccessPeer. src = %" PRId32
           ", dst = %" PRId32 "\n",
           SrcDevId, DstDevId);
        CUDA_ERR_STRING(Err);
        return memcpyDtoD(SrcPtr, DstPtr, Size, Stream);
      }

      if (!CanAccessPeer) {
        DP("P2P memcpy not supported so fall back to D2D memcpy");
        return memcpyDtoD(SrcPtr, DstPtr, Size, Stream);
      }

      Err = cuCtxEnablePeerAccess(DeviceData[DstDevId].Context, 0);
      if (Err != CUDA_SUCCESS) {
        DP("Error returned from cuCtxEnablePeerAccess. src = %" PRId32
           ", dst = %" PRId32 "\n",
           SrcDevId, DstDevId);
        CUDA_ERR_STRING(Err);
        return memcpyDtoD(SrcPtr, DstPtr, Size, Stream);
      }

      Err = cuMemcpyPeerAsync((CUdeviceptr)DstPtr, DeviceData[DstDevId].Context,
                              (CUdeviceptr)SrcPtr, DeviceData[SrcDevId].Context,
                              Size, Stream);
      if (Err == CUDA_SUCCESS)
        return OFFLOAD_SUCCESS;

      DP("Error returned from cuMemcpyPeerAsync. src_ptr = " DPxMOD
         ", src_id =%" PRId32 ", dst_ptr = " DPxMOD ", dst_id =%" PRId32 "\n",
         DPxPTR(SrcPtr), SrcDevId, DPxPTR(DstPtr), DstDevId);
      CUDA_ERR_STRING(Err);
    }

    return memcpyDtoD(SrcPtr, DstPtr, Size, Stream);
  }

  int dataDelete(const int DeviceId, void *TgtPtr) const {
    CUresult Err = cuCtxSetCurrent(DeviceData[DeviceId].Context);
    if (!checkResult(Err, "Error returned from cuCtxSetCurrent\n"))
      return OFFLOAD_FAIL;

    Err = cuMemFree((CUdeviceptr)TgtPtr);
    if (!checkResult(Err, "Error returned from cuMemFree\n"))
      return OFFLOAD_FAIL;

    return OFFLOAD_SUCCESS;
  }

  int runTargetTeamRegion(const int DeviceId, void *TgtEntryPtr, void **TgtArgs,
                          ptrdiff_t *TgtOffsets, const int ArgNum,
                          const int TeamNum, const int ThreadLimit,
                          const unsigned int LoopTripCount,
                          __tgt_async_info *AsyncInfo) const {
    CUresult Err = cuCtxSetCurrent(DeviceData[DeviceId].Context);
    if (!checkResult(Err, "Error returned from cuCtxSetCurrent\n"))
      return OFFLOAD_FAIL;

    // All args are references.
    std::vector<void *> Args(ArgNum);
    std::vector<void *> Ptrs(ArgNum);

    for (int I = 0; I < ArgNum; ++I) {
      Ptrs[I] = (void *)((intptr_t)TgtArgs[I] + TgtOffsets[I]);
      Args[I] = &Ptrs[I];
    }

    KernelTy *KernelInfo = reinterpret_cast<KernelTy *>(TgtEntryPtr);

    int CudaThreadsPerBlock;
    if (ThreadLimit > 0) {
      DP("Setting CUDA threads per block to requested %d\n", ThreadLimit);
      CudaThreadsPerBlock = ThreadLimit;
      // Add master warp if necessary
      if (KernelInfo->ExecutionMode == GENERIC) {
        DP("Adding master warp: +%d threads\n", DeviceData[DeviceId].WarpSize);
        CudaThreadsPerBlock += DeviceData[DeviceId].WarpSize;
      }
    } else {
      DP("Setting CUDA threads per block to default %d\n",
         DeviceData[DeviceId].NumThreads);
      CudaThreadsPerBlock = DeviceData[DeviceId].NumThreads;
    }

    if (CudaThreadsPerBlock > DeviceData[DeviceId].ThreadsPerBlock) {
      DP("Threads per block capped at device limit %d\n",
         DeviceData[DeviceId].ThreadsPerBlock);
      CudaThreadsPerBlock = DeviceData[DeviceId].ThreadsPerBlock;
    }

    if (!KernelInfo->MaxThreadsPerBlock) {
      Err = cuFuncGetAttribute(&KernelInfo->MaxThreadsPerBlock,
                               CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                               KernelInfo->Func);
      if (!checkResult(Err, "Error returned from cuFuncGetAttribute\n"))
        return OFFLOAD_FAIL;
    }

    if (KernelInfo->MaxThreadsPerBlock < CudaThreadsPerBlock) {
      DP("Threads per block capped at kernel limit %d\n",
         KernelInfo->MaxThreadsPerBlock);
      CudaThreadsPerBlock = KernelInfo->MaxThreadsPerBlock;
    }

    unsigned int CudaBlocksPerGrid;
    if (TeamNum <= 0) {
      if (LoopTripCount > 0 && EnvNumTeams < 0) {
        if (KernelInfo->ExecutionMode == SPMD) {
          // We have a combined construct, i.e. `target teams distribute
          // parallel for [simd]`. We launch so many teams so that each thread
          // will execute one iteration of the loop. round up to the nearest
          // integer
          CudaBlocksPerGrid = ((LoopTripCount - 1) / CudaThreadsPerBlock) + 1;
        } else {
          // If we reach this point, then we have a non-combined construct, i.e.
          // `teams distribute` with a nested `parallel for` and each team is
          // assigned one iteration of the `distribute` loop. E.g.:
          //
          // #pragma omp target teams distribute
          // for(...loop_tripcount...) {
          //   #pragma omp parallel for
          //   for(...) {}
          // }
          //
          // Threads within a team will execute the iterations of the `parallel`
          // loop.
          CudaBlocksPerGrid = LoopTripCount;
        }
        DP("Using %d teams due to loop trip count %" PRIu32
           " and number of threads per block %d\n",
           CudaBlocksPerGrid, LoopTripCount, CudaThreadsPerBlock);
      } else {
        DP("Using default number of teams %d\n", DeviceData[DeviceId].NumTeams);
        CudaBlocksPerGrid = DeviceData[DeviceId].NumTeams;
      }
    } else if (TeamNum > DeviceData[DeviceId].BlocksPerGrid) {
      DP("Capping number of teams to team limit %d\n",
         DeviceData[DeviceId].BlocksPerGrid);
      CudaBlocksPerGrid = DeviceData[DeviceId].BlocksPerGrid;
    } else {
      DP("Using requested number of teams %d\n", TeamNum);
      CudaBlocksPerGrid = TeamNum;
    }

    // Run on the device.
    DP("Launch kernel with %d blocks and %d threads\n", CudaBlocksPerGrid,
       CudaThreadsPerBlock);

    CUstream Stream = getStream(DeviceId, AsyncInfo);
    Err = cuLaunchKernel(KernelInfo->Func, CudaBlocksPerGrid, /* gridDimY */ 1,
                         /* gridDimZ */ 1, CudaThreadsPerBlock,
                         /* blockDimY */ 1, /* blockDimZ */ 1,
                         /* sharedMemBytes */ 0, Stream, &Args[0], nullptr);
    if (!checkResult(Err, "Error returned from cuLaunchKernel\n"))
      return OFFLOAD_FAIL;

    DP("Launch of entry point at " DPxMOD " successful!\n",
       DPxPTR(TgtEntryPtr));

    return OFFLOAD_SUCCESS;
  }

  int synchronize(const int DeviceId, __tgt_async_info *AsyncInfoPtr) const {
    CUstream Stream = reinterpret_cast<CUstream>(AsyncInfoPtr->Queue);
    CUresult Err = cuStreamSynchronize(Stream);
    if (Err != CUDA_SUCCESS) {
      DP("Error when synchronizing stream. stream = " DPxMOD
         ", async info ptr = " DPxMOD "\n",
         DPxPTR(Stream), DPxPTR(AsyncInfoPtr));
      CUDA_ERR_STRING(Err);
      return OFFLOAD_FAIL;
    }

    // Once the stream is synchronized, return it to stream pool and reset
    // async_info. This is to make sure the synchronization only works for its
    // own tasks.
    StreamManager->returnStream(
        DeviceId, reinterpret_cast<CUstream>(AsyncInfoPtr->Queue));
    AsyncInfoPtr->Queue = nullptr;

    return OFFLOAD_SUCCESS;
  }
};

DeviceRTLTy DeviceRTL;
} // namespace

// Exposed library API function
#ifdef __cplusplus
extern "C" {
#endif

int32_t __tgt_rtl_is_valid_binary(__tgt_device_image *image) {
  return elf_check_machine(image, /* EM_CUDA */ 190);
}

int32_t __tgt_rtl_number_of_devices() { return DeviceRTL.getNumOfDevices(); }

int64_t __tgt_rtl_init_requires(int64_t RequiresFlags) {
  DP("Init requires flags to %" PRId64 "\n", RequiresFlags);
  DeviceRTL.setRequiresFlag(RequiresFlags);
  return RequiresFlags;
}

int32_t __tgt_rtl_is_data_exchangable(int32_t src_dev_id, int dst_dev_id) {
  if (DeviceRTL.isValidDeviceId(src_dev_id) &&
      DeviceRTL.isValidDeviceId(dst_dev_id))
    return 1;

  return 0;
}

int32_t __tgt_rtl_init_device(int32_t device_id) {
  assert(DeviceRTL.isValidDeviceId(device_id) && "device_id is invalid");

  return DeviceRTL.initDevice(device_id);
}

__tgt_target_table *__tgt_rtl_load_binary(int32_t device_id,
                                          __tgt_device_image *image) {
  assert(DeviceRTL.isValidDeviceId(device_id) && "device_id is invalid");

  return DeviceRTL.loadBinary(device_id, image);
}

void *__tgt_rtl_data_alloc(int32_t device_id, int64_t size, void *) {
  assert(DeviceRTL.isValidDeviceId(device_id) && "device_id is invalid");

  return DeviceRTL.dataAlloc(device_id, size);
}

int32_t __tgt_rtl_data_submit(int32_t device_id, void *tgt_ptr, void *hst_ptr,
                              int64_t size) {
  assert(DeviceRTL.isValidDeviceId(device_id) && "device_id is invalid");

  __tgt_async_info async_info;
  const int32_t rc = __tgt_rtl_data_submit_async(device_id, tgt_ptr, hst_ptr,
                                                 size, &async_info);
  if (rc != OFFLOAD_SUCCESS)
    return OFFLOAD_FAIL;

  return __tgt_rtl_synchronize(device_id, &async_info);
}

int32_t __tgt_rtl_data_submit_async(int32_t device_id, void *tgt_ptr,
                                    void *hst_ptr, int64_t size,
                                    __tgt_async_info *async_info_ptr) {
  assert(DeviceRTL.isValidDeviceId(device_id) && "device_id is invalid");
  assert(async_info_ptr && "async_info_ptr is nullptr");

  return DeviceRTL.dataSubmit(device_id, tgt_ptr, hst_ptr, size,
                              async_info_ptr);
}

int32_t __tgt_rtl_data_retrieve(int32_t device_id, void *hst_ptr, void *tgt_ptr,
                                int64_t size) {
  assert(DeviceRTL.isValidDeviceId(device_id) && "device_id is invalid");

  __tgt_async_info async_info;
  const int32_t rc = __tgt_rtl_data_retrieve_async(device_id, hst_ptr, tgt_ptr,
                                                   size, &async_info);
  if (rc != OFFLOAD_SUCCESS)
    return OFFLOAD_FAIL;

  return __tgt_rtl_synchronize(device_id, &async_info);
}

int32_t __tgt_rtl_data_retrieve_async(int32_t device_id, void *hst_ptr,
                                      void *tgt_ptr, int64_t size,
                                      __tgt_async_info *async_info_ptr) {
  assert(DeviceRTL.isValidDeviceId(device_id) && "device_id is invalid");
  assert(async_info_ptr && "async_info_ptr is nullptr");

  return DeviceRTL.dataRetrieve(device_id, hst_ptr, tgt_ptr, size,
                                async_info_ptr);
}

int32_t __tgt_rtl_data_exchange_async(int32_t src_dev_id, void *src_ptr,
                                      int dst_dev_id, void *dst_ptr,
                                      int64_t size,
                                      __tgt_async_info *async_info_ptr) {
  assert(DeviceRTL.isValidDeviceId(src_dev_id) && "src_dev_id is invalid");
  assert(DeviceRTL.isValidDeviceId(dst_dev_id) && "dst_dev_id is invalid");
  assert(async_info_ptr && "async_info_ptr is nullptr");

  return DeviceRTL.dataExchange(src_dev_id, src_ptr, dst_dev_id, dst_ptr, size,
                                async_info_ptr);
}

int32_t __tgt_rtl_data_exchange(int32_t src_dev_id, void *src_ptr,
                                int32_t dst_dev_id, void *dst_ptr,
                                int64_t size) {
  assert(DeviceRTL.isValidDeviceId(src_dev_id) && "src_dev_id is invalid");
  assert(DeviceRTL.isValidDeviceId(dst_dev_id) && "dst_dev_id is invalid");

  __tgt_async_info async_info;
  const int32_t rc = __tgt_rtl_data_exchange_async(
      src_dev_id, src_ptr, dst_dev_id, dst_ptr, size, &async_info);
  if (rc != OFFLOAD_SUCCESS)
    return OFFLOAD_FAIL;

  return __tgt_rtl_synchronize(src_dev_id, &async_info);
}

int32_t __tgt_rtl_data_delete(int32_t device_id, void *tgt_ptr) {
  assert(DeviceRTL.isValidDeviceId(device_id) && "device_id is invalid");

  return DeviceRTL.dataDelete(device_id, tgt_ptr);
}

int32_t __tgt_rtl_run_target_team_region(int32_t device_id, void *tgt_entry_ptr,
                                         void **tgt_args,
                                         ptrdiff_t *tgt_offsets,
                                         int32_t arg_num, int32_t team_num,
                                         int32_t thread_limit,
                                         uint64_t loop_tripcount) {
  assert(DeviceRTL.isValidDeviceId(device_id) && "device_id is invalid");

  __tgt_async_info async_info;
  const int32_t rc = __tgt_rtl_run_target_team_region_async(
      device_id, tgt_entry_ptr, tgt_args, tgt_offsets, arg_num, team_num,
      thread_limit, loop_tripcount, &async_info);
  if (rc != OFFLOAD_SUCCESS)
    return OFFLOAD_FAIL;

  return __tgt_rtl_synchronize(device_id, &async_info);
}

int32_t __tgt_rtl_run_target_team_region_async(
    int32_t device_id, void *tgt_entry_ptr, void **tgt_args,
    ptrdiff_t *tgt_offsets, int32_t arg_num, int32_t team_num,
    int32_t thread_limit, uint64_t loop_tripcount,
    __tgt_async_info *async_info_ptr) {
  assert(DeviceRTL.isValidDeviceId(device_id) && "device_id is invalid");

  return DeviceRTL.runTargetTeamRegion(
      device_id, tgt_entry_ptr, tgt_args, tgt_offsets, arg_num, team_num,
      thread_limit, loop_tripcount, async_info_ptr);
}

int32_t __tgt_rtl_run_target_region(int32_t device_id, void *tgt_entry_ptr,
                                    void **tgt_args, ptrdiff_t *tgt_offsets,
                                    int32_t arg_num) {
  assert(DeviceRTL.isValidDeviceId(device_id) && "device_id is invalid");

  __tgt_async_info async_info;
  const int32_t rc = __tgt_rtl_run_target_region_async(
      device_id, tgt_entry_ptr, tgt_args, tgt_offsets, arg_num, &async_info);
  if (rc != OFFLOAD_SUCCESS)
    return OFFLOAD_FAIL;

  return __tgt_rtl_synchronize(device_id, &async_info);
}

int32_t __tgt_rtl_run_target_region_async(int32_t device_id,
                                          void *tgt_entry_ptr, void **tgt_args,
                                          ptrdiff_t *tgt_offsets,
                                          int32_t arg_num,
                                          __tgt_async_info *async_info_ptr) {
  assert(DeviceRTL.isValidDeviceId(device_id) && "device_id is invalid");

  return __tgt_rtl_run_target_team_region_async(
      device_id, tgt_entry_ptr, tgt_args, tgt_offsets, arg_num,
      /* team num*/ 1, /* thread_limit */ 1, /* loop_tripcount */ 0,
      async_info_ptr);
}

int32_t __tgt_rtl_synchronize(int32_t device_id,
                              __tgt_async_info *async_info_ptr) {
  assert(DeviceRTL.isValidDeviceId(device_id) && "device_id is invalid");
  assert(async_info_ptr && "async_info_ptr is nullptr");
  assert(async_info_ptr->Queue && "async_info_ptr->Queue is nullptr");

  return DeviceRTL.synchronize(device_id, async_info_ptr);
}

#ifdef __cplusplus
}
#endif
