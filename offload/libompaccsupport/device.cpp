//===--------- device.cpp - Target independent OpenMP target RTL ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functionality for managing devices that are handled by RTL plugins.
//
//===----------------------------------------------------------------------===//

#include "device.h"
#include "OffloadEntry.h"
#include "OmpAccError.h"
#include "OpenMP/Mapping.h"
#include "OpenMP/OMPT/Callback.h"
#include "OpenMP/OMPT/Interface.h"
#include "OpenMP/OMPT/OmptCommonDefs.h"
#include "OpenMP/OMPT/OmptTracing.h"
#include "PluginManager.h"
#ifdef OMPT_SUPPORT
// Downstream: provides 'ompt::lookupDeviceTracingFn' and the device pointer to
// user-device-number mapping used when emitting trace records.
#include "OmptDeviceTracing.h"
#endif
#include "Shared/APITypes.h"
#include "Shared/Debug.h"
#include "omptarget.h"
#include "rtl.h"

#include "Shared/EnvironmentVar.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <mutex>
#include <string>
#include <thread>

#ifdef OMPT_SUPPORT
using namespace llvm::omp::target;
using namespace ompt;
#endif

using namespace llvm::omp::target::error;
using namespace llvm::omp::target::plugin;
using namespace llvm::omp::target::debug;

namespace llvm::offload::tmp {
GenericPluginTy *__ol_tgt_GetPluginFromPlatform(ol_platform_handle_t Platform);
int32_t __ol_tgt_GetPluginDeviceId(ol_device_handle_t Device);
RecordReplayTy *__ol_tgt_GetRecordReplay(ol_device_handle_t Device);
llvm::omp::target::plugin::DeviceImageTy *
__ol_tgt_GetDeviceImageFromProgram(ol_program_handle_t Program);
} // namespace llvm::offload::tmp

int HostDataToTargetTy::addEventIfNecessary(DeviceTy &Device,
                                            AsyncInfoTy &AsyncInfo) const {
  // First, check if the user disabled atomic map transfer/malloc/dealloc.
  if (!MappingConfig::get().UseEventsForAtomicTransfers)
    return OFFLOAD_SUCCESS;

  // We cannot assume the event should not be nullptr because we don't
  // know if the target support event. But if a target doesn't,
  // recordEvent should always return success.
  void *Event = getEvent();
  if (Device.recordEvent(&Event, AsyncInfo) != OFFLOAD_SUCCESS) {
    REPORT() << "Failed to set dependence on event " << Event;
    return OFFLOAD_FAIL;
  }

  setEvent(Event);

  return OFFLOAD_SUCCESS;
}

DeviceTy::DeviceTy(int32_t DeviceID, ol_platform_handle_t Platform,
                   ol_device_handle_t DeviceHandle, PluginManager *PM)
    : DeviceID(DeviceID), Platform(Platform), DeviceHandle(DeviceHandle),
      PM(PM), ForceSynchronousTargetRegions(false), MappingInfo(*this) {}

DeviceTy::~DeviceTy() {
  if (DeviceID == -1 || !(getInfoLevel() & OMP_INFOTYPE_DUMP_TABLE))
    return;

  ident_t Loc = {0, 0, 0, 0, ";libomptarget;libomptarget;0;0;;"};
  dumpTargetPointerMappings(&Loc, *this);
}

/// Used to set the asynchronous execution mode
inline void setAsyncInfoSynchronous(__tgt_async_info *AI, bool SetSynchronous) {
  // AI is null if the AsyncInfoTy failed to create its queue.
  if (SetSynchronous && AI)
    AI->ExecAsync = false;
}

llvm::Error DeviceTy::initGeometryProps() {
  auto Query = [&](DeviceInfo Info, auto &Dest) -> llvm::Error {
    if (auto Res =
            olGetDeviceInfo(DeviceHandle, static_cast<ol_device_info_t>(Info),
                            sizeof(Dest), &Dest))
      return createError(ErrorCode::BackendFailure,
                         "failed to get property %d of device %d: %s",
                         static_cast<int>(Info), DeviceID, Res->Details);
    return llvm::Error::success();
  };

  if (auto Err = Query(DeviceInfo::NUM_LANES, GeometryProps.WarpSize))
    return Err;
  // A block is one dimensional, so it is limited both by the maximum number of
  // threads a block can have and by the maximum size of its first dimension.
  uint32_t MaxBlockSize = 0;
  if (auto Err = Query(DeviceInfo::MAX_WORK_GROUP_SIZE, MaxBlockSize))
    return Err;
  ol_dimensions_t MaxBlockSizePerDim{};
  if (auto Err = Query(DeviceInfo::MAX_WORK_GROUP_SIZE_PER_DIMENSION,
                       MaxBlockSizePerDim))
    return Err;
  GeometryProps.ThreadLimit = std::min(MaxBlockSize, MaxBlockSizePerDim.x);
  if (auto Err = Query(DeviceInfo::LAUNCH_DEFAULT_WORK_GROUP_SIZE,
                       GeometryProps.DefaultNumThreads))
    return Err;
  if (auto Err = Query(DeviceInfo::LAUNCH_DEFAULT_NUM_WORK_GROUPS,
                       GeometryProps.DefaultNumBlocks))
    return Err;
  if (auto Err = Query(DeviceInfo::LAUNCH_MAX_NUM_WORK_GROUPS,
                       GeometryProps.MaxNumBlocks))
    return Err;
  if (auto Err =
          Query(DeviceInfo::LAUNCH_MAX_GRID_SIZE, GeometryProps.MaxGridSize))
    return Err;
  // This is the same property reported by omp_get_gprivate_limit().
  if (auto Err = Query(DeviceInfo::WORK_GROUP_LOCAL_MEM_SIZE,
                       GeometryProps.MaxBlockSharedMemSize))
    return Err;
  // Not every device reports its number of compute units; zero turns off the
  // heuristics that need it.
  GeometryProps.NumComputeUnits =
      getInfo<uint32_t>(DeviceInfo::NUM_COMPUTE_UNITS);
  // Some defaults depend on the backend. If it is unknown, the generic ones
  // apply.
  GeometryProps.Backend = Backend;

  return llvm::Error::success();
}

uint32_t DeviceTy::getThreadLimit() const {
  static Int32Envar OMP_TeamsThreadLimit("OMP_TEAMS_THREAD_LIMIT");

  if (OMP_TeamsThreadLimit > 0)
    return std::min(GeometryProps.ThreadLimit, uint32_t(OMP_TeamsThreadLimit));
  return GeometryProps.ThreadLimit;
}

llvm::Error DeviceTy::init() {
  if (auto Res = olCreateContext(1, &DeviceHandle, &Context))
    return createError(ErrorCode::BackendFailure,
                       "failed to create context for device %d: %s", DeviceID,
                       Res->Details);
  IsHostDevice =
      getInfo<ol_device_type_t>(DeviceInfo::TYPE) == OL_DEVICE_TYPE_HOST;

  RTL = llvm::offload::tmp::__ol_tgt_GetPluginFromPlatform(Platform);
  RTLDeviceID = llvm::offload::tmp::__ol_tgt_GetPluginDeviceId(DeviceHandle);

  // Needs to be initialized before the GeometryProps.
  if (olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND, sizeof(Backend),
                        &Backend))
    Backend = OL_PLATFORM_BACKEND_UNKNOWN;

  getInfoString(DeviceInfo::UID, Uid);
  if (Uid.empty())
    ODBG(ODT_Init) << "Failed to get UID for device " << DeviceID;

  if (auto Err = initGeometryProps())
    return Err;

  OMPT_IF_BUILT({
    // Downstream: tools get the plugin device rather than this DeviceTy, since
    // the device tracing entry points (ompt_start_trace, ...) cast it back.
    GenericDeviceTy &GenericDevice = RTL->getDevice(RTLDeviceID);
    auto *DevicePtr = reinterpret_cast<ompt_device_t *>(&GenericDevice);
    // Register the device->user-id mapping unconditionally: a device can be
    // initialized before the tool connects, yet the plugin still needs this to
    // recover the user device number via 'ompt::getDeviceId()' when it later
    // emits trace records.
    ompt::setDeviceId(DevicePtr, DeviceID);
    performIfOmptInitialized({
      llvm::SmallString<32> ComputeUnitKind;
      getInfoString(DeviceInfo::COMPUTE_UNIT_KIND, ComputeUnitKind);
      // 'lookupDeviceTracingFn' is the only lookup that resolves the device
      // tracing entry points (ompt_set_trace_ompt, ompt_start_trace, ...).
      performOmptCallback(device_initialize, DeviceID, ComputeUnitKind.c_str(),
                          DevicePtr, ompt::lookupDeviceTracingFn,
                          /*documentation=*/nullptr);
    });
  });

  // Envar that indicates whether mapped host buffers should be locked
  // automatically. The possible values are boolean (on/off) and a special:
  //   off:       Mapped host buffers are not locked.
  //   on:        Mapped host buffers are locked in a best-effort approach.
  //              Failure to lock the buffers are silent.
  //   mandatory: Mapped host buffers are always locked and failures to lock
  //              a buffer results in a fatal error.
  StringEnvar OMPX_LockMappedBuffers("LIBOMPTARGET_LOCK_MAPPED_HOST_BUFFERS",
                                     "off");
  bool Enabled;
  if (StringParser::parse(OMPX_LockMappedBuffers.get().data(), Enabled)) {
    // Parsed as a boolean value. Enable the feature if necessary.
    LockMappedBuffers = Enabled;
    IgnoreLockMappedFailures = true;
  } else if (OMPX_LockMappedBuffers.get() == "mandatory") {
    // Enable the feature and failures are fatal.
    LockMappedBuffers = true;
    IgnoreLockMappedFailures = false;
  } else {
    // Disable by default.
    ODBG(ODT_Alloc) << "Invalid value LIBOMPTARGET_LOCK_MAPPED_HOST_BUFFERS="
                    << OMPX_LockMappedBuffers.get();
    LockMappedBuffers = false;
    IgnoreLockMappedFailures = true;
  }

  // Enables recording kernels if set.
  BoolEnvar OMPX_RecordKernel("LIBOMPTARGET_RECORD", false);
  if (OMPX_RecordKernel) {
    BoolEnvar OMPX_RecordOutput("LIBOMPTARGET_RECORD_OUTPUT", true);
    Int64Envar OMPX_RecordMemSize("LIBOMPTARGET_RECORD_MEMSIZE",
                                  8 * 1024 * 1024 * 1024ULL);
    Int32Envar OMPX_RecordDevice("LIBOMPTARGET_RECORD_DEVICE", 0);
    StringEnvar OMPX_RecordOutputDir("LIBOMPTARGET_RECORD_DIR", "");
    BoolEnvar OMPX_EmitRecordReport("LIBOMPTARGET_RECORD_REPORT", false);
    StringEnvar OMPX_RecordReportFilename("LIBOMPTARGET_RECORD_REPORT_FILENAME",
                                          "");
    if (OMPX_RecordDevice != RTLDeviceID)
      return llvm::Error::success();

    // Print report if it was enabled explicitly or a report file was indicated.
    bool EmitReport =
        OMPX_EmitRecordReport || !OMPX_RecordReportFilename.get().empty();

    int32_t Ret = RTL->initialize_record_replay(
        RTLDeviceID, OMPX_RecordMemSize, nullptr,
        /*IsRecord=*/true, /*IsNative=*/true, OMPX_RecordOutput, EmitReport,
        OMPX_RecordReportFilename.get().c_str(),
        OMPX_RecordOutputDir.get().c_str());
    if (Ret != OFFLOAD_SUCCESS)
      return createError(ErrorCode::BackendFailure,
                         "failed to initialize RR in device %d\n", DeviceID);
  }

  return llvm::Error::success();
}

llvm::Error DeviceTy::deinit() {
  OMPT_IF_BUILT({
    performIfOmptInitialized(performOmptCallback(device_finalize, DeviceID));
    // Pairs the unconditional 'setDeviceId' in DeviceTy::init().
    ompt::removeDeviceId(
        reinterpret_cast<ompt_device_t *>(&RTL->getDevice(RTLDeviceID)));
  });

  if (!Context)
    return llvm::Error::success();
  if (auto Res = olDestroyContext(Context))
    return createError(ErrorCode::BackendFailure,
                       "failed to destroy context for device %d: %s", DeviceID,
                       Res->Details);
  Context = nullptr;
  return llvm::Error::success();
}

llvm::Expected<void *>
DeviceTy::getAndRecordGlobalAddress(const ProgramTy &Program, const char *Name,
                                    size_t ExpectedSize) {
  size_t Size = 0;
  auto AddrOrErr = Program.getGlobalAddress(Name, &Size);
  if (!AddrOrErr)
    return AddrOrErr;
  // Some backends return size 0 if the size is unknown.
  // Only check the size if it is non-zero.
  if (Size && Size != ExpectedSize)
    return createError(ErrorCode::InvalidBinary,
                       "size of global symbol %s does not match expected size "
                       "(expected %zu, got %zu)",
                       Name, ExpectedSize, Size);

  RecordReplayTy *RecordReplay =
      llvm::offload::tmp::__ol_tgt_GetRecordReplay(DeviceHandle);
  if (RecordReplay && RecordReplay->isRecording())
    RecordReplay->addGlobal(Name, Size, *AddrOrErr);

  return AddrOrErr;
}

// Extract the mapping of host function pointers to device function pointers
// from the entry table. Functions marked as 'indirect' in OpenMP will have
// offloading entries generated for them which map the host's function pointer
// to a global containing the corresponding function pointer on the device.
static llvm::Expected<std::pair<void *, uint64_t>>
setupIndirectCallTable(DeviceTy &Device, __tgt_device_image *Image,
                       const ProgramTy &Program) {
  llvm::ArrayRef<llvm::offloading::EntryTy> Entries(Image->EntriesBegin,
                                                    Image->EntriesEnd);
  llvm::SmallVector<std::pair<void *, void *>> IndirectCallTable;
  for (const auto &Entry : Entries) {
    if (Entry.Kind != llvm::object::OffloadKind::OFK_OpenMP ||
        Entry.Size == 0 ||
        (!(Entry.Flags & OMP_DECLARE_TARGET_INDIRECT) &&
         !(Entry.Flags & OMP_DECLARE_TARGET_INDIRECT_VTABLE)))
      continue;

    size_t PtrSize = sizeof(void *);
    if (Entry.Flags & OMP_DECLARE_TARGET_INDIRECT_VTABLE) {
      // This is a VTable entry, the current entry is the first index of the
      // VTable and Entry.Size is the total size of the VTable. Unlike the
      // indirect function case below, the Global is not of size Entry.Size and
      // is instead of size PtrSize (sizeof(void*)).
      void *res;
      auto VtableOrErr =
          Device.getAndRecordGlobalAddress(Program, Entry.SymbolName, PtrSize);
      if (!VtableOrErr)
        return VtableOrErr.takeError();
      void *Vtable = *VtableOrErr;

      // HstPtr = Entry.Address;
      if (Device.retrieveDataSync(&res, Vtable, PtrSize))
        return createError(ErrorCode::InvalidBinary, "failed to load %s",
                           Entry.SymbolName);
      // Calculate and emplace entire Vtable from first Vtable byte
      for (uint64_t i = 0; i < Entry.Size / PtrSize; ++i) {
        auto &[HstPtr, DevPtr] = IndirectCallTable.emplace_back();
        HstPtr = reinterpret_cast<void *>(
            reinterpret_cast<uintptr_t>(Entry.Address) + i * PtrSize);
        DevPtr = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(res) +
                                          i * PtrSize);
      }
    } else {
      // Indirect function case: Entry.Size should equal PtrSize since we're
      // dealing with a single function pointer (not a VTable)
      assert(Entry.Size == PtrSize && "Global not a function pointer?");
      auto &[HstPtr, DevPtr] = IndirectCallTable.emplace_back();
      auto PtrOrErr = Device.getAndRecordGlobalAddress(
          Program, Entry.SymbolName, Entry.Size);
      if (!PtrOrErr)
        return PtrOrErr.takeError();
      void *Ptr = *PtrOrErr;

      HstPtr = Entry.Address;
      if (Device.retrieveDataSync(&DevPtr, Ptr, Entry.Size))
        return createError(ErrorCode::InvalidBinary, "failed to load %s",
                           Entry.SymbolName);
    }
  }

  // If we do not have any indirect globals we exit early.
  if (IndirectCallTable.empty())
    return std::pair{nullptr, 0};

  // Sort the array to allow for more efficient lookup of device pointers.
  llvm::sort(IndirectCallTable,
             [](const auto &x, const auto &y) { return x.first < y.first; });

  uint64_t TableSize =
      IndirectCallTable.size() * sizeof(std::pair<void *, void *>);
  void *DevicePtr = Device.allocData(TableSize, nullptr, TARGET_ALLOC_DEVICE);
  // The IndirectCallTable is on the stack, so the copy must be synchronous to
  // ensure the data is copied before we return.
  if (Device.submitDataSync(DevicePtr, IndirectCallTable.data(), TableSize,
                            /*Entry=*/nullptr,
                            /*HDTTMapPtr=*/nullptr, /*Loc=*/nullptr,
                            /*Name=*/"IndirectCallTable"))
    return createError(ErrorCode::InvalidBinary, "failed to copy data");

  return std::pair<void *, uint64_t>(DevicePtr, IndirectCallTable.size());
}

// Load binary to device and perform global initialization if needed.
llvm::Expected<ProgramTy> DeviceTy::loadBinary(__tgt_device_image *Img) {
  auto ProgramOrErr = ProgramTy::create(Context, DeviceHandle, Img);
  if (!ProgramOrErr)
    return ProgramOrErr.takeError();
  ProgramTy Program = std::move(*ProgramOrErr);

  // Dispatch this before the fast-reduction probe below: that probe is a
  // downstream addition and its device-to-host read would otherwise be
  // reported to the tool ahead of the load of the image it reads from.
  OMPT_IF_BUILT_AND_INITIALIZED({
    auto &Image = *llvm::offload::tmp::__ol_tgt_GetDeviceImageFromProgram(
        Program.getHandle());
    performOmptCallback(device_load, DeviceID, /*FileName=*/nullptr,
                        /*FileOffset=*/0, /*VmaInFile=*/nullptr,
                        Image.getSize(), const_cast<void *>(Image.getStart()),
                        /*DeviceAddr=*/nullptr, /*ModuleId=*/0);
  });

  AsyncInfoTy AsyncInfo(*this);
  // From the image, read whether fast reduction is enabled (optional symbol).
  auto IsFastReductionEnabledPtrOrErr = getAndRecordGlobalAddress(
      Program, "__omp_plugin_enable_fast_reduction", sizeof(int8_t));
  if (!IsFastReductionEnabledPtrOrErr) {
    llvm::consumeError(IsFastReductionEnabledPtrOrErr.takeError());
  } else {
    int8_t IsFastReductionEnabled;
    if (!retrieveData(&IsFastReductionEnabled, *IsFastReductionEnabledPtrOrErr,
                      sizeof(int8_t), AsyncInfo) &&
        !synchronize(AsyncInfo))
      RTL->getDevice(RTLDeviceID)
          .setIsFastReductionEnabled(IsFastReductionEnabled);
  }

  // This symbol is optional.
  auto DeviceEnvironmentPtrOrErr = getAndRecordGlobalAddress(
      Program, "__omp_rtl_device_environment", sizeof(DeviceEnvironmentTy));
  if (!DeviceEnvironmentPtrOrErr) {
    llvm::consumeError(DeviceEnvironmentPtrOrErr.takeError());
    return Program;
  }
  void *DeviceEnvironmentPtr = *DeviceEnvironmentPtrOrErr;

  // Obtain a table mapping host function pointers to device function pointers.
  auto CallTablePairOrErr = setupIndirectCallTable(*this, Img, Program);
  if (!CallTablePairOrErr)
    return CallTablePairOrErr.takeError();

  // Debug flags for the device runtime.
  static Int32Envar OMPX_DebugKind("LIBOMPTARGET_DEVICE_RTL_DEBUG");

  DeviceEnvironmentTy DeviceEnvironment;
  DeviceEnvironment.DeviceDebugKind = OMPX_DebugKind.get();
  DeviceEnvironment.NumDevices = RTL->getNumDevices();
  // TODO: The device ID used here is not the real device ID used by OpenMP.
  DeviceEnvironment.DeviceNum = RTLDeviceID;
  DeviceEnvironment.DynamicMemSize = 0;
  DeviceEnvironment.ClockFrequency =
      getInfo<uint64_t>(DeviceInfo::TIMESTAMP_FREQUENCY);
  DeviceEnvironment.IndirectCallTable =
      reinterpret_cast<uintptr_t>(CallTablePairOrErr->first);
  DeviceEnvironment.IndirectCallTableSize = CallTablePairOrErr->second;
  DeviceEnvironment.HardwareParallelism =
      getInfo<uint64_t>(DeviceInfo::HARDWARE_PARALLELISM);

  // DeviceEnvironment is on the stack, so the copy must be synchronous.
  if (submitDataSync(DeviceEnvironmentPtr, &DeviceEnvironment,
                     sizeof(DeviceEnvironment),
                     /*Entry=*/nullptr,
                     /*HDTTMapPtr=*/nullptr,
                     /*Loc=*/nullptr,
                     /*Name=*/"DeviceEnvironment"))
    return createError(ErrorCode::InvalidBinary, "failed to copy data");

  return Program;
}

void *DeviceTy::allocData(int64_t Size, void *HstPtr, int32_t Kind) {
  /// RAII to establish tool anchors before and after data allocation
  void *TargetPtr = nullptr;
  OMPT_IF_BUILT(
      InterfaceRAII TargetDataAllocRAII(
          RegionInterface.getCallbacks<ompt_target_data_alloc>(), DeviceID,
          HstPtr, &TargetPtr, Size,
          /*CodePtr=*/OMPT_GET_RETURN_ADDRESS);
      InterfaceRAII TargetDataAllocTraceRAII(
          RegionInterface.getTraceGenerators<ompt_target_data_alloc>(),
          RTLDeviceID, HstPtr, &TargetPtr, Size,
          /*CodePtr=*/OMPT_GET_RETURN_ADDRESS);)

  TargetPtr = RTL->data_alloc(RTLDeviceID, Size, HstPtr, Kind);
  return TargetPtr;
}

int32_t DeviceTy::deleteData(void *TgtAllocBegin, int32_t Kind) {
  /// RAII to establish tool anchors before and after data deletion
  OMPT_IF_BUILT(
      InterfaceRAII TargetDataDeleteRAII(
          RegionInterface.getCallbacks<ompt_target_data_delete>(), DeviceID,
          TgtAllocBegin,
          /*CodePtr=*/OMPT_GET_RETURN_ADDRESS);
      InterfaceRAII TargetDataDeleteTraceRAII(
          RegionInterface.getTraceGenerators<ompt_target_data_delete>(),
          DeviceID, TgtAllocBegin,
          /*CodePtr=*/OMPT_GET_RETURN_ADDRESS);)
  return RTL->data_delete(RTLDeviceID, TgtAllocBegin, Kind);
}

llvm::Expected<void *> DeviceTy::allocCtxMem(int64_t Size,
                                             ol_alloc_type_t Type) {
  void *Ptr = nullptr;
  if (auto Res = olMemAlloc(Context, DeviceHandle, Type, Size, &Ptr))
    return createError(ErrorCode::BackendFailure,
                       "failed to allocate %" PRId64 " bytes on device %d: %s",
                       Size, DeviceID, Res->Details);
  return Ptr;
}

llvm::Error DeviceTy::freeCtxMem(void *Ptr) {
  if (auto Res = olMemFree(Context, Ptr))
    return createError(ErrorCode::BackendFailure,
                       "failed to free memory %p on device %d: %s", Ptr,
                       DeviceID, Res->Details);
  return llvm::Error::success();
}

namespace llvm::offload::tmp {
__tgt_async_info *__ol_tgt_GetAsyncInfoFromQueue(ol_queue_handle_t Queue);
} // namespace llvm::offload::tmp

// Submit data to device
int32_t DeviceTy::submitData(void *TgtPtrBegin, void *HstPtrBegin, int64_t Size,
                             AsyncInfoTy &AsyncInfo, HostDataToTargetTy *Entry,
                             MappingInfoTy::HDTTMapAccessorTy *HDTTMapPtr,
                             const ident_t *Loc, const char *Name) {
  ol_queue_handle_t Queue = AsyncInfo.getQueue();
  if (!Queue)
    return OFFLOAD_FAIL;

  return submitDataImpl(TgtPtrBegin, HstPtrBegin, Size, Queue, Entry,
                        HDTTMapPtr, Loc, Name);
}

// Submit data to device synchronously
int32_t DeviceTy::submitDataSync(void *TgtPtrBegin, void *HstPtrBegin,
                                 int64_t Size, HostDataToTargetTy *Entry,
                                 MappingInfoTy::HDTTMapAccessorTy *HDTTMapPtr,
                                 const ident_t *Loc, const char *Name) {
  // Downstream: use a queue, so that OMPT device tracing records the transfer.
  AsyncInfoTy AsyncInfo(*this);
  if (submitData(TgtPtrBegin, HstPtrBegin, Size, AsyncInfo, Entry, HDTTMapPtr,
                 Loc, Name) != OFFLOAD_SUCCESS)
    return OFFLOAD_FAIL;
  return AsyncInfo.synchronize();
}

int32_t DeviceTy::submitDataImpl(void *TgtPtrBegin, void *HstPtrBegin,
                                 int64_t Size, ol_queue_handle_t Queue,
                                 HostDataToTargetTy *Entry,
                                 MappingInfoTy::HDTTMapAccessorTy *HDTTMapPtr,
                                 const ident_t *Loc, const char *Name) {
  if (getInfoLevel() & OMP_INFOTYPE_DATA_TRANSFER)
    MappingInfo.printCopyInfo(TgtPtrBegin, HstPtrBegin, Size, /*H2D=*/true,
                              Entry, HDTTMapPtr, Loc, Name);

  // Downstream: OMPT device tracing and forced synchronization act on the
  // queue's async info. Transfers without a queue are synchronous and are
  // not traced.
  __tgt_async_info *AI =
      Queue ? llvm::offload::tmp::__ol_tgt_GetAsyncInfoFromQueue(Queue)
            : nullptr;

  /// RAII to establish tool anchors before and after data submit
  OMPT_IF_BUILT(
      InterfaceRAII TargetDataSubmitRAII(
          RegionInterface.getCallbacks<ompt_target_data_transfer_to_device>(),
          omp_initial_device, HstPtrBegin, DeviceID, TgtPtrBegin, Size,
          /*CodePtr=*/OMPT_GET_RETURN_ADDRESS);
      // Only if 'TracedDeviceId' is actually traced, AsyncInfo->OmptEventInfo
      // is set and a trace record generated. Otherwise: No OMPT device tracing.
      TracerInterfaceRAII TargetDataSubmitTraceRAII(
          RegionInterface
              .getTraceGenerators<ompt_target_data_transfer_to_device>(),
          AI, RTL->getProfiler(), /*TracedDeviceId=*/DeviceID,
          /*EventType=*/ompt_callback_target_data_op, omp_initial_device,
          HstPtrBegin, DeviceID, TgtPtrBegin, Size,
          /*CodePtr=*/OMPT_GET_RETURN_ADDRESS);)

  setAsyncInfoSynchronous(AI, ForceSynchronousTargetRegions);
  if (auto Res = olMemcpy(Queue, TgtPtrBegin, DeviceHandle, HstPtrBegin,
                          /*Src=*/nullptr, Size)) {
    REPORT() << "Failure to copy data from host to device. Pointers: host "
             << "= " << HstPtrBegin << ", device = " << TgtPtrBegin
             << ", size = " << Size << ": " << Res->Details;
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

// Retrieve data from device
int32_t DeviceTy::retrieveData(void *HstPtrBegin, void *TgtPtrBegin,
                               int64_t Size, AsyncInfoTy &AsyncInfo,
                               HostDataToTargetTy *Entry,
                               MappingInfoTy::HDTTMapAccessorTy *HDTTMapPtr,
                               const ident_t *Loc) {
  ol_queue_handle_t Queue = AsyncInfo.getQueue();
  if (!Queue)
    return OFFLOAD_FAIL;

  return retrieveDataImpl(HstPtrBegin, TgtPtrBegin, Size, Queue, Entry,
                          HDTTMapPtr, Loc);
}

// Retrieve data from device synchronously
int32_t DeviceTy::retrieveDataSync(void *HstPtrBegin, void *TgtPtrBegin,
                                   int64_t Size, HostDataToTargetTy *Entry,
                                   MappingInfoTy::HDTTMapAccessorTy *HDTTMapPtr,
                                   const ident_t *Loc) {
  // Downstream: use a queue, so that OMPT device tracing records the transfer.
  AsyncInfoTy AsyncInfo(*this);
  if (retrieveData(HstPtrBegin, TgtPtrBegin, Size, AsyncInfo, Entry, HDTTMapPtr,
                   Loc) != OFFLOAD_SUCCESS)
    return OFFLOAD_FAIL;
  return AsyncInfo.synchronize();
}

int32_t DeviceTy::retrieveDataImpl(void *HstPtrBegin, void *TgtPtrBegin,
                                   int64_t Size, ol_queue_handle_t Queue,
                                   HostDataToTargetTy *Entry,
                                   MappingInfoTy::HDTTMapAccessorTy *HDTTMapPtr,
                                   const ident_t *Loc) {
  if (getInfoLevel() & OMP_INFOTYPE_DATA_TRANSFER)
    MappingInfo.printCopyInfo(TgtPtrBegin, HstPtrBegin, Size, /*H2D=*/false,
                              Entry, HDTTMapPtr, Loc);

  // Downstream: OMPT device tracing and forced synchronization act on the
  // queue's async info. Transfers without a queue are synchronous and are
  // not traced.
  __tgt_async_info *AI =
      Queue ? llvm::offload::tmp::__ol_tgt_GetAsyncInfoFromQueue(Queue)
            : nullptr;

  /// RAII to establish tool anchors before and after data retrieval
  OMPT_IF_BUILT(
      InterfaceRAII TargetDataRetrieveRAII(
          RegionInterface.getCallbacks<ompt_target_data_transfer_from_device>(),
          DeviceID, TgtPtrBegin, omp_initial_device, HstPtrBegin, Size,
          /*CodePtr=*/OMPT_GET_RETURN_ADDRESS);
      // Only if 'TracedDeviceId' is actually traced, AsyncInfo->OmptEventInfo
      // is set and a trace record generated. Otherwise: No OMPT device tracing.
      TracerInterfaceRAII TargetDataSubmitTraceRAII(
          RegionInterface
              .getTraceGenerators<ompt_target_data_transfer_from_device>(),
          AI, RTL->getProfiler(), /*TracedDeviceId=*/DeviceID,
          /*EventType=*/ompt_callback_target_data_op, DeviceID, TgtPtrBegin,
          omp_initial_device, HstPtrBegin, Size,
          /*CodePtr=*/OMPT_GET_RETURN_ADDRESS);)

  setAsyncInfoSynchronous(AI, ForceSynchronousTargetRegions);
  if (auto Res = olMemcpy(Queue, HstPtrBegin, /*Dst=*/nullptr, TgtPtrBegin,
                          DeviceHandle, Size)) {
    REPORT() << "Failure to copy data from device to host. Pointers: host "
             << "= " << HstPtrBegin << ", device = " << TgtPtrBegin
             << ", size = " << Size << ": " << Res->Details;
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

// Copy data from current device to destination device directly
int32_t DeviceTy::dataExchange(void *SrcPtr, DeviceTy &DstDev, void *DstPtr,
                               int64_t Size, AsyncInfoTy &AsyncInfo) {
  /// RAII to establish tool anchors before and after data exchange
  /// Note: Despite the fact that this is a data exchange, we use 'from_device'
  ///       operation enum (w.r.t. ompt_target_data_op_t) as there is currently
  ///       no better alternative. It is still possible to distinguish this
  ///       scenario from a real data retrieve by checking if both involved
  ///       device numbers are less than omp_get_num_devices().
  OMPT_IF_BUILT(
      InterfaceRAII TargetDataExchangeRAII(
          RegionInterface.getCallbacks<ompt_target_data_transfer_from_device>(),
          RTLDeviceID, SrcPtr, DstDev.RTLDeviceID, DstPtr, Size,
          /*CodePtr=*/OMPT_GET_RETURN_ADDRESS);
      // Only if 'TracedDeviceId' is actually traced, AsyncInfo->OmptEventInfo
      // is set and a trace record generated. Otherwise: No OMPT device tracing.
      TracerInterfaceRAII TargetDataExchangeTraceRAII(
          RegionInterface
              .getTraceGenerators<ompt_target_data_transfer_from_device>(),
          AsyncInfo, RTL->getProfiler(), /*TracedDeviceId=*/RTLDeviceID,
          /*EventType=*/ompt_callback_target_data_op, RTLDeviceID, SrcPtr,
          DstDev.RTLDeviceID, DstPtr, Size,
          /*CodePtr=*/OMPT_GET_RETURN_ADDRESS);)

  setAsyncInfoSynchronous(AsyncInfo, ForceSynchronousTargetRegions);
  ol_queue_handle_t queue = AsyncInfo.getQueue();
  if (!queue)
    return OFFLOAD_FAIL;
  if (auto Res = olMemcpy(queue, DstPtr, DstDev.DeviceHandle, SrcPtr,
                          DeviceHandle, Size)) {
    REPORT() << "Failure to copy data from device (" << RTLDeviceID
             << ") to device (" << DstDev.RTLDeviceID
             << "). Pointers: host = " << SrcPtr << ", device = " << DstPtr
             << ", size = " << Size << ": " << Res->Details;
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

llvm::Expected<void *> DeviceTy::registerMemory(void *HstPtr, int64_t Size,
                                                bool LockMemory) {
  void *LockedPtr = nullptr;
  ol_memory_register_flags_t Flags =
      LockMemory ? OL_MEMORY_REGISTER_FLAG_LOCK_MEMORY : 0;
  if (auto Res = olMemRegister(DeviceHandle, HstPtr, Size, Flags, &LockedPtr))
    return createError(ErrorCode::BackendFailure,
                       "failed to lock memory %p: %s", HstPtr, Res->Details);
  return LockedPtr;
}

llvm::Error DeviceTy::unregisterMemory(void *HstPtr, bool UnlockMemory) {
  ol_memory_register_flags_t Flags =
      UnlockMemory ? OL_MEMORY_REGISTER_FLAG_UNLOCK_MEMORY : 0;
  if (auto Res = olMemUnregister(DeviceHandle, HstPtr, Flags))
    return createError(ErrorCode::BackendFailure,
                       "failed to unlock memory %p: %s", HstPtr, Res->Details);
  return llvm::Error::success();
}

int32_t DeviceTy::notifyDataMapped(void *HstPtr, int64_t Size) {
  ODBG(ODT_Mapping) << "Notifying about new mapping: HstPtr=" << HstPtr
                    << ", Size=" << Size;

  auto LockedPtrOrErr = registerMemory(HstPtr, Size, LockMappedBuffers);
  if (!LockedPtrOrErr) {
    if (!IgnoreLockMappedFailures) {
      REPORT() << "Notifying about data mapping failed: "
               << llvm::toString(LockedPtrOrErr.takeError());
      return OFFLOAD_FAIL;
    }
    llvm::consumeError(LockedPtrOrErr.takeError());
  }
  return OFFLOAD_SUCCESS;
}

int32_t DeviceTy::notifyDataUnmapped(void *HstPtr) {
  ODBG(ODT_Mapping) << "Notifying about an unmapping: HstPtr=" << HstPtr;

  if (auto Err = unregisterMemory(HstPtr, LockMappedBuffers)) {
    if (!IgnoreLockMappedFailures) {
      REPORT() << "Notifying about data unmapping failed: "
               << llvm::toString(std::move(Err));
      return OFFLOAD_FAIL;
    }
    llvm::consumeError(std::move(Err));
  }
  return OFFLOAD_SUCCESS;
}

namespace {
/// The launch configuration for a kernel.
struct LaunchConfigTy {
  /// Launch geometry and the amount of dynamic shared memory to request.
  ol_kernel_launch_size_args_t Size = {/*Dimensions=*/3, {}, {}, 0};
  /// Number of kernel arguments in \p Args.
  uint32_t NumArgs = 0;
  /// Array of \p NumArgs pointers, each pointing at one argument's value,
  /// with any offsets already resolved.
  void **Args = nullptr;
  /// Size in bytes of each argument, one entry per \p Args element, or null
  /// if this launch does not know the argument sizes.
  const size_t *ArgSizes = nullptr;
  /// Whether this kernel was spawned as cooperative.
  bool Cooperative = false;
};
} // namespace

/// Resolve \p NumArgs (base pointer, offset) pairs into a flattened array of
/// argument-value pointers suitable for a kernel launch, writing the result
/// into \p Config.NumArgs/Args.
static void resolveKernelLaunchParams(void **const TgtArgs,
                                      ptrdiff_t *const TgtOffsets,
                                      uint32_t NumArgs,
                                      llvm::SmallVector<void *> &Args,
                                      llvm::SmallVector<void *> &Ptrs,
                                      LaunchConfigTy &Config) {
  Config.NumArgs = NumArgs;
  Args.resize(NumArgs);
  Ptrs.resize(NumArgs);

  if (NumArgs == 0)
    return;

  for (uint32_t I = 0; I < NumArgs; ++I) {
    Args[I] = reinterpret_cast<void *>(reinterpret_cast<intptr_t>(TgtArgs[I]) +
                                       TgtOffsets[I]);
    Ptrs[I] = &Args[I];
  }

  Config.Args = &Ptrs[0];
}

namespace {
/// Configuration of dynamic block memory needed for launching a kernel.
struct DynBlockMemConfTy {
  /// The size of the dynamic block memory buffer.
  uint32_t Size = 0;
  /// The size of dynamic shared memory natively provided by the device.
  uint32_t NativeSize = 0;
  /// The fallback that was triggered (if any).
  DynCGroupMemFallbackType Fallback = DynCGroupMemFallbackType::None;
  /// The fallback pointer if global memory was used as alternative.
  void *FallbackPtr = nullptr;
};
} // namespace

/// Prepare the block memory buffer requested for the kernel and execute the
/// specified fallback if necessary.
static llvm::Expected<DynBlockMemConfTy>
prepareBlockMemory(DeviceTy &Device, const KernelLaunchInfoTy &KernelEnv,
                   uint32_t DynCGroupMem,
                   DynCGroupMemFallbackType DynCGroupMemFallback,
                   uint64_t NumBlocks) {

  // If the fallback is abort, don't try to adjust the block memory size.
  if (DynCGroupMemFallback == DynCGroupMemFallbackType::Abort)
    return DynBlockMemConfTy{DynCGroupMem, DynCGroupMem,
                             DynCGroupMemFallbackType::Abort, nullptr};

  uint32_t MaxBlockMemSize = Device.getMaxBlockSharedMemSize();
  uint32_t DynBlockMemSize = DynCGroupMem;
  uint32_t TotalBlockMemSize = KernelEnv.StaticBlockMemSize + DynBlockMemSize;
  uint32_t DynNativeBlockMemSize = DynBlockMemSize;
  void *DynFallbackPtr = nullptr;

  DynCGroupMemFallbackType DynFallback = DynCGroupMemFallbackType::None;
  if (DynBlockMemSize && TotalBlockMemSize > MaxBlockMemSize) {
    // Launch without native dynamic block memory.
    DynNativeBlockMemSize = 0;
    DynFallback = DynCGroupMemFallback;
    if (DynFallback != DynCGroupMemFallbackType::DefaultMem) {
      // Do not provide any memory as fallback.
      DynBlockMemSize = 0;
    } else {
      // Get global memory as fallback.
      auto AllocOrErr = Device.allocCtxMem(
          NumBlocks * uint64_t(DynBlockMemSize), OL_ALLOC_TYPE_DEVICE);
      if (!AllocOrErr)
        return AllocOrErr.takeError();
      DynFallbackPtr = *AllocOrErr;
    }
  }
  return DynBlockMemConfTy{DynBlockMemSize, DynNativeBlockMemSize, DynFallback,
                           DynFallbackPtr};
}

static void freeAfterSynchronization(DeviceTy &Device, AsyncInfoTy &AsyncInfo,
                                     void *Ptr) {
  AsyncInfo.addPostProcessingFunction([&Device, Ptr]() -> int {
    if (auto Err = Device.freeCtxMem(Ptr)) {
      REPORT() << "Failure to free device memory " << Ptr << ": "
               << toString(std::move(Err));
      return OFFLOAD_FAIL;
    }
    return OFFLOAD_SUCCESS;
  });
}

/// Downstream: free \p Ptr, which was allocated through the plugin's
/// allocator with \p Kind, once \p AsyncInfo is synchronized.
static void freePluginMemAfterSynchronization(GenericDeviceTy &GenericDevice,
                                              AsyncInfoTy &AsyncInfo, void *Ptr,
                                              TargetAllocTy Kind) {
  AsyncInfo.addPostProcessingFunction([&GenericDevice, Ptr, Kind]() -> int {
    if (auto Err = GenericDevice.dataDelete(Ptr, Kind)) {
      REPORT() << "Failure to free memory " << Ptr << ": "
               << toString(std::move(Err));
      return OFFLOAD_FAIL;
    }
    return OFFLOAD_SUCCESS;
  });
}

/// Return a device pointer to a new kernel launch environment, or null if
/// this launch has no reserved dyn_ptr slot to store one in. \p OmpABIVersion
/// is the kernel-argument ABI version this launch was built against.
/// \p NumBlocks0 is the number of blocks for this launch and is used to size
/// the reduction buffer.
static llvm::Expected<KernelLaunchEnvironmentTy *>
getKernelLaunchEnvironment(DeviceTy &Device, uint32_t OmpABIVersion,
                           const KernelLaunchInfoTy &KernelEnv,
                           const DynBlockMemConfTy &DynBlockMemConf,
                           uint32_t DynCGroupMem, void **DynPtrSlot,
                           AsyncInfoTy &AsyncInfo, uint32_t NumBlocks0) {
  RecordReplayTy *RecordReplay =
      llvm::offload::tmp::__ol_tgt_GetRecordReplay(Device.DeviceHandle);

  // Ctor/Dtor have no arguments, replaying uses the original kernel launch
  // environment, and launches with no reserved dyn_ptr slot (e.g. older
  // compiler versions, or non-OpenMP launches) have nowhere to store one.
  if ((RecordReplay && RecordReplay->isReplaying()) || !DynPtrSlot)
    return nullptr;

  // Downstream: specialized kernels don't use the kernel launch environment.
  // Since the dynamic pointer is still generated by the compiler for these
  // execution modes, ~0 is returned. Note that a cross-team reduction kernel
  // must not be listed here: it is a plain SPMD kernel that needs the launch
  // environment for its reduction buffer, which is allocated below.
  if (KernelEnv.isBigJumpLoopMode() || KernelEnv.isNoLoopMode())
    return reinterpret_cast<KernelLaunchEnvironmentTy *>(~0);

  const bool NeedsReductionBuffer = KernelEnv.ReductionDataSize != 0;
  if (NeedsReductionBuffer && OmpABIVersion < OMP_KERNEL_ARG_VERSION)
    return createError(
        ErrorCode::InvalidBinary,
        "kernel was built against an older OpenMP kernel-launch-environment "
        "ABI (v%u); current runtime requires v%u for cross-team reductions",
        OmpABIVersion, OMP_KERNEL_ARG_VERSION);
  if (!NeedsReductionBuffer && !DynCGroupMem)
    return reinterpret_cast<KernelLaunchEnvironmentTy *>(~0);

  auto AllocOrErr = Device.allocCtxMem(sizeof(KernelLaunchEnvironmentTy),
                                       OL_ALLOC_TYPE_DEVICE);
  if (!AllocOrErr)
    return AllocOrErr.takeError();

  // Remember to free the memory later.
  freeAfterSynchronization(Device, AsyncInfo, *AllocOrErr);

  // Use the KLE in the __tgt_async_info to ensure a stable address for the
  // async data transfer.
  auto &LocalKLE =
      static_cast<__tgt_async_info *>(AsyncInfo)->KernelLaunchEnvironment;
  LocalKLE = KernelLaunchEnvironmentTy{};
  LocalKLE.DynCGroupMemSize = DynBlockMemConf.Size;
  LocalKLE.DynCGroupMemFbPtr = DynBlockMemConf.FallbackPtr;
  LocalKLE.DynCGroupMemFb = DynBlockMemConf.Fallback;
  LocalKLE.ReductionBuffer = nullptr;

  if (NeedsReductionBuffer) {
    // Use number of teams many buffer elements.
    auto ReductionAllocOrErr =
        Device.allocCtxMem(uint64_t(KernelEnv.ReductionDataSize) * NumBlocks0,
                           OL_ALLOC_TYPE_DEVICE);
    if (!ReductionAllocOrErr)
      return ReductionAllocOrErr.takeError();
    LocalKLE.ReductionBuffer = *ReductionAllocOrErr;
    // Remember to free the memory later.
    freeAfterSynchronization(Device, AsyncInfo, *ReductionAllocOrErr);
  }

  // Downstream: stage the launch environment in pinned memory if the plugin
  // transfers it faster from there, so the copy below can take that path.
  GenericDeviceTy &GenericDevice = Device.RTL->getDevice(Device.RTLDeviceID);
  const void *LaunchEnvSrc = &LocalKLE;
  KernelLaunchEnvironmentTy *PinnedKLE =
      GenericDevice.getPinnedLaunchEnvBuffer();
  if (PinnedKLE) {
    *PinnedKLE = LocalKLE;
    LaunchEnvSrc = PinnedKLE;
  }

  // Downstream: the ProfilerData at this point will have a callback for a
  // kernel launch, not a data-op. This is due to the "external" operation
  // being a kernel launch and the data submit here being an implementation
  // detail. We temporarily set the ProfilerData to nullptr, such that we
  // disable the timing etc. further down to not trigger assertions or report
  // implementation detail. For the same reason, copy with olMemcpy rather
  // than Device.submitData(), which would report an OMPT data operation.
  __tgt_async_info *AI = AsyncInfo;
  void *ProfilerData = AI->ProfilerData;
  AI->ProfilerData = nullptr;
  auto Res = olMemcpy(AsyncInfo.getQueue(), *AllocOrErr, Device.DeviceHandle,
                      LaunchEnvSrc, /*SrcDevice=*/nullptr,
                      sizeof(KernelLaunchEnvironmentTy));
  AI->ProfilerData = ProfilerData;
  if (Res)
    return createError(ErrorCode::BackendFailure,
                       "failed to copy the kernel launch environment: %s",
                       Res->Details);

  // Register the staging buffer only now that the transfer reading it has been
  // issued. Registering it earlier would let a concurrent finalization that
  // observes the queue as complete release it while the transfer is still in
  // flight.
  if (PinnedKLE)
    freePluginMemAfterSynchronization(GenericDevice, AsyncInfo, PinnedKLE,
                                      TARGET_ALLOC_HOST);

  return static_cast<KernelLaunchEnvironmentTy *>(*AllocOrErr);
}

/// Get the effective number of threads for the kernel based on the
/// user-defined number of threads.
static uint32_t getEffectiveNumThreads(const DeviceGeometryPropsTy &Props,
                                       uint32_t UserThreadLimit,
                                       const KernelLaunchInfoTy &KernelEnv) {
  assert(!KernelEnv.isBareMode() &&
         "bare kernel should not call this function");

  if (UserThreadLimit > 0 && KernelEnv.isGenericMode()) {
    // Downstream: a thread limit of -1 requests the preferred number of
    // threads.
    if (UserThreadLimit == (uint32_t)-1)
      UserThreadLimit = KernelEnv.PreferredNumThreads;
    else
      UserThreadLimit += Props.WarpSize;
  }

  return std::min(KernelEnv.MaxNumThreads, (UserThreadLimit > 0)
                                               ? UserThreadLimit
                                               : KernelEnv.PreferredNumThreads);
}

/// Get the number of teams the user set with OMP_NUM_TEAMS, or zero.
static uint32_t getOMPNumTeams() {
  static Int32Envar OMP_NumTeams("OMP_NUM_TEAMS");
  return OMP_NumTeams > 0 ? uint32_t(OMP_NumTeams) : 0;
}

/// Get the maximum number of blocks the device can launch for a kernel using
/// \p NumThreads threads per block, further limited by OMP_NUM_TEAMS if the
/// user set it.
static uint32_t getDeviceBlockLimit(const DeviceGeometryPropsTy &Props,
                                    uint32_t NumThreads) {
  uint32_t BlockLimit = Props.MaxNumBlocks;
  // Some devices limit the total number of threads of a launch, which makes
  // the block limit depend on the thread count.
  if (Props.MaxGridSize && NumThreads > 0)
    BlockLimit = std::min<uint64_t>(BlockLimit, Props.MaxGridSize / NumThreads);
  if (uint32_t NumTeams = getOMPNumTeams())
    return std::min(BlockLimit, NumTeams);
  return BlockLimit;
}

/// Threshold for the reduction data size (in bytes) above which a cross-team
/// reduction classifies as "large".
static constexpr uint32_t LargeTeamsReductionBytes = 32;

/// Downstream: maximum number of blocks of a big-jump-loop kernel.
static constexpr uint64_t BigJumpLoopMaxNumBlocks = 1024 * 1024;

/// Get the number of blocks (each having \p NumThreads threads) to launch a
/// kernel with a cross-team reduction, or zero to handle it like a
/// non-reduction kernel.
///
/// Every block adds to the cost of combining the partial results of the blocks.
/// For a "large" reduction (see \p LargeTeamsReductionBytes), this cost
/// dominates -> one block / compute unit. A small/scalar reduction prefers more
/// blocks -> as many as needed given the device's number of waves per compute
/// unit, but at least one / compute unit. A loop too short to give every
/// compute unit a block is handled like a non-reduction kernel.
static uint64_t
getXTeamReductionNumBlocks(const DeviceGeometryPropsTy &Props,
                           uint32_t NumThreads, uint64_t LoopTripCount,
                           const KernelLaunchInfoTy &KernelEnv) {
  if (!KernelEnv.ReductionDataSize || !NumThreads)
    return 0;

  // In Generic and Generic-SPMD mode, each block runs one iteration of the
  // distribute loop. For these modes, there currently is only performance data
  // for large reductions.
  bool IsLarge = KernelEnv.ReductionDataSize > LargeTeamsReductionBytes;
  bool IsSPMD = KernelEnv.isSPMDMode();
  if (!IsSPMD && !(IsLarge && (KernelEnv.isGenericMode() ||
                               KernelEnv.isGenericSPMDMode())))
    return 0;

  // Envar to override the device's number of waves per compute unit for a
  // kernel with a scalar cross-team reduction. Zero keeps the device's value.
  static UInt32Envar OMPX_TeamsReductionWavesPerCU(
      "LIBOMPTARGET_TEAMS_REDUCTION_WAVES_PER_CU", 0);
  uint32_t ScalarWavesPerCU = OMPX_TeamsReductionWavesPerCU.get();
  // Measured on MI210, MI300A, MI300X, MI350X, and MI450: scalar reductions
  // with many iterations prefer more waves, ones with fewer iterations or more
  // memory traffic prefer less waves. 16 waves were never far from the best.
  if (!ScalarWavesPerCU && Props.Backend == OL_PLATFORM_BACKEND_AMDGPU)
    ScalarWavesPerCU = 16;
  uint64_t NumCUs = Props.NumComputeUnits;
  if (!ScalarWavesPerCU || !NumCUs)
    return 0;

  uint32_t WavesPerBlock = llvm::divideCeil(NumThreads, Props.WarpSize);
  uint64_t BlocksPerCU =
      IsLarge ? 1 : std::max(1u, ScalarWavesPerCU / WavesPerBlock);
  uint64_t NumBlocks = NumCUs * BlocksPerCU;
  if (LoopTripCount > 0) {
    // No more blocks than there is work for.
    uint64_t TripCountNumBlocks =
        IsSPMD ? llvm::divideCeil(LoopTripCount, NumThreads) : LoopTripCount;
    if (TripCountNumBlocks < NumCUs)
      return 0;
    NumBlocks = std::min(NumBlocks, TripCountNumBlocks);
  }
  return NumBlocks;
}

/// Get the default number of blocks for a kernel in Generic-SPMD mode.
static uint32_t
getDefaultNumBlocksGenericSPMD(const DeviceGeometryPropsTy &Props) {
  // Envar to set the default number of blocks per compute unit for a kernel in
  // Generic-SPMD mode. Currently only used on AMDGPU devices. Zero uses the
  // default number of blocks of other kernels.
  static UInt32Envar OMPX_GenericSPMDTeamsPerCU(
      "LIBOMPTARGET_GENERIC_SPMD_TEAMS_PER_CU", 6);
  // Downstream: the former name of the envar, honored if the new one is unset.
  static UInt32Envar OMPX_AMDGPUGenericSPMDTeamsPerCU(
      "LIBOMPTARGET_AMDGPU_GENERIC_SPMD_TEAMS_PER_CU", 6);
  uint32_t TeamsPerCU = !OMPX_GenericSPMDTeamsPerCU.isPresent() &&
                                OMPX_AMDGPUGenericSPMDTeamsPerCU.isPresent()
                            ? OMPX_AMDGPUGenericSPMDTeamsPerCU.get()
                            : OMPX_GenericSPMDTeamsPerCU.get();

  if (Props.Backend != OL_PLATFORM_BACKEND_AMDGPU || !TeamsPerCU ||
      !Props.NumComputeUnits)
    return Props.DefaultNumBlocks;
  return Props.NumComputeUnits * TeamsPerCU;
}

/// Get the number of blocks for each thread to run one iteration of a loop of
/// \p LoopTripCount iterations. If that results in fewer than
/// \p DefaultNumBlocks blocks, lower the number of threads per block
/// \p EffectiveNumThreads, unless the user set it, to increase the outer
/// parallelism.
static uint64_t getOneIterationPerThreadNumBlocks(uint64_t LoopTripCount,
                                                  uint64_t DefaultNumBlocks,
                                                  uint32_t &EffectiveNumThreads,
                                                  bool IsNumThreadsStrict,
                                                  bool IsNumThreadsFromUser) {
  assert(LoopTripCount > 0 && "Expected a known loop trip count");

  // Envar that sets the minimum number of threads to use for such a loop.
  static UInt32Envar OMPX_MinThreadsForLowTripCount(
      "LIBOMPTARGET_MIN_THREADS_FOR_LOW_TRIP_COUNT", 32);

  uint32_t MinThreads =
      std::min(OMPX_MinThreadsForLowTripCount.get(), EffectiveNumThreads);

  // Honor the thread_limit clause; only lower the number of threads.
  [[maybe_unused]] auto OldNumThreads = EffectiveNumThreads;
  uint64_t NumBlocks;
  if (LoopTripCount >= DefaultNumBlocks * EffectiveNumThreads ||
      IsNumThreadsFromUser || IsNumThreadsStrict) {
    // Enough parallelism for blocks and threads.
    NumBlocks = ((LoopTripCount - 1) / EffectiveNumThreads) + 1;
    assert(IsNumThreadsFromUser ||
           NumBlocks >= DefaultNumBlocks &&
               "Expected sufficient outer parallelism.");
  } else if (LoopTripCount >= DefaultNumBlocks * MinThreads) {
    // Enough parallelism for blocks, limit threads.

    // This case is hard; for now, we force "full warps":
    // First, compute a thread count assuming DefaultNumBlocks.
    auto NumThreadsDefaultBlocks =
        (LoopTripCount + DefaultNumBlocks - 1) / DefaultNumBlocks;
    // Now get a power of two that is larger or equal.
    auto NumThreadsDefaultBlocksP2 =
        llvm::PowerOf2Ceil(NumThreadsDefaultBlocks);
    // Do not increase a thread limit given be the user.
    EffectiveNumThreads =
        std::min(EffectiveNumThreads, uint32_t(NumThreadsDefaultBlocksP2));
    assert(EffectiveNumThreads >= MinThreads &&
           "Expected sufficient inner parallelism.");
    NumBlocks = ((LoopTripCount - 1) / EffectiveNumThreads) + 1;
  } else {
    // Not enough parallelism for blocks and threads, limit both.
    EffectiveNumThreads = std::min(EffectiveNumThreads, MinThreads);
    NumBlocks = ((LoopTripCount - 1) / EffectiveNumThreads) + 1;
  }

  assert(EffectiveNumThreads * NumBlocks >= LoopTripCount &&
         "Expected sufficient parallelism");
  assert(OldNumThreads >= EffectiveNumThreads &&
         "Number of threads cannot be increased!");
  return NumBlocks;
}

/// Downstream: the low-trip-count knobs of the former AMDGPU launch-geometry
static uint64_t getLowTripCountOverrideNumBlocks(
    const DeviceGeometryPropsTy &Props, uint32_t UserNumBlocks,
    uint64_t LoopTripCount, uint32_t &EffectiveNumThreads,
    bool IsNumThreadsStrict, bool IsNumThreadsFromUser,
    const KernelLaunchInfoTy &KernelEnv) {
  static UInt32Envar OMPX_LowTripCount("LIBOMPTARGET_AMDGPU_LOW_TRIPCOUNT", 0);
  static UInt32Envar OMPX_NumBlocksForLowTripCount(
      "LIBOMPTARGET_BLOCKS_FOR_LOW_TRIP_COUNT", 0);
  static UInt32Envar OMPX_MinThreadsForLowTripCount(
      "LIBOMPTARGET_MIN_THREADS_FOR_LOW_TRIP_COUNT", 32);
  static Int32Envar OMP_TeamsThreadLimit("OMP_TEAMS_THREAD_LIMIT");

  if (!LoopTripCount || LoopTripCount > OMPX_LowTripCount.get())
    return 0;
  bool IsNoLoop = KernelEnv.isNoLoopMode();
  bool IsBigJumpLoop = KernelEnv.isBigJumpLoopMode();
  if (!KernelEnv.isSPMDMode() && !IsNoLoop && !IsBigJumpLoop)
    return 0;
  if (IsNumThreadsFromUser || IsNumThreadsStrict || OMP_TeamsThreadLimit > 0)
    return 0;

  uint32_t MinThreads = OMPX_MinThreadsForLowTripCount.get();
  uint32_t NumThreads = EffectiveNumThreads;
  while (NumThreads > MinThreads)
    NumThreads >>= 1;
  if (!NumThreads)
    return 0;
  if (KernelEnv.doesTeamsReduction())
    NumThreads = llvm::omp::getBlockSizeAsPowerOfTwo(NumThreads);
  EffectiveNumThreads = NumThreads;

  uint64_t TripCountNumBlocks = llvm::divideCeil(LoopTripCount, NumThreads);
  // A no-loop kernel has no loop to run more than one iteration per thread.
  if (IsNoLoop)
    return TripCountNumBlocks;

  uint64_t NumBlocks = TripCountNumBlocks;
  if (UserNumBlocks > 0)
    NumBlocks = IsBigJumpLoop ? std::min(uint64_t(UserNumBlocks), NumBlocks)
                              : UserNumBlocks;
  else if (uint64_t LowTripCountBlocks = OMPX_NumBlocksForLowTripCount.get())
    NumBlocks = std::min(LowTripCountBlocks,
                         llvm::divideCeil(LoopTripCount, MinThreads));
  if (IsBigJumpLoop)
    NumBlocks = std::min(NumBlocks, BigJumpLoopMaxNumBlocks);
  return std::min(NumBlocks, uint64_t(getDeviceBlockLimit(Props, NumThreads)));
}

/// Get the effective number of blocks for the kernel based on the
/// user-defined number of blocks and the loop trip count.
/// The number of threads \p EffectiveNumThreads can be adjusted by this
/// method. \p IsNumThreadsFromUser is true if \p EffectiveNumThreads is
/// defined by the user via the thread_limit clause.
static uint32_t getEffectiveNumBlocks(const DeviceGeometryPropsTy &Props,
                                      uint32_t UserNumBlocks,
                                      uint64_t LoopTripCount,
                                      uint32_t &EffectiveNumThreads,
                                      bool IsNumThreadsStrict,
                                      bool IsNumThreadsFromUser,
                                      const KernelLaunchInfoTy &KernelEnv) {
  assert(!KernelEnv.isBareMode() &&
         "bare kernel should not call this function");

  // Downstream: opt-in override for loops with a low trip count.
  if (uint64_t NumBlocks = getLowTripCountOverrideNumBlocks(
          Props, UserNumBlocks, LoopTripCount, EffectiveNumThreads,
          IsNumThreadsStrict, IsNumThreadsFromUser, KernelEnv))
    return NumBlocks;

  // NOTE: This clamps the user-requested number of blocks to the device limit
  // rather than honoring it exactly, which is non-standard behavior. Truly
  // honoring an arbitrary value would require launching multiple kernels or
  // reusing blocks until the requested count has been served.
  if (UserNumBlocks > 0)
    return std::min(UserNumBlocks,
                    getDeviceBlockLimit(Props, EffectiveNumThreads));

  // Return the number of blocks required to cover the loop iterations. As for
  // SPMD below, trade threads for blocks if the loop is short.
  // Downstream: a big-jump-loop kernel has no per-block overhead, so it also
  // gets one thread per iteration. It loops over the grid, so the number of
  // blocks may be capped. Only AMDGPU devices run big-jump-loop kernels.
  if (KernelEnv.isNoLoopMode() || KernelEnv.isBigJumpLoopMode()) {
    uint64_t NumBlocks =
        LoopTripCount > 0
            ? getOneIterationPerThreadNumBlocks(
                  LoopTripCount, Props.DefaultNumBlocks, EffectiveNumThreads,
                  IsNumThreadsStrict, IsNumThreadsFromUser)
            : 1;
    if (KernelEnv.isBigJumpLoopMode())
      NumBlocks =
          std::min({NumBlocks, BigJumpLoopMaxNumBlocks,
                    uint64_t(getDeviceBlockLimit(Props, EffectiveNumThreads))});
    return NumBlocks;
  }

  if (uint64_t NumBlocks = getXTeamReductionNumBlocks(
          Props, EffectiveNumThreads, LoopTripCount, KernelEnv))
    return std::min(NumBlocks,
                    uint64_t(getDeviceBlockLimit(Props, EffectiveNumThreads)));

  uint64_t DefaultNumBlocks = KernelEnv.isGenericSPMDMode()
                                  ? getDefaultNumBlocksGenericSPMD(Props)
                                  : Props.DefaultNumBlocks;
  uint64_t TripCountNumBlocks = std::numeric_limits<uint64_t>::max();
  if (LoopTripCount > 0) {
    if (KernelEnv.isSPMDMode()) {
      // We have a combined construct, i.e. `target teams distribute
      // parallel for [simd]`. We launch so many blocks so that each thread
      // will execute one iteration of the loop; rounded up to the nearest
      // integer. However, if that results in too few blocks, we artificially
      // reduce the thread count per block to increase the outer parallelism.
      TripCountNumBlocks = getOneIterationPerThreadNumBlocks(
          LoopTripCount, DefaultNumBlocks, EffectiveNumThreads,
          IsNumThreadsStrict, IsNumThreadsFromUser);
    } else {
      assert((KernelEnv.isGenericMode() || KernelEnv.isGenericSPMDMode()) &&
             "Unexpected execution mode!");
      // If we reach this point, then we have a non-combined construct, i.e.
      // `teams distribute` with a nested `parallel for` and each block is
      // assigned one iteration of the `distribute` loop. E.g.:
      //
      // #pragma omp target teams distribute
      // for(...loop_tripcount...) {
      //   #pragma omp parallel for
      //   for(...) {}
      // }
      //
      // Threads within a block will execute the iterations of the `parallel`
      // loop.
      TripCountNumBlocks = LoopTripCount;
    }
  }

  uint32_t PreferredNumBlocks = TripCountNumBlocks;
  // If the loops are long running we rather reuse blocks than spawn too many.
  static BoolEnvar OMPX_ReuseBlocksForHighTripCount(
      "LIBOMPTARGET_REUSE_BLOCKS_FOR_HIGH_TRIP_COUNT", true);
  if (OMPX_ReuseBlocksForHighTripCount.get())
    PreferredNumBlocks = std::min(TripCountNumBlocks, DefaultNumBlocks);

  // In Generic-SPMD mode, the threads of a block run the iterations of the
  // inner parallel loops, whose trip counts are unknown. If they are lower than
  // the number of threads, most threads idle. So trade threads for blocks, down
  // to one warp per block, as long as the distribute loop has an iteration for
  // every block. This keeps the device as busy if the inner trip counts are
  // high. The trade is not done if the user set the number of threads, and the
  // number of blocks stays within the device's limit, which OMP_NUM_TEAMS may
  // lower.
  if (KernelEnv.isGenericSPMDMode() && !IsNumThreadsFromUser &&
      !IsNumThreadsStrict) {
    while (EffectiveNumThreads > Props.WarpSize) {
      uint64_t MaxNumBlocks = std::min<uint64_t>(
          LoopTripCount, getDeviceBlockLimit(Props, EffectiveNumThreads / 2));
      if (uint64_t(PreferredNumBlocks) * 2 > MaxNumBlocks)
        break;
      EffectiveNumThreads /= 2;
      PreferredNumBlocks *= 2;
    }
  }

  return std::min(PreferredNumBlocks,
                  getDeviceBlockLimit(Props, EffectiveNumThreads));
}

/// Build the base launch configuration for a launch from the public
/// KernelArgsTy.
static LaunchConfigTy buildLaunchConfig(const KernelArgsTy &KernelArgs) {
  static_assert(sizeof(size_t) == sizeof(int64_t),
                "KernelArgsTy argument sizes are not reinterpretable as the "
                "size_t array olLaunchKernel expects");

  LaunchConfigTy Config;
  Config.Size.NumGroups = {KernelArgs.UserNumBlocks[0],
                           KernelArgs.UserNumBlocks[1],
                           KernelArgs.UserNumBlocks[2]};
  Config.Size.GroupSize = {KernelArgs.UserThreadLimit[0],
                           KernelArgs.UserThreadLimit[1],
                           KernelArgs.UserThreadLimit[2]};
  Config.ArgSizes = reinterpret_cast<const size_t *>(KernelArgs.ArgSizes);
  Config.Cooperative = KernelArgs.Flags.Cooperative;
  return Config;
}

/// Build the record/replay view of a launch, once the effective geometry and
/// dynamic block memory in \p Config have been resolved.
static KernelRecordArgsTy
buildRecordArgs(const KernelArgsTy &KernelArgs, const LaunchConfigTy &Config,
                const KernelLaunchInfoTy &KernelEnv,
                KernelReplayOutcomeTy *ReplayOutcome) {
  KernelRecordArgsTy RecordArgs;
  RecordArgs.NumArgs = Config.NumArgs;
  RecordArgs.Args = Config.Args;
  RecordArgs.Tripcount = KernelArgs.Tripcount;
  // The value requested by the program, before adjustEffectiveGeometry()
  // computed the effective number of blocks.
  RecordArgs.RequestedNumBlocks = KernelArgs.UserNumBlocks[0];
  RecordArgs.MaxNumThreads = KernelEnv.MaxNumThreads;
  RecordArgs.NumBlocks[0] = Config.Size.NumGroups.x;
  RecordArgs.NumBlocks[1] = Config.Size.NumGroups.y;
  RecordArgs.NumBlocks[2] = Config.Size.NumGroups.z;
  RecordArgs.NumThreads[0] = Config.Size.GroupSize.x;
  RecordArgs.NumThreads[1] = Config.Size.GroupSize.y;
  RecordArgs.NumThreads[2] = Config.Size.GroupSize.z;
  RecordArgs.DynCGroupMem = uint32_t(Config.Size.DynSharedMemory);
  RecordArgs.ReplayOutcome = ReplayOutcome;
  return RecordArgs;
}

/// Assert the launch geometry invariants expected by the plugin layer.
static void checkLaunchInvariants(const LaunchConfigTy &Config,
                                  const KernelArgsTy &KernelArgs,
                                  const KernelLaunchInfoTy &KernelLaunchInfo) {
  const auto &NumGroups = Config.Size.NumGroups;
  const auto &GroupSize = Config.Size.GroupSize;

  // Multidimensional is only supported with bare mode for now.
  assert(KernelLaunchInfo.isBareMode() ||
         GroupSize.y == 1 && GroupSize.z == 1 && NumGroups.y == 1 &&
             NumGroups.z == 1 &&
             "Non-bare mode should only use the first thread and block "
             "dimensions");

  assert(!KernelArgs.Flags.StrictBlocks ||
         NumGroups.x > 0 && NumGroups.y > 0 && NumGroups.z > 0 &&
             "Strict requires number of blocks greater than zero");
  assert(!KernelArgs.Flags.StrictThreads ||
         GroupSize.x > 0 && GroupSize.y > 0 && GroupSize.z > 0 &&
             "Strict requires number of threads greater than zero");
}

/// Calculate or adjust, in place, the effective number of threads and blocks
/// for the first dimension, unless the caller requested strict counts.
static void adjustEffectiveGeometry(const DeviceGeometryPropsTy &Props,
                                    LaunchConfigTy &Config,
                                    const KernelArgsTy &KernelArgs,
                                    const KernelLaunchInfoTy &KernelEnv) {
  const bool StrictBlocks = KernelArgs.Flags.StrictBlocks;
  const bool StrictThreads = KernelArgs.Flags.StrictThreads;
  if (StrictThreads && StrictBlocks)
    return;

  assert(!KernelEnv.isBareMode() &&
         "bare kernel launches must request strict thread/block counts");

  // Record whether the user actually requested a thread limit (thread_limit
  // clause) before possibly overwriting the group size below with the
  // computed effective value.
  const bool ThreadLimitFromUser = Config.Size.GroupSize.x > 0;

  uint32_t EffectiveNumThreads = Config.Size.GroupSize.x;
  if (!StrictThreads)
    EffectiveNumThreads =
        getEffectiveNumThreads(Props, EffectiveNumThreads, KernelEnv);

  if (!StrictBlocks)
    Config.Size.NumGroups.x = getEffectiveNumBlocks(
        Props, Config.Size.NumGroups.x, KernelArgs.Tripcount,
        EffectiveNumThreads, StrictThreads, ThreadLimitFromUser, KernelEnv);

  Config.Size.GroupSize.x = EffectiveNumThreads;
}

/// Flatten the kernel arguments into \p Config.Args. Returns the address
/// of the element reserved for the kernel launch environment (dyn_ptr), or
/// null if this launch has no such slot.
static void **resolveArgsAndDynPtrSlot(KernelArgsTy &KernelArgs,
                                       void **TgtVarsPtr, ptrdiff_t *TgtOffsets,
                                       llvm::SmallVector<void *> &Args,
                                       llvm::SmallVector<void *> &Ptrs,
                                       llvm::SmallVector<size_t> &ArgSizes,
                                       LaunchConfigTy &Config) {
  if (KernelArgs.Flags.IsCUDA) {
    // Kernel languages (CUDA/HIP) pass an already-flattened argument-pointer
    // array through KernelArgs.ArgPtrs instead of using the OpenMP
    // base-pointer/offset argument scheme.
    auto *LaunchParams =
        reinterpret_cast<KernelLaunchParamsTy *>(KernelArgs.ArgPtrs);
    Config.NumArgs = LaunchParams->NumArgs;
    Config.Args = LaunchParams->Args;
    return nullptr;
  }

  resolveKernelLaunchParams(TgtVarsPtr, TgtOffsets, KernelArgs.NumArgs, Args,
                            Ptrs, Config);

  if (KernelArgs.NumArgs == 0 ||
      KernelArgs.Version < OMP_KERNEL_ARG_MIN_VERSION_WITH_DYN_PTR)
    return nullptr;

  // The dyn_ptr slot is reserved by the host (version >= 4) or by
  // upgradeKernelArgs (version 3) as the last element of the argument array.
  // Version 3 device kernels expect it first instead, so rotate it to the
  // front to match that ABI.
  if (KernelArgs.Version != OMP_KERNEL_ARG_MIN_VERSION_WITH_DYN_PTR)
    return &Args[KernelArgs.NumArgs - 1];

  std::rotate(Args.begin(), Args.end() - 1, Args.end());

  // Keep ArgSizes in sync with the rotated Args, if present.
  if (Config.ArgSizes) {
    ArgSizes.assign(Config.ArgSizes, Config.ArgSizes + KernelArgs.NumArgs);
    std::rotate(ArgSizes.begin(), ArgSizes.end() - 1, ArgSizes.end());
    Config.ArgSizes = ArgSizes.data();
  }
  return &Args[0];
}

/// Compute the dynamic block-memory configuration for this launch, filling in
/// \p Config.Size.DynSharedMemory with the native size to request, and, if
/// this launch has a reserved dyn_ptr slot (\p DynPtrSlot), the device-side
/// kernel launch environment.
static llvm::Error
prepareDynamicLaunchState(DeviceTy &Device, uint32_t OmpABIVersion,
                          const KernelLaunchInfoTy &KernelEnv,
                          LaunchConfigTy &Config, uint32_t DynCGroupMem,
                          DynCGroupMemFallbackType DynCGroupMemFallback,
                          void **DynPtrSlot, AsyncInfoTy &AsyncInfo) {
  const auto &NumGroups = Config.Size.NumGroups;
  uint64_t NumGroupsTotal = uint64_t(NumGroups.x) * NumGroups.y * NumGroups.z;
  auto DynBlockMemConfOrErr = prepareBlockMemory(
      Device, KernelEnv, DynCGroupMem, DynCGroupMemFallback, NumGroupsTotal);
  if (!DynBlockMemConfOrErr)
    return DynBlockMemConfOrErr.takeError();

  DynBlockMemConfTy &DynBlockMemConf = *DynBlockMemConfOrErr;
  Config.Size.DynSharedMemory = DynBlockMemConf.NativeSize;
  if (DynBlockMemConf.FallbackPtr)
    freeAfterSynchronization(Device, AsyncInfo, DynBlockMemConf.FallbackPtr);

  auto KernelLaunchEnvOrErr = getKernelLaunchEnvironment(
      Device, OmpABIVersion, KernelEnv, DynBlockMemConf, DynCGroupMem,
      DynPtrSlot, AsyncInfo, NumGroups.x);
  if (!KernelLaunchEnvOrErr)
    return KernelLaunchEnvOrErr.takeError();

  // Fill in the kernel launch environment (dyn_ptr) if this launch has a
  // reserved slot for it. When replaying, getKernelLaunchEnvironment()
  // returns null so the recorded value already in the slot is preserved.
  if (DynPtrSlot && *KernelLaunchEnvOrErr)
    *DynPtrSlot = *KernelLaunchEnvOrErr;

  return llvm::Error::success();
}

// Run region on device
int32_t DeviceTy::launchKernel(void *TgtEntryPtr, void **TgtVarsPtr,
                               ptrdiff_t *TgtOffsets, KernelArgsTy &KernelArgs,
                               KernelReplayOutcomeTy *ReplayOutcome,
                               AsyncInfoTy &AsyncInfo) {
  setAsyncInfoSynchronous(AsyncInfo, ForceSynchronousTargetRegions);

  llvm::SmallVector<void *> Args, Ptrs;
  llvm::SmallVector<size_t> ArgSizes;

  auto KernelHandle = static_cast<ol_symbol_handle_t>(TgtEntryPtr);
  KernelLaunchInfoTy KernelLaunchInfo = getKernelLaunchInfo(TgtEntryPtr);
  LaunchConfigTy Config = buildLaunchConfig(KernelArgs);

  // Host kernels are only supported for the latest ABI.
  if (KernelArgs.Version < OMP_KERNEL_ARG_VERSION && isHostDevice()) {
    REPORT() << "Incompatible kernel argument version for kernel "
             << KernelLaunchInfo.Name << " on host device";
    return OFFLOAD_FAIL;
  }

  checkLaunchInvariants(Config, KernelArgs, KernelLaunchInfo);
  adjustEffectiveGeometry(GeometryProps, Config, KernelArgs, KernelLaunchInfo);

  void **DynPtrSlot = resolveArgsAndDynPtrSlot(
      KernelArgs, TgtVarsPtr, TgtOffsets, Args, Ptrs, ArgSizes, Config);

  auto DynCGroupMemFallback = static_cast<DynCGroupMemFallbackType>(
      KernelArgs.Flags.DynCGroupMemFallback);
  if (auto Err = prepareDynamicLaunchState(
          *this, KernelArgs.Version, KernelLaunchInfo, Config,
          KernelArgs.DynCGroupMem, DynCGroupMemFallback, DynPtrSlot,
          AsyncInfo)) {
    REPORT() << "Failure to prepare launch state for kernel " << TgtEntryPtr
             << ": " << toString(std::move(Err));
    return OFFLOAD_FAIL;
  }

  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceID,
       "Launching kernel %s with [%u,%u,%u] blocks and [%u,%u,%u] threads in "
       "%s mode with tripcount %" PRIu64 "\n",
       KernelLaunchInfo.Name, Config.Size.NumGroups.x, Config.Size.NumGroups.y,
       Config.Size.NumGroups.z, Config.Size.GroupSize.x,
       Config.Size.GroupSize.y, Config.Size.GroupSize.z,
       KernelLaunchInfo.getExecutionModeName(), KernelArgs.Tripcount);

  // Record the kernel prologue data before the launch.
  RecordReplayTy *RecordReplay =
      llvm::offload::tmp::__ol_tgt_GetRecordReplay(DeviceHandle);
  RecordReplayTy::HandleTy RRHandle;
  KernelInfoTy Kernel;
  if (RecordReplay) {
    Kernel = KernelInfoTy(KernelLaunchInfo.Name, KernelLaunchInfo.Image,
                          KernelLaunchInfo.IRImage);

    // Record replay requires synchronization of any previous operation.
    if (synchronize(AsyncInfo) != OFFLOAD_SUCCESS)
      return OFFLOAD_FAIL;

    KernelRecordArgsTy RecordArgs =
        buildRecordArgs(KernelArgs, Config, KernelLaunchInfo, ReplayOutcome);
    auto RRHandleOrErr = RecordReplay->recordPrologue(Kernel, RecordArgs);
    if (!RRHandleOrErr) {
      REPORT() << "Failure to record prologue for kernel "
               << KernelLaunchInfo.Name << ": "
               << toString(RRHandleOrErr.takeError());
      return OFFLOAD_FAIL;
    }
    RRHandle = *RRHandleOrErr;
  }

  // Downstream: the AMDGPU kernel trace prints the execution mode and the trip
  // count, so pass the launch info and the trip count along.
  uint64_t Tripcount = KernelArgs.Tripcount;
  ol_kernel_launch_prop_t LaunchProps[] = {
      {OL_KERNEL_LAUNCH_PROP_TYPE_IS_COOPERATIVE, &Config.Cooperative},
      {OL_KERNEL_LAUNCH_PROP_TYPE_OMP_KERNEL_LAUNCH_INFO, &KernelLaunchInfo},
      {OL_KERNEL_LAUNCH_PROP_TYPE_OMP_TRIPCOUNT, &Tripcount},
      OL_KERNEL_LAUNCH_PROP_END};

  llvm::SmallVector<size_t> PointerArgSizes;
  if (!Config.ArgSizes && Config.NumArgs) {
    PointerArgSizes.assign(Config.NumArgs, sizeof(void *));
    Config.ArgSizes = PointerArgSizes.data();
  }

  if (auto Res = olLaunchKernel(AsyncInfo.getQueue(), DeviceHandle,
                                KernelHandle, &Config.Size, LaunchProps,
                                Config.NumArgs, Config.Args, Config.ArgSizes)) {
    REPORT() << "Failure to run target region " << KernelLaunchInfo.Name
             << " in device " << DeviceID << ": " << Res->Details;
    return OFFLOAD_FAIL;
  }

  // Record the epilogue data after kernel synchronization.
  if (RecordReplay) {
    if (synchronize(AsyncInfo) != OFFLOAD_SUCCESS)
      return OFFLOAD_FAIL;

    if (auto Err = RecordReplay->recordEpilogue(Kernel, RRHandle)) {
      REPORT() << "Failure to record epilogue for kernel "
               << KernelLaunchInfo.Name << ": " << toString(std::move(Err));
      return OFFLOAD_FAIL;
    }
  }

  return OFFLOAD_SUCCESS;
}

int32_t DeviceTy::synchronize(AsyncInfoTy &AsyncInfo) {
  ol_queue_handle_t Queue = AsyncInfo.getQueue();
  if (!Queue)
    return OFFLOAD_SUCCESS;
  if (auto Res = olSyncQueue(Queue)) {
    REPORT() << "Failure to synchronize stream " << Queue << ": "
             << Res->Details;
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

int32_t DeviceTy::queryAsync(AsyncInfoTy &AsyncInfo) {
  ol_queue_handle_t Queue = AsyncInfo.getQueue();
  if (!Queue)
    return OFFLOAD_SUCCESS;
  if (auto Res = olQueryQueue(Queue, /*IsComplete=*/nullptr)) {
    REPORT() << "Failure to query stream " << Queue << ": " << Res->Details;
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

int32_t DeviceTy::recordEvent(void **Event, AsyncInfoTy &AsyncInfo) {
  ol_event_handle_t NewEvent;
  if (auto Res =
          olCreateEvent(AsyncInfo.getQueue(), OL_EVENT_FLAGS_NONE, &NewEvent)) {
    REPORT() << "Failure to record event: " << Res->Details;
    return OFFLOAD_FAIL;
  }

  if (*Event) {
    if (auto Res = olDestroyEvent(static_cast<ol_event_handle_t>(*Event)))
      REPORT() << "Failure to destroy previous event " << *Event << ": "
               << Res->Details;
  }

  *Event = NewEvent;
  return OFFLOAD_SUCCESS;
}

int32_t DeviceTy::waitEvent(void *Event, AsyncInfoTy &AsyncInfo) {
  ol_event_handle_t E = static_cast<ol_event_handle_t>(Event);
  if (auto Res = olWaitEvents(AsyncInfo.getQueue(), &E, 1)) {
    REPORT() << "Failure to wait for event " << Event << ": " << Res->Details;
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

int32_t DeviceTy::syncEvent(void *Event) {
  if (auto Res = olSyncEvent(static_cast<ol_event_handle_t>(Event))) {
    REPORT() << "Failure to synchronize event " << Event << ": "
             << Res->Details;
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

int32_t DeviceTy::destroyEvent(void *Event) {
  if (auto Res = olDestroyEvent(static_cast<ol_event_handle_t>(Event))) {
    REPORT() << "Failure to destroy event " << Event << ": " << Res->Details;
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

void DeviceTy::dumpOffloadEntries() {
  fprintf(stderr, "Device %i offload entries:\n", DeviceID);
  for (auto &It : *DeviceOffloadEntries.getExclusiveAccessor()) {
    const char *Kind = "kernel";
    if (It.second.isLink())
      Kind = "link";
    else if (It.second.isGlobal())
      Kind = "global var.";
    fprintf(stderr, "  %11s: %s\n", Kind, It.second.getNameAsCStr());
  }
}

bool DeviceTy::useAutoZeroCopy() {
  if (PM->getRequirements() & OMP_REQ_UNIFIED_SHARED_MEMORY)
    return false;

  return RTL->use_auto_zero_copy(RTLDeviceID);
}

bool DeviceTy::checkIfAPU() { return RTL->has_apu_device(RTLDeviceID); }

bool DeviceTy::supportsUnifiedMemory() {
  return RTL->supports_unified_memory(RTLDeviceID);
}

void DeviceTy::zeroCopySanityChecksAndDiag(bool isUnifiedSharedMemory,
                                           bool isAutoZeroCopy,
                                           bool isEagerMaps) {
  RTL->zero_copy_sanity_checks_and_diag(RTLDeviceID, isUnifiedSharedMemory,
                                        isAutoZeroCopy, isEagerMaps);
}

bool DeviceTy::isAccessiblePtr(const void *Ptr, size_t Size) {
  bool IsAccessible = false;
  if (auto Res = olMemIsAccessible(/*Context=*/nullptr, DeviceHandle, Ptr, Size,
                                   &IsAccessible)) {
    REPORT() << "Failure to check pointer accessibility: " << Res->Details;
    return false;
  }
  return IsAccessible;
}
