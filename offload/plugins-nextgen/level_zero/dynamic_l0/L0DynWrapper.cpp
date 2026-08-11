//===--- level_zero/dynamic_level_zero/level_zero.cpp ------------- C++ -*-===//
//
// Implement wrapper for level_zero API calls through dlopen
//
//===----------------------------------------------------------------------===//

#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>
#include <memory>
#include <mutex>

#include "APIHelpers.h"
#include "DLWrap.h"
#include "Shared/Debug.h"
#include "llvm/Support/DynamicLibrary.h"

using namespace llvm::offload::debug;

DLWRAP_INITIALIZE()

DLWRAP_INTERNAL(zeInit, 1)
DLWRAP(zeDriverGet, 2)
DLWRAP(zeDeviceGet, 3)
DLWRAP(zeDeviceGetSubDevices, 3)
DLWRAP(zeModuleCreate, 5)
DLWRAP(zeModuleGetProperties, 2)
DLWRAP(zeModuleBuildLogDestroy, 1)
DLWRAP(zeModuleBuildLogGetString, 3)
DLWRAP(zeModuleGetKernelNames, 3)
DLWRAP(zeModuleDestroy, 1)
DLWRAP(zeCommandListAppendBarrier, 4)
DLWRAP(zeCommandListAppendLaunchKernel, 6)
DLWRAP(zeCommandListAppendLaunchCooperativeKernel, 6)
DLWRAP(zeCommandListAppendLaunchKernelWithArguments, 9)
DLWRAP(zeCommandListAppendMemoryCopy, 7)
DLWRAP(zeCommandListAppendMemoryCopyRegion, 12)
DLWRAP(zeCommandListAppendMemoryFill, 8)
DLWRAP(zeCommandListAppendMemoryPrefetch, 3)
DLWRAP(zeCommandListAppendMemAdvise, 5)
DLWRAP(zeCommandListClose, 1)
DLWRAP(zeCommandListCreate, 4)
DLWRAP(zeCommandListCreateImmediate, 4)
DLWRAP(zeCommandListDestroy, 1)
DLWRAP(zeCommandListReset, 1)
DLWRAP(zeCommandQueueCreate, 4)
DLWRAP(zeCommandQueueDestroy, 1)
DLWRAP(zeCommandQueueExecuteCommandLists, 4)
DLWRAP(zeCommandQueueSynchronize, 2)
DLWRAP(zeContextCreate, 3)
DLWRAP(zeContextDestroy, 1)
DLWRAP(zeContextMakeMemoryResident, 4)
DLWRAP(zeDeviceCanAccessPeer, 3)
DLWRAP(zeDeviceGetProperties, 2)
DLWRAP(zeDeviceGetCommandQueueGroupProperties, 3)
DLWRAP(zeDeviceGetComputeProperties, 2)
DLWRAP(zeDeviceGetModuleProperties, 2)
DLWRAP(zeDeviceGetMemoryProperties, 3)
DLWRAP(zeDeviceGetCacheProperties, 3)
DLWRAP(zeDeviceGetGlobalTimestamps, 3)
DLWRAP(zeDriverGetApiVersion, 2)
DLWRAP(zeDriverGetExtensionFunctionAddress, 3)
DLWRAP(zeDriverGetExtensionProperties, 3)
DLWRAP(zeEventCreate, 3)
DLWRAP(zeEventDestroy, 1)
DLWRAP(zeEventHostReset, 1)
DLWRAP(zeEventHostSynchronize, 2)
DLWRAP(zeEventPoolCreate, 5)
DLWRAP(zeEventPoolDestroy, 1)
DLWRAP(zeEventQueryKernelTimestamp, 2)
DLWRAP(zeFenceCreate, 3)
DLWRAP(zeFenceDestroy, 1)
DLWRAP(zeFenceHostSynchronize, 2)
DLWRAP(zeKernelCreate, 3)
DLWRAP(zeKernelDestroy, 1)
DLWRAP(zeKernelGetName, 3)
DLWRAP(zeKernelGetProperties, 2)
DLWRAP(zeKernelSetArgumentValue, 4)
DLWRAP(zeKernelSetGroupSize, 4)
DLWRAP(zeKernelSetIndirectAccess, 2)
DLWRAP(zeKernelSuggestGroupSize, 7)
DLWRAP(zeKernelSuggestMaxCooperativeGroupCount, 2)
DLWRAP(zeMemAllocDevice, 6)
DLWRAP(zeMemAllocHost, 5)
DLWRAP(zeMemAllocShared, 7)
DLWRAP(zeMemFree, 2)
DLWRAP(zeMemGetAddressRange, 4)
DLWRAP(zeMemGetAllocProperties, 4)
DLWRAP(zeModuleDynamicLink, 3)
DLWRAP(zeModuleGetGlobalPointer, 4)
DLWRAP(zeModuleGetNativeBinary, 3)
DLWRAP(zeModuleGetFunctionPointer, 3)
DLWRAP(zesDeviceEnumMemoryModules, 3)
DLWRAP(zesMemoryGetState, 2)
DLWRAP(zeCommandListHostSynchronize, 2)
DLWRAP(zeCommandListAppendSignalEvent, 2)
DLWRAP(zeCommandListAppendWaitOnEvents, 3)
DLWRAP(zeEventQueryStatus, 1)
DLWRAP(zeDriverGetDefaultContext, 1)
DLWRAP(zeCommandListAppendHostFunction, 7)

DLWRAP_FINALIZE()

#ifdef _WIN32
#define LEVEL_ZERO_LIBRARY "ze_loader.dll"
#else
#define LEVEL_ZERO_LIBRARY "libze_loader.so"
#endif // _WIN32

#ifndef TARGET_NAME
#error "Missing TARGET_NAME macro"
#endif
#ifndef DEBUG_PREFIX
#define DEBUG_PREFIX "TARGET " GETNAME(TARGET_NAME) " RTL"
#endif

// Macro used to make APIs that are returning
// ZE_RESULT_ERROR_UNSUPPORTED_FEATURE nullptr in dlwrap.
#define INVALIDATE_LEVEL_ZERO_API(function)                                    \
  if (dlwrap::function##_loaded() &&                                           \
      api_helper::callWithDefaultArgs(function) ==                             \
          ZE_RESULT_ERROR_UNSUPPORTED_FEATURE) {                               \
    for (size_t I = 0; I < dlwrap::size(); I++) {                              \
      const char *Sym = dlwrap::symbol(I);                                     \
      if (std::strcmp(Sym, #function) != 0)                                    \
        continue;                                                              \
      ODBG(OLDT_Init) << #function                                             \
                      << " returns ZE_RESULT_ERROR_UNSUPPORTED_FEATURE, "      \
                         "fallback might be available";                        \
      *dlwrap::pointer(I) = nullptr;                                           \
    }                                                                          \
  }

// Extension function pointer for getting argument sizes.
static ze_result_t (*zexKernelGetArgumentSize_ptr)(ze_kernel_handle_t, uint32_t,
                                                   uint32_t *) = nullptr;

static bool zeCommandListAppendLaunchKernelWithArgumentsFallbackAvailable() {
  static std::once_flag zexKernelGetArgumentSize_once;

  // Load zexKernelGetArgumentSize extension if available.
  std::call_once(zexKernelGetArgumentSize_once, []() {
    uint32_t DriverCount = 0;
    if (zeDriverGet(&DriverCount, nullptr) == ZE_RESULT_SUCCESS &&
        DriverCount > 0) {
      ze_driver_handle_t Driver;
      DriverCount = 1;
      if (zeDriverGet(&DriverCount, &Driver) == ZE_RESULT_SUCCESS) {
        void *ExtFunc = nullptr;
        if (zeDriverGetExtensionFunctionAddress(
                Driver, "zexKernelGetArgumentSize", &ExtFunc) ==
                ZE_RESULT_SUCCESS &&
            ExtFunc) {
          zexKernelGetArgumentSize_ptr =
              reinterpret_cast<decltype(zexKernelGetArgumentSize_ptr)>(ExtFunc);
          ODBG(OLDT_Init) << "Loaded zexKernelGetArgumentSize extension";
        }
      }
    }
  });
  return zexKernelGetArgumentSize_ptr != nullptr;
}

static ze_result_t zeCommandListAppendLaunchKernelWithArgumentsFallback(
    ze_command_list_handle_t hCommandList, ze_kernel_handle_t hKernel,
    const ze_group_count_t groupCounts, const ze_group_size_t groupSizes,
    void **pArguments, const void *pNext, ze_event_handle_t hSignalEvent,
    uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents) {

  ze_result_t Res;
  Res = zeKernelSetGroupSize(hKernel, groupSizes.groupSizeX,
                             groupSizes.groupSizeY, groupSizes.groupSizeZ);
  if (Res != ZE_RESULT_SUCCESS)
    return Res;

  ze_kernel_properties_t KernelProps = {};
  KernelProps.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
  Res = zeKernelGetProperties(hKernel, &KernelProps);
  if (Res != ZE_RESULT_SUCCESS)
    return Res;

  uint32_t NumKernelArgs = KernelProps.numKernelArgs;

  for (uint32_t KernelArg = 0; KernelArg < NumKernelArgs; KernelArg++) {
    uint32_t ArgSize = 0;

    Res = zexKernelGetArgumentSize_ptr(hKernel, KernelArg, &ArgSize);
    if (Res != ZE_RESULT_SUCCESS)
      return Res;

    Res = zeKernelSetArgumentValue(hKernel, KernelArg, ArgSize,
                                   pArguments[KernelArg]);
    if (Res != ZE_RESULT_SUCCESS)
      return Res;
  }

  bool IsCooperative = false;
  if (pNext) {
    const ze_command_list_append_launch_kernel_param_cooperative_desc_t
        *CoopDesc = static_cast<
            const ze_command_list_append_launch_kernel_param_cooperative_desc_t
                *>(pNext);
    if (CoopDesc->stype ==
        ZE_STRUCTURE_TYPE_COMMAND_LIST_APPEND_PARAM_COOPERATIVE_DESC)
      IsCooperative = CoopDesc->isCooperative;
  }

  if (IsCooperative)
    return zeCommandListAppendLaunchCooperativeKernel(
        hCommandList, hKernel, &groupCounts, hSignalEvent, numWaitEvents,
        phWaitEvents);
  return zeCommandListAppendLaunchKernel(hCommandList, hKernel, &groupCounts,
                                         hSignalEvent, numWaitEvents,
                                         phWaitEvents);
}

static struct {
  const char *Name;
  void *FallbackFunc;
  bool (*FallbackAvailable)();
} ZeFallbacksTbl[] = {
    {"zeCommandListAppendLaunchKernelWithArguments",
     reinterpret_cast<void *>(
         &zeCommandListAppendLaunchKernelWithArgumentsFallback),
     zeCommandListAppendLaunchKernelWithArgumentsFallbackAvailable}};
constexpr size_t ZeFallbacksTblSz =
    sizeof(ZeFallbacksTbl) / sizeof(ZeFallbacksTbl[0]);

static void *findZeFallback(std::string_view Name) {
  for (size_t i = 0; i < ZeFallbacksTblSz; i++) {
    if (Name == ZeFallbacksTbl[i].Name) {
      if (!ZeFallbacksTbl[i].FallbackAvailable()) {
        ODBG(OLDT_Init)
            << "Symbol '" << Name
            << "' has fallback but it's not compatible with the platform!";
        // In theory we could have multiple fallback entries for one
        // symbol, continue the search
        continue;
      }

      return ZeFallbacksTbl[i].FallbackFunc;
    }
  }
  return nullptr;
}

static bool loadLevelZero() {
  std::string L0Library{LEVEL_ZERO_LIBRARY};
  std::string ErrMsg;

  ODBG(OLDT_Init) << "Trying to load " << L0Library;
  auto DynlibHandle = std::make_unique<llvm::sys::DynamicLibrary>(
      llvm::sys::DynamicLibrary::getPermanentLibrary(L0Library.c_str(),
                                                     &ErrMsg));

  // Update the following comment and the MinVersion when the plugin starts to
  // use a new Level Zero API routine.
  // zeCommandListHostSynchronize was introduced in loader 1.10.0 (API 1.6.0).
  constexpr uint32_t MinVersion{ZE_MAKE_VERSION(1, 10)};
  auto EmitCheckVersion = [&]() {
    ODBG(OLDT_Init) << "Level Zero Loader compatible with version "
                    << ZE_MAJOR_VERSION(MinVersion) << "."
                    << ZE_MINOR_VERSION(MinVersion) << " is required";
  };

#ifndef _WIN32
  if (!DynlibHandle->isValid()) {
    // Try to open loader with major version number on Linux.
    L0Library +=
        std::string{"."} + std::to_string(ZE_MAJOR_VERSION(MinVersion));
    ErrMsg.clear();
    ODBG(OLDT_Init) << "Trying to load " << L0Library;
    DynlibHandle = std::make_unique<llvm::sys::DynamicLibrary>(
        llvm::sys::DynamicLibrary::getPermanentLibrary(L0Library.c_str(),
                                                       &ErrMsg));
  }
#endif
  if (!DynlibHandle->isValid()) {
    if (ErrMsg.empty())
      ErrMsg = "unknown error";
    ODBG(OLDT_Init) << "Unable to load library '" << L0Library
                    << "': " << ErrMsg << "!";
    EmitCheckVersion();
    return false;
  }

  for (size_t I = 0; I < dlwrap::size(); I++) {
    const char *Sym = dlwrap::symbol(I);

    void *P = DynlibHandle->getAddressOfSymbol(Sym);
    if (P)
      ODBG(OLDT_Init) << "Implementing " << Sym << " with dlsym(" << Sym
                      << ") -> " << P;
    *dlwrap::pointer(I) = P;
  }

  return true;
}

// Some APIs might be invalid in some Level Zero and
// compute runtime environments. They will return
// `ZE_RESULT_ERROR_UNSUPPORTED_FEATURE`. This
// function sets their dlwrap pointers to null
// in this situation.
static void invalidateLevelZeroSymbols() {
  INVALIDATE_LEVEL_ZERO_API(zeCommandListAppendLaunchKernelWithArguments);
}

static void addLevelZeroFallbacks() {
  std::string L0Library{LEVEL_ZERO_LIBRARY};

  for (size_t I = 0; I < dlwrap::size(); I++) {
    if (*dlwrap::pointer(I) != nullptr)
      continue;

    // Sym is missing, try to find fallback
    const char *Sym = dlwrap::symbol(I);
    void *Fallback = nullptr;

    Fallback = findZeFallback(Sym);
    if (Fallback == nullptr) {
      ODBG(OLDT_Init) << "Symbol '" << Sym << "' not found in '" << L0Library
                      << "' and no fallback is available!";
      continue;
    }

    ODBG(OLDT_Init) << "Symbol '" << Sym << "' not found in '" << L0Library
                    << "'. Using fallback implementation -> " << Fallback;
    *dlwrap::pointer(I) = Fallback;
  }
}

ze_result_t ZE_APICALL zeInit(ze_init_flags_t flags) {
  if (!loadLevelZero())
    return ZE_RESULT_ERROR_UNKNOWN;

  auto InitResult = dlwrap_zeInit(flags);

  if (InitResult != ZE_RESULT_SUCCESS)
    return InitResult;

  // Some functions might be present in loader but return
  // ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
  invalidateLevelZeroSymbols();

  // Add fallbacks after calling zeInit, so we can
  // use level_zero APIs to check if they work
  addLevelZeroFallbacks();

  return InitResult;
}
