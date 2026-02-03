//===--- level_zero/dynamic_level_zero/level_zero.cpp ------------- C++ -*-===//
//
// Implement wrapper for level_zero API calls through dlopen
//
//===----------------------------------------------------------------------===//

#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>
#include <memory>

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

static bool loadLevelZero() {
  const char *L0Library = LEVEL_ZERO_LIBRARY;
  std::string ErrMsg;

  ODBG(OLDT_Init) << "Trying to load " << L0Library;
  auto DynlibHandle = std::make_unique<llvm::sys::DynamicLibrary>(
      llvm::sys::DynamicLibrary::getPermanentLibrary(L0Library, &ErrMsg));
  if (!DynlibHandle->isValid()) {
    if (ErrMsg.empty())
      ErrMsg = "unknown error";
    ODBG(OLDT_Init) << "Unable to load library '" << L0Library
                    << "': " << ErrMsg << "!";
    return false;
  }

  for (size_t I = 0; I < dlwrap::size(); I++) {
    const char *Sym = dlwrap::symbol(I);

    void *P = DynlibHandle->getAddressOfSymbol(Sym);
    if (P == nullptr) {
      ODBG(OLDT_Init) << "Unable to find '" << Sym << "' in '" << L0Library
                      << "'!";
      return false;
    }
    ODBG(OLDT_Init) << "Implementing " << Sym << " with dlsym(" << Sym
                    << ") -> " << P;

    *dlwrap::pointer(I) = P;
  }

  return true;
}

ze_result_t ZE_APICALL zeInit(ze_init_flags_t flags) {
  if (!loadLevelZero())
    return ZE_RESULT_ERROR_UNKNOWN;
  return dlwrap_zeInit(flags);
}
