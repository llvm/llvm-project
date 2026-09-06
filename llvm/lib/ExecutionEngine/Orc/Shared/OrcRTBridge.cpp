//===------ OrcRTBridge.cpp - Executor functions for bootstrap -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"

namespace llvm {
namespace orc {
namespace rt {

const char *SimpleExecutorMemoryManagerInstanceName =
    "__llvm_orc_SimpleExecutorMemoryManager_Instance";
const char *SimpleExecutorMemoryManagerReserveWrapperName =
    "__llvm_orc_SimpleExecutorMemoryManager_reserve_wrapper";
const char *SimpleExecutorMemoryManagerInitializeWrapperName =
    "__llvm_orc_SimpleExecutorMemoryManager_initialize_wrapper";
const char *SimpleExecutorMemoryManagerDeinitializeWrapperName =
    "__llvm_orc_SimpleExecutorMemoryManager_deinitialize_wrapper";
const char *SimpleExecutorMemoryManagerReleaseWrapperName =
    "__llvm_orc_SimpleExecutorMemoryManager_release_wrapper";

const char *ExecutorSharedMemoryMapperServiceInstanceName =
    "__llvm_orc_ExecutorSharedMemoryMapperService_Instance";
const char *ExecutorSharedMemoryMapperServiceReserveWrapperName =
    "__llvm_orc_ExecutorSharedMemoryMapperService_Reserve";
const char *ExecutorSharedMemoryMapperServiceInitializeWrapperName =
    "__llvm_orc_ExecutorSharedMemoryMapperService_Initialize";
const char *ExecutorSharedMemoryMapperServiceDeinitializeWrapperName =
    "__llvm_orc_ExecutorSharedMemoryMapperService_Deinitialize";
const char *ExecutorSharedMemoryMapperServiceReleaseWrapperName =
    "__llvm_orc_ExecutorSharedMemoryMapperService_Release";

const char *RegisterEHFrameSectionAllocActionName =
    "llvm_orc_registerEHFrameAllocAction";
const char *DeregisterEHFrameSectionAllocActionName =
    "llvm_orc_deregisterEHFrameAllocAction";

const char *RegisterJITLoaderGDBAllocActionName =
    "orc_rt_ci_aa_sps_GDBJITRegistrar_register";
const char *DeregisterJITLoaderGDBAllocActionName =
    "orc_rt_ci_aa_sps_GDBJITRegistrar_deregister";

const char *const DispatchName = "__orc_rt_jit_dispatch";
const char *const DispatchCtxName = "__orc_rt_jit_dispatch_ctx";

const MachOUnwindInfoRegistrarSymbolNames
    orc_rt_MachOUnwindInfoRegistrarSPSSymbols = {
        "orc_rt_ci_aa_sps_MachOUnwindInfoRegistrar_registerSections",
        "orc_rt_ci_aa_sps_MachOUnwindInfoRegistrar_deregisterSections"};

} // end namespace rt
namespace rt_alt {
const char *UnwindInfoManagerRegisterActionName =
    "orc_rt_alt_UnwindInfoManager_register";
const char *UnwindInfoManagerDeregisterActionName =
    "orc_rt_alt_UnwindInfoManager_deregister";

} // end namespace rt_alt
} // end namespace orc
} // end namespace llvm
