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

const char *MemoryReserveWrapperName =
    "__llvm_orc_bootstrap_mem_reserve_wrapper";
const char *MemoryFinalizeWrapperName =
    "__llvm_orc_bootstrap_mem_finalize_wrapper";
const char *MemoryDeallocateWrapperName =
    "__llvm_orc_bootstrap_mem_deallocate_wrapper";
const char *MemoryWriteUInt8sWrapperName =
    "__llvm_orc_bootstrap_mem_write_uint8s_wrapper";
const char *MemoryWriteUInt16sWrapperName =
    "__llvm_orc_bootstrap_mem_write_uint16s_wrapper";
const char *MemoryWriteUInt32sWrapperName =
    "__llvm_orc_bootstrap_mem_write_uint32s_wrapper";
const char *MemoryWriteUInt64sWrapperName =
    "__llvm_orc_bootstrap_mem_write_uint64s_wrapper";
const char *MemoryWriteBuffersWrapperName =
    "__llvm_orc_bootstrap_mem_write_buffers_wrapper";
const char *RunAsMainWrapperName = "__llvm_orc_bootstrap_run_as_main_wrapper";

} // end namespace rt
} // end namespace orc
} // end namespace llvm
