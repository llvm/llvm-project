//===--- amdgpu/dynamic_hsa/hsa.cpp ------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement subset of hsa api by calling into hsa library via dlopen
// Does the dlopen/dlsym calls as part of the call to hsa_init
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/DynamicLibrary.h"

#include "Debug.h"
#include "dlwrap.h"
#include "hsa.h"
#include "hsa_ext_amd.h"
#include <memory>

DLWRAP_INITIALIZE()

DLWRAP_INTERNAL(hsa_init, 0)

DLWRAP(hsa_status_string, 2)
DLWRAP(hsa_shut_down, 0)
DLWRAP(hsa_system_get_info, 2)
DLWRAP(hsa_agent_get_info, 3)
DLWRAP(hsa_isa_get_info_alt, 3)
DLWRAP(hsa_iterate_agents, 2)
DLWRAP(hsa_agent_iterate_isas, 3)
DLWRAP(hsa_signal_create, 4)
DLWRAP(hsa_signal_destroy, 1)
DLWRAP(hsa_signal_store_relaxed, 2)
DLWRAP(hsa_signal_store_screlease, 2)
DLWRAP(hsa_signal_wait_scacquire, 5)
DLWRAP(hsa_signal_load_scacquire, 1)
DLWRAP(hsa_signal_subtract_screlease, 2)
DLWRAP(hsa_queue_create, 8)
DLWRAP(hsa_queue_destroy, 1)
DLWRAP(hsa_queue_load_read_index_scacquire, 1)
DLWRAP(hsa_queue_add_write_index_relaxed, 2)
DLWRAP(hsa_memory_copy, 3)
DLWRAP(hsa_executable_create, 4)
DLWRAP(hsa_executable_create_alt, 4)
DLWRAP(hsa_executable_destroy, 1)
DLWRAP(hsa_executable_freeze, 2)
DLWRAP(hsa_executable_validate, 2)
DLWRAP(hsa_executable_symbol_get_info, 3)
DLWRAP(hsa_executable_get_symbol_by_name, 4)
DLWRAP(hsa_executable_iterate_symbols, 3)
DLWRAP(hsa_code_object_deserialize, 4)
DLWRAP(hsa_executable_load_code_object, 4)
DLWRAP(hsa_code_object_destroy, 1)
DLWRAP(hsa_amd_agent_memory_pool_get_info, 4)
DLWRAP(hsa_amd_agent_iterate_memory_pools, 3)
DLWRAP(hsa_amd_memory_pool_allocate, 4)
DLWRAP(hsa_amd_memory_pool_free, 1)
DLWRAP(hsa_amd_memory_async_copy, 8)
DLWRAP(hsa_amd_memory_pool_get_info, 3)
DLWRAP(hsa_amd_agents_allow_access, 4)
DLWRAP(hsa_amd_memory_lock, 5)
DLWRAP(hsa_amd_memory_unlock, 1)
DLWRAP(hsa_amd_memory_fill, 3)
DLWRAP(hsa_amd_register_system_event_handler, 2)
DLWRAP(hsa_amd_signal_create, 5)
DLWRAP(hsa_amd_signal_async_handler, 5)
DLWRAP(hsa_amd_pointer_info, 5)

DLWRAP_FINALIZE()

#ifndef DYNAMIC_HSA_PATH
#define DYNAMIC_HSA_PATH "libhsa-runtime64.so"
#endif

#ifndef TARGET_NAME
#error "Missing TARGET_NAME macro"
#endif
#define DEBUG_PREFIX "Target " GETNAME(TARGET_NAME) " RTL"

static bool checkForHSA() {
  // return true if dlopen succeeded and all functions found

  const char *HsaLib = DYNAMIC_HSA_PATH;
  std::string ErrMsg;
  auto DynlibHandle = std::make_unique<llvm::sys::DynamicLibrary>(
      llvm::sys::DynamicLibrary::getPermanentLibrary(HsaLib, &ErrMsg));
  if (!DynlibHandle->isValid()) {
    DP("Unable to load library '%s': %s!\n", HsaLib, ErrMsg.c_str());
    return false;
  }

  for (size_t I = 0; I < dlwrap::size(); I++) {
    const char *Sym = dlwrap::symbol(I);

    void *P = DynlibHandle->getAddressOfSymbol(Sym);
    if (P == nullptr) {
      DP("Unable to find '%s' in '%s'!\n", Sym, HsaLib);
      return false;
    }
    DP("Implementing %s with dlsym(%s) -> %p\n", Sym, Sym, P);

    *dlwrap::pointer(I) = P;
  }

  return true;
}

hsa_status_t hsa_init() {
  if (!checkForHSA()) {
    return HSA_STATUS_ERROR;
  }
  return dlwrap_hsa_init();
}
