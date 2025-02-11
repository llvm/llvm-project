//===------- Offload API tests - helper for test device code --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__AMDGPU__)
#define KERNEL [[clang::amdgpu_kernel]]
#define get_thread_id_x() __builtin_amdgcn_workitem_id_x()
#elif defined(__NVPTX__)
#define KERNEL [[clang::nvptx_kernel]]
#define get_thread_id_x() __nvvm_read_ptx_sreg_tid_x()
#else
#error "Unsupported target"
#endif
