//===---- devsanitizer.cpp: Definition of handler for Sanitizer Service ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/*  Copyright (c) 2023 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */

#include "execute_service.h"
#include "urilocator.h"
#include <algorithm>
#include <assert.h>   //to exp
#include <inttypes.h> //to exp
#include <string>
#include <tuple>
#include <vector>

// Address sanitizer runtime entry-function to report the invalid device memory
// access this will be defined in llvm-project/compiler-rt/lib/asan, and will
// have effect only when compiler-rt is build for AMDGPU. Note: This API is
// runtime interface of asan library and only defined for linux os.
extern "C" void __asan_report_nonself_error(
    uint64_t *callstack, uint32_t n_callstack, uint64_t *addr, uint32_t naddr,
    uint64_t *entity_ids, uint32_t n_entities, bool is_write,
    uint32_t access_size, bool is_abort, const char *name, int64_t vma_adjust,
    int fd, uint64_t file_extent_size, uint64_t file_extent_start = 0);

namespace {
extern "C" void handler_SERVICE_SANITIZER(payload_t *packt_payload,
                                          uint64_t activemask,
                                          uint32_t gpu_device,
                                          UriLocator *uri_locator) {
  // An address results in invalid access in each active lane
  uint64_t device_failing_addresses[64];
  // An array of identifications of entities requesting a report.
  // index 0       - contains device id
  // index 1,2,3   - contains wg_idx, wg_idy, wg_idz respectively.
  // index 4 to 67 - contains reporting wave ids in a wave-front.
  uint64_t entity_id[68], callstack[1];
#if SANITIZER_AMDGPU
#if defined(__linux__)
  uint32_t n_activelanes = __builtin_popcountl(activemask);
  uint64_t access_info = 0, access_size = 0;
  bool is_abort = true;
#endif
#endif
  entity_id[0] = gpu_device;

  assert(packt_payload != nullptr && "packet payload is null?");

  int indx = 0, en_idx = 1;
  bool first_workitem = false;
  for (uint32_t wi = 0; wi != 64; ++wi) {
    uint64_t flag = activemask & ((uint64_t)1 << wi);
    if (flag == 0)
      continue;

    auto data_slot = packt_payload->slots[wi];
    // encoding of packet payload arguments is
    // defined in device-libs/asanrtl/src/report.cl
    if (!first_workitem) {
      device_failing_addresses[indx] = data_slot[0];
      callstack[0] = data_slot[1];
      entity_id[en_idx] = data_slot[2];
      entity_id[++en_idx] = data_slot[3];
      entity_id[++en_idx] = data_slot[4];
      entity_id[++en_idx] = data_slot[5];
#if SANITIZER_AMDGPU
#if defined(__linux__)
      access_info = data_slot[6];
      access_size = data_slot[7];
#endif
#endif
      first_workitem = true;
    } else {
      device_failing_addresses[indx] = data_slot[0];
      entity_id[en_idx] = data_slot[5];
    }
    indx++;
    en_idx++;
  }

#if SANITIZER_AMDGPU
#if defined(__linux__)
  bool is_write = false;
  if (access_info & 0xFFFFFFFF00000000)
    is_abort = false;
  if (access_info & 1)
    is_write = true;
#endif
#endif

  std::string fileuri;
  uint64_t size = 0, offset = 0;
#if SANITIZER_AMDGPU
#if defined(__linux__)
  int64_t loadAddrAdjust = 0;
#endif
#endif
  int uri_fd = -1;

  if (uri_locator) {
    UriLocator::UriInfo fileuri_info = uri_locator->lookUpUri(callstack[0]);
    std::tie(offset, size) =
        uri_locator->decodeUriAndGetFd(fileuri_info, &uri_fd);
#if SANITIZER_AMDGPU
#if defined(__linux__)
    loadAddrAdjust = fileuri_info.loadAddressDiff;
#endif
#endif
  }

#if SANITIZER_AMDGPU
#if defined(__linux__)
  __asan_report_nonself_error(
      callstack, 1, device_failing_addresses, n_activelanes, entity_id,
      n_activelanes + 4, is_write, access_size, is_abort,
      /*thread key*/ "amdgpu", loadAddrAdjust, uri_fd, size, offset);
#endif
#endif
}
} // end anonymous namespace
