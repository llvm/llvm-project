//===---- amdgcn_urilocator.cpp - services support for urilocator  --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to support hsa UriLocator in hostrpc.
//
//===----------------------------------------------------------------------===//

/* Copyright (c) 2023 Advanced Micro Devices, Inc.

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

#include "urilocator.h"
#include <cstdlib>
#include <fcntl.h>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

static bool GetFileHandle(const char *fname, int *fd_ptr, size_t *sz_ptr) {
  if ((fd_ptr == nullptr) || (sz_ptr == nullptr)) {
    return false;
  }

  // open system function call, return false on fail
  struct stat stat_buf;
  *fd_ptr = open(fname, O_RDONLY);
  if (*fd_ptr < 0) {
    return false;
  }

  // Retrieve stat info and size
  if (fstat(*fd_ptr, &stat_buf) != 0) {
    close(*fd_ptr);
    return false;
  }

  *sz_ptr = stat_buf.st_size;

  return true;
}

hsa_status_t UriLocator::createUriRangeTable() {

  auto execCb = [](hsa_executable_t exec, void *data) -> hsa_status_t {
    int execState = 0;
    hsa_status_t status;
    status =
        hsa_executable_get_info(exec, HSA_EXECUTABLE_INFO_STATE, &execState);
    if (status != HSA_STATUS_SUCCESS)
      return status;
    if (execState != HSA_EXECUTABLE_STATE_FROZEN)
      return status;

    auto loadedCodeObjectCb = [](hsa_executable_t exec,
                                 hsa_loaded_code_object_t lcobj,
                                 void *data) -> hsa_status_t {
      hsa_status_t result;
      uint64_t loadBAddr = 0, loadSize = 0;
      uint32_t uriLen = 0;
      int64_t delta = 0;
      uint64_t *argsCb = static_cast<uint64_t *>(data);
      hsa_ven_amd_loader_1_03_pfn_t *fnTab =
          reinterpret_cast<hsa_ven_amd_loader_1_03_pfn_t *>(argsCb[0]);
      std::vector<UriRange> *rangeTab =
          reinterpret_cast<std::vector<UriRange> *>(argsCb[1]);

      if (!fnTab->hsa_ven_amd_loader_loaded_code_object_get_info)
        return HSA_STATUS_ERROR;

      result = fnTab->hsa_ven_amd_loader_loaded_code_object_get_info(
          lcobj, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
          (void *)&loadBAddr);
      if (result != HSA_STATUS_SUCCESS)
        return result;

      result = fnTab->hsa_ven_amd_loader_loaded_code_object_get_info(
          lcobj, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE,
          (void *)&loadSize);
      if (result != HSA_STATUS_SUCCESS)
        return result;

      result = fnTab->hsa_ven_amd_loader_loaded_code_object_get_info(
          lcobj, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH,
          (void *)&uriLen);
      if (result != HSA_STATUS_SUCCESS)
        return result;

      result = fnTab->hsa_ven_amd_loader_loaded_code_object_get_info(
          lcobj, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA,
          (void *)&delta);
      if (result != HSA_STATUS_SUCCESS)
        return result;

      char *uri = new char[uriLen + 1];
      uri[uriLen] = '\0';
      result = fnTab->hsa_ven_amd_loader_loaded_code_object_get_info(
          lcobj, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI, (void *)uri);
      if (result != HSA_STATUS_SUCCESS)
        return result;

      rangeTab->push_back(UriRange{loadBAddr, loadBAddr + loadSize - 1, delta,
                                   std::string{uri, uriLen + 1}});
      delete[] uri;
      return HSA_STATUS_SUCCESS;
    };

    uint64_t *args = static_cast<uint64_t *>(data);
    hsa_ven_amd_loader_1_03_pfn_t *fnExtTab =
        reinterpret_cast<hsa_ven_amd_loader_1_03_pfn_t *>(args[0]);
    return fnExtTab->hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
        exec, loadedCodeObjectCb, data);
  };

  if (!fn_table_.hsa_ven_amd_loader_iterate_executables)
    return HSA_STATUS_ERROR;

  uint64_t callbackArgs[2] = {(uint64_t)&fn_table_, (uint64_t)&rangeTab_};
  return fn_table_.hsa_ven_amd_loader_iterate_executables(execCb,
                                                          (void *)callbackArgs);
}

// Encoding of uniform-resource-identifier(URI) is detailed in
// https://llvm.org/docs/AMDGPUUsage.html#loaded-code-object-path-uniform-resource-identifier-uri
// The below code currently extracts the uri of loaded code object using
// file-uri.
std::pair<uint64_t, uint64_t> UriLocator::decodeUriAndGetFd(UriInfo &uri,
                                                            int *uri_fd) {

  std::ostringstream ss;
  char cur;
  uint64_t offset = 0, size = 0;
  if (uri.uriPath.size() == 0)
    return {0, 0};
  auto pos = uri.uriPath.find("//");
  if (pos == std::string::npos || uri.uriPath.substr(0, pos) != "file:") {
    uri.uriPath = "";
    return {0, 0};
  }
  auto rspos = uri.uriPath.find('#');
  if (rspos != std::string::npos) {
    // parse range specifier
    std::string offprefix = "offset=", sizeprefix = "size=";
    auto sbeg = uri.uriPath.find('&', rspos);
    auto offbeg = rspos + offprefix.size() + 1;
    std::string offstr = uri.uriPath.substr(offbeg, sbeg - offbeg);
    auto sizebeg = sbeg + sizeprefix.size() + 1;
    std::string sizestr =
        uri.uriPath.substr(sizebeg, uri.uriPath.size() - sizebeg);
    offset = std::stoull(offstr, nullptr, 0);
    size = std::stoull(sizestr, nullptr, 0);
    rspos -= 1;
  } else {
    rspos = uri.uriPath.size() - 1;
  }
  pos += 2;
  // decode filepath
  for (auto i = pos; i <= rspos;) {
    cur = uri.uriPath[i];
    if (isalnum(cur) || cur == '/' || cur == '-' || cur == '_' || cur == '.' ||
        cur == '~') {
      ss << cur;
      i++;
    } else {
      // characters prefix with '%' char
      char tbits = uri.uriPath[i + 1], lbits = uri.uriPath[i + 2];
      uint8_t t = (tbits < 58) ? (tbits - 48) : ((tbits - 65) + 10);
      uint8_t l = (lbits < 58) ? (lbits - 48) : ((lbits - 65) + 10);
      ss << (char)(((0b00000000 | t) << 4) | l);
      i += 3;
    }
  }
  uri.uriPath = ss.str();
  size_t fd_size;
  GetFileHandle(uri.uriPath.c_str(), uri_fd, &fd_size);
  // As per URI locator syntax, range_specifier is optional
  // if range_specifier is absent return total size of the file
  // and set offset to begin at 0.
  if (size == 0)
    size = fd_size;
  return {offset, size};
}

UriLocator::UriInfo UriLocator::lookUpUri(uint64_t device_pc) {
  UriInfo errorstate{"", 0};

  if (!init_) {

    hsa_status_t result;
    result = hsa_system_get_major_extension_table(
        HSA_EXTENSION_AMD_LOADER, 1, sizeof(fn_table_), &fn_table_);
    if (result != HSA_STATUS_SUCCESS)
      return errorstate;
    result = createUriRangeTable();
    if (result != HSA_STATUS_SUCCESS) {
      rangeTab_.clear();
      return errorstate;
    }
    init_ = true;
  }

  for (auto &seg : rangeTab_)
    if (seg.startAddr_ <= device_pc && device_pc <= seg.endAddr_)
      return UriInfo{seg.Uri_.c_str(), seg.elfDelta_};

  return errorstate;
}
