/*
//===--- UriLocator.h: Schema of URI Locator  -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc.

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

#ifndef URILOCATOR_H
#define URILOCATOR_H
#include "hsa/hsa_ven_amd_loader.h"
#include <string>
#include <vector>

class UriLocator {

public:
  struct UriInfo {
    std::string uriPath;
    int64_t loadAddressDiff;
  };

  struct UriRange {
    uint64_t startAddr_, endAddr_;
    int64_t elfDelta_;
    std::string Uri_;
  };

  bool init_ = false;
  std::vector<UriRange> rangeTab_;
  hsa_ven_amd_loader_1_03_pfn_t fn_table_;

  hsa_status_t createUriRangeTable();

  ~UriLocator() {}

  UriInfo lookUpUri(uint64_t device_pc);
  std::pair<uint64_t, uint64_t> decodeUriAndGetFd(UriInfo &uri_path,
                                                  int *uri_fd);
};
#endif