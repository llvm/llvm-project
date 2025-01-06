//===------- Offload API tests - Helpers for platform info query testing --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <vector>

// TODO: We could autogenerate these

inline std::vector<ol_platform_info_t> PlatformQueries = {
    OL_PLATFORM_INFO_NAME, OL_PLATFORM_INFO_VENDOR_NAME,
    OL_PLATFORM_INFO_VERSION, OL_PLATFORM_INFO_BACKEND};

inline std::unordered_map<ol_platform_info_t, size_t> PlatformInfoSizeMap = {
    {OL_PLATFORM_INFO_BACKEND, sizeof(ol_platform_backend_t)},
};
