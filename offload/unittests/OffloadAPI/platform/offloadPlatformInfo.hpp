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

inline std::vector<offload_platform_info_t> PlatformQueries = {
    OFFLOAD_PLATFORM_INFO_NAME, OFFLOAD_PLATFORM_INFO_VENDOR_NAME,
    OFFLOAD_PLATFORM_INFO_VERSION, OFFLOAD_PLATFORM_INFO_BACKEND};

inline std::unordered_map<offload_platform_info_t, size_t> PlatformInfoSizeMap =
    {
        {OFFLOAD_PLATFORM_INFO_BACKEND, sizeof(offload_platform_backend_t)},
};
