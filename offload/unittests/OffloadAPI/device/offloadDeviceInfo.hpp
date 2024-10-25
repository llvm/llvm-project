//===------- Offload API tests - Helpers for device info query testing ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <unordered_map>
#include <vector>

// TODO: We could autogenerate these
inline std::vector<offload_device_info_t> DeviceQueries = {
    OFFLOAD_DEVICE_INFO_TYPE, OFFLOAD_DEVICE_INFO_PLATFORM,
    OFFLOAD_DEVICE_INFO_NAME, OFFLOAD_DEVICE_INFO_VENDOR,
    OFFLOAD_DEVICE_INFO_DRIVER_VERSION};

inline std::unordered_map<offload_device_info_t, size_t> DeviceInfoSizeMap = {
    {OFFLOAD_DEVICE_INFO_TYPE, sizeof(offload_device_type_t)},
    {OFFLOAD_DEVICE_INFO_PLATFORM, sizeof(offload_platform_handle_t)},
};
