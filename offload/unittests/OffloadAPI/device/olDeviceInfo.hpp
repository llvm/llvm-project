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
inline std::vector<ol_device_info_t> DeviceQueries = {
    OL_DEVICE_INFO_TYPE, OL_DEVICE_INFO_PLATFORM, OL_DEVICE_INFO_NAME,
    OL_DEVICE_INFO_VENDOR, OL_DEVICE_INFO_DRIVER_VERSION};

inline std::unordered_map<ol_device_info_t, size_t> DeviceInfoSizeMap = {
    {OL_DEVICE_INFO_TYPE, sizeof(ol_device_type_t)},
    {OL_DEVICE_INFO_PLATFORM, sizeof(ol_platform_handle_t)},
};
