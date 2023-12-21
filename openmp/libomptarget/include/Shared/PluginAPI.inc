//===-- Shared/PluginAPI.inc - Target independent plugin API ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the names of the interface functions between target
// independent offload runtime library and target dependent plugins.
//
//===----------------------------------------------------------------------===//

// No include guards!

PLUGIN_API_HANDLE(init_plugin, true);
PLUGIN_API_HANDLE(is_valid_binary, true);
PLUGIN_API_HANDLE(is_valid_binary_info, false);
PLUGIN_API_HANDLE(is_data_exchangable, false);
PLUGIN_API_HANDLE(number_of_devices, true);
PLUGIN_API_HANDLE(init_device, true);
PLUGIN_API_HANDLE(load_binary, true);
PLUGIN_API_HANDLE(data_alloc, true);
PLUGIN_API_HANDLE(data_submit, true);
PLUGIN_API_HANDLE(data_submit_async, false);
PLUGIN_API_HANDLE(data_retrieve, true);
PLUGIN_API_HANDLE(data_retrieve_async, false);
PLUGIN_API_HANDLE(data_exchange, false);
PLUGIN_API_HANDLE(data_exchange_async, false);
PLUGIN_API_HANDLE(data_delete, true);
PLUGIN_API_HANDLE(launch_kernel, true);
PLUGIN_API_HANDLE(init_requires, false);
PLUGIN_API_HANDLE(synchronize, false);
PLUGIN_API_HANDLE(query_async, false);
PLUGIN_API_HANDLE(supports_empty_images, false);
PLUGIN_API_HANDLE(set_info_flag, false);
PLUGIN_API_HANDLE(print_device_info, false);
PLUGIN_API_HANDLE(create_event, false);
PLUGIN_API_HANDLE(record_event, false);
PLUGIN_API_HANDLE(wait_event, false);
PLUGIN_API_HANDLE(sync_event, false);
PLUGIN_API_HANDLE(destroy_event, false);
PLUGIN_API_HANDLE(init_async_info, false);
PLUGIN_API_HANDLE(init_device_info, false);
PLUGIN_API_HANDLE(data_lock, false);
PLUGIN_API_HANDLE(data_unlock, false);
PLUGIN_API_HANDLE(data_notify_mapped, false);
PLUGIN_API_HANDLE(data_notify_unmapped, false);
PLUGIN_API_HANDLE(set_device_offset, false);
PLUGIN_API_HANDLE(initialize_record_replay, false);
