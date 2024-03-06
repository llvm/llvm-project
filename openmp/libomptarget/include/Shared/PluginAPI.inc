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

PLUGIN_API_HANDLE(init_plugin);
PLUGIN_API_HANDLE(is_valid_binary);
PLUGIN_API_HANDLE(is_data_exchangable);
PLUGIN_API_HANDLE(number_of_devices);
PLUGIN_API_HANDLE(init_device);
PLUGIN_API_HANDLE(load_binary);
PLUGIN_API_HANDLE(get_global);
PLUGIN_API_HANDLE(get_function);
PLUGIN_API_HANDLE(data_alloc);
PLUGIN_API_HANDLE(data_submit);
PLUGIN_API_HANDLE(data_submit_async);
PLUGIN_API_HANDLE(data_retrieve);
PLUGIN_API_HANDLE(data_retrieve_async);
PLUGIN_API_HANDLE(data_exchange);
PLUGIN_API_HANDLE(data_exchange_async);
PLUGIN_API_HANDLE(data_delete);
PLUGIN_API_HANDLE(launch_kernel);
PLUGIN_API_HANDLE(init_requires);
PLUGIN_API_HANDLE(synchronize);
PLUGIN_API_HANDLE(query_async);
PLUGIN_API_HANDLE(set_info_flag);
PLUGIN_API_HANDLE(print_device_info);
PLUGIN_API_HANDLE(create_event);
PLUGIN_API_HANDLE(record_event);
PLUGIN_API_HANDLE(wait_event);
PLUGIN_API_HANDLE(sync_event);
PLUGIN_API_HANDLE(destroy_event);
PLUGIN_API_HANDLE(init_async_info);
PLUGIN_API_HANDLE(init_device_info);
PLUGIN_API_HANDLE(data_lock);
PLUGIN_API_HANDLE(data_unlock);
PLUGIN_API_HANDLE(data_notify_mapped);
PLUGIN_API_HANDLE(data_notify_unmapped);
PLUGIN_API_HANDLE(set_device_offset);
PLUGIN_API_HANDLE(initialize_record_replay);
PLUGIN_API_HANDLE(use_auto_zero_copy);
PLUGIN_API_HANDLE(check_invalid_image);
PLUGIN_API_HANDLE(has_apu_device);
PLUGIN_API_HANDLE(has_USM_capable_dGPU);
PLUGIN_API_HANDLE(supports_unified_memory);
PLUGIN_API_HANDLE(is_fine_grained_memory_enabled);
PLUGIN_API_HANDLE(is_system_supporting_managed_memory);
PLUGIN_API_HANDLE(number_of_team_procs);
PLUGIN_API_HANDLE(launch_kernel_sync);
PLUGIN_API_HANDLE(set_coarse_grain_mem_region);
PLUGIN_API_HANDLE(prepopulate_page_table);
PLUGIN_API_HANDLE(query_coarse_grain_mem_region);
PLUGIN_API_HANDLE(enable_access_to_all_agents);
PLUGIN_API_HANDLE(zero_copy_sanity_checks_and_diag);
