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
PLUGIN_API_HANDLE(is_data_exchangable, false);
PLUGIN_API_HANDLE(number_of_devices, true);
PLUGIN_API_HANDLE(init_device, true);
PLUGIN_API_HANDLE(load_binary, true);
PLUGIN_API_HANDLE(get_global, true);
PLUGIN_API_HANDLE(get_function, true);
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
PLUGIN_API_HANDLE(check_invalid_image, true);
PLUGIN_API_HANDLE(can_use_host_globals, true);
PLUGIN_API_HANDLE(has_apu_device, true);
PLUGIN_API_HANDLE(has_USM_capable_dGPU, true);
PLUGIN_API_HANDLE(are_allocations_for_maps_on_apus_disabled, true);
PLUGIN_API_HANDLE(requested_prepopulate_gpu_page_table, true);
PLUGIN_API_HANDLE(is_fine_grained_memory_enabled, true);
PLUGIN_API_HANDLE(is_system_supporting_managed_memory, true);
PLUGIN_API_HANDLE(number_of_team_procs, true);
PLUGIN_API_HANDLE(launch_kernel_sync, true);
PLUGIN_API_HANDLE(set_coarse_grain_mem_region, true);
PLUGIN_API_HANDLE(prepopulate_page_table, true);
PLUGIN_API_HANDLE(query_coarse_grain_mem_region, true);
PLUGIN_API_HANDLE(enable_access_to_all_agents, false);
PLUGIN_API_HANDLE(release_async_info, false);
PLUGIN_API_HANDLE(activate_record_replay, false);
PLUGIN_API_HANDLE(set_up_env, true);
