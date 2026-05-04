//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements parsing, storage and precedence resolution for the OpenMP 6.0
// device-scope environment variable extensions described in OpenMP 6.0
// Section 3.2 (ICV initialization) and Chapter 4 (Environment Variables):
//
//   <ENV>          -- sets the host device ICV.
//   <ENV>_ALL      -- sets the host and non-host device ICVs that are not
//                     overridden by a more specific form.
//   <ENV>_DEV      -- sets all non-host device ICVs that are not overridden
//                     by an `<ENV>_DEV_<d>` form.
//   <ENV>_DEV_<d>  -- sets the ICV for non-host device with id `d` (a
//                     non-negative integer).
//
// Precedence:
//   Host:     <ENV>           > <ENV>_ALL                            > default
//   Device d: <ENV>_DEV_<d>   > <ENV>_DEV    > <ENV>_ALL             > default
//
// Restrictions enforced:
//   * Device-specific environment variables are only accepted for env vars
//     listed in `__kmp_device_env_table` (i.e. those that initialize
//     device-scope ICVs and are not global-scope or `OMP_DEFAULT_DEVICE`).
//     Suffix forms applied to other env vars are ignored with a warning.
//   * `<ENV>_DEV_<token>` where <token> is not a non-negative integer is
//     rejected with a warning. The spec requires that device-specific
//     environment variables must not specify the host device; we accept only
//     non-negative integer non-host device ids.
//   * Conflicting settings of `<ENV>_DEV_<d>` (same d) follow last-parsed-wins
//     semantics, mirroring existing libomp env handling.

#ifndef KMP_DEVICE_ENV_H
#define KMP_DEVICE_ENV_H

#ifdef __cplusplus
extern "C" {
#endif

// Consume `full_name=value` if it is a device-scope variant. Returns:
//   1 -- consumed (stored, OR rejected with a warning); caller must skip it.
//   0 -- not a device-scope variant; caller continues normal processing.
int __kmp_device_env_record(char const *full_name, char const *value);

// Resolve the effective string for `base_name` on `device_id` per the
// precedence rules above. Pass `device_id == -1` to request the host;
// any other negative value returns NULL (defensive). Returns NULL when no
// source applies (caller falls back to its own default).
char const *__kmp_resolve_device_env(char const *base_name, int device_id);

// Record an unsuffixed `<ENV>=value` pair so the host query is consistent
// with the host ICV. No-op for non-eligible names.
void __kmp_device_env_observe_host(char const *full_name, char const *value);

// Free all storage owned by the registry.
void __kmp_device_env_reset(void);

// Iterate the eligible base-name table; returns NULL once exhausted.
char const *__kmp_device_env_eligible_name(int index);

#ifdef __cplusplus
}
#endif

#endif // KMP_DEVICE_ENV_H
