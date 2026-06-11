//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpenMP 6.0 device-scope env-var registry. See kmp_device_env.h for the
// public-internal contract;
//
//===----------------------------------------------------------------------===//

#include "kmp.h"
#include "kmp_device_env.h"
#include "kmp_i18n.h"
#include "kmp_str.h"

#include <stdlib.h>
#include <string.h>

// Eligible-var table. Only env vars associated with non-global-scope ICVs
// (and not `OMP_DEFAULT_DEVICE`) are eligible for `_ALL`/`_DEV[_d]` forms,
// per OpenMP 6.0.
static char const *const __kmp_device_env_eligible_names[] = {
    "OMP_NUM_THREADS",
    NULL,
};

// Denylist of well-known OpenMP env vars that initialize *global-scope* ICVs
// or `OMP_DEFAULT_DEVICE`.
static char const *const __kmp_device_env_denied_bases[] = {
    "OMP_DEFAULT_DEVICE",
    "OMP_MAX_TASK_PRIORITY",
    "OMP_TARGET_OFFLOAD",
    "OMP_DISPLAY_ENV",
    "OMP_DISPLAY_AFFINITY",
    "OMP_AFFINITY_FORMAT",
    "OMP_CANCELLATION",
    "OMP_TOOL",
    "OMP_TOOL_LIBRARIES",
    "OMP_TOOL_VERBOSE_INIT",
    "OMP_DEBUG",
    NULL,
};

struct kmp_device_env_dev_node_t {
  int device_id;
  char *value;
  struct kmp_device_env_dev_node_t *next;
};

struct kmp_device_env_state_t {
  char *host_value; // <ENV>
  char *all_value; // <ENV>_ALL
  char *dev_default_value; // <ENV>_DEV
  kmp_device_env_dev_node_t *per_device; // <ENV>_DEV_<d>
};

static kmp_device_env_state_t *__kmp_device_env_states = NULL;

static int __kmp_device_env_count(void) {
  int count = 0;
  while (__kmp_device_env_eligible_names[count] != NULL)
    ++count;
  return count;
}

// Invariant: the eligible-name and denied-base tables must be disjoint --
// the classifier short-circuits on the first eligible match and never
// consults the denylist (see `__kmp_classify_device_env_name`). Verified
// once-per-process in debug builds to protect future maintainers.
static void __kmp_device_env_assert_tables_disjoint(void) {
#ifdef KMP_DEBUG
  for (int i = 0; __kmp_device_env_eligible_names[i] != NULL; ++i) {
    for (int j = 0; __kmp_device_env_denied_bases[j] != NULL; ++j) {
      KMP_DEBUG_ASSERT(strcmp(__kmp_device_env_eligible_names[i],
                              __kmp_device_env_denied_bases[j]) != 0);
    }
  }
#endif
}

static void __kmp_device_env_lazy_init(void) {
  if (__kmp_device_env_states != NULL)
    return;
  __kmp_device_env_assert_tables_disjoint();
  int count = __kmp_device_env_count();
  __kmp_device_env_states = (kmp_device_env_state_t *)KMP_INTERNAL_MALLOC(
      sizeof(kmp_device_env_state_t) * count);
  for (int i = 0; i < count; ++i) {
    __kmp_device_env_states[i].host_value = NULL;
    __kmp_device_env_states[i].all_value = NULL;
    __kmp_device_env_states[i].dev_default_value = NULL;
    __kmp_device_env_states[i].per_device = NULL;
  }
}

static int __kmp_device_env_index(char const *base_name) {
  if (base_name == NULL)
    return -1;
  for (int i = 0; __kmp_device_env_eligible_names[i] != NULL; ++i)
    if (strcmp(__kmp_device_env_eligible_names[i], base_name) == 0)
      return i;
  return -1;
}

extern "C" char const *__kmp_device_env_eligible_name(int index) {
  if (index < 0)
    return NULL;
  int count = __kmp_device_env_count();
  if (index >= count)
    return NULL;
  return __kmp_device_env_eligible_names[index];
}

// Match `full_name` against `base_name + suffix`. Returns the pointer just
// after the matched suffix on success, or NULL on failure.
static char const *__kmp_strip_prefix_and_suffix(char const *full_name,
                                                 char const *base_name,
                                                 char const *suffix) {
  size_t base_len = strlen(base_name);
  if (strncmp(full_name, base_name, base_len) != 0)
    return NULL;
  char const *rest = full_name + base_len;
  size_t sfx_len = strlen(suffix);
  if (strncmp(rest, suffix, sfx_len) != 0)
    return NULL;
  return rest + sfx_len;
}

// True if `s` is non-empty and consists entirely of ASCII decimal digits.
static int __kmp_is_nonneg_int(char const *s) {
  if (s == NULL || *s == '\0')
    return 0;
  for (char const *p = s; *p; ++p) {
    if (*p < '0' || *p > '9')
      return 0;
  }
  return 1;
}

// Returns the parsed value on success, -1 on overflow/empty/non-digit input.
// We cap valid device ids at INT_MAX-1 to leave INT_MAX as a sentinel.
static int __kmp_parse_dev_id(char const *s) {
  if (!__kmp_is_nonneg_int(s))
    return -1;
  // Reject obviously-overflowing inputs early.
  size_t len = strlen(s);
  if (len > 10)
    return -1;
  long long v = 0;
  for (char const *p = s; *p; ++p) {
    v = v * 10 + (*p - '0');
    if (v >= 2147483647LL)
      return -1;
  }
  return (int)v;
}

enum kmp_dev_env_kind_t {
  kmp_dev_env_none = 0,
  kmp_dev_env_all, // <ENV>_ALL
  kmp_dev_env_dev_default, // <ENV>_DEV
  kmp_dev_env_dev_id, // <ENV>_DEV_<d>
};

// Try to classify `full_name` as `<base>_<suffix>` where suffix is `_ALL`,
// `_DEV`, or `_DEV_<token>`. On match, returns the kind and (for `_DEV_<n>`)
// fills `*out_dev_id` (or sets it to -1 to flag a malformed token, e.g. a
// non-integer or overflowing token). Returns kmp_dev_env_none if `full_name`
// does not have any of those exact suffixes for `base`.
static kmp_dev_env_kind_t
__kmp_match_suffix(char const *full_name, char const *base, int *out_dev_id) {
  *out_dev_id = -1;
  if (char const *tail = __kmp_strip_prefix_and_suffix(full_name, base, "_ALL"))
    if (*tail == '\0')
      return kmp_dev_env_all;
  if (char const *tail =
          __kmp_strip_prefix_and_suffix(full_name, base, "_DEV_")) {
    int dev_id = __kmp_parse_dev_id(tail);
    *out_dev_id = dev_id; // -1 signals malformed/overflowing token
    return kmp_dev_env_dev_id;
  }
  if (char const *tail = __kmp_strip_prefix_and_suffix(full_name, base, "_DEV"))
    if (*tail == '\0')
      return kmp_dev_env_dev_default;
  return kmp_dev_env_none;
}

// Decompose `full_name` into base index, kind and (optionally) device id.
// Sets `*out_dev_id` only when the kind is `kmp_dev_env_dev_id`.
//
// `*out_index` is set to the eligible-table index when the base matches an
// eligible name, or -1 if `full_name` matches a *denied* (global-scope)
// base + suffix (the caller emits a warning in that case).
//
// Returns the kind. `kmp_dev_env_none` means the name is not a device-scope
// variant of either an eligible base or a known denied base. The caller
// should treat such names normally (no warning) so unrelated env vars like
// `OMP_DEV_LIST` are not misinterpreted.
static kmp_dev_env_kind_t __kmp_classify_device_env_name(char const *full_name,
                                                         int *out_index,
                                                         int *out_dev_id) {
  *out_index = -1;
  *out_dev_id = -1;

  if (full_name == NULL || *full_name == '\0')
    return kmp_dev_env_none;

  // First, try to recognize a known eligible base + suffix pairing.
  for (int i = 0; __kmp_device_env_eligible_names[i] != NULL; ++i) {
    char const *base = __kmp_device_env_eligible_names[i];
    int dev_id = -1;
    kmp_dev_env_kind_t k = __kmp_match_suffix(full_name, base, &dev_id);
    if (k != kmp_dev_env_none) {
      *out_index = i;
      *out_dev_id = dev_id;
      return k;
    }
  }

  // Second, check the denylist of well-known global-scope OMP env vars. Only
  // these well-defined bases trigger the global-scope-rejection warning;
  // unrelated user-defined env vars (e.g. `OMP_DEV_LIST`, `KMP_X`) are
  // ignored silently.
  for (int i = 0; __kmp_device_env_denied_bases[i] != NULL; ++i) {
    char const *base = __kmp_device_env_denied_bases[i];
    int dev_id = -1;
    kmp_dev_env_kind_t k = __kmp_match_suffix(full_name, base, &dev_id);
    if (k != kmp_dev_env_none)
      return k; // *out_index stays -1 -- denylist hit
  }

  return kmp_dev_env_none;
}

static void __kmp_device_env_set_string(char **slot, char const *value) {
  if (*slot != NULL) {
    __kmp_str_free(slot);
  }
  *slot = __kmp_str_format("%s", value);
}

static void __kmp_device_env_set_per_device(kmp_device_env_state_t *st,
                                            int device_id, char const *value) {
  for (kmp_device_env_dev_node_t *n = st->per_device; n != NULL; n = n->next) {
    if (n->device_id == device_id) {
      __kmp_str_free(&n->value);
      n->value = __kmp_str_format("%s", value);
      return;
    }
  }
  kmp_device_env_dev_node_t *node =
      (kmp_device_env_dev_node_t *)KMP_INTERNAL_MALLOC(
          sizeof(kmp_device_env_dev_node_t));
  node->device_id = device_id;
  node->value = __kmp_str_format("%s", value);
  node->next = st->per_device;
  st->per_device = node;
}

extern "C" int __kmp_device_env_record(char const *full_name,
                                       char const *value) {
  if (full_name == NULL || value == NULL)
    return 0;

  int idx = -1;
  int dev_id = -1;
  kmp_dev_env_kind_t kind =
      __kmp_classify_device_env_name(full_name, &idx, &dev_id);
  if (kind == kmp_dev_env_none)
    return 0; // not a device-scope variant -- caller handles normally

  // Suffix recognized but base is on the denylist (global-scope ICV). Per the
  // OpenMP 6.0 restriction, reject with a warning.
  if (idx < 0) {
    KMP_WARNING(DeviceEnvVarOnGlobalScope, full_name);
    return 1;
  }

  // For the eligible-base path, malformed `<ENV>_DEV_<token>` (non-integer or
  // overflowing) is rejected before we touch the registry.
  if (kind == kmp_dev_env_dev_id && dev_id < 0) {
    KMP_WARNING(MalformedDeviceEnvVar, full_name);
    return 1;
  }

  __kmp_device_env_lazy_init();
  kmp_device_env_state_t *st = &__kmp_device_env_states[idx];

  switch (kind) {
  case kmp_dev_env_all:
    __kmp_device_env_set_string(&st->all_value, value);
    return 1;
  case kmp_dev_env_dev_default:
    __kmp_device_env_set_string(&st->dev_default_value, value);
    return 1;
  case kmp_dev_env_dev_id:
    __kmp_device_env_set_per_device(st, dev_id, value);
    return 1;
  case kmp_dev_env_none:
    break;
  }
  return 0;
}

extern "C" char const *__kmp_resolve_device_env(char const *base_name,
                                                int device_id) {
  int idx = __kmp_device_env_index(base_name);
  if (idx < 0 || __kmp_device_env_states == NULL)
    return NULL;
  kmp_device_env_state_t *st = &__kmp_device_env_states[idx];

  // Host: <ENV> > <ENV>_ALL
  if (device_id == -1) {
    if (st->host_value != NULL)
      return st->host_value;
    // Defensive: after `__kmp_env_initialize` completes, `host_value` is
    // always populated whenever `<ENV>` or `<ENV>_ALL` was set (the post-pass
    // calls `observe_host` after replaying `_ALL`). This `all_value` fallback
    // covers the narrow window of an early query during init bootstrap (e.g.
    // a query issued from inside `__kmp_stg_parse` while the post-pass has
    // not yet run). Kept deliberately so the contract holds end-to-end.
    if (st->all_value != NULL)
      return st->all_value;
    return NULL;
  }
  if (device_id < 0)
    return NULL;

  // Non-host device d: <ENV>_DEV_<d> > <ENV>_DEV > <ENV>_ALL > default.
  for (kmp_device_env_dev_node_t *n = st->per_device; n != NULL; n = n->next) {
    if (n->device_id == device_id)
      return n->value;
  }
  if (st->dev_default_value != NULL)
    return st->dev_default_value;
  if (st->all_value != NULL)
    return st->all_value;
  return NULL;
}

extern "C" void __kmp_device_env_observe_host(char const *full_name,
                                              char const *value) {
  int idx = __kmp_device_env_index(full_name);
  if (idx < 0 || value == NULL)
    return;
  __kmp_device_env_lazy_init();
  __kmp_device_env_set_string(&__kmp_device_env_states[idx].host_value, value);
}

extern "C" void __kmp_device_env_reset(void) {
  if (__kmp_device_env_states == NULL)
    return;
  int count = __kmp_device_env_count();
  for (int i = 0; i < count; ++i) {
    kmp_device_env_state_t *st = &__kmp_device_env_states[i];
    __kmp_str_free(&st->host_value);
    __kmp_str_free(&st->all_value);
    __kmp_str_free(&st->dev_default_value);
    kmp_device_env_dev_node_t *n = st->per_device;
    while (n) {
      kmp_device_env_dev_node_t *next = n->next;
      __kmp_str_free(&n->value);
      KMP_INTERNAL_FREE(n);
      n = next;
    }
    st->per_device = NULL;
  }
  KMP_INTERNAL_FREE(__kmp_device_env_states);
  __kmp_device_env_states = NULL;
}

// Public test/query helper. Returns the resolved value for `name` on
// `device_id`.
extern "C" char const *__kmpc_get_resolved_device_env(char const *name,
                                                      int device_id) {
  return __kmp_resolve_device_env(name, device_id);
}
