//===- PluginAPI.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CAS_PLUGINAPI_H
#define LLVM_LIB_CAS_PLUGINAPI_H

#include "llvm-c/CAS/PluginAPI_types.h"

/// See documentation in \c "llvm-c/CAS/PluginAPI_functions.h" for how these
/// functions are used.
struct llcas_functions_t {
  void (*get_plugin_version)(unsigned *major, unsigned *minor);

  void (*string_dispose)(char *);

  void (*cancellable_cancel)(llcas_cancellable_t);

  void (*cancellable_dispose)(llcas_cancellable_t);

  llcas_cas_options_t (*cas_options_create)(void);

  void (*cas_options_dispose)(llcas_cas_options_t);

  void (*cas_options_set_client_version)(llcas_cas_options_t, unsigned major,
                                         unsigned minor);

  void (*cas_options_set_ondisk_path)(llcas_cas_options_t, const char *path);

  bool (*cas_options_set_option)(llcas_cas_options_t, const char *name,
                                 const char *value, char **error);

  llcas_cas_t (*cas_create)(llcas_cas_options_t, char **error);

  void (*cas_dispose)(llcas_cas_t);

  int64_t (*cas_get_ondisk_size)(llcas_cas_t, char **error);

  bool (*cas_set_ondisk_size_limit)(llcas_cas_t, uint64_t size_limit,
                                    char **error);

  bool (*cas_prune_ondisk_data)(llcas_cas_t, char **error);

  bool (*cas_validate)(llcas_cas_t, bool check_hash, char **error);

  unsigned (*digest_parse)(llcas_cas_t, const char *printed_digest,
                           uint8_t *bytes, size_t bytes_size, char **error);

  bool (*digest_print)(llcas_cas_t, llcas_digest_t, char **printed_id,
                       char **error);

  char *(*cas_get_hash_schema_name)(llcas_cas_t);

  bool (*cas_get_objectid)(llcas_cas_t, llcas_digest_t, llcas_objectid_t *,
                           char **error);

  llcas_digest_t (*objectid_get_digest)(llcas_cas_t, llcas_objectid_t);

  llcas_lookup_result_t (*cas_contains_object)(llcas_cas_t, llcas_objectid_t,
                                               bool globally, char **error);

  llcas_lookup_result_t (*cas_load_object)(llcas_cas_t, llcas_objectid_t,
                                           llcas_loaded_object_t *,
                                           char **error);
  void (*cas_load_object_async)(llcas_cas_t, llcas_objectid_t, void *ctx_cb,
                                llcas_cas_load_object_cb,
                                llcas_cancellable_t *);

  bool (*cas_store_object)(llcas_cas_t, llcas_data_t,
                           const llcas_objectid_t *refs, size_t refs_count,
                           llcas_objectid_t *, char **error);

  llcas_data_t (*loaded_object_get_data)(llcas_cas_t, llcas_loaded_object_t);

  llcas_object_refs_t (*loaded_object_get_refs)(llcas_cas_t,
                                                llcas_loaded_object_t);

  size_t (*object_refs_get_count)(llcas_cas_t, llcas_object_refs_t);

  llcas_objectid_t (*object_refs_get_id)(llcas_cas_t, llcas_object_refs_t,
                                         size_t index);

  /*===--------------------------------------------------------------------===*\
  |* Action cache API
  \*===--------------------------------------------------------------------===*/

  llcas_lookup_result_t (*actioncache_get_for_digest)(llcas_cas_t,
                                                      llcas_digest_t key,
                                                      llcas_objectid_t *p_value,
                                                      bool globally,
                                                      char **error);

  void (*actioncache_get_for_digest_async)(llcas_cas_t, llcas_digest_t key,
                                           bool globally, void *ctx_cb,
                                           llcas_actioncache_get_cb,
                                           llcas_cancellable_t *);

  bool (*actioncache_put_for_digest)(llcas_cas_t, llcas_digest_t key,
                                     llcas_objectid_t value, bool globally,
                                     char **error);

  void (*actioncache_put_for_digest_async)(llcas_cas_t, llcas_digest_t key,
                                           llcas_objectid_t value,
                                           bool globally, void *ctx_cb,
                                           llcas_actioncache_put_cb,
                                           llcas_cancellable_t *);
};

#endif // LLVM_LIB_CAS_PLUGINAPI_H
