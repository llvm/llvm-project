/*
 * ompd-specific.cpp -- OpenMP debug support
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ompd-specific.h"

#if OMPD_SUPPORT

/**
 * Declaration of symbols to hold struct size and member offset information
 */

#define ompd_declare_access(t, m) uint64_t ompd_access__##t##__##m;
OMPD_FOREACH_ACCESS(ompd_declare_access)
#undef ompd_declare_access

#define ompd_declare_sizeof_member(t, m) uint64_t ompd_sizeof__##t##__##m;
OMPD_FOREACH_ACCESS(ompd_declare_sizeof_member)
#undef ompd_declare_sizeof_member

#define ompd_declare_bitfield(t, m) uint64_t ompd_bitfield__##t##__##m;
OMPD_FOREACH_BITFIELD(ompd_declare_bitfield)
#undef ompd_declare_bitfield

#define ompd_declare_sizeof(t) uint64_t ompd_sizeof__##t;
OMPD_FOREACH_SIZEOF(ompd_declare_sizeof)
#undef ompd_declare_sizeof

volatile const char **ompd_dll_locations = NULL;
uint64_t ompd_state = 0;

char *ompd_env_block = NULL;
ompd_size_t ompd_env_block_size = 0;

void ompd_init() {

  static int ompd_initialized = 0;

  if (ompd_initialized)
    return;

    /**
     * Calculate member offsets for structs and unions
     */

#define ompd_init_access(t, m)                                                 \
  ompd_access__##t##__##m = (uint64_t) & (((t *)0)->m);
  OMPD_FOREACH_ACCESS(ompd_init_access)
#undef ompd_init_access

  /**
   * Create bit mask for bitfield access
   */

#define ompd_init_bitfield(t, m)                                               \
  ompd_bitfield__##t##__##m = 0;                                               \
  ((t *)(&ompd_bitfield__##t##__##m))->m = 1;
  OMPD_FOREACH_BITFIELD(ompd_init_bitfield)
#undef ompd_init_bitfield

  /**
   * Calculate type size information
   */

#define ompd_init_sizeof_member(t, m)                                          \
  ompd_sizeof__##t##__##m = sizeof(((t *)0)->m);
  OMPD_FOREACH_ACCESS(ompd_init_sizeof_member)
#undef ompd_init_sizeof_member

#define ompd_init_sizeof(t) ompd_sizeof__##t = sizeof(t);
  OMPD_FOREACH_SIZEOF(ompd_init_sizeof)
#undef ompd_init_sizeof

  char *libname = NULL;

#if KMP_OS_UNIX
  // Find the location of libomp.so thru dladdr and replace the libomp with
  // libompd to get the full path of libompd
  Dl_info dl_info;
  int ret = dladdr((void *)ompd_init, &dl_info);
  if (!ret) {
    fprintf(stderr, "%s\n", dlerror());
  }
  int lib_path_length;
  if (strrchr(dl_info.dli_fname, '/')) {
    lib_path_length = strrchr(dl_info.dli_fname, '/') - dl_info.dli_fname;

    libname =
        (char *)malloc(lib_path_length + 12 /*for '/libompd.so' and '\0'*/);
    strcpy(libname, dl_info.dli_fname);
    memcpy(strrchr(libname, '/'), "/libompd.so\0", 12);
  }
#endif

  const char *ompd_env_var = getenv("OMP_DEBUG");
  if (ompd_env_var && !strcmp(ompd_env_var, "enabled")) {
    fprintf(stderr, "OMP_OMPD active\n");
    ompt_enabled.enabled = 1;
    ompd_state |= OMPD_ENABLE_BP;
  }

  ompd_initialized = 1;
  ompd_dll_locations = (volatile const char **)malloc(3 * sizeof(const char *));
  ompd_dll_locations[0] = "libompd.so";
  ompd_dll_locations[1] = libname;
  ompd_dll_locations[2] = NULL;
  ompd_dll_locations_valid();
}

void __attribute__((noinline)) ompd_dll_locations_valid(void) {
  /* naive way of implementing hard to opt-out empty function
     we might want to use a separate object file? */
  asm("");
}

void ompd_bp_parallel_begin(void) {
  /* naive way of implementing hard to opt-out empty function
     we might want to use a separate object file? */
  asm("");
}
void ompd_bp_parallel_end(void) {
  /* naive way of implementing hard to opt-out empty function
     we might want to use a separate object file? */
  asm("");
}
void ompd_bp_task_begin(void) {
  /* naive way of implementing hard to opt-out empty function
     we might want to use a separate object file? */
  asm("");
}
void ompd_bp_task_end(void) {
  /* naive way of implementing hard to opt-out empty function
     we might want to use a separate object file? */
  asm("");
}
void ompd_bp_thread_begin(void) {
  /* naive way of implementing hard to opt-out empty function
     we might want to use a separate object file? */
  asm("");
}
void ompd_bp_thread_end(void) {
  /* naive way of implementing hard to opt-out empty function
     we might want to use a separate object file? */
  asm("");
}

#endif /* OMPD_SUPPORT */
