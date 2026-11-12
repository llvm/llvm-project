/*
 * ompd-private.h
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SRC_OMPD_PRIVATE_H_
#define SRC_OMPD_PRIVATE_H_

/*
 * Definition of OMPD states, taken from OMPT
 */
#define FOREACH_OMPD_STATE(macro)                                              \
                                                                               \
  /* first available state */                                                  \
  macro(ompt_state_undefined, 0x102) /* undefined thread state */              \
                                                                               \
      /* work states (0..15) */                                                \
      macro(ompt_state_work_serial, 0x000)    /* working outside parallel */   \
      macro(ompt_state_work_parallel, 0x001)  /* working within parallel */    \
      macro(ompt_state_work_reduction, 0x002) /* performing a reduction */     \
                                                                               \
      /* barrier wait states (16..31) */                                       \
      macro(ompt_state_wait_barrier, 0x010) /* waiting at a barrier */         \
      macro(ompt_state_wait_barrier_implicit_parallel,                         \
            0x011) /* implicit barrier at the end of parallel region */        \
      macro(ompt_state_wait_barrier_implicit_workshare,                        \
            0x012) /* implicit barrier at the end of worksharing */            \
      macro(ompt_state_wait_barrier_implicit, 0x013) /* implicit barrier */    \
      macro(ompt_state_wait_barrier_explicit, 0x014) /* explicit barrier */    \
                                                                               \
      /* task wait states (32..63) */                                          \
      macro(ompt_state_wait_taskwait, 0x020)  /* waiting at a taskwait */      \
      macro(ompt_state_wait_taskgroup, 0x021) /* waiting at a taskgroup */     \
                                                                               \
      /* mutex wait states (64..127) */                                        \
      macro(ompt_state_wait_mutex, 0x040)                                      \
          macro(ompt_state_wait_lock, 0x041) /* waiting for lock */            \
      macro(ompt_state_wait_critical, 0x042) /* waiting for critical */        \
      macro(ompt_state_wait_atomic, 0x043)   /* waiting for atomic */          \
      macro(ompt_state_wait_ordered, 0x044)  /* waiting for ordered */         \
                                                                               \
      /* target wait states (128..255) */                                      \
      macro(ompt_state_wait_target, 0x080) /* waiting for target region */     \
      macro(ompt_state_wait_target_map,                                        \
            0x081) /* waiting for target data mapping operation */             \
      macro(ompt_state_wait_target_update,                                     \
            0x082) /* waiting for target update operation */                   \
                                                                               \
      /* misc (256..511) */                                                    \
      macro(ompt_state_idle, 0x100)     /* waiting for work */                 \
      macro(ompt_state_overhead, 0x101) /* overhead excluding wait states */   \
                                                                               \
      /* implementation-specific states (512..) */

#define OMPD_LAST_OMP_STATE ompt_state_overhead

/**
 * Primitive types.
 */
typedef enum ompd_target_prim_types_t {
  ompd_type_invalid = -1,
  ompd_type_char = 0,
  ompd_type_short = 1,
  ompd_type_int = 2,
  ompd_type_long = 3,
  ompd_type_long_long = 4,
  ompd_type_pointer = 5,
  ompd_type_max
} ompd_target_prim_types_t;

#include "ompd-types.h"
#endif /*SRC_OMPD_PRIVATE_H*/
