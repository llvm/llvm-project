/*
 * @@name:   ompd-types.h
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __OMPD_TYPES_H
#define __OMPD_TYPES_H

extern "C" {
#include "omp-tools.h"
}

#define OMPD_TYPES_VERSION 20180906 /* YYYYMMDD Format */

/* Kinds of device threads  */
#define OMPD_THREAD_ID_PTHREAD ((ompd_thread_id_t)0)
#define OMPD_THREAD_ID_LWP ((ompd_thread_id_t)1)
#define OMPD_THREAD_ID_WINTHREAD ((ompd_thread_id_t)2)
/* The range of non-standard implementation defined values */
#define OMPD_THREAD_ID_LO ((ompd_thread_id_t)1000000)
#define OMPD_THREAD_ID_HI ((ompd_thread_id_t)1100000)

/* Memory Access Segment definitions for Host and Target Devices */
#define OMPD_SEGMENT_UNSPECIFIED ((ompd_seg_t)0)

/* Kinds of device device address spaces */
#define OMPD_DEVICE_KIND_HOST ((ompd_device_t)1)
/* The range of non-standard implementation defined values */
#define OMPD_DEVICE_IMPL_LO ((ompd_device_t)1000000)
#define OMPD_DEVICE_IMPL_HI ((ompd_device_t)1100000)
#endif
