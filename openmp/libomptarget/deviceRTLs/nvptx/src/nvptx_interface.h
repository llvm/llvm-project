//===--- nvptx_interface.h - OpenMP interface definitions -------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _NVPTX_INTERFACE_H_
#define _NVPTX_INTERFACE_H_

#include <stdint.h>

#define EXTERN extern "C" __device__
typedef uint32_t __kmpc_impl_lanemask_t;

#endif
