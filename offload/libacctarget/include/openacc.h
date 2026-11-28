//===- openacc.h - OpenACC runtime API ----------------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPENACC_H
#define OPENACC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define ACC_API_EXPORT extern "C"
#else
#define ACC_API_EXPORT
#endif

#include "openacc/AccCEnums.inc"

#endif // OPENACC_H
