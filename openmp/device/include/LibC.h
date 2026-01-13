//===--------- LibC.h - Simple implementation of libc functions --- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_LIBC_H
#define OMPTARGET_LIBC_H

#include "DeviceTypes.h"

namespace ompx {

int printf(const char *Format, ...);

} // namespace ompx

#endif
