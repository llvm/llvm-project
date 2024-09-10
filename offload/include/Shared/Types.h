//===-- Shared/Types.h - Type defs shared between host and device - C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Environments shared between host and device.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_SHARED_TYPES_H
#define OMPTARGET_SHARED_TYPES_H

#ifndef OMPTARGET_DEVICE_RUNTIME
#include <cstdint>
#else
#include "DeviceTypes.h"
#endif

#endif // OMPTARGET_SHARED_TYPES_H
