//===-- PlatformInterfaces.cpp - Platform interface implementations -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation file for PlatformInterfaces.h.
///
//===----------------------------------------------------------------------===//

#include "streamexecutor/PlatformInterfaces.h"

namespace streamexecutor {

PlatformStreamHandle::~PlatformStreamHandle() = default;

PlatformDevice::~PlatformDevice() = default;

} // namespace streamexecutor
