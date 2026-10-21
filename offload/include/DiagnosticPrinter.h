//===-------- DiagnosticPrinter.h --------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_DIAGNOSTIC_PRINTER_H
#define OMPTARGET_DIAGNOSTIC_PRINTER_H

#include "OffloadAPI.h"

#include <cstdint>
#include <string>

namespace llvm::omp::target {

/// Return the handler that prints the diagnostic error messages of the device
/// \p DeviceId. The records of the allocations are kept per device id, and the
/// plugins' own device ids mean nothing to libomptarget, so the handler carries
/// the id of the device it is installed on.
ol_diagnostic_handler_t getDiagnosticPrinter(int32_t DeviceId);

/// Record that \p Type memory was allocated at \p DevicePtr on the device
/// \p DeviceId for the host data at \p HstPtr, which is null when the
/// allocation does not back any host data. The plugins do not know what an
/// allocation is for, nor what kind of memory the OpenMP runtime asked for.
/// Nothing is recorded unless allocation tracking was requested.
void recordAllocation(int32_t DeviceId, void *DevicePtr, void *HstPtr,
                      ol_alloc_type_t Type);

/// Record that the allocation at \p DevicePtr on the device \p DeviceId
/// was deallocated, without checking anything about the deallocation. Nothing
/// is recorded unless allocation tracking was requested.
void markDeallocated(int32_t DeviceId, void *DevicePtr);

/// Check that \p DevicePtr on the device \p DeviceId can be deallocated
/// as \p Type memory and record its deallocation. If it was allocated as
/// another kind of memory, return false and describe the problem in
/// \p Summary. The deallocation of a pointer that was never allocated, and the
/// double-free of one, are detected by the plugins. Nothing is checked unless
/// allocation tracking was requested.
bool checkDeallocation(int32_t DeviceId, void *DevicePtr, ol_alloc_type_t Type,
                       std::string &Summary);

} // namespace llvm::omp::target

#endif // OMPTARGET_DIAGNOSTIC_PRINTER_H
