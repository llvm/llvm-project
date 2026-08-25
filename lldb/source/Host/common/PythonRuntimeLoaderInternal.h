//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_HOST_COMMON_PYTHONRUNTIMELOADERINTERNAL_H
#define LLDB_SOURCE_HOST_COMMON_PYTHONRUNTIMELOADERINTERNAL_H

#include "llvm/ADT/STLFunctionalExtras.h"

namespace lldb_private {

/// Visits candidate Python runtime paths in priority order, stopping at
/// the first call that returns true. A callback (rather than a vector)
/// lets platforms defer expensive synthesis until cheaper candidates miss.
/// Paths are null-terminated for direct use with the dynamic loader API.
void ForEachPythonRuntimeCandidate(
    llvm::function_ref<bool(const char *)> callback);

} // namespace lldb_private

#endif // LLDB_SOURCE_HOST_COMMON_PYTHONRUNTIMELOADERINTERNAL_H
