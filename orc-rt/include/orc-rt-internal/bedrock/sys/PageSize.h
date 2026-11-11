//===- PageSize.h - Host page-size detection --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Exactly one implementation is compiled into the runtime, chosen by the
// build: see lib/bedrock/sys/posix/PageSize.cpp and its siblings.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_INTERNAL_BEDROCK_SYS_PAGESIZE_H
#define ORC_RT_INTERNAL_BEDROCK_SYS_PAGESIZE_H

#include "orc-rt/support/Error.h"

#include <cstddef>

namespace orc_rt::sys {

/// Returns the host process's page size, or an error if it could not be
/// determined.
Expected<size_t> detectPageSize() noexcept;

} // namespace orc_rt::sys

#endif // ORC_RT_INTERNAL_BEDROCK_SYS_PAGESIZE_H
