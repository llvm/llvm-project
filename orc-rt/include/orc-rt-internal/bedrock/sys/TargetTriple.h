//===- TargetTriple.h - Host target-triple detection ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Exactly one implementation is compiled into the runtime, chosen by the
// build: see lib/bedrock/sys/darwin/TargetTriple.cpp and its siblings.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_INTERNAL_BEDROCK_SYS_TARGETTRIPLE_H
#define ORC_RT_INTERNAL_BEDROCK_SYS_TARGETTRIPLE_H

#include <string>

namespace orc_rt::sys {

/// Returns a target-triple string for the host process. Detection may
/// involve system calls, so the result is cached.
std::string detectTargetTriple() noexcept;

} // namespace orc_rt::sys

#endif // ORC_RT_INTERNAL_BEDROCK_SYS_TARGETTRIPLE_H
