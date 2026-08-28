//===- Environment.h - ORC-RT executor environment access -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers for reading the process environment.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_LIB_EXECUTOR_ENVIRONMENT_H
#define ORC_RT_LIB_EXECUTOR_ENVIRONMENT_H

namespace orc_rt {

/// Like getenv, but returns null when the process is running with elevated
/// privileges (e.g. a set-user-ID or set-group-ID program), so that a variable
/// in an attacker-controlled environment cannot influence a privileged host.
///
/// Use this for any environment variable whose value has security-relevant
/// effects, e.g. choosing a file to open.
const char *secureGetenv(const char *Name);

} // namespace orc_rt

#endif // ORC_RT_LIB_EXECUTOR_ENVIRONMENT_H
