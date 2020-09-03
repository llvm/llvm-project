//===-- DarwinLogTypes.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_DEBUGSERVER_SOURCE_MACOSX_DARWINLOG_DARWINLOGTYPES_H
#define LLDB_TOOLS_DEBUGSERVER_SOURCE_MACOSX_DARWINLOG_DARWINLOGTYPES_H

enum FilterTarget {
  eFilterTargetInvalid,
  eFilterTargetActivity,
  eFilterTargetActivityChain,
  eFilterTargetCategory,
  eFilterTargetMessage,
  eFilterTargetSubsystem
};

#endif // LLDB_TOOLS_DEBUGSERVER_SOURCE_MACOSX_DARWINLOG_DARWINLOGTYPES_H
