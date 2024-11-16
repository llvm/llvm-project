//===-- HostInfoQNX.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/qnx/HostInfoQNX.h"

using namespace lldb_private;

llvm::VersionTuple HostInfoQNX::GetOSVersion() { return llvm::VersionTuple(); }

std::optional<std::string> HostInfoQNX::GetOSBuildString() {
  return std::nullopt;
}

FileSpec HostInfoQNX::GetProgramFileSpec() {
  static FileSpec g_program_filespec;
  return g_program_filespec;
}
