//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements \c OutputConfig class methods.
///
//===----------------------------------------------------------------------===//

#include "llvm/Support/VirtualOutputConfig.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::vfs;

OutputConfig &OutputConfig::setOpenFlags(const sys::fs::OpenFlags &Flags) {
  // Ignore CRLF on its own as invalid.
  using namespace llvm::sys::fs;
  return Flags & OF_Text
             ? setText().setCRLF(Flags & OF_CRLF).setAppend(Flags & OF_Append)
             : setBinary().setAppend(Flags & OF_Append);
}

void OutputConfig::print(raw_ostream &OS) const {
  OS << "{";
  bool IsFirst = true;
  auto printFlag = [&](StringRef FlagName, bool Value) {
    if (IsFirst)
      IsFirst = false;
    else
      OS << ",";
    if (!Value)
      OS << "No";
    OS << FlagName;
  };

#define HANDLE_OUTPUT_CONFIG_FLAG(NAME, DEFAULT)                               \
  if (get##NAME() != DEFAULT)                                                  \
    printFlag(#NAME, get##NAME());
#include "llvm/Support/VirtualOutputConfig.def"
  OS << "}";
}

LLVM_DUMP_METHOD void OutputConfig::dump() const { print(dbgs()); }

raw_ostream &llvm::operator<<(raw_ostream &OS, OutputConfig Config) {
  Config.print(OS);
  return OS;
}
