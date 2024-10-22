//===- VirtualOutputBackend.cpp - Virtualize compiler outputs -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements vfs::OutputBackend.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/VirtualOutputBackend.h"
#include "llvm/ADT/StringExtras.h"

using namespace llvm;
using namespace llvm::vfs;

void OutputBackend::anchor() {}

Expected<OutputFile>
OutputBackend::createFile(const Twine &Path_,
                          std::optional<OutputConfig> Config) {
  SmallString<128> Path;
  Path_.toVector(Path);

  if (Config) {
    // Check for invalid configs.
    if (!Config->getText() && Config->getCRLF())
      return make_error<OutputConfigError>(*Config, Path);
  }

  std::unique_ptr<OutputFileImpl> Impl;
  if (Error E = createFileImpl(Path, Config).moveInto(Impl))
    return std::move(E);
  assert(Impl && "Expected valid Impl or Error");
  return OutputFile(Path, std::move(Impl));
}
