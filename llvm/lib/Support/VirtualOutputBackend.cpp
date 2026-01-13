//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements \c vfs::OutputBackend class methods.
///
//===----------------------------------------------------------------------===//

#include "llvm/Support/VirtualOutputBackend.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/VirtualOutputError.h"

using namespace llvm;
using namespace llvm::vfs;

void OutputBackend::anchor() {}

Expected<OutputFile>
OutputBackend::createFile(const Twine &Path,
                          std::optional<OutputConfig> Config) {
  SmallString<128> PathStorage;
  Path.toVector(PathStorage);

  if (Config) {
    // Check for invalid configs.
    if (!Config->getText() && Config->getCRLF())
      return make_error<OutputConfigError>(*Config, PathStorage);
  }

  std::unique_ptr<OutputFileImpl> Impl;
  if (Error E = createFileImpl(PathStorage, Config).moveInto(Impl))
    return std::move(E);
  assert(Impl && "Expected valid Impl or Error");
  return OutputFile(PathStorage, std::move(Impl));
}
