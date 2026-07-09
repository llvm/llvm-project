//===- llvm/DebugInfod/BuildIDFetcher.cpp - Build ID fetcher --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines a DIFetcher implementation for obtaining debug info
/// from debuginfod.
///
//===----------------------------------------------------------------------===//

#include "llvm/Debuginfod/BuildIDFetcher.h"

#include "llvm/Debuginfod/Debuginfod.h"
#include "llvm/Support/Error.h"

using namespace llvm;

Expected<std::string>
DebuginfodFetcher::fetch(ArrayRef<uint8_t> BuildID) const {
  Expected<std::string> Path = BuildIDFetcher::fetch(BuildID);
  if (Path)
    return Path;
  // Most users will not care why this failed.
  assert(errorToErrorCode(Path.takeError()) ==
             std::errc::no_such_file_or_directory &&
         "BuildIDFetcher::fetch() failed in an unexpected way");
  consumeError(Path.takeError());
  return getCachedOrDownloadDebuginfo(BuildID);
}
