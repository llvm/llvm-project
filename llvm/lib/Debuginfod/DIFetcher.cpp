//===- llvm/DebugInfod/DIFetcher.cpp - Debug info fetcher -----------------===//
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

#include "llvm/Debuginfod/DIFetcher.h"

#include "llvm/Debuginfod/Debuginfod.h"

using namespace llvm;

Optional<std::string>
DebuginfodDIFetcher::fetchBuildID(ArrayRef<uint8_t> BuildID) const {
  Expected<std::string> PathOrErr = getCachedOrDownloadDebuginfo(BuildID);
  if (PathOrErr)
    return *PathOrErr;
  consumeError(PathOrErr.takeError());
  return None;
}
