//===- llvm/DebugInfod/DIFetcher.h - Debug info fetcher----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares a DIFetcher implementation for obtaining debug info from
/// debuginfod.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFOD_DIFETCHER_H
#define LLVM_DEBUGINFOD_DIFETCHER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/Symbolize/DIFetcher.h"

namespace llvm {

class DebuginfodDIFetcher : public symbolize::DIFetcher {
public:
  virtual ~DebuginfodDIFetcher() = default;

  /// Fetches the given Build ID using debuginfod and returns a local path to
  /// the resulting debug binary.
  Optional<std::string> fetchBuildID(ArrayRef<uint8_t> BuildID) const override;
};

} // namespace llvm

#endif // LLVM_DEBUGINFOD_DIFETCHER_H
