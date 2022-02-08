//===-- llvm/DebugInfo/Symbolize/DIFetcher.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares a DIFetcher abstraction for obtaining debug info from an
/// arbitrary outside source.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_SYMBOLIZE_DIFETCHER_H
#define LLVM_DEBUGINFO_SYMBOLIZE_DIFETCHER_H

#include <cstdint>
#include <string>

#include "llvm/ADT/ArrayRef.h"

namespace llvm {
namespace symbolize {

/// The DIFetcher interface provides arbitrary mechanisms for obtaining debug
/// info from an outside source.
class DIFetcher {
public:
  virtual ~DIFetcher() = default;
  virtual Optional<std::string>
  fetchBuildID(ArrayRef<uint8_t> BuildID) const = 0;
};

/// LocalDIFetcher searches local cache directories for debug info.
class LocalDIFetcher : public DIFetcher {
public:
  LocalDIFetcher(ArrayRef<std::string> DebugFileDirectory)
      : DebugFileDirectory(DebugFileDirectory){};
  virtual ~LocalDIFetcher() = default;

  Optional<std::string> fetchBuildID(ArrayRef<uint8_t> BuildID) const override;

private:
  const ArrayRef<std::string> DebugFileDirectory;
};

} // end namespace symbolize
} // end namespace llvm

#endif // LLVM_DEBUGINFO_SYMBOLIZE_DIFETCHER_H
