//===-- lib/DebugInfo/Symbolize/DIFetcher.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the implementation of the local debug info fetcher, which
/// searches cache directories.
///
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/Symbolize/DIFetcher.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace llvm {
namespace symbolize {

Optional<std::string>
LocalDIFetcher::fetchBuildID(ArrayRef<uint8_t> BuildID) const {
  auto GetDebugPath = [&](StringRef Directory) {
    SmallString<128> Path{Directory};
    sys::path::append(Path, ".build-id",
                      llvm::toHex(BuildID[0], /*LowerCase=*/true),
                      llvm::toHex(BuildID.slice(1), /*LowerCase=*/true));
    Path += ".debug";
    return Path;
  };
  if (DebugFileDirectory.empty()) {
    SmallString<128> Path = GetDebugPath(
#if defined(__NetBSD__)
        // Try /usr/libdata/debug/.build-id/../...
        "/usr/libdata/debug"
#else
        // Try /usr/lib/debug/.build-id/../...
        "/usr/lib/debug"
#endif
    );
    if (llvm::sys::fs::exists(Path))
      return std::string(Path);
  } else {
    for (const auto &Directory : DebugFileDirectory) {
      // Try <debug-file-directory>/.build-id/../...
      SmallString<128> Path = GetDebugPath(Directory);
      if (llvm::sys::fs::exists(Path))
        return std::string(Path);
    }
  }
  return None;
}

} // namespace symbolize
} // namespace llvm
