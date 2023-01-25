//===-- HostInfoSwift.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/common/HostInfoSwift.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfoBase.h"
#include "lldb/Host/Config.h"
#include "lldb/Utility/Log.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "lldb/Utility/LLDBLog.h"

bool lldb_private::VerifySwiftPath(const llvm::Twine &swift_path) {
  if (FileSystem::Instance().IsDirectory(swift_path))
    return true;
  Log *log = GetLog(LLDBLog::Host);
  if (log)
    log->Printf("VerifySwiftPath(): "
                "failed to stat swift resource directory at \"%s\"",
                swift_path.str().c_str());
  return false;
}

bool lldb_private::DefaultComputeSwiftResourceDirectory(
    FileSpec &lldb_shlib_spec, FileSpec &file_spec, bool verify) {
  if (!lldb_shlib_spec)
    return false;
  Log *log = GetLog(LLDBLog::Host);
  std::string raw_path = lldb_shlib_spec.GetPath();
  // Drop bin (windows) or lib
  llvm::StringRef parent_path = llvm::sys::path::parent_path(raw_path);

  static const llvm::StringRef kResourceDirSuffixes[] = {
      "lib/swift",
      LLDB_INSTALL_LIBDIR_BASENAME "/lldb/swift",
  };
  for (const auto &Suffix : kResourceDirSuffixes) {
    llvm::SmallString<256> swift_path(parent_path);
    llvm::SmallString<32> relative_path(Suffix);
    llvm::sys::path::append(swift_path, relative_path);
    if (!verify || VerifySwiftPath(swift_path)) {
      if (log)
        log->Printf("DefaultComputeSwiftResourceDir: Setting SwiftResourceDir "
                    "to \"%s\", verify = %s",
                    swift_path.str().str().c_str(), verify ? "true" : "false");
      file_spec.SetDirectory(swift_path);
      FileSystem::Instance().Resolve(file_spec);
      return true;
    }
  }
  return false;
}
