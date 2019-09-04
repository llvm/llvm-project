//===-- SwiftHost.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SwiftHost.h"

#include "lldb/Host/Config.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Log.h"

#include <string>

using namespace lldb_private;

static bool VerifySwiftPath(const llvm::Twine &swift_path) {
  if (FileSystem::Instance().IsDirectory(swift_path))
    return true;
  Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
  if (log)
    log->Printf("VerifySwiftPath(): "
                "failed to stat swift resource directory at \"%s\"",
                swift_path.str().c_str());
  return false;
}

static bool DefaultComputeSwiftResourceDirectory(FileSpec &lldb_shlib_spec,
                                                 FileSpec &file_spec,
                                                 bool verify) {
  if (!lldb_shlib_spec)
    return false;
  Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
  std::string raw_path = lldb_shlib_spec.GetPath();
  // Drop bin (windows) or lib
  llvm::StringRef parent_path = llvm::sys::path::parent_path(raw_path);

  static const llvm::StringRef kResourceDirSuffixes[] = {
      "lib/swift",
      "lib" LLDB_LIBDIR_SUFFIX "/lldb/swift",
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
      file_spec.GetDirectory().SetString(swift_path);
      FileSystem::Instance().Resolve(file_spec);
      return true;
    }
  }
  return false;
}

bool lldb_private::ComputeSwiftResourceDirectory(FileSpec &lldb_shlib_spec,
                                                 FileSpec &file_spec,
                                                 bool verify) {
#if !defined(__APPLE__)
  return DefaultComputeSwiftResourceDirectory(lldb_shlib_spec, file_spec,
                                              verify);
#else
  if (!lldb_shlib_spec)
    return false;

  std::string raw_path = lldb_shlib_spec.GetPath();
  size_t framework_pos = raw_path.find("LLDB.framework");
  if (framework_pos == std::string::npos)
    return DefaultComputeSwiftResourceDirectory(lldb_shlib_spec, file_spec,
                                                verify);

  framework_pos += strlen("LLDB.framework");
  raw_path.resize(framework_pos);
  raw_path.append("/Resources/Swift");
  if (!verify || VerifySwiftPath(raw_path)) {
    file_spec.GetDirectory().SetString(raw_path);
    FileSystem::Instance().Resolve(file_spec);
    return true;
  }
  return true;
#endif // __APPLE__
}

FileSpec lldb_private::GetSwiftResourceDir() {
  static std::once_flag g_once_flag;
  static FileSpec g_swift_resource_dir;
  std::call_once(g_once_flag, []() {
    FileSpec lldb_file_spec = HostInfo::GetShlibDir();
    ComputeSwiftResourceDirectory(lldb_file_spec, g_swift_resource_dir, true);
    Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
    LLDB_LOG(log, "swift dir -> '{0}'", g_swift_resource_dir);
  });
  return g_swift_resource_dir;
}
