//===-- CppModuleConfiguration.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CppModuleConfiguration.h"

#include "ClangHost.h"
#include "lldb/Host/FileSystem.h"

using namespace lldb_private;

bool CppModuleConfiguration::SetOncePath::TrySet(llvm::StringRef path) {
  // Setting for the first time always works.
  if (m_first) {
    m_path = path.str();
    m_valid = true;
    m_first = false;
    return true;
  }
  // Changing the path to the same value is fine.
  if (m_path == path)
    return true;

  // Changing the path after it was already set is not allowed.
  m_valid = false;
  return false;
}

bool CppModuleConfiguration::analyzeFile(const FileSpec &f) {
  using namespace llvm::sys::path;
  // Convert to slashes to make following operations simpler.
  std::string dir_buffer = convert_to_slash(f.GetDirectory().GetStringRef());
  llvm::StringRef posix_dir(dir_buffer);

  // Check for /c++/vX/ that is used by libc++.
  static llvm::Regex libcpp_regex(R"regex(/c[+][+]/v[0-9]/)regex");
  // If the path is in the libc++ include directory use it as the found libc++
  // path. Ignore subdirectories such as /c++/v1/experimental as those don't
  // need to be specified in the header search.
  if (libcpp_regex.match(f.GetPath()) &&
      parent_path(posix_dir, Style::posix).endswith("c++")) {
    return m_std_inc.TrySet(posix_dir);
  }

  // Check for /usr/include. On Linux this might be /usr/include/bits, so
  // we should remove that '/bits' suffix to get the actual include directory.
  if (posix_dir.endswith("/usr/include/bits"))
    posix_dir.consume_back("/bits");
  if (posix_dir.endswith("/usr/include"))
    return m_c_inc.TrySet(posix_dir);

  // File wasn't interesting, continue analyzing.
  return true;
}

bool CppModuleConfiguration::hasValidConfig() {
  // We all these include directories to have a valid usable configuration.
  return m_c_inc.Valid() && m_std_inc.Valid();
}

CppModuleConfiguration::CppModuleConfiguration(
    const FileSpecList &support_files) {
  // Analyze all files we were given to build the configuration.
  bool error = !llvm::all_of(support_files,
                             std::bind(&CppModuleConfiguration::analyzeFile,
                                       this, std::placeholders::_1));
  // If we have a valid configuration at this point, set the
  // include directories and module list that should be used.
  if (!error && hasValidConfig()) {
    // Calculate the resource directory for LLDB.
    llvm::SmallString<256> resource_dir;
    llvm::sys::path::append(resource_dir, GetClangResourceDir().GetPath(),
                            "include");
    m_resource_inc = std::string(resource_dir.str());

    // This order matches the way Clang orders these directories.
    m_include_dirs = {m_std_inc.Get(), m_resource_inc, m_c_inc.Get()};
    m_imported_modules = {"std"};
  }
}
