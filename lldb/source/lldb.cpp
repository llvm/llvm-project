//===-- lldb.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private.h"

using namespace lldb;
using namespace lldb_private;

#include "clang/Basic/Version.h"
#ifdef LLDB_ENABLE_SWIFT
#include "swift/Basic/Version.h"
#endif // LLDB_ENABLE_SWIFT

#ifdef HAVE_VCS_VERSION_INC
#include "VCSVersion.inc"
#endif

static const char *GetLLDBRevision() {
#ifdef LLDB_REVISION
  return LLDB_REVISION;
#else
  return NULL;
#endif
}

static const char *GetLLDBRepository() {
#ifdef LLDB_REPOSITORY
  return LLDB_REPOSITORY;
#else
  return NULL;
#endif
}

#if LLDB_IS_BUILDBOT_BUILD
static std::string GetBuildDate() {
#if defined(LLDB_BUILD_DATE)
  return std::string(LLDB_BUILD_DATE);
#else
  return std::string();
#endif
}
#endif

#define QUOTE(str) #str
#define EXPAND_AND_QUOTE(str) QUOTE(str)

const char *lldb_private::GetVersion() {
  // On platforms other than Darwin, report a version number in the same style
  // as the clang tool.
  static std::string g_version_str;
  if (g_version_str.empty()) {
    g_version_str += "lldb version ";
    g_version_str += CLANG_VERSION_STRING;

    const char *lldb_repo = GetLLDBRepository();
    const char *lldb_rev = GetLLDBRevision();
    if (lldb_repo || lldb_rev) {
      g_version_str += " (";
      if (lldb_repo)
        g_version_str += lldb_repo;
      if (lldb_repo && lldb_rev)
        g_version_str += " ";
      if (lldb_rev) {
        g_version_str += "revision ";
        g_version_str += lldb_rev;
      }
      g_version_str += ")";
    }
    
#if LLDB_IS_BUILDBOT_BUILD
    std::string build_date = GetBuildDate();
    if(!build_date.empty())
      g_version_str += " (buildbot " + build_date + ")";
#endif

#ifdef LLDB_ENABLE_SWIFT
    auto const swift_version = swift::version::getSwiftFullVersion();
    g_version_str += "\n" + swift_version;
#endif // LLDB_ENABLE_SWIFT

    // getSwiftFullVersion() also prints clang and llvm versions, no
    // need to print them again. We keep this code here to not diverge
    // too much from upstream.
#undef LLDB_UPSTREAM
#ifdef LLDB_UPSTREAM
    std::string clang_rev(clang::getClangRevision());
    if (clang_rev.length() > 0) {
      g_version_str += "\n  clang revision ";
      g_version_str += clang_rev;
    }
    std::string llvm_rev(clang::getLLVMRevision());
    if (llvm_rev.length() > 0) {
      g_version_str += "\n  llvm revision ";
      g_version_str += llvm_rev;
    }
#endif // LLDB_UPSTREAM
  }
  return g_version_str.c_str();
}
