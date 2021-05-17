//===-- lldb.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VCSVersion.inc"
#include "lldb/lldb-private.h"
#include "clang/Basic/Version.h"

using namespace lldb;
using namespace lldb_private;

// LLDB_VERSION_STRING is set through a define so unlike the other defines
// expanded with CMake, it lacks the double quotes.
#define QUOTE(str) #str
#define EXPAND_AND_QUOTE(str) QUOTE(str)

static const char *GetLLDBVersion() {
#ifdef LLDB_VERSION_STRING
  return EXPAND_AND_QUOTE(LLDB_VERSION_STRING);
#else
  return "lldb version " CLANG_VERSION_STRING;
#endif
}

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

const char *lldb_private::GetVersion() {
  static std::string g_version_str;
  if (g_version_str.empty()) {
    const char *lldb_version = GetLLDBVersion();
    const char *lldb_repo = GetLLDBRepository();
    const char *lldb_rev = GetLLDBRevision();
    g_version_str += lldb_version;
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
  }
  return g_version_str.c_str();
}
