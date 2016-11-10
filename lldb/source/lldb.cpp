//===-- lldb.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private.h"
#include "llvm/ADT/StringExtras.h"

using namespace lldb;
using namespace lldb_private;

#include "clang/Basic/Version.h"
#include "swift/Basic/Version.h"

#ifdef HAVE_SVN_VERSION_INC
#  include "SVNVersion.inc"
#endif

#ifdef HAVE_APPLE_VERSION_INC
#  include "AppleVersion.inc"
#endif

static const char *GetLLDBRevision() {
#ifdef LLDB_REVISION
  static const char *s_revision = LLDB_REVISION;
#else
  static const char *s_revision = nullptr;
#endif

  // If LLDB_REVISION is defined but isn't set to a string, it
  // can still be the equivalent of NULL.  Hence we always do
  // this check below and return an empty string when we don't
  // otherwise have a valid const string for it.
  if (s_revision != nullptr)
    return s_revision;
  else
    return "";
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
    if (lldb_repo) {
      g_version_str += " (";
      g_version_str += lldb_repo;
    }

    const char *lldb_rev = GetLLDBRevision();
    if (lldb_rev) {
      g_version_str += " revision ";
      g_version_str += lldb_rev;
      g_version_str += ")";
    }
#ifdef LLDB_VERSION_STRING
    g_version_str += " (";
    g_version_str += EXPAND_AND_QUOTE(LLDB_VERSION_STRING);
#if LLDB_IS_BUILDBOT_BUILD
    g_version_str = "buildbot";
    std::string build_date = GetBuildDate();
    if(!build_date.empty())
      g_version_str += " (" + build_date + ")";
#endif
    g_version_str += ")";
#endif

    auto const swift_version = swift::version::getSwiftNumericVersion();
    g_version_str += "\n  Swift-";
    g_version_str += llvm::utostr(swift_version.first) + ".";
    g_version_str += llvm::utostr(swift_version.second);
    std::string swift_rev(swift::version::getSwiftRevision());
    if (swift_rev.length() > 0) {
      g_version_str += " (revision " + swift_rev + ")";
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
