//===-- lldb.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private.h"

using namespace lldb;
using namespace lldb_private;

#if defined(__APPLE__)
// Xcode writes this file out in a build step.
// cmake builds get this through another mechanism.
// Both produce LLDB_REVISION.
#include "lldb_revision.h"
extern "C" const unsigned char liblldb_coreVersionString[];
#endif

#if defined(LLDB_USE_OSS_VERSION_SCHEME)

#include "swift/Basic/Version.h"
#include "llvm/Support/raw_ostream.h"

#endif

#if !defined(__APPLE__) || defined(LLDB_USE_OSS_VERSION_SCHEME)

#include "clang/Basic/Version.h"

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

#endif

#if !defined(__APPLE__)

static const char *GetLLDBRepository() {
#ifdef LLDB_REPOSITORY
  return LLDB_REPOSITORY;
#else
  return NULL;
#endif
}

#endif

#if defined(LLDB_USE_OSS_VERSION_SCHEME)

// TODO remove this function once swift revision is directly exposed.
std::string ExtractSwiftRevision(const std::string &fullVersion) {
  // Find spot right before Swift revision.
  const std::string search_prefix = "Swift ";
  const size_t prefix_start_pos = fullVersion.rfind(search_prefix);
  if (prefix_start_pos == std::string::npos)
    return "";

  // Find spot where Swift revision ends.
  const size_t revision_end_pos = fullVersion.rfind(')');
  if (revision_end_pos == std::string::npos)
    return "";

  // Return the revision.
  return fullVersion.substr(prefix_start_pos + search_prefix.length(),
                            revision_end_pos - prefix_start_pos -
                                search_prefix.length());
}

static std::string GetBuildDate() {
// The following code helps distinguish between a defined preprocessor
// value that gets the default value assigned vs. one with an explicit value.
#define LLDB_DO_MACRO_EXPAND(macro_value) macro_value##1
#define LLDB_MACRO_EXPAND(macro_value) LLDB_DO_MACRO_EXPAND(macro_value)

#if defined(LLDB_BUILD_DATE)
  return std::string(LLDB_BUILD_DATE);
#else
  return std::string();
#endif
}

static const char *_GetVersionOSS() {
  static std::string g_version_string;
  if (g_version_string.empty()) {
    std::string build_string;
    llvm::raw_string_ostream out(build_string);

    std::string build_flavor = "local";
#if defined(LLDB_IS_BUILDBOT_BUILD)
#if (LLDB_IS_BUILDBOT_BUILD != 0)
    build_flavor = "buildbot";
#endif
#endif
    out << "lldb-" << build_flavor;

    // We only run this code when the build date is both set and non-default.
    // Otherwise this code doesn't compile.
    const std::string build_date(GetBuildDate());
    if (!build_date.empty())
      out << "-" << build_date;

    out << " (";

    std::string lldb_revision = GetLLDBRevision();
    if (lldb_revision.length() > 0) {
      const size_t MAX_REVISION_LENGTH = 10;

      out << "LLDB ";
      out << lldb_revision.substr(0, MAX_REVISION_LENGTH);

      const std::string llvm_revision = clang::getLLVMRevision();
      if (!llvm_revision.empty())
        out << ", LLVM " << llvm_revision.substr(0, MAX_REVISION_LENGTH);

      const std::string clang_revision = clang::getClangRevision();
      if (!clang_revision.empty())
        out << ", Clang " << clang_revision.substr(0, MAX_REVISION_LENGTH);

      // TODO replace this with a swift::version::GetSwiftRevision() call
      // once added.
      const std::string swift_revision =
          ExtractSwiftRevision(swift::version::getSwiftFullVersion());
      if (!swift_revision.empty()) {
        auto const swift_version = swift::version::getSwiftNumericVersion();
        out << ", Swift-" << swift_version.first << "." << swift_version.second
            << " " << swift_revision.substr(0, MAX_REVISION_LENGTH);
      }
    }
    out << ")";

    g_version_string = out.str();
  }
  return g_version_string.c_str();
}

#endif

#if defined(__APPLE__) && !defined(LLDB_USE_OSS_VERSION_SCHEME)

static const char *_GetVersionAppleStandard() {
  static char g_version_string[32];
  if (g_version_string[0] == '\0') {
    const char *version_string =
        ::strstr((const char *)liblldb_coreVersionString, "PROJECT:");

    if (version_string)
      version_string += sizeof("PROJECT:") - 1;
    else
      version_string = "unknown";

    const char *newline_loc = strchr(version_string, '\n');

    size_t version_len = sizeof(g_version_string) - 1;

    if (newline_loc &&
        (newline_loc - version_string < static_cast<ptrdiff_t>(version_len)))
      version_len = newline_loc - version_string;

    ::snprintf(g_version_string, version_len + 1, "%s", version_string);
  }

  return g_version_string;
}

#endif

const char *lldb_private::GetVersion() {
#if defined(__APPLE__)
#if defined(LLDB_USE_OSS_VERSION_SCHEME)
  return _GetVersionOSS();
#else
  return _GetVersionAppleStandard();
#endif
#else
#if defined(LLDB_USE_OSS_VERSION_SCHEME)
  return _GetVersionOSS();
#else
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
    }
    std::string clang_rev(clang::getClangRevision());
    if (clang_rev.length() > 0) {
      g_version_str += " clang revision ";
      g_version_str += clang_rev;
    }
    std::string llvm_rev(clang::getLLVMRevision());
    if (llvm_rev.length() > 0) {
      g_version_str += " llvm revision ";
      g_version_str += llvm_rev;
    }

    if (lldb_repo)
      g_version_str += ")";
  }
  return g_version_str.c_str();
#endif
#endif
}
