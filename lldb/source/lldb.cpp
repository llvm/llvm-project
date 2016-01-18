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

#if defined (__APPLE__)
extern "C" const unsigned char liblldb_coreVersionString[];
#else

#include "clang/Basic/Version.h"

static const char *
GetLLDBRevision()
{
#ifdef LLDB_REVISION
    return LLDB_REVISION;
#else
    return NULL;
#endif
}

static const char *
GetLLDBRepository()
{
#ifdef LLDB_REPOSITORY
    return LLDB_REPOSITORY;
#else
    return NULL;
#endif
}

#endif

const char *
lldb_private::GetVersion ()
{
#if defined (__APPLE__)
    static char g_version_string[32];
    if (g_version_string[0] == '\0')
    {
        const char *version_string = ::strstr ((const char *)liblldb_coreVersionString, "PROJECT:");
        
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
#else
    // On Linux/FreeBSD/Windows, report a version number in the same style as the clang tool.
    static std::string g_version_str;
    if (g_version_str.empty())
    {
        g_version_str += "lldb version ";
        g_version_str += CLANG_VERSION_STRING;
        const char * lldb_repo = GetLLDBRepository();
        if (lldb_repo)
        {
            g_version_str += " (";
            g_version_str += lldb_repo;
        }

        const char *lldb_rev = GetLLDBRevision();
        if (lldb_rev)
        {
            g_version_str += " revision ";
            g_version_str += lldb_rev;
        }
        std::string clang_rev (clang::getClangRevision());
        if (clang_rev.length() > 0)
        {
            g_version_str += " clang revision ";
            g_version_str += clang_rev;
        }
        std::string llvm_rev (clang::getLLVMRevision());
        if (llvm_rev.length() > 0)
        {
            g_version_str += " llvm revision ";
            g_version_str += llvm_rev;
        }

        if (lldb_repo)
            g_version_str += ")";
    }
    return g_version_str.c_str();
#endif
}
