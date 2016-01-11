//===-- HostInfoBase.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/HostInfoBase.h"

#include "llvm/ADT/Triple.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <thread>
#include <mutex> // std::once

using namespace lldb;
using namespace lldb_private;

namespace
{
    void
    CleanupProcessSpecificLLDBTempDir()
    {
        // Get the process specific LLDB temporary directory and delete it.
        FileSpec tmpdir_file_spec;
        if (!HostInfo::GetLLDBPath(ePathTypeLLDBTempSystemDir, tmpdir_file_spec))
            return;

        // Remove the LLDB temporary directory if we have one. Set "recurse" to
        // true to all files that were created for the LLDB process can be cleaned up.
        FileSystem::DeleteDirectory(tmpdir_file_spec, true);
    }

    //----------------------------------------------------------------------
    // The HostInfoBaseFields is a work around for windows not supporting
    // static variables correctly in a thread safe way. Really each of the
    // variables in HostInfoBaseFields should live in the functions in which
    // they are used and each one should be static, but the work around is
    // in place to avoid this restriction. Ick.
    //----------------------------------------------------------------------

    struct HostInfoBaseFields
    {
        uint32_t m_number_cpus;
        std::string m_vendor_string;
        std::string m_os_string;
        std::string m_host_triple;

        ArchSpec m_host_arch_32;
        ArchSpec m_host_arch_64;

        FileSpec m_lldb_so_dir;
        FileSpec m_lldb_support_exe_dir;
        FileSpec m_lldb_headers_dir;
        FileSpec m_lldb_python_dir;
        FileSpec m_lldb_clang_resource_dir;
        FileSpec m_lldb_system_plugin_dir;
        FileSpec m_lldb_user_plugin_dir;
        FileSpec m_lldb_process_tmp_dir;
        FileSpec m_lldb_global_tmp_dir;
    };
    
    HostInfoBaseFields *g_fields = nullptr;
}

void
HostInfoBase::Initialize()
{
    g_fields = new HostInfoBaseFields();
}

uint32_t
HostInfoBase::GetNumberCPUS()
{
    static std::once_flag g_once_flag;
    std::call_once(g_once_flag,  []() {
        g_fields->m_number_cpus = std::thread::hardware_concurrency();
    });
    return g_fields->m_number_cpus;
}

uint32_t
HostInfoBase::GetMaxThreadNameLength()
{
    return 0;
}

llvm::StringRef
HostInfoBase::GetVendorString()
{
    static std::once_flag g_once_flag;
    std::call_once(g_once_flag,  []() {
        g_fields->m_vendor_string = HostInfo::GetArchitecture().GetTriple().getVendorName().str();
    });
    return g_fields->m_vendor_string;
}

llvm::StringRef
HostInfoBase::GetOSString()
{
    static std::once_flag g_once_flag;
    std::call_once(g_once_flag,  []() {
        g_fields->m_os_string = std::move(HostInfo::GetArchitecture().GetTriple().getOSName());
    });
    return g_fields->m_os_string;
}

llvm::StringRef
HostInfoBase::GetTargetTriple()
{
    static std::once_flag g_once_flag;
    std::call_once(g_once_flag,  []() {
        g_fields->m_host_triple = HostInfo::GetArchitecture().GetTriple().getTriple();
    });
    return g_fields->m_host_triple;
}

const ArchSpec &
HostInfoBase::GetArchitecture(ArchitectureKind arch_kind)
{
    static std::once_flag g_once_flag;
    std::call_once(g_once_flag,  []() {
        HostInfo::ComputeHostArchitectureSupport(g_fields->m_host_arch_32, g_fields->m_host_arch_64);
    });

    // If an explicit 32 or 64-bit architecture was requested, return that.
    if (arch_kind == eArchKind32)
        return g_fields->m_host_arch_32;
    if (arch_kind == eArchKind64)
        return g_fields->m_host_arch_64;

    // Otherwise prefer the 64-bit architecture if it is valid.
    return (g_fields->m_host_arch_64.IsValid()) ? g_fields->m_host_arch_64 : g_fields->m_host_arch_32;
}

bool
HostInfoBase::GetLLDBPath(lldb::PathType type, FileSpec &file_spec)
{
    file_spec.Clear();

#if defined(LLDB_DISABLE_PYTHON)
    if (type == lldb::ePathTypePythonDir)
        return false;
#endif

    FileSpec *result = nullptr;
    switch (type)
    {
        case lldb::ePathTypeLLDBShlibDir:
            {
                static std::once_flag g_once_flag;
                static bool success = false;
                std::call_once(g_once_flag,  []() {
                    success = HostInfo::ComputeSharedLibraryDirectory (g_fields->m_lldb_so_dir);
                    Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
                    if (log)
                        log->Printf("HostInfoBase::GetLLDBPath(ePathTypeLLDBShlibDir) => '%s'", g_fields->m_lldb_so_dir.GetPath().c_str());
                });
                if (success)
                    result = &g_fields->m_lldb_so_dir;
            }
            break;
        case lldb::ePathTypeSupportExecutableDir:
            {
                static std::once_flag g_once_flag;
                static bool success = false;
                std::call_once(g_once_flag,  []() {
                    success = HostInfo::ComputeSupportExeDirectory (g_fields->m_lldb_support_exe_dir);
                    Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
                    if (log)
                        log->Printf("HostInfoBase::GetLLDBPath(ePathTypeSupportExecutableDir) => '%s'",
                                    g_fields->m_lldb_support_exe_dir.GetPath().c_str());
                });
                if (success)
                    result = &g_fields->m_lldb_support_exe_dir;
            }
            break;
        case lldb::ePathTypeHeaderDir:
            {
                static std::once_flag g_once_flag;
                static bool success = false;
                std::call_once(g_once_flag,  []() {
                    success = HostInfo::ComputeHeaderDirectory (g_fields->m_lldb_headers_dir);
                    Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
                    if (log)
                        log->Printf("HostInfoBase::GetLLDBPath(ePathTypeHeaderDir) => '%s'", g_fields->m_lldb_headers_dir.GetPath().c_str());
                });
                if (success)
                    result = &g_fields->m_lldb_headers_dir;
            }
            break;
        case lldb::ePathTypePythonDir:
            {
                static std::once_flag g_once_flag;
                static bool success = false;
                std::call_once(g_once_flag,  []() {
                    success = HostInfo::ComputePythonDirectory (g_fields->m_lldb_python_dir);
                    Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
                    if (log)
                        log->Printf("HostInfoBase::GetLLDBPath(ePathTypePythonDir) => '%s'", g_fields->m_lldb_python_dir.GetPath().c_str());
                });
                if (success)
                    result = &g_fields->m_lldb_python_dir;
            }
            break;
        case lldb::ePathTypeClangDir:
            {
                static std::once_flag g_once_flag;
                static bool success = false;
                std::call_once(g_once_flag,  []() {
                    success = HostInfo::ComputeClangDirectory (g_fields->m_lldb_clang_resource_dir);
                    Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
                    if (log)
                        log->Printf("HostInfoBase::GetLLDBPath(ePathTypeClangResourceDir) => '%s'", g_fields->m_lldb_clang_resource_dir.GetPath().c_str());
                });
                if (success)
                    result = &g_fields->m_lldb_clang_resource_dir;
            }
            break;
        case lldb::ePathTypeLLDBSystemPlugins:
            {
                static std::once_flag g_once_flag;
                static bool success = false;
                std::call_once(g_once_flag,  []() {
                    success = HostInfo::ComputeSystemPluginsDirectory (g_fields->m_lldb_system_plugin_dir);
                    Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
                    if (log)
                        log->Printf("HostInfoBase::GetLLDBPath(ePathTypeLLDBSystemPlugins) => '%s'",
                                    g_fields->m_lldb_system_plugin_dir.GetPath().c_str());
                });
                if (success)
                    result = &g_fields->m_lldb_system_plugin_dir;
            }
            break;
        case lldb::ePathTypeLLDBUserPlugins:
            {
                static std::once_flag g_once_flag;
                static bool success = false;
                std::call_once(g_once_flag,  []() {
                    success = HostInfo::ComputeUserPluginsDirectory (g_fields->m_lldb_user_plugin_dir);
                    Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
                    if (log)
                        log->Printf("HostInfoBase::GetLLDBPath(ePathTypeLLDBUserPlugins) => '%s'",
                                    g_fields->m_lldb_user_plugin_dir.GetPath().c_str());
                });
                if (success)
                    result = &g_fields->m_lldb_user_plugin_dir;
            }
            break;
        case lldb::ePathTypeLLDBTempSystemDir:
            {
                static std::once_flag g_once_flag;
                static bool success = false;
                std::call_once(g_once_flag,  []() {
                    success = HostInfo::ComputeProcessTempFileDirectory (g_fields->m_lldb_process_tmp_dir);
                    Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
                    if (log)
                        log->Printf("HostInfoBase::GetLLDBPath(ePathTypeLLDBTempSystemDir) => '%s'", g_fields->m_lldb_process_tmp_dir.GetPath().c_str());
                });
                if (success)
                    result = &g_fields->m_lldb_process_tmp_dir;
            }
            break;
        case lldb::ePathTypeGlobalLLDBTempSystemDir:
            {
                static std::once_flag g_once_flag;
                static bool success = false;
                std::call_once(g_once_flag,  []() {
                    success = HostInfo::ComputeGlobalTempFileDirectory (g_fields->m_lldb_global_tmp_dir);
                    Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
                    if (log)
                        log->Printf("HostInfoBase::GetLLDBPath(ePathTypeGlobalLLDBTempSystemDir) => '%s'", g_fields->m_lldb_global_tmp_dir.GetPath().c_str());
                });
                if (success)
                    result = &g_fields->m_lldb_global_tmp_dir;
            }
            break;
    }

    if (!result)
        return false;
    file_spec = *result;
    return true;
}

bool
HostInfoBase::ComputeSharedLibraryDirectory(FileSpec &file_spec)
{
    // To get paths related to LLDB we get the path to the executable that
    // contains this function. On MacOSX this will be "LLDB.framework/.../LLDB",
    // on linux this is assumed to be the "lldb" main executable. If LLDB on
    // linux is actually in a shared library (liblldb.so) then this function will
    // need to be modified to "do the right thing".

    FileSpec lldb_file_spec(
        Host::GetModuleFileSpecForHostAddress(reinterpret_cast<void *>(reinterpret_cast<intptr_t>(HostInfoBase::GetLLDBPath))));
    
    // This is necessary because when running the testsuite the shlib might be a symbolic link inside the Python resource dir.
    FileSystem::ResolveSymbolicLink(lldb_file_spec, lldb_file_spec);
    
    // Remove the filename so that this FileSpec only represents the directory.
    file_spec.GetDirectory() = lldb_file_spec.GetDirectory();

    return (bool)file_spec.GetDirectory();
}

bool
HostInfoBase::ComputeSupportExeDirectory(FileSpec &file_spec)
{
    return GetLLDBPath(lldb::ePathTypeLLDBShlibDir, file_spec);
}

bool
HostInfoBase::ComputeProcessTempFileDirectory(FileSpec &file_spec)
{
    FileSpec temp_file_spec;
    if (!HostInfo::ComputeGlobalTempFileDirectory(temp_file_spec))
        return false;

    std::string pid_str{std::to_string(Host::GetCurrentProcessID())};
    temp_file_spec.AppendPathComponent(pid_str);
    if (!FileSystem::MakeDirectory(temp_file_spec, eFilePermissionsDirectoryDefault).Success())
        return false;

    // Make an atexit handler to clean up the process specify LLDB temp dir
    // and all of its contents.
    ::atexit(CleanupProcessSpecificLLDBTempDir);
    file_spec.GetDirectory().SetCString(temp_file_spec.GetCString());
    return true;
}

bool
HostInfoBase::ComputeTempFileBaseDirectory(FileSpec &file_spec)
{
    llvm::SmallVector<char, 16> tmpdir;
    llvm::sys::path::system_temp_directory(/*ErasedOnReboot*/ true, tmpdir);
    file_spec = FileSpec(std::string(tmpdir.data(), tmpdir.size()), true);
    return true;
}

bool
HostInfoBase::ComputeGlobalTempFileDirectory(FileSpec &file_spec)
{
    file_spec.Clear();

    FileSpec temp_file_spec;
    if (!HostInfo::ComputeTempFileBaseDirectory(temp_file_spec))
        return false;

    temp_file_spec.AppendPathComponent("lldb");
    if (!FileSystem::MakeDirectory(temp_file_spec, eFilePermissionsDirectoryDefault).Success())
        return false;

    file_spec.GetDirectory().SetCString(temp_file_spec.GetCString());
    return true;
}

bool
HostInfoBase::ComputeHeaderDirectory(FileSpec &file_spec)
{
    // TODO(zturner): Figure out how to compute the header directory for all platforms.
    return false;
}

bool
HostInfoBase::ComputeSystemPluginsDirectory(FileSpec &file_spec)
{
    // TODO(zturner): Figure out how to compute the system plugins directory for all platforms.
    return false;
}

bool
HostInfoBase::ComputeClangDirectory(FileSpec &file_spec)
{
    return false;
}

bool
HostInfoBase::ComputeUserPluginsDirectory(FileSpec &file_spec)
{
    // TODO(zturner): Figure out how to compute the user plugins directory for all platforms.
    return false;
}

void
HostInfoBase::ComputeHostArchitectureSupport(ArchSpec &arch_32, ArchSpec &arch_64)
{
    llvm::Triple triple(llvm::sys::getProcessTriple());

    arch_32.Clear();
    arch_64.Clear();

    switch (triple.getArch())
    {
        default:
            arch_32.SetTriple(triple);
            break;

        case llvm::Triple::aarch64:
        case llvm::Triple::ppc64:
        case llvm::Triple::x86_64:
            arch_64.SetTriple(triple);
            arch_32.SetTriple(triple.get32BitArchVariant());
            break;

        case llvm::Triple::mips64:
        case llvm::Triple::mips64el:
        case llvm::Triple::sparcv9:
            arch_64.SetTriple(triple);
            break;
    }
}
