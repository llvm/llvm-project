//===-- HostInfoWindows.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/windows.h"

#include <objbase.h>

#include <mutex> // std::once

#include "lldb/Host/windows/HostInfoWindows.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Path.h"

using namespace lldb_private;

FileSpec HostInfoWindows::m_program_filespec;

void
HostInfoWindows::Initialize()
{
    ::CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    HostInfoBase::Initialize();
}

void
HostInfoWindows::Terminate()
{
    HostInfoBase::Terminate();
    ::CoUninitialize();
}

size_t
HostInfoWindows::GetPageSize()
{
    SYSTEM_INFO systemInfo;
    GetNativeSystemInfo(&systemInfo);
    return systemInfo.dwPageSize;
}

bool
HostInfoWindows::GetOSVersion(uint32_t &major, uint32_t &minor, uint32_t &update)
{
    OSVERSIONINFOEX info;

    ZeroMemory(&info, sizeof(OSVERSIONINFOEX));
    info.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);
#pragma warning(push)
#pragma warning(disable : 4996)
    // Starting with Microsoft SDK for Windows 8.1, this function is deprecated in favor of the
    // new Windows Version Helper APIs.  Since we don't specify a minimum SDK version, it's easier
    // to simply disable the warning rather than try to support both APIs.
    if (GetVersionEx((LPOSVERSIONINFO)&info) == 0)
    {
        return false;
    }
#pragma warning(pop)

    major = info.dwMajorVersion;
    minor = info.dwMinorVersion;
    update = info.wServicePackMajor;

    return true;
}

bool
HostInfoWindows::GetOSBuildString(std::string &s)
{
    s.clear();
    uint32_t major, minor, update;
    if (!GetOSVersion(major, minor, update))
        return false;

    llvm::raw_string_ostream stream(s);
    stream << "Windows NT " << major << "." << minor << "." << update;
    return true;
}

bool
HostInfoWindows::GetOSKernelDescription(std::string &s)
{
    return GetOSBuildString(s);
}

bool
HostInfoWindows::GetHostname(std::string &s)
{
    char buffer[MAX_COMPUTERNAME_LENGTH + 1];
    DWORD dwSize = MAX_COMPUTERNAME_LENGTH + 1;
    if (!::GetComputerName(buffer, &dwSize))
        return false;

    s.assign(buffer, buffer + dwSize);
    return true;
}

FileSpec
HostInfoWindows::GetProgramFileSpec()
{
    static std::once_flag g_once_flag;
    std::call_once(g_once_flag,  []() {
        char buffer[PATH_MAX];
        ::GetModuleFileName(NULL, buffer, sizeof(buffer));
        m_program_filespec.SetFile(buffer, false);
    });
    return m_program_filespec;
}

FileSpec
HostInfoWindows::GetDefaultShell()
{
    return FileSpec(::getenv("ComSpec"), false);
}

bool
HostInfoWindows::ComputePythonDirectory(FileSpec &file_spec)
{
    FileSpec lldb_file_spec;
    if (!GetLLDBPath(lldb::ePathTypeLLDBShlibDir, lldb_file_spec))
        return false;
    llvm::SmallString<64> path(lldb_file_spec.GetDirectory().AsCString());
    llvm::sys::path::remove_filename(path);
    llvm::sys::path::append(path, "lib", "site-packages");
    std::replace(path.begin(), path.end(), '\\', '/');
    file_spec.GetDirectory().SetString(path.c_str());
    return true;
}
