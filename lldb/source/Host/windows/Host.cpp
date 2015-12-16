//===-- source/Host/windows/Host.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <stdio.h>
#include "lldb/Host/windows/windows.h"
#include "lldb/Host/windows/AutoHandle.h"

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Target/Process.h"

#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/StructuredData.h"

// Windows includes
#include <TlHelp32.h>

using namespace lldb;
using namespace lldb_private;

namespace
{
    bool GetTripleForProcess(const FileSpec &executable, llvm::Triple &triple)
    {
        // Open the PE File as a binary file, and parse just enough information to determine the
        // machine type.
        File imageBinary(
            executable.GetPath().c_str(),
            File::eOpenOptionRead,
            lldb::eFilePermissionsUserRead);
        imageBinary.SeekFromStart(0x3c);
        int32_t peOffset = 0;
        uint32_t peHead = 0;
        uint16_t machineType = 0;
        size_t readSize = sizeof(peOffset);
        imageBinary.Read(&peOffset, readSize);
        imageBinary.SeekFromStart(peOffset);
        imageBinary.Read(&peHead, readSize);
        if (peHead != 0x00004550) // "PE\0\0", little-endian
            return false;       // Error: Can't find PE header
        readSize = 2;
        imageBinary.Read(&machineType, readSize);
        triple.setVendor(llvm::Triple::PC);
        triple.setOS(llvm::Triple::Win32);
        triple.setArch(llvm::Triple::UnknownArch);
        if (machineType == 0x8664)
            triple.setArch(llvm::Triple::x86_64);
        else if (machineType == 0x14c)
            triple.setArch(llvm::Triple::x86);

        return true;
    }

    bool GetExecutableForProcess(const AutoHandle &handle, std::string &path)
    {
        // Get the process image path.  MAX_PATH isn't long enough, paths can actually be up to 32KB.
        std::vector<char> buffer(32768);
        DWORD dwSize = buffer.size();
        if (!::QueryFullProcessImageNameA(handle.get(), 0, &buffer[0], &dwSize))
            return false;
        path.assign(&buffer[0]);
        return true;
    }

    void GetProcessExecutableAndTriple(const AutoHandle &handle, ProcessInstanceInfo &process)
    {
        // We may not have permissions to read the path from the process.  So start off by
        // setting the executable file to whatever Toolhelp32 gives us, and then try to
        // enhance this with more detailed information, but fail gracefully.
        std::string executable;
        llvm::Triple triple;
        triple.setVendor(llvm::Triple::PC);
        triple.setOS(llvm::Triple::Win32);
        triple.setArch(llvm::Triple::UnknownArch);
        if (GetExecutableForProcess(handle, executable))
        {
            FileSpec executableFile(executable.c_str(), false);
            process.SetExecutableFile(executableFile, true);
            GetTripleForProcess(executableFile, triple);
        }
        process.SetArchitecture(ArchSpec(triple));

        // TODO(zturner): Add the ability to get the process user name.
    }
}

lldb::DataBufferSP
Host::GetAuxvData(lldb_private::Process *process)
{
    return 0;
}

lldb::tid_t
Host::GetCurrentThreadID()
{
    return lldb::tid_t(::GetCurrentThreadId());
}

lldb::thread_t
Host::GetCurrentThread ()
{
    return lldb::thread_t(::GetCurrentThread());
}

lldb::thread_key_t
Host::ThreadLocalStorageCreate(ThreadLocalStorageCleanupCallback callback)
{
    return TlsAlloc();
}

void*
Host::ThreadLocalStorageGet(lldb::thread_key_t key)
{
    return ::TlsGetValue (key);
}

void
Host::ThreadLocalStorageSet(lldb::thread_key_t key, void *value)
{
   ::TlsSetValue (key, value);
}

void
Host::Kill(lldb::pid_t pid, int signo)
{
    TerminateProcess((HANDLE) pid, 1);
}


const char *
Host::GetSignalAsCString(int signo)
{
    return NULL;
}

FileSpec
Host::GetModuleFileSpecForHostAddress (const void *host_addr)
{
    FileSpec module_filespec;

    HMODULE hmodule = NULL;
    if (!::GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, (LPCTSTR)host_addr, &hmodule))
        return module_filespec;

    std::vector<char> buffer(MAX_PATH);
    DWORD chars_copied = 0;
    do {
        chars_copied = ::GetModuleFileName(hmodule, &buffer[0], buffer.size());
        if (chars_copied == buffer.size() && ::GetLastError() == ERROR_INSUFFICIENT_BUFFER)
            buffer.resize(buffer.size() * 2);
    } while (chars_copied >= buffer.size());

    module_filespec.SetFile(&buffer[0], false);
    return module_filespec;
}

uint32_t
Host::FindProcesses (const ProcessInstanceInfoMatch &match_info, ProcessInstanceInfoList &process_infos)
{
    process_infos.Clear();

    AutoHandle snapshot(CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0));
    if (!snapshot.IsValid())
        return 0;

    PROCESSENTRY32 pe = {0};
    pe.dwSize = sizeof(PROCESSENTRY32);
    if (Process32First(snapshot.get(), &pe))
    {
        do
        {
            AutoHandle handle(::OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pe.th32ProcessID), nullptr);

            ProcessInstanceInfo process;
            process.SetExecutableFile(FileSpec(pe.szExeFile, false), true);
            process.SetProcessID(pe.th32ProcessID);
            process.SetParentProcessID(pe.th32ParentProcessID);
            GetProcessExecutableAndTriple(handle, process);

            if (match_info.MatchAllProcesses() || match_info.Matches(process))
                process_infos.Append(process);
        } while (Process32Next(snapshot.get(), &pe));
    }
    return process_infos.GetSize();
}

bool
Host::GetProcessInfo (lldb::pid_t pid, ProcessInstanceInfo &process_info)
{
    process_info.Clear();

    AutoHandle handle(::OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid),
                      nullptr);
    if (!handle.IsValid())
        return false;
    
    process_info.SetProcessID(pid);
    GetProcessExecutableAndTriple(handle, process_info);

    // Need to read the PEB to get parent process and command line arguments.
    return true;
}

HostThread
Host::StartMonitoringChildProcess(Host::MonitorChildProcessCallback callback, void *callback_baton, lldb::pid_t pid, bool monitor_signals)
{
    return HostThread();
}

Error
Host::ShellExpandArguments (ProcessLaunchInfo &launch_info)
{
    Error error;
    if (launch_info.GetFlags().Test(eLaunchFlagShellExpandArguments))
    {
        FileSpec expand_tool_spec;
        if (!HostInfo::GetLLDBPath(lldb::ePathTypeSupportExecutableDir, expand_tool_spec))
        {
            error.SetErrorString("could not find support executable directory for the lldb-argdumper tool");
            return error;
        }
        expand_tool_spec.AppendPathComponent("lldb-argdumper.exe");
        if (!expand_tool_spec.Exists())
        {
            error.SetErrorString("could not find the lldb-argdumper tool");
            return error;
        }
        
        std::string quoted_cmd_string;
        launch_info.GetArguments().GetQuotedCommandString(quoted_cmd_string);
        std::replace(quoted_cmd_string.begin(), quoted_cmd_string.end(), '\\', '/');
        StreamString expand_command;
        
        expand_command.Printf("\"%s\" %s",
                              expand_tool_spec.GetPath().c_str(),
                              quoted_cmd_string.c_str());
        
        int status;
        std::string output;
        RunShellCommand(expand_command.GetData(), launch_info.GetWorkingDirectory(), &status, nullptr, &output, 10);
        
        if (status != 0)
        {
            error.SetErrorStringWithFormat("lldb-argdumper exited with error %d", status);
            return error;
        }
        
        auto data_sp = StructuredData::ParseJSON(output);
        if (!data_sp)
        {
            error.SetErrorString("invalid JSON");
            return error;
        }
        
        auto dict_sp = data_sp->GetAsDictionary();
        if (!data_sp)
        {
            error.SetErrorString("invalid JSON");
            return error;
        }
        
        auto args_sp = dict_sp->GetObjectForDotSeparatedPath("arguments");
        if (!args_sp)
        {
            error.SetErrorString("invalid JSON");
            return error;
        }
        
        auto args_array_sp = args_sp->GetAsArray();
        if (!args_array_sp)
        {
            error.SetErrorString("invalid JSON");
            return error;
        }
        
        launch_info.GetArguments().Clear();
        
        for (size_t i = 0;
             i < args_array_sp->GetSize();
             i++)
        {
            auto item_sp = args_array_sp->GetItemAtIndex(i);
            if (!item_sp)
                continue;
            auto str_sp = item_sp->GetAsString();
            if (!str_sp)
                continue;
            
            launch_info.GetArguments().AppendArgument(str_sp->GetValue().c_str());
        }
    }
    
    return error;
}

size_t
Host::GetEnvironment(StringList &env)
{
    // The environment block on Windows is a contiguous buffer of NULL terminated strings,
    // where the end of the environment block is indicated by two consecutive NULLs.
    LPCH environment_block = ::GetEnvironmentStrings();
    env.Clear();
    while (*environment_block != '\0')
    {
        llvm::StringRef current_var(environment_block);
        if (current_var[0] != '=')
            env.AppendString(current_var);

        environment_block += current_var.size()+1;
    }
    return env.GetSize();
}
