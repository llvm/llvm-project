//===-- source/Host/freebsd/Host.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <stdio.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <sys/types.h>
#include <sys/user.h>
#include <sys/utsname.h>
#include <sys/sysctl.h>

#include <sys/ptrace.h>
#include <sys/exec.h>
#include <machine/elf.h>


// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Error.h"
#include "lldb/Host/Endian.h"
#include "lldb/Host/Host.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Target/Process.h"

#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "llvm/Support/Host.h"


extern "C" {
    extern char **environ;
}

using namespace lldb;
using namespace lldb_private;


class FreeBSDThread
{
public:
    FreeBSDThread(const char *thread_name)
    {
        Host::SetThreadName (LLDB_INVALID_PROCESS_ID, LLDB_INVALID_THREAD_ID, thread_name);
    }
    static void PThreadDestructor (void *v)
    {
        delete (FreeBSDThread*)v;
    }
};

static pthread_once_t g_thread_create_once = PTHREAD_ONCE_INIT;
static pthread_key_t g_thread_create_key = 0;

static void
InitThreadCreated()
{
    ::pthread_key_create (&g_thread_create_key, FreeBSDThread::PThreadDestructor);
}

void
Host::ThreadCreated (const char *thread_name)
{
    ::pthread_once (&g_thread_create_once, InitThreadCreated);
    if (g_thread_create_key)
    {
        ::pthread_setspecific (g_thread_create_key, new FreeBSDThread(thread_name));
    }
}

void
Host::Backtrace (Stream &strm, uint32_t max_frames)
{
    char backtrace_path[] = "/tmp/lldb-backtrace-tmp-XXXXXX";
    int backtrace_fd = ::mkstemp (backtrace_path);
    if (backtrace_fd != -1)
    {
        std::vector<void *> frame_buffer (max_frames, NULL);
        int count = ::backtrace (&frame_buffer[0], frame_buffer.size());
        ::backtrace_symbols_fd (&frame_buffer[0], count, backtrace_fd);

        const off_t buffer_size = ::lseek(backtrace_fd, 0, SEEK_CUR);

        if (::lseek(backtrace_fd, 0, SEEK_SET) == 0)
        {
            char *buffer = (char *)::malloc (buffer_size);
            if (buffer)
            {
                ssize_t bytes_read = ::read (backtrace_fd, buffer, buffer_size);
                if (bytes_read > 0)
                    strm.Write(buffer, bytes_read);
                ::free (buffer);
            }
        }
        ::close (backtrace_fd);
        ::unlink (backtrace_path);
    }
}

size_t
Host::GetEnvironment (StringList &env)
{
    char *v;
    char **var = environ;
    for (; var != NULL && *var != NULL; ++var) {
        v = strchr(*var, (int)'-');
        if (v == NULL)
            continue;
        env.AppendString(v);
    }
    return env.GetSize();
}

bool
Host::GetOSVersion(uint32_t &major, 
                   uint32_t &minor, 
                   uint32_t &update)
{
    struct utsname un;
    int status;

    if (uname(&un) < 0)
        return false;

    status = sscanf(un.release, "%u.%u-%u", &major, &minor, &update);
    return status == 3;
}

Error
Host::LaunchProcess (ProcessLaunchInfo &launch_info)
{
    Error error;
    assert(!"Not implemented yet!!!");
    return error;
}

bool
Host::GetOSBuildString (std::string &s)
{
    int mib[2] = { CTL_KERN, KERN_OSREV };
    char cstr[PATH_MAX];
    size_t cstr_len = sizeof(cstr);
    if (::sysctl (mib, 2, cstr, &cstr_len, NULL, 0) == 0)
    {
        s.assign (cstr, cstr_len);
        return true;
    }
    s.clear();
    return false;
}

bool
Host::GetOSKernelDescription (std::string &s)
{
    int mib[2] = { CTL_KERN, KERN_VERSION };
    char cstr[PATH_MAX];
    size_t cstr_len = sizeof(cstr);
    if (::sysctl (mib, 2, cstr, &cstr_len, NULL, 0) == 0)
    {
        s.assign (cstr, cstr_len);
        return true;
    }
    s.clear();
    return false;
}

static bool
GetFreeBSDProcessArgs (const ProcessInstanceInfoMatch *match_info_ptr,
                      ProcessInstanceInfo &process_info)
{
    if (process_info.ProcessIDIsValid()) {
        int mib[4] = { CTL_KERN, KERN_PROC, KERN_PROC_ARGS, (int)process_info.GetProcessID() };

        char arg_data[8192];
        size_t arg_data_size = sizeof(arg_data);
        if (::sysctl (mib, 4, arg_data, &arg_data_size , NULL, 0) == 0)
        {
            DataExtractor data (arg_data, arg_data_size, lldb::endian::InlHostByteOrder(), sizeof(void *));
            lldb::offset_t offset = 0;
            const char *cstr;

            cstr = data.GetCStr (&offset);
            if (cstr)
            {
                process_info.GetExecutableFile().SetFile(cstr, false);

                if (!(match_info_ptr == NULL || 
                    NameMatches (process_info.GetExecutableFile().GetFilename().GetCString(),
                                 match_info_ptr->GetNameMatchType(),
                                 match_info_ptr->GetProcessInfo().GetName())))
                    return false;

                Args &proc_args = process_info.GetArguments();
                while (1)
                {
                    const uint8_t *p = data.PeekData(offset, 1);
                    while ((p != NULL) && (*p == '\0') && offset < arg_data_size)
                    {
                        ++offset;
                        p = data.PeekData(offset, 1);
                    }
                    if (p == NULL || offset >= arg_data_size)
                        return true;

                    cstr = data.GetCStr(&offset);
                    if (cstr)
                        proc_args.AppendArgument(cstr);
                    else
                        return true;
                }
            }
        } 
    }
    return false;
}

static bool
GetFreeBSDProcessCPUType (ProcessInstanceInfo &process_info)
{
    if (process_info.ProcessIDIsValid()) {
        process_info.GetArchitecture() = Host::GetArchitecture (Host::eSystemDefaultArchitecture);
        return true;
    }
    process_info.GetArchitecture().Clear();
    return false;
}

static bool
GetFreeBSDProcessUserAndGroup(ProcessInstanceInfo &process_info)
{
    struct kinfo_proc proc_kinfo;
    size_t proc_kinfo_size;

    if (process_info.ProcessIDIsValid()) 
    {
        int mib[4] = { CTL_KERN, KERN_PROC, KERN_PROC_PID,
            (int)process_info.GetProcessID() };
        proc_kinfo_size = sizeof(struct kinfo_proc);

        if (::sysctl (mib, 4, &proc_kinfo, &proc_kinfo_size, NULL, 0) == 0)
        {
            if (proc_kinfo_size > 0)
            {
                process_info.SetParentProcessID (proc_kinfo.ki_ppid);
                process_info.SetUserID (proc_kinfo.ki_ruid);
                process_info.SetGroupID (proc_kinfo.ki_rgid);
                process_info.SetEffectiveUserID (proc_kinfo.ki_uid);
                if (proc_kinfo.ki_ngroups > 0)
                    process_info.SetEffectiveGroupID (proc_kinfo.ki_groups[0]);
                else
                    process_info.SetEffectiveGroupID (UINT32_MAX);
                return true;
            }
        }
    }
    process_info.SetParentProcessID (LLDB_INVALID_PROCESS_ID);
    process_info.SetUserID (UINT32_MAX);
    process_info.SetGroupID (UINT32_MAX);
    process_info.SetEffectiveUserID (UINT32_MAX);
    process_info.SetEffectiveGroupID (UINT32_MAX);
    return false;
}

bool
Host::GetProcessInfo (lldb::pid_t pid, ProcessInstanceInfo &process_info)
{
    process_info.SetProcessID(pid);
    if (GetFreeBSDProcessArgs(NULL, process_info)) {
        // should use libprocstat instead of going right into sysctl?
        GetFreeBSDProcessCPUType(process_info);
        GetFreeBSDProcessUserAndGroup(process_info);
        return true;
    }
    process_info.Clear();
    return false;
}

lldb::DataBufferSP
Host::GetAuxvData(lldb_private::Process *process)
{
   int mib[2] = { CTL_KERN, KERN_PS_STRINGS };
   void *ps_strings_addr, *auxv_addr;
   size_t ps_strings_size = sizeof(void *);
   Elf_Auxinfo aux_info[AT_COUNT];
   struct ps_strings ps_strings;
   struct ptrace_io_desc pid;
   DataBufferSP buf_sp;
   std::unique_ptr<DataBufferHeap> buf_ap(new DataBufferHeap(1024, 0));

   if (::sysctl(mib, 2, &ps_strings_addr, &ps_strings_size, NULL, 0) == 0) {
           pid.piod_op = PIOD_READ_D;
           pid.piod_addr = &ps_strings;
           pid.piod_offs = ps_strings_addr;
           pid.piod_len = sizeof(ps_strings);
           if (::ptrace(PT_IO, process->GetID(), (caddr_t)&pid, 0)) {
                   perror("failed to fetch ps_strings");
                   buf_ap.release();
                   goto done;
           }

           auxv_addr = ps_strings.ps_envstr + ps_strings.ps_nenvstr + 1;

           pid.piod_addr = aux_info;
           pid.piod_offs = auxv_addr;
           pid.piod_len = sizeof(aux_info);
           if (::ptrace(PT_IO, process->GetID(), (caddr_t)&pid, 0)) {
                   perror("failed to fetch aux_info");
                   buf_ap.release();
                   goto done;
           }
           memcpy(buf_ap->GetBytes(), aux_info, pid.piod_len);
           buf_sp.reset(buf_ap.release());
   } else {
           perror("sysctl failed on ps_strings");
   }

   done:
   return buf_sp;
}
