//===-- NativeProcessLinux.cpp -------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NativeProcessLinux.h"

// C Includes
#include <errno.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

// C++ Includes
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

// Other libraries and framework includes
#include "lldb/Core/EmulateInstruction.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/State.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Host/common/NativeBreakpoint.h"
#include "lldb/Host/common/NativeRegisterContext.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/ProcessLaunchInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/PseudoTerminal.h"
#include "lldb/Utility/StringExtractor.h"

#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#include "NativeThreadLinux.h"
#include "ProcFileReader.h"
#include "Procfs.h"

// System includes - They have to be included after framework includes because they define some
// macros which collide with variable names in other modules
#include <linux/unistd.h>
#include <sys/socket.h>

#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/user.h>
#include <sys/wait.h>

#include "lldb/Host/linux/Personality.h"
#include "lldb/Host/linux/Ptrace.h"
#include "lldb/Host/linux/Uio.h"

#define LLDB_PERSONALITY_GET_CURRENT_SETTINGS  0xffffffff

// Support hardware breakpoints in case it has not been defined
#ifndef TRAP_HWBKPT
  #define TRAP_HWBKPT 4
#endif

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_linux;
using namespace llvm;

// Private bits we only need internally.

static bool ProcessVmReadvSupported()
{
    static bool is_supported;
    static std::once_flag flag;

    std::call_once(flag, [] {
        Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

        uint32_t source = 0x47424742;
        uint32_t dest = 0;

        struct iovec local, remote;
        remote.iov_base = &source;
        local.iov_base = &dest;
        remote.iov_len = local.iov_len = sizeof source;

        // We shall try if cross-process-memory reads work by attempting to read a value from our own process.
        ssize_t res = process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
        is_supported = (res == sizeof(source) && source == dest);
        if (log)
        {
            if (is_supported)
                log->Printf("%s: Detected kernel support for process_vm_readv syscall. Fast memory reads enabled.",
                        __FUNCTION__);
            else
                log->Printf("%s: syscall process_vm_readv failed (error: %s). Fast memory reads disabled.",
                        __FUNCTION__, strerror(errno));
        }
    });

    return is_supported;
}

namespace
{
Error
ResolveProcessArchitecture(lldb::pid_t pid, ArchSpec &arch)
{
    // Grab process info for the running process.
    ProcessInstanceInfo process_info;
    if (!Host::GetProcessInfo(pid, process_info))
        return Error("failed to get process info");

    // Resolve the executable module.
    ModuleSpecList module_specs;
    if (!ObjectFile::GetModuleSpecifications(process_info.GetExecutableFile(), 0, 0, module_specs))
        return Error("failed to get module specifications");
    assert(module_specs.GetSize() == 1);

    arch = module_specs.GetModuleSpecRefAtIndex(0).GetArchitecture();
    if (arch.IsValid())
        return Error();
    else
        return Error("failed to retrieve a valid architecture from the exe module");
}

// Used to notify the parent about which part of the launch sequence failed.
enum LaunchCallSpecifier
{
    ePtraceFailed,
    eDupStdinFailed,
    eDupStdoutFailed,
    eDupStderrFailed,
    eChdirFailed,
    eExecFailed,
    eSetGidFailed,
    eSetSigMaskFailed,
    eLaunchCallMax = eSetSigMaskFailed
};

static uint8_t LLVM_ATTRIBUTE_NORETURN
ExitChildAbnormally(LaunchCallSpecifier spec)
{
    static_assert(eLaunchCallMax < 0x8, "Have more launch calls than we are able to represent");
    // This may truncate the topmost bits of the errno because the exit code is only 8 bits wide.
    // However, it should still give us a pretty good indication of what went wrong. (And the
    // most common errors have small numbers anyway).
    _exit(unsigned(spec) | (errno << 3));
}

// The second member is the errno (or its 5 lowermost bits anyway).
inline std::pair<LaunchCallSpecifier, uint8_t>
DecodeChildExitCode(int exit_code)
{
    return std::make_pair(LaunchCallSpecifier(exit_code & 0x7), exit_code >> 3);
}

void
MaybeLogLaunchInfo(const ProcessLaunchInfo &info)
{
    Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));
    if (!log)
        return;

    if (const FileAction *action = info.GetFileActionForFD(STDIN_FILENO))
        log->Printf("%s: setting STDIN to '%s'", __FUNCTION__, action->GetFileSpec().GetCString());
    else
        log->Printf("%s leaving STDIN as is", __FUNCTION__);

    if (const FileAction *action = info.GetFileActionForFD(STDOUT_FILENO))
        log->Printf("%s setting STDOUT to '%s'", __FUNCTION__, action->GetFileSpec().GetCString());
    else
        log->Printf("%s leaving STDOUT as is", __FUNCTION__);

    if (const FileAction *action = info.GetFileActionForFD(STDERR_FILENO))
        log->Printf("%s setting STDERR to '%s'", __FUNCTION__, action->GetFileSpec().GetCString());
    else
        log->Printf("%s leaving STDERR as is", __FUNCTION__);

    int i = 0;
    for (const char **args = info.GetArguments().GetConstArgumentVector(); *args; ++args, ++i)
        log->Printf("%s arg %d: \"%s\"", __FUNCTION__, i, *args ? *args : "nullptr");
}

void
DisplayBytes(StreamString &s, void *bytes, uint32_t count)
{
    uint8_t *ptr = (uint8_t *)bytes;
    const uint32_t loop_count = std::min<uint32_t>(DEBUG_PTRACE_MAXBYTES, count);
    for (uint32_t i = 0; i < loop_count; i++)
    {
        s.Printf("[%x]", *ptr);
        ptr++;
    }
}

    void
    PtraceDisplayBytes(int &req, void *data, size_t data_size)
    {
        StreamString buf;
        Log *verbose_log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (
                    POSIX_LOG_PTRACE | POSIX_LOG_VERBOSE));

        if (verbose_log)
        {
            switch(req)
            {
            case PTRACE_POKETEXT:
            {
                DisplayBytes(buf, &data, 8);
                verbose_log->Printf("PTRACE_POKETEXT %s", buf.GetData());
                break;
            }
            case PTRACE_POKEDATA:
            {
                DisplayBytes(buf, &data, 8);
                verbose_log->Printf("PTRACE_POKEDATA %s", buf.GetData());
                break;
            }
            case PTRACE_POKEUSER:
            {
                DisplayBytes(buf, &data, 8);
                verbose_log->Printf("PTRACE_POKEUSER %s", buf.GetData());
                break;
            }
            case PTRACE_SETREGS:
            {
                DisplayBytes(buf, data, data_size);
                verbose_log->Printf("PTRACE_SETREGS %s", buf.GetData());
                break;
            }
            case PTRACE_SETFPREGS:
            {
                DisplayBytes(buf, data, data_size);
                verbose_log->Printf("PTRACE_SETFPREGS %s", buf.GetData());
                break;
            }
            case PTRACE_SETSIGINFO:
            {
                DisplayBytes(buf, data, sizeof(siginfo_t));
                verbose_log->Printf("PTRACE_SETSIGINFO %s", buf.GetData());
                break;
            }
            case PTRACE_SETREGSET:
            {
                // Extract iov_base from data, which is a pointer to the struct IOVEC
                DisplayBytes(buf, *(void **)data, data_size);
                verbose_log->Printf("PTRACE_SETREGSET %s", buf.GetData());
                break;
            }
            default:
            {
            }
            }
        }
    }

    static constexpr unsigned k_ptrace_word_size = sizeof(void*);
    static_assert(sizeof(long) >= k_ptrace_word_size, "Size of long must be larger than ptrace word size");
} // end of anonymous namespace

// Simple helper function to ensure flags are enabled on the given file
// descriptor.
static Error
EnsureFDFlags(int fd, int flags)
{
    Error error;

    int status = fcntl(fd, F_GETFL);
    if (status == -1)
    {
        error.SetErrorToErrno();
        return error;
    }

    if (fcntl(fd, F_SETFL, status | flags) == -1)
    {
        error.SetErrorToErrno();
        return error;
    }

    return error;
}

// -----------------------------------------------------------------------------
// Public Static Methods
// -----------------------------------------------------------------------------

Error
NativeProcessProtocol::Launch (
    ProcessLaunchInfo &launch_info,
    NativeProcessProtocol::NativeDelegate &native_delegate,
    MainLoop &mainloop,
    NativeProcessProtocolSP &native_process_sp)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    Error error;

    // Verify the working directory is valid if one was specified.
    FileSpec working_dir{launch_info.GetWorkingDirectory()};
    if (working_dir &&
            (!working_dir.ResolvePath() ||
             working_dir.GetFileType() != FileSpec::eFileTypeDirectory))
    {
        error.SetErrorStringWithFormat ("No such file or directory: %s",
                working_dir.GetCString());
        return error;
    }

    // Create the NativeProcessLinux in launch mode.
    native_process_sp.reset (new NativeProcessLinux ());

    if (!native_process_sp->RegisterNativeDelegate (native_delegate))
    {
        native_process_sp.reset ();
        error.SetErrorStringWithFormat ("failed to register the native delegate");
        return error;
    }

    error = std::static_pointer_cast<NativeProcessLinux>(native_process_sp)->LaunchInferior(mainloop, launch_info);

    if (error.Fail ())
    {
        native_process_sp.reset ();
        if (log)
            log->Printf ("NativeProcessLinux::%s failed to launch process: %s", __FUNCTION__, error.AsCString ());
        return error;
    }

    launch_info.SetProcessID (native_process_sp->GetID ());

    return error;
}

Error
NativeProcessProtocol::Attach (
    lldb::pid_t pid,
    NativeProcessProtocol::NativeDelegate &native_delegate,
    MainLoop &mainloop,
    NativeProcessProtocolSP &native_process_sp)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    if (log && log->GetMask ().Test (POSIX_LOG_VERBOSE))
        log->Printf ("NativeProcessLinux::%s(pid = %" PRIi64 ")", __FUNCTION__, pid);

    // Retrieve the architecture for the running process.
    ArchSpec process_arch;
    Error error = ResolveProcessArchitecture(pid, process_arch);
    if (!error.Success ())
        return error;

    std::shared_ptr<NativeProcessLinux> native_process_linux_sp (new NativeProcessLinux ());

    if (!native_process_linux_sp->RegisterNativeDelegate (native_delegate))
    {
        error.SetErrorStringWithFormat ("failed to register the native delegate");
        return error;
    }

    native_process_linux_sp->AttachToInferior (mainloop, pid, error);
    if (!error.Success ())
        return error;

    native_process_sp = native_process_linux_sp;
    return error;
}

// -----------------------------------------------------------------------------
// Public Instance Methods
// -----------------------------------------------------------------------------

NativeProcessLinux::NativeProcessLinux () :
    NativeProcessProtocol (LLDB_INVALID_PROCESS_ID),
    m_arch (),
    m_supports_mem_region (eLazyBoolCalculate),
    m_mem_region_cache (),
    m_pending_notification_tid(LLDB_INVALID_THREAD_ID)
{
}

void
NativeProcessLinux::AttachToInferior (MainLoop &mainloop, lldb::pid_t pid, Error &error)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf ("NativeProcessLinux::%s (pid = %" PRIi64 ")", __FUNCTION__, pid);

    m_sigchld_handle = mainloop.RegisterSignal(SIGCHLD,
            [this] (MainLoopBase &) { SigchldHandler(); }, error);
    if (! m_sigchld_handle)
        return;

    error = ResolveProcessArchitecture(pid, m_arch);
    if (!error.Success())
        return;

    // Set the architecture to the exe architecture.
    if (log)
        log->Printf ("NativeProcessLinux::%s (pid = %" PRIi64 ") detected architecture %s", __FUNCTION__, pid, m_arch.GetArchitectureName ());

    m_pid = pid;
    SetState(eStateAttaching);

    Attach(pid, error);
}

void
NativeProcessLinux::ChildFunc(const ProcessLaunchInfo &info)
{
    // Start tracing this child that is about to exec.
    if (ptrace(PTRACE_TRACEME, 0, nullptr, nullptr) == -1)
        ExitChildAbnormally(ePtraceFailed);

    // Do not inherit setgid powers.
    if (setgid(getgid()) != 0)
        ExitChildAbnormally(eSetGidFailed);

    // Attempt to have our own process group.
    if (setpgid(0, 0) != 0)
    {
        // FIXME log that this failed. This is common.
        // Don't allow this to prevent an inferior exec.
    }

    // Dup file descriptors if needed.
    if (const FileAction *action = info.GetFileActionForFD(STDIN_FILENO))
        if (!DupDescriptor(action->GetFileSpec(), STDIN_FILENO, O_RDONLY))
            ExitChildAbnormally(eDupStdinFailed);

    if (const FileAction *action = info.GetFileActionForFD(STDOUT_FILENO))
        if (!DupDescriptor(action->GetFileSpec(), STDOUT_FILENO, O_WRONLY | O_CREAT | O_TRUNC))
            ExitChildAbnormally(eDupStdoutFailed);

    if (const FileAction *action = info.GetFileActionForFD(STDERR_FILENO))
        if (!DupDescriptor(action->GetFileSpec(), STDERR_FILENO, O_WRONLY | O_CREAT | O_TRUNC))
            ExitChildAbnormally(eDupStderrFailed);

    // Close everything besides stdin, stdout, and stderr that has no file
    // action to avoid leaking
    for (int fd = 3; fd < sysconf(_SC_OPEN_MAX); ++fd)
        if (!info.GetFileActionForFD(fd))
            close(fd);

    // Change working directory
    if (info.GetWorkingDirectory() && 0 != ::chdir(info.GetWorkingDirectory().GetCString()))
        ExitChildAbnormally(eChdirFailed);

    // Disable ASLR if requested.
    if (info.GetFlags().Test(lldb::eLaunchFlagDisableASLR))
    {
        const int old_personality = personality(LLDB_PERSONALITY_GET_CURRENT_SETTINGS);
        if (old_personality == -1)
        {
            // Can't retrieve Linux personality.  Cannot disable ASLR.
        }
        else
        {
            const int new_personality = personality(ADDR_NO_RANDOMIZE | old_personality);
            if (new_personality == -1)
            {
                // Disabling ASLR failed.
            }
            else
            {
                // Disabling ASLR succeeded.
            }
        }
    }

    // Clear the signal mask to prevent the child from being affected by
    // any masking done by the parent.
    sigset_t set;
    if (sigemptyset(&set) != 0 || pthread_sigmask(SIG_SETMASK, &set, nullptr) != 0)
        ExitChildAbnormally(eSetSigMaskFailed);

    const char **argv = info.GetArguments().GetConstArgumentVector();

    // Propagate the environment if one is not supplied.
    const char **envp = info.GetEnvironmentEntries().GetConstArgumentVector();
    if (envp == NULL || envp[0] == NULL)
        envp = const_cast<const char **>(environ);

    // Execute.  We should never return...
    execve(argv[0], const_cast<char *const *>(argv), const_cast<char *const *>(envp));

    if (errno == ETXTBSY)
    {
        // On android M and earlier we can get this error because the adb deamon can hold a write
        // handle on the executable even after it has finished uploading it. This state lasts
        // only a short time and happens only when there are many concurrent adb commands being
        // issued, such as when running the test suite. (The file remains open when someone does
        // an "adb shell" command in the fork() child before it has had a chance to exec.) Since
        // this state should clear up quickly, wait a while and then give it one more go.
        usleep(50000);
        execve(argv[0], const_cast<char *const *>(argv), const_cast<char *const *>(envp));
    }

    // ...unless exec fails.  In which case we definitely need to end the child here.
    ExitChildAbnormally(eExecFailed);
}

Error
NativeProcessLinux::LaunchInferior(MainLoop &mainloop, ProcessLaunchInfo &launch_info)
{
    Error error;
    m_sigchld_handle = mainloop.RegisterSignal(SIGCHLD, [this](MainLoopBase &) { SigchldHandler(); }, error);
    if (!m_sigchld_handle)
        return error;

    SetState(eStateLaunching);

    lldb_utility::PseudoTerminal terminal;
    const size_t err_len = 1024;
    char err_str[err_len];
    lldb::pid_t pid;

    MaybeLogLaunchInfo(launch_info);

    if ((pid = terminal.Fork(err_str, err_len)) == static_cast<lldb::pid_t> (-1))
    {
        error.SetErrorToGenericError();
        error.SetErrorStringWithFormat("Process fork failed: %s", err_str);
        return error;
    }

    // Child process.
    if (pid == 0)
    {
        // First, make sure we disable all logging. If we are logging to stdout, our logs can be
        // mistaken for inferior output.
        Log::DisableAllLogChannels(nullptr);

        // terminal has already dupped the tty descriptors to stdin/out/err.
        // This closes original fd from which they were copied (and avoids
        // leaking descriptors to the debugged process.
        terminal.CloseSlaveFileDescriptor();

        ChildFunc(launch_info);
    }

    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    // Wait for the child process to trap on its call to execve.
    ::pid_t wpid;
    int status;
    if ((wpid = waitpid(pid, &status, 0)) < 0)
    {
        error.SetErrorToErrno();
        if (log)
            log->Printf ("NativeProcessLinux::%s waitpid for inferior failed with %s",
                    __FUNCTION__, error.AsCString ());

        // Mark the inferior as invalid.
        // FIXME this could really use a new state - eStateLaunchFailure.  For now, using eStateInvalid.
        SetState (StateType::eStateInvalid);

        return error;
    }
    else if (WIFEXITED(status))
    {
        auto p = DecodeChildExitCode(WEXITSTATUS(status));
        Error child_error(p.second, eErrorTypePOSIX);
        const char *failure_reason;
        switch (p.first)
        {
            case ePtraceFailed:
                failure_reason = "Child ptrace failed";
                break;
            case eDupStdinFailed:
                failure_reason = "Child open stdin failed";
                break;
            case eDupStdoutFailed:
                failure_reason = "Child open stdout failed";
                break;
            case eDupStderrFailed:
                failure_reason = "Child open stderr failed";
                break;
            case eChdirFailed:
                failure_reason = "Child failed to set working directory";
                break;
            case eExecFailed:
                failure_reason = "Child exec failed";
                break;
            case eSetGidFailed:
                failure_reason = "Child setgid failed";
                break;
            case eSetSigMaskFailed:
                failure_reason = "Child failed to set signal mask";
                break;
        }
        error.SetErrorStringWithFormat("%s: %d - %s (error code truncated)", failure_reason, child_error.GetError(), child_error.AsCString());

        if (log)
        {
            log->Printf ("NativeProcessLinux::%s inferior exited with status %d before issuing a STOP",
                    __FUNCTION__,
                    WEXITSTATUS(status));
        }

        // Mark the inferior as invalid.
        // FIXME this could really use a new state - eStateLaunchFailure.  For now, using eStateInvalid.
        SetState (StateType::eStateInvalid);

        return error;
    }
    assert(WIFSTOPPED(status) && (wpid == static_cast< ::pid_t> (pid)) &&
           "Could not sync with inferior process.");

    if (log)
        log->Printf ("NativeProcessLinux::%s inferior started, now in stopped state", __FUNCTION__);

    error = SetDefaultPtraceOpts(pid);
    if (error.Fail())
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s inferior failed to set default ptrace options: %s",
                    __FUNCTION__, error.AsCString ());

        // Mark the inferior as invalid.
        // FIXME this could really use a new state - eStateLaunchFailure.  For now, using eStateInvalid.
        SetState (StateType::eStateInvalid);

        return error;
    }

    // Release the master terminal descriptor and pass it off to the
    // NativeProcessLinux instance.  Similarly stash the inferior pid.
    m_terminal_fd = terminal.ReleaseMasterFileDescriptor();
    m_pid = pid;
    launch_info.SetProcessID(pid);

    // Set the terminal fd to be in non blocking mode (it simplifies the
    // implementation of ProcessLinux::GetSTDOUT to have a non-blocking
    // descriptor to read from).
    error = EnsureFDFlags(m_terminal_fd, O_NONBLOCK);
    if (error.Fail())
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s inferior EnsureFDFlags failed for ensuring terminal O_NONBLOCK setting: %s",
                    __FUNCTION__, error.AsCString ());

        // Mark the inferior as invalid.
        // FIXME this could really use a new state - eStateLaunchFailure.  For now, using eStateInvalid.
        SetState (StateType::eStateInvalid);

        return error;
    }

    if (log)
        log->Printf ("NativeProcessLinux::%s() adding pid = %" PRIu64, __FUNCTION__, pid);

    ResolveProcessArchitecture(m_pid, m_arch);
    NativeThreadLinuxSP thread_sp = AddThread(pid);
    assert (thread_sp && "AddThread() returned a nullptr thread");
    thread_sp->SetStoppedBySignal(SIGSTOP);
    ThreadWasCreated(*thread_sp);

    // Let our process instance know the thread has stopped.
    SetCurrentThreadID (thread_sp->GetID ());
    SetState (StateType::eStateStopped);

    if (log)
    {
        if (error.Success ())
            log->Printf("NativeProcessLinux::%s inferior launching succeeded", __FUNCTION__);
        else
            log->Printf("NativeProcessLinux::%s inferior launching failed: %s", __FUNCTION__, error.AsCString());
    }
    return error;
}

::pid_t
NativeProcessLinux::Attach(lldb::pid_t pid, Error &error)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    // Use a map to keep track of the threads which we have attached/need to attach.
    Host::TidMap tids_to_attach;
    if (pid <= 1)
    {
        error.SetErrorToGenericError();
        error.SetErrorString("Attaching to process 1 is not allowed.");
        return -1;
    }

    while (Host::FindProcessThreads(pid, tids_to_attach))
    {
        for (Host::TidMap::iterator it = tids_to_attach.begin();
             it != tids_to_attach.end();)
        {
            if (it->second == false)
            {
                lldb::tid_t tid = it->first;

                // Attach to the requested process.
                // An attach will cause the thread to stop with a SIGSTOP.
                error = PtraceWrapper(PTRACE_ATTACH, tid);
                if (error.Fail())
                {
                    // No such thread. The thread may have exited.
                    // More error handling may be needed.
                    if (error.GetError() == ESRCH)
                    {
                        it = tids_to_attach.erase(it);
                        continue;
                    }
                    else
                        return -1;
                }

                int status;
                // Need to use __WALL otherwise we receive an error with errno=ECHLD
                // At this point we should have a thread stopped if waitpid succeeds.
                if ((status = waitpid(tid, NULL, __WALL)) < 0)
                {
                    // No such thread. The thread may have exited.
                    // More error handling may be needed.
                    if (errno == ESRCH)
                    {
                        it = tids_to_attach.erase(it);
                        continue;
                    }
                    else
                    {
                        error.SetErrorToErrno();
                        return -1;
                    }
                }

                error = SetDefaultPtraceOpts(tid);
                if (error.Fail())
                    return -1;

                if (log)
                    log->Printf ("NativeProcessLinux::%s() adding tid = %" PRIu64, __FUNCTION__, tid);

                it->second = true;

                // Create the thread, mark it as stopped.
                NativeThreadLinuxSP thread_sp (AddThread(static_cast<lldb::tid_t>(tid)));
                assert (thread_sp && "AddThread() returned a nullptr");

                // This will notify this is a new thread and tell the system it is stopped.
                thread_sp->SetStoppedBySignal(SIGSTOP);
                ThreadWasCreated(*thread_sp);
                SetCurrentThreadID (thread_sp->GetID ());
            }

            // move the loop forward
            ++it;
        }
    }

    if (tids_to_attach.size() > 0)
    {
        m_pid = pid;
        // Let our process instance know the thread has stopped.
        SetState (StateType::eStateStopped);
    }
    else
    {
        error.SetErrorToGenericError();
        error.SetErrorString("No such process.");
        return -1;
    }

    return pid;
}

Error
NativeProcessLinux::SetDefaultPtraceOpts(lldb::pid_t pid)
{
    long ptrace_opts = 0;

    // Have the child raise an event on exit.  This is used to keep the child in
    // limbo until it is destroyed.
    ptrace_opts |= PTRACE_O_TRACEEXIT;

    // Have the tracer trace threads which spawn in the inferior process.
    // TODO: if we want to support tracing the inferiors' child, add the
    // appropriate ptrace flags here (PTRACE_O_TRACEFORK, PTRACE_O_TRACEVFORK)
    ptrace_opts |= PTRACE_O_TRACECLONE;

    // Have the tracer notify us before execve returns
    // (needed to disable legacy SIGTRAP generation)
    ptrace_opts |= PTRACE_O_TRACEEXEC;

    return PtraceWrapper(PTRACE_SETOPTIONS, pid, nullptr, (void*)ptrace_opts);
}

static ExitType convert_pid_status_to_exit_type (int status)
{
    if (WIFEXITED (status))
        return ExitType::eExitTypeExit;
    else if (WIFSIGNALED (status))
        return ExitType::eExitTypeSignal;
    else if (WIFSTOPPED (status))
        return ExitType::eExitTypeStop;
    else
    {
        // We don't know what this is.
        return ExitType::eExitTypeInvalid;
    }
}

static int convert_pid_status_to_return_code (int status)
{
    if (WIFEXITED (status))
        return WEXITSTATUS (status);
    else if (WIFSIGNALED (status))
        return WTERMSIG (status);
    else if (WIFSTOPPED (status))
        return WSTOPSIG (status);
    else
    {
        // We don't know what this is.
        return ExitType::eExitTypeInvalid;
    }
}

// Handles all waitpid events from the inferior process.
void
NativeProcessLinux::MonitorCallback(lldb::pid_t pid,
                                    bool exited,
                                    int signal,
                                    int status)
{
    Log *log (GetLogIfAnyCategoriesSet (LIBLLDB_LOG_PROCESS));

    // Certain activities differ based on whether the pid is the tid of the main thread.
    const bool is_main_thread = (pid == GetID ());

    // Handle when the thread exits.
    if (exited)
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s() got exit signal(%d) , tid = %"  PRIu64 " (%s main thread)", __FUNCTION__, signal, pid, is_main_thread ? "is" : "is not");

        // This is a thread that exited.  Ensure we're not tracking it anymore.
        const bool thread_found = StopTrackingThread (pid);

        if (is_main_thread)
        {
            // We only set the exit status and notify the delegate if we haven't already set the process
            // state to an exited state.  We normally should have received a SIGTRAP | (PTRACE_EVENT_EXIT << 8)
            // for the main thread.
            const bool already_notified = (GetState() == StateType::eStateExited) || (GetState () == StateType::eStateCrashed);
            if (!already_notified)
            {
                if (log)
                    log->Printf ("NativeProcessLinux::%s() tid = %"  PRIu64 " handling main thread exit (%s), expected exit state already set but state was %s instead, setting exit state now", __FUNCTION__, pid, thread_found ? "stopped tracking thread metadata" : "thread metadata not found", StateAsCString (GetState ()));
                // The main thread exited.  We're done monitoring.  Report to delegate.
                SetExitStatus (convert_pid_status_to_exit_type (status), convert_pid_status_to_return_code (status), nullptr, true);

                // Notify delegate that our process has exited.
                SetState (StateType::eStateExited, true);
            }
            else
            {
                if (log)
                    log->Printf ("NativeProcessLinux::%s() tid = %"  PRIu64 " main thread now exited (%s)", __FUNCTION__, pid, thread_found ? "stopped tracking thread metadata" : "thread metadata not found");
            }
        }
        else
        {
            // Do we want to report to the delegate in this case?  I think not.  If this was an orderly
            // thread exit, we would already have received the SIGTRAP | (PTRACE_EVENT_EXIT << 8) signal,
            // and we would have done an all-stop then.
            if (log)
                log->Printf ("NativeProcessLinux::%s() tid = %"  PRIu64 " handling non-main thread exit (%s)", __FUNCTION__, pid, thread_found ? "stopped tracking thread metadata" : "thread metadata not found");
        }
        return;
    }

    siginfo_t info;
    const auto info_err = GetSignalInfo(pid, &info);
    auto thread_sp = GetThreadByID(pid);

    if (! thread_sp)
    {
        // Normally, the only situation when we cannot find the thread is if we have just
        // received a new thread notification. This is indicated by GetSignalInfo() returning
        // si_code == SI_USER and si_pid == 0
        if (log)
            log->Printf("NativeProcessLinux::%s received notification about an unknown tid %" PRIu64 ".", __FUNCTION__, pid);

        if (info_err.Fail())
        {
            if (log)
                log->Printf("NativeProcessLinux::%s (tid %" PRIu64 ") GetSignalInfo failed (%s). Ingoring this notification.", __FUNCTION__, pid, info_err.AsCString());
            return;
        }

        if (log && (info.si_code != SI_USER || info.si_pid != 0))
            log->Printf("NativeProcessLinux::%s (tid %" PRIu64 ") unexpected signal info (si_code: %d, si_pid: %d). Treating as a new thread notification anyway.", __FUNCTION__, pid, info.si_code, info.si_pid);

        auto thread_sp = AddThread(pid);
        // Resume the newly created thread.
        ResumeThread(*thread_sp, eStateRunning, LLDB_INVALID_SIGNAL_NUMBER);
        ThreadWasCreated(*thread_sp);
        return;
    }

    // Get details on the signal raised.
    if (info_err.Success())
    {
        // We have retrieved the signal info.  Dispatch appropriately.
        if (info.si_signo == SIGTRAP)
            MonitorSIGTRAP(info, *thread_sp);
        else
            MonitorSignal(info, *thread_sp, exited);
    }
    else
    {
        if (info_err.GetError() == EINVAL)
        {
            // This is a group stop reception for this tid.
            // We can reach here if we reinject SIGSTOP, SIGSTP, SIGTTIN or SIGTTOU into the
            // tracee, triggering the group-stop mechanism. Normally receiving these would stop
            // the process, pending a SIGCONT. Simulating this state in a debugger is hard and is
            // generally not needed (one use case is debugging background task being managed by a
            // shell). For general use, it is sufficient to stop the process in a signal-delivery
            // stop which happens before the group stop. This done by MonitorSignal and works
            // correctly for all signals.
            if (log)
                log->Printf("NativeProcessLinux::%s received a group stop for pid %" PRIu64 " tid %" PRIu64 ". Transparent handling of group stops not supported, resuming the thread.", __FUNCTION__, GetID (), pid);
            ResumeThread(*thread_sp, thread_sp->GetState(), LLDB_INVALID_SIGNAL_NUMBER);
        }
        else
        {
            // ptrace(GETSIGINFO) failed (but not due to group-stop).

            // A return value of ESRCH means the thread/process is no longer on the system,
            // so it was killed somehow outside of our control.  Either way, we can't do anything
            // with it anymore.

            // Stop tracking the metadata for the thread since it's entirely off the system now.
            const bool thread_found = StopTrackingThread (pid);

            if (log)
                log->Printf ("NativeProcessLinux::%s GetSignalInfo failed: %s, tid = %" PRIu64 ", signal = %d, status = %d (%s, %s, %s)",
                             __FUNCTION__, info_err.AsCString(), pid, signal, status, info_err.GetError() == ESRCH ? "thread/process killed" : "unknown reason", is_main_thread ? "is main thread" : "is not main thread", thread_found ? "thread metadata removed" : "thread metadata not found");

            if (is_main_thread)
            {
                // Notify the delegate - our process is not available but appears to have been killed outside
                // our control.  Is eStateExited the right exit state in this case?
                SetExitStatus (convert_pid_status_to_exit_type (status), convert_pid_status_to_return_code (status), nullptr, true);
                SetState (StateType::eStateExited, true);
            }
            else
            {
                // This thread was pulled out from underneath us.  Anything to do here? Do we want to do an all stop?
                if (log)
                    log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " tid %" PRIu64 " non-main thread exit occurred, didn't tell delegate anything since thread disappeared out from underneath us", __FUNCTION__, GetID (), pid);
            }
        }
    }
}

void
NativeProcessLinux::WaitForNewThread(::pid_t tid)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    NativeThreadLinuxSP new_thread_sp = GetThreadByID(tid);

    if (new_thread_sp)
    {
        // We are already tracking the thread - we got the event on the new thread (see
        // MonitorSignal) before this one. We are done.
        return;
    }

    // The thread is not tracked yet, let's wait for it to appear.
    int status = -1;
    ::pid_t wait_pid;
    do
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s() received thread creation event for tid %" PRIu32 ". tid not tracked yet, waiting for thread to appear...", __FUNCTION__, tid);
        wait_pid = waitpid(tid, &status, __WALL);
    }
    while (wait_pid == -1 && errno == EINTR);
    // Since we are waiting on a specific tid, this must be the creation event. But let's do
    // some checks just in case.
    if (wait_pid != tid) {
        if (log)
            log->Printf ("NativeProcessLinux::%s() waiting for tid %" PRIu32 " failed. Assuming the thread has disappeared in the meantime", __FUNCTION__, tid);
        // The only way I know of this could happen is if the whole process was
        // SIGKILLed in the mean time. In any case, we can't do anything about that now.
        return;
    }
    if (WIFEXITED(status))
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s() waiting for tid %" PRIu32 " returned an 'exited' event. Not tracking the thread.", __FUNCTION__, tid);
        // Also a very improbable event.
        return;
    }

    siginfo_t info;
    Error error = GetSignalInfo(tid, &info);
    if (error.Fail())
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s() GetSignalInfo for tid %" PRIu32 " failed. Assuming the thread has disappeared in the meantime.", __FUNCTION__, tid);
        return;
    }

    if (((info.si_pid != 0) || (info.si_code != SI_USER)) && log)
    {
        // We should be getting a thread creation signal here, but we received something
        // else. There isn't much we can do about it now, so we will just log that. Since the
        // thread is alive and we are receiving events from it, we shall pretend that it was
        // created properly.
        log->Printf ("NativeProcessLinux::%s() GetSignalInfo for tid %" PRIu32 " received unexpected signal with code %d from pid %d.", __FUNCTION__, tid, info.si_code, info.si_pid);
    }

    if (log)
        log->Printf ("NativeProcessLinux::%s() pid = %" PRIu64 ": tracking new thread tid %" PRIu32,
                 __FUNCTION__, GetID (), tid);

    new_thread_sp = AddThread(tid);
    ResumeThread(*new_thread_sp, eStateRunning, LLDB_INVALID_SIGNAL_NUMBER);
    ThreadWasCreated(*new_thread_sp);
}

void
NativeProcessLinux::MonitorSIGTRAP(const siginfo_t &info, NativeThreadLinux &thread)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    const bool is_main_thread = (thread.GetID() == GetID ());

    assert(info.si_signo == SIGTRAP && "Unexpected child signal!");

    switch (info.si_code)
    {
    // TODO: these two cases are required if we want to support tracing of the inferiors' children.  We'd need this to debug a monitor.
    // case (SIGTRAP | (PTRACE_EVENT_FORK << 8)):
    // case (SIGTRAP | (PTRACE_EVENT_VFORK << 8)):

    case (SIGTRAP | (PTRACE_EVENT_CLONE << 8)):
    {
        // This is the notification on the parent thread which informs us of new thread
        // creation.
        // We don't want to do anything with the parent thread so we just resume it. In case we
        // want to implement "break on thread creation" functionality, we would need to stop
        // here.

        unsigned long event_message = 0;
        if (GetEventMessage(thread.GetID(), &event_message).Fail())
        {
            if (log)
                log->Printf ("NativeProcessLinux::%s() pid %" PRIu64 " received thread creation event but GetEventMessage failed so we don't know the new tid", __FUNCTION__, thread.GetID());
        } else 
            WaitForNewThread(event_message);

        ResumeThread(thread, thread.GetState(), LLDB_INVALID_SIGNAL_NUMBER);
        break;
    }

    case (SIGTRAP | (PTRACE_EVENT_EXEC << 8)):
    {
        NativeThreadLinuxSP main_thread_sp;
        if (log)
            log->Printf ("NativeProcessLinux::%s() received exec event, code = %d", __FUNCTION__, info.si_code ^ SIGTRAP);

        // Exec clears any pending notifications.
        m_pending_notification_tid = LLDB_INVALID_THREAD_ID;

        // Remove all but the main thread here.  Linux fork creates a new process which only copies the main thread.
        if (log)
            log->Printf ("NativeProcessLinux::%s exec received, stop tracking all but main thread", __FUNCTION__);

        for (auto thread_sp : m_threads)
        {
            const bool is_main_thread = thread_sp && thread_sp->GetID () == GetID ();
            if (is_main_thread)
            {
                main_thread_sp = std::static_pointer_cast<NativeThreadLinux>(thread_sp);
                if (log)
                    log->Printf ("NativeProcessLinux::%s found main thread with tid %" PRIu64 ", keeping", __FUNCTION__, main_thread_sp->GetID ());
            }
            else
            {
                if (log)
                    log->Printf ("NativeProcessLinux::%s discarding non-main-thread tid %" PRIu64 " due to exec", __FUNCTION__, thread_sp->GetID ());
            }
        }

        m_threads.clear ();

        if (main_thread_sp)
        {
            m_threads.push_back (main_thread_sp);
            SetCurrentThreadID (main_thread_sp->GetID ());
            main_thread_sp->SetStoppedByExec();
        }
        else
        {
            SetCurrentThreadID (LLDB_INVALID_THREAD_ID);
            if (log)
                log->Printf ("NativeProcessLinux::%s pid %" PRIu64 "no main thread found, discarded all threads, we're in a no-thread state!", __FUNCTION__, GetID ());
        }

        // Tell coordinator about about the "new" (since exec) stopped main thread.
        ThreadWasCreated(*main_thread_sp);

        // Let our delegate know we have just exec'd.
        NotifyDidExec ();

        // If we have a main thread, indicate we are stopped.
        assert (main_thread_sp && "exec called during ptraced process but no main thread metadata tracked");

        // Let the process know we're stopped.
        StopRunningThreads(main_thread_sp->GetID());

        break;
    }

    case (SIGTRAP | (PTRACE_EVENT_EXIT << 8)):
    {
        // The inferior process or one of its threads is about to exit.
        // We don't want to do anything with the thread so we just resume it. In case we
        // want to implement "break on thread exit" functionality, we would need to stop
        // here.

        unsigned long data = 0;
        if (GetEventMessage(thread.GetID(), &data).Fail())
            data = -1;

        if (log)
        {
            log->Printf ("NativeProcessLinux::%s() received PTRACE_EVENT_EXIT, data = %lx (WIFEXITED=%s,WIFSIGNALED=%s), pid = %" PRIu64 " (%s)",
                         __FUNCTION__,
                         data, WIFEXITED (data) ? "true" : "false", WIFSIGNALED (data) ? "true" : "false",
                         thread.GetID(),
                    is_main_thread ? "is main thread" : "not main thread");
        }

        if (is_main_thread)
        {
            SetExitStatus (convert_pid_status_to_exit_type (data), convert_pid_status_to_return_code (data), nullptr, true);
        }

        StateType state = thread.GetState();
        if (! StateIsRunningState(state))
        {
            // Due to a kernel bug, we may sometimes get this stop after the inferior gets a
            // SIGKILL. This confuses our state tracking logic in ResumeThread(), since normally,
            // we should not be receiving any ptrace events while the inferior is stopped. This
            // makes sure that the inferior is resumed and exits normally.
            state = eStateRunning;
        }
        ResumeThread(thread, state, LLDB_INVALID_SIGNAL_NUMBER);

        break;
    }

    case 0:
    case TRAP_TRACE:  // We receive this on single stepping.
    case TRAP_HWBKPT: // We receive this on watchpoint hit
    {
        // If a watchpoint was hit, report it
        uint32_t wp_index;
        Error error = thread.GetRegisterContext()->GetWatchpointHitIndex(wp_index, (uintptr_t)info.si_addr);
        if (error.Fail() && log)
            log->Printf("NativeProcessLinux::%s() "
                        "received error while checking for watchpoint hits, "
                        "pid = %" PRIu64 " error = %s",
                        __FUNCTION__, thread.GetID(), error.AsCString());
        if (wp_index != LLDB_INVALID_INDEX32)
        {
            MonitorWatchpoint(thread, wp_index);
            break;
        }

        // Otherwise, report step over
        MonitorTrace(thread);
        break;
    }

    case SI_KERNEL:
#if defined __mips__
        // For mips there is no special signal for watchpoint
        // So we check for watchpoint in kernel trap
    {
        // If a watchpoint was hit, report it
        uint32_t wp_index;
        Error error = thread.GetRegisterContext()->GetWatchpointHitIndex(wp_index, LLDB_INVALID_ADDRESS);
        if (error.Fail() && log)
            log->Printf("NativeProcessLinux::%s() "
                        "received error while checking for watchpoint hits, "
                        "pid = %" PRIu64 " error = %s",
                        __FUNCTION__, thread.GetID(), error.AsCString());
        if (wp_index != LLDB_INVALID_INDEX32)
        {
            MonitorWatchpoint(thread, wp_index);
            break;
        }
    }
        // NO BREAK
#endif
    case TRAP_BRKPT:
        MonitorBreakpoint(thread);
        break;

    case SIGTRAP:
    case (SIGTRAP | 0x80):
        if (log)
            log->Printf ("NativeProcessLinux::%s() received unknown SIGTRAP system call stop event, pid %" PRIu64 "tid %" PRIu64 ", resuming", __FUNCTION__, GetID (), thread.GetID());

        // Ignore these signals until we know more about them.
        ResumeThread(thread, thread.GetState(), LLDB_INVALID_SIGNAL_NUMBER);
        break;

    default:
        assert(false && "Unexpected SIGTRAP code!");
        if (log)
            log->Printf ("NativeProcessLinux::%s() pid %" PRIu64 "tid %" PRIu64 " received unhandled SIGTRAP code: 0x%d",
                    __FUNCTION__, GetID(), thread.GetID(), info.si_code);
        break;
        
    }
}

void
NativeProcessLinux::MonitorTrace(NativeThreadLinux &thread)
{
    Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf("NativeProcessLinux::%s() received trace event, pid = %" PRIu64 " (single stepping)",
                __FUNCTION__, thread.GetID());

    // This thread is currently stopped.
    thread.SetStoppedByTrace();

    StopRunningThreads(thread.GetID());
}

void
NativeProcessLinux::MonitorBreakpoint(NativeThreadLinux &thread)
{
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_BREAKPOINTS));
    if (log)
        log->Printf("NativeProcessLinux::%s() received breakpoint event, pid = %" PRIu64,
                __FUNCTION__, thread.GetID());

    // Mark the thread as stopped at breakpoint.
    thread.SetStoppedByBreakpoint();
    Error error = FixupBreakpointPCAsNeeded(thread);
    if (error.Fail())
        if (log)
            log->Printf("NativeProcessLinux::%s() pid = %" PRIu64 " fixup: %s",
                    __FUNCTION__, thread.GetID(), error.AsCString());

    if (m_threads_stepping_with_breakpoint.find(thread.GetID()) != m_threads_stepping_with_breakpoint.end())
        thread.SetStoppedByTrace();

    StopRunningThreads(thread.GetID());
}

void
NativeProcessLinux::MonitorWatchpoint(NativeThreadLinux &thread, uint32_t wp_index)
{
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_WATCHPOINTS));
    if (log)
        log->Printf("NativeProcessLinux::%s() received watchpoint event, "
                    "pid = %" PRIu64 ", wp_index = %" PRIu32,
                    __FUNCTION__, thread.GetID(), wp_index);

    // Mark the thread as stopped at watchpoint.
    // The address is at (lldb::addr_t)info->si_addr if we need it.
    thread.SetStoppedByWatchpoint(wp_index);

    // We need to tell all other running threads before we notify the delegate about this stop.
    StopRunningThreads(thread.GetID());
}

void
NativeProcessLinux::MonitorSignal(const siginfo_t &info, NativeThreadLinux &thread, bool exited)
{
    const int signo = info.si_signo;
    const bool is_from_llgs = info.si_pid == getpid ();

    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    // POSIX says that process behaviour is undefined after it ignores a SIGFPE,
    // SIGILL, SIGSEGV, or SIGBUS *unless* that signal was generated by a
    // kill(2) or raise(3).  Similarly for tgkill(2) on Linux.
    //
    // IOW, user generated signals never generate what we consider to be a
    // "crash".
    //
    // Similarly, ACK signals generated by this monitor.

    // Handle the signal.
    if (info.si_code == SI_TKILL || info.si_code == SI_USER)
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s() received signal %s (%d) with code %s, (siginfo pid = %d (%s), waitpid pid = %" PRIu64 ")",
                            __FUNCTION__,
                            Host::GetSignalAsCString(signo),
                            signo,
                            (info.si_code == SI_TKILL ? "SI_TKILL" : "SI_USER"),
                            info.si_pid,
                            is_from_llgs ? "from llgs" : "not from llgs",
                            thread.GetID());
    }

    // Check for thread stop notification.
    if (is_from_llgs && (info.si_code == SI_TKILL) && (signo == SIGSTOP))
    {
        // This is a tgkill()-based stop.
        if (log)
            log->Printf ("NativeProcessLinux::%s() pid %" PRIu64 " tid %" PRIu64 ", thread stopped",
                         __FUNCTION__,
                         GetID (),
                         thread.GetID());

        // Check that we're not already marked with a stop reason.
        // Note this thread really shouldn't already be marked as stopped - if we were, that would imply that
        // the kernel signaled us with the thread stopping which we handled and marked as stopped,
        // and that, without an intervening resume, we received another stop.  It is more likely
        // that we are missing the marking of a run state somewhere if we find that the thread was
        // marked as stopped.
        const StateType thread_state = thread.GetState();
        if (!StateIsStoppedState (thread_state, false))
        {
            // An inferior thread has stopped because of a SIGSTOP we have sent it.
            // Generally, these are not important stops and we don't want to report them as
            // they are just used to stop other threads when one thread (the one with the
            // *real* stop reason) hits a breakpoint (watchpoint, etc...). However, in the
            // case of an asynchronous Interrupt(), this *is* the real stop reason, so we
            // leave the signal intact if this is the thread that was chosen as the
            // triggering thread.
            if (m_pending_notification_tid != LLDB_INVALID_THREAD_ID)
            {
                if (m_pending_notification_tid == thread.GetID())
                    thread.SetStoppedBySignal(SIGSTOP, &info);
                else
                    thread.SetStoppedWithNoReason();

                SetCurrentThreadID (thread.GetID ());
                SignalIfAllThreadsStopped();
            }
            else
            {
                // We can end up here if stop was initiated by LLGS but by this time a
                // thread stop has occurred - maybe initiated by another event.
                Error error = ResumeThread(thread, thread.GetState(), 0);
                if (error.Fail() && log)
                {
                    log->Printf("NativeProcessLinux::%s failed to resume thread tid  %" PRIu64 ": %s",
                            __FUNCTION__, thread.GetID(), error.AsCString());
                }
            }
        }
        else
        {
            if (log)
            {
                // Retrieve the signal name if the thread was stopped by a signal.
                int stop_signo = 0;
                const bool stopped_by_signal = thread.IsStopped(&stop_signo);
                const char *signal_name = stopped_by_signal ? Host::GetSignalAsCString(stop_signo) : "<not stopped by signal>";
                if (!signal_name)
                    signal_name = "<no-signal-name>";

                log->Printf ("NativeProcessLinux::%s() pid %" PRIu64 " tid %" PRIu64 ", thread was already marked as a stopped state (state=%s, signal=%d (%s)), leaving stop signal as is",
                             __FUNCTION__,
                             GetID (),
                             thread.GetID(),
                             StateAsCString (thread_state),
                             stop_signo,
                             signal_name);
            }
            SignalIfAllThreadsStopped();
        }

        // Done handling.
        return;
    }

    if (log)
        log->Printf ("NativeProcessLinux::%s() received signal %s", __FUNCTION__, Host::GetSignalAsCString(signo));

    // This thread is stopped.
    thread.SetStoppedBySignal(signo, &info);

    // Send a stop to the debugger after we get all other threads to stop.
    StopRunningThreads(thread.GetID());
}

namespace {

struct EmulatorBaton
{
    NativeProcessLinux* m_process;
    NativeRegisterContext* m_reg_context;

    // eRegisterKindDWARF -> RegsiterValue
    std::unordered_map<uint32_t, RegisterValue> m_register_values;

    EmulatorBaton(NativeProcessLinux* process, NativeRegisterContext* reg_context) :
            m_process(process), m_reg_context(reg_context) {}
};

} // anonymous namespace

static size_t
ReadMemoryCallback (EmulateInstruction *instruction,
                    void *baton,
                    const EmulateInstruction::Context &context, 
                    lldb::addr_t addr, 
                    void *dst,
                    size_t length)
{
    EmulatorBaton* emulator_baton = static_cast<EmulatorBaton*>(baton);

    size_t bytes_read;
    emulator_baton->m_process->ReadMemory(addr, dst, length, bytes_read);
    return bytes_read;
}

static bool
ReadRegisterCallback (EmulateInstruction *instruction,
                      void *baton,
                      const RegisterInfo *reg_info,
                      RegisterValue &reg_value)
{
    EmulatorBaton* emulator_baton = static_cast<EmulatorBaton*>(baton);

    auto it = emulator_baton->m_register_values.find(reg_info->kinds[eRegisterKindDWARF]);
    if (it != emulator_baton->m_register_values.end())
    {
        reg_value = it->second;
        return true;
    }

    // The emulator only fill in the dwarf regsiter numbers (and in some case
    // the generic register numbers). Get the full register info from the
    // register context based on the dwarf register numbers.
    const RegisterInfo* full_reg_info = emulator_baton->m_reg_context->GetRegisterInfo(
            eRegisterKindDWARF, reg_info->kinds[eRegisterKindDWARF]);

    Error error = emulator_baton->m_reg_context->ReadRegister(full_reg_info, reg_value);
    if (error.Success())
        return true;

    return false;
}

static bool
WriteRegisterCallback (EmulateInstruction *instruction,
                       void *baton,
                       const EmulateInstruction::Context &context,
                       const RegisterInfo *reg_info,
                       const RegisterValue &reg_value)
{
    EmulatorBaton* emulator_baton = static_cast<EmulatorBaton*>(baton);
    emulator_baton->m_register_values[reg_info->kinds[eRegisterKindDWARF]] = reg_value;
    return true;
}

static size_t
WriteMemoryCallback (EmulateInstruction *instruction,
                     void *baton,
                     const EmulateInstruction::Context &context, 
                     lldb::addr_t addr, 
                     const void *dst,
                     size_t length)
{
    return length;
}

static lldb::addr_t
ReadFlags (NativeRegisterContext* regsiter_context)
{
    const RegisterInfo* flags_info = regsiter_context->GetRegisterInfo(
            eRegisterKindGeneric, LLDB_REGNUM_GENERIC_FLAGS);
    return regsiter_context->ReadRegisterAsUnsigned(flags_info, LLDB_INVALID_ADDRESS);
}

Error
NativeProcessLinux::SetupSoftwareSingleStepping(NativeThreadLinux &thread)
{
    Error error;
    NativeRegisterContextSP register_context_sp = thread.GetRegisterContext();

    std::unique_ptr<EmulateInstruction> emulator_ap(
        EmulateInstruction::FindPlugin(m_arch, eInstructionTypePCModifying, nullptr));

    if (emulator_ap == nullptr)
        return Error("Instruction emulator not found!");

    EmulatorBaton baton(this, register_context_sp.get());
    emulator_ap->SetBaton(&baton);
    emulator_ap->SetReadMemCallback(&ReadMemoryCallback);
    emulator_ap->SetReadRegCallback(&ReadRegisterCallback);
    emulator_ap->SetWriteMemCallback(&WriteMemoryCallback);
    emulator_ap->SetWriteRegCallback(&WriteRegisterCallback);

    if (!emulator_ap->ReadInstruction())
        return Error("Read instruction failed!");

    bool emulation_result = emulator_ap->EvaluateInstruction(eEmulateInstructionOptionAutoAdvancePC);

    const RegisterInfo* reg_info_pc = register_context_sp->GetRegisterInfo(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
    const RegisterInfo* reg_info_flags = register_context_sp->GetRegisterInfo(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_FLAGS);

    auto pc_it = baton.m_register_values.find(reg_info_pc->kinds[eRegisterKindDWARF]);
    auto flags_it = baton.m_register_values.find(reg_info_flags->kinds[eRegisterKindDWARF]);

    lldb::addr_t next_pc;
    lldb::addr_t next_flags;
    if (emulation_result)
    {
        assert(pc_it != baton.m_register_values.end() && "Emulation was successfull but PC wasn't updated");
        next_pc = pc_it->second.GetAsUInt64();

        if (flags_it != baton.m_register_values.end())
            next_flags = flags_it->second.GetAsUInt64();
        else
            next_flags = ReadFlags (register_context_sp.get());
    }
    else if (pc_it == baton.m_register_values.end())
    {
        // Emulate instruction failed and it haven't changed PC. Advance PC
        // with the size of the current opcode because the emulation of all
        // PC modifying instruction should be successful. The failure most
        // likely caused by a not supported instruction which don't modify PC.
        next_pc = register_context_sp->GetPC() + emulator_ap->GetOpcode().GetByteSize();
        next_flags = ReadFlags (register_context_sp.get());
    }
    else
    {
        // The instruction emulation failed after it modified the PC. It is an
        // unknown error where we can't continue because the next instruction is
        // modifying the PC but we don't  know how.
        return Error ("Instruction emulation failed unexpectedly.");
    }

    if (m_arch.GetMachine() == llvm::Triple::arm)
    {
        if (next_flags & 0x20)
        {
            // Thumb mode
            error = SetSoftwareBreakpoint(next_pc, 2);
        }
        else
        {
            // Arm mode
            error = SetSoftwareBreakpoint(next_pc, 4);
        }
    }
    else if (m_arch.GetMachine() == llvm::Triple::mips64
            || m_arch.GetMachine() == llvm::Triple::mips64el
            || m_arch.GetMachine() == llvm::Triple::mips
            || m_arch.GetMachine() == llvm::Triple::mipsel)
        error = SetSoftwareBreakpoint(next_pc, 4);
    else
    {
        // No size hint is given for the next breakpoint
        error = SetSoftwareBreakpoint(next_pc, 0);
    }

    if (error.Fail())
        return error;

    m_threads_stepping_with_breakpoint.insert({thread.GetID(), next_pc});

    return Error();
}

bool
NativeProcessLinux::SupportHardwareSingleStepping() const
{
    if (m_arch.GetMachine() == llvm::Triple::arm
        || m_arch.GetMachine() == llvm::Triple::mips64 || m_arch.GetMachine() == llvm::Triple::mips64el
        || m_arch.GetMachine() == llvm::Triple::mips || m_arch.GetMachine() == llvm::Triple::mipsel)
        return false;
    return true;
}

Error
NativeProcessLinux::Resume (const ResumeActionList &resume_actions)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_THREAD));
    if (log)
        log->Printf ("NativeProcessLinux::%s called: pid %" PRIu64, __FUNCTION__, GetID ());

    bool software_single_step = !SupportHardwareSingleStepping();

    if (software_single_step)
    {
        for (auto thread_sp : m_threads)
        {
            assert (thread_sp && "thread list should not contain NULL threads");

            const ResumeAction *const action = resume_actions.GetActionForThread (thread_sp->GetID (), true);
            if (action == nullptr)
                continue;

            if (action->state == eStateStepping)
            {
                Error error = SetupSoftwareSingleStepping(static_cast<NativeThreadLinux &>(*thread_sp));
                if (error.Fail())
                    return error;
            }
        }
    }

    for (auto thread_sp : m_threads)
    {
        assert (thread_sp && "thread list should not contain NULL threads");

        const ResumeAction *const action = resume_actions.GetActionForThread (thread_sp->GetID (), true);

        if (action == nullptr)
        {
            if (log)
                log->Printf ("NativeProcessLinux::%s no action specified for pid %" PRIu64 " tid %" PRIu64,
                    __FUNCTION__, GetID (), thread_sp->GetID ());
            continue;
        }

        if (log)
        {
            log->Printf ("NativeProcessLinux::%s processing resume action state %s for pid %" PRIu64 " tid %" PRIu64, 
                    __FUNCTION__, StateAsCString (action->state), GetID (), thread_sp->GetID ());
        }

        switch (action->state)
        {
        case eStateRunning:
        case eStateStepping:
        {
            // Run the thread, possibly feeding it the signal.
            const int signo = action->signal;
            ResumeThread(static_cast<NativeThreadLinux &>(*thread_sp), action->state, signo);
            break;
        }

        case eStateSuspended:
        case eStateStopped:
            lldbassert(0 && "Unexpected state");

        default:
            return Error ("NativeProcessLinux::%s (): unexpected state %s specified for pid %" PRIu64 ", tid %" PRIu64,
                    __FUNCTION__, StateAsCString (action->state), GetID (), thread_sp->GetID ());
        }
    }

    return Error();
}

Error
NativeProcessLinux::Halt ()
{
    Error error;

    if (kill (GetID (), SIGSTOP) != 0)
        error.SetErrorToErrno ();

    return error;
}

Error
NativeProcessLinux::Detach ()
{
    Error error;

    // Stop monitoring the inferior.
    m_sigchld_handle.reset();

    // Tell ptrace to detach from the process.
    if (GetID () == LLDB_INVALID_PROCESS_ID)
        return error;

    for (auto thread_sp : m_threads)
    {
        Error e = Detach(thread_sp->GetID());
        if (e.Fail())
            error = e; // Save the error, but still attempt to detach from other threads.
    }

    return error;
}

Error
NativeProcessLinux::Signal (int signo)
{
    Error error;

    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf ("NativeProcessLinux::%s: sending signal %d (%s) to pid %" PRIu64,
                __FUNCTION__, signo, Host::GetSignalAsCString(signo), GetID());

    if (kill(GetID(), signo))
        error.SetErrorToErrno();

    return error;
}

Error
NativeProcessLinux::Interrupt ()
{
    // Pick a running thread (or if none, a not-dead stopped thread) as
    // the chosen thread that will be the stop-reason thread.
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    NativeThreadProtocolSP running_thread_sp;
    NativeThreadProtocolSP stopped_thread_sp;
        
    if (log)
        log->Printf ("NativeProcessLinux::%s selecting running thread for interrupt target", __FUNCTION__);

    for (auto thread_sp : m_threads)
    {
        // The thread shouldn't be null but lets just cover that here.
        if (!thread_sp)
            continue;

        // If we have a running or stepping thread, we'll call that the
        // target of the interrupt.
        const auto thread_state = thread_sp->GetState ();
        if (thread_state == eStateRunning ||
            thread_state == eStateStepping)
        {
            running_thread_sp = thread_sp;
            break;
        }
        else if (!stopped_thread_sp && StateIsStoppedState (thread_state, true))
        {
            // Remember the first non-dead stopped thread.  We'll use that as a backup if there are no running threads.
            stopped_thread_sp = thread_sp;
        }
    }

    if (!running_thread_sp && !stopped_thread_sp)
    {
        Error error("found no running/stepping or live stopped threads as target for interrupt");
        if (log)
            log->Printf ("NativeProcessLinux::%s skipping due to error: %s", __FUNCTION__, error.AsCString ());

        return error;
    }

    NativeThreadProtocolSP deferred_signal_thread_sp = running_thread_sp ? running_thread_sp : stopped_thread_sp;

    if (log)
        log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " %s tid %" PRIu64 " chosen for interrupt target",
                     __FUNCTION__,
                     GetID (),
                     running_thread_sp ? "running" : "stopped",
                     deferred_signal_thread_sp->GetID ());

    StopRunningThreads(deferred_signal_thread_sp->GetID());

    return Error();
}

Error
NativeProcessLinux::Kill ()
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf ("NativeProcessLinux::%s called for PID %" PRIu64, __FUNCTION__, GetID ());

    Error error;

    switch (m_state)
    {
        case StateType::eStateInvalid:
        case StateType::eStateExited:
        case StateType::eStateCrashed:
        case StateType::eStateDetached:
        case StateType::eStateUnloaded:
            // Nothing to do - the process is already dead.
            if (log)
                log->Printf ("NativeProcessLinux::%s ignored for PID %" PRIu64 " due to current state: %s", __FUNCTION__, GetID (), StateAsCString (m_state));
            return error;

        case StateType::eStateConnected:
        case StateType::eStateAttaching:
        case StateType::eStateLaunching:
        case StateType::eStateStopped:
        case StateType::eStateRunning:
        case StateType::eStateStepping:
        case StateType::eStateSuspended:
            // We can try to kill a process in these states.
            break;
    }

    if (kill (GetID (), SIGKILL) != 0)
    {
        error.SetErrorToErrno ();
        return error;
    }

    return error;
}

static Error
ParseMemoryRegionInfoFromProcMapsLine (const std::string &maps_line, MemoryRegionInfo &memory_region_info)
{
    memory_region_info.Clear();

    StringExtractor line_extractor (maps_line.c_str ());

    // Format: {address_start_hex}-{address_end_hex} perms offset  dev   inode   pathname
    // perms: rwxp   (letter is present if set, '-' if not, final character is p=private, s=shared).

    // Parse out the starting address
    lldb::addr_t start_address = line_extractor.GetHexMaxU64 (false, 0);

    // Parse out hyphen separating start and end address from range.
    if (!line_extractor.GetBytesLeft () || (line_extractor.GetChar () != '-'))
        return Error ("malformed /proc/{pid}/maps entry, missing dash between address range");

    // Parse out the ending address
    lldb::addr_t end_address = line_extractor.GetHexMaxU64 (false, start_address);

    // Parse out the space after the address.
    if (!line_extractor.GetBytesLeft () || (line_extractor.GetChar () != ' '))
        return Error ("malformed /proc/{pid}/maps entry, missing space after range");

    // Save the range.
    memory_region_info.GetRange ().SetRangeBase (start_address);
    memory_region_info.GetRange ().SetRangeEnd (end_address);

    // Any memory region in /proc/{pid}/maps is by definition mapped into the process.
    memory_region_info.SetMapped(MemoryRegionInfo::OptionalBool::eYes);

    // Parse out each permission entry.
    if (line_extractor.GetBytesLeft () < 4)
        return Error ("malformed /proc/{pid}/maps entry, missing some portion of permissions");

    // Handle read permission.
    const char read_perm_char = line_extractor.GetChar ();
    if (read_perm_char == 'r')
        memory_region_info.SetReadable (MemoryRegionInfo::OptionalBool::eYes);
    else if (read_perm_char == '-')
        memory_region_info.SetReadable (MemoryRegionInfo::OptionalBool::eNo);
    else
        return Error ("unexpected /proc/{pid}/maps read permission char");

    // Handle write permission.
    const char write_perm_char = line_extractor.GetChar ();
    if (write_perm_char == 'w')
        memory_region_info.SetWritable (MemoryRegionInfo::OptionalBool::eYes);
    else if (write_perm_char == '-')
        memory_region_info.SetWritable (MemoryRegionInfo::OptionalBool::eNo);
    else
        return Error ("unexpected /proc/{pid}/maps write permission char");

    // Handle execute permission.
    const char exec_perm_char = line_extractor.GetChar ();
    if (exec_perm_char == 'x')
        memory_region_info.SetExecutable (MemoryRegionInfo::OptionalBool::eYes);
    else if (exec_perm_char == '-')
        memory_region_info.SetExecutable (MemoryRegionInfo::OptionalBool::eNo);
    else
        return Error ("unexpected /proc/{pid}/maps exec permission char");

    return Error ();
}

Error
NativeProcessLinux::GetMemoryRegionInfo (lldb::addr_t load_addr, MemoryRegionInfo &range_info)
{
    // FIXME review that the final memory region returned extends to the end of the virtual address space,
    // with no perms if it is not mapped.

    // Use an approach that reads memory regions from /proc/{pid}/maps.
    // Assume proc maps entries are in ascending order.
    // FIXME assert if we find differently.

    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    Error error;

    if (m_supports_mem_region == LazyBool::eLazyBoolNo)
    {
        // We're done.
        error.SetErrorString ("unsupported");
        return error;
    }

    // If our cache is empty, pull the latest.  There should always be at least one memory region
    // if memory region handling is supported.
    if (m_mem_region_cache.empty ())
    {
        error = ProcFileReader::ProcessLineByLine (GetID (), "maps",
             [&] (const std::string &line) -> bool
             {
                 MemoryRegionInfo info;
                 const Error parse_error = ParseMemoryRegionInfoFromProcMapsLine (line, info);
                 if (parse_error.Success ())
                 {
                     m_mem_region_cache.push_back (info);
                     return true;
                 }
                 else
                 {
                     if (log)
                         log->Printf ("NativeProcessLinux::%s failed to parse proc maps line '%s': %s", __FUNCTION__, line.c_str (), error.AsCString ());
                     return false;
                 }
             });

        // If we had an error, we'll mark unsupported.
        if (error.Fail ())
        {
            m_supports_mem_region = LazyBool::eLazyBoolNo;
            return error;
        }
        else if (m_mem_region_cache.empty ())
        {
            // No entries after attempting to read them.  This shouldn't happen if /proc/{pid}/maps
            // is supported.  Assume we don't support map entries via procfs.
            if (log)
                log->Printf ("NativeProcessLinux::%s failed to find any procfs maps entries, assuming no support for memory region metadata retrieval", __FUNCTION__);
            m_supports_mem_region = LazyBool::eLazyBoolNo;
            error.SetErrorString ("not supported");
            return error;
        }

        if (log)
            log->Printf ("NativeProcessLinux::%s read %" PRIu64 " memory region entries from /proc/%" PRIu64 "/maps", __FUNCTION__, static_cast<uint64_t> (m_mem_region_cache.size ()), GetID ());

        // We support memory retrieval, remember that.
        m_supports_mem_region = LazyBool::eLazyBoolYes;
    }
    else
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s reusing %" PRIu64 " cached memory region entries", __FUNCTION__, static_cast<uint64_t> (m_mem_region_cache.size ()));
    }

    lldb::addr_t prev_base_address = 0;

    // FIXME start by finding the last region that is <= target address using binary search.  Data is sorted.
    // There can be a ton of regions on pthreads apps with lots of threads.
    for (auto it = m_mem_region_cache.begin(); it != m_mem_region_cache.end (); ++it)
    {
        MemoryRegionInfo &proc_entry_info = *it;

        // Sanity check assumption that /proc/{pid}/maps entries are ascending.
        assert ((proc_entry_info.GetRange ().GetRangeBase () >= prev_base_address) && "descending /proc/pid/maps entries detected, unexpected");
        prev_base_address = proc_entry_info.GetRange ().GetRangeBase ();

        // If the target address comes before this entry, indicate distance to next region.
        if (load_addr < proc_entry_info.GetRange ().GetRangeBase ())
        {
            range_info.GetRange ().SetRangeBase (load_addr);
            range_info.GetRange ().SetByteSize (proc_entry_info.GetRange ().GetRangeBase () - load_addr);
            range_info.SetReadable (MemoryRegionInfo::OptionalBool::eNo);
            range_info.SetWritable (MemoryRegionInfo::OptionalBool::eNo);
            range_info.SetExecutable (MemoryRegionInfo::OptionalBool::eNo);
            range_info.SetMapped(MemoryRegionInfo::OptionalBool::eNo);

            return error;
        }
        else if (proc_entry_info.GetRange ().Contains (load_addr))
        {
            // The target address is within the memory region we're processing here.
            range_info = proc_entry_info;
            return error;
        }

        // The target memory address comes somewhere after the region we just parsed.
    }

    // If we made it here, we didn't find an entry that contained the given address. Return the
    // load_addr as start and the amount of bytes betwwen load address and the end of the memory as
    // size.
    range_info.GetRange ().SetRangeBase (load_addr);
    range_info.GetRange ().SetRangeEnd(LLDB_INVALID_ADDRESS);
    range_info.SetReadable (MemoryRegionInfo::OptionalBool::eNo);
    range_info.SetWritable (MemoryRegionInfo::OptionalBool::eNo);
    range_info.SetExecutable (MemoryRegionInfo::OptionalBool::eNo);
    range_info.SetMapped(MemoryRegionInfo::OptionalBool::eNo);
    return error;
}

void
NativeProcessLinux::DoStopIDBumped (uint32_t newBumpId)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf ("NativeProcessLinux::%s(newBumpId=%" PRIu32 ") called", __FUNCTION__, newBumpId);

        if (log)
            log->Printf ("NativeProcessLinux::%s clearing %" PRIu64 " entries from the cache", __FUNCTION__, static_cast<uint64_t> (m_mem_region_cache.size ()));
        m_mem_region_cache.clear ();
}

Error
NativeProcessLinux::AllocateMemory(size_t size, uint32_t permissions, lldb::addr_t &addr)
{
    // FIXME implementing this requires the equivalent of
    // InferiorCallPOSIX::InferiorCallMmap, which depends on
    // functional ThreadPlans working with Native*Protocol.
#if 1
    return Error ("not implemented yet");
#else
    addr = LLDB_INVALID_ADDRESS;

    unsigned prot = 0;
    if (permissions & lldb::ePermissionsReadable)
        prot |= eMmapProtRead;
    if (permissions & lldb::ePermissionsWritable)
        prot |= eMmapProtWrite;
    if (permissions & lldb::ePermissionsExecutable)
        prot |= eMmapProtExec;

    // TODO implement this directly in NativeProcessLinux
    // (and lift to NativeProcessPOSIX if/when that class is
    // refactored out).
    if (InferiorCallMmap(this, addr, 0, size, prot,
                         eMmapFlagsAnon | eMmapFlagsPrivate, -1, 0)) {
        m_addr_to_mmap_size[addr] = size;
        return Error ();
    } else {
        addr = LLDB_INVALID_ADDRESS;
        return Error("unable to allocate %" PRIu64 " bytes of memory with permissions %s", size, GetPermissionsAsCString (permissions));
    }
#endif
}

Error
NativeProcessLinux::DeallocateMemory (lldb::addr_t addr)
{
    // FIXME see comments in AllocateMemory - required lower-level
    // bits not in place yet (ThreadPlans)
    return Error ("not implemented");
}

lldb::addr_t
NativeProcessLinux::GetSharedLibraryInfoAddress ()
{
    // punt on this for now
    return LLDB_INVALID_ADDRESS;
}

size_t
NativeProcessLinux::UpdateThreads ()
{
    // The NativeProcessLinux monitoring threads are always up to date
    // with respect to thread state and they keep the thread list
    // populated properly. All this method needs to do is return the
    // thread count.
    return m_threads.size ();
}

bool
NativeProcessLinux::GetArchitecture (ArchSpec &arch) const
{
    arch = m_arch;
    return true;
}

Error
NativeProcessLinux::GetSoftwareBreakpointPCOffset(uint32_t &actual_opcode_size)
{
    // FIXME put this behind a breakpoint protocol class that can be
    // set per architecture.  Need ARM, MIPS support here.
    static const uint8_t g_i386_opcode [] = { 0xCC };
    static const uint8_t g_s390x_opcode[] = { 0x00, 0x01 };

    switch (m_arch.GetMachine ())
    {
        case llvm::Triple::x86:
        case llvm::Triple::x86_64:
            actual_opcode_size = static_cast<uint32_t> (sizeof(g_i386_opcode));
            return Error ();

        case llvm::Triple::systemz:
            actual_opcode_size = static_cast<uint32_t> (sizeof(g_s390x_opcode));
            return Error ();

        case llvm::Triple::arm:
        case llvm::Triple::aarch64:
        case llvm::Triple::mips64:
        case llvm::Triple::mips64el:
        case llvm::Triple::mips:
        case llvm::Triple::mipsel:
            // On these architectures the PC don't get updated for breakpoint hits
            actual_opcode_size = 0;
            return Error ();
        
        default:
            assert(false && "CPU type not supported!");
            return Error ("CPU type not supported");
    }
}

Error
NativeProcessLinux::SetBreakpoint (lldb::addr_t addr, uint32_t size, bool hardware)
{
    if (hardware)
        return Error ("NativeProcessLinux does not support hardware breakpoints");
    else
        return SetSoftwareBreakpoint (addr, size);
}

Error
NativeProcessLinux::GetSoftwareBreakpointTrapOpcode (size_t trap_opcode_size_hint,
                                                     size_t &actual_opcode_size,
                                                     const uint8_t *&trap_opcode_bytes)
{
    // FIXME put this behind a breakpoint protocol class that can be set per
    // architecture.  Need MIPS support here.
    static const uint8_t g_aarch64_opcode[] = { 0x00, 0x00, 0x20, 0xd4 };
    // The ARM reference recommends the use of 0xe7fddefe and 0xdefe but the
    // linux kernel does otherwise.
    static const uint8_t g_arm_breakpoint_opcode[] = { 0xf0, 0x01, 0xf0, 0xe7 };
    static const uint8_t g_i386_opcode [] = { 0xCC };
    static const uint8_t g_mips64_opcode[] = { 0x00, 0x00, 0x00, 0x0d };
    static const uint8_t g_mips64el_opcode[] = { 0x0d, 0x00, 0x00, 0x00 };
    static const uint8_t g_s390x_opcode[] = { 0x00, 0x01 };
    static const uint8_t g_thumb_breakpoint_opcode[] = { 0x01, 0xde };

    switch (m_arch.GetMachine ())
    {
    case llvm::Triple::aarch64:
        trap_opcode_bytes = g_aarch64_opcode;
        actual_opcode_size = sizeof(g_aarch64_opcode);
        return Error ();

    case llvm::Triple::arm:
        switch (trap_opcode_size_hint)
        {
        case 2:
            trap_opcode_bytes = g_thumb_breakpoint_opcode;
            actual_opcode_size = sizeof(g_thumb_breakpoint_opcode);
            return Error ();
        case 4:
            trap_opcode_bytes = g_arm_breakpoint_opcode;
            actual_opcode_size = sizeof(g_arm_breakpoint_opcode);
            return Error ();
        default:
            assert(false && "Unrecognised trap opcode size hint!");
            return Error ("Unrecognised trap opcode size hint!");
        }

    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
        trap_opcode_bytes = g_i386_opcode;
        actual_opcode_size = sizeof(g_i386_opcode);
        return Error ();

    case llvm::Triple::mips:
    case llvm::Triple::mips64:
        trap_opcode_bytes = g_mips64_opcode;
        actual_opcode_size = sizeof(g_mips64_opcode);
        return Error ();

    case llvm::Triple::mipsel:
    case llvm::Triple::mips64el:
        trap_opcode_bytes = g_mips64el_opcode;
        actual_opcode_size = sizeof(g_mips64el_opcode);
        return Error ();

    case llvm::Triple::systemz:
        trap_opcode_bytes = g_s390x_opcode;
        actual_opcode_size = sizeof(g_s390x_opcode);
        return Error ();

    default:
        assert(false && "CPU type not supported!");
        return Error ("CPU type not supported");
    }
}

#if 0
ProcessMessage::CrashReason
NativeProcessLinux::GetCrashReasonForSIGSEGV(const siginfo_t *info)
{
    ProcessMessage::CrashReason reason;
    assert(info->si_signo == SIGSEGV);

    reason = ProcessMessage::eInvalidCrashReason;

    switch (info->si_code)
    {
    default:
        assert(false && "unexpected si_code for SIGSEGV");
        break;
    case SI_KERNEL:
        // Linux will occasionally send spurious SI_KERNEL codes.
        // (this is poorly documented in sigaction)
        // One way to get this is via unaligned SIMD loads.
        reason = ProcessMessage::eInvalidAddress; // for lack of anything better
        break;
    case SEGV_MAPERR:
        reason = ProcessMessage::eInvalidAddress;
        break;
    case SEGV_ACCERR:
        reason = ProcessMessage::ePrivilegedAddress;
        break;
    }

    return reason;
}
#endif


#if 0
ProcessMessage::CrashReason
NativeProcessLinux::GetCrashReasonForSIGILL(const siginfo_t *info)
{
    ProcessMessage::CrashReason reason;
    assert(info->si_signo == SIGILL);

    reason = ProcessMessage::eInvalidCrashReason;

    switch (info->si_code)
    {
    default:
        assert(false && "unexpected si_code for SIGILL");
        break;
    case ILL_ILLOPC:
        reason = ProcessMessage::eIllegalOpcode;
        break;
    case ILL_ILLOPN:
        reason = ProcessMessage::eIllegalOperand;
        break;
    case ILL_ILLADR:
        reason = ProcessMessage::eIllegalAddressingMode;
        break;
    case ILL_ILLTRP:
        reason = ProcessMessage::eIllegalTrap;
        break;
    case ILL_PRVOPC:
        reason = ProcessMessage::ePrivilegedOpcode;
        break;
    case ILL_PRVREG:
        reason = ProcessMessage::ePrivilegedRegister;
        break;
    case ILL_COPROC:
        reason = ProcessMessage::eCoprocessorError;
        break;
    case ILL_BADSTK:
        reason = ProcessMessage::eInternalStackError;
        break;
    }

    return reason;
}
#endif

#if 0
ProcessMessage::CrashReason
NativeProcessLinux::GetCrashReasonForSIGFPE(const siginfo_t *info)
{
    ProcessMessage::CrashReason reason;
    assert(info->si_signo == SIGFPE);

    reason = ProcessMessage::eInvalidCrashReason;

    switch (info->si_code)
    {
    default:
        assert(false && "unexpected si_code for SIGFPE");
        break;
    case FPE_INTDIV:
        reason = ProcessMessage::eIntegerDivideByZero;
        break;
    case FPE_INTOVF:
        reason = ProcessMessage::eIntegerOverflow;
        break;
    case FPE_FLTDIV:
        reason = ProcessMessage::eFloatDivideByZero;
        break;
    case FPE_FLTOVF:
        reason = ProcessMessage::eFloatOverflow;
        break;
    case FPE_FLTUND:
        reason = ProcessMessage::eFloatUnderflow;
        break;
    case FPE_FLTRES:
        reason = ProcessMessage::eFloatInexactResult;
        break;
    case FPE_FLTINV:
        reason = ProcessMessage::eFloatInvalidOperation;
        break;
    case FPE_FLTSUB:
        reason = ProcessMessage::eFloatSubscriptRange;
        break;
    }

    return reason;
}
#endif

#if 0
ProcessMessage::CrashReason
NativeProcessLinux::GetCrashReasonForSIGBUS(const siginfo_t *info)
{
    ProcessMessage::CrashReason reason;
    assert(info->si_signo == SIGBUS);

    reason = ProcessMessage::eInvalidCrashReason;

    switch (info->si_code)
    {
    default:
        assert(false && "unexpected si_code for SIGBUS");
        break;
    case BUS_ADRALN:
        reason = ProcessMessage::eIllegalAlignment;
        break;
    case BUS_ADRERR:
        reason = ProcessMessage::eIllegalAddress;
        break;
    case BUS_OBJERR:
        reason = ProcessMessage::eHardwareError;
        break;
    }

    return reason;
}
#endif

Error
NativeProcessLinux::ReadMemory (lldb::addr_t addr, void *buf, size_t size, size_t &bytes_read)
{
    if (ProcessVmReadvSupported()) {
        // The process_vm_readv path is about 50 times faster than ptrace api. We want to use
        // this syscall if it is supported.

        const ::pid_t pid = GetID();

        struct iovec local_iov, remote_iov;
        local_iov.iov_base = buf;
        local_iov.iov_len = size;
        remote_iov.iov_base = reinterpret_cast<void *>(addr);
        remote_iov.iov_len = size;

        bytes_read = process_vm_readv(pid, &local_iov, 1, &remote_iov, 1, 0);
        const bool success = bytes_read == size;

        Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
        if (log)
            log->Printf ("NativeProcessLinux::%s using process_vm_readv to read %zd bytes from inferior address 0x%" PRIx64": %s",
                    __FUNCTION__, size, addr, success ? "Success" : strerror(errno));

        if (success)
            return Error();
        // else
        //     the call failed for some reason, let's retry the read using ptrace api.
    }

    unsigned char *dst = static_cast<unsigned char*>(buf);
    size_t remainder;
    long data;

    Log *log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_ALL));
    if (log)
        ProcessPOSIXLog::IncNestLevel();
    if (log && ProcessPOSIXLog::AtTopNestLevel() && log->GetMask().Test(POSIX_LOG_MEMORY))
        log->Printf ("NativeProcessLinux::%s(%p, %p, %zd, _)", __FUNCTION__, (void*)addr, buf, size);

    for (bytes_read = 0; bytes_read < size; bytes_read += remainder)
    {
        Error error = NativeProcessLinux::PtraceWrapper(PTRACE_PEEKDATA, GetID(), (void*)addr, nullptr, 0, &data);
        if (error.Fail())
        {
            if (log)
                ProcessPOSIXLog::DecNestLevel();
            return error;
        }

        remainder = size - bytes_read;
        remainder = remainder > k_ptrace_word_size ? k_ptrace_word_size : remainder;

        // Copy the data into our buffer
        memcpy(dst, &data, remainder);

        if (log && ProcessPOSIXLog::AtTopNestLevel() &&
                (log->GetMask().Test(POSIX_LOG_MEMORY_DATA_LONG) ||
                        (log->GetMask().Test(POSIX_LOG_MEMORY_DATA_SHORT) &&
                                size <= POSIX_LOG_MEMORY_SHORT_BYTES)))
        {
            uintptr_t print_dst = 0;
            // Format bytes from data by moving into print_dst for log output
            for (unsigned i = 0; i < remainder; ++i)
                print_dst |= (((data >> i*8) & 0xFF) << i*8);
            log->Printf ("NativeProcessLinux::%s() [0x%" PRIx64 "]:0x%" PRIx64 " (0x%" PRIx64 ")",
                    __FUNCTION__, addr, uint64_t(print_dst), uint64_t(data));
        }
        addr += k_ptrace_word_size;
        dst += k_ptrace_word_size;
    }

    if (log)
        ProcessPOSIXLog::DecNestLevel();
    return Error();
}

Error
NativeProcessLinux::ReadMemoryWithoutTrap(lldb::addr_t addr, void *buf, size_t size, size_t &bytes_read)
{
    Error error = ReadMemory(addr, buf, size, bytes_read);
    if (error.Fail()) return error;
    return m_breakpoint_list.RemoveTrapsFromBuffer(addr, buf, size);
}

Error
NativeProcessLinux::WriteMemory(lldb::addr_t addr, const void *buf, size_t size, size_t &bytes_written)
{
    const unsigned char *src = static_cast<const unsigned char*>(buf);
    size_t remainder;
    Error error;

    Log *log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_ALL));
    if (log)
        ProcessPOSIXLog::IncNestLevel();
    if (log && ProcessPOSIXLog::AtTopNestLevel() && log->GetMask().Test(POSIX_LOG_MEMORY))
        log->Printf ("NativeProcessLinux::%s(0x%" PRIx64 ", %p, %zu)", __FUNCTION__, addr, buf, size);

    for (bytes_written = 0; bytes_written < size; bytes_written += remainder)
    {
        remainder = size - bytes_written;
        remainder = remainder > k_ptrace_word_size ? k_ptrace_word_size : remainder;

        if (remainder == k_ptrace_word_size)
        {
            unsigned long data = 0;
            memcpy(&data, src, k_ptrace_word_size);

            if (log && ProcessPOSIXLog::AtTopNestLevel() &&
                    (log->GetMask().Test(POSIX_LOG_MEMORY_DATA_LONG) ||
                            (log->GetMask().Test(POSIX_LOG_MEMORY_DATA_SHORT) &&
                                    size <= POSIX_LOG_MEMORY_SHORT_BYTES)))
                log->Printf ("NativeProcessLinux::%s() [%p]:0x%lx (0x%lx)", __FUNCTION__,
                        (void*)addr, *(const unsigned long*)src, data);

            error = NativeProcessLinux::PtraceWrapper(PTRACE_POKEDATA, GetID(), (void*)addr, (void*)data);
            if (error.Fail())
            {
                if (log)
                    ProcessPOSIXLog::DecNestLevel();
                return error;
            }
        }
        else
        {
            unsigned char buff[8];
            size_t bytes_read;
            error = ReadMemory(addr, buff, k_ptrace_word_size, bytes_read);
            if (error.Fail())
            {
                if (log)
                    ProcessPOSIXLog::DecNestLevel();
                return error;
            }

            memcpy(buff, src, remainder);

            size_t bytes_written_rec;
            error = WriteMemory(addr, buff, k_ptrace_word_size, bytes_written_rec);
            if (error.Fail())
            {
                if (log)
                    ProcessPOSIXLog::DecNestLevel();
                return error;
            }

            if (log && ProcessPOSIXLog::AtTopNestLevel() &&
                    (log->GetMask().Test(POSIX_LOG_MEMORY_DATA_LONG) ||
                            (log->GetMask().Test(POSIX_LOG_MEMORY_DATA_SHORT) &&
                                    size <= POSIX_LOG_MEMORY_SHORT_BYTES)))
                log->Printf ("NativeProcessLinux::%s() [%p]:0x%lx (0x%lx)", __FUNCTION__,
                        (void*)addr, *(const unsigned long*)src, *(unsigned long*)buff);
        }

        addr += k_ptrace_word_size;
        src += k_ptrace_word_size;
    }
    if (log)
        ProcessPOSIXLog::DecNestLevel();
    return error;
}

Error
NativeProcessLinux::GetSignalInfo(lldb::tid_t tid, void *siginfo)
{
    return PtraceWrapper(PTRACE_GETSIGINFO, tid, nullptr, siginfo);
}

Error
NativeProcessLinux::GetEventMessage(lldb::tid_t tid, unsigned long *message)
{
    return PtraceWrapper(PTRACE_GETEVENTMSG, tid, nullptr, message);
}

Error
NativeProcessLinux::Detach(lldb::tid_t tid)
{
    if (tid == LLDB_INVALID_THREAD_ID)
        return Error();

    return PtraceWrapper(PTRACE_DETACH, tid);
}

bool
NativeProcessLinux::DupDescriptor(const FileSpec &file_spec, int fd, int flags)
{
    int target_fd = open(file_spec.GetCString(), flags, 0666);

    if (target_fd == -1)
        return false;

    if (dup2(target_fd, fd) == -1)
        return false;

    return (close(target_fd) == -1) ? false : true;
}

bool
NativeProcessLinux::HasThreadNoLock (lldb::tid_t thread_id)
{
    for (auto thread_sp : m_threads)
    {
        assert (thread_sp && "thread list should not contain NULL threads");
        if (thread_sp->GetID () == thread_id)
        {
            // We have this thread.
            return true;
        }
    }

    // We don't have this thread.
    return false;
}

bool
NativeProcessLinux::StopTrackingThread (lldb::tid_t thread_id)
{
    Log *const log = GetLogIfAllCategoriesSet (LIBLLDB_LOG_THREAD);

    if (log)
        log->Printf("NativeProcessLinux::%s (tid: %" PRIu64 ")", __FUNCTION__, thread_id);

    bool found = false;

    for (auto it = m_threads.begin (); it != m_threads.end (); ++it)
    {
        if (*it && ((*it)->GetID () == thread_id))
        {
            m_threads.erase (it);
            found = true;
            break;
        }
    }

    SignalIfAllThreadsStopped();

    return found;
}

NativeThreadLinuxSP
NativeProcessLinux::AddThread (lldb::tid_t thread_id)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_THREAD));

    if (log)
    {
        log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " adding thread with tid %" PRIu64,
                __FUNCTION__,
                GetID (),
                thread_id);
    }

    assert (!HasThreadNoLock (thread_id) && "attempted to add a thread by id that already exists");

    // If this is the first thread, save it as the current thread
    if (m_threads.empty ())
        SetCurrentThreadID (thread_id);

    auto thread_sp = std::make_shared<NativeThreadLinux>(this, thread_id);
    m_threads.push_back (thread_sp);
    return thread_sp;
}

Error
NativeProcessLinux::FixupBreakpointPCAsNeeded(NativeThreadLinux &thread)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS));

    Error error;

    // Find out the size of a breakpoint (might depend on where we are in the code).
    NativeRegisterContextSP context_sp = thread.GetRegisterContext();
    if (!context_sp)
    {
        error.SetErrorString ("cannot get a NativeRegisterContext for the thread");
        if (log)
            log->Printf ("NativeProcessLinux::%s failed: %s", __FUNCTION__, error.AsCString ());
        return error;
    }

    uint32_t breakpoint_size = 0;
    error = GetSoftwareBreakpointPCOffset(breakpoint_size);
    if (error.Fail ())
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s GetBreakpointSize() failed: %s", __FUNCTION__, error.AsCString ());
        return error;
    }
    else
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s breakpoint size: %" PRIu32, __FUNCTION__, breakpoint_size);
    }

    // First try probing for a breakpoint at a software breakpoint location: PC - breakpoint size.
    const lldb::addr_t initial_pc_addr = context_sp->GetPCfromBreakpointLocation ();
    lldb::addr_t breakpoint_addr = initial_pc_addr;
    if (breakpoint_size > 0)
    {
        // Do not allow breakpoint probe to wrap around.
        if (breakpoint_addr >= breakpoint_size)
            breakpoint_addr -= breakpoint_size;
    }

    // Check if we stopped because of a breakpoint.
    NativeBreakpointSP breakpoint_sp;
    error = m_breakpoint_list.GetBreakpoint (breakpoint_addr, breakpoint_sp);
    if (!error.Success () || !breakpoint_sp)
    {
        // We didn't find one at a software probe location.  Nothing to do.
        if (log)
            log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " no lldb breakpoint found at current pc with adjustment: 0x%" PRIx64, __FUNCTION__, GetID (), breakpoint_addr);
        return Error ();
    }

    // If the breakpoint is not a software breakpoint, nothing to do.
    if (!breakpoint_sp->IsSoftwareBreakpoint ())
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " breakpoint found at 0x%" PRIx64 ", not software, nothing to adjust", __FUNCTION__, GetID (), breakpoint_addr);
        return Error ();
    }

    //
    // We have a software breakpoint and need to adjust the PC.
    //

    // Sanity check.
    if (breakpoint_size == 0)
    {
        // Nothing to do!  How did we get here?
        if (log)
            log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " breakpoint found at 0x%" PRIx64 ", it is software, but the size is zero, nothing to do (unexpected)", __FUNCTION__, GetID (), breakpoint_addr);
        return Error ();
    }

    // Change the program counter.
    if (log)
        log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " tid %" PRIu64 ": changing PC from 0x%" PRIx64 " to 0x%" PRIx64, __FUNCTION__, GetID(), thread.GetID(), initial_pc_addr, breakpoint_addr);

    error = context_sp->SetPC (breakpoint_addr);
    if (error.Fail ())
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " tid %" PRIu64 ": failed to set PC: %s", __FUNCTION__, GetID(), thread.GetID(), error.AsCString ());
        return error;
    }

    return error;
}

Error
NativeProcessLinux::GetLoadedModuleFileSpec(const char* module_path, FileSpec& file_spec)
{
    FileSpec module_file_spec(module_path, true);

    bool found = false;
    file_spec.Clear();
    ProcFileReader::ProcessLineByLine(GetID(), "maps",
        [&] (const std::string &line)
        {
            SmallVector<StringRef, 16> columns;
            StringRef(line).split(columns, " ", -1, false);
            if (columns.size() < 6)
                return true; // continue searching

            FileSpec this_file_spec(columns[5].str().c_str(), false);
            if (this_file_spec.GetFilename() != module_file_spec.GetFilename())
                return true; // continue searching

            file_spec = this_file_spec;
            found = true;
            return false; // we are done
        });

    if (! found)
        return Error("Module file (%s) not found in /proc/%" PRIu64 "/maps file!",
                module_file_spec.GetFilename().AsCString(), GetID());

    return Error();
}

Error
NativeProcessLinux::GetFileLoadAddress(const llvm::StringRef& file_name, lldb::addr_t& load_addr)
{
    load_addr = LLDB_INVALID_ADDRESS;
    Error error = ProcFileReader::ProcessLineByLine (GetID (), "maps",
        [&] (const std::string &line) -> bool
        {
            StringRef maps_row(line);
 
            SmallVector<StringRef, 16> maps_columns;
            maps_row.split(maps_columns, StringRef(" "), -1, false);
 
            if (maps_columns.size() < 6)
            {
                // Return true to continue reading the proc file
                return true;
            }

            if (maps_columns[5] == file_name)
            {
                StringExtractor addr_extractor(maps_columns[0].str().c_str());
                load_addr = addr_extractor.GetHexMaxU64(false, LLDB_INVALID_ADDRESS); 

                // Return false to stop reading the proc file further
                return false;
            }
 
            // Return true to continue reading the proc file
            return true;
        });
    return error; 
}

NativeThreadLinuxSP
NativeProcessLinux::GetThreadByID(lldb::tid_t tid)
{
    return std::static_pointer_cast<NativeThreadLinux>(NativeProcessProtocol::GetThreadByID(tid));
}

Error
NativeProcessLinux::ResumeThread(NativeThreadLinux &thread, lldb::StateType state, int signo)
{
    Log *const log = GetLogIfAllCategoriesSet (LIBLLDB_LOG_THREAD);

    if (log)
        log->Printf("NativeProcessLinux::%s (tid: %" PRIu64 ")",
                __FUNCTION__, thread.GetID());
    
    // Before we do the resume below, first check if we have a pending
    // stop notification that is currently waiting for
    // all threads to stop.  This is potentially a buggy situation since
    // we're ostensibly waiting for threads to stop before we send out the
    // pending notification, and here we are resuming one before we send
    // out the pending stop notification.
    if (m_pending_notification_tid != LLDB_INVALID_THREAD_ID && log)
    {
        log->Printf("NativeProcessLinux::%s about to resume tid %" PRIu64 " per explicit request but we have a pending stop notification (tid %" PRIu64 ") that is actively waiting for this thread to stop. Valid sequence of events?", __FUNCTION__, thread.GetID(), m_pending_notification_tid);
    }

    // Request a resume.  We expect this to be synchronous and the system
    // to reflect it is running after this completes.
    switch (state)
    {
    case eStateRunning:
    {
        const auto resume_result = thread.Resume(signo);
        if (resume_result.Success())
            SetState(eStateRunning, true);
        return resume_result;
    }
    case eStateStepping:
    {
        const auto step_result = thread.SingleStep(signo);
        if (step_result.Success())
            SetState(eStateRunning, true);
        return step_result;
    }
    default:
        if (log)
            log->Printf("NativeProcessLinux::%s Unhandled state %s.",
                    __FUNCTION__, StateAsCString(state));
        llvm_unreachable("Unhandled state for resume");
    }
}

//===----------------------------------------------------------------------===//

void
NativeProcessLinux::StopRunningThreads(const lldb::tid_t triggering_tid)
{
    Log *const log = GetLogIfAllCategoriesSet (LIBLLDB_LOG_THREAD);

    if (log)
    {
        log->Printf("NativeProcessLinux::%s about to process event: (triggering_tid: %" PRIu64 ")",
                __FUNCTION__, triggering_tid);
    }

    m_pending_notification_tid = triggering_tid;

    // Request a stop for all the thread stops that need to be stopped
    // and are not already known to be stopped.
    for (const auto &thread_sp: m_threads)
    {
        if (StateIsRunningState(thread_sp->GetState()))
            static_pointer_cast<NativeThreadLinux>(thread_sp)->RequestStop();
    }

    SignalIfAllThreadsStopped();

    if (log)
    {
        log->Printf("NativeProcessLinux::%s event processing done", __FUNCTION__);
    }
}

void
NativeProcessLinux::SignalIfAllThreadsStopped()
{
    if (m_pending_notification_tid == LLDB_INVALID_THREAD_ID)
        return; // No pending notification. Nothing to do.

    for (const auto &thread_sp: m_threads)
    {
        if (StateIsRunningState(thread_sp->GetState()))
            return; // Some threads are still running. Don't signal yet.
    }

    // We have a pending notification and all threads have stopped.
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_BREAKPOINTS));

    // Clear any temporary breakpoints we used to implement software single stepping.
    for (const auto &thread_info: m_threads_stepping_with_breakpoint)
    {
        Error error = RemoveBreakpoint (thread_info.second);
        if (error.Fail())
            if (log)
                log->Printf("NativeProcessLinux::%s() pid = %" PRIu64 " remove stepping breakpoint: %s",
                        __FUNCTION__, thread_info.first, error.AsCString());
    }
    m_threads_stepping_with_breakpoint.clear();

    // Notify the delegate about the stop
    SetCurrentThreadID(m_pending_notification_tid);
    SetState(StateType::eStateStopped, true);
    m_pending_notification_tid = LLDB_INVALID_THREAD_ID;
}

void
NativeProcessLinux::ThreadWasCreated(NativeThreadLinux &thread)
{
    Log *const log = GetLogIfAllCategoriesSet (LIBLLDB_LOG_THREAD);

    if (log)
        log->Printf("NativeProcessLinux::%s (tid: %" PRIu64 ")", __FUNCTION__, thread.GetID());

    if (m_pending_notification_tid != LLDB_INVALID_THREAD_ID && StateIsRunningState(thread.GetState()))
    {
        // We will need to wait for this new thread to stop as well before firing the
        // notification.
        thread.RequestStop();
    }
}

void
NativeProcessLinux::SigchldHandler()
{
    Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));
    // Process all pending waitpid notifications.
    while (true)
    {
        int status = -1;
        ::pid_t wait_pid = waitpid(-1, &status, __WALL | __WNOTHREAD | WNOHANG);

        if (wait_pid == 0)
            break; // We are done.

        if (wait_pid == -1)
        {
            if (errno == EINTR)
                continue;

            Error error(errno, eErrorTypePOSIX);
            if (log)
                log->Printf("NativeProcessLinux::%s waitpid (-1, &status, __WALL | __WNOTHREAD | WNOHANG) failed: %s",
                        __FUNCTION__, error.AsCString());
            break;
        }

        bool exited = false;
        int signal = 0;
        int exit_status = 0;
        const char *status_cstr = nullptr;
        if (WIFSTOPPED(status))
        {
            signal = WSTOPSIG(status);
            status_cstr = "STOPPED";
        }
        else if (WIFEXITED(status))
        {
            exit_status = WEXITSTATUS(status);
            status_cstr = "EXITED";
            exited = true;
        }
        else if (WIFSIGNALED(status))
        {
            signal = WTERMSIG(status);
            status_cstr = "SIGNALED";
            if (wait_pid == static_cast< ::pid_t>(GetID())) {
                exited = true;
                exit_status = -1;
            }
        }
        else
            status_cstr = "(\?\?\?)";

        if (log)
            log->Printf("NativeProcessLinux::%s: waitpid (-1, &status, __WALL | __WNOTHREAD | WNOHANG)"
                "=> pid = %" PRIi32 ", status = 0x%8.8x (%s), signal = %i, exit_state = %i",
                __FUNCTION__, wait_pid, status, status_cstr, signal, exit_status);

        MonitorCallback (wait_pid, exited, signal, exit_status);
    }
}

// Wrapper for ptrace to catch errors and log calls.
// Note that ptrace sets errno on error because -1 can be a valid result (i.e. for PTRACE_PEEK*)
Error
NativeProcessLinux::PtraceWrapper(int req, lldb::pid_t pid, void *addr, void *data, size_t data_size, long *result)
{
    Error error;
    long int ret;

    Log *log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_PTRACE));

    PtraceDisplayBytes(req, data, data_size);

    errno = 0;
    if (req == PTRACE_GETREGSET || req == PTRACE_SETREGSET)
        ret = ptrace(static_cast<__ptrace_request>(req), static_cast< ::pid_t>(pid), *(unsigned int *)addr, data);
    else
        ret = ptrace(static_cast<__ptrace_request>(req), static_cast< ::pid_t>(pid), addr, data);

    if (ret == -1)
        error.SetErrorToErrno();

    if (result)
        *result = ret;

    if (log)
        log->Printf("ptrace(%d, %" PRIu64 ", %p, %p, %zu)=%lX", req, pid, addr, data, data_size, ret);

    PtraceDisplayBytes(req, data, data_size);

    if (log && error.GetError() != 0)
    {
        const char* str;
        switch (error.GetError())
        {
        case ESRCH:  str = "ESRCH"; break;
        case EINVAL: str = "EINVAL"; break;
        case EBUSY:  str = "EBUSY"; break;
        case EPERM:  str = "EPERM"; break;
        default:     str = error.AsCString();
        }
        log->Printf("ptrace() failed; errno=%d (%s)", error.GetError(), str);
    }

    return error;
}
