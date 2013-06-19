//===-- ProcessMonitor.h -------------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessMonitor_H_
#define liblldb_ProcessMonitor_H_

// C Includes
#include <semaphore.h>
#include <signal.h>

// C++ Includes
// Other libraries and framework includes
#include "lldb/lldb-types.h"
#include "lldb/Host/Mutex.h"

namespace lldb_private
{
class Error;
class Module;
class Scalar;
} // End lldb_private namespace.

class ProcessLinux;
class Operation;

/// @class ProcessMonitor
/// @brief Manages communication with the inferior (debugee) process.
///
/// Upon construction, this class prepares and launches an inferior process for
/// debugging.
///
/// Changes in the inferior process state are propagated to the associated
/// ProcessLinux instance by calling ProcessLinux::SendMessage with the
/// appropriate ProcessMessage events.
///
/// A purposely minimal set of operations are provided to interrogate and change
/// the inferior process state.
class ProcessMonitor
{
public:

    /// Launches an inferior process ready for debugging.  Forms the
    /// implementation of Process::DoLaunch.
    ProcessMonitor(ProcessPOSIX *process,
                   lldb_private::Module *module,
                   char const *argv[],
                   char const *envp[],
                   const char *stdin_path,
                   const char *stdout_path,
                   const char *stderr_path,
                   const char *working_dir,
                   lldb_private::Error &error);

    ProcessMonitor(ProcessPOSIX *process,
                   lldb::pid_t pid,
                   lldb_private::Error &error);

    ~ProcessMonitor();

    enum ResumeSignals 
    {
        eResumeSignalNone = 0
    };

    /// Provides the process number of debugee.
    lldb::pid_t
    GetPID() const { return m_pid; }

    /// Returns the process associated with this ProcessMonitor.
    ProcessLinux &
    GetProcess() { return *m_process; }

    /// Returns a file descriptor to the controlling terminal of the inferior
    /// process.
    ///
    /// Reads from this file descriptor yield both the standard output and
    /// standard error of this debugee.  Even if stderr and stdout were
    /// redirected on launch it may still happen that data is available on this
    /// descriptor (if the inferior process opens /dev/tty, for example).
    ///
    /// If this monitor was attached to an existing process this method returns
    /// -1.
    int
    GetTerminalFD() const { return m_terminal_fd; }

    /// Reads @p size bytes from address @vm_adder in the inferior process
    /// address space.
    ///
    /// This method is provided to implement Process::DoReadMemory.
    size_t
    ReadMemory(lldb::addr_t vm_addr, void *buf, size_t size,
               lldb_private::Error &error);

    /// Writes @p size bytes from address @p vm_adder in the inferior process
    /// address space.
    ///
    /// This method is provided to implement Process::DoWriteMemory.
    size_t
    WriteMemory(lldb::addr_t vm_addr, const void *buf, size_t size,
                lldb_private::Error &error);

    /// Reads the contents from the register identified by the given (architecture
    /// dependent) offset.
    ///
    /// This method is provided for use by RegisterContextLinux derivatives.
    bool
    ReadRegisterValue(lldb::tid_t tid, unsigned offset, const char *reg_name,
                      unsigned size, lldb_private::RegisterValue &value);

    /// Writes the given value to the register identified by the given
    /// (architecture dependent) offset.
    ///
    /// This method is provided for use by RegisterContextLinux derivatives.
    bool
    WriteRegisterValue(lldb::tid_t tid, unsigned offset, const char *reg_name,
                       const lldb_private::RegisterValue &value);

    /// Reads all general purpose registers into the specified buffer.
    bool
    ReadGPR(lldb::tid_t tid, void *buf, size_t buf_size);

    /// Reads generic floating point registers into the specified buffer.
    bool
    ReadFPR(lldb::tid_t tid, void *buf, size_t buf_size);

    /// Reads the specified register set into the specified buffer.
    /// For instance, the extended floating-point register set.
    bool
    ReadRegisterSet(lldb::tid_t tid, void *buf, size_t buf_size, unsigned int regset);

    /// Writes all general purpose registers into the specified buffer.
    bool
    WriteGPR(lldb::tid_t tid, void *buf, size_t buf_size);

    /// Writes generic floating point registers into the specified buffer.
    bool
    WriteFPR(lldb::tid_t tid, void *buf, size_t buf_size);

    /// Writes the specified register set into the specified buffer.
    /// For instance, the extended floating-point register set.
    bool
    WriteRegisterSet(lldb::tid_t tid, void *buf, size_t buf_size, unsigned int regset);

    /// Writes a siginfo_t structure corresponding to the given thread ID to the
    /// memory region pointed to by @p siginfo.
    bool
    GetSignalInfo(lldb::tid_t tid, void *siginfo, int &ptrace_err);

    /// Writes the raw event message code (vis-a-vis PTRACE_GETEVENTMSG)
    /// corresponding to the given thread IDto the memory pointed to by @p
    /// message.
    bool
    GetEventMessage(lldb::tid_t tid, unsigned long *message);

    /// Resumes the given thread.  If @p signo is anything but
    /// LLDB_INVALID_SIGNAL_NUMBER, deliver that signal to the thread.
    bool
    Resume(lldb::tid_t tid, uint32_t signo);

    /// Single steps the given thread.  If @p signo is anything but
    /// LLDB_INVALID_SIGNAL_NUMBER, deliver that signal to the thread.
    bool
    SingleStep(lldb::tid_t tid, uint32_t signo);

    /// Sends the inferior process a PTRACE_KILL signal.  The inferior will
    /// still exists and can be interrogated.  Once resumed it will exit as
    /// though it received a SIGKILL.
    bool
    BringProcessIntoLimbo();

    lldb_private::Error
    Detach(lldb::tid_t tid);

    /// Stops the requested thread and waits for the stop signal.
    bool
    StopThread(lldb::tid_t tid);

private:
    ProcessLinux *m_process;

    lldb::thread_t m_operation_thread;
    lldb::thread_t m_monitor_thread;
    lldb::pid_t m_pid;
    int m_terminal_fd;


    lldb_private::Mutex m_server_mutex;
    int m_client_fd;
    int m_server_fd;

    struct OperationArgs
    {
        OperationArgs(ProcessMonitor *monitor);

        ~OperationArgs();

        ProcessMonitor *m_monitor;      // The monitor performing the attach.
        sem_t m_semaphore;              // Posted to once operation complete.
        lldb_private::Error m_error;    // Set if process operation failed.
    };

    /// @class LauchArgs
    ///
    /// @brief Simple structure to pass data to the thread responsible for
    /// launching a child process.
    struct LaunchArgs : OperationArgs
    {
        LaunchArgs(ProcessMonitor *monitor,
                   lldb_private::Module *module,
                   char const **argv,
                   char const **envp,
                   const char *stdin_path,
                   const char *stdout_path,
                   const char *stderr_path,
                   const char *working_dir);

        ~LaunchArgs();

        lldb_private::Module *m_module; // The executable image to launch.
        char const **m_argv;            // Process arguments.
        char const **m_envp;            // Process environment.
        const char *m_stdin_path;       // Redirect stdin or NULL.
        const char *m_stdout_path;      // Redirect stdout or NULL.
        const char *m_stderr_path;      // Redirect stderr or NULL.
        const char *m_working_dir;      // Working directory or NULL.
    };

    void
    StartLaunchOpThread(LaunchArgs *args, lldb_private::Error &error);

    static void *
    LaunchOpThread(void *arg);

    static bool
    Launch(LaunchArgs *args);

    bool
    EnableIPC();

    struct AttachArgs : OperationArgs
    {
        AttachArgs(ProcessMonitor *monitor,
                   lldb::pid_t pid);

        ~AttachArgs();

        lldb::pid_t m_pid;              // pid of the process to be attached.
    };

    void
    StartAttachOpThread(AttachArgs *args, lldb_private::Error &error);

    static void *
    AttachOpThread(void *args);

    static bool
    Attach(AttachArgs *args);

    static bool
    SetDefaultPtraceOpts(const lldb::pid_t);

    static void
    ServeOperation(OperationArgs *args);

    static bool
    DupDescriptor(const char *path, int fd, int flags);

    static bool
    MonitorCallback(void *callback_baton,
                    lldb::pid_t pid, bool exited, int signal, int status);

    static ProcessMessage
    MonitorSIGTRAP(ProcessMonitor *monitor,
                   const siginfo_t *info, lldb::pid_t pid);

    static ProcessMessage
    MonitorSignal(ProcessMonitor *monitor, 
                  const siginfo_t *info, lldb::pid_t pid);

    static ProcessMessage::CrashReason
    GetCrashReasonForSIGSEGV(const siginfo_t *info);

    static ProcessMessage::CrashReason
    GetCrashReasonForSIGILL(const siginfo_t *info);

    static ProcessMessage::CrashReason
    GetCrashReasonForSIGFPE(const siginfo_t *info);

    static ProcessMessage::CrashReason
    GetCrashReasonForSIGBUS(const siginfo_t *info);

    void
    DoOperation(Operation *op);

    /// Stops the child monitor thread.
    void
    StopMonitoringChildProcess();

    void 
    StopMonitor();

    /// Stops the operation thread used to attach/launch a process.
    void
    StopOpThread();

    void
    CloseFD(int &fd);
};

#endif // #ifndef liblldb_ProcessMonitor_H_
