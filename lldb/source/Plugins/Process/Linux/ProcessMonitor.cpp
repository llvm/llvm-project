//===-- ProcessMonitor.cpp ------------------------------------ -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <errno.h>
#include <poll.h>
#include <string.h>
#include <unistd.h>
#include <sys/ptrace.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/user.h>
#include <sys/wait.h>

// C++ Includes
// Other libraries and framework includes
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Utility/PseudoTerminal.h"

#include "POSIXThread.h"
#include "ProcessLinux.h"
#include "ProcessPOSIXLog.h"
#include "ProcessMonitor.h"


#define DEBUG_PTRACE_MAXBYTES 20

using namespace lldb_private;

// FIXME: this code is host-dependent with respect to types and
// endianness and needs to be fixed.  For example, lldb::addr_t is
// hard-coded to uint64_t, but on a 32-bit Linux host, ptrace requires
// 32-bit pointer arguments.  This code uses casts to work around the
// problem.

// We disable the tracing of ptrace calls for integration builds to
// avoid the additional indirection and checks.
#ifndef LLDB_CONFIGURATION_BUILDANDINTEGRATION

static void
DisplayBytes (lldb_private::StreamString &s, void *bytes, uint32_t count)
{
    uint8_t *ptr = (uint8_t *)bytes;
    const uint32_t loop_count = std::min<uint32_t>(DEBUG_PTRACE_MAXBYTES, count);
    for(uint32_t i=0; i<loop_count; i++)
    {
        s.Printf ("[%x]", *ptr);
        ptr++;
    }
}

static void PtraceDisplayBytes(__ptrace_request &req, void *data)
{
    StreamString buf;
    LogSP verbose_log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (
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
                DisplayBytes(buf, data, sizeof(user_regs_struct));
                verbose_log->Printf("PTRACE_SETREGS %s", buf.GetData());
                break;
            }
        case PTRACE_SETFPREGS:
            {
                DisplayBytes(buf, data, sizeof(user_fpregs_struct));
                verbose_log->Printf("PTRACE_SETFPREGS %s", buf.GetData());
                break;
            }
        case PTRACE_SETSIGINFO:
            {
                DisplayBytes(buf, data, sizeof(siginfo_t));
                verbose_log->Printf("PTRACE_SETSIGINFO %s", buf.GetData());
                break;
            }
        default:
            {
            }
        }
    }
}

// Wrapper for ptrace to catch errors and log calls.
extern long
PtraceWrapper(__ptrace_request req, pid_t pid, void *addr, void *data,
              const char* reqName, const char* file, int line)
{
    long int result;

    LogSP log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_PTRACE));

    if (log)
        log->Printf("ptrace(%s, %u, %p, %p) called from file %s line %d",
                    reqName, pid, addr, data, file, line);

    PtraceDisplayBytes(req, data);

    errno = 0;
    result = ptrace(req, pid, addr, data);

    PtraceDisplayBytes(req, data);

    if (log && (result == -1 || errno != 0))
    {
        const char* str;
        switch (errno)
        {
        case ESRCH:  str = "ESRCH"; break;
        case EINVAL: str = "EINVAL"; break;
        case EBUSY:  str = "EBUSY"; break;
        case EPERM:  str = "EPERM"; break;
        default:     str = "<unknown>";
        }
        log->Printf("ptrace() failed; errno=%d (%s)", errno, str);
    }

    return result;
}

#define PTRACE(req, pid, addr, data) \
    PtraceWrapper((req), (pid), (addr), (data), #req, __FILE__, __LINE__)
#else
#define PTRACE ptrace
#endif

//------------------------------------------------------------------------------
// Static implementations of ProcessMonitor::ReadMemory and
// ProcessMonitor::WriteMemory.  This enables mutual recursion between these
// functions without needed to go thru the thread funnel.

static size_t
DoReadMemory(lldb::pid_t pid,
             lldb::addr_t vm_addr, void *buf, size_t size, Error &error)
{
    // ptrace word size is determined by the host, not the child
    static const unsigned word_size = sizeof(void*);
    unsigned char *dst = static_cast<unsigned char*>(buf);
    size_t bytes_read;
    size_t remainder;
    long data;

    LogSP log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_ALL));
    if (log)
        ProcessPOSIXLog::IncNestLevel();
    if (log && ProcessPOSIXLog::AtTopNestLevel() && log->GetMask().Test(POSIX_LOG_MEMORY))
        log->Printf ("ProcessMonitor::%s(%d, %d, %p, %p, %d, _)", __FUNCTION__,
                     pid, word_size, (void*)vm_addr, buf, size);

    assert(sizeof(data) >= word_size);
    for (bytes_read = 0; bytes_read < size; bytes_read += remainder)
    {
        errno = 0;
        data = PTRACE(PTRACE_PEEKDATA, pid, (void*)vm_addr, NULL);
        if (data == -1L && errno)
        {
            error.SetErrorToErrno();
            if (log)
                ProcessPOSIXLog::DecNestLevel();
            return bytes_read;
        }

        remainder = size - bytes_read;
        remainder = remainder > word_size ? word_size : remainder;

        // Copy the data into our buffer
        if (log)
            memset(dst, 0, sizeof(dst));
        for (unsigned i = 0; i < remainder; ++i)
            dst[i] = ((data >> i*8) & 0xFF);

        if (log && ProcessPOSIXLog::AtTopNestLevel() &&
            (log->GetMask().Test(POSIX_LOG_MEMORY_DATA_LONG) ||
             (log->GetMask().Test(POSIX_LOG_MEMORY_DATA_SHORT) &&
              size <= POSIX_LOG_MEMORY_SHORT_BYTES)))
            log->Printf ("ProcessMonitor::%s() [%p]:0x%lx (0x%lx)", __FUNCTION__,
                         (void*)vm_addr, *(unsigned long*)dst, (unsigned long)data);

        vm_addr += word_size;
        dst += word_size;
    }

    if (log)
        ProcessPOSIXLog::DecNestLevel();
    return bytes_read;
}

static size_t
DoWriteMemory(lldb::pid_t pid,
              lldb::addr_t vm_addr, const void *buf, size_t size, Error &error)
{
    // ptrace word size is determined by the host, not the child
    static const unsigned word_size = sizeof(void*);
    const unsigned char *src = static_cast<const unsigned char*>(buf);
    size_t bytes_written = 0;
    size_t remainder;

    LogSP log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_ALL));
    if (log)
        ProcessPOSIXLog::IncNestLevel();
    if (log && ProcessPOSIXLog::AtTopNestLevel() && log->GetMask().Test(POSIX_LOG_MEMORY))
        log->Printf ("ProcessMonitor::%s(%d, %d, %p, %p, %d, _)", __FUNCTION__,
                     pid, word_size, (void*)vm_addr, buf, size);

    for (bytes_written = 0; bytes_written < size; bytes_written += remainder)
    {
        remainder = size - bytes_written;
        remainder = remainder > word_size ? word_size : remainder;

        if (remainder == word_size)
        {
            unsigned long data = 0;
            assert(sizeof(data) >= word_size);
            for (unsigned i = 0; i < word_size; ++i)
                data |= (unsigned long)src[i] << i*8;

            if (log && ProcessPOSIXLog::AtTopNestLevel() &&
                (log->GetMask().Test(POSIX_LOG_MEMORY_DATA_LONG) ||
                 (log->GetMask().Test(POSIX_LOG_MEMORY_DATA_SHORT) &&
                  size <= POSIX_LOG_MEMORY_SHORT_BYTES)))
                 log->Printf ("ProcessMonitor::%s() [%p]:0x%lx (0x%lx)", __FUNCTION__,
                              (void*)vm_addr, *(unsigned long*)src, data);

            if (PTRACE(PTRACE_POKEDATA, pid, (void*)vm_addr, (void*)data))
            {
                error.SetErrorToErrno();
                if (log)
                    ProcessPOSIXLog::DecNestLevel();
                return bytes_written;
            }
        }
        else
        {
            unsigned char buff[8];
            if (DoReadMemory(pid, vm_addr,
                             buff, word_size, error) != word_size)
            {
                if (log)
                    ProcessPOSIXLog::DecNestLevel();
                return bytes_written;
            }

            memcpy(buff, src, remainder);

            if (DoWriteMemory(pid, vm_addr,
                              buff, word_size, error) != word_size)
            {
                if (log)
                    ProcessPOSIXLog::DecNestLevel();
                return bytes_written;
            }

            if (log && ProcessPOSIXLog::AtTopNestLevel() &&
                (log->GetMask().Test(POSIX_LOG_MEMORY_DATA_LONG) ||
                 (log->GetMask().Test(POSIX_LOG_MEMORY_DATA_SHORT) &&
                  size <= POSIX_LOG_MEMORY_SHORT_BYTES)))
                 log->Printf ("ProcessMonitor::%s() [%p]:0x%lx (0x%lx)", __FUNCTION__,
                              (void*)vm_addr, *(unsigned long*)src, *(unsigned long*)buff);
        }

        vm_addr += word_size;
        src += word_size;
    }
    if (log)
        ProcessPOSIXLog::DecNestLevel();
    return bytes_written;
}

// Simple helper function to ensure flags are enabled on the given file
// descriptor.
static bool
EnsureFDFlags(int fd, int flags, Error &error)
{
    int status;

    if ((status = fcntl(fd, F_GETFL)) == -1)
    {
        error.SetErrorToErrno();
        return false;
    }

    if (fcntl(fd, F_SETFL, status | flags) == -1)
    {
        error.SetErrorToErrno();
        return false;
    }

    return true;
}

//------------------------------------------------------------------------------
/// @class Operation
/// @brief Represents a ProcessMonitor operation.
///
/// Under Linux, it is not possible to ptrace() from any other thread but the
/// one that spawned or attached to the process from the start.  Therefore, when
/// a ProcessMonitor is asked to deliver or change the state of an inferior
/// process the operation must be "funneled" to a specific thread to perform the
/// task.  The Operation class provides an abstract base for all services the
/// ProcessMonitor must perform via the single virtual function Execute, thus
/// encapsulating the code that needs to run in the privileged context.
class Operation
{
public:
    virtual void Execute(ProcessMonitor *monitor) = 0;
};

//------------------------------------------------------------------------------
/// @class ReadOperation
/// @brief Implements ProcessMonitor::ReadMemory.
class ReadOperation : public Operation
{
public:
    ReadOperation(lldb::addr_t addr, void *buff, size_t size,
                  Error &error, size_t &result)
        : m_addr(addr), m_buff(buff), m_size(size),
          m_error(error), m_result(result)
        { }

    void Execute(ProcessMonitor *monitor);

private:
    lldb::addr_t m_addr;
    void *m_buff;
    size_t m_size;
    Error &m_error;
    size_t &m_result;
};

void
ReadOperation::Execute(ProcessMonitor *monitor)
{
    lldb::pid_t pid = monitor->GetPID();

    m_result = DoReadMemory(pid, m_addr, m_buff, m_size, m_error);
}

//------------------------------------------------------------------------------
/// @class ReadOperation
/// @brief Implements ProcessMonitor::WriteMemory.
class WriteOperation : public Operation
{
public:
    WriteOperation(lldb::addr_t addr, const void *buff, size_t size,
                   Error &error, size_t &result)
        : m_addr(addr), m_buff(buff), m_size(size),
          m_error(error), m_result(result)
        { }

    void Execute(ProcessMonitor *monitor);

private:
    lldb::addr_t m_addr;
    const void *m_buff;
    size_t m_size;
    Error &m_error;
    size_t &m_result;
};

void
WriteOperation::Execute(ProcessMonitor *monitor)
{
    lldb::pid_t pid = monitor->GetPID();

    m_result = DoWriteMemory(pid, m_addr, m_buff, m_size, m_error);
}


//------------------------------------------------------------------------------
/// @class ReadRegOperation
/// @brief Implements ProcessMonitor::ReadRegisterValue.
class ReadRegOperation : public Operation
{
public:
    ReadRegOperation(unsigned offset, RegisterValue &value, bool &result)
        : m_offset(offset), m_value(value), m_result(result)
        { }

    void Execute(ProcessMonitor *monitor);

private:
    unsigned m_offset;
    RegisterValue &m_value;
    bool &m_result;
};

void
ReadRegOperation::Execute(ProcessMonitor *monitor)
{
    lldb::pid_t pid = monitor->GetPID();
    LogSP log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_REGISTERS));

    // Set errno to zero so that we can detect a failed peek.
    errno = 0;
    lldb::addr_t data = PTRACE(PTRACE_PEEKUSER, pid, (void*)m_offset, NULL);
    if (data == -1UL && errno)
        m_result = false;
    else
    {
        m_value = data;
        m_result = true;
    }
    if (log)
        log->Printf ("ProcessMonitor::%s() reg %s: 0x%x", __FUNCTION__,
                     POSIXThread::GetRegisterNameFromOffset(m_offset), data);
}

//------------------------------------------------------------------------------
/// @class WriteRegOperation
/// @brief Implements ProcessMonitor::WriteRegisterValue.
class WriteRegOperation : public Operation
{
public:
    WriteRegOperation(unsigned offset, const RegisterValue &value, bool &result)
        : m_offset(offset), m_value(value), m_result(result)
        { }

    void Execute(ProcessMonitor *monitor);

private:
    unsigned m_offset;
    const RegisterValue &m_value;
    bool &m_result;
};

void
WriteRegOperation::Execute(ProcessMonitor *monitor)
{
    void* buf;
    lldb::pid_t pid = monitor->GetPID();
    LogSP log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_REGISTERS));

    if (sizeof(void*) == sizeof(uint64_t))
        buf = (void*) m_value.GetAsUInt64();
    else
    {
        assert(sizeof(void*) == sizeof(uint32_t));
        buf = (void*) m_value.GetAsUInt32();
    }

    if (log)
        log->Printf ("ProcessMonitor::%s() reg %s: %p", __FUNCTION__,
                     POSIXThread::GetRegisterNameFromOffset(m_offset), buf);
    if (PTRACE(PTRACE_POKEUSER, pid, (void*)m_offset, buf))
        m_result = false;
    else
        m_result = true;
}

//------------------------------------------------------------------------------
/// @class ReadGPROperation
/// @brief Implements ProcessMonitor::ReadGPR.
class ReadGPROperation : public Operation
{
public:
    ReadGPROperation(void *buf, bool &result)
        : m_buf(buf), m_result(result)
        { }

    void Execute(ProcessMonitor *monitor);

private:
    void *m_buf;
    bool &m_result;
};

void
ReadGPROperation::Execute(ProcessMonitor *monitor)
{
    if (PTRACE(PTRACE_GETREGS, monitor->GetPID(), NULL, m_buf) < 0)
        m_result = false;
    else
        m_result = true;
}

//------------------------------------------------------------------------------
/// @class ReadFPROperation
/// @brief Implements ProcessMonitor::ReadFPR.
class ReadFPROperation : public Operation
{
public:
    ReadFPROperation(void *buf, bool &result)
        : m_buf(buf), m_result(result)
        { }

    void Execute(ProcessMonitor *monitor);

private:
    void *m_buf;
    bool &m_result;
};

void
ReadFPROperation::Execute(ProcessMonitor *monitor)
{
    if (PTRACE(PTRACE_GETFPREGS, monitor->GetPID(), NULL, m_buf) < 0)
        m_result = false;
    else
        m_result = true;
}

//------------------------------------------------------------------------------
/// @class WriteGPROperation
/// @brief Implements ProcessMonitor::WriteGPR.
class WriteGPROperation : public Operation
{
public:
    WriteGPROperation(void *buf, bool &result)
        : m_buf(buf), m_result(result)
        { }

    void Execute(ProcessMonitor *monitor);

private:
    void *m_buf;
    bool &m_result;
};

void
WriteGPROperation::Execute(ProcessMonitor *monitor)
{
    if (PTRACE(PTRACE_SETREGS, monitor->GetPID(), NULL, m_buf) < 0)
        m_result = false;
    else
        m_result = true;
}

//------------------------------------------------------------------------------
/// @class WriteFPROperation
/// @brief Implements ProcessMonitor::WriteFPR.
class WriteFPROperation : public Operation
{
public:
    WriteFPROperation(void *buf, bool &result)
        : m_buf(buf), m_result(result)
        { }

    void Execute(ProcessMonitor *monitor);

private:
    void *m_buf;
    bool &m_result;
};

void
WriteFPROperation::Execute(ProcessMonitor *monitor)
{
    if (PTRACE(PTRACE_SETFPREGS, monitor->GetPID(), NULL, m_buf) < 0)
        m_result = false;
    else
        m_result = true;
}

//------------------------------------------------------------------------------
/// @class ResumeOperation
/// @brief Implements ProcessMonitor::Resume.
class ResumeOperation : public Operation
{
public:
    ResumeOperation(lldb::tid_t tid, uint32_t signo, bool &result) :
        m_tid(tid), m_signo(signo), m_result(result) { }

    void Execute(ProcessMonitor *monitor);

private:
    lldb::tid_t m_tid;
    uint32_t m_signo;
    bool &m_result;
};

void
ResumeOperation::Execute(ProcessMonitor *monitor)
{
    int data = 0;

    if (m_signo != LLDB_INVALID_SIGNAL_NUMBER)
        data = m_signo;

    if (PTRACE(PTRACE_CONT, m_tid, NULL, (void*)data))
        m_result = false;
    else
        m_result = true;
}

//------------------------------------------------------------------------------
/// @class ResumeOperation
/// @brief Implements ProcessMonitor::SingleStep.
class SingleStepOperation : public Operation
{
public:
    SingleStepOperation(lldb::tid_t tid, uint32_t signo, bool &result)
        : m_tid(tid), m_signo(signo), m_result(result) { }

    void Execute(ProcessMonitor *monitor);

private:
    lldb::tid_t m_tid;
    uint32_t m_signo;
    bool &m_result;
};

void
SingleStepOperation::Execute(ProcessMonitor *monitor)
{
    int data = 0;

    if (m_signo != LLDB_INVALID_SIGNAL_NUMBER)
        data = m_signo;

    if (PTRACE(PTRACE_SINGLESTEP, m_tid, NULL, (void*)data))
        m_result = false;
    else
        m_result = true;
}

//------------------------------------------------------------------------------
/// @class SiginfoOperation
/// @brief Implements ProcessMonitor::GetSignalInfo.
class SiginfoOperation : public Operation
{
public:
    SiginfoOperation(lldb::tid_t tid, void *info, bool &result)
        : m_tid(tid), m_info(info), m_result(result) { }

    void Execute(ProcessMonitor *monitor);

private:
    lldb::tid_t m_tid;
    void *m_info;
    bool &m_result;
};

void
SiginfoOperation::Execute(ProcessMonitor *monitor)
{
    if (PTRACE(PTRACE_GETSIGINFO, m_tid, NULL, m_info))
        m_result = false;
    else
        m_result = true;
}

//------------------------------------------------------------------------------
/// @class EventMessageOperation
/// @brief Implements ProcessMonitor::GetEventMessage.
class EventMessageOperation : public Operation
{
public:
    EventMessageOperation(lldb::tid_t tid, unsigned long *message, bool &result)
        : m_tid(tid), m_message(message), m_result(result) { }

    void Execute(ProcessMonitor *monitor);

private:
    lldb::tid_t m_tid;
    unsigned long *m_message;
    bool &m_result;
};

void
EventMessageOperation::Execute(ProcessMonitor *monitor)
{
    if (PTRACE(PTRACE_GETEVENTMSG, m_tid, NULL, m_message))
        m_result = false;
    else
        m_result = true;
}

//------------------------------------------------------------------------------
/// @class KillOperation
/// @brief Implements ProcessMonitor::BringProcessIntoLimbo.
class KillOperation : public Operation
{
public:
    KillOperation(bool &result) : m_result(result) { }

    void Execute(ProcessMonitor *monitor);

private:
    bool &m_result;
};

void
KillOperation::Execute(ProcessMonitor *monitor)
{
    lldb::pid_t pid = monitor->GetPID();

    if (PTRACE(PTRACE_KILL, pid, NULL, NULL))
        m_result = false;
    else
        m_result = true;
}

//------------------------------------------------------------------------------
/// @class KillOperation
/// @brief Implements ProcessMonitor::BringProcessIntoLimbo.
class DetachOperation : public Operation
{
public:
    DetachOperation(Error &result) : m_error(result) { }

    void Execute(ProcessMonitor *monitor);

private:
    Error &m_error;
};

void
DetachOperation::Execute(ProcessMonitor *monitor)
{
    lldb::pid_t pid = monitor->GetPID();

    if (ptrace(PT_DETACH, pid, NULL, 0) < 0)
        m_error.SetErrorToErrno();

}

ProcessMonitor::OperationArgs::OperationArgs(ProcessMonitor *monitor)
    : m_monitor(monitor)
{
    sem_init(&m_semaphore, 0, 0);
}

ProcessMonitor::OperationArgs::~OperationArgs()
{
    sem_destroy(&m_semaphore);
}

ProcessMonitor::LaunchArgs::LaunchArgs(ProcessMonitor *monitor,
                                       lldb_private::Module *module,
                                       char const **argv,
                                       char const **envp,
                                       const char *stdin_path,
                                       const char *stdout_path,
                                       const char *stderr_path)
    : OperationArgs(monitor),
      m_module(module),
      m_argv(argv),
      m_envp(envp),
      m_stdin_path(stdin_path),
      m_stdout_path(stdout_path),
      m_stderr_path(stderr_path) { }

ProcessMonitor::LaunchArgs::~LaunchArgs()
{ }

ProcessMonitor::AttachArgs::AttachArgs(ProcessMonitor *monitor,
                                       lldb::pid_t pid)
    : OperationArgs(monitor), m_pid(pid) { }

ProcessMonitor::AttachArgs::~AttachArgs()
{ }

//------------------------------------------------------------------------------
/// The basic design of the ProcessMonitor is built around two threads.
///
/// One thread (@see SignalThread) simply blocks on a call to waitpid() looking
/// for changes in the debugee state.  When a change is detected a
/// ProcessMessage is sent to the associated ProcessLinux instance.  This thread
/// "drives" state changes in the debugger.
///
/// The second thread (@see OperationThread) is responsible for two things 1)
/// launching or attaching to the inferior process, and then 2) servicing
/// operations such as register reads/writes, stepping, etc.  See the comments
/// on the Operation class for more info as to why this is needed.
ProcessMonitor::ProcessMonitor(ProcessPOSIX *process,
                               Module *module,
                               const char *argv[],
                               const char *envp[],
                               const char *stdin_path,
                               const char *stdout_path,
                               const char *stderr_path,
                               lldb_private::Error &error)
    : m_process(static_cast<ProcessLinux *>(process)),
      m_operation_thread(LLDB_INVALID_HOST_THREAD),
      m_pid(LLDB_INVALID_PROCESS_ID),
      m_terminal_fd(-1),
      m_monitor_thread(LLDB_INVALID_HOST_THREAD),
      m_client_fd(-1),
      m_server_fd(-1)
{
    std::auto_ptr<LaunchArgs> args;

    args.reset(new LaunchArgs(this, module, argv, envp,
                              stdin_path, stdout_path, stderr_path));

    // Server/client descriptors.
    if (!EnableIPC())
    {
        error.SetErrorToGenericError();
        error.SetErrorString("Monitor failed to initialize.");
    }

    StartLaunchOpThread(args.get(), error);
    if (!error.Success())
        return;

WAIT_AGAIN:
    // Wait for the operation thread to initialize.
    if (sem_wait(&args->m_semaphore))
    {
        if (errno == EINTR)
            goto WAIT_AGAIN;
        else
        {
            error.SetErrorToErrno();
            return;
        }
    }

    // Check that the launch was a success.
    if (!args->m_error.Success())
    {
        StopOpThread();
        error = args->m_error;
        return;
    }

    // Finally, start monitoring the child process for change in state.
    m_monitor_thread = Host::StartMonitoringChildProcess(
        ProcessMonitor::MonitorCallback, this, GetPID(), true);
    if (!IS_VALID_LLDB_HOST_THREAD(m_monitor_thread))
    {
        error.SetErrorToGenericError();
        error.SetErrorString("Process launch failed.");
        return;
    }
}

ProcessMonitor::ProcessMonitor(ProcessPOSIX *process,
                               lldb::pid_t pid,
                               lldb_private::Error &error)
  : m_process(static_cast<ProcessLinux *>(process)),
      m_operation_thread(LLDB_INVALID_HOST_THREAD),
      m_pid(LLDB_INVALID_PROCESS_ID),
      m_terminal_fd(-1),
      m_monitor_thread(LLDB_INVALID_HOST_THREAD),
      m_client_fd(-1),
      m_server_fd(-1)
{
    std::auto_ptr<AttachArgs> args;

    args.reset(new AttachArgs(this, pid));

    // Server/client descriptors.
    if (!EnableIPC())
    {
        error.SetErrorToGenericError();
        error.SetErrorString("Monitor failed to initialize.");
    }

    StartAttachOpThread(args.get(), error);
    if (!error.Success())
        return;

WAIT_AGAIN:
    // Wait for the operation thread to initialize.
    if (sem_wait(&args->m_semaphore))
    {
        if (errno == EINTR)
            goto WAIT_AGAIN;
        else
        {
            error.SetErrorToErrno();
            return;
        }
    }

    // Check that the attach was a success.
    if (!args->m_error.Success())
    {
        StopOpThread();
        error = args->m_error;
        return;
    }

    // Finally, start monitoring the child process for change in state.
    m_monitor_thread = Host::StartMonitoringChildProcess(
        ProcessMonitor::MonitorCallback, this, GetPID(), true);
    if (!IS_VALID_LLDB_HOST_THREAD(m_monitor_thread))
    {
        error.SetErrorToGenericError();
        error.SetErrorString("Process attach failed.");
        return;
    }
}

ProcessMonitor::~ProcessMonitor()
{
    StopMonitor();
}

//------------------------------------------------------------------------------
// Thread setup and tear down.
void
ProcessMonitor::StartLaunchOpThread(LaunchArgs *args, Error &error)
{
    static const char *g_thread_name = "lldb.process.linux.operation";

    if (IS_VALID_LLDB_HOST_THREAD(m_operation_thread))
        return;

    m_operation_thread =
        Host::ThreadCreate(g_thread_name, LaunchOpThread, args, &error);
}

void *
ProcessMonitor::LaunchOpThread(void *arg)
{
    LaunchArgs *args = static_cast<LaunchArgs*>(arg);

    if (!Launch(args)) {
        sem_post(&args->m_semaphore);
        return NULL;
    }

    ServeOperation(args);
    return NULL;
}

bool
ProcessMonitor::Launch(LaunchArgs *args)
{
    ProcessMonitor *monitor = args->m_monitor;
    ProcessLinux &process = monitor->GetProcess();
    const char **argv = args->m_argv;
    const char **envp = args->m_envp;
    const char *stdin_path = args->m_stdin_path;
    const char *stdout_path = args->m_stdout_path;
    const char *stderr_path = args->m_stderr_path;

    lldb_utility::PseudoTerminal terminal;
    const size_t err_len = 1024;
    char err_str[err_len];
    lldb::pid_t pid;

    lldb::ThreadSP inferior;
    LogSP log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_PROCESS));

    // Propagate the environment if one is not supplied.
    if (envp == NULL || envp[0] == NULL)
        envp = const_cast<const char **>(environ);

    // Pseudo terminal setup.
    if (!terminal.OpenFirstAvailableMaster(O_RDWR | O_NOCTTY, err_str, err_len))
    {
        args->m_error.SetErrorToGenericError();
        args->m_error.SetErrorString("Could not open controlling TTY.");
        goto FINISH;
    }

    if ((pid = terminal.Fork(err_str, err_len)) < 0)
    {
        args->m_error.SetErrorToGenericError();
        args->m_error.SetErrorString("Process fork failed.");
        goto FINISH;
    }

    // Recognized child exit status codes.
    enum {
        ePtraceFailed = 1,
        eDupStdinFailed,
        eDupStdoutFailed,
        eDupStderrFailed,
        eExecFailed
    };

    // Child process.
    if (pid == 0)
    {
        // Trace this process.
        if (PTRACE(PTRACE_TRACEME, 0, NULL, NULL) < 0)
            exit(ePtraceFailed);

        // Do not inherit setgid powers.
        setgid(getgid());

        // Let us have our own process group.
        setpgid(0, 0);

        // Dup file descriptors if needed.
        //
        // FIXME: If two or more of the paths are the same we needlessly open
        // the same file multiple times.
        if (stdin_path != NULL && stdin_path[0])
            if (!DupDescriptor(stdin_path, STDIN_FILENO, O_RDONLY))
                exit(eDupStdinFailed);

        if (stdout_path != NULL && stdout_path[0])
            if (!DupDescriptor(stdout_path, STDOUT_FILENO, O_WRONLY | O_CREAT))
                exit(eDupStdoutFailed);

        if (stderr_path != NULL && stderr_path[0])
            if (!DupDescriptor(stderr_path, STDERR_FILENO, O_WRONLY | O_CREAT))
                exit(eDupStderrFailed);

        // Execute.  We should never return.
        execve(argv[0],
               const_cast<char *const *>(argv),
               const_cast<char *const *>(envp));
        exit(eExecFailed);
    }

    // Wait for the child process to to trap on its call to execve.
    pid_t wpid;
    int status;
    if ((wpid = waitpid(pid, &status, 0)) < 0)
    {
        args->m_error.SetErrorToErrno();
        goto FINISH;
    }
    else if (WIFEXITED(status))
    {
        // open, dup or execve likely failed for some reason.
        args->m_error.SetErrorToGenericError();
        switch (WEXITSTATUS(status))
        {
            case ePtraceFailed:
                args->m_error.SetErrorString("Child ptrace failed.");
                break;
            case eDupStdinFailed:
                args->m_error.SetErrorString("Child open stdin failed.");
                break;
            case eDupStdoutFailed:
                args->m_error.SetErrorString("Child open stdout failed.");
                break;
            case eDupStderrFailed:
                args->m_error.SetErrorString("Child open stderr failed.");
                break;
            case eExecFailed:
                args->m_error.SetErrorString("Child exec failed.");
                break;
            default:
                args->m_error.SetErrorString("Child returned unknown exit status.");
                break;
        }
        goto FINISH;
    }
    assert(WIFSTOPPED(status) && wpid == pid &&
           "Could not sync with inferior process.");

    // Have the child raise an event on exit.  This is used to keep the child in
    // limbo until it is destroyed.
    if (PTRACE(PTRACE_SETOPTIONS, pid, NULL, (void*)PTRACE_O_TRACEEXIT) < 0)
    {
        args->m_error.SetErrorToErrno();
        goto FINISH;
    }

    // Release the master terminal descriptor and pass it off to the
    // ProcessMonitor instance.  Similarly stash the inferior pid.
    monitor->m_terminal_fd = terminal.ReleaseMasterFileDescriptor();
    monitor->m_pid = pid;

    // Set the terminal fd to be in non blocking mode (it simplifies the
    // implementation of ProcessLinux::GetSTDOUT to have a non-blocking
    // descriptor to read from).
    if (!EnsureFDFlags(monitor->m_terminal_fd, O_NONBLOCK, args->m_error))
        goto FINISH;

    // Update the process thread list with this new thread.
    // FIXME: should we be letting UpdateThreadList handle this?
    // FIXME: by using pids instead of tids, we can only support one thread.
    inferior.reset(new POSIXThread(process, pid));
    if (log)
        log->Printf ("ProcessMonitor::%s() adding pid = %i", __FUNCTION__, pid);
    process.GetThreadList().AddThread(inferior);

    // Let our process instance know the thread has stopped.
    process.SendMessage(ProcessMessage::Trace(pid));

FINISH:
    return args->m_error.Success();
}

bool
ProcessMonitor::EnableIPC()
{
    int fd[2];

    if (socketpair(AF_UNIX, SOCK_STREAM, 0, fd))
        return false;

    m_client_fd = fd[0];
    m_server_fd = fd[1];
    return true;
}

void
ProcessMonitor::StartAttachOpThread(AttachArgs *args, lldb_private::Error &error)
{
    static const char *g_thread_name = "lldb.process.linux.operation";

    if (IS_VALID_LLDB_HOST_THREAD(m_operation_thread))
        return;

    m_operation_thread =
        Host::ThreadCreate(g_thread_name, AttachOpThread, args, &error);
}

void *
ProcessMonitor::AttachOpThread(void *arg)
{
    AttachArgs *args = static_cast<AttachArgs*>(arg);

    if (!Attach(args)) {
        sem_post(&args->m_semaphore);
        return NULL;
    }

    ServeOperation(args);
    return NULL;
}

bool
ProcessMonitor::Attach(AttachArgs *args)
{
    lldb::pid_t pid = args->m_pid;

    ProcessMonitor *monitor = args->m_monitor;
    ProcessLinux &process = monitor->GetProcess();
    lldb::ThreadSP inferior;
    LogSP log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_PROCESS));

    if (pid <= 1)
    {
        args->m_error.SetErrorToGenericError();
        args->m_error.SetErrorString("Attaching to process 1 is not allowed.");
        goto FINISH;
    }

    // Attach to the requested process.
    if (PTRACE(PTRACE_ATTACH, pid, NULL, NULL) < 0)
    {
        args->m_error.SetErrorToErrno();
        goto FINISH;
    }

    int status;
    if ((status = waitpid(pid, NULL, 0)) < 0)
    {
        args->m_error.SetErrorToErrno();
        goto FINISH;
    }

    monitor->m_pid = pid;

    // Update the process thread list with the attached thread.
    inferior.reset(new POSIXThread(process, pid));
    if (log)
        log->Printf ("ProcessMonitor::%s() adding tid = %i", __FUNCTION__, pid);
    process.GetThreadList().AddThread(inferior);

    // Let our process instance know the thread has stopped.
    process.SendMessage(ProcessMessage::Trace(pid));

 FINISH:
    return args->m_error.Success();
}

bool
ProcessMonitor::MonitorCallback(void *callback_baton,
                                lldb::pid_t pid,
                                bool exited,
                                int signal,
                                int status)
{
    ProcessMessage message;
    ProcessMonitor *monitor = static_cast<ProcessMonitor*>(callback_baton);
    ProcessLinux *process = monitor->m_process;
    assert(process);
    bool stop_monitoring;
    siginfo_t info;

    if (!monitor->GetSignalInfo(pid, &info))
        stop_monitoring = true; // pid is gone.  Bail.
    else {
        switch (info.si_signo)
        {
        case SIGTRAP:
            message = MonitorSIGTRAP(monitor, &info, pid);
            break;

        default:
            message = MonitorSignal(monitor, &info, pid);
            break;
        }

        process->SendMessage(message);
        stop_monitoring = message.GetKind() == ProcessMessage::eExitMessage;
    }

    return stop_monitoring;
}

ProcessMessage
ProcessMonitor::MonitorSIGTRAP(ProcessMonitor *monitor,
                               const siginfo_t *info, lldb::pid_t pid)
{
    ProcessMessage message;

    assert(monitor);
    assert(info && info->si_signo == SIGTRAP && "Unexpected child signal!");

    switch (info->si_code)
    {
    default:
        assert(false && "Unexpected SIGTRAP code!");
        break;

    case (SIGTRAP | (PTRACE_EVENT_EXIT << 8)):
    {
        // The inferior process is about to exit.  Maintain the process in a
        // state of "limbo" until we are explicitly commanded to detach,
        // destroy, resume, etc.
        unsigned long data = 0;
        if (!monitor->GetEventMessage(pid, &data))
            data = -1;
        message = ProcessMessage::Limbo(pid, (data >> 8));
        break;
    }

    case 0:
    case TRAP_TRACE:
        message = ProcessMessage::Trace(pid);
        break;

    case SI_KERNEL:
    case TRAP_BRKPT:
        message = ProcessMessage::Break(pid);
        break;
    }

    return message;
}

ProcessMessage
ProcessMonitor::MonitorSignal(ProcessMonitor *monitor,
                              const siginfo_t *info, lldb::pid_t pid)
{
    ProcessMessage message;
    int signo = info->si_signo;

    // POSIX says that process behaviour is undefined after it ignores a SIGFPE,
    // SIGILL, SIGSEGV, or SIGBUS *unless* that signal was generated by a
    // kill(2) or raise(3).  Similarly for tgkill(2) on Linux.
    //
    // IOW, user generated signals never generate what we consider to be a
    // "crash".
    //
    // Similarly, ACK signals generated by this monitor.
    if (info->si_code == SI_TKILL || info->si_code == SI_USER)
    {
        if (info->si_pid == getpid())
            return ProcessMessage::SignalDelivered(pid, signo);
        else
            return ProcessMessage::Signal(pid, signo);
    }

    if (signo == SIGSEGV) {
        lldb::addr_t fault_addr = reinterpret_cast<lldb::addr_t>(info->si_addr);
        ProcessMessage::CrashReason reason = GetCrashReasonForSIGSEGV(info);
        return ProcessMessage::Crash(pid, reason, signo, fault_addr);
    }

    if (signo == SIGILL) {
        lldb::addr_t fault_addr = reinterpret_cast<lldb::addr_t>(info->si_addr);
        ProcessMessage::CrashReason reason = GetCrashReasonForSIGILL(info);
        return ProcessMessage::Crash(pid, reason, signo, fault_addr);
    }

    if (signo == SIGFPE) {
        lldb::addr_t fault_addr = reinterpret_cast<lldb::addr_t>(info->si_addr);
        ProcessMessage::CrashReason reason = GetCrashReasonForSIGFPE(info);
        return ProcessMessage::Crash(pid, reason, signo, fault_addr);
    }

    if (signo == SIGBUS) {
        lldb::addr_t fault_addr = reinterpret_cast<lldb::addr_t>(info->si_addr);
        ProcessMessage::CrashReason reason = GetCrashReasonForSIGBUS(info);
        return ProcessMessage::Crash(pid, reason, signo, fault_addr);
    }

    // Everything else is "normal" and does not require any special action on
    // our part.
    return ProcessMessage::Signal(pid, signo);
}

ProcessMessage::CrashReason
ProcessMonitor::GetCrashReasonForSIGSEGV(const siginfo_t *info)
{
    ProcessMessage::CrashReason reason;
    assert(info->si_signo == SIGSEGV);

    reason = ProcessMessage::eInvalidCrashReason;

    switch (info->si_code)
    {
    default:
        assert(false && "unexpected si_code for SIGSEGV");
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

ProcessMessage::CrashReason
ProcessMonitor::GetCrashReasonForSIGILL(const siginfo_t *info)
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

ProcessMessage::CrashReason
ProcessMonitor::GetCrashReasonForSIGFPE(const siginfo_t *info)
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

ProcessMessage::CrashReason
ProcessMonitor::GetCrashReasonForSIGBUS(const siginfo_t *info)
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

void
ProcessMonitor::ServeOperation(OperationArgs *args)
{
    int status;
    pollfd fdset;

    ProcessMonitor *monitor = args->m_monitor;

    fdset.fd = monitor->m_server_fd;
    fdset.events = POLLIN | POLLPRI;
    fdset.revents = 0;

    // We are finised with the arguments and are ready to go.  Sync with the
    // parent thread and start serving operations on the inferior.
    sem_post(&args->m_semaphore);

    for (;;)
    {
        if ((status = poll(&fdset, 1, -1)) < 0)
        {
            switch (errno)
            {
            default:
                assert(false && "Unexpected poll() failure!");
                continue;

            case EINTR: continue; // Just poll again.
            case EBADF: return;   // Connection terminated.
            }
        }

        assert(status == 1 && "Too many descriptors!");

        if (fdset.revents & POLLIN)
        {
            Operation *op = NULL;

        READ_AGAIN:
            if ((status = read(fdset.fd, &op, sizeof(op))) < 0)
            {
                // There is only one acceptable failure.
                assert(errno == EINTR);
                goto READ_AGAIN;
            }

            assert(status == sizeof(op));
            op->Execute(monitor);
            write(fdset.fd, &op, sizeof(op));
        }
    }
}

void
ProcessMonitor::DoOperation(Operation *op)
{
    int status;
    Operation *ack = NULL;
    Mutex::Locker lock(m_server_mutex);

    // FIXME: Do proper error checking here.
    write(m_client_fd, &op, sizeof(op));

READ_AGAIN:
    if ((status = read(m_client_fd, &ack, sizeof(ack))) < 0)
    {
        // If interrupted by a signal handler try again.  Otherwise the monitor
        // thread probably died and we have a stale file descriptor -- abort the
        // operation.
        if (errno == EINTR)
            goto READ_AGAIN;
        return;
    }

    assert(status == sizeof(ack));
    assert(ack == op && "Invalid monitor thread response!");
}

size_t
ProcessMonitor::ReadMemory(lldb::addr_t vm_addr, void *buf, size_t size,
                           Error &error)
{
    size_t result;
    ReadOperation op(vm_addr, buf, size, error, result);
    DoOperation(&op);
    return result;
}

size_t
ProcessMonitor::WriteMemory(lldb::addr_t vm_addr, const void *buf, size_t size,
                            lldb_private::Error &error)
{
    size_t result;
    WriteOperation op(vm_addr, buf, size, error, result);
    DoOperation(&op);
    return result;
}

bool
ProcessMonitor::ReadRegisterValue(unsigned offset, unsigned size, RegisterValue &value)
{
    bool result;
    ReadRegOperation op(offset, value, result);
    DoOperation(&op);
    return result;
}

bool
ProcessMonitor::WriteRegisterValue(unsigned offset, const RegisterValue &value)
{
    bool result;
    WriteRegOperation op(offset, value, result);
    DoOperation(&op);
    return result;
}

bool
ProcessMonitor::ReadGPR(void *buf)
{
    bool result;
    ReadGPROperation op(buf, result);
    DoOperation(&op);
    return result;
}

bool
ProcessMonitor::ReadFPR(void *buf)
{
    bool result;
    ReadFPROperation op(buf, result);
    DoOperation(&op);
    return result;
}

bool
ProcessMonitor::WriteGPR(void *buf)
{
    bool result;
    WriteGPROperation op(buf, result);
    DoOperation(&op);
    return result;
}

bool
ProcessMonitor::WriteFPR(void *buf)
{
    bool result;
    WriteFPROperation op(buf, result);
    DoOperation(&op);
    return result;
}

bool
ProcessMonitor::Resume(lldb::tid_t tid, uint32_t signo)
{
    bool result;
    ResumeOperation op(tid, signo, result);
    DoOperation(&op);
    return result;
}

bool
ProcessMonitor::SingleStep(lldb::tid_t tid, uint32_t signo)
{
    bool result;
    SingleStepOperation op(tid, signo, result);
    DoOperation(&op);
    return result;
}

bool
ProcessMonitor::BringProcessIntoLimbo()
{
    bool result;
    KillOperation op(result);
    DoOperation(&op);
    return result;
}

bool
ProcessMonitor::GetSignalInfo(lldb::tid_t tid, void *siginfo)
{
    bool result;
    SiginfoOperation op(tid, siginfo, result);
    DoOperation(&op);
    return result;
}

bool
ProcessMonitor::GetEventMessage(lldb::tid_t tid, unsigned long *message)
{
    bool result;
    EventMessageOperation op(tid, message, result);
    DoOperation(&op);
    return result;
}

lldb_private::Error
ProcessMonitor::Detach()
{
    lldb_private::Error error;
    if (m_pid != LLDB_INVALID_PROCESS_ID) {
        DetachOperation op(error);
        DoOperation(&op);
    }
    StopMonitor();
    return error;
}

bool
ProcessMonitor::DupDescriptor(const char *path, int fd, int flags)
{
    int target_fd = open(path, flags, 0666);

    if (target_fd == -1)
        return false;

    return (dup2(target_fd, fd) == -1) ? false : true;
}

void
ProcessMonitor::StopMonitoringChildProcess()
{
    lldb::thread_result_t thread_result;

    if (IS_VALID_LLDB_HOST_THREAD(m_monitor_thread))
    {
        Host::ThreadCancel(m_monitor_thread, NULL);
        Host::ThreadJoin(m_monitor_thread, &thread_result, NULL);
        m_monitor_thread = LLDB_INVALID_HOST_THREAD;
    }
}

void
ProcessMonitor::StopMonitor()
{
    StopMonitoringChildProcess();
    StopOpThread();
    CloseFD(m_terminal_fd);
    CloseFD(m_client_fd);
    CloseFD(m_server_fd);
}

void
ProcessMonitor::StopOpThread()
{
    lldb::thread_result_t result;

    if (!IS_VALID_LLDB_HOST_THREAD(m_operation_thread))
        return;

    Host::ThreadCancel(m_operation_thread, NULL);
    Host::ThreadJoin(m_operation_thread, &result, NULL);
}

void
ProcessMonitor::CloseFD(int &fd)
{
    if (fd != -1)
    {
        close(fd);
        fd = -1;
    }
}
