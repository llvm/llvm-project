//===-- MachTask.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//----------------------------------------------------------------------
//
//  MachTask.cpp
//  debugserver
//
//  Created by Greg Clayton on 12/5/08.
//
//===----------------------------------------------------------------------===//

#include "MachTask.h"

// C Includes

#include <mach-o/dyld_images.h>
#include <mach/mach_vm.h>
#import <sys/sysctl.h>

// C++ Includes
#include <iomanip>
#include <sstream>

// Other libraries and framework includes
// Project includes
#include "CFUtils.h"
#include "DNB.h"
#include "DNBError.h"
#include "DNBLog.h"
#include "MachProcess.h"
#include "DNBDataRef.h"
#include "stack_logging.h"

#ifdef WITH_SPRINGBOARD

#include <CoreFoundation/CoreFoundation.h>
#include <SpringBoardServices/SpringBoardServer.h>
#include <SpringBoardServices/SBSWatchdogAssertion.h>

#endif

//----------------------------------------------------------------------
// MachTask constructor
//----------------------------------------------------------------------
MachTask::MachTask(MachProcess *process) :
    m_process (process),
    m_task (TASK_NULL),
    m_vm_memory (),
    m_exception_thread (0),
    m_exception_port (MACH_PORT_NULL)
{
    memset(&m_exc_port_info, 0, sizeof(m_exc_port_info));
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
MachTask::~MachTask()
{
    Clear();
}


//----------------------------------------------------------------------
// MachTask::Suspend
//----------------------------------------------------------------------
kern_return_t
MachTask::Suspend()
{
    DNBError err;
    task_t task = TaskPort();
    err = ::task_suspend (task);
    if (DNBLogCheckLogBit(LOG_TASK) || err.Fail())
        err.LogThreaded("::task_suspend ( target_task = 0x%4.4x )", task);
    return err.Error();
}


//----------------------------------------------------------------------
// MachTask::Resume
//----------------------------------------------------------------------
kern_return_t
MachTask::Resume()
{
    struct task_basic_info task_info;
    task_t task = TaskPort();
    if (task == TASK_NULL)
        return KERN_INVALID_ARGUMENT;

    DNBError err;
    err = BasicInfo(task, &task_info);

    if (err.Success())
    {
        // task_resume isn't counted like task_suspend calls are, are, so if the 
        // task is not suspended, don't try and resume it since it is already 
        // running
        if (task_info.suspend_count > 0)
        {
            err = ::task_resume (task);
            if (DNBLogCheckLogBit(LOG_TASK) || err.Fail())
                err.LogThreaded("::task_resume ( target_task = 0x%4.4x )", task);
        }
    }
    return err.Error();
}

//----------------------------------------------------------------------
// MachTask::ExceptionPort
//----------------------------------------------------------------------
mach_port_t
MachTask::ExceptionPort() const
{
    return m_exception_port;
}

//----------------------------------------------------------------------
// MachTask::ExceptionPortIsValid
//----------------------------------------------------------------------
bool
MachTask::ExceptionPortIsValid() const
{
    return MACH_PORT_VALID(m_exception_port);
}


//----------------------------------------------------------------------
// MachTask::Clear
//----------------------------------------------------------------------
void
MachTask::Clear()
{
    // Do any cleanup needed for this task
    m_task = TASK_NULL;
    m_exception_thread = 0;
    m_exception_port = MACH_PORT_NULL;

}


//----------------------------------------------------------------------
// MachTask::SaveExceptionPortInfo
//----------------------------------------------------------------------
kern_return_t
MachTask::SaveExceptionPortInfo()
{
    return m_exc_port_info.Save(TaskPort());
}

//----------------------------------------------------------------------
// MachTask::RestoreExceptionPortInfo
//----------------------------------------------------------------------
kern_return_t
MachTask::RestoreExceptionPortInfo()
{
    return m_exc_port_info.Restore(TaskPort());
}


//----------------------------------------------------------------------
// MachTask::ReadMemory
//----------------------------------------------------------------------
nub_size_t
MachTask::ReadMemory (nub_addr_t addr, nub_size_t size, void *buf)
{
    nub_size_t n = 0;
    task_t task = TaskPort();
    if (task != TASK_NULL)
    {
        n = m_vm_memory.Read(task, addr, buf, size);

        DNBLogThreadedIf(LOG_MEMORY, "MachTask::ReadMemory ( addr = 0x%8.8llx, size = %llu, buf = %p) => %llu bytes read", (uint64_t)addr, (uint64_t)size, buf, (uint64_t)n);
        if (DNBLogCheckLogBit(LOG_MEMORY_DATA_LONG) || (DNBLogCheckLogBit(LOG_MEMORY_DATA_SHORT) && size <= 8))
        {
            DNBDataRef data((uint8_t*)buf, n, false);
            data.Dump(0, n, addr, DNBDataRef::TypeUInt8, 16);
        }
    }
    return n;
}


//----------------------------------------------------------------------
// MachTask::WriteMemory
//----------------------------------------------------------------------
nub_size_t
MachTask::WriteMemory (nub_addr_t addr, nub_size_t size, const void *buf)
{
    nub_size_t n = 0;
    task_t task = TaskPort();
    if (task != TASK_NULL)
    {
        n = m_vm_memory.Write(task, addr, buf, size);
        DNBLogThreadedIf(LOG_MEMORY, "MachTask::WriteMemory ( addr = 0x%8.8llx, size = %llu, buf = %p) => %llu bytes written", (uint64_t)addr, (uint64_t)size, buf, (uint64_t)n);
        if (DNBLogCheckLogBit(LOG_MEMORY_DATA_LONG) || (DNBLogCheckLogBit(LOG_MEMORY_DATA_SHORT) && size <= 8))
        {
            DNBDataRef data((uint8_t*)buf, n, false);
            data.Dump(0, n, addr, DNBDataRef::TypeUInt8, 16);
        }
    }
    return n;
}

//----------------------------------------------------------------------
// MachTask::MemoryRegionInfo
//----------------------------------------------------------------------
int
MachTask::GetMemoryRegionInfo (nub_addr_t addr, DNBRegionInfo *region_info)
{
    task_t task = TaskPort();
    if (task == TASK_NULL)
        return -1;

    int ret = m_vm_memory.GetMemoryRegionInfo(task, addr, region_info);
    DNBLogThreadedIf(LOG_MEMORY, "MachTask::MemoryRegionInfo ( addr = 0x%8.8llx ) => %i  (start = 0x%8.8llx, size = 0x%8.8llx, permissions = %u)",
                     (uint64_t)addr, 
                     ret,
                     (uint64_t)region_info->addr,
                     (uint64_t)region_info->size,
                     region_info->permissions);
    return ret;
}

#define TIME_VALUE_TO_TIMEVAL(a, r) do {        \
(r)->tv_sec = (a)->seconds;                     \
(r)->tv_usec = (a)->microseconds;               \
} while (0)

// We should consider moving this into each MacThread.
static void get_threads_profile_data(DNBProfileDataScanType scanType, task_t task, nub_process_t pid, std::vector<uint64_t> &threads_id, std::vector<std::string> &threads_name, std::vector<uint64_t> &threads_used_usec)
{
    kern_return_t kr;
    thread_act_array_t threads;
    mach_msg_type_number_t tcnt;
    
    kr = task_threads(task, &threads, &tcnt);
    if (kr != KERN_SUCCESS)
        return;
    
    for (int i = 0; i < tcnt; i++)
    {
        thread_identifier_info_data_t identifier_info;
        mach_msg_type_number_t count = THREAD_IDENTIFIER_INFO_COUNT;
        kr = ::thread_info(threads[i], THREAD_IDENTIFIER_INFO, (thread_info_t)&identifier_info, &count);
        if (kr != KERN_SUCCESS) continue;
        
        thread_basic_info_data_t basic_info;
        count = THREAD_BASIC_INFO_COUNT;
        kr = ::thread_info(threads[i], THREAD_BASIC_INFO, (thread_info_t)&basic_info, &count);
        if (kr != KERN_SUCCESS) continue;

        if ((basic_info.flags & TH_FLAGS_IDLE) == 0)
        {
            nub_thread_t tid = MachThread::GetGloballyUniqueThreadIDForMachPortID (threads[i]);
            threads_id.push_back(tid);
            
            if ((scanType & eProfileThreadName) && (identifier_info.thread_handle != 0))
            {
                struct proc_threadinfo proc_threadinfo;
                int len = ::proc_pidinfo(pid, PROC_PIDTHREADINFO, identifier_info.thread_handle, &proc_threadinfo, PROC_PIDTHREADINFO_SIZE);
                if (len && proc_threadinfo.pth_name[0])
                {
                    threads_name.push_back(proc_threadinfo.pth_name);
                }
                else
                {
                    threads_name.push_back("");
                }
            }
            else
            {
                threads_name.push_back("");
            }
            struct timeval tv;
            struct timeval thread_tv;
            TIME_VALUE_TO_TIMEVAL(&basic_info.user_time, &thread_tv);
            TIME_VALUE_TO_TIMEVAL(&basic_info.system_time, &tv);
            timeradd(&thread_tv, &tv, &thread_tv);
            uint64_t used_usec = thread_tv.tv_sec * 1000000ULL + thread_tv.tv_usec;
            threads_used_usec.push_back(used_usec);
        }
        
        kr = mach_port_deallocate(mach_task_self(), threads[i]);
    }
    kr = mach_vm_deallocate(mach_task_self(), (mach_vm_address_t)(uintptr_t)threads, tcnt * sizeof(*threads));
}

#define RAW_HEXBASE     std::setfill('0') << std::hex << std::right
#define DECIMAL         std::dec << std::setfill(' ')
std::string
MachTask::GetProfileData (DNBProfileDataScanType scanType)
{
    std::string result;
    
    static int32_t numCPU = -1;
    struct host_cpu_load_info host_info;
    if (scanType & eProfileHostCPU)
    {
        int32_t mib[] = {CTL_HW, HW_AVAILCPU};
        size_t len = sizeof(numCPU);
        if (numCPU == -1)
        {
            if (sysctl(mib, sizeof(mib) / sizeof(int32_t), &numCPU, &len, NULL, 0) != 0)
                return result;
        }
        
        mach_port_t localHost = mach_host_self();
        mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
        kern_return_t kr = host_statistics(localHost, HOST_CPU_LOAD_INFO, (host_info_t)&host_info, &count);
        if (kr != KERN_SUCCESS)
            return result;
    }
    
    task_t task = TaskPort();
    if (task == TASK_NULL)
        return result;
    
    struct task_basic_info task_info;
    DNBError err;
    err = BasicInfo(task, &task_info);
    
    if (!err.Success())
        return result;
    
    uint64_t elapsed_usec = 0;
    uint64_t task_used_usec = 0;
    if (scanType & eProfileCPU)
    {
        // Get current used time.
        struct timeval current_used_time;
        struct timeval tv;
        TIME_VALUE_TO_TIMEVAL(&task_info.user_time, &current_used_time);
        TIME_VALUE_TO_TIMEVAL(&task_info.system_time, &tv);
        timeradd(&current_used_time, &tv, &current_used_time);
        task_used_usec = current_used_time.tv_sec * 1000000ULL + current_used_time.tv_usec;
        
        struct timeval current_elapsed_time;
        int res = gettimeofday(&current_elapsed_time, NULL);
        if (res == 0)
        {
            elapsed_usec = current_elapsed_time.tv_sec * 1000000ULL + current_elapsed_time.tv_usec;
        }
    }
    
    std::vector<uint64_t> threads_id;
    std::vector<std::string> threads_name;
    std::vector<uint64_t> threads_used_usec;

    if (scanType & eProfileThreadsCPU)
    {
        get_threads_profile_data(scanType, task, m_process->ProcessID(), threads_id, threads_name, threads_used_usec);
    }
    
    struct vm_statistics vm_stats;
    uint64_t physical_memory;
    mach_vm_size_t rprvt = 0;
    mach_vm_size_t rsize = 0;
    mach_vm_size_t vprvt = 0;
    mach_vm_size_t vsize = 0;
    mach_vm_size_t dirty_size = 0;
    mach_vm_size_t purgeable = 0;
    mach_vm_size_t anonymous = 0;
    if (m_vm_memory.GetMemoryProfile(scanType, task, task_info, m_process->GetCPUType(), m_process->ProcessID(), vm_stats, physical_memory, rprvt, rsize, vprvt, vsize, dirty_size, purgeable, anonymous))
    {
        std::ostringstream profile_data_stream;
        
        if (scanType & eProfileHostCPU)
        {
            profile_data_stream << "num_cpu:" << numCPU << ';';
            profile_data_stream << "host_user_ticks:" << host_info.cpu_ticks[CPU_STATE_USER] << ';';
            profile_data_stream << "host_sys_ticks:" << host_info.cpu_ticks[CPU_STATE_SYSTEM] << ';';
            profile_data_stream << "host_idle_ticks:" << host_info.cpu_ticks[CPU_STATE_IDLE] << ';';
        }
        
        if (scanType & eProfileCPU)
        {
            profile_data_stream << "elapsed_usec:" << elapsed_usec << ';';
            profile_data_stream << "task_used_usec:" << task_used_usec << ';';
        }
        
        if (scanType & eProfileThreadsCPU)
        {
            int num_threads = threads_id.size();
            for (int i=0; i<num_threads; i++)
            {
                profile_data_stream << "thread_used_id:" << std::hex << threads_id[i] << std::dec << ';';
                profile_data_stream << "thread_used_usec:" << threads_used_usec[i] << ';';
                
                if (scanType & eProfileThreadName)
                {
                    profile_data_stream << "thread_used_name:";
                    int len = threads_name[i].size();
                    if (len)
                    {
                        const char *thread_name = threads_name[i].c_str();
                        // Make sure that thread name doesn't interfere with our delimiter.
                        profile_data_stream << RAW_HEXBASE << std::setw(2);
                        const uint8_t *ubuf8 = (const uint8_t *)(thread_name);
                        for (int j=0; j<len; j++)
                        {
                            profile_data_stream << (uint32_t)(ubuf8[j]);
                        }
                        // Reset back to DECIMAL.
                        profile_data_stream << DECIMAL;
                    }
                    profile_data_stream << ';';
                }
            }
        }
        
        if (scanType & eProfileHostMemory)
            profile_data_stream << "total:" << physical_memory << ';';
        
        if (scanType & eProfileMemory)
        {
            static vm_size_t pagesize;
            static bool calculated = false;
            if (!calculated)
            {
                calculated = true;
                pagesize = PageSize();
            }
            
            profile_data_stream << "wired:" << vm_stats.wire_count * pagesize << ';';
            profile_data_stream << "active:" << vm_stats.active_count * pagesize << ';';
            profile_data_stream << "inactive:" << vm_stats.inactive_count * pagesize << ';';
            uint64_t total_used_count = vm_stats.wire_count + vm_stats.inactive_count + vm_stats.active_count;
            profile_data_stream << "used:" << total_used_count * pagesize << ';';
            profile_data_stream << "free:" << vm_stats.free_count * pagesize << ';';
            
            profile_data_stream << "rprvt:" << rprvt << ';';
            profile_data_stream << "rsize:" << rsize << ';';
            profile_data_stream << "vprvt:" << vprvt << ';';
            profile_data_stream << "vsize:" << vsize << ';';
            
            if (scanType & eProfileMemoryDirtyPage)
                profile_data_stream << "dirty:" << dirty_size << ';';

            if (scanType & eProfileMemoryAnonymous)
            {
                profile_data_stream << "purgeable:" << purgeable << ';';
                profile_data_stream << "anonymous:" << anonymous << ';';
            }
        }
        
        profile_data_stream << "--end--;";
        
        result = profile_data_stream.str();
    }
    
    return result;
}


//----------------------------------------------------------------------
// MachTask::TaskPortForProcessID
//----------------------------------------------------------------------
task_t
MachTask::TaskPortForProcessID (DNBError &err)
{
    if (m_task == TASK_NULL && m_process != NULL)
        m_task = MachTask::TaskPortForProcessID(m_process->ProcessID(), err);
    return m_task;
}

//----------------------------------------------------------------------
// MachTask::TaskPortForProcessID
//----------------------------------------------------------------------
task_t
MachTask::TaskPortForProcessID (pid_t pid, DNBError &err, uint32_t num_retries, uint32_t usec_interval)
{
    if (pid != INVALID_NUB_PROCESS)
    {
        DNBError err;
        mach_port_t task_self = mach_task_self ();  
        task_t task = TASK_NULL;
        for (uint32_t i=0; i<num_retries; i++)
        {   
            err = ::task_for_pid ( task_self, pid, &task);

            if (DNBLogCheckLogBit(LOG_TASK) || err.Fail())
            {
                char str[1024];
                ::snprintf (str,
                            sizeof(str),
                            "::task_for_pid ( target_tport = 0x%4.4x, pid = %d, &task ) => err = 0x%8.8x (%s)",
                            task_self,
                            pid,
                            err.Error(),
                            err.AsString() ? err.AsString() : "success");
                if (err.Fail())
                    err.SetErrorString(str);
                err.LogThreaded(str);
            }

            if (err.Success())
                return task;

            // Sleep a bit and try again
            ::usleep (usec_interval);
        }
    }
    return TASK_NULL;
}


//----------------------------------------------------------------------
// MachTask::BasicInfo
//----------------------------------------------------------------------
kern_return_t
MachTask::BasicInfo(struct task_basic_info *info)
{
    return BasicInfo (TaskPort(), info);
}

//----------------------------------------------------------------------
// MachTask::BasicInfo
//----------------------------------------------------------------------
kern_return_t
MachTask::BasicInfo(task_t task, struct task_basic_info *info)
{
    if (info == NULL)
        return KERN_INVALID_ARGUMENT;

    DNBError err;
    mach_msg_type_number_t count = TASK_BASIC_INFO_COUNT;
    err = ::task_info (task, TASK_BASIC_INFO, (task_info_t)info, &count);
    const bool log_process = DNBLogCheckLogBit(LOG_TASK);
    if (log_process || err.Fail())
        err.LogThreaded("::task_info ( target_task = 0x%4.4x, flavor = TASK_BASIC_INFO, task_info_out => %p, task_info_outCnt => %u )", task, info, count);
    if (DNBLogCheckLogBit(LOG_TASK) && DNBLogCheckLogBit(LOG_VERBOSE) && err.Success())
    {
        float user = (float)info->user_time.seconds + (float)info->user_time.microseconds / 1000000.0f;
        float system = (float)info->user_time.seconds + (float)info->user_time.microseconds / 1000000.0f;
        DNBLogThreaded ("task_basic_info = { suspend_count = %i, virtual_size = 0x%8.8llx, resident_size = 0x%8.8llx, user_time = %f, system_time = %f }",
                        info->suspend_count, 
                        (uint64_t)info->virtual_size, 
                        (uint64_t)info->resident_size, 
                        user, 
                        system);
    }
    return err.Error();
}


//----------------------------------------------------------------------
// MachTask::IsValid
//
// Returns true if a task is a valid task port for a current process.
//----------------------------------------------------------------------
bool
MachTask::IsValid () const
{
    return MachTask::IsValid(TaskPort());
}

//----------------------------------------------------------------------
// MachTask::IsValid
//
// Returns true if a task is a valid task port for a current process.
//----------------------------------------------------------------------
bool
MachTask::IsValid (task_t task)
{
    if (task != TASK_NULL)
    {
        struct task_basic_info task_info;
        return BasicInfo(task, &task_info) == KERN_SUCCESS;
    }
    return false;
}


bool
MachTask::StartExceptionThread(DNBError &err)
{
    DNBLogThreadedIf(LOG_EXCEPTIONS, "MachTask::%s ( )", __FUNCTION__);
    task_t task = TaskPortForProcessID(err);
    if (MachTask::IsValid(task))
    {
        // Got the mach port for the current process
        mach_port_t task_self = mach_task_self ();

        // Allocate an exception port that we will use to track our child process
        err = ::mach_port_allocate (task_self, MACH_PORT_RIGHT_RECEIVE, &m_exception_port);
        if (err.Fail())
            return false;

        // Add the ability to send messages on the new exception port
        err = ::mach_port_insert_right (task_self, m_exception_port, m_exception_port, MACH_MSG_TYPE_MAKE_SEND);
        if (err.Fail())
            return false;

        // Save the original state of the exception ports for our child process
        SaveExceptionPortInfo();

        // We weren't able to save the info for our exception ports, we must stop...
        if (m_exc_port_info.mask == 0)
        {
            err.SetErrorString("failed to get exception port info");
            return false;
        }

        // Set the ability to get all exceptions on this port
        err = ::task_set_exception_ports (task, m_exc_port_info.mask, m_exception_port, EXCEPTION_DEFAULT | MACH_EXCEPTION_CODES, THREAD_STATE_NONE);
        if (DNBLogCheckLogBit(LOG_EXCEPTIONS) || err.Fail())
        {
            err.LogThreaded("::task_set_exception_ports ( task = 0x%4.4x, exception_mask = 0x%8.8x, new_port = 0x%4.4x, behavior = 0x%8.8x, new_flavor = 0x%8.8x )",
                            task,
                            m_exc_port_info.mask,
                            m_exception_port,
                            (EXCEPTION_DEFAULT | MACH_EXCEPTION_CODES),
                            THREAD_STATE_NONE);
        }

        if (err.Fail())
            return false;

        // Create the exception thread
        err = ::pthread_create (&m_exception_thread, NULL, MachTask::ExceptionThread, this);
        return err.Success();
    }
    else
    {
        DNBLogError("MachTask::%s (): task invalid, exception thread start failed.", __FUNCTION__);
    }
    return false;
}

kern_return_t
MachTask::ShutDownExcecptionThread()
{
    DNBError err;

    err = RestoreExceptionPortInfo();

    // NULL our our exception port and let our exception thread exit
    mach_port_t exception_port = m_exception_port;
    m_exception_port = NULL;

    err.SetError(::pthread_cancel(m_exception_thread), DNBError::POSIX);
    if (DNBLogCheckLogBit(LOG_TASK) || err.Fail())
        err.LogThreaded("::pthread_cancel ( thread = %p )", m_exception_thread);

    err.SetError(::pthread_join(m_exception_thread, NULL), DNBError::POSIX);
    if (DNBLogCheckLogBit(LOG_TASK) || err.Fail())
        err.LogThreaded("::pthread_join ( thread = %p, value_ptr = NULL)", m_exception_thread);

    // Deallocate our exception port that we used to track our child process
    mach_port_t task_self = mach_task_self ();
    err = ::mach_port_deallocate (task_self, exception_port);
    if (DNBLogCheckLogBit(LOG_TASK) || err.Fail())
        err.LogThreaded("::mach_port_deallocate ( task = 0x%4.4x, name = 0x%4.4x )", task_self, exception_port);

    return err.Error();
}


void *
MachTask::ExceptionThread (void *arg)
{
    if (arg == NULL)
        return NULL;

    MachTask *mach_task = (MachTask*) arg;
    MachProcess *mach_proc = mach_task->Process();
    DNBLogThreadedIf(LOG_EXCEPTIONS, "MachTask::%s ( arg = %p ) starting thread...", __FUNCTION__, arg);

    // We keep a count of the number of consecutive exceptions received so
    // we know to grab all exceptions without a timeout. We do this to get a
    // bunch of related exceptions on our exception port so we can process
    // then together. When we have multiple threads, we can get an exception
    // per thread and they will come in consecutively. The main loop in this
    // thread can stop periodically if needed to service things related to this
    // process.
    // flag set in the options, so we will wait forever for an exception on
    // our exception port. After we get one exception, we then will use the
    // MACH_RCV_TIMEOUT option with a zero timeout to grab all other current
    // exceptions for our process. After we have received the last pending
    // exception, we will get a timeout which enables us to then notify
    // our main thread that we have an exception bundle avaiable. We then wait
    // for the main thread to tell this exception thread to start trying to get
    // exceptions messages again and we start again with a mach_msg read with
    // infinite timeout.
    uint32_t num_exceptions_received = 0;
    DNBError err;
    task_t task = mach_task->TaskPort();
    mach_msg_timeout_t periodic_timeout = 0;

#ifdef WITH_SPRINGBOARD
    mach_msg_timeout_t watchdog_elapsed = 0;
    mach_msg_timeout_t watchdog_timeout = 60 * 1000;
    pid_t pid = mach_proc->ProcessID();
    CFReleaser<SBSWatchdogAssertionRef> watchdog;

    if (mach_proc->ProcessUsingSpringBoard())
    {
        // Request a renewal for every 60 seconds if we attached using SpringBoard
        watchdog.reset(::SBSWatchdogAssertionCreateForPID(NULL, pid, 60));
        DNBLogThreadedIf(LOG_TASK, "::SBSWatchdogAssertionCreateForPID (NULL, %4.4x, 60 ) => %p", pid, watchdog.get());

        if (watchdog.get())
        {
            ::SBSWatchdogAssertionRenew (watchdog.get());

            CFTimeInterval watchdogRenewalInterval = ::SBSWatchdogAssertionGetRenewalInterval (watchdog.get());
            DNBLogThreadedIf(LOG_TASK, "::SBSWatchdogAssertionGetRenewalInterval ( %p ) => %g seconds", watchdog.get(), watchdogRenewalInterval);
            if (watchdogRenewalInterval > 0.0)
            {
                watchdog_timeout = (mach_msg_timeout_t)watchdogRenewalInterval * 1000;
                if (watchdog_timeout > 3000)
                    watchdog_timeout -= 1000;   // Give us a second to renew our timeout
                else if (watchdog_timeout > 1000)
                    watchdog_timeout -= 250;    // Give us a quarter of a second to renew our timeout
            }
        }
        if (periodic_timeout == 0 || periodic_timeout > watchdog_timeout)
            periodic_timeout = watchdog_timeout;
    }
#endif  // #ifdef WITH_SPRINGBOARD

    while (mach_task->ExceptionPortIsValid())
    {
        ::pthread_testcancel ();

        MachException::Message exception_message;


        if (num_exceptions_received > 0)
        {
            // No timeout, just receive as many exceptions as we can since we already have one and we want
            // to get all currently available exceptions for this task
            err = exception_message.Receive(mach_task->ExceptionPort(), MACH_RCV_MSG | MACH_RCV_INTERRUPT | MACH_RCV_TIMEOUT, 0);
        }
        else if (periodic_timeout > 0)
        {
            // We need to stop periodically in this loop, so try and get a mach message with a valid timeout (ms)
            err = exception_message.Receive(mach_task->ExceptionPort(), MACH_RCV_MSG | MACH_RCV_INTERRUPT | MACH_RCV_TIMEOUT, periodic_timeout);
        }
        else
        {
            // We don't need to parse all current exceptions or stop periodically,
            // just wait for an exception forever.
            err = exception_message.Receive(mach_task->ExceptionPort(), MACH_RCV_MSG | MACH_RCV_INTERRUPT, 0);
        }

        if (err.Error() == MACH_RCV_INTERRUPTED)
        {
            // If we have no task port we should exit this thread
            if (!mach_task->ExceptionPortIsValid())
            {
                DNBLogThreadedIf(LOG_EXCEPTIONS, "thread cancelled...");
                break;
            }

            // Make sure our task is still valid
            if (MachTask::IsValid(task))
            {
                // Task is still ok
                DNBLogThreadedIf(LOG_EXCEPTIONS, "interrupted, but task still valid, continuing...");
                continue;
            }
            else
            {
                DNBLogThreadedIf(LOG_EXCEPTIONS, "task has exited...");
                mach_proc->SetState(eStateExited);
                // Our task has died, exit the thread.
                break;
            }
        }
        else if (err.Error() == MACH_RCV_TIMED_OUT)
        {
            if (num_exceptions_received > 0)
            {
                // We were receiving all current exceptions with a timeout of zero
                // it is time to go back to our normal looping mode
                num_exceptions_received = 0;

                // Notify our main thread we have a complete exception message
                // bundle available.
                mach_proc->ExceptionMessageBundleComplete();

                // in case we use a timeout value when getting exceptions...
                // Make sure our task is still valid
                if (MachTask::IsValid(task))
                {
                    // Task is still ok
                    DNBLogThreadedIf(LOG_EXCEPTIONS, "got a timeout, continuing...");
                    continue;
                }
                else
                {
                    DNBLogThreadedIf(LOG_EXCEPTIONS, "task has exited...");
                    mach_proc->SetState(eStateExited);
                    // Our task has died, exit the thread.
                    break;
                }
                continue;
            }

#ifdef WITH_SPRINGBOARD
            if (watchdog.get())
            {
                watchdog_elapsed += periodic_timeout;
                if (watchdog_elapsed >= watchdog_timeout)
                {
                    DNBLogThreadedIf(LOG_TASK, "SBSWatchdogAssertionRenew ( %p )", watchdog.get());
                    ::SBSWatchdogAssertionRenew (watchdog.get());
                    watchdog_elapsed = 0;
                }
            }
#endif
        }
        else if (err.Error() != KERN_SUCCESS)
        {
            DNBLogThreadedIf(LOG_EXCEPTIONS, "got some other error, do something about it??? nah, continuing for now...");
            // TODO: notify of error?
        }
        else
        {
            if (exception_message.CatchExceptionRaise(task))
            {
                ++num_exceptions_received;
                mach_proc->ExceptionMessageReceived(exception_message);
            }
        }
    }

#ifdef WITH_SPRINGBOARD
    if (watchdog.get())
    {
        // TODO: change SBSWatchdogAssertionRelease to SBSWatchdogAssertionCancel when we
        // all are up and running on systems that support it. The SBS framework has a #define
        // that will forward SBSWatchdogAssertionRelease to SBSWatchdogAssertionCancel for now
        // so it should still build either way.
        DNBLogThreadedIf(LOG_TASK, "::SBSWatchdogAssertionRelease(%p)", watchdog.get());
        ::SBSWatchdogAssertionRelease (watchdog.get());
    }
#endif  // #ifdef WITH_SPRINGBOARD

    DNBLogThreadedIf(LOG_EXCEPTIONS, "MachTask::%s (%p): thread exiting...", __FUNCTION__, arg);
    return NULL;
}


// So the TASK_DYLD_INFO used to just return the address of the all image infos
// as a single member called "all_image_info". Then someone decided it would be
// a good idea to rename this first member to "all_image_info_addr" and add a
// size member called "all_image_info_size". This of course can not be detected
// using code or #defines. So to hack around this problem, we define our own
// version of the TASK_DYLD_INFO structure so we can guarantee what is inside it.

struct hack_task_dyld_info {
    mach_vm_address_t   all_image_info_addr;
    mach_vm_size_t      all_image_info_size;
};

nub_addr_t
MachTask::GetDYLDAllImageInfosAddress (DNBError& err)
{
    struct hack_task_dyld_info dyld_info;
    mach_msg_type_number_t count = TASK_DYLD_INFO_COUNT;
    // Make sure that COUNT isn't bigger than our hacked up struct hack_task_dyld_info.
    // If it is, then make COUNT smaller to match.
    if (count > (sizeof(struct hack_task_dyld_info) / sizeof(natural_t)))
        count = (sizeof(struct hack_task_dyld_info) / sizeof(natural_t));

    task_t task = TaskPortForProcessID (err);
    if (err.Success())
    {
        err = ::task_info (task, TASK_DYLD_INFO, (task_info_t)&dyld_info, &count);
        if (err.Success())
        {
            // We now have the address of the all image infos structure
            return dyld_info.all_image_info_addr;
        }
    }
    return INVALID_NUB_ADDRESS;
}


//----------------------------------------------------------------------
// MachTask::AllocateMemory
//----------------------------------------------------------------------
nub_addr_t
MachTask::AllocateMemory (size_t size, uint32_t permissions)
{
    mach_vm_address_t addr;
    task_t task = TaskPort();
    if (task == TASK_NULL)
        return INVALID_NUB_ADDRESS;

    DNBError err;
    err = ::mach_vm_allocate (task, &addr, size, TRUE);
    if (err.Error() == KERN_SUCCESS)
    {
        // Set the protections:
        vm_prot_t mach_prot = VM_PROT_NONE;
        if (permissions & eMemoryPermissionsReadable)
            mach_prot |= VM_PROT_READ;
        if (permissions & eMemoryPermissionsWritable)
            mach_prot |= VM_PROT_WRITE;
        if (permissions & eMemoryPermissionsExecutable)
            mach_prot |= VM_PROT_EXECUTE;


        err = ::mach_vm_protect (task, addr, size, 0, mach_prot);
        if (err.Error() == KERN_SUCCESS)
        {
            m_allocations.insert (std::make_pair(addr, size));
            return addr;
        }
        ::mach_vm_deallocate (task, addr, size);
    }
    return INVALID_NUB_ADDRESS;
}

//----------------------------------------------------------------------
// MachTask::DeallocateMemory
//----------------------------------------------------------------------
nub_bool_t
MachTask::DeallocateMemory (nub_addr_t addr)
{
    task_t task = TaskPort();
    if (task == TASK_NULL)
        return false;

    // We have to stash away sizes for the allocations...
    allocation_collection::iterator pos, end = m_allocations.end();
    for (pos = m_allocations.begin(); pos != end; pos++)
    {
        if ((*pos).first == addr)
        {
            m_allocations.erase(pos);
#define ALWAYS_ZOMBIE_ALLOCATIONS 0
            if (ALWAYS_ZOMBIE_ALLOCATIONS || getenv ("DEBUGSERVER_ZOMBIE_ALLOCATIONS"))
            {
                ::mach_vm_protect (task, (*pos).first, (*pos).second, 0, VM_PROT_NONE);
                return true;
            }
            else
                return ::mach_vm_deallocate (task, (*pos).first, (*pos).second) == KERN_SUCCESS;
        }
        
    }
    return false;
}

static void foundStackLog(mach_stack_logging_record_t record, void *context) {
    *((bool*)context) = true;
}

bool
MachTask::HasMallocLoggingEnabled ()
{
    bool found = false;
    
    __mach_stack_logging_enumerate_records(m_task, 0x0, foundStackLog, &found);
    return found;
}

struct history_enumerator_impl_data
{
    MachMallocEvent *buffer;
    uint32_t        *position;
    uint32_t         count;
};

static void history_enumerator_impl(mach_stack_logging_record_t record, void* enum_obj)
{
    history_enumerator_impl_data *data = (history_enumerator_impl_data*)enum_obj;
    
    if (*data->position >= data->count)
        return;
    
    data->buffer[*data->position].m_base_address = record.address;
    data->buffer[*data->position].m_size = record.argument;
    data->buffer[*data->position].m_event_id = record.stack_identifier;
    data->buffer[*data->position].m_event_type = record.type_flags == stack_logging_type_alloc ?   eMachMallocEventTypeAlloc :
                                                 record.type_flags == stack_logging_type_dealloc ? eMachMallocEventTypeDealloc :
                                                                                                   eMachMallocEventTypeOther;
    *data->position+=1;
}

bool
MachTask::EnumerateMallocRecords (MachMallocEvent *event_buffer,
                                  uint32_t buffer_size,
                                  uint32_t *count)
{
    return EnumerateMallocRecords(0,
                                  event_buffer,
                                  buffer_size,
                                  count);
}

bool
MachTask::EnumerateMallocRecords (mach_vm_address_t address,
                                  MachMallocEvent *event_buffer,
                                  uint32_t buffer_size,
                                  uint32_t *count)
{
    if (!event_buffer || !count)
        return false;
    
    if (buffer_size == 0)
        return false;
    
    *count = 0;
    history_enumerator_impl_data data = { event_buffer, count, buffer_size };
    __mach_stack_logging_enumerate_records(m_task, address, history_enumerator_impl, &data);
    return (*count > 0);
}

bool
MachTask::EnumerateMallocFrames (MachMallocEventId event_id,
                                 mach_vm_address_t *function_addresses_buffer,
                                 uint32_t buffer_size,
                                 uint32_t *count)
{
    if (!function_addresses_buffer || !count)
        return false;
    
    if (buffer_size == 0)
        return false;
    
    __mach_stack_logging_frames_for_uniqued_stack(m_task, event_id, &function_addresses_buffer[0], buffer_size, count);
    *count -= 1;
    if (function_addresses_buffer[*count-1] < PageSize())
        *count -= 1;
    return (*count > 0);
}

nub_size_t
MachTask::PageSize ()
{
    return m_vm_memory.PageSize (m_task);
}
