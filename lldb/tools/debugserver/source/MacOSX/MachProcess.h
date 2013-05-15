//===-- MachProcess.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 6/15/07.
//
//===----------------------------------------------------------------------===//

#ifndef __MachProcess_h__
#define __MachProcess_h__

#include "DNBDefs.h"
#include "DNBBreakpoint.h"
#include "DNBError.h"
#include "DNBThreadResumeActions.h"
//#include "MachDYLD.h"
#include "MachException.h"
#include "MachVMMemory.h"
#include "MachTask.h"
#include "MachThreadList.h"
#include "PThreadCondition.h"
#include "PThreadEvent.h"
#include "PThreadMutex.h"

#include <mach/mach.h>
#include <sys/signal.h>
#include <pthread.h>
#include <vector>

class DNBThreadResumeActions;

class MachProcess
{
public:
    //----------------------------------------------------------------------
    // Constructors and Destructors
    //----------------------------------------------------------------------
    MachProcess ();
    ~MachProcess ();

    //----------------------------------------------------------------------
    // Child process control
    //----------------------------------------------------------------------
    pid_t                   AttachForDebug (pid_t pid, char *err_str, size_t err_len);
    pid_t                   LaunchForDebug (const char *path, 
                                            char const *argv[], 
                                            char const *envp[], 
                                            const char *working_directory,
                                            const char *stdin_path,
                                            const char *stdout_path,
                                            const char *stderr_path,
                                            bool no_stdio, 
                                            nub_launch_flavor_t launch_flavor, 
                                            int disable_aslr, 
                                            DNBError &err);

    static uint32_t         GetCPUTypeForLocalProcess (pid_t pid);
    static pid_t            ForkChildForPTraceDebugging (const char *path, char const *argv[], char const *envp[], MachProcess* process, DNBError &err);
    static pid_t            PosixSpawnChildForPTraceDebugging (const char *path, 
                                                               cpu_type_t cpu_type, 
                                                               char const *argv[], 
                                                               char const *envp[], 
                                                               const char *working_directory,
                                                               const char *stdin_path,
                                                               const char *stdout_path,
                                                               const char *stderr_path,
                                                               bool no_stdio, 
                                                               MachProcess* process, 
                                                               int disable_aslr, 
                                                               DNBError& err);
    nub_addr_t              GetDYLDAllImageInfosAddress ();
    static const void *     PrepareForAttach (const char *path, nub_launch_flavor_t launch_flavor, bool waitfor, DNBError &err_str);
    static void             CleanupAfterAttach (const void *attach_token, bool success, DNBError &err_str);
    static nub_process_t    CheckForProcess (const void *attach_token);
#ifdef WITH_SPRINGBOARD
    pid_t                   SBLaunchForDebug (const char *app_bundle_path, char const *argv[], char const *envp[], bool no_stdio, DNBError &launch_err);
    static pid_t            SBForkChildForPTraceDebugging (const char *path, char const *argv[], char const *envp[], bool no_stdio, MachProcess* process, DNBError &launch_err);
#endif
    nub_addr_t              LookupSymbol (const char *name, const char *shlib);
    void                    SetNameToAddressCallback (DNBCallbackNameToAddress callback, void *baton)
                            {
                                m_name_to_addr_callback = callback;
                                m_name_to_addr_baton    = baton;
                            }
    void                    SetSharedLibraryInfoCallback (DNBCallbackCopyExecutableImageInfos callback, void *baton)
                            {
                                m_image_infos_callback    = callback;
                                m_image_infos_baton        = baton;
                            }

    bool                    Resume (const DNBThreadResumeActions& thread_actions);
    bool                    Signal  (int signal, const struct timespec *timeout_abstime = NULL);
    bool                    Kill (const struct timespec *timeout_abstime = NULL);
    bool                    Detach ();
    nub_size_t              ReadMemory (nub_addr_t addr, nub_size_t size, void *buf);
    nub_size_t              WriteMemory (nub_addr_t addr, nub_size_t size, const void *buf);

    //----------------------------------------------------------------------
    // Path and arg accessors
    //----------------------------------------------------------------------
    const char *            Path () const { return m_path.c_str(); }
    size_t                  ArgumentCount () const { return m_args.size(); }
    const char *            ArgumentAtIndex (size_t arg_idx) const
                            {
                                if (arg_idx < m_args.size())
                                    return m_args[arg_idx].c_str();
                                return NULL;
                            }

    //----------------------------------------------------------------------
    // Breakpoint functions
    //----------------------------------------------------------------------
    nub_break_t             CreateBreakpoint (nub_addr_t addr, nub_size_t length, bool hardware, thread_t thread);
    bool                    DisableBreakpoint (nub_break_t breakID, bool remove);
    nub_size_t              DisableAllBreakpoints (bool remove);
    bool                    EnableBreakpoint (nub_break_t breakID);
    void                    DumpBreakpoint(nub_break_t breakID) const;
    DNBBreakpointList&      Breakpoints() { return m_breakpoints; }
    const DNBBreakpointList& Breakpoints() const { return m_breakpoints; }

    //----------------------------------------------------------------------
    // Watchpoint functions
    //----------------------------------------------------------------------
    nub_watch_t             CreateWatchpoint (nub_addr_t addr, nub_size_t length, uint32_t watch_type, bool hardware, thread_t thread);
    bool                    DisableWatchpoint (nub_watch_t watchID, bool remove);
    nub_size_t              DisableAllWatchpoints (bool remove);
    bool                    EnableWatchpoint (nub_watch_t watchID);
    void                    DumpWatchpoint(nub_watch_t watchID) const;
    uint32_t                GetNumSupportedHardwareWatchpoints () const;
    DNBBreakpointList&      Watchpoints() { return m_watchpoints; }
    const DNBBreakpointList& Watchpoints() const { return m_watchpoints; }

    //----------------------------------------------------------------------
    // Exception thread functions
    //----------------------------------------------------------------------
    bool                    StartSTDIOThread ();
    static void *           STDIOThread (void *arg);
    void                    ExceptionMessageReceived (const MachException::Message& exceptionMessage);
    void                    ExceptionMessageBundleComplete ();
    void                    SharedLibrariesUpdated ();
    nub_size_t              CopyImageInfos (struct DNBExecutableImageInfo **image_infos, bool only_changed);
    
    //----------------------------------------------------------------------
    // Profile functions
    //----------------------------------------------------------------------
    void                    SetEnableAsyncProfiling (bool enable, uint64_t internal_usec, DNBProfileDataScanType scan_type);
    bool                    IsProfilingEnabled () { return m_profile_enabled; }
    uint64_t                ProfileInterval () { return m_profile_interval_usec; }
    bool                    StartProfileThread ();
    static void *           ProfileThread (void *arg);
    void                    SignalAsyncProfileData (const char *info);
    size_t                  GetAsyncProfileData (char *buf, size_t buf_size);

    //----------------------------------------------------------------------
    // Accessors
    //----------------------------------------------------------------------
    pid_t                   ProcessID () const { return m_pid; }
    bool                    ProcessIDIsValid () const { return m_pid > 0; }
    pid_t                   SetProcessID (pid_t pid);
    MachTask&               Task() { return m_task; }
    const MachTask&         Task() const { return m_task; }

    PThreadEvent&           Events() { return m_events; }
    const DNBRegisterSetInfo *
                            GetRegisterSetInfo (nub_thread_t tid, nub_size_t *num_reg_sets) const;
    bool                    GetRegisterValue (nub_thread_t tid, uint32_t set, uint32_t reg, DNBRegisterValue *reg_value) const;
    bool                    SetRegisterValue (nub_thread_t tid, uint32_t set, uint32_t reg, const DNBRegisterValue *value) const;
    nub_bool_t              SyncThreadState (nub_thread_t tid);
    const char *            ThreadGetName (nub_thread_t tid);
    nub_state_t             ThreadGetState (nub_thread_t tid);
    nub_size_t              GetNumThreads () const;
    nub_thread_t            GetThreadAtIndex (nub_size_t thread_idx) const;
    nub_thread_t            GetCurrentThread ();
    nub_thread_t            GetCurrentThreadMachPort ();
    nub_thread_t            SetCurrentThread (nub_thread_t tid);
    MachThreadList &        GetThreadList() { return m_thread_list; }
    bool                    GetThreadStoppedReason(nub_thread_t tid, struct DNBThreadStopInfo *stop_info) const;
    void                    DumpThreadStoppedReason(nub_thread_t tid) const;
    const char *            GetThreadInfo (nub_thread_t tid) const;

    nub_thread_t            GetThreadIDForMachPortNumber (thread_t mach_port_number) const;

    uint32_t                GetCPUType ();
    nub_state_t             GetState ();
    void                    SetState (nub_state_t state);
    bool                    IsRunning (nub_state_t state)
                            {
                                return    state == eStateRunning || IsStepping(state);
                            }
    bool                    IsStepping (nub_state_t state)
                            {
                                return    state == eStateStepping;
                            }
    bool                    CanResume (nub_state_t state)
                            {
                                return state == eStateStopped;
                            }

    bool                    GetExitStatus(int* status)
                            {
                                if (GetState() == eStateExited)
                                {
                                    if (status)
                                        *status = m_exit_status;
                                    return true;
                                }
                                return false;
                            }
    void                    SetExitStatus(int status)
                            {
                                m_exit_status = status;
                                SetState(eStateExited);
                            }

    uint32_t                StopCount() const { return m_stop_count; }
    void                    SetChildFileDescriptors (int stdin_fileno, int stdout_fileno, int stderr_fileno)
                            {
                                m_child_stdin   = stdin_fileno;
                                m_child_stdout  = stdout_fileno;
                                m_child_stderr  = stderr_fileno;
                            }

    int                     GetStdinFileDescriptor () const { return m_child_stdin; }
    int                     GetStdoutFileDescriptor () const { return m_child_stdout; }
    int                     GetStderrFileDescriptor () const { return m_child_stderr; }
    void                    AppendSTDOUT (char* s, size_t len);
    size_t                  GetAvailableSTDOUT (char *buf, size_t buf_size);
    size_t                  GetAvailableSTDERR (char *buf, size_t buf_size);
    void                    CloseChildFileDescriptors ()
                            {
                                if (m_child_stdin >= 0)
                                {
                                    ::close (m_child_stdin);
                                    m_child_stdin = -1;
                                }
                                if (m_child_stdout >= 0)
                                {
                                    ::close (m_child_stdout);
                                    m_child_stdout = -1;
                                }
                                if (m_child_stderr >= 0)
                                {
                                    ::close (m_child_stderr);
                                    m_child_stderr = -1;
                                }
                            }

    bool                    ProcessUsingSpringBoard() const { return (m_flags & eMachProcessFlagsUsingSBS) != 0; }
    
    DNBProfileDataScanType  GetProfileScanType () { return m_profile_scan_type; }
    
private:
    enum
    {
        eMachProcessFlagsNone = 0,
        eMachProcessFlagsAttached = (1 << 0),
        eMachProcessFlagsUsingSBS = (1 << 1)
    };
    void                    Clear ();
    void                    ReplyToAllExceptions ();
    void                    PrivateResume ();
    nub_size_t              RemoveTrapsFromBuffer (nub_addr_t addr, nub_size_t size, uint8_t *buf) const;

    uint32_t                Flags () const { return m_flags; }
    nub_state_t             DoSIGSTOP (bool clear_bps_and_wps, bool allow_running, uint32_t *thread_idx_ptr);

    pid_t                       m_pid;                      // Process ID of child process
    cpu_type_t                  m_cpu_type;                 // The CPU type of this process
    int                         m_child_stdin;
    int                         m_child_stdout;
    int                         m_child_stderr;
    std::string                 m_path;                     // A path to the executable if we have one
    std::vector<std::string>    m_args;                     // The arguments with which the process was lauched
    int                         m_exit_status;              // The exit status for the process
    MachTask                    m_task;                     // The mach task for this process
    uint32_t                    m_flags;                    // Process specific flags (see eMachProcessFlags enums)
    uint32_t                    m_stop_count;               // A count of many times have we stopped
    pthread_t                   m_stdio_thread;             // Thread ID for the thread that watches for child process stdio
    PThreadMutex                m_stdio_mutex;              // Multithreaded protection for stdio
    std::string                 m_stdout_data;
    
    bool                        m_profile_enabled;          // A flag to indicate if profiling is enabled
    uint64_t                    m_profile_interval_usec;    // If enable, the profiling interval in microseconds
    DNBProfileDataScanType      m_profile_scan_type;        // Indicates what needs to be profiled
    pthread_t                   m_profile_thread;           // Thread ID for the thread that profiles the inferior
    PThreadMutex                m_profile_data_mutex;       // Multithreaded protection for profile info data
    std::vector<std::string>    m_profile_data;             // Profile data, must be protected by m_profile_data_mutex
    
    DNBThreadResumeActions      m_thread_actions;           // The thread actions for the current MachProcess::Resume() call
    MachException::Message::collection
                                m_exception_messages;       // A collection of exception messages caught when listening to the exception port
    PThreadMutex                m_exception_messages_mutex; // Multithreaded protection for m_exception_messages

    MachThreadList              m_thread_list;               // A list of threads that is maintained/updated after each stop
    nub_state_t                 m_state;                    // The state of our process
    PThreadMutex                m_state_mutex;              // Multithreaded protection for m_state
    PThreadEvent                m_events;                   // Process related events in the child processes lifetime can be waited upon
    DNBBreakpointList           m_breakpoints;              // Breakpoint list for this process
    DNBBreakpointList           m_watchpoints;              // Watchpoint list for this process
    DNBCallbackNameToAddress    m_name_to_addr_callback;
    void *                      m_name_to_addr_baton;
    DNBCallbackCopyExecutableImageInfos
                                m_image_infos_callback;
    void *                      m_image_infos_baton;
};


#endif // __MachProcess_h__
