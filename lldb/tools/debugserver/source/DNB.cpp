//===-- DNB.cpp -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 3/23/07.
//
//===----------------------------------------------------------------------===//

#include "DNB.h"
#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sys/sysctl.h>
#include <map>
#include <vector>
#include <libproc.h>

#include "MacOSX/MachProcess.h"
#include "MacOSX/MachTask.h"
#include "CFString.h"
#include "DNBLog.h"
#include "DNBDataRef.h"
#include "DNBThreadResumeActions.h"
#include "DNBTimer.h"
#include "CFBundle.h"


typedef std::shared_ptr<MachProcess> MachProcessSP;
typedef std::map<nub_process_t, MachProcessSP> ProcessMap;
typedef ProcessMap::iterator ProcessMapIter;
typedef ProcessMap::const_iterator ProcessMapConstIter;

size_t GetAllInfos (std::vector<struct kinfo_proc>& proc_infos);
static size_t GetAllInfosMatchingName (const char *process_name, std::vector<struct kinfo_proc>& matching_proc_infos);

//----------------------------------------------------------------------
// A Thread safe singleton to get a process map pointer.
//
// Returns a pointer to the existing process map, or a pointer to a
// newly created process map if CAN_CREATE is non-zero.
//----------------------------------------------------------------------
static ProcessMap*
GetProcessMap(bool can_create)
{
    static ProcessMap* g_process_map_ptr = NULL;

    if (can_create && g_process_map_ptr == NULL)
    {
        static pthread_mutex_t g_process_map_mutex = PTHREAD_MUTEX_INITIALIZER;
        PTHREAD_MUTEX_LOCKER (locker, &g_process_map_mutex);
        if (g_process_map_ptr == NULL)
            g_process_map_ptr = new ProcessMap;
    }
    return g_process_map_ptr;
}

//----------------------------------------------------------------------
// Add PID to the shared process pointer map.
//
// Return non-zero value if we succeed in adding the process to the map.
// The only time this should fail is if we run out of memory and can't
// allocate a ProcessMap.
//----------------------------------------------------------------------
static nub_bool_t
AddProcessToMap (nub_process_t pid, MachProcessSP& procSP)
{
    ProcessMap* process_map = GetProcessMap(true);
    if (process_map)
    {
        process_map->insert(std::make_pair(pid, procSP));
        return true;
    }
    return false;
}

//----------------------------------------------------------------------
// Remove the shared pointer for PID from the process map.
//
// Returns the number of items removed from the process map.
//----------------------------------------------------------------------
static size_t
RemoveProcessFromMap (nub_process_t pid)
{
    ProcessMap* process_map = GetProcessMap(false);
    if (process_map)
    {
        return process_map->erase(pid);
    }
    return 0;
}

//----------------------------------------------------------------------
// Get the shared pointer for PID from the existing process map.
//
// Returns true if we successfully find a shared pointer to a
// MachProcess object.
//----------------------------------------------------------------------
static nub_bool_t
GetProcessSP (nub_process_t pid, MachProcessSP& procSP)
{
    ProcessMap* process_map = GetProcessMap(false);
    if (process_map != NULL)
    {
        ProcessMapIter pos = process_map->find(pid);
        if (pos != process_map->end())
        {
            procSP = pos->second;
            return true;
        }
    }
    procSP.reset();
    return false;
}


static void *
waitpid_thread (void *arg)
{
    const pid_t pid = (pid_t)(intptr_t)arg;
    int status;
    while (1)
    {
        pid_t child_pid = waitpid(pid, &status, 0);
        DNBLogThreadedIf(LOG_PROCESS, "waitpid_process_thread (): waitpid (pid = %i, &status, 0) => %i, status = %i, errno = %i", pid, child_pid, status, errno);

        if (child_pid < 0)
        {
            if (errno == EINTR)
                continue;
            break;
        }
        else
        {
            if (WIFSTOPPED(status))
            {
                continue;
            }
            else// if (WIFEXITED(status) || WIFSIGNALED(status))
            {
                DNBLogThreadedIf(LOG_PROCESS, "waitpid_process_thread (): setting exit status for pid = %i to %i", child_pid, status);
                DNBProcessSetExitStatus (child_pid, status);
                return NULL;
            }
        }
    }

    // We should never exit as long as our child process is alive, so if we
    // do something else went wrong and we should exit...
    DNBLogThreadedIf(LOG_PROCESS, "waitpid_process_thread (): main loop exited, setting exit status to an invalid value (-1) for pid %i", pid);
    DNBProcessSetExitStatus (pid, -1);
    return NULL;
}

static bool
spawn_waitpid_thread (pid_t pid)
{
    pthread_t thread = THREAD_NULL;
    ::pthread_create (&thread, NULL, waitpid_thread, (void *)(intptr_t)pid);
    if (thread != THREAD_NULL)
    {
        ::pthread_detach (thread);
        return true;
    }
    return false;
}

nub_process_t
DNBProcessLaunch (const char *path,
                  char const *argv[],
                  const char *envp[],
                  const char *working_directory, // NULL => dont' change, non-NULL => set working directory for inferior to this
                  const char *stdin_path,
                  const char *stdout_path,
                  const char *stderr_path,
                  bool no_stdio,
                  nub_launch_flavor_t launch_flavor,
                  int disable_aslr,
                  char *err_str,
                  size_t err_len)
{
    DNBLogThreadedIf(LOG_PROCESS, "%s ( path='%s', argv = %p, envp = %p, working_dir=%s, stdin=%s, stdout=%s, stderr=%s, no-stdio=%i, launch_flavor = %u, disable_aslr = %d, err = %p, err_len = %llu) called...",
                     __FUNCTION__, 
                     path, 
                     argv, 
                     envp, 
                     working_directory,
                     stdin_path,
                     stdout_path,
                     stderr_path,
                     no_stdio,
                     launch_flavor, 
                     disable_aslr, 
                     err_str, 
                     (uint64_t)err_len);
    
    if (err_str && err_len > 0)
        err_str[0] = '\0';
    struct stat path_stat;
    if (::stat(path, &path_stat) == -1)
    {
        char stat_error[256];
        ::strerror_r (errno, stat_error, sizeof(stat_error));
        snprintf(err_str, err_len, "%s (%s)", stat_error, path);
        return INVALID_NUB_PROCESS;
    }

    MachProcessSP processSP (new MachProcess);
    if (processSP.get())
    {
        DNBError launch_err;
        pid_t pid = processSP->LaunchForDebug (path, 
                                               argv, 
                                               envp, 
                                               working_directory, 
                                               stdin_path, 
                                               stdout_path, 
                                               stderr_path, 
                                               no_stdio, 
                                               launch_flavor, 
                                               disable_aslr, 
                                               launch_err);
        if (err_str)
        {
            *err_str = '\0';
            if (launch_err.Fail())
            {
                const char *launch_err_str = launch_err.AsString();
                if (launch_err_str)
                {
                    strncpy(err_str, launch_err_str, err_len-1);
                    err_str[err_len-1] = '\0';  // Make sure the error string is terminated
                }
            }
        }

        DNBLogThreadedIf(LOG_PROCESS, "(DebugNub) new pid is %d...", pid);

        if (pid != INVALID_NUB_PROCESS)
        {
            // Spawn a thread to reap our child inferior process...
            spawn_waitpid_thread (pid);

            if (processSP->Task().TaskPortForProcessID (launch_err) == TASK_NULL)
            {
                // We failed to get the task for our process ID which is bad.
                // Kill our process otherwise it will be stopped at the entry
                // point and get reparented to someone else and never go away.
                DNBLog ("Could not get task port for process, sending SIGKILL and exiting.");
                kill (SIGKILL, pid);

                if (err_str && err_len > 0)
                {
                    if (launch_err.AsString())
                    {
                        ::snprintf (err_str, err_len, "failed to get the task for process %i (%s)", pid, launch_err.AsString());
                    }
                    else
                    {
                        ::snprintf (err_str, err_len, "failed to get the task for process %i", pid);
                    }
                }
            }
            else
            {
                bool res = AddProcessToMap(pid, processSP);
                assert(res && "Couldn't add process to map!");
                return pid;
            }
        }
    }
    return INVALID_NUB_PROCESS;
}

nub_process_t
DNBProcessAttachByName (const char *name, struct timespec *timeout, char *err_str, size_t err_len)
{
    if (err_str && err_len > 0)
        err_str[0] = '\0';
    std::vector<struct kinfo_proc> matching_proc_infos;
    size_t num_matching_proc_infos = GetAllInfosMatchingName(name, matching_proc_infos);
    if (num_matching_proc_infos == 0)
    {
        DNBLogError ("error: no processes match '%s'\n", name);
        return INVALID_NUB_PROCESS;
    }
    else if (num_matching_proc_infos > 1)
    {
        DNBLogError ("error: %llu processes match '%s':\n", (uint64_t)num_matching_proc_infos, name);
        size_t i;
        for (i=0; i<num_matching_proc_infos; ++i)
            DNBLogError ("%6u - %s\n", matching_proc_infos[i].kp_proc.p_pid, matching_proc_infos[i].kp_proc.p_comm);
        return INVALID_NUB_PROCESS;
    }
    
    return DNBProcessAttach (matching_proc_infos[0].kp_proc.p_pid, timeout, err_str, err_len);
}

nub_process_t
DNBProcessAttach (nub_process_t attach_pid, struct timespec *timeout, char *err_str, size_t err_len)
{
    if (err_str && err_len > 0)
        err_str[0] = '\0';

    pid_t pid = INVALID_NUB_PROCESS;
    MachProcessSP processSP(new MachProcess);
    if (processSP.get())
    {
        DNBLogThreadedIf(LOG_PROCESS, "(DebugNub) attaching to pid %d...", attach_pid);
        pid = processSP->AttachForDebug (attach_pid, err_str,  err_len);

        if (pid != INVALID_NUB_PROCESS)
        {
            bool res = AddProcessToMap(pid, processSP);
            assert(res && "Couldn't add process to map!");
            spawn_waitpid_thread(pid);
        }
    }

    while (pid != INVALID_NUB_PROCESS)
    {
        // Wait for process to start up and hit entry point
        DNBLogThreadedIf (LOG_PROCESS, 
                          "%s DNBProcessWaitForEvent (%4.4x, eEventProcessRunningStateChanged | eEventProcessStoppedStateChanged, true, INFINITE)...",
                          __FUNCTION__, 
                          pid);
        nub_event_t set_events = DNBProcessWaitForEvents (pid,
                                                          eEventProcessRunningStateChanged | eEventProcessStoppedStateChanged,
                                                          true, 
                                                          timeout);

        DNBLogThreadedIf (LOG_PROCESS, 
                          "%s DNBProcessWaitForEvent (%4.4x, eEventProcessRunningStateChanged | eEventProcessStoppedStateChanged, true, INFINITE) => 0x%8.8x",
                          __FUNCTION__, 
                          pid, 
                          set_events);

        if (set_events == 0)
        {
            if (err_str && err_len > 0)
                snprintf(err_str, err_len, "operation timed out");
            pid = INVALID_NUB_PROCESS;
        }
        else
        {
            if (set_events & (eEventProcessRunningStateChanged | eEventProcessStoppedStateChanged))
            {
                nub_state_t pid_state = DNBProcessGetState (pid);
                DNBLogThreadedIf (LOG_PROCESS, "%s process %4.4x state changed (eEventProcessStateChanged): %s",
                        __FUNCTION__, pid, DNBStateAsString(pid_state));

                switch (pid_state)
                {
                    default:
                    case eStateInvalid:
                    case eStateUnloaded:
                    case eStateAttaching:
                    case eStateLaunching:
                    case eStateSuspended:
                        break;  // Ignore
                        
                    case eStateRunning:
                    case eStateStepping:
                        // Still waiting to stop at entry point...
                        break;
                        
                    case eStateStopped:
                    case eStateCrashed:
                        return pid;

                    case eStateDetached:
                    case eStateExited:
                        if (err_str && err_len > 0)
                            snprintf(err_str, err_len, "process exited");
                        return INVALID_NUB_PROCESS;
                }
            }

            DNBProcessResetEvents(pid, set_events);
        }
    }

    return INVALID_NUB_PROCESS;
}

size_t
GetAllInfos (std::vector<struct kinfo_proc>& proc_infos)
{
    size_t size = 0;
    int name[] = { CTL_KERN, KERN_PROC, KERN_PROC_ALL };
    u_int namelen = sizeof(name)/sizeof(int);
    int err;

    // Try to find out how many processes are around so we can
    // size the buffer appropriately.  sysctl's man page specifically suggests
    // this approach, and says it returns a bit larger size than needed to
    // handle any new processes created between then and now.

    err = ::sysctl (name, namelen, NULL, &size, NULL, 0);

    if ((err < 0) && (err != ENOMEM))
    {
        proc_infos.clear();
        perror("sysctl (mib, miblen, NULL, &num_processes, NULL, 0)");
        return 0;
    }


    // Increase the size of the buffer by a few processes in case more have
    // been spawned
    proc_infos.resize (size / sizeof(struct kinfo_proc));
    size = proc_infos.size() * sizeof(struct kinfo_proc);   // Make sure we don't exceed our resize...
    err = ::sysctl (name, namelen, &proc_infos[0], &size, NULL, 0);
    if (err < 0)
    {
        proc_infos.clear();
        return 0;
    }

    // Trim down our array to fit what we actually got back
    proc_infos.resize(size / sizeof(struct kinfo_proc));
    return proc_infos.size();
}

static size_t
GetAllInfosMatchingName(const char *full_process_name, std::vector<struct kinfo_proc>& matching_proc_infos)
{

    matching_proc_infos.clear();
    if (full_process_name && full_process_name[0])
    {
        // We only get the process name, not the full path, from the proc_info.  So just take the
        // base name of the process name...
        const char *process_name;
        process_name = strrchr (full_process_name, '/');
        if (process_name == NULL)
            process_name = full_process_name;
        else
            process_name++;

        const int process_name_len = strlen(process_name);
        std::vector<struct kinfo_proc> proc_infos;
        const size_t num_proc_infos = GetAllInfos(proc_infos);
        if (num_proc_infos > 0)
        {
            uint32_t i;
            for (i=0; i<num_proc_infos; i++)
            {
                // Skip zombie processes and processes with unset status
                if (proc_infos[i].kp_proc.p_stat == 0 || proc_infos[i].kp_proc.p_stat == SZOMB)
                    continue;

                // Check for process by name. We only check the first MAXCOMLEN
                // chars as that is all that kp_proc.p_comm holds.
                
                if (::strncasecmp(process_name, proc_infos[i].kp_proc.p_comm, MAXCOMLEN) == 0)
                {
                    if (process_name_len > MAXCOMLEN)
                    {
                        // We found a matching process name whose first MAXCOMLEN
                        // characters match, but there is more to the name than
                        // this. We need to get the full process name.  Use proc_pidpath, which will get
                        // us the full path to the executed process.

                        char proc_path_buf[PATH_MAX];
                        
                        int return_val = proc_pidpath (proc_infos[i].kp_proc.p_pid, proc_path_buf, PATH_MAX);
                        if (return_val > 0)
                        {
                            // Okay, now search backwards from that to see if there is a
                            // slash in the name.  Note, even though we got all the args we don't care
                            // because the list data is just a bunch of concatenated null terminated strings
                            // so strrchr will start from the end of argv0.
                            
                            const char *argv_basename = strrchr(proc_path_buf, '/');
                            if (argv_basename)
                            {
                                // Skip the '/'
                                ++argv_basename;
                            }
                            else
                            {
                                // We didn't find a directory delimiter in the process argv[0], just use what was in there
                                argv_basename = proc_path_buf;
                            }

                            if (argv_basename)
                            {
                                if (::strncasecmp(process_name, argv_basename, PATH_MAX) == 0)
                                {
                                    matching_proc_infos.push_back(proc_infos[i]);
                                }
                            }
                        }
                    }
                    else
                    {
                        // We found a matching process, add it to our list
                        matching_proc_infos.push_back(proc_infos[i]);
                    }
                }
            }
        }
    }
    // return the newly added matches.
    return matching_proc_infos.size();
}

nub_process_t
DNBProcessAttachWait (const char *waitfor_process_name, 
                      nub_launch_flavor_t launch_flavor,
                      bool ignore_existing,
                      struct timespec *timeout_abstime, 
                      useconds_t waitfor_interval,
                      char *err_str, 
                      size_t err_len,
                      DNBShouldCancelCallback should_cancel_callback,
                      void *callback_data)
{
    DNBError prepare_error;
    std::vector<struct kinfo_proc> exclude_proc_infos;
    size_t num_exclude_proc_infos;

    // If the PrepareForAttach returns a valid token, use  MachProcess to check
    // for the process, otherwise scan the process table.

    const void *attach_token = MachProcess::PrepareForAttach (waitfor_process_name, launch_flavor, true, prepare_error);

    if (prepare_error.Fail())
    {
        DNBLogError ("Error in PrepareForAttach: %s", prepare_error.AsString());
        return INVALID_NUB_PROCESS;
    }

    if (attach_token == NULL)
    {
        if (ignore_existing)
            num_exclude_proc_infos = GetAllInfosMatchingName (waitfor_process_name, exclude_proc_infos);
        else
            num_exclude_proc_infos = 0;
    }

    DNBLogThreadedIf (LOG_PROCESS, "Waiting for '%s' to appear...\n", waitfor_process_name);

    // Loop and try to find the process by name
    nub_process_t waitfor_pid = INVALID_NUB_PROCESS;

    while (waitfor_pid == INVALID_NUB_PROCESS)
    {
        if (attach_token != NULL)
        {
            nub_process_t pid;
            pid = MachProcess::CheckForProcess(attach_token);
            if (pid != INVALID_NUB_PROCESS)
            {
                waitfor_pid = pid;
                break;
            }
        }
        else
        {

            // Get the current process list, and check for matches that
            // aren't in our original list. If anyone wants to attach
            // to an existing process by name, they should do it with
            // --attach=PROCNAME. Else we will wait for the first matching
            // process that wasn't in our exclusion list.
            std::vector<struct kinfo_proc> proc_infos;
            const size_t num_proc_infos = GetAllInfosMatchingName (waitfor_process_name, proc_infos);
            for (size_t i=0; i<num_proc_infos; i++)
            {
                nub_process_t curr_pid = proc_infos[i].kp_proc.p_pid;
                for (size_t j=0; j<num_exclude_proc_infos; j++)
                {
                    if (curr_pid == exclude_proc_infos[j].kp_proc.p_pid)
                    {
                        // This process was in our exclusion list, don't use it.
                        curr_pid = INVALID_NUB_PROCESS;
                        break;
                    }
                }

                // If we didn't find CURR_PID in our exclusion list, then use it.
                if (curr_pid != INVALID_NUB_PROCESS)
                {
                    // We found our process!
                    waitfor_pid = curr_pid;
                    break;
                }
            }
        }

        // If we haven't found our process yet, check for a timeout
        // and then sleep for a bit until we poll again.
        if (waitfor_pid == INVALID_NUB_PROCESS)
        {
            if (timeout_abstime != NULL)
            {
                // Check to see if we have a waitfor-duration option that
                // has timed out?
                if (DNBTimer::TimeOfDayLaterThan(*timeout_abstime))
                {
                    if (err_str && err_len > 0)
                        snprintf(err_str, err_len, "operation timed out");
                    DNBLogError ("error: waiting for process '%s' timed out.\n", waitfor_process_name);
                    return INVALID_NUB_PROCESS;
                }
            }

            // Call the should cancel callback as well...

            if (should_cancel_callback != NULL
                && should_cancel_callback (callback_data))
            {
                DNBLogThreadedIf (LOG_PROCESS, "DNBProcessAttachWait cancelled by should_cancel callback.");
                waitfor_pid = INVALID_NUB_PROCESS;
                break;
            }

            ::usleep (waitfor_interval);    // Sleep for WAITFOR_INTERVAL, then poll again
        }
    }

    if (waitfor_pid != INVALID_NUB_PROCESS)
    {
        DNBLogThreadedIf (LOG_PROCESS, "Attaching to %s with pid %i...\n", waitfor_process_name, waitfor_pid);
        waitfor_pid = DNBProcessAttach (waitfor_pid, timeout_abstime, err_str, err_len);
    }

    bool success = waitfor_pid != INVALID_NUB_PROCESS;
    MachProcess::CleanupAfterAttach (attach_token, success, prepare_error);

    return waitfor_pid;
}

nub_bool_t
DNBProcessDetach (nub_process_t pid)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
    {
        return procSP->Detach();
    }
    return false;
}

nub_bool_t
DNBProcessKill (nub_process_t pid)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
    {
        return procSP->Kill ();
    }
    return false;
}

nub_bool_t
DNBProcessSignal (nub_process_t pid, int signal)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
    {
        return procSP->Signal (signal);
    }
    return false;
}


nub_bool_t
DNBProcessIsAlive (nub_process_t pid)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
    {
        return MachTask::IsValid (procSP->Task().TaskPort());
    }
    return eStateInvalid;
}

//----------------------------------------------------------------------
// Process and Thread state information
//----------------------------------------------------------------------
nub_state_t
DNBProcessGetState (nub_process_t pid)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
    {
        return procSP->GetState();
    }
    return eStateInvalid;
}

//----------------------------------------------------------------------
// Process and Thread state information
//----------------------------------------------------------------------
nub_bool_t
DNBProcessGetExitStatus (nub_process_t pid, int* status)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
    {
        return procSP->GetExitStatus(status);
    }
    return false;
}

nub_bool_t
DNBProcessSetExitStatus (nub_process_t pid, int status)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
    {
        procSP->SetExitStatus(status);
        return true;
    }
    return false;
}


const char *
DNBThreadGetName (nub_process_t pid, nub_thread_t tid)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        return procSP->ThreadGetName(tid);
    return NULL;
}


nub_bool_t
DNBThreadGetIdentifierInfo (nub_process_t pid, nub_thread_t tid, thread_identifier_info_data_t *ident_info)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        return procSP->GetThreadList().GetIdentifierInfo(tid, ident_info);
    return false;
}

nub_state_t
DNBThreadGetState (nub_process_t pid, nub_thread_t tid)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
    {
        return procSP->ThreadGetState(tid);
    }
    return eStateInvalid;
}

const char *
DNBStateAsString(nub_state_t state)
{
    switch (state)
    {
    case eStateInvalid:     return "Invalid";
    case eStateUnloaded:    return "Unloaded";
    case eStateAttaching:   return "Attaching";
    case eStateLaunching:   return "Launching";
    case eStateStopped:     return "Stopped";
    case eStateRunning:     return "Running";
    case eStateStepping:    return "Stepping";
    case eStateCrashed:     return "Crashed";
    case eStateDetached:    return "Detached";
    case eStateExited:      return "Exited";
    case eStateSuspended:   return "Suspended";
    }
    return "nub_state_t ???";
}

const char *
DNBProcessGetExecutablePath (nub_process_t pid)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
    {
        return procSP->Path();
    }
    return NULL;
}

nub_size_t
DNBProcessGetArgumentCount (nub_process_t pid)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
    {
        return procSP->ArgumentCount();
    }
    return 0;
}

const char *
DNBProcessGetArgumentAtIndex (nub_process_t pid, nub_size_t idx)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
    {
        return procSP->ArgumentAtIndex (idx);
    }
    return NULL;
}


//----------------------------------------------------------------------
// Execution control
//----------------------------------------------------------------------
nub_bool_t
DNBProcessResume (nub_process_t pid, const DNBThreadResumeAction *actions, size_t num_actions)
{
    DNBLogThreadedIf(LOG_PROCESS, "%s(pid = %4.4x)", __FUNCTION__, pid);
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
    {
        DNBThreadResumeActions thread_actions (actions, num_actions);

        // Below we add a default thread plan just in case one wasn't
        // provided so all threads always know what they were supposed to do
        if (thread_actions.IsEmpty())
        {
            // No thread plans were given, so the default it to run all threads
            thread_actions.SetDefaultThreadActionIfNeeded (eStateRunning, 0);
        }
        else
        {
            // Some thread plans were given which means anything that wasn't
            // specified should remain stopped.
            thread_actions.SetDefaultThreadActionIfNeeded (eStateStopped, 0);
        }
        return procSP->Resume (thread_actions);
    }
    return false;
}

nub_bool_t
DNBProcessHalt (nub_process_t pid)
{
    DNBLogThreadedIf(LOG_PROCESS, "%s(pid = %4.4x)", __FUNCTION__, pid);
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        return procSP->Signal (SIGSTOP);
    return false;
}
//
//nub_bool_t
//DNBThreadResume (nub_process_t pid, nub_thread_t tid, nub_bool_t step)
//{
//    DNBLogThreadedIf(LOG_THREAD, "%s(pid = %4.4x, tid = %4.4x, step = %u)", __FUNCTION__, pid, tid, (uint32_t)step);
//    MachProcessSP procSP;
//    if (GetProcessSP (pid, procSP))
//    {
//        return procSP->Resume(tid, step, 0);
//    }
//    return false;
//}
//
//nub_bool_t
//DNBThreadResumeWithSignal (nub_process_t pid, nub_thread_t tid, nub_bool_t step, int signal)
//{
//    DNBLogThreadedIf(LOG_THREAD, "%s(pid = %4.4x, tid = %4.4x, step = %u, signal = %i)", __FUNCTION__, pid, tid, (uint32_t)step, signal);
//    MachProcessSP procSP;
//    if (GetProcessSP (pid, procSP))
//    {
//        return procSP->Resume(tid, step, signal);
//    }
//    return false;
//}

nub_event_t
DNBProcessWaitForEvents (nub_process_t pid, nub_event_t event_mask, bool wait_for_set, struct timespec* timeout)
{
    nub_event_t result = 0;
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
    {
        if (wait_for_set)
            result = procSP->Events().WaitForSetEvents(event_mask, timeout);
        else
            result = procSP->Events().WaitForEventsToReset(event_mask, timeout);
    }
    return result;
}

void
DNBProcessResetEvents (nub_process_t pid, nub_event_t event_mask)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        procSP->Events().ResetEvents(event_mask);
}

void
DNBProcessInterruptEvents (nub_process_t pid)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        procSP->Events().SetEvents(eEventProcessAsyncInterrupt);
}


// Breakpoints
nub_break_t
DNBBreakpointSet (nub_process_t pid, nub_addr_t addr, nub_size_t size, nub_bool_t hardware)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
    {
        return procSP->CreateBreakpoint(addr, size, hardware, THREAD_NULL);
    }
    return INVALID_NUB_BREAK_ID;
}

nub_bool_t
DNBBreakpointClear (nub_process_t pid, nub_break_t breakID)
{
    if (NUB_BREAK_ID_IS_VALID(breakID))
    {
        MachProcessSP procSP;
        if (GetProcessSP (pid, procSP))
        {
            return procSP->DisableBreakpoint(breakID, true);
        }
    }
    return false; // Failed
}

nub_ssize_t
DNBBreakpointGetHitCount (nub_process_t pid, nub_break_t breakID)
{
    if (NUB_BREAK_ID_IS_VALID(breakID))
    {
        MachProcessSP procSP;
        if (GetProcessSP (pid, procSP))
        {
            DNBBreakpoint *bp = procSP->Breakpoints().FindByID(breakID);
            if (bp)
                return bp->GetHitCount();
        }
    }
    return 0;
}

nub_ssize_t
DNBBreakpointGetIgnoreCount (nub_process_t pid, nub_break_t breakID)
{
    if (NUB_BREAK_ID_IS_VALID(breakID))
    {
        MachProcessSP procSP;
        if (GetProcessSP (pid, procSP))
        {
            DNBBreakpoint *bp = procSP->Breakpoints().FindByID(breakID);
            if (bp)
                return bp->GetIgnoreCount();
        }
    }
    return 0;
}

nub_bool_t
DNBBreakpointSetIgnoreCount (nub_process_t pid, nub_break_t breakID, nub_size_t ignore_count)
{
    if (NUB_BREAK_ID_IS_VALID(breakID))
    {
        MachProcessSP procSP;
        if (GetProcessSP (pid, procSP))
        {
            DNBBreakpoint *bp = procSP->Breakpoints().FindByID(breakID);
            if (bp)
            {
                bp->SetIgnoreCount(ignore_count);
                return true;
            }
        }
    }
    return false;
}

// Set the callback function for a given breakpoint. The callback function will
// get called as soon as the breakpoint is hit. The function will be called
// with the process ID, thread ID, breakpoint ID and the baton, and can return
//
nub_bool_t
DNBBreakpointSetCallback (nub_process_t pid, nub_break_t breakID, DNBCallbackBreakpointHit callback, void *baton)
{
    if (NUB_BREAK_ID_IS_VALID(breakID))
    {
        MachProcessSP procSP;
        if (GetProcessSP (pid, procSP))
        {
            DNBBreakpoint *bp = procSP->Breakpoints().FindByID(breakID);
            if (bp)
            {
                bp->SetCallback(callback, baton);
                return true;
            }
        }
    }
    return false;
}

//----------------------------------------------------------------------
// Dump the breakpoints stats for process PID for a breakpoint by ID.
//----------------------------------------------------------------------
void
DNBBreakpointPrint (nub_process_t pid, nub_break_t breakID)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        procSP->DumpBreakpoint(breakID);
}

//----------------------------------------------------------------------
// Watchpoints
//----------------------------------------------------------------------
nub_watch_t
DNBWatchpointSet (nub_process_t pid, nub_addr_t addr, nub_size_t size, uint32_t watch_flags, nub_bool_t hardware)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
    {
        return procSP->CreateWatchpoint(addr, size, watch_flags, hardware, THREAD_NULL);
    }
    return INVALID_NUB_WATCH_ID;
}

nub_bool_t
DNBWatchpointClear (nub_process_t pid, nub_watch_t watchID)
{
    if (NUB_WATCH_ID_IS_VALID(watchID))
    {
        MachProcessSP procSP;
        if (GetProcessSP (pid, procSP))
        {
            return procSP->DisableWatchpoint(watchID, true);
        }
    }
    return false; // Failed
}

nub_ssize_t
DNBWatchpointGetHitCount (nub_process_t pid, nub_watch_t watchID)
{
    if (NUB_WATCH_ID_IS_VALID(watchID))
    {
        MachProcessSP procSP;
        if (GetProcessSP (pid, procSP))
        {
            DNBBreakpoint *bp = procSP->Watchpoints().FindByID(watchID);
            if (bp)
                return bp->GetHitCount();
        }
    }
    return 0;
}

nub_ssize_t
DNBWatchpointGetIgnoreCount (nub_process_t pid, nub_watch_t watchID)
{
    if (NUB_WATCH_ID_IS_VALID(watchID))
    {
        MachProcessSP procSP;
        if (GetProcessSP (pid, procSP))
        {
            DNBBreakpoint *bp = procSP->Watchpoints().FindByID(watchID);
            if (bp)
                return bp->GetIgnoreCount();
        }
    }
    return 0;
}

nub_bool_t
DNBWatchpointSetIgnoreCount (nub_process_t pid, nub_watch_t watchID, nub_size_t ignore_count)
{
    if (NUB_WATCH_ID_IS_VALID(watchID))
    {
        MachProcessSP procSP;
        if (GetProcessSP (pid, procSP))
        {
            DNBBreakpoint *bp = procSP->Watchpoints().FindByID(watchID);
            if (bp)
            {
                bp->SetIgnoreCount(ignore_count);
                return true;
            }
        }
    }
    return false;
}

// Set the callback function for a given watchpoint. The callback function will
// get called as soon as the watchpoint is hit. The function will be called
// with the process ID, thread ID, watchpoint ID and the baton, and can return
//
nub_bool_t
DNBWatchpointSetCallback (nub_process_t pid, nub_watch_t watchID, DNBCallbackBreakpointHit callback, void *baton)
{
    if (NUB_WATCH_ID_IS_VALID(watchID))
    {
        MachProcessSP procSP;
        if (GetProcessSP (pid, procSP))
        {
            DNBBreakpoint *bp = procSP->Watchpoints().FindByID(watchID);
            if (bp)
            {
                bp->SetCallback(callback, baton);
                return true;
            }
        }
    }
    return false;
}

//----------------------------------------------------------------------
// Dump the watchpoints stats for process PID for a watchpoint by ID.
//----------------------------------------------------------------------
void
DNBWatchpointPrint (nub_process_t pid, nub_watch_t watchID)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        procSP->DumpWatchpoint(watchID);
}

//----------------------------------------------------------------------
// Return the number of supported hardware watchpoints.
//----------------------------------------------------------------------
uint32_t
DNBWatchpointGetNumSupportedHWP (nub_process_t pid)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        return procSP->GetNumSupportedHardwareWatchpoints();
    return 0;
}

//----------------------------------------------------------------------
// Read memory in the address space of process PID. This call will take
// care of setting and restoring permissions and breaking up the memory
// read into multiple chunks as required.
//
// RETURNS: number of bytes actually read
//----------------------------------------------------------------------
nub_size_t
DNBProcessMemoryRead (nub_process_t pid, nub_addr_t addr, nub_size_t size, void *buf)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        return procSP->ReadMemory(addr, size, buf);
    return 0;
}

//----------------------------------------------------------------------
// Write memory to the address space of process PID. This call will take
// care of setting and restoring permissions and breaking up the memory
// write into multiple chunks as required.
//
// RETURNS: number of bytes actually written
//----------------------------------------------------------------------
nub_size_t
DNBProcessMemoryWrite (nub_process_t pid, nub_addr_t addr, nub_size_t size, const void *buf)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        return procSP->WriteMemory(addr, size, buf);
    return 0;
}

nub_addr_t
DNBProcessMemoryAllocate (nub_process_t pid, nub_size_t size, uint32_t permissions)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        return procSP->Task().AllocateMemory (size, permissions);
    return 0;
}

nub_bool_t
DNBProcessMemoryDeallocate (nub_process_t pid, nub_addr_t addr)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        return procSP->Task().DeallocateMemory (addr);
    return 0;
}

//----------------------------------------------------------------------
// Find attributes of the memory region that contains ADDR for process PID,
// if possible, and return a string describing those attributes.
//
// Returns 1 if we could find attributes for this region and OUTBUF can
// be sent to the remote debugger.
//
// Returns 0 if we couldn't find the attributes for a region of memory at
// that address and OUTBUF should not be sent.
//
// Returns -1 if this platform cannot look up information about memory regions
// or if we do not yet have a valid launched process.
//
//----------------------------------------------------------------------
int
DNBProcessMemoryRegionInfo (nub_process_t pid, nub_addr_t addr, DNBRegionInfo *region_info)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        return procSP->Task().GetMemoryRegionInfo (addr, region_info);

    return -1;
}

std::string
DNBProcessGetProfileData (nub_process_t pid, DNBProfileDataScanType scanType)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        return procSP->Task().GetProfileData(scanType);
    
    return std::string("");
}

nub_bool_t
DNBProcessSetEnableAsyncProfiling (nub_process_t pid, nub_bool_t enable, uint64_t interval_usec, DNBProfileDataScanType scan_type)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
    {
        procSP->SetEnableAsyncProfiling(enable, interval_usec, scan_type);
        return true;
    }
    
    return false;    
}

//----------------------------------------------------------------------
// Formatted output that uses memory and registers from process and
// thread in place of arguments.
//----------------------------------------------------------------------
nub_size_t
DNBPrintf (nub_process_t pid, nub_thread_t tid, nub_addr_t base_addr, FILE *file, const char *format)
{
    if (file == NULL)
        return 0;
    enum printf_flags
    {
        alternate_form          = (1 << 0),
        zero_padding            = (1 << 1),
        negative_field_width    = (1 << 2),
        blank_space             = (1 << 3),
        show_sign               = (1 << 4),
        show_thousands_separator= (1 << 5),
    };

    enum printf_length_modifiers
    {
        length_mod_h            = (1 << 0),
        length_mod_hh           = (1 << 1),
        length_mod_l            = (1 << 2),
        length_mod_ll           = (1 << 3),
        length_mod_L            = (1 << 4),
        length_mod_j            = (1 << 5),
        length_mod_t            = (1 << 6),
        length_mod_z            = (1 << 7),
        length_mod_q            = (1 << 8),
    };

    nub_addr_t addr = base_addr;
    char *end_format = (char*)format + strlen(format);
    char *end = NULL;    // For strtoXXXX calls;
    std::basic_string<uint8_t> buf;
    nub_size_t total_bytes_read = 0;
    DNBDataRef data;
    const char *f;
    for (f = format; *f != '\0' && f < end_format; f++)
    {
        char ch = *f;
        switch (ch)
        {
        case '%':
            {
                f++;    // Skip the '%' character
//                int min_field_width = 0;
//                int precision = 0;
                //uint32_t flags = 0;
                uint32_t length_modifiers = 0;
                uint32_t byte_size = 0;
                uint32_t actual_byte_size = 0;
                bool is_string = false;
                bool is_register = false;
                DNBRegisterValue register_value;
                int64_t    register_offset = 0;
                nub_addr_t register_addr = INVALID_NUB_ADDRESS;

                // Create the format string to use for this conversion specification
                // so we can remove and mprintf specific flags and formatters.
                std::string fprintf_format("%");

                // Decode any flags
                switch (*f)
                {
                case '#': fprintf_format += *f++; break; //flags |= alternate_form;          break;
                case '0': fprintf_format += *f++; break; //flags |= zero_padding;            break;
                case '-': fprintf_format += *f++; break; //flags |= negative_field_width;    break;
                case ' ': fprintf_format += *f++; break; //flags |= blank_space;             break;
                case '+': fprintf_format += *f++; break; //flags |= show_sign;               break;
                case ',': fprintf_format += *f++; break; //flags |= show_thousands_separator;break;
                case '{':
                case '[':
                    {
                        // We have a register name specification that can take two forms:
                        // ${regname} or ${regname+offset}
                        //        The action is to read the register value and add the signed offset
                        //        (if any) and use that as the value to format.
                        // $[regname] or $[regname+offset]
                        //        The action is to read the register value and add the signed offset
                        //        (if any) and use the result as an address to dereference. The size
                        //        of what is dereferenced is specified by the actual byte size that
                        //        follows the minimum field width and precision (see comments below).
                        switch (*f)
                        {
                        case '{':
                        case '[':
                            {
                                char open_scope_ch = *f;
                                f++;
                                const char *reg_name = f;
                                size_t reg_name_length = strcspn(f, "+-}]");
                                if (reg_name_length > 0)
                                {
                                    std::string register_name(reg_name, reg_name_length);
                                    f += reg_name_length;
                                    register_offset = strtoll(f, &end, 0);
                                    if (f < end)
                                        f = end;
                                    if ((open_scope_ch == '{' && *f != '}') || (open_scope_ch == '[' && *f != ']'))
                                    {
                                        fprintf(file, "error: Invalid register format string. Valid formats are %%{regname} or %%{regname+offset}, %%[regname] or %%[regname+offset]\n");
                                        return total_bytes_read;
                                    }
                                    else
                                    {
                                        f++;
                                        if (DNBThreadGetRegisterValueByName(pid, tid, REGISTER_SET_ALL, register_name.c_str(), &register_value))
                                        {
                                            // Set the address to dereference using the register value plus the offset
                                            switch (register_value.info.size)
                                            {
                                            default:
                                            case 0:
                                                fprintf (file, "error: unsupported register size of %u.\n", register_value.info.size);
                                                return total_bytes_read;

                                            case 1:        register_addr = register_value.value.uint8  + register_offset; break;
                                            case 2:        register_addr = register_value.value.uint16 + register_offset; break;
                                            case 4:        register_addr = register_value.value.uint32 + register_offset; break;
                                            case 8:        register_addr = register_value.value.uint64 + register_offset; break;
                                            case 16:
                                                if (open_scope_ch == '[')
                                                {
                                                    fprintf (file, "error: register size (%u) too large for address.\n", register_value.info.size);
                                                    return total_bytes_read;
                                                }
                                                break;
                                            }

                                            if (open_scope_ch == '{')
                                            {
                                                byte_size = register_value.info.size;
                                                is_register = true;    // value is in a register

                                            }
                                            else
                                            {
                                                addr = register_addr;    // Use register value and offset as the address
                                            }
                                        }
                                        else
                                        {
                                            fprintf(file, "error: unable to read register '%s' for process %#.4x and thread %#.8" PRIx64 "\n", register_name.c_str(), pid, tid);
                                            return total_bytes_read;
                                        }
                                    }
                                }
                            }
                            break;

                        default:
                            fprintf(file, "error: %%$ must be followed by (regname + n) or [regname + n]\n");
                            return total_bytes_read;
                        }
                    }
                    break;
                }

                // Check for a minimum field width
                if (isdigit(*f))
                {
                    //min_field_width = strtoul(f, &end, 10);
                    strtoul(f, &end, 10);
                    if (end > f)
                    {
                        fprintf_format.append(f, end - f);
                        f = end;
                    }
                }


                // Check for a precision
                if (*f == '.')
                {
                    f++;
                    if (isdigit(*f))
                    {
                        fprintf_format += '.';
                        //precision = strtoul(f, &end, 10);
                        strtoul(f, &end, 10);
                        if (end > f)
                        {
                            fprintf_format.append(f, end - f);
                            f = end;
                        }
                    }
                }


                // mprintf specific: read the optional actual byte size (abs)
                // after the standard minimum field width (mfw) and precision (prec).
                // Standard printf calls you can have "mfw.prec" or ".prec", but
                // mprintf can have "mfw.prec.abs", ".prec.abs" or "..abs". This is nice
                // for strings that may be in a fixed size buffer, but may not use all bytes
                // in that buffer for printable characters.
                if (*f == '.')
                {
                    f++;
                    actual_byte_size = strtoul(f, &end, 10);
                    if (end > f)
                    {
                        byte_size = actual_byte_size;
                        f = end;
                    }
                }

                // Decode the length modifiers
                switch (*f)
                {
                case 'h':    // h and hh length modifiers
                    fprintf_format += *f++;
                    length_modifiers |= length_mod_h;
                    if (*f == 'h')
                    {
                        fprintf_format += *f++;
                        length_modifiers |= length_mod_hh;
                    }
                    break;

                case 'l': // l and ll length modifiers
                    fprintf_format += *f++;
                    length_modifiers |= length_mod_l;
                    if (*f == 'h')
                    {
                        fprintf_format += *f++;
                        length_modifiers |= length_mod_ll;
                    }
                    break;

                case 'L':    fprintf_format += *f++;    length_modifiers |= length_mod_L;    break;
                case 'j':    fprintf_format += *f++;    length_modifiers |= length_mod_j;    break;
                case 't':    fprintf_format += *f++;    length_modifiers |= length_mod_t;    break;
                case 'z':    fprintf_format += *f++;    length_modifiers |= length_mod_z;    break;
                case 'q':    fprintf_format += *f++;    length_modifiers |= length_mod_q;    break;
                }

                // Decode the conversion specifier
                switch (*f)
                {
                case '_':
                    // mprintf specific format items
                    {
                        ++f;    // Skip the '_' character
                        switch (*f)
                        {
                        case 'a':    // Print the current address
                            ++f;
                            fprintf_format += "ll";
                            fprintf_format += *f;    // actual format to show address with folows the 'a' ("%_ax")
                            fprintf (file, fprintf_format.c_str(), addr);
                            break;
                        case 'o':    // offset from base address
                            ++f;
                            fprintf_format += "ll";
                            fprintf_format += *f;    // actual format to show address with folows the 'a' ("%_ox")
                            fprintf(file, fprintf_format.c_str(), addr - base_addr);
                            break;
                        default:
                            fprintf (file, "error: unsupported mprintf specific format character '%c'.\n", *f);
                            break;
                        }
                        continue;
                    }
                    break;

                case 'D':
                case 'O':
                case 'U':
                    fprintf_format += *f;
                    if (byte_size == 0)
                        byte_size = sizeof(long int);
                    break;

                case 'd':
                case 'i':
                case 'o':
                case 'u':
                case 'x':
                case 'X':
                    fprintf_format += *f;
                    if (byte_size == 0)
                    {
                        if (length_modifiers & length_mod_hh)
                            byte_size = sizeof(char);
                        else if (length_modifiers & length_mod_h)
                            byte_size = sizeof(short);
                        else if (length_modifiers & length_mod_ll)
                            byte_size = sizeof(long long);
                        else if (length_modifiers & length_mod_l)
                            byte_size = sizeof(long);
                        else
                            byte_size = sizeof(int);
                    }
                    break;

                case 'a':
                case 'A':
                case 'f':
                case 'F':
                case 'e':
                case 'E':
                case 'g':
                case 'G':
                    fprintf_format += *f;
                    if (byte_size == 0)
                    {
                        if (length_modifiers & length_mod_L)
                            byte_size = sizeof(long double);
                        else
                            byte_size = sizeof(double);
                    }
                    break;

                case 'c':
                    if ((length_modifiers & length_mod_l) == 0)
                    {
                        fprintf_format += *f;
                        if (byte_size == 0)
                            byte_size = sizeof(char);
                        break;
                    }
                    // Fall through to 'C' modifier below...

                case 'C':
                    fprintf_format += *f;
                    if (byte_size == 0)
                        byte_size = sizeof(wchar_t);
                    break;

                case 's':
                    fprintf_format += *f;
                    if (is_register || byte_size == 0)
                        is_string = 1;
                    break;

                case 'p':
                    fprintf_format += *f;
                    if (byte_size == 0)
                        byte_size = sizeof(void*);
                    break;
                }

                if (is_string)
                {
                    std::string mem_string;
                    const size_t string_buf_len = 4;
                    char string_buf[string_buf_len+1];
                    char *string_buf_end = string_buf + string_buf_len;
                    string_buf[string_buf_len] = '\0';
                    nub_size_t bytes_read;
                    nub_addr_t str_addr = is_register ? register_addr : addr;
                    while ((bytes_read = DNBProcessMemoryRead(pid, str_addr, string_buf_len, &string_buf[0])) > 0)
                    {
                        // Did we get a NULL termination character yet?
                        if (strchr(string_buf, '\0') == string_buf_end)
                        {
                            // no NULL terminator yet, append as a std::string
                            mem_string.append(string_buf, string_buf_len);
                            str_addr += string_buf_len;
                        }
                        else
                        {
                            // yep
                            break;
                        }
                    }
                    // Append as a C-string so we don't get the extra NULL
                    // characters in the temp buffer (since it was resized)
                    mem_string += string_buf;
                    size_t mem_string_len = mem_string.size() + 1;
                    fprintf(file, fprintf_format.c_str(), mem_string.c_str());
                    if (mem_string_len > 0)
                    {
                        if (!is_register)
                        {
                            addr += mem_string_len;
                            total_bytes_read += mem_string_len;
                        }
                    }
                    else
                        return total_bytes_read;
                }
                else
                if (byte_size > 0)
                {
                    buf.resize(byte_size);
                    nub_size_t bytes_read = 0;
                    if (is_register)
                        bytes_read = register_value.info.size;
                    else
                        bytes_read = DNBProcessMemoryRead(pid, addr, buf.size(), &buf[0]);
                    if (bytes_read > 0)
                    {
                        if (!is_register)
                            total_bytes_read += bytes_read;

                        if (bytes_read == byte_size)
                        {
                            switch (*f)
                            {
                            case 'd':
                            case 'i':
                            case 'o':
                            case 'u':
                            case 'X':
                            case 'x':
                            case 'a':
                            case 'A':
                            case 'f':
                            case 'F':
                            case 'e':
                            case 'E':
                            case 'g':
                            case 'G':
                            case 'p':
                            case 'c':
                            case 'C':
                                {
                                    if (is_register)
                                        data.SetData(&register_value.value.v_uint8[0], register_value.info.size);
                                    else
                                        data.SetData(&buf[0], bytes_read);
                                    DNBDataRef::offset_t data_offset = 0;
                                    if (byte_size <= 4)
                                    {
                                        uint32_t u32 = data.GetMax32(&data_offset, byte_size);
                                        // Show the actual byte width when displaying hex
                                        fprintf(file, fprintf_format.c_str(), u32);
                                    }
                                    else if (byte_size <= 8)
                                    {
                                        uint64_t u64 = data.GetMax64(&data_offset, byte_size);
                                        // Show the actual byte width when displaying hex
                                        fprintf(file, fprintf_format.c_str(), u64);
                                    }
                                    else
                                    {
                                        fprintf(file, "error: integer size not supported, must be 8 bytes or less (%u bytes).\n", byte_size);
                                    }
                                    if (!is_register)
                                        addr += byte_size;
                                }
                                break;

                            case 's':
                                fprintf(file, fprintf_format.c_str(), buf.c_str());
                                addr += byte_size;
                                break;

                            default:
                                fprintf(file, "error: unsupported conversion specifier '%c'.\n", *f);
                                break;
                            }
                        }
                    }
                }
                else
                    return total_bytes_read;
            }
            break;

        case '\\':
            {
                f++;
                switch (*f)
                {
                case 'e': ch = '\e'; break;
                case 'a': ch = '\a'; break;
                case 'b': ch = '\b'; break;
                case 'f': ch = '\f'; break;
                case 'n': ch = '\n'; break;
                case 'r': ch = '\r'; break;
                case 't': ch = '\t'; break;
                case 'v': ch = '\v'; break;
                case '\'': ch = '\''; break;
                case '\\': ch = '\\'; break;
                case '0':
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                case '6':
                case '7':
                    ch = strtoul(f, &end, 8);
                    f = end;
                    break;
                default:
                    ch = *f;
                    break;
                }
                fputc(ch, file);
            }
            break;

        default:
            fputc(ch, file);
            break;
        }
    }
    return total_bytes_read;
}


//----------------------------------------------------------------------
// Get the number of threads for the specified process.
//----------------------------------------------------------------------
nub_size_t
DNBProcessGetNumThreads (nub_process_t pid)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        return procSP->GetNumThreads();
    return 0;
}

//----------------------------------------------------------------------
// Get the thread ID of the current thread.
//----------------------------------------------------------------------
nub_thread_t
DNBProcessGetCurrentThread (nub_process_t pid)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        return procSP->GetCurrentThread();
    return 0;
}

//----------------------------------------------------------------------
// Get the mach port number of the current thread.
//----------------------------------------------------------------------
nub_thread_t
DNBProcessGetCurrentThreadMachPort (nub_process_t pid)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        return procSP->GetCurrentThreadMachPort();
    return 0;
}

//----------------------------------------------------------------------
// Change the current thread.
//----------------------------------------------------------------------
nub_thread_t
DNBProcessSetCurrentThread (nub_process_t pid, nub_thread_t tid)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        return procSP->SetCurrentThread (tid);
    return INVALID_NUB_THREAD;
}


//----------------------------------------------------------------------
// Dump a string describing a thread's stop reason to the specified file
// handle
//----------------------------------------------------------------------
nub_bool_t
DNBThreadGetStopReason (nub_process_t pid, nub_thread_t tid, struct DNBThreadStopInfo *stop_info)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        return procSP->GetThreadStoppedReason (tid, stop_info);
    return false;
}

//----------------------------------------------------------------------
// Return string description for the specified thread.
//
// RETURNS: NULL if the thread isn't valid, else a NULL terminated C
// string from a static buffer that must be copied prior to subsequent
// calls.
//----------------------------------------------------------------------
const char *
DNBThreadGetInfo (nub_process_t pid, nub_thread_t tid)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        return procSP->GetThreadInfo (tid);
    return NULL;
}

//----------------------------------------------------------------------
// Get the thread ID given a thread index.
//----------------------------------------------------------------------
nub_thread_t
DNBProcessGetThreadAtIndex (nub_process_t pid, size_t thread_idx)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        return procSP->GetThreadAtIndex (thread_idx);
    return INVALID_NUB_THREAD;
}

//----------------------------------------------------------------------
// Do whatever is needed to sync the thread's register state with it's kernel values.
//----------------------------------------------------------------------
nub_bool_t
DNBProcessSyncThreadState (nub_process_t pid, nub_thread_t tid)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        return procSP->SyncThreadState (tid);
    return false;

}

nub_addr_t
DNBProcessGetSharedLibraryInfoAddress (nub_process_t pid)
{
    MachProcessSP procSP;
    DNBError err;
    if (GetProcessSP (pid, procSP))
        return procSP->Task().GetDYLDAllImageInfosAddress (err);
    return INVALID_NUB_ADDRESS;
}


nub_bool_t
DNBProcessSharedLibrariesUpdated(nub_process_t pid)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
    {
        procSP->SharedLibrariesUpdated ();
        return true;
    }
    return false;
}

//----------------------------------------------------------------------
// Get the current shared library information for a process. Only return
// the shared libraries that have changed since the last shared library
// state changed event if only_changed is non-zero.
//----------------------------------------------------------------------
nub_size_t
DNBProcessGetSharedLibraryInfo (nub_process_t pid, nub_bool_t only_changed, struct DNBExecutableImageInfo **image_infos)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        return procSP->CopyImageInfos (image_infos, only_changed);

    // If we have no process, then return NULL for the shared library info
    // and zero for shared library count
    *image_infos = NULL;
    return 0;
}

//----------------------------------------------------------------------
// Get the register set information for a specific thread.
//----------------------------------------------------------------------
const DNBRegisterSetInfo *
DNBGetRegisterSetInfo (nub_size_t *num_reg_sets)
{
    return DNBArchProtocol::GetRegisterSetInfo (num_reg_sets);
}


//----------------------------------------------------------------------
// Read a register value by register set and register index.
//----------------------------------------------------------------------
nub_bool_t
DNBThreadGetRegisterValueByID (nub_process_t pid, nub_thread_t tid, uint32_t set, uint32_t reg, DNBRegisterValue *value)
{
    MachProcessSP procSP;
    ::bzero (value, sizeof(DNBRegisterValue));
    if (GetProcessSP (pid, procSP))
    {
        if (tid != INVALID_NUB_THREAD)
            return procSP->GetRegisterValue (tid, set, reg, value);
    }
    return false;
}

nub_bool_t
DNBThreadSetRegisterValueByID (nub_process_t pid, nub_thread_t tid, uint32_t set, uint32_t reg, const DNBRegisterValue *value)
{
    if (tid != INVALID_NUB_THREAD)
    {
        MachProcessSP procSP;
        if (GetProcessSP (pid, procSP))
            return procSP->SetRegisterValue (tid, set, reg, value);
    }
    return false;
}

nub_size_t
DNBThreadGetRegisterContext (nub_process_t pid, nub_thread_t tid, void *buf, size_t buf_len)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
    {
        if (tid != INVALID_NUB_THREAD)
            return procSP->GetThreadList().GetRegisterContext (tid, buf, buf_len);
    }
    ::bzero (buf, buf_len);
    return 0;

}

nub_size_t
DNBThreadSetRegisterContext (nub_process_t pid, nub_thread_t tid, const void *buf, size_t buf_len)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
    {
        if (tid != INVALID_NUB_THREAD)
            return procSP->GetThreadList().SetRegisterContext (tid, buf, buf_len);
    }
    return 0;
}

//----------------------------------------------------------------------
// Read a register value by name.
//----------------------------------------------------------------------
nub_bool_t
DNBThreadGetRegisterValueByName (nub_process_t pid, nub_thread_t tid, uint32_t reg_set, const char *reg_name, DNBRegisterValue *value)
{
    MachProcessSP procSP;
    ::bzero (value, sizeof(DNBRegisterValue));
    if (GetProcessSP (pid, procSP))
    {
        const struct DNBRegisterSetInfo *set_info;
        nub_size_t num_reg_sets = 0;
        set_info = DNBGetRegisterSetInfo (&num_reg_sets);
        if (set_info)
        {
            uint32_t set = reg_set;
            uint32_t reg;
            if (set == REGISTER_SET_ALL)
            {
                for (set = 1; set < num_reg_sets; ++set)
                {
                    for (reg = 0; reg < set_info[set].num_registers; ++reg)
                    {
                        if (strcasecmp(reg_name, set_info[set].registers[reg].name) == 0)
                            return procSP->GetRegisterValue (tid, set, reg, value);
                    }
                }
            }
            else
            {
                for (reg = 0; reg < set_info[set].num_registers; ++reg)
                {
                    if (strcasecmp(reg_name, set_info[set].registers[reg].name) == 0)
                        return procSP->GetRegisterValue (tid, set, reg, value);
                }
            }
        }
    }
    return false;
}


//----------------------------------------------------------------------
// Read a register set and register number from the register name.
//----------------------------------------------------------------------
nub_bool_t
DNBGetRegisterInfoByName (const char *reg_name, DNBRegisterInfo* info)
{
    const struct DNBRegisterSetInfo *set_info;
    nub_size_t num_reg_sets = 0;
    set_info = DNBGetRegisterSetInfo (&num_reg_sets);
    if (set_info)
    {
        uint32_t set, reg;
        for (set = 1; set < num_reg_sets; ++set)
        {
            for (reg = 0; reg < set_info[set].num_registers; ++reg)
            {
                if (strcasecmp(reg_name, set_info[set].registers[reg].name) == 0)
                {
                    *info = set_info[set].registers[reg];
                    return true;
                }
            }
        }

        for (set = 1; set < num_reg_sets; ++set)
        {
            uint32_t reg;
            for (reg = 0; reg < set_info[set].num_registers; ++reg)
            {
                if (set_info[set].registers[reg].alt == NULL)
                    continue;

                if (strcasecmp(reg_name, set_info[set].registers[reg].alt) == 0)
                {
                    *info = set_info[set].registers[reg];
                    return true;
                }
            }
        }
    }

    ::bzero (info, sizeof(DNBRegisterInfo));
    return false;
}


//----------------------------------------------------------------------
// Set the name to address callback function that this nub can use
// for any name to address lookups that are needed.
//----------------------------------------------------------------------
nub_bool_t
DNBProcessSetNameToAddressCallback (nub_process_t pid, DNBCallbackNameToAddress callback, void *baton)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
    {
        procSP->SetNameToAddressCallback (callback, baton);
        return true;
    }
    return false;
}


//----------------------------------------------------------------------
// Set the name to address callback function that this nub can use
// for any name to address lookups that are needed.
//----------------------------------------------------------------------
nub_bool_t
DNBProcessSetSharedLibraryInfoCallback (nub_process_t pid, DNBCallbackCopyExecutableImageInfos callback, void  *baton)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
    {
        procSP->SetSharedLibraryInfoCallback (callback, baton);
        return true;
    }
    return false;
}

nub_addr_t
DNBProcessLookupAddress (nub_process_t pid, const char *name, const char *shlib)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
    {
        return procSP->LookupSymbol (name, shlib);
    }
    return INVALID_NUB_ADDRESS;
}


nub_size_t
DNBProcessGetAvailableSTDOUT (nub_process_t pid, char *buf, nub_size_t buf_size)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        return procSP->GetAvailableSTDOUT (buf, buf_size);
    return 0;
}

nub_size_t
DNBProcessGetAvailableSTDERR (nub_process_t pid, char *buf, nub_size_t buf_size)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        return procSP->GetAvailableSTDERR (buf, buf_size);
    return 0;
}

nub_size_t
DNBProcessGetAvailableProfileData (nub_process_t pid, char *buf, nub_size_t buf_size)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        return procSP->GetAsyncProfileData (buf, buf_size);
    return 0;
}

nub_size_t
DNBProcessGetStopCount (nub_process_t pid)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        return procSP->StopCount();
    return 0;
}

uint32_t
DNBProcessGetCPUType (nub_process_t pid)
{
    MachProcessSP procSP;
    if (GetProcessSP (pid, procSP))
        return procSP->GetCPUType ();
    return 0;
    
}

nub_bool_t
DNBResolveExecutablePath (const char *path, char *resolved_path, size_t resolved_path_size)
{
    if (path == NULL || path[0] == '\0')
        return false;

    char max_path[PATH_MAX];
    std::string result;
    CFString::GlobPath(path, result);

    if (result.empty())
        result = path;
    
    struct stat path_stat;
    if (::stat(path, &path_stat) == 0)
    {
        if ((path_stat.st_mode & S_IFMT) == S_IFDIR)
        {
            CFBundle bundle (path);
            CFReleaser<CFURLRef> url(bundle.CopyExecutableURL ());
            if (url.get())
            {
                if (::CFURLGetFileSystemRepresentation (url.get(), true, (UInt8*)resolved_path, resolved_path_size))
                    return true;
            }
        }
    }

    if (realpath(path, max_path))
    {
        // Found the path relatively...
        ::strncpy(resolved_path, max_path, resolved_path_size);
        return strlen(resolved_path) + 1 < resolved_path_size;
    }
    else
    {
        // Not a relative path, check the PATH environment variable if the
        const char *PATH = getenv("PATH");
        if (PATH)
        {
            const char *curr_path_start = PATH;
            const char *curr_path_end;
            while (curr_path_start && *curr_path_start)
            {
                curr_path_end = strchr(curr_path_start, ':');
                if (curr_path_end == NULL)
                {
                    result.assign(curr_path_start);
                    curr_path_start = NULL;
                }
                else if (curr_path_end > curr_path_start)
                {
                    size_t len = curr_path_end - curr_path_start;
                    result.assign(curr_path_start, len);
                    curr_path_start += len + 1;
                }
                else
                    break;

                result += '/';
                result += path;
                struct stat s;
                if (stat(result.c_str(), &s) == 0)
                {
                    ::strncpy(resolved_path, result.c_str(), resolved_path_size);
                    return result.size() + 1 < resolved_path_size;
                }
            }
        }
    }
    return false;
}


void
DNBInitialize()
{
    DNBLogThreadedIf (LOG_PROCESS, "DNBInitialize ()");
#if defined (__i386__) || defined (__x86_64__)
    DNBArchImplI386::Initialize();
    DNBArchImplX86_64::Initialize();
#elif defined (__arm__)
    DNBArchMachARM::Initialize();
#endif
}

void
DNBTerminate()
{
}

nub_bool_t
DNBSetArchitecture (const char *arch)
{
    if (arch && arch[0])
    {
        if (strcasecmp (arch, "i386") == 0)
            return DNBArchProtocol::SetArchitecture (CPU_TYPE_I386);
        else if (strcasecmp (arch, "x86_64") == 0)
            return DNBArchProtocol::SetArchitecture (CPU_TYPE_X86_64);
        else if (strstr (arch, "arm") == arch)
            return DNBArchProtocol::SetArchitecture (CPU_TYPE_ARM);
    }
    return false;
}
