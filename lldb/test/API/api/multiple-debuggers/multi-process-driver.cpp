
// This program creates NUMBER_OF_SIMULTANEOUS_DEBUG_SESSIONS of pthreads, 
// creates an lldb Debugger on each thread, creates targets, inserts two
// breakpoints, runs to the first breakpoint, backtraces, runs to the second
// breakpoint, backtraces, kills the inferior process, closes down the
// debugger.

// The main thread keeps track of which pthreads have completed and which 
// pthreads have completed successfully, and exits when all pthreads have
// completed successfully, or our time limit has been exceeded.

// This test file helps to uncover race conditions and locking mistakes
// that are hit when lldb is being used to debug multiple processes
// simultaneously.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lldb/API/LLDB.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBDebugger.h"

#include <chrono>
#include <thread>

#define NUMBER_OF_SIMULTANEOUS_DEBUG_SESSIONS 10

#define DEBUG 0

using namespace lldb;

bool *completed_threads_array = 0;
bool *successful_threads_array  = 0;

const char *inferior_process_name = "testprog";

bool
wait_for_stop_event (SBProcess process, SBListener listener)
{
    bool stopped = false;
    while (!stopped)
    {
        SBEvent event;
        bool waitfor_ret = listener.WaitForEvent (2, event);
        if (event.GetType() == SBProcess::eBroadcastBitStateChanged)
        {
            if (process.GetState() == StateType::eStateStopped
                || process.GetState() == StateType::eStateCrashed
                || process.GetState() == StateType::eStateDetached
                || process.GetState() == StateType::eStateExited)
            {
                stopped = true;
            }
        }
    }
    return stopped;
}

bool
walk_stack_to_main (SBThread thread)
{
    if (thread.IsValid() == 0)
    {
        return false;
    }

    bool found_main = false;
    uint32_t curr_frame = 0;
    const uint32_t framecount = thread.GetNumFrames();
    while (!found_main && curr_frame < framecount)
    {
        SBFrame frame = thread.GetFrameAtIndex (curr_frame);
        if (strcmp (frame.GetFunctionName(), "main") == 0)
        {
            found_main = true;
            break;
        }
        curr_frame += 1;
    }
    return found_main;
}

void *do_one_debugger (void *in)
{
    uint64_t threadnum = (uint64_t) in;

#if defined (__APPLE__)
    char *threadname;
    asprintf (&threadname, "thread #%lld", threadnum);
    pthread_setname_np (threadname);
    free (threadname);
#endif

#if DEBUG == 1
    printf ("#%lld: Starting debug session\n", threadnum);
#endif

    SBDebugger debugger = lldb::SBDebugger::Create (false);
    if (debugger.IsValid ())
    {
        debugger.SetAsync (true);
        SBTarget target = debugger.CreateTargetWithFileAndArch(inferior_process_name, "x86_64");
        SBCommandInterpreter command_interp = debugger.GetCommandInterpreter();
        if (target.IsValid())
        {
            SBBreakpoint bar_br = target.BreakpointCreateByName ("bar", "testprog");
            if (!bar_br.IsValid())
            {
                printf ("#%lld: failed to set breakpoint on bar, exiting.\n", threadnum);
                exit (1);
            }
            SBBreakpoint foo_br = target.BreakpointCreateByName ("foo", "testprog");
            if (!foo_br.IsValid())
            {
                printf ("#%lld: Failed to set breakpoint on foo()\n", threadnum);
            }

            SBLaunchInfo launch_info (NULL);
            SBError error;
            SBProcess process = target.Launch (launch_info, error);
            if (process.IsValid())
            {
                SBListener listener = debugger.GetListener();
                SBBroadcaster broadcaster = process.GetBroadcaster();
                uint32_t rc = broadcaster.AddListener (listener, SBProcess::eBroadcastBitStateChanged);
                if (rc == 0)
                {
                    printf ("adding listener failed\n");
                    exit (1);
                }

                wait_for_stop_event (process, listener);

                if (!walk_stack_to_main (process.GetThreadAtIndex(0)))
                {
                    printf ("#%lld: backtrace while @ foo() failed\n", threadnum);
                    completed_threads_array[threadnum] = true;
                    return (void *) 1;
                }

                if (strcmp (process.GetThreadAtIndex(0).GetFrameAtIndex(0).GetFunctionName(), "foo") != 0)
                {
#if DEBUG == 1
                    printf ("#%lld: First breakpoint did not stop at foo(), instead stopped at '%s'\n", threadnum, process.GetThreadAtIndex(0).GetFrameAtIndex(0).GetFunctionName());
#endif
                    completed_threads_array[threadnum] = true;
                    return (void*) 1;
                }

                process.Continue();

                wait_for_stop_event (process, listener);

                if (process.GetState() == StateType::eStateExited)
                {
                    printf ("#%lld: Process exited\n", threadnum);
                    completed_threads_array[threadnum] = true;
                    return (void *) 1;
                }


                if (!walk_stack_to_main (process.GetThreadAtIndex(0)))
                {
                    printf ("#%lld: backtrace while @ bar() failed\n", threadnum);
                    completed_threads_array[threadnum] = true;
                    return (void *) 1;
                }

                if (strcmp (process.GetThreadAtIndex(0).GetFrameAtIndex(0).GetFunctionName(), "bar") != 0)
                {
                    printf ("#%lld: First breakpoint did not stop at bar()\n", threadnum);
                    completed_threads_array[threadnum] = true;
                    return (void*) 1;
                }

                process.Kill();

                wait_for_stop_event (process, listener);

                SBDebugger::Destroy(debugger);

#if DEBUG == 1
                printf ("#%lld: All good!\n", threadnum);
#endif
                successful_threads_array[threadnum] = true;
                completed_threads_array[threadnum] = true;
                return (void*) 0;
            }
            else
            {
                printf("#%lld: process failed to launch\n", threadnum);
                successful_threads_array[threadnum] = false;
                completed_threads_array[threadnum] = true;
                return (void*) 0;
            }
        }
        else
        {
            printf ("#%lld: did not get valid target\n", threadnum);
            successful_threads_array[threadnum] = false;
            completed_threads_array[threadnum] = true;
            return (void*) 0;
        }
    }
    else
    {
        printf ("#%lld: did not get debugger\n", threadnum);
        successful_threads_array[threadnum] = false;
        completed_threads_array[threadnum] = true;
        return (void*) 0;
    }
    completed_threads_array[threadnum] = true;
    return (void*) 1;
}

int main (int argc, char **argv)
{
#if !defined(_MSC_VER)
  signal(SIGPIPE, SIG_IGN);
#endif
  
    SBDebugger::Initialize();

    completed_threads_array = (bool *) malloc (sizeof (bool) * NUMBER_OF_SIMULTANEOUS_DEBUG_SESSIONS);
    memset (completed_threads_array, 0, sizeof (bool) * NUMBER_OF_SIMULTANEOUS_DEBUG_SESSIONS);
    successful_threads_array = (bool *) malloc (sizeof (bool) * NUMBER_OF_SIMULTANEOUS_DEBUG_SESSIONS);
    memset (successful_threads_array, 0, sizeof (bool) * NUMBER_OF_SIMULTANEOUS_DEBUG_SESSIONS);

    if (argc > 1 && argv[1] != NULL)
    {
        inferior_process_name = argv[1];
    }

    std::thread threads[NUMBER_OF_SIMULTANEOUS_DEBUG_SESSIONS];
    for (uint64_t i = 0; i< NUMBER_OF_SIMULTANEOUS_DEBUG_SESSIONS; i++)
    {
        threads[i] = std::move(std::thread(do_one_debugger, (void*)i));
    }


    int max_time_to_wait = 20;  // 20 iterations, or 60 seconds
    int iter = 0;
    while (1)
    {
        std::this_thread::sleep_for(std::chrono::seconds(3));
        bool all_done = true;
        int successful_threads = 0;
        int total_completed_threads = 0;
        for (uint64_t i = 0; i < NUMBER_OF_SIMULTANEOUS_DEBUG_SESSIONS; i++)
        {
            if (successful_threads_array[i] == true)
                successful_threads++;
            if (completed_threads_array[i] == true)
                total_completed_threads++;
            if (completed_threads_array[i] == false)
            {
                all_done = false;
            }
        }
        if (all_done)
        {
#if DEBUG == 1
            printf ("All threads completed.\n");
            printf ("%d threads completed successfully out of %d\n", successful_threads, NUMBER_OF_SIMULTANEOUS_DEBUG_SESSIONS);
#endif
            SBDebugger::Terminate();
            exit(0);
        }
        else
        {
#if DEBUG == 1
            printf ("%d threads completed so far (%d successfully), out of %d\n", total_completed_threads, successful_threads, NUMBER_OF_SIMULTANEOUS_DEBUG_SESSIONS);
#endif
        }
        if (iter++ == max_time_to_wait)
        {
            printf ("reached maximum timeout but only %d threads have completed so far (%d successfully), out of %d.  Exiting.\n", total_completed_threads, successful_threads, NUMBER_OF_SIMULTANEOUS_DEBUG_SESSIONS);
            break;
        }
    }


    SBDebugger::Terminate();
    exit (1);
}
