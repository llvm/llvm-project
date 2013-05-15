//===-- lldb-platform.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

// C Includes
#include <errno.h>
#include <getopt.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// C++ Includes

// Other libraries and framework includes
#include "lldb/Core/Error.h"
#include "lldb/Core/ConnectionFileDescriptor.h"
#include "lldb/Core/ConnectionMachPort.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/StreamFile.h"
#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServer.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// option descriptors for getopt_long_only()
//----------------------------------------------------------------------

int g_debug = 0;
int g_verbose = 0;

static struct option g_long_options[] =
{
    { "debug",              no_argument,        &g_debug,           1   },
    { "verbose",            no_argument,        &g_verbose,         1   },
    { "log-file",           required_argument,  NULL,               'l' },
    { "log-flags",          required_argument,  NULL,               'f' },
    { "listen",             required_argument,  NULL,               'L' },
    { NULL,                 0,                  NULL,               0   }
};

//----------------------------------------------------------------------
// Watch for signals
//----------------------------------------------------------------------
int g_sigpipe_received = 0;
void
signal_handler(int signo)
{
    switch (signo)
    {
    case SIGPIPE:
        g_sigpipe_received = 1;
        break;
    }
}

//----------------------------------------------------------------------
// main
//----------------------------------------------------------------------
int
main (int argc, char *argv[])
{
    signal (SIGPIPE, signal_handler);
    int long_option_index = 0;
    StreamSP log_stream_sp;
    Args log_args;
    Error error;
    std::string listen_host_port;
    int ch;
    Debugger::Initialize();
    
//    ConnectionMachPort a;
//    ConnectionMachPort b;
//
//    lldb::ConnectionStatus status;
//    const char *bootstrap_service_name = "HelloWorld";
//    status = a.BootstrapCheckIn(bootstrap_service_name, &error);
//    
//    if (status != eConnectionStatusSuccess)
//    {
//        fprintf(stderr, "%s", error.AsCString());
//        return 1;
//    }
//    status = b.BootstrapLookup (bootstrap_service_name, &error);
//    if (status != eConnectionStatusSuccess)
//    {
//        fprintf(stderr, "%s", error.AsCString());
//        return 2;
//    }
//    
//    if (a.Write ("hello", 5, status, &error) == 5)
//    {
//        char buf[32];
//        memset(buf, 0, sizeof(buf));
//        if (b.Read (buf, 5, status, &error))
//        {
//            printf("read returned bytes: %s", buf);
//        }
//        else
//        {
//            fprintf(stderr, "%s", error.AsCString());
//            return 4;
//        }
//    }
//    else
//    {
//        fprintf(stderr, "%s", error.AsCString());
//        return 3;
//    }
    
    while ((ch = getopt_long_only(argc, argv, "l:f:L:", g_long_options, &long_option_index)) != -1)
    {
//        DNBLogDebug("option: ch == %c (0x%2.2x) --%s%c%s\n",
//                    ch, (uint8_t)ch,
//                    g_long_options[long_option_index].name,
//                    g_long_options[long_option_index].has_arg ? '=' : ' ',
//                    optarg ? optarg : "");
        switch (ch)
        {
        case 0:   // Any optional that auto set themselves will return 0
            break;

        case 'l': // Set Log File
            if (optarg && optarg[0])
            {
                if ((strcasecmp(optarg, "stdout") == 0) || (strcmp(optarg, "/dev/stdout") == 0))
                {
                    log_stream_sp.reset (new StreamFile (stdout, false));
                }
                else if ((strcasecmp(optarg, "stderr") == 0) || (strcmp(optarg, "/dev/stderr") == 0))
                {
                    log_stream_sp.reset (new StreamFile (stderr, false));
                }
                else
                {
                    FILE *log_file = fopen(optarg, "w");
                    if (log_file)
                    {
                        setlinebuf(log_file);
                        log_stream_sp.reset (new StreamFile (log_file, true));
                    }
                    else
                    {
                        const char *errno_str = strerror(errno);
                        fprintf (stderr, "Failed to open log file '%s' for writing: errno = %i (%s)", optarg, errno, errno_str ? errno_str : "unknown error");
                    }

                }
                
            }
            break;

        case 'f': // Log Flags
            if (optarg && optarg[0])
                log_args.AppendArgument(optarg);
            break;
        
        case 'L':
            listen_host_port.append (optarg);
            break;
        }
    }
    
    if (log_stream_sp)
    {
        if (log_args.GetArgumentCount() == 0)
            log_args.AppendArgument("default");
        ProcessGDBRemoteLog::EnableLog (log_stream_sp, 0,log_args.GetConstArgumentVector(), log_stream_sp.get());
    }

    // Skip any options we consumed with getopt_long_only
    argc -= optind;
    argv += optind;


    GDBRemoteCommunicationServer gdb_server (true);
    if (!listen_host_port.empty())
    {
        std::unique_ptr<ConnectionFileDescriptor> conn_ap(new ConnectionFileDescriptor());
        if (conn_ap.get())
        {
            std::string connect_url ("listen://");
            connect_url.append(listen_host_port.c_str());

            printf ("Listening for a connection on %s...\n", listen_host_port.c_str());
            if (conn_ap->Connect(connect_url.c_str(), &error) == eConnectionStatusSuccess)
            {
                printf ("Connection established.\n");
                gdb_server.SetConnection (conn_ap.release());
            }
        }
    }


    if (gdb_server.IsConnected())
    {
        // After we connected, we need to get an initial ack from...
        if (gdb_server.HandshakeWithClient(&error))
        {
            bool interrupt = false;
            bool done = false;
            while (!interrupt && !done)
            {
                if (!gdb_server.GetPacketAndSendResponse (UINT32_MAX, error, interrupt, done))
                    break;
            }
            
            if (error.Fail())
            {
                fprintf(stderr, "error: %s\n", error.AsCString());
            }
        }
        else
        {
            fprintf(stderr, "error: handshake with client failed\n");
        }
    }

    Debugger::Terminate();

    return 0;
}
