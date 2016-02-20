//===-- lldb-server.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Initialization/SystemLifetimeManager.h"
#include "lldb/Initialization/SystemInitializerCommon.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ManagedStatic.h"

#include <stdio.h>
#include <stdlib.h>

static llvm::ManagedStatic<lldb_private::SystemLifetimeManager> g_debugger_lifetime;

static void
display_usage (const char *progname)
{
    fprintf(stderr, "Usage:\n"
            "  %s g[dbserver] [options]\n"
            "  %s p[latform] [options]\n"
            "Invoke subcommand for additional help\n", progname, progname);
    exit(0);
}

// Forward declarations of subcommand main methods.
int main_gdbserver (int argc, char *argv[]);
int main_platform (int argc, char *argv[]);

static void
initialize ()
{
    g_debugger_lifetime->Initialize(llvm::make_unique<lldb_private::SystemInitializerCommon>(), nullptr);
}

static void
terminate ()
{
    g_debugger_lifetime->Terminate();
}

//----------------------------------------------------------------------
// main
//----------------------------------------------------------------------
int
main (int argc, char *argv[])
{
    int option_error = 0;
    const char *progname = argv[0];
    if (argc < 2)
    {
        display_usage(progname);
        exit(option_error);
    }
    else if (argv[1][0] == 'g')
    {
        initialize();
        main_gdbserver(argc, argv);
        terminate();
    }
    else if (argv[1][0] == 'p')
    {
        initialize();
        main_platform(argc, argv);
        terminate();
    }
    else {
        display_usage(progname);
        exit(option_error);
    }
}
