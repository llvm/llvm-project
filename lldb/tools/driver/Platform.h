//===-- Platform.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Platform_h_
#define lldb_Platform_h_

#include "lldb/Host/HostGetOpt.h"

#if defined( _WIN32 )

    // this will stop signal.h being included
    #define _INC_SIGNAL
    #include <io.h>
#if defined( _MSC_VER )
    #include <eh.h>
#endif
    #include <inttypes.h>
    #include "lldb/Host/windows/windows.h"

    struct winsize
    {
        long ws_col;
    };

    typedef unsigned char   cc_t;
    typedef unsigned int    speed_t;
    typedef unsigned int    tcflag_t;

    // fcntl.h
    #define O_NOCTTY 0400

    // ioctls.h
    #define TIOCGWINSZ 0x5413


    // signal handler function pointer type
    typedef void(*sighandler_t)(int);

    // signal.h
    #define SIGINT 2
    // default handler
    #define SIG_DFL ( (sighandler_t) -1 )
    // ignored
    #define SIG_IGN ( (sighandler_t) -2 )

    // signal.h
    #define SIGPIPE  13
    #define SIGCONT  18
    #define SIGTSTP  20
    #define SIGWINCH 28

    // tcsetattr arguments
    #define TCSANOW 0

    #define NCCS 32
    struct termios
    {
        tcflag_t c_iflag;  // input mode flags
        tcflag_t c_oflag;  // output mode flags
        tcflag_t c_cflag;  // control mode flags
        tcflag_t c_lflag;  // local mode flags
        cc_t c_line;       // line discipline
        cc_t c_cc[NCCS];   // control characters
        speed_t c_ispeed;  // input speed
        speed_t c_ospeed;  // output speed
    };



#ifdef _MSC_VER
    struct timeval
    {
        long tv_sec;
        long tv_usec;
    };
    typedef long pid_t;
    #define snprintf _snprintf
    extern sighandler_t signal( int sig, sighandler_t );
    #define PATH_MAX MAX_PATH
#endif

    #define STDIN_FILENO 0

    extern int  ioctl( int d, int request, ... );
    extern int  kill ( pid_t pid, int sig      );
    extern int  tcsetattr( int fd, int optional_actions, const struct termios *termios_p );
    extern int  tcgetattr( int fildes, struct termios *termios_p );

#else

    #include <inttypes.h>

    #include <libgen.h>
    #include <sys/ioctl.h>
    #include <termios.h>
    #include <unistd.h>

    #include <pthread.h>
    #include <sys/time.h>
#endif

#endif // lldb_Platform_h_
