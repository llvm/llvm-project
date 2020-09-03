#include <stdio.h>
#include <sys/types.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#include <signal.h>
#endif

// This simple program is to test the lldb Python API related to process.

int main (int argc, char const *argv[])
{
#if defined(_WIN32)
		::ExitProcess(1);
#else
    kill(getpid(), SIGINT); // Set break point at this line and setup signal ignores.
#endif
    return 0;
}
