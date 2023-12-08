#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

void handler(int signo)
{
    _exit(signo);
}

int main (int argc, char *argv[])
{
    if (signal(SIGTRAP, handler) == SIG_ERR)
    {
        perror("signal(SIGTRAP)");
        return 1;
    }
#ifndef __APPLE__
    // Real time signals not supported on apple platforms.
    if (signal(SIGRTMIN, handler) == SIG_ERR)
    {
        perror("signal(SIGRTMIN)");
        return 1;
    }
#endif

    if (argc < 2)
    {
        puts("Please specify a signal to raise");
        return 1;
    }

    if (strcmp(argv[1], "SIGSTOP") == 0)
        raise(SIGSTOP);
    else if (strcmp(argv[1], "SIGTRAP") == 0)
        raise(SIGTRAP);
#ifndef __APPLE__
    else if (strcmp(argv[1], "SIGRTMIN") == 0)
        raise(SIGRTMIN);
#endif
    else
    {
        printf("Unknown signal: %s\n", argv[1]);
        return 1;
    }

    return 0;
}

