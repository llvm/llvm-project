#include <stdlib.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

void handler (int in)
{
    puts ("in handler routine");
    while (1)
        ;
}

void
foo ()
{
    puts ("in foo ()");
    kill (getpid(), SIGUSR1);
}
int main ()
{
    puts ("in main");           // Set breakpoint here
    signal (SIGUSR1, handler);
    puts ("signal handler set up");
    foo();
    puts ("exiting");
    return 0;
}
