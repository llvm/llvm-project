#include <stdio.h>
#include <string.h>
#include <unistd.h>

int 
main (int argc, char **argv)
{
    lldb_enable_attach();

    int do_crash = 0;
    int do_wait = 0;

    int idx;
    for (idx = 1; idx < argc; idx++)
    {
        if (strcmp(argv[idx], "CRASH") == 0)
            do_crash = 1;
        if (strcmp(argv[idx], "WAIT") == 0)
            do_wait = 1;
    }
    printf("PID: %d END\n", getpid());

    if (do_wait)
    {
        int keep_waiting = 1;
        while (keep_waiting)
        {
            printf ("Waiting\n");
            sleep(1); // Stop here to unset keep_waiting
        }
    }

    if (do_crash)
    {
      char *touch_me_not = (char *) 0;
      printf ("About to crash.\n");
      touch_me_not[0] = 'a';
    }
    printf ("Got there on time and it did not crash.\n");
    return 0;
}
