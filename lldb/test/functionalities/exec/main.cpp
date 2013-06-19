#include <errno.h>
#include <mach/mach.h>
#include <signal.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <spawn.h>
#include <unistd.h>

static void
exit_with_errno (int err, const char *prefix)
{
    if (err)
    {
        fprintf (stderr,
                 "%s%s",
                 prefix ? prefix : "",
                 strerror(err));
        exit (err);
    }
}

static pid_t
spawn_process (const char **argv,
               const char **envp,
               cpu_type_t cpu_type,
               int &err)
{
    pid_t pid = 0;

    const posix_spawn_file_actions_t *file_actions = NULL;
    posix_spawnattr_t attr;
    err = posix_spawnattr_init (&attr);
    if (err)
        return pid;

    short flags = POSIX_SPAWN_SETEXEC | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK;
    err = posix_spawnattr_setflags (&attr, flags);
    if (err == 0)
    {
        // Use the default signal masks
        sigset_t no_signals;
        sigset_t all_signals;
        sigemptyset (&no_signals);
        sigfillset (&all_signals);
        posix_spawnattr_setsigmask(&attr, &no_signals);
        posix_spawnattr_setsigdefault(&attr, &all_signals);

        if (cpu_type != 0)
        {
            size_t ocount = 0;
            err = posix_spawnattr_setbinpref_np (&attr, 1, &cpu_type, &ocount);
        }

        if (err == 0)
        {
            err = posix_spawn (&pid,
                               argv[0],
                               file_actions,
                               &attr,
                               (char * const *)argv,
                               (char * const *)envp);
        }
        
        posix_spawnattr_destroy(&attr);
    }
    return pid;
}

int 
main (int argc, char const **argv)
{
    printf ("pid %i: Pointer size is %zu.\n", getpid(), sizeof(void *));
    int err = 0;    // Set breakpoint 1 here
#if defined (__x86_64__)
    if (sizeof(void *) == 8)
    {
        spawn_process (argv, NULL, CPU_TYPE_I386, err);
        if (err)
            exit_with_errno (err, "posix_spawn i386 error");
    }
    else
    {
        spawn_process (argv, NULL, CPU_TYPE_X86_64, err);
        if (err)
            exit_with_errno (err, "posix_spawn x86_64 error");
    }
#else
    spawn_process (argv, NULL, 0, err);
    if (err)
        exit_with_errno (err, "posix_spawn x86_64 error");
#endif
    return 0;
}
