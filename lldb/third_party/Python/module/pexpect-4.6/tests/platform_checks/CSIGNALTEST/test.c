/* I built this with "gcc -lutil test.c -otest" */
#include <sys/types.h>  /* include this before any other sys headers */
#include <sys/wait.h>   /* header for waitpid() and various macros */
#include <signal.h>     /* header for signal functions */
#include <stdio.h>      /* header for fprintf() */
#include <unistd.h>     /* header for fork() */
#ifdef LINUX
#include <pty.h>
#else
#include <util.h>	/* header for forkpty, compile with -lutil */
#endif

void sig_chld(int);  /* prototype for our SIGCHLD handler */

int main() 
{
    struct sigaction act;
    int pid;
    int fdm;
    char slave_name [20];


    /* Assign sig_chld as our SIGCHLD handler.
       We don't want to block any other signals in this example 
       We're only interested in children that have terminated, not ones
       which have been stopped (eg user pressing control-Z at terminal).
       Finally, make these values effective. If we were writing a real 
       application, we would save the old value instead of passing NULL.
     */
    act.sa_handler = sig_chld;
    sigemptyset(&act.sa_mask);
    act.sa_flags = SA_NOCLDSTOP;
    if (sigaction(SIGCHLD, &act, NULL) < 0) 
    {
        fprintf(stderr, "sigaction failed\n");
        return 1;
    }

    /* Do the Fork thing. 
    */
    pid = forkpty (&fdm, slave_name, NULL, NULL);
    /* pid = fork(); */

    switch (pid)
    {
	    case -1:
		fprintf(stderr, "fork failed\n");
		return 1;
	    break;

	    case 0: /* Child process. */     
		printf ("This child output will cause trouble.\n");
		_exit(7);
	    break;

	    default: /* Parent process. */
		sleep(1);
		printf ("Child pid: %d\n", pid); 
		sleep(10);  /* let child finish -- crappy way to avoid race. */
	    break;
    }

    return 0;
}
 
void sig_chld(int signo) 
{
    int status, wpid, child_val;

    printf ("In sig_chld signal handler.\n");

    /* Wait for any child without blocking */
    wpid = waitpid (-1, & status, WNOHANG);
    printf ("\tWaitpid found status for pid: %d\n", wpid);
    if (wpid < 0)
    {
        fprintf(stderr, "\twaitpid failed\n");
        return;
    }
    printf("\tWaitpid status: %d\n", status);

    if (WIFEXITED(status)) /* did child exit normally? */
    {
        child_val = WEXITSTATUS(status); 
        printf("\tchild exited normally with status %d\n", child_val);
    }
    printf ("End of sig_chld.\n");
}


