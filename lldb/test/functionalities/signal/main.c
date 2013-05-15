#include <sys/signal.h>
#include <stdio.h>
#include <unistd.h>

void handler_usr1 (int i)
{
  puts ("got signal usr1");
}

void handler_alrm (int i)
{
  puts ("got signal ALRM");
}

int main ()
{
  int i = 0;

  signal (SIGUSR1, handler_usr1);
  signal (SIGALRM, handler_alrm);

  puts ("Put breakpoint here");

  while (i++ < 20)
     sleep (1);
}

