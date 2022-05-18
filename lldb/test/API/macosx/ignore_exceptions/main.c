#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <stdio.h>

int g_ints[] = {10, 20, 30, 40, 50, 60};

void
saction_handler(int signo, siginfo_t info, void *baton) {
  printf("Got into handler.\n");
  mprotect(g_ints, sizeof(g_ints), PROT_READ|PROT_WRITE); // stop here in the signal handler
  g_ints[0] = 20;
}
int
main()
{
  mprotect(g_ints, 10*sizeof(int) , PROT_NONE);
  struct sigaction my_action;
  sigemptyset(&my_action.sa_mask);
  my_action.sa_handler = (void (*)(int)) saction_handler;
  my_action.sa_flags = SA_SIGINFO;

  sigaction(SIGBUS, &my_action, NULL); // Stop here to get things going.
  int local_value = g_ints[1];
  return local_value; // Break here to make sure we got past the signal handler
}
