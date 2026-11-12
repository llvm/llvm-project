#include <unistd.h>
#include <stdio.h>
#include <signal.h>

static int sigchld_no;
static int nosig_no;
static int weird_value;

void
sigchld_handler (int signo)
{
  sigchld_no++;
  printf ("Got sigchld %d.\n", sigchld_no);
}

int
call_me (int some_value)
{
  int ret_val = 0;
  int i;
  for (i = 0; i < some_value; i++)
    {
      int result = 0;
      if (i%2 == 0)
          result = kill (getpid(), SIGCHLD);
      else
        sigchld_no++;

      usleep(1000);
      if (result == 0)
        ret_val++;
    }
  usleep (10000);
  return ret_val;
}

int
call_me_nosig (int some_value)
{
  int ret_val = 0;
  int i;
  for (i = 0; i < some_value; i++)
    weird_value += i % 4;

  nosig_no += some_value;
  return some_value;
}

int 
main ()
{
  int ret_val;
  signal (SIGCHLD, sigchld_handler);
  
  ret_val = call_me (2);  // Stop here in main.

  ret_val = call_me_nosig (10);

  return 0;

}
