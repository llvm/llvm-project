#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdint.h>

int 
wait_a_while (useconds_t interval)
{
  int num_times = 0;
  int return_value = 1;

  struct timeval start_time;
  gettimeofday(&start_time, NULL);
  uint64_t target = start_time.tv_sec * 1000000 + start_time.tv_usec + interval;

  while (1)
    {
      num_times++;
      return_value = usleep (interval);
      if (return_value != 0)
        {
          struct timeval now;
          gettimeofday(&now, NULL);
          interval = target - now.tv_sec * 1000000 + now.tv_usec;
        }
      else
        break;
    }
  return num_times;
}

int
main (int argc, char **argv)
{
  printf ("stop here in main.\n");
  int num_times = wait_a_while (argc * 1000);
  printf ("Done, took %d times.\n", num_times);

  return 0;

}
