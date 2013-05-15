#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sched.h>
#include <math.h>

#ifndef POLYBENCH_CACHE_SIZE_KB
# define POLYBENCH_CACHE_SIZE_KB 8192
#endif

/* Timer code (gettimeofday). */
double polybench_t_start, polybench_t_end;

static
inline
double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0)
      printf("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

inline
void polybench_flush_cache()
{
  int cs = POLYBENCH_CACHE_SIZE_KB * 1024 / sizeof(double);
  double* flush = (double*) calloc(cs, sizeof(double));
  int i;
  double tmp = 0.0;
  for (i = 0; i < cs; i++)
    tmp += flush[i];
  /* This prevents DCE on the cache flush code. */
  assert (tmp <= 10.0);
}

#ifdef POLYBENCH_LINUX_FIFO_SCHEDULER
inline
void polybench_linux_fifo_scheduler()
{
  /* Use FIFO scheduler to limit OS interference. Program must be run
     as root, and this works only for Linux kernels. */
  struct sched_param schedParam;
  schedParam.sched_priority = sched_get_priority_max(SCHED_FIFO);
  sched_setscheduler(0, SCHED_FIFO, &schedParam);
}

inline
void polybench_linux_standard_scheduler()
{
  /* Restore to standard scheduler policy. */
  struct sched_param schedParam;
  schedParam.sched_priority = sched_get_priority_max(SCHED_OTHER);
  sched_setscheduler(0, SCHED_OTHER, &schedParam);
}
#endif

void polybench_timer_start()
{
#ifndef POLYBENCH_NO_FLUSH_CACHE
  polybench_flush_cache();
#endif
#ifdef POLYBENCH_LINUX_FIFO_SCHEDULER
  polybench_linux_fifo_scheduler();
#endif
  polybench_t_start = rtclock();
}

void polybench_timer_stop()
{
#ifdef POLYBENCH_LINUX_FIFO_SCHEDULER
  polybench_linux_standard_scheduler();
#endif
  polybench_t_end = rtclock();
}

void polybench_timer_print()
{
    printf("%0.6lfs\n", polybench_t_end - polybench_t_start);
}
