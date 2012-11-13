#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>


#define polybench_start_instruments
#define polybench_stop_instruments
#define polybench_print_instruments

#ifdef POLYBENCH_TIME
# undef polybench_start_instruments
# undef polybench_stop_instruments
# undef polybench_print_instruments
# define polybench_start_instruments polybench_timer_start();
# define polybench_stop_instruments polybench_timer_stop();
# define polybench_print_instruments polybench_timer_print();
#endif


extern void polybench_timer_start();
extern void polybench_timer_stop();
extern void polybench_timer_print();
