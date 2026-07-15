#ifndef MY_SLEEP_H
#define MY_SLEEP_H

/*! Utility function to have a sleep function with better resolution and
 *  which only stops one thread. */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>

#if defined(_WIN32)
# include <windows.h>
// Windows version of my_sleep() function
static void my_sleep(double sleeptime) {
  DWORD ms = (DWORD) (sleeptime * 1000.0);
  Sleep(ms);
}


#else // _WIN32

// Unices version of my_sleep() function
static void my_sleep(double sleeptime) {
  struct timespec ts;
  ts.tv_sec = (time_t)sleeptime;
  ts.tv_nsec = (long)((sleeptime - (double)ts.tv_sec) * 1E9);
  nanosleep(&ts, NULL);
}

#endif // _WIN32

#endif // MY_SLEEP_H
