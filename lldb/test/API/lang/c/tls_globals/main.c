#include <stdio.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

#ifdef _WIN32
typedef DWORD ThreadReturn;

static void do_sleep(int msec) { Sleep(msec); }
static void run_thread(ThreadReturn (*fn)(void *)) {
  CreateThread(NULL, 0, fn, NULL, 0, NULL);
}
#else
typedef void *ThreadReturn;

static void do_sleep(int msec) { usleep(msec * 1000); }
static void run_thread(ThreadReturn (*fn)(void *)) {
  pthread_t handle;
  pthread_create(&handle, NULL, fn, NULL);
}
#endif

void LLDB_DYLIB_IMPORT shared_check();
// On some OS's (darwin) you must actually access a thread local variable
// before you can read it
int LLDB_DYLIB_IMPORT touch_shared();

// Create some TLS storage within the static executable.
__thread int var_static = 44;
__thread int var_static2 = 22;

static ThreadReturn fn_static(void *param) {
  var_static *= 2;
  var_static2 *= 3;
  shared_check();
  do_sleep(1); // thread breakpoint
  for (;;)
    do_sleep(1);

  return 0;
}

int main(int argc, char const *argv[]) {
  run_thread(&fn_static);
  touch_shared();
  for (; var_static;) {
    printf(""); // main breakpoint
  }

  return 0;
}
