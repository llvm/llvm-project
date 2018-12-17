// RUN: %libomp-compile-and-run

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define XSTR(x) #x
#define STR(x) XSTR(x)

#define streqls(s1, s2) (!strcmp(s1, s2))

#define check(condition)                                                       \
  if (!(condition)) {                                                          \
    fprintf(stderr, "error: %s: %d: " STR(condition) "\n", __FILE__,           \
            __LINE__);                                                         \
    exit(1);                                                                   \
  }

#if defined(_WIN32)
#include <windows.h>
#define getpid _getpid
typedef int pid_t;
#define gettid GetCurrentThreadId
#define my_gethostname(buf, sz) GetComputerNameA(buf, &(sz))
#else
#include <unistd.h>
#include <sys/types.h>
#define my_gethostname(buf, sz) gethostname(buf, sz)
#endif

#define BUFFER_SIZE 256

int get_integer() {
  int n, retval;
  char buf[BUFFER_SIZE];
  size_t needed = omp_capture_affinity(buf, BUFFER_SIZE, NULL);
  check(needed < BUFFER_SIZE);
  n = sscanf(buf, "%d", &retval);
  check(n == 1);
  return retval;
}

char* get_string() {
  int n, retval;
  char buf[BUFFER_SIZE];
  size_t needed = omp_capture_affinity(buf, BUFFER_SIZE, NULL);
  check(needed < BUFFER_SIZE);
  return strdup(buf);
}

void check_integer(const char* formats[2], int(*func)()) {
  int i;
  for (i = 0; i < 2; ++i) {
    omp_set_affinity_format(formats[i]);
    #pragma omp parallel num_threads(8)
    {
      check(get_integer() == func());
      #pragma omp parallel num_threads(3)
      {
        check(get_integer() == func());
      }
      check(get_integer() == func());
    }
  }
}

void check_nesting_level() {
  // Check %{nesting_level} and %L
  const char* formats[2] = {"%{nesting_level}", "%L"};
  check_integer(formats, omp_get_level);
}

void check_thread_num() {
  // Check %{thread_num} and %n
  const char* formats[2] = {"%{thread_num}", "%n"};
  check_integer(formats, omp_get_thread_num);
}

void check_num_threads() {
  // Check %{num_threads} and %N
  const char* formats[2] = {"%{num_threads}", "%N"};
  check_integer(formats, omp_get_num_threads);
}

int ancestor_helper() {
  return omp_get_ancestor_thread_num(omp_get_level() - 1);
}
void check_ancestor_tnum() {
  // Check %{ancestor_tnum} and %a
  const char* formats[2] = {"%{ancestor_tnum}", "%a"};
  check_integer(formats, ancestor_helper);
}

int my_get_pid() { return (int)getpid(); }
void check_process_id() {
  // Check %{process_id} and %P
  const char* formats[2] = {"%{process_id}", "%P"};
  check_integer(formats, my_get_pid);
}

/*
int my_get_tid() { return (int)gettid(); }
void check_native_thread_id() {
  // Check %{native_thread_id} and %i
  const char* formats[2] = {"%{native_thread_id}", "%i"};
  check_integer(formats, my_get_tid);
}
*/

void check_host() {
  int i;
  int buffer_size = 256;
  const char* formats[2] = {"%{host}", "%H"};
  char hostname[256];
  my_gethostname(hostname, buffer_size);
  for (i = 0; i < 2; ++i) {
    omp_set_affinity_format(formats[i]);
    #pragma omp parallel num_threads(8)
    {
      char* host = get_string();
      check(streqls(host, hostname));
      free(host);
    }
  }
}

void check_undefined() {
  int i;
  const char* formats[2] = {"%{foobar}", "%X"};
  for (i = 0; i < 2; ++i) {
    omp_set_affinity_format(formats[i]);
    #pragma omp parallel num_threads(8)
    {
      char* undef = get_string();
      check(streqls(undef, "undefined"));
      free(undef);
    }
  }
}

int main(int argc, char** argv) {
  omp_set_nested(1);
  check_nesting_level();
  check_num_threads();
  check_ancestor_tnum();
  check_process_id();
  //check_native_thread_id();
  check_host();
  check_undefined();
  return 0;
}
