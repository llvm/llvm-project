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

#define BUFFER_SIZE 1024

char buf[BUFFER_SIZE];
#pragma omp threadprivate(buf)

char* get_string(size_t check_needed) {
  size_t needed = omp_capture_affinity(buf, BUFFER_SIZE, NULL);
  //printf("buf = %s\n", buf);
  check(needed < BUFFER_SIZE);
  if (check_needed != 0) {
    check(needed == check_needed);
  }
  return buf;
}

void check_thread_num_padded_rjustified() {
  int i;
  const char* formats[2] = {"%0.8{thread_num}", "%0.8n"};
  for (i = 0; i < sizeof(formats)/sizeof(formats[0]); ++i) {
    omp_set_affinity_format(formats[i]);
    #pragma omp parallel num_threads(8)
    {
      int j;
      int tid = omp_get_thread_num();
      char ctid = '0' + (char)tid;
      char* s = get_string(8);
      for (j = 0; j < 7; ++j) {
        check(s[j] == '0');
      }
      check(s[j] == ctid);
    }
  }
}

void check_thread_num_rjustified() {
  int i;
  const char* formats[2] = {"%.12{thread_num}", "%.12n"};
  for (i = 0; i < sizeof(formats)/sizeof(formats[0]); ++i) {
    omp_set_affinity_format(formats[i]);
    #pragma omp parallel num_threads(8)
    {
      int j;
      int tid = omp_get_thread_num();
      char ctid = '0' + (char)tid;
      char* s = get_string(12);
      for (j = 0; j < 11; ++j) {
        check(s[j] == ' ');
      }
      check(s[j] == ctid);
    }
  }
}

void check_thread_num_ljustified() {
  int i;
  const char* formats[2] = {"%5{thread_num}", "%5n"};
  for (i = 0; i < sizeof(formats)/sizeof(formats[0]); ++i) {
    omp_set_affinity_format(formats[i]);
    #pragma omp parallel num_threads(8)
    {
      int j;
      int tid = omp_get_thread_num();
      char ctid = '0' + (char)tid;
      char* s = get_string(5);
      check(s[0] == ctid);
      for (j = 1; j < 5; ++j) {
        check(s[j] == ' ');
      }
    }
  }
}

void check_thread_num_padded_ljustified() {
  int i;
  const char* formats[2] = {"%018{thread_num}", "%018n"};
  for (i = 0; i < sizeof(formats)/sizeof(formats[0]); ++i) {
    omp_set_affinity_format(formats[i]);
    #pragma omp parallel num_threads(8)
    {
      int j;
      int tid = omp_get_thread_num();
      char ctid = '0' + (char)tid;
      char* s = get_string(18);
      check(s[0] == ctid);
      for (j = 1; j < 18; ++j) {
        check(s[j] == ' ');
      }
    }
  }
}

int main(int argc, char** argv) {
  check_thread_num_ljustified();
  check_thread_num_rjustified();
  check_thread_num_padded_ljustified();
  check_thread_num_padded_rjustified();
  return 0;
}
