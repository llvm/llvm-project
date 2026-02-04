#include <dispatch/dispatch.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

void do_work_level_5(void) {
  // Frame 0 will have these variables.
  int frame0_var = 100;
  const char *frame0_string = "frame_zero";
  float frame0_float = 1.5f;

  // This is where we'll set the breakpoint.
  printf("Level 5 work executing\n"); // Break here.
  while (1)
    sleep(1);
}

void do_work_level_4(void) {
  // Frame 1 will have these variables.
  int frame1_var = 200;
  const char *frame1_string = "frame_one";
  long frame1_long = 9876543210L;

  do_work_level_5();
}

void do_work_level_3(void) {
  // Frame 2 will have these variables.
  int test_variable = 42;
  const char *test_string = "test_value";
  double test_double = 3.14159;

  do_work_level_4();
}

void do_work_level_2(void) { do_work_level_3(); }

void do_work_level_1(void *context) { do_work_level_2(); }

int main(int argc, const char *argv[]) {
  // Create a serial dispatch queue.
  dispatch_queue_t worker_queue =
      dispatch_queue_create("com.test.worker_queue", DISPATCH_QUEUE_SERIAL);
  dispatch_queue_t submitter_queue =
      dispatch_queue_create("com.test.submitter_queue", DISPATCH_QUEUE_SERIAL);

  // Submit work from one queue to another to create extended backtrace.
  dispatch_async_f(submitter_queue, &worker_queue, do_work_level_1);

  // Keep main thread alive.
  dispatch_main();
  return 0;
}
