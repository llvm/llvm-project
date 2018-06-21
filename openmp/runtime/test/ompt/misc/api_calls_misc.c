// RUN: %libomp-compile && %libomp-run | FileCheck %s
// REQUIRES: ompt
#include "callback.h"
#include <omp.h>

int main() {
#pragma omp parallel num_threads(1)
  {
    // ompt_get_callback()
    ompt_callback_t callback;
    ompt_get_callback(ompt_callback_thread_begin, &callback);
    printf("%" PRIu64 ": &on_ompt_callback_thread_begin=%p\n",
           ompt_get_thread_data()->value, &on_ompt_callback_thread_begin);
    printf("%" PRIu64 ": ompt_get_callback() result=%p\n",
           ompt_get_thread_data()->value, callback);

    // ompt_get_state()
    printf("%" PRIu64 ": ompt_get_state()=%d\n", ompt_get_thread_data()->value,
           ompt_get_state(NULL));

    // ompt_enumerate_states()
    int state = omp_state_undefined;
    const char *state_name;
    int steps = 0;
    while (ompt_enumerate_states(state, &state, &state_name) && steps < 1000) {
      steps++;
      if (!state_name)
        printf("%" PRIu64 ": state_name is NULL\n",
               ompt_get_thread_data()->value);
    }
    if (steps >= 1000) {
      // enumeration did not end after 1000 steps
      printf("%" PRIu64 ": states enumeration did not end\n",
             ompt_get_thread_data()->value);
    }

    // ompt_enumerate_mutex_impls()
    int impl = ompt_mutex_impl_unknown;
    const char *impl_name;
    steps = 0;
    while (ompt_enumerate_mutex_impls(impl, &impl, &impl_name) &&
           steps < 1000) {
      steps++;
      if (!impl_name)
        printf("%" PRIu64 ": impl_name is NULL\n",
               ompt_get_thread_data()->value);
    }
    if (steps >= 1000) {
      // enumeration did not end after 1000 steps
      printf("%" PRIu64 ": mutex_impls enumeration did not end\n",
             ompt_get_thread_data()->value);
    }
  }

  // Check if libomp supports the callbacks for this test.

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: &on_ompt_callback_thread_begin
  // CHECK-SAME: =[[FUNCTION_POINTER:0x[0-f]+]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_get_callback() result=[[FUNCTION_POINTER]]

  // CHECK: {{^}}[[THREAD_ID]]: ompt_get_state()=1

  // CHECK-NOT: {{^}}[[THREAD_ID]]: state_name is NULL
  // CHECK-NOT: {{^}}[[THREAD_ID]]: states enumeration did not end

  // CHECK-NOT: {{^}}[[THREAD_ID]]: impl_name is NULL
  // CHECK-NOT: {{^}}[[THREAD_ID]]: mutex_impls enumeration did not end

  return 0;
}
