#include <Python.h>
#include <dlfcn.h>
#include <errno.h>
#include <omp-tools.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void *ompd_library;

struct _ompd_aspace_cont {
  int id;
};
struct _ompd_thread_cont {
  int id;
};
ompd_address_space_context_t context = {42};
ompd_address_space_context_t invalidcontext = {99};

// call back functions for ompd_initialize
ompd_rc_t _alloc(ompd_size_t bytes, void **ptr);
ompd_rc_t _free(void *ptr);
ompd_rc_t _sizes(ompd_address_space_context_t *_acontext,
                 ompd_device_type_sizes_t *sizes);
ompd_rc_t _sym_addr(ompd_address_space_context_t *context,
                    ompd_thread_context_t *tcontext, const char *symbol_name,
                    ompd_address_t *symbol_addr, const char *file_name);
ompd_rc_t _read(ompd_address_space_context_t *context,
                ompd_thread_context_t *tcontext, const ompd_address_t *addr,
                ompd_size_t nbytes, void *buffer);
ompd_rc_t _read_string(ompd_address_space_context_t *context,
                       ompd_thread_context_t *tcontext,
                       const ompd_address_t *addr, ompd_size_t nbytes,
                       void *buffer);
ompd_rc_t _endianess(ompd_address_space_context_t *address_space_context,
                     const void *input, ompd_size_t unit_size,
                     ompd_size_t count, void *output);
ompd_rc_t _thread_context(ompd_address_space_context_t *context,
                          ompd_thread_id_t kind, ompd_size_t sizeof_thread_id,
                          const void *thread_id,
                          ompd_thread_context_t **thread_context);
ompd_rc_t _print(const char *str, int category);

/*
  Test API: ompd_get_thread_handle

  ompdtestapi threadandparallel

  Program:
    1.    #include <stdio.h>
    2.    #include <omp.h>
    3.    int main () {
    4.        omp_set_num_threads(2);
    5.        #pragma omp parallel
    6.        {
    7.            printf("Parallel level 1, thread num = %d",
                     omp_get_thread_num());
    8.        }
    9.        return 0;
    10.   }

  GDB Commands:
    ompd init
    b 7
    c
    ompdtestapi ompd_get_thread_handle

  for ompd_rc_unavailable:
    ompd init
    ompdtestapi ompd_get_thread_handle
*/

PyObject *test_ompd_get_thread_handle(PyObject *self, PyObject *args) {
  printf("Testing \"ompd_get_thread_handle\"...\n");

  PyObject *addrSpaceTup = PyTuple_GetItem(args, 0);
  ompd_address_space_handle_t *addr_handle =
      (ompd_address_space_handle_t *)PyCapsule_GetPointer(addrSpaceTup,
                                                          "AddressSpace");

  PyObject *threadIdTup = PyTuple_GetItem(args, 1);
  uint64_t threadID = (uint64_t)PyLong_AsLong(threadIdTup);

  ompd_size_t sizeof_thread_id = sizeof(threadID);
  ompd_thread_handle_t *thread_handle;

  // should be successful
  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc = ompd_get_thread_handle(
      addr_handle, 1 /*lwp*/, sizeof_thread_id, &threadID, &thread_handle);

  if (rc == ompd_rc_unavailable) {
    // ompd_rc_unavailable if the thread is not an OpenMP thread.
    printf("Success. ompd_rc_unavailable, OpenMP is disabled.\n");
    printf("This is not a Parallel Region, No more testing is possible.\n");
    return Py_None;
  } else if (rc != ompd_rc_ok)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  // as in ompd-types.h, only 0-3 are valid for thread kind
  // ompd_rc_unsupported if thread kind is not supported.
  printf("Test: Unsupported thread kind.\n");
  rc = ompd_get_thread_handle(addr_handle, 4, sizeof_thread_id, &threadID,
                              &thread_handle);
  if (rc != ompd_rc_unsupported)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  // ompd_rc_bad_input: if a different value in sizeof_thread_id is expected for
  // a thread kind.
  // sizeof_thread_id is validated at thread_context which is call back function
  // "_thread_context" where we expect size to be sizeof(long int)
  printf("Test: Wrong value for sizeof threadID.\n");
  rc = ompd_get_thread_handle(addr_handle, 1 /*lwp*/, sizeof_thread_id - 1,
                              &threadID, &thread_handle);
  if (rc != ompd_rc_bad_input)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  // Random checks with null and invalid args.
  /*
     ompd_rc_stale_handle: is returned when the specified handle is no
     longer valid;
     ompd_rc_bad_input: is returned when the input parameters
     (other than handle) are invalid;
     ompd_rc_error:    is returned when a fatal error occurred;
  */

  printf("Test: Expecting ompd_rc_bad_input for NULL thread_handle.\n");
  rc = ompd_get_thread_handle(addr_handle, 1 /*lwp*/, sizeof_thread_id,
                              &threadID, NULL);
  if (rc != ompd_rc_bad_input)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  printf(
      "Test: Expecting ompd_rc_error or stale_handle for NULL addr_handle.\n");
  rc = ompd_get_thread_handle(NULL, 1 /*lwp*/, sizeof_thread_id, &threadID,
                              &thread_handle);
  if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  return Py_None;
}

/*
  Test API: ompd_get_curr_parallel_handle.

  Program:
    1.    #include <stdio.h>
    2.    #include <omp.h>
    3.    int main () {
    4.        omp_set_num_threads(2);
    5.        #pragma omp parallel
    6.        {
    7.            printf("Parallel level 1, thread num = %d",
                     omp_get_thread_num());
    8.        }
    9.        return 0;
    10.   }

  GDB Commands:
    ompd init
    b 7
    omptestapi ompd_get_curr_parallel_handle

    for ompd_rc_unavailable
    ompd init
    omptestapi ompd_get_curr_parallel_handle (or break at line 4
    before this)
*/

PyObject *test_ompd_get_curr_parallel_handle(PyObject *self, PyObject *args) {
  printf("Testing \"ompd_get_curr_parallel_handle\"...\n");

  PyObject *threadHandlePy = PyTuple_GetItem(args, 0);
  ompd_thread_handle_t *thread_handle =
      (ompd_thread_handle_t *)(PyCapsule_GetPointer(threadHandlePy,
                                                    "ThreadHandle"));

  ompd_parallel_handle_t *parallel_handle;

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc = ompd_get_curr_parallel_handle(thread_handle, &parallel_handle);
  if (rc == ompd_rc_unavailable) {
    // ompd_rc_unavailable if the thread is not currently part of a team

    // ToCheck: Even in non parallel region, error code is stale_handle
    // Need to find a test case for ompd_rc_unavailable ?????
    printf("Success. ompd_rc_unavailable, Not in parallel region\n");
    printf("No more testing is possible.\n");
    return Py_None;
  } else if (rc == ompd_rc_stale_handle) {
    printf("Return code is stale_handle, may be in non-parallel region.\n");
    printf("No more testing is possible.\n");
    return Py_None;
  } else if (rc != ompd_rc_ok)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  // Random checks with  null and invalid args.
  /*
     ompd_rc_stale_handle: is returned when the specified handle is no
     longer valid;
     ompd_rc_bad_input: is returned when the input parameters
     (other than handle) are invalid;
     ompd_rc_error:    is returned when a fatal error occurred;
  */

  printf("Test: Expecting ompd_rc_bad_input for NULL parallel_handle.\n");
  rc = ompd_get_curr_parallel_handle(thread_handle, NULL);
  if (rc != ompd_rc_bad_input)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  printf("Test: Expecting ompd_rc_error or stale_handle for NULL "
         "thread_handle.\n");
  rc = ompd_get_curr_parallel_handle(NULL, &parallel_handle);
  if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  return Py_None;
}

/*
  Test API: ompd_get_thread_in_parallel.

  Program:
    1.    #include <stdio.h>
    2.    #include <omp.h>
    3.    int main () {
    4.        omp_set_num_threads(3);
    5.        #pragma omp parallel
    6.        {
    7.            printf("Parallel level 1, thread num = %d",
                     omp_get_thread_num());
    8.        }
    9.        return 0;
    10.   }

  GDB Commands:
    ompd init
    b 7
    omptestapi ompd_get_thread_in_parallel
*/
PyObject *test_ompd_get_thread_in_parallel(PyObject *self, PyObject *args) {
  printf("Testing \"ompd_get_thread_in_parallel\"...\n");

  PyObject *parallelHandlePy = PyTuple_GetItem(args, 0);
  ompd_parallel_handle_t *parallel_handle =
      (ompd_parallel_handle_t *)(PyCapsule_GetPointer(parallelHandlePy,
                                                      "ParallelHandle"));
  ompd_thread_handle_t *thread_handle;

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc = ompd_get_thread_in_parallel(
      parallel_handle, 1 /* lesser than team-size-var*/, &thread_handle);
  if (rc != ompd_rc_ok) {
    printf("Failed, with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success.\n");

  // ompd_rc_bad_input: if the thread_num argument is greater than or equal to
  // the team-size-var ICV or negative
  printf("Test: Invalid thread num (199).\n");
  rc = ompd_get_thread_in_parallel(parallel_handle, 199, &thread_handle);
  if (rc != ompd_rc_bad_input)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  printf("Test: Invalid thread num (-5).\n");
  rc = ompd_get_thread_in_parallel(parallel_handle, -5, &thread_handle);
  if (rc != ompd_rc_bad_input)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  // Random checks with  null and invalid args.
  /*
     ompd_rc_stale_handle: is returned when the specified handle is no
     longer valid;
     ompd_rc_bad_input: is returned when the input parameters
     (other than handle) are invalid;
     ompd_rc_error:    is returned when a fatal error occurred;
  */

  printf("Test: Expecting ompd_rc_bad_input for NULL thread_handle.\n");
  rc = ompd_get_thread_in_parallel(parallel_handle, 1, NULL);
  if (rc != ompd_rc_bad_input)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  printf("Test: Expecting ompd_rc_error or stale_handle for NULL "
         "parallel_handle.\n");
  rc = ompd_get_thread_in_parallel(NULL, 1, &thread_handle);
  if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  return Py_None;
}

/*
  Test API: ompd_thread_handle_compare.

  Program:
    1.    #include <stdio.h>
    2.    #include <omp.h>
    3.    int main () {
    4.        omp_set_num_threads(4);
    5.        #pragma omp parallel
    6.        {
    7.            printf("Parallel level 1, thread num = %d",
                     omp_get_thread_num());
    8.        }
    9.        return 0;
    10.   }

  GDB Commands:
    ompd init
    b 7
    omptestapi ompd_thread_handle_compare
*/

PyObject *test_ompd_thread_handle_compare(PyObject *self, PyObject *args) {
  printf("Testing \"ompd_thread_handle_compare\"...\n");

  PyObject *threadHandlePy1 = PyTuple_GetItem(args, 0);
  ompd_thread_handle_t *thread_handle1 =
      (ompd_thread_handle_t *)(PyCapsule_GetPointer(threadHandlePy1,
                                                    "ThreadHandle"));
  PyObject *threadHandlePy2 = PyTuple_GetItem(args, 1);
  ompd_thread_handle_t *thread_handle2 =
      (ompd_thread_handle_t *)(PyCapsule_GetPointer(threadHandlePy2,
                                                    "ThreadHandle"));

  int cmp_value;

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc =
      ompd_thread_handle_compare(thread_handle1, thread_handle2, &cmp_value);
  if (rc != ompd_rc_ok) {
    printf("Failed, with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success.\n");

  if (cmp_value == 0) {
    printf("Threads are Equal.\n");
  } else {
    // a value less than, equal to, or greater than 0 indicates that the thread
    // corresponding to thread_handle_1 is, respectively, less than, equal to,
    // or greater than that corresponding to thread_handle_2.
    if (cmp_value <= 0) {
      printf("Thread 1 is lesser than thread 2, cmp_val = %d\n", cmp_value);
      printf("Test: Changing the order.\n");
      rc = ompd_thread_handle_compare(thread_handle2, thread_handle1,
                                      &cmp_value);
      if (rc != ompd_rc_ok) {
        printf("Failed, with return code = %d\n", rc);
        return Py_None;
      }
      if (cmp_value >= 0)
        printf("Success now cmp_value is greater, %d.\n", cmp_value);
      else
        printf("Failed.\n");
    } else {
      printf("Thread 1 is greater than thread 2.\n");
      printf("Test: Changing the order.\n");
      rc = ompd_thread_handle_compare(thread_handle2, thread_handle1,
                                      &cmp_value);
      if (rc != ompd_rc_ok) {
        printf("Failed, with return code = %d\n", rc);
        return Py_None;
      }
      if (cmp_value <= 0)
        printf("Success now cmp_value is lesser, %d.\n", cmp_value);
      else
        printf("Failed.\n");
    }

    // Random checks with  null and invalid args.
    /*
     ompd_rc_stale_handle: is returned when the specified handle is no
     longer valid;
     ompd_rc_bad_input: is returned when the input parameters
     (other than handle) are invalid;
     ompd_rc_error:    is returned when a fatal error occurred;
    */

    printf("Test: Expecting ompd_rc_bad_input for NULL cmp_value.\n");
    rc = ompd_thread_handle_compare(thread_handle2, thread_handle1, NULL);
    if (rc != ompd_rc_bad_input)
      printf("Failed, with return code = %d\n", rc);
    else
      printf("Success.\n");

    printf("Test: Expecting ompd_rc_error or stale_handle for NULL "
           "thread_handle.\n");
    rc = ompd_thread_handle_compare(NULL, thread_handle1, &cmp_value);
    if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
      printf("Failed, with return code = %d\n", rc);
    else
      printf("Success.\n");
  }

  return Py_None;
}

/*
  Test API: ompd_get_thread_id.

  Program:
    1.    #include <stdio.h>
    2.    #include <omp.h>
    3.    int main () {
    4.        omp_set_num_threads(2);
    5.        #pragma omp parallel
    6.        {
    7.            printf("Parallel level 1, thread num = %d",
                     omp_get_thread_num());
    8.        }
    9.        return 0;
    10.   }

  GDB Commands:
    ompd init
    b 7
    omptestapi ompd_get_thread_id
*/

PyObject *test_ompd_get_thread_id(PyObject *self, PyObject *args) {
  printf("Testing \"ompd_get_thread_id\"...\n");

  PyObject *threadHandlePy = PyTuple_GetItem(args, 0);
  ompd_thread_handle_t *thread_handle =
      (ompd_thread_handle_t *)(PyCapsule_GetPointer(threadHandlePy,
                                                    "ThreadHandle"));

  uint64_t threadID;
  ompd_size_t sizeof_thread_id = sizeof(threadID);

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc = ompd_get_thread_id(thread_handle, 0 /*OMPD_THREAD_ID_PTHREAD*/,
                                    sizeof_thread_id, &threadID);
  if (rc != ompd_rc_ok) {
    printf("Failed, with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success. Thread id = %ld\n", threadID);

  // ompd_rc_bad_input: if a different value in sizeof_thread_id is expected for
  // a thread kind of kind
  printf("Test: Wrong sizeof_thread_id.\n");
  rc = ompd_get_thread_id(thread_handle, 0 /*OMPD_THREAD_ID_PTHREAD*/,
                          sizeof_thread_id - 1, &threadID);
  if (rc != ompd_rc_bad_input)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  // ompd_rc_unsupported: if the kind of thread is not supported
  printf("Test: Unsupported thread kind.\n");
  // thread kind currently support from 0-3, refer in ompd-types.h
  rc = ompd_get_thread_id(thread_handle, 4, sizeof_thread_id - 1, &threadID);
  if (rc != ompd_rc_unsupported)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  // Random checks with  null and invalid args.
  /*
     ompd_rc_stale_handle: is returned when the specified handle is no
     longer valid;
     ompd_rc_bad_input: is returned when the input parameters
     (other than handle) are invalid;
     ompd_rc_error:    is returned when a fatal error occurred;
  */

  printf("Test: Expecting ompd_rc_bad_input for NULL threadID.\n");
  rc = ompd_get_thread_id(thread_handle, 0 /*OMPD_THREAD_ID_PTHREAD*/,
                          sizeof_thread_id, NULL);
  if (rc != ompd_rc_bad_input)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  printf("Test: Expecting ompd_rc_error for NULL thread_handle.\n");
  rc = ompd_get_thread_id(NULL, 0 /*OMPD_THREAD_ID_PTHREAD*/, sizeof_thread_id,
                          &threadID);
  if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  return Py_None;
}

/*
  Test API: ompd_rel_thread_handle

  Program:
    1.    #include <stdio.h>
    2.    #include <omp.h>
    3.    int main () {
    4.        omp_set_num_threads(2);
    5.        #pragma omp parallel
    6.        {
    7.            printf("Parallel level 1, thread num = %d",
                     omp_get_thread_num());
    8.        }
    9.        return 0;
    10.   }

  GDB Commands:
    ompd init
    b 7
    omptestapi ompd_rel_thread_handle
*/

//  TODO: This might not be the right way to do,as this handle comes from
// python not generated by ompd API

PyObject *test_ompd_rel_thread_handle(PyObject *self, PyObject *args) {
  printf("Testing Not enabled for \"ompd_rel_thread_handle\"...\n");
  printf("Disabled.\n");
  return Py_None;
}

/*
  Test API: ompd_get_enclosing_parallel_handle.

  Program:
    1.    #include <stdio.h>
    2.    #include <omp.h>
    3.    int main () {
    4.        omp_set_num_threads(2);
    5.        #pragma omp parallel
    6.        {
    7.            printf("Parallel level 1, thread num = %d",
                          omp_get_thread_num());
    8.            omp_set_num_threads(3);
    9.            #pragma omp parallel
    10.           {
    11.             printf ("Parallel level 2, thread num = %d",
                        omp_get_thread_num());
    12.           }
    13.       }
    14.       return 0;
    15.   }

  GDB Commands:
    ompd init
    b 11
    ompdtestapi ompd_get_enclosing_parallel_handle

    for "ompd_rc_unavailable":
    ompd init
    omptestapi ompd_get_enclosing_parallel_handle
                (or break at line 4 before this)
*/

PyObject *test_ompd_get_enclosing_parallel_handle(PyObject *self,
                                                  PyObject *args) {
  printf("Testing \"ompd_get_enclosing_parallel_handle\"...\n");

  PyObject *parallelHandlePy = PyTuple_GetItem(args, 0);
  ompd_parallel_handle_t *parallel_handle =
      (ompd_parallel_handle_t *)(PyCapsule_GetPointer(parallelHandlePy,
                                                      "ParallelHandle"));
  ompd_parallel_handle_t *enclosing_parallel_handle;

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc = ompd_get_enclosing_parallel_handle(parallel_handle,
                                                    &enclosing_parallel_handle);
  if (rc == ompd_rc_unavailable) {
    // ompd_rc_unavailable: if no enclosing parallel region exists.
    printf("Success. return code is ompd_rc_unavailable, Not in parallel "
           "region\n");
    printf("No more testing is possible.\n");
    return Py_None;
  } else if (rc != ompd_rc_ok) {
    printf("Failed, with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success.\n");

  // Random checks with  null and invalid args.
  /*
     ompd_rc_stale_handle: is returned when the specified handle is no
     longer valid;
     ompd_rc_bad_input: is returned when the input parameters
     (other than handle) are invalid;
     ompd_rc_error:    is returned when a fatal error occurred;
  */

  printf("Test: Expecting ompd_rc_bad_input for NULL "
         "enclosing_parallel_handle.\n");
  rc = ompd_get_enclosing_parallel_handle(parallel_handle, NULL);
  if (rc != ompd_rc_bad_input)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  printf("Test: Expecting ompd_rc_error or stale_handle for NULL "
         "parallel_handle.\n");
  rc = ompd_get_enclosing_parallel_handle(NULL, &enclosing_parallel_handle);
  if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  return Py_None;
}

/*
  Test API: ompd_parallel_handle_compare.

  Program:
    1.    #include <stdio.h>
    2.    #include <omp.h>
    3.    int main () {
    4.        omp_set_num_threads(2);
    5.        #pragma omp parallel
    6.        {
    7.            printf("Parallel level 1, thread num = %d",
                          omp_get_thread_num());
    8.            omp_set_num_threads(3);
    9.            #pragma omp parallel
    10.           {
    11.             printf ("Parallel level 2, thread num = %d",
                        omp_get_thread_num());
    12.           }
    13.       }
    14.       return 0;
    15.   }

  GDB Commands:
    ompd init
    b 11
    ompdtestapi ompd_parallel_handle_compare
*/

PyObject *test_ompd_parallel_handle_compare(PyObject *self, PyObject *args) {
  printf("Testing \"ompd_parallel_handle_compare\"...\n");

  PyObject *parallelHandlePy1 = PyTuple_GetItem(args, 0);
  ompd_parallel_handle_t *parallel_handle1 =
      (ompd_parallel_handle_t *)(PyCapsule_GetPointer(parallelHandlePy1,
                                                      "ParallelHandle"));
  PyObject *parallelHandlePy2 = PyTuple_GetItem(args, 1);
  ompd_parallel_handle_t *parallel_handle2 =
      (ompd_parallel_handle_t *)(PyCapsule_GetPointer(parallelHandlePy2,
                                                      "ParallelHandle"));

  int cmp_value;

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc = ompd_parallel_handle_compare(parallel_handle1,
                                              parallel_handle2, &cmp_value);
  if (rc != ompd_rc_ok) {
    printf("Failed, with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success.\n");

  if (cmp_value == 0) {
    printf("Parallel regions are Same.\n");
  } else {
    // A value less than, equal to, or greater than 0 indicates that the region
    // corresponding to parallel_handle_1 is, respectively, less than, equal to,
    // or greater than that corresponding to parallel_handle_2
    if (cmp_value <= 0) {
      printf("Parallel handle 1 is lesser than handle 2, cmp_val = %d\n",
             cmp_value);
      printf("Test: Changing the order.\n");
      rc = ompd_parallel_handle_compare(parallel_handle2, parallel_handle1,
                                        &cmp_value);
      if (rc != ompd_rc_ok) {
        printf("Failed, with return code = %d\n", rc);
        return Py_None;
      }
      if (cmp_value >= 0)
        printf("Success now cmp_value is greater, %d.\n", cmp_value);
      else
        printf("Failed.\n");
    } else {
      printf("Parallel 1 is greater than handle 2.\n");
      printf("Test: Changing the order.\n");
      rc = ompd_parallel_handle_compare(parallel_handle2, parallel_handle1,
                                        &cmp_value);
      if (rc != ompd_rc_ok) {
        printf("Failed, with return code = %d\n", rc);
        return Py_None;
      }
      if (cmp_value <= 0)
        printf("Success now cmp_value is lesser, %d.\n", cmp_value);
      else
        printf("Failed.\n");
    }

    // Random checks with  null and invalid args.
    /*
     ompd_rc_stale_handle: is returned when the specified handle is no
     longer valid;
     ompd_rc_bad_input: is returned when the input parameters
     (other than handle) are invalid;
     ompd_rc_error:    is returned when a fatal error occurred;
    */

    printf("Test: Expecting ompd_rc_bad_input for NULL cmp_value.\n");
    rc = ompd_parallel_handle_compare(parallel_handle2, parallel_handle1, NULL);
    if (rc != ompd_rc_bad_input)
      printf("Failed, with return code = %d\n", rc);
    else
      printf("Success.\n");

    printf("Test: Expecting ompd_rc_error or stale_handle for NULL "
           "thread_handle.\n");
    rc = ompd_parallel_handle_compare(NULL, parallel_handle1, &cmp_value);
    if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
      printf("Failed, with return code = %d\n", rc);
    else
      printf("Success.\n");
  }

  return Py_None;
}

/*
  Test API: ompd_rel_parallel_handle

  Program:
    1.    #include <stdio.h>
    2.    #include <omp.h>
    3.    int main () {
    4.        omp_set_num_threads(2);
    5.        #pragma omp parallel
    6.        {
    7.            printf("Parallel level 1, thread num = %d",
                     omp_get_thread_num());
    8.        }
    9.        return 0;
    10.   }

  GDB Commands:
    ompd init
    b 7
    omptestapi ompd_rel_parallel_handle
*/

// TODO: Same as thread_rel_handle, might not be a right way to test
// What released should be provided by ompd API, this address is actually from
// python
PyObject *test_ompd_rel_parallel_handle(PyObject *self, PyObject *args) {
  printf("Testing NOT enabled for \"ompd_rel_parallel_handle\"...\n");
  printf("Disabled.\n");
  return Py_None;
}

/*
  Test API: ompd_initialize

  Program:
    1.    #include <stdio.h>
    2.    #include <omp.h>
    3.    int main () {
    4.        omp_set_num_threads(2);
    5.        #pragma omp parallel
    6.        {
    7.            printf("Parallel level 1, thread num = %d",
                     omp_get_thread_num());
    8.        }
    9.        return 0;
    10.   }

  GDB Commands:
    b 4
    ompdtestapi ompd_initialize\
*/
PyObject *test_ompd_initialize(PyObject *self, PyObject *noargs) {
  printf("Testing \"test_ompd_initialize\"...\n");

  ompd_word_t version;
  ompd_rc_t rc = ompd_get_api_version(&version);
  if (rc != ompd_rc_ok) {
    printf("Failed in \"ompd_get_api_version\".\n");
    return Py_None;
  }

  static ompd_callbacks_t table = {
      _alloc, _free,        _print,     _sizes,     _sym_addr,      _read,
      NULL,   _read_string, _endianess, _endianess, _thread_context};

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t (*my_ompd_init)(ompd_word_t version, ompd_callbacks_t *) =
      dlsym(ompd_library, "ompd_initialize");
  rc = my_ompd_init(version, &table);
  if (rc != ompd_rc_ok) {
    printf("Failed, with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success.\n");

  static ompd_callbacks_t invalid_table = {
      NULL,       /*      _alloc, */
      NULL,       /*      _free, */
      NULL,       /*      _print,*/
      NULL,       /*      _sizes, */
      NULL,       /*      _sym_addr, */
      NULL,       /*      _read,*/
      NULL, NULL, /*      _read_string, */
      NULL,       /*      _endianess, */
      NULL,       /*      _endianess, */
      NULL,       /*      _thread_context */
  };

  // ompd_rc_bad_input: if invalid callbacks are provided
  printf("Test: Invalid callbacks.\n");
  rc = my_ompd_init(version, &invalid_table);
  if (rc != ompd_rc_bad_input)
    printf("Warning, with return code = %d\n", rc);
  else
    printf("Success.\n");

  // ompd_rc_unsupported: if the requested API version cannot be provided
  printf("Test: Wrong API version.\n");
  rc = my_ompd_init(150847, &table);
  if (rc != ompd_rc_unsupported)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  // Random checks with  null and invalid args.
  /*
     ompd_rc_stale_handle: is returned when the specified handle is no
     longer valid;
     ompd_rc_bad_input: is returned when the input parameters
     (other than handle) are invalid;
     ompd_rc_error:    is returned when a fatal error occurred;
  */

  printf("Test: Expecting ompd_rc_bad_input for NULL table.\n");
  rc = my_ompd_init(version, NULL);
  if (rc != ompd_rc_bad_input)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  printf("Test: Expecting ompd_rc_error or ompd_rc_bad_input for NULL\n");
  rc = my_ompd_init(0, &table);
  if (rc != ompd_rc_unsupported && rc != ompd_rc_bad_input)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  return Py_None;
}

/*
  Test API: ompd_get_api_version

  Program:
    1.    #include <stdio.h>
    2.    #include <omp.h>
    3.    int main () {
    4.        omp_set_num_threads(2);
    5.        #pragma omp parallel
    6.        {
    7.            printf("Parallel level 1, thread num = %d",
                     omp_get_thread_num());
    8.        }
    9.        return 0;
    10.   }

  GDB Commands:
    ompd init
    b 7
    ompdtestapi ompd_get_version

*/

PyObject *test_ompd_get_api_version(PyObject *self, PyObject *noargs) {
  printf("Testing \"ompd_get_api_version\"...\n");

  ompd_word_t version;

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc = ompd_get_api_version(&version);
  if (rc != ompd_rc_ok) {
    printf("Failed, with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success. API version is %ld\n", version);

  printf(
      "Test: Expecting ompd_rc_error or ompd_rc_bad_input for NULL version\n");
  rc = ompd_get_api_version(NULL);
  if (rc != ompd_rc_error && rc != ompd_rc_bad_input)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  return Py_None;
}

/*
  Test API: ompd_get_version_string

  Program:
    1.    #include <stdio.h>
    2.    #include <omp.h>
    3.    int main () {
    4.        omp_set_num_threads(2);
    5.        #pragma omp parallel
    6.        {
    7.            printf("Parallel level 1, thread num = %d",
                     omp_get_thread_num());
    8.        }
    9.        return 0;
    10.   }

  GDB Commands:
    ompd init
    b 7
    omptestapi ompd_get_version_string

*/

PyObject *test_ompd_get_version_string(PyObject *self, PyObject *noargs) {
  printf("Testing \"ompd_get_version_string\"...\n");

  const char *string;

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc = ompd_get_version_string(&string);
  if (rc != ompd_rc_ok) {
    printf("Failed, with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success. API version is %s\n", string);

  printf(
      "Test: Expecting ompd_rc_error or ompd_rc_bad_input for NULL version\n");
  rc = ompd_get_version_string(NULL);
  if (rc != ompd_rc_error && rc != ompd_rc_bad_input)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  return Py_None;
}

/*
  Test API: ompd_finalize

  Program:
    1.    #include <stdio.h>
    2.    #include <omp.h>
    3.    int main () {
    4.        omp_set_num_threads(2);
    5.        #pragma omp parallel
    6.        {
    7.            printf("Parallel level 1, thread num = %d",
                     omp_get_thread_num());
    8.        }
    9.        return 0;
    10.   }

  GDB Commands:
    ompd init
    b 7
    ompdtestapi ompd_finalize


    b 4
    r
    ompdtestapi ompd_finalize
*/

PyObject *test_ompd_finalize(PyObject *self, PyObject *noargs) {
  printf("Testing \"ompd_finalize\"...\n");

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc = ompd_finalize();
  if (rc == ompd_rc_ok)
    printf("Ret code: ompd_rc_ok, Success if ompd is initialized.\n");
  // ompd_rc_unsupported: if the OMPD library is not initialized.
  else if (rc == ompd_rc_unsupported)
    printf(
        "Ret code: ompd_rc_unsupported, Success if ompd is NOT initialized.\n");
  else
    printf("Failed: Return code is %d.\n", rc);

  return Py_None;
}

/*
  Test API: ompd_process_initialize

  Program:
    1.    #include <stdio.h>
    2.    #include <omp.h>
    3.    int main () {
    4.        omp_set_num_threads(2);
    5.        #pragma omp parallel
    6.        {
    7.            printf("Parallel level 1, thread num = %d",
                     omp_get_thread_num());
    8.        }
    9.        return 0;
    10.   }

  GDB Commands:

*/

PyObject *test_ompd_process_initialize(PyObject *self, PyObject *noargs) {

  printf("Testing \"ompd_process_initialize\"....\n");

  ompd_address_space_handle_t *addr_handle;

  //  ompd_address_space_context_t context = {42};

  printf("Test: with correct Args.\n");
  ompd_rc_t rc = ompd_process_initialize(&context, &addr_handle);
  if (rc != ompd_rc_ok) {
    printf("Failed, with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success.\n");

  printf("Test: With Unsupported library.\n");
  printf("Warning: Have to test manually with 32 and 64 bit combination.\n");

  //  ompd_address_space_context_t invalidcontext = {99};
  printf("Test: with wrong context value.\n");
  rc = ompd_process_initialize(&invalidcontext, &addr_handle);
  if ((rc != ompd_rc_bad_input) && (rc != ompd_rc_incompatible) &&
      (rc != ompd_rc_stale_handle))
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  // Random checks with  null and invalid args.
  /*
     ompd_rc_stale_handle: is returned when the specified handle is no
     longer valid;
     ompd_rc_bad_input: is returned when the input parameters
     (other than handle) are invalid;
     ompd_rc_error:    is returned when a fatal error occurred;
  */

  printf("Test: Expecting stale handle or bad_input for NULL addr_handle.\n");
  rc = ompd_process_initialize(&context, NULL);
  if ((rc != ompd_rc_bad_input) && (rc != ompd_rc_stale_handle))
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  return Py_None;
}

/*
  Test API: ompd_device_initialize

  Program:
    1.    #include <stdio.h>
    2.    #include <omp.h>
    3.    int main () {
    4.        omp_set_num_threads(2);
    5.        #pragma omp parallel
    6.        {
    7.            printf("Parallel level 1, thread num = %d",
                     omp_get_thread_num());
    8.        }
    9.        return 0;
    10.   }

  GDB Commands:

*/

PyObject *test_ompd_device_initialize(PyObject *self, PyObject *noargs) {
  printf("Testing Not enabled for \"ompd_device_initialize\".\n");
  printf("Disabled.\n");

  return Py_None;
}

/*
  Test API: ompd_rel_address_space_handle

  Program:
    1.    #include <stdio.h>
    2.    #include <omp.h>
    3.    int main () {
    4.        omp_set_num_threads(2);
    5.        #pragma omp parallel
    6.        {
    7.            printf("Parallel level 1, thread num = %d",
                     omp_get_thread_num());
    8.        }
    9.        return 0;
    10.   }

  GDB Commands:

*/
PyObject *test_ompd_rel_address_space_handle(PyObject *self, PyObject *noargs) {
  printf("Testing Not enabled for \"ompd_rel_address_space_handle\".\n");
  printf("Disabled.\n");

  return Py_None;
}

/*
  Test API: ompd_get_omp_version

  Program:
    1.    #include <stdio.h>
    2.    #include <omp.h>
    3.    int main () {
    4.        omp_set_num_threads(2);
    5.        #pragma omp parallel
    6.        {
    7.            printf("Parallel level 1, thread num = %d",
                     omp_get_thread_num());
    8.        }
    9.        return 0;
    10.   }

  GDB Commands:
    ompd init
    b 10
    c
    ompdtestapi ompd_get_omp_version
*/
PyObject *test_ompd_get_omp_version(PyObject *self, PyObject *args) {
  printf("Testing \"ompd_get_omp_version\" ...\n");

  PyObject *addrSpaceTup = PyTuple_GetItem(args, 0);
  ompd_address_space_handle_t *addr_handle =
      (ompd_address_space_handle_t *)PyCapsule_GetPointer(addrSpaceTup,
                                                          "AddressSpace");

  ompd_word_t omp_version;

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc = ompd_get_omp_version(addr_handle, &omp_version);
  if (rc != ompd_rc_ok) {
    printf("Failed, with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success. API version is %ld\n", omp_version);

  // Random checks with  null and invalid args.
  /*
     ompd_rc_stale_handle: is returned when the specified handle is no
     longer valid;
     ompd_rc_bad_input: is returned when the input parameters
     (other than handle) are invalid;
     ompd_rc_error:    is returned when a fatal error occurred;
  */

  printf("Test: Expecting stale handle or bad_input for NULL addr_handle.\n");
  rc = ompd_get_omp_version(NULL, &omp_version);
  if ((rc != ompd_rc_bad_input) && (rc != ompd_rc_stale_handle))
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  printf("Test: Expecting ompd_rc_error or bad_input for NULL omp_version.\n");
  rc = ompd_get_omp_version(addr_handle, NULL);
  if (rc != ompd_rc_error && rc != ompd_rc_bad_input)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  return Py_None;
}

/*
  Test API: ompd_get_omp_version_string

  Program:
    1.    #include <stdio.h>
    2.    #include <omp.h>
    3.    int main () {
    4.        omp_set_num_threads(2);
    5.        #pragma omp parallel
    6.        {
    7.            printf("Parallel level 1, thread num = %d",
                     omp_get_thread_num());
    8.        }
    9.        return 0;
    10.   }

  GDB Commands:
    ompd init
    b 7
    ompdtestapi ompd_get_omp_version_string
*/
PyObject *test_ompd_get_omp_version_string(PyObject *self, PyObject *args) {
  printf("Testing \"ompd_get_omp_version_string\" ...\n");

  PyObject *addrSpaceTup = PyTuple_GetItem(args, 0);
  ompd_address_space_handle_t *addr_handle =
      (ompd_address_space_handle_t *)PyCapsule_GetPointer(addrSpaceTup,
                                                          "AddressSpace");

  const char *string;

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc = ompd_get_omp_version_string(addr_handle, &string);
  if (rc != ompd_rc_ok) {
    printf("Failed, with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success. API version is %s\n", string);

  // Random checks with  null and invalid args.
  /*
     ompd_rc_stale_handle: is returned when the specified handle is no
     longer valid;
     ompd_rc_bad_input: is returned when the input parameters
     (other than handle) are invalid;
     ompd_rc_error:    is returned when a fatal error occurred;
  */

  printf("Test: Expecting stale handle or bad_input for NULL addr_handle.\n");
  rc = ompd_get_omp_version_string(NULL, &string);
  if ((rc != ompd_rc_bad_input) && (rc != ompd_rc_stale_handle))
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  printf("Test: Expecting ompd_rc_error or bad_input for NULL omp_version.\n");
  rc = ompd_get_omp_version_string(addr_handle, NULL);
  if (rc != ompd_rc_error && rc != ompd_rc_bad_input)
    printf("Failed, with return code = %d\n", rc);
  else
    printf("Success.\n");

  return Py_None;
}

/*
  Test API: ompd_get_curr_task_handle

  Program:
    1 #include <stdio.h>
    2 #include <omp.h>
    3 int get_fib_num (int num)
    4 {
    5   int t1, t2;
    6   if (num < 2)
    7     return num;
    8   else {
    9     #pragma omp task shared(t1)
    10       t1 = get_fib_num(num-1);
    11       #pragma omp task shared(t2)
    12       t2 = get_fib_num(num-2);
    13       #pragma omp taskwait
    14       return t1+t2;
    15     }
    16 }
    17
    18 int main () {
    19     int ret = 0;
    20     omp_set_num_threads(2);
    21     #pragma omp parallel
    22     {
    23       ret = get_fib_num(10);
    24     }
    25     printf ("Fib of 10 is %d", ret);
    26    return 0;
    27 }

  GDB Commands:
    ompd init
    b 10
    c
   ompdtestapi ompd_get_curr_task_handle
*/

PyObject *test_ompd_get_curr_task_handle(PyObject *self, PyObject *args) {
  printf("Testing \"ompd_get_curr_task_handle\"...\n");

  PyObject *threadHandlePy = PyTuple_GetItem(args, 0);
  ompd_thread_handle_t *thread_handle =
      (ompd_thread_handle_t *)(PyCapsule_GetPointer(threadHandlePy,
                                                    "ThreadHandle"));

  ompd_task_handle_t *task_handle;

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc = ompd_get_curr_task_handle(thread_handle, &task_handle);
  if (rc == ompd_rc_unavailable) {
    // ompd_rc_unavailable if the thread is not currently executing a task

    printf(
        "Success. Return code is ompd_rc_unavailable, Not executing a task.\n");
    printf("No more testing is possible.\n");
    return Py_None;
  } else if (rc == ompd_rc_stale_handle) {
    printf("Return code is stale_handle, may be in non parallel region.\n");
    printf("No more testing is possible.\n");
    return Py_None;
  } else if (rc != ompd_rc_ok)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  // Random checks with  null and invalid args.
  /*
     ompd_rc_stale_handle: is returned when the specified handle is no
     longer valid;
     ompd_rc_bad_input: is returned when the input parameters
     (other than handle) are invalid;
     ompd_rc_error:    is returned when a fatal error occurred;
  */

  printf("Test: Expecting ompd_rc_bad_input for NULL parallel_handle.\n");
  rc = ompd_get_curr_task_handle(thread_handle, NULL);
  if (rc != ompd_rc_bad_input)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  printf("Test: Expecting ompd_rc_error or stale_handle for NULL "
         "thread_handle.\n");
  rc = ompd_get_curr_task_handle(NULL, &task_handle);
  if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  return Py_None;
}

/*
  Test API: ompd_get_task_parallel_handle

  Program:
    1 #include <stdio.h>
    2 #include <omp.h>
    3 int get_fib_num (int num)
    4 {
    5   int t1, t2;
    6   if (num < 2)
    7     return num;
    8   else {
    9     #pragma omp task shared(t1)
    10       t1 = get_fib_num(num-1);
    11       #pragma omp task shared(t2)
    12       t2 = get_fib_num(num-2);
    13       #pragma omp taskwait
    14       return t1+t2;
    15     }
    16 }
    17
    18 int main () {
    19     int ret = 0;
    20     omp_set_num_threads(2);
    21     #pragma omp parallel
    22     {
    23       ret = get_fib_num(10);
    24     }
    25     printf ("Fib of 10 is %d", ret);
    26    return 0;
    27 }

  GDB Commands:
    ompd init
    b 10
    c
    ompdtestapi ompd_get_task_parallel_handle
*/
PyObject *test_ompd_get_task_parallel_handle(PyObject *self, PyObject *args) {

  printf("Testing \"ompd_get_task_parallel_handle\"...\n");

  PyObject *taskHandlePy = PyTuple_GetItem(args, 0);
  ompd_task_handle_t *task_handle =
      PyCapsule_GetPointer(taskHandlePy, "TaskHandle");

  ompd_parallel_handle_t *task_parallel_handle;

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc =
      ompd_get_task_parallel_handle(task_handle, &task_parallel_handle);
  if (rc != ompd_rc_ok) {
    printf("Failed. with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success.\n");

  // Random checks with  null and invalid args.
  /*
     ompd_rc_stale_handle: is returned when the specified handle is no
     longer valid;
     ompd_rc_bad_input: is returned when the input parameters
     (other than handle) are invalid;
     ompd_rc_error:    is returned when a fatal error occurred;
  */

  printf("Test: Expecting ompd_rc_bad_input for NULL task_parallel_handle.\n");
  rc = ompd_get_task_parallel_handle(task_handle, NULL);
  if (rc != ompd_rc_bad_input)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  printf(
      "Test: Expecting ompd_rc_error or stale_handle for NULL task_handle.\n");
  rc = ompd_get_task_parallel_handle(NULL, &task_parallel_handle);
  if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  return Py_None;
}

/*
  Test API: ompd_get_generating_task_handle

  Program:
    1 #include <stdio.h>
    2 #include <omp.h>
    3 int get_fib_num (int num)
    4 {
    5   int t1, t2;
    6   if (num < 2)
    7     return num;
    8   else {
    9     #pragma omp task shared(t1)
    10       t1 = get_fib_num(num-1);
    11       #pragma omp task shared(t2)
    12       t2 = get_fib_num(num-2);
    13       #pragma omp taskwait
    14       return t1+t2;
    15     }
    16 }
    17
    18 int main () {
    19     int ret = 0;
    20     omp_set_num_threads(2);
    21     #pragma omp parallel
    22     {
    23       ret = get_fib_num(10);
    24     }
    25     printf ("Fib of 10 is %d", ret);
    26    return 0;
    27 }

  GDB Commands:
    ompd init
    b 10
    c
    c // may or may not be needed
    ompdtestapi ompd_get_generating_task_handle
*/
PyObject *test_ompd_get_generating_task_handle(PyObject *self, PyObject *args) {
  printf("Testing \"ompd_get_generating_task_handle\"...\n");

  PyObject *taskHandlePy = PyTuple_GetItem(args, 0);
  ompd_task_handle_t *task_handle =
      (ompd_task_handle_t *)(PyCapsule_GetPointer(taskHandlePy, "TaskHandle"));
  ompd_task_handle_t *generating_task_handle;

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc =
      ompd_get_generating_task_handle(task_handle, &generating_task_handle);
  if (rc == ompd_rc_unavailable) {
    // ompd_rc_unavailable if no generating task handle exists.
    printf("Success. Return code is ompd_rc_unavailable\n");
    printf("No more testing is possible.\n");
    return Py_None;
  } else if (rc != ompd_rc_ok) {
    printf("Failed. with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success.\n");

  // Random checks with  null and invalid args.
  /*
     ompd_rc_stale_handle: is returned when the specified handle is no
     longer valid;
     ompd_rc_bad_input: is returned when the input parameters
     (other than handle) are invalid;
     ompd_rc_error:    is returned when a fatal error occurred;
  */

  printf(
      "Test: Expecting ompd_rc_bad_input for NULL generating_task_handle.\n");
  rc = ompd_get_generating_task_handle(task_handle, NULL);
  if (rc != ompd_rc_bad_input)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  printf(
      "Test: Expecting ompd_rc_error or stale_handle for NULL task_handle.\n");
  rc = ompd_get_generating_task_handle(NULL, &generating_task_handle);
  if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  return Py_None;
}

/*
  Test API: ompd_get_scheduling_task_handle

  Program:
    1 #include <stdio.h>
    2 #include <omp.h>
    3 int get_fib_num (int num)
    4 {
    5   int t1, t2;
    6   if (num < 2)
    7     return num;
    8   else {
    9     #pragma omp task shared(t1)
    10       t1 = get_fib_num(num-1);
    11       #pragma omp task shared(t2)
    12       t2 = get_fib_num(num-2);
    13       #pragma omp taskwait
    14       return t1+t2;
    15     }
    16 }
    17
    18 int main () {
    19     int ret = 0;
    20     omp_set_num_threads(2);
    21     #pragma omp parallel
    22     {
    23       ret = get_fib_num(10);
    24     }
    25     printf ("Fib of 10 is %d", ret);
    26    return 0;
    27 }

  GDB Commands:
    ompd init
    b 10
    c
    ompdtestapi ompd_get_scheduling_task_handle
*/
PyObject *test_ompd_get_scheduling_task_handle(PyObject *self, PyObject *args) {
  printf("Testing \"ompd_get_scheduling_task_handle\"...\n");

  PyObject *taskHandlePy = PyTuple_GetItem(args, 0);
  ompd_task_handle_t *task_handle =
      (ompd_task_handle_t *)(PyCapsule_GetPointer(taskHandlePy, "TaskHandle"));
  ompd_task_handle_t *scheduling_task_handle;

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc =
      ompd_get_scheduling_task_handle(task_handle, &scheduling_task_handle);
  if (rc == ompd_rc_unavailable) {
    // ompd_rc_unavailable if no generating task handle exists.
    printf(
        "Success. Return code is ompd_rc_unavailable, No scheduling task.\n");
    printf("No more testing is possible.\n");
    return Py_None;
  } else if (rc != ompd_rc_ok) {
    printf("Failed. with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success.\n");

  // Random checks with  null and invalid args.
  /*
     ompd_rc_stale_handle: is returned when the specified handle is no
     longer valid;
     ompd_rc_bad_input: is returned when the input parameters
     (other than handle) are invalid;
     ompd_rc_error:    is returned when a fatal error occurred;
  */

  printf(
      "Test: Expecting ompd_rc_bad_input for NULL scheduling_task_handle.\n");
  rc = ompd_get_scheduling_task_handle(task_handle, NULL);
  if (rc != ompd_rc_bad_input)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  printf(
      "Test: Expecting ompd_rc_error or stale_handle for NULL task_handle.\n");
  rc = ompd_get_scheduling_task_handle(NULL, &scheduling_task_handle);
  if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  return Py_None;
}

/*
  Test API: ompd_get_task_in_parallel

  Program:
  Program:
    1.    #include <stdio.h>
    2.    #include <omp.h>
    3.    int main () {
    4.        omp_set_num_threads(4);
    5.        #pragma omp parallel
    6.        {
    7.            printf("Parallel level 1, thread num = %d",
                     omp_get_thread_num());
    8.        }
    9.        return 0;
    10.   }

  GDB Commands:
    ompd init
    b 7
    c
   ompdtestapi ompd_get_task_in_parallel
*/
PyObject *test_ompd_get_task_in_parallel(PyObject *self, PyObject *args) {
  printf("Testing \"ompd_get_task_in_parallel\"...\n");

  PyObject *parallelHandlePy = PyTuple_GetItem(args, 0);
  ompd_parallel_handle_t *parallel_handle =
      (ompd_parallel_handle_t *)(PyCapsule_GetPointer(parallelHandlePy,
                                                      "ParallelHandle"));
  ompd_task_handle_t *task_handle;

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc = ompd_get_task_in_parallel(
      parallel_handle, 1 /* lesser than team-size-var*/, &task_handle);
  if (rc != ompd_rc_ok) {
    printf("Failed. with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success.\n");

  // ompd_rc_bad_input if the thread_num argument is greater than or equal to
  // the team-size-var ICV or negative
  printf("Test: Invalid thread num (199).\n");
  rc = ompd_get_task_in_parallel(parallel_handle, 199, &task_handle);
  if (rc != ompd_rc_bad_input)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  printf("Test: Invalid thread num (-5).\n");
  rc = ompd_get_task_in_parallel(parallel_handle, -5, &task_handle);
  if (rc != ompd_rc_bad_input)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  // Random checks with  null and invalid args.
  /*
     ompd_rc_stale_handle: is returned when the specified handle is no
     longer valid;
     ompd_rc_bad_input: is returned when the input parameters
     (other than handle) are invalid;
     ompd_rc_error:    is returned when a fatal error occurred;
  */

  printf("Test: Expecting ompd_rc_bad_input for NULL task_handle.\n");
  rc = ompd_get_task_in_parallel(parallel_handle, 1, NULL);
  if (rc != ompd_rc_bad_input)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  printf("Test: Expecting ompd_rc_error or stale_handle for NULL "
         "parallel_handle.\n");
  rc = ompd_get_task_in_parallel(NULL, 1, &task_handle);
  if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  return Py_None;
}

/*
  Test API: ompd_rel_task_handle

  Program:
    1 #include <stdio.h>
    2 #include <omp.h>
    3 int get_fib_num (int num)
    4 {
    5   int t1, t2;
    6   if (num < 2)
    7     return num;
    8   else {
    9     #pragma omp task shared(t1)
    10       t1 = get_fib_num(num-1);
    11       #pragma omp task shared(t2)
    12       t2 = get_fib_num(num-2);
    13       #pragma omp taskwait
    14       return t1+t2;
    15     }
    16 }
    17
    18 int main () {
    19     int ret = 0;
    20     omp_set_num_threads(2);
    21     #pragma omp parallel
    22     {
    23       ret = get_fib_num(10);
    24     }
    25     printf ("Fib of 10 is %d", ret);
    26    return 0;
    27 }

  GDB Commands:
    ompd init
    b 10
    c
    ompdtestapi ompd_rel_task_handle
*/
PyObject *test_ompd_rel_task_handle(PyObject *self, PyObject *noargs) {
  printf("Testing Not enabled for \"ompd_rel_task_handle\".\n");
  printf("Disabled.\n");

  return Py_None;
}

/*
  Test API: ompd_task_handle_compare

  Program:
    1 #include <stdio.h>
    2 #include <omp.h>
    3 int get_fib_num (int num)
    4 {
    5   int t1, t2;
    6   if (num < 2)
    7     return num;
    8   else {
    9     #pragma omp task shared(t1)
    10       t1 = get_fib_num(num-1);
    11       #pragma omp task shared(t2)
    12       t2 = get_fib_num(num-2);
    13       #pragma omp taskwait
    14       return t1+t2;
    15     }
    16 }
    17
    18 int main () {
    19     int ret = 0;
    20     omp_set_num_threads(2);
    21     #pragma omp parallel
    22     {
    23       ret = get_fib_num(10);
    24     }
    25     printf ("Fib of 10 is %d", ret);
    26    return 0;
    27 }

  GDB Commands:
    ompd init
    b 10
    c
    c
   ompdtestapi ompd_task_handle_compare
*/
PyObject *test_ompd_task_handle_compare(PyObject *self, PyObject *args) {
  printf("Testing \"ompd_task_handle_compare\"...\n");

  PyObject *taskHandlePy1 = PyTuple_GetItem(args, 0);
  ompd_task_handle_t *task_handle1 =
      (ompd_task_handle_t *)(PyCapsule_GetPointer(taskHandlePy1, "TaskHandle"));
  PyObject *taskHandlePy2 = PyTuple_GetItem(args, 1);
  ompd_task_handle_t *task_handle2 =
      (ompd_task_handle_t *)(PyCapsule_GetPointer(taskHandlePy2, "TaskHandle"));

  int cmp_value;

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc =
      ompd_task_handle_compare(task_handle1, task_handle2, &cmp_value);
  if (rc != ompd_rc_ok) {
    printf("Failed. with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success.\n");

  if (cmp_value == 0) {
    printf("Task Handles are Same.\n");
  } else {
    // a value less than, equal to, or greater than 0 indicates that the task
    // that corresponds to task_handle_1 is, respectively, less than, equal to,
    // or greater than the task that corresponds to task_handle_2.
    if (cmp_value <= 0) {
      printf("Task handle 1 is lesser than handle 2, cmp_val = %d\n",
             cmp_value);
      printf("Test: Changing the order.\n");
      rc = ompd_task_handle_compare(task_handle2, task_handle1, &cmp_value);
      if (rc != ompd_rc_ok) {
        printf("Failed. with return code = %d\n", rc);
        return Py_None;
      }
      if (cmp_value >= 0)
        printf("Success now cmp_value is greater, %d.\n", cmp_value);
      else
        printf("Failed.\n");
    } else {
      printf("Task 1 is greater than handle 2.\n");
      printf("Test: Changing the order.\n");
      rc = ompd_task_handle_compare(task_handle2, task_handle1, &cmp_value);
      if (rc != ompd_rc_ok) {
        printf("Failed. with return code = %d\n", rc);
        return Py_None;
      }
      if (cmp_value <= 0)
        printf("Success now cmp_value is lesser, %d.\n", cmp_value);
      else
        printf("Failed.\n");
    }

    // Random checks with  null and invalid args.
    /*
     ompd_rc_stale_handle: is returned when the specified handle is no
     longer valid;
     ompd_rc_bad_input: is returned when the input parameters
     (other than handle) are invalid;
     ompd_rc_error:    is returned when a fatal error occurred;
    */

    printf("Test: Expecting ompd_rc_bad_input for NULL cmp_value.\n");
    rc = ompd_task_handle_compare(task_handle2, task_handle1, NULL);
    if (rc != ompd_rc_bad_input)
      printf("Failed. with return code = %d\n", rc);
    else
      printf("Success.\n");

    printf("Test: Expecting ompd_rc_error or stale_handle for NULL "
           "task_handle.\n");
    rc = ompd_task_handle_compare(NULL, task_handle1, &cmp_value);
    if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
      printf("Failed. with return code = %d\n", rc);
    else
      printf("Success.\n");
  }

  return Py_None;
}

/*
  Test API: ompd_get_task_function

  Program:
    1 #include <stdio.h>
    2 #include <omp.h>
    3 int get_fib_num (int num)
    4 {
    5   int t1, t2;
    6   if (num < 2)
    7     return num;
    8   else {
    9     #pragma omp task shared(t1)
    10       t1 = get_fib_num(num-1);
    11       #pragma omp task shared(t2)
    12       t2 = get_fib_num(num-2);
    13       #pragma omp taskwait
    14       return t1+t2;
    15     }
    16 }
    17
    18 int main () {
    19     int ret = 0;
    20     omp_set_num_threads(2);
    21     #pragma omp parallel
    22     {
    23       ret = get_fib_num(10);
    24     }
    25     printf ("Fib of 10 is %d", ret);
    26    return 0;
    27 }

  GDB Commands:
    ompd init
    b 10
    c
   ompdtestapi ompd_get_task_function
*/
PyObject *test_ompd_get_task_function(PyObject *self, PyObject *args) {
  printf("Testing \"ompd_get_task_function\"...\n");

  PyObject *taskHandlePy = PyTuple_GetItem(args, 0);
  ompd_task_handle_t *task_handle =
      (ompd_task_handle_t *)(PyCapsule_GetPointer(taskHandlePy, "TaskHandle"));

  ompd_address_t entry_point;

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc = ompd_get_task_function(task_handle, &entry_point);
  if (rc != ompd_rc_ok) {
    printf("Failed. with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success. Entry point is %lx.\n", entry_point.address);

  // Random checks with  null and invalid args.
  /*
     ompd_rc_stale_handle: is returned when the specified handle is no
     longer valid;
     ompd_rc_bad_input: is returned when the input parameters
     (other than handle) are invalid;
     ompd_rc_error:    is returned when a fatal error occurred;
  */

  printf("Test: Expecting ompd_rc_bad_input for NULL entry_point.\n");
  rc = ompd_get_task_function(task_handle, NULL);
  if (rc != ompd_rc_bad_input)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  printf(
      "Test: Expecting ompd_rc_error or stale_handle for NULL task_handle.\n");
  rc = ompd_get_task_function(NULL, &entry_point);
  if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  return Py_None;
}

/*
  Test API: ompd_get_task_frame

  Program:
    1 #include <stdio.h>
    2 #include <omp.h>
    3 int get_fib_num (int num)
    4 {
    5   int t1, t2;
    6   if (num < 2)
    7     return num;
    8   else {
    9     #pragma omp task shared(t1)
    10       t1 = get_fib_num(num-1);
    11       #pragma omp task shared(t2)
    12       t2 = get_fib_num(num-2);
    13       #pragma omp taskwait
    14       return t1+t2;
    15     }
    16 }
    17
    18 int main () {
    19     int ret = 0;
    20     omp_set_num_threads(2);
    21     #pragma omp parallel
    22     {
    23       ret = get_fib_num(10);
    24     }
    25     printf ("Fib of 10 is %d", ret);
    26    return 0;
    27 }

  GDB Commands:
    ompd init
    b 10
    c
   ompdtestapi ompd_get_task_frame
*/
PyObject *test_ompd_get_task_frame(PyObject *self, PyObject *args) {
  printf("Testing \"ompd_get_task_frame\"...\n");

  PyObject *taskHandlePy = PyTuple_GetItem(args, 0);
  ompd_task_handle_t *task_handle =
      (ompd_task_handle_t *)(PyCapsule_GetPointer(taskHandlePy, "TaskHandle"));

  ompd_frame_info_t exit_frame;
  ompd_frame_info_t enter_frame;

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc = ompd_get_task_frame(task_handle, &exit_frame, &enter_frame);
  if (rc != ompd_rc_ok) {
    printf("Failed. with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success.\n");

  // Random checks with  null and invalid args.
  /*
     ompd_rc_stale_handle: is returned when the specified handle is no
     longer valid;
     ompd_rc_bad_input: is returned when the input parameters
     (other than handle) are invalid;
     ompd_rc_error:    is returned when a fatal error occurred;
  */

  printf("Test: Expecting ompd_rc_bad_input for NULL exit and enter frame.\n");
  rc = ompd_get_task_frame(task_handle, NULL, NULL);
  if (rc != ompd_rc_bad_input)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  printf(
      "Test: Expecting ompd_rc_error or stale handle for NULL task_handle.\n");
  rc = ompd_get_task_frame(NULL, &exit_frame, &enter_frame);
  if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  return Py_None;
}

/*
  Test API: ompd_get_state

  Program:
  Program:
    1.    #include <stdio.h>
    2.    #include <omp.h>
    3.    int main () {
    4.        omp_set_num_threads(4);
    5.        #pragma omp parallel
    6.        {
    7.            printf("Parallel level 1, thread num = %d",
                     omp_get_thread_num());
    8.        }
    9.        return 0;
    10.   }

  GDB Commands:
    ompd init
    b 7
    c
   ompdtestapi ompd_get_state
*/
PyObject *test_ompd_get_state(PyObject *self, PyObject *args) {
  printf("Testing \"ompd_get_state\"...\n");

  PyObject *threadHandlePy = PyTuple_GetItem(args, 0);
  ompd_thread_handle_t *thread_handle =
      (ompd_thread_handle_t *)(PyCapsule_GetPointer(threadHandlePy,
                                                    "ThreadHandle"));

  ompd_word_t state;
  ompt_wait_id_t wait_id;

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc = ompd_get_state(thread_handle, &state, &wait_id);
  if (rc != ompd_rc_ok) {
    printf("Failed. with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success.\n");

  // Random checks with  null and invalid args.
  /*
     ompd_rc_stale_handle: is returned when the specified handle is no
     longer valid;
     ompd_rc_bad_input: is returned when the input parameters
     (other than handle) are invalid;
     ompd_rc_error:    is returned when a fatal error occurred;
  */

  printf("Test: Expecting ompd_rc_error or stale handle for NULL "
         "thread_handle.\n");
  rc = ompd_get_state(NULL, &state, &wait_id);
  if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  return Py_None;
}

/*
  Test API: ompd_get_display_control_vars

  Program:
    1 #include <stdio.h>
    2 #include <omp.h>
    3 int get_fib_num (int num)
    4 {
    5   int t1, t2;
    6   if (num < 2)
    7     return num;
    8   else {
    9     #pragma omp task shared(t1)
    10       t1 = get_fib_num(num-1);
    11       #pragma omp task shared(t2)
    12       t2 = get_fib_num(num-2);
    13       #pragma omp taskwait
    14       return t1+t2;
    15     }
    16 }
    17
    18 int main () {
    19     int ret = 0;
    20     omp_set_num_threads(2);
    21     #pragma omp parallel
    22     {
    23       ret = get_fib_num(10);
    24     }
    25     printf ("Fib of 10 is %d", ret);
    26    return 0;
    27 }

  GDB Commands:
    ompd init
    b 10
    c
   ompdtestapi ompd_get_display_control_vars
*/
PyObject *test_ompd_get_display_control_vars(PyObject *self, PyObject *args) {
  printf("Testing \"ompd_get_display_control_vars\" ...\n");

  PyObject *addrSpaceTup = PyTuple_GetItem(args, 0);
  ompd_address_space_handle_t *addr_handle =
      (ompd_address_space_handle_t *)PyCapsule_GetPointer(addrSpaceTup,
                                                          "AddressSpace");

  const char *const *control_vars;

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc = ompd_get_display_control_vars(addr_handle, &control_vars);
  if (rc != ompd_rc_ok) {
    printf("Failed. with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success.\n");

  // Random checks with  null and invalid args.
  /*
     ompd_rc_stale_handle: is returned when the specified handle is no
     longer valid;
     ompd_rc_bad_input: is returned when the input parameters
     (other than handle) are invalid;
     ompd_rc_error:    is returned when a fatal error occurred;
  */

  printf("Test: Expecting stale handle or bad_input for NULL addr_handle.\n");
  rc = ompd_get_display_control_vars(NULL, &control_vars);
  if ((rc != ompd_rc_bad_input) && (rc != ompd_rc_stale_handle))
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  printf("Test: Expecting ompd_rc_error or bad_input for NULL control_vars.\n");
  rc = ompd_get_display_control_vars(addr_handle, NULL);
  if (rc != ompd_rc_error && rc != ompd_rc_bad_input)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  return Py_None;
}

/*
  Test API: ompd_rel_display_control_vars

  Program:
    1 #include <stdio.h>
    2 #include <omp.h>
    3 int get_fib_num (int num)
    4 {
    5   int t1, t2;
    6   if (num < 2)
    7     return num;
    8   else {
    9     #pragma omp task shared(t1)
    10       t1 = get_fib_num(num-1);
    11       #pragma omp task shared(t2)
    12       t2 = get_fib_num(num-2);
    13       #pragma omp taskwait
    14       return t1+t2;
    15     }
    16 }
    17
    18 int main () {
    19     int ret = 0;
    20     omp_set_num_threads(2);
    21     #pragma omp parallel
    22     {
    23       ret = get_fib_num(10);
    24     }
    25     printf ("Fib of 10 is %d", ret);
    26    return 0;
    27 }

  GDB Commands:
    ompd init
    b 10
    c
   ompdtestapi ompd_rel_display_control_vars
*/
PyObject *test_ompd_rel_display_control_vars(PyObject *self, PyObject *noargs) {
  printf("Testing Not enabled for \"ompd_rel_display_control_vars\".\n");
  printf("Disabled.\n");

  return Py_None;
}

/*
   Test API: ompd_enumerate_icvs

  Program:
  Program:
    1.    #include <stdio.h>
    2.    #include <omp.h>
    3.    int main () {
    4.        omp_set_num_threads(2);
    5.        #pragma omp parallel
    6.        {
    7.            printf("Parallel level 1, thread num = %d",
                     omp_get_thread_num());
    8.        }
    9.        return 0;
    10.   }

  GDB Commands:
    ompd init
    b 7
    c
   ompdtestapi ompd_enumerate_icvs
*/

PyObject *test_ompd_enumerate_icvs(PyObject *self, PyObject *args) {
  printf("Testing \"ompd_enumerate_icvs\"...\n");

  PyObject *addrSpaceTup = PyTuple_GetItem(args, 0);
  ompd_address_space_handle_t *addr_handle =
      (ompd_address_space_handle_t *)PyCapsule_GetPointer(addrSpaceTup,
                                                          "AddressSpace");

  ompd_icv_id_t current = 0; // To begin enumerating the ICVs, a tool should
                             // pass ompd_icv_undefined as the value of current
  ompd_icv_id_t next_id;
  const char *next_icv_name;
  ompd_scope_t next_scope;
  int more;

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc = ompd_enumerate_icvs(addr_handle, current, &next_id,
                                     &next_icv_name, &next_scope, &more);
  if (rc != ompd_rc_ok) {
    printf("Failed. with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success.\n");

  // ompd_rc_bad_input if an unknown value is provided in current
  printf("Test: Unknown current value.\n");
  rc = ompd_enumerate_icvs(
      addr_handle,
      99 /*unknown current value: greater than enum "ompd_icvompd_icv" */,
      &next_id, &next_icv_name, &next_scope, &more);
  if (rc != ompd_rc_bad_input)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  // Random checks with  null and invalid args.
  /*
     ompd_rc_stale_handle: is returned when the specified handle is no
     longer valid;
     ompd_rc_bad_input: is returned when the input parameters
     (other than handle) are invalid;
     ompd_rc_error:    is returned when a fatal error occurred;
  */

  printf(
      "Test: Expecting ompd_rc_bad_input for NULL next_id and next_icv_name\n");
  rc =
      ompd_enumerate_icvs(addr_handle, current, NULL, NULL, &next_scope, &more);
  if (rc != ompd_rc_bad_input)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  printf(
      "Test: Expecting ompd_rc_error or stale_handle for NULL addr_handle.\n");
  rc = ompd_enumerate_icvs(NULL, current, &next_id, &next_icv_name, &next_scope,
                           &more);
  if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  return Py_None;
}

/*
  Test API: ompd_get_icv_from_scope

  Program:
    1 #include <stdio.h>
    2 #include <omp.h>
    3 int get_fib_num (int num)
    4 {
    5   int t1, t2;
    6   if (num < 2)
    7     return num;
    8   else {
    9     #pragma omp task shared(t1)
    10       t1 = get_fib_num(num-1);
    11       #pragma omp task shared(t2)
    12       t2 = get_fib_num(num-2);
    13       #pragma omp taskwait
    14       return t1+t2;
    15     }
    16 }
    17
    18 int main () {
    19     int ret = 0;
    20     omp_set_num_threads(2);
    21     #pragma omp parallel
    22     {
    23       ret = get_fib_num(10);
    24     }
    25     printf ("Fib of 10 is %d", ret);
    26    return 0;
    27 }

  GDB Commands:
    ompd init
    b 10
    c
   ompdtestapi ompd_get_icv_from_scope
*/
PyObject *test_ompd_get_icv_from_scope_with_addr_handle(PyObject *self,
                                                        PyObject *args) {
  printf("Testing \"ompd_get_icv_from_scope with addr_handle\"...\n");

  PyObject *addrSpaceTup = PyTuple_GetItem(args, 0);
  ompd_address_space_handle_t *addr_handle =
      (ompd_address_space_handle_t *)PyCapsule_GetPointer(addrSpaceTup,
                                                          "AddressSpace");

  ompd_word_t icv_value;

  printf("Test: With Correct Arguments.\n");
  // cannot import enum ompd_icv from omp-icv.cpp, hardcoding as of now, if enum
  // changes it also requires modification
  ompd_rc_t rc = ompd_get_icv_from_scope(
      addr_handle, ompd_scope_address_space,
      19 /* ompd_icv_num_procs_var: check enum ompd_icv in omp-icv.cpp */,
      &icv_value);
  if (rc != ompd_rc_ok) {
    printf("Failed. with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success.\n");

  // ompd_rc_bad_input if an unknown value is provided in icv_id.
  printf("Test: bad_input for unknown icv_id.\n");
  rc = ompd_get_icv_from_scope(addr_handle, ompd_scope_address_space,
                               99 /*wrong value*/, &icv_value);
  if (rc != ompd_rc_bad_input)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  // ompd_rc_incompatible if the ICV cannot be represented as an integer;
  printf("Test: rc_incompatible for ICV that cant be represented as an "
         "integer.\n");
  rc = ompd_get_icv_from_scope(addr_handle, ompd_scope_address_space,
                               12 /*ompd_icv_tool_libraries_var*/, &icv_value);
  if (rc != ompd_rc_incompatible)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  // Random checks with  null and invalid args.
  /*
     ompd_rc_stale_handle: is returned when the specified handle is no
     longer valid;
     ompd_rc_bad_input: is returned when the input parameters
     (other than handle) are invalid;
     ompd_rc_error:    is returned when a fatal error occurred;
  */

  printf("Test: Expecting ompd_rc_bad_input for NULL icv_value.\n");
  rc = ompd_get_icv_from_scope(addr_handle, ompd_scope_address_space,
                               19 /*ompd_icv_num_procs_var*/, NULL);
  if (rc != ompd_rc_bad_input)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  printf("Test: Expecting ompd_rc_error for NULL handle.\n");
  rc = ompd_get_icv_from_scope(NULL, ompd_scope_address_space,
                               19 /*ompd_icv_num_procs_var*/, &icv_value);
  if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  return Py_None;
}

PyObject *test_ompd_get_icv_from_scope_with_thread_handle(PyObject *self,
                                                          PyObject *args) {
  printf("Testing \"ompd_get_icv_from_scope with thread_handle\"...\n");

  PyObject *threadHandlePy = PyTuple_GetItem(args, 0);
  ompd_thread_handle_t *thread_handle =
      (ompd_thread_handle_t *)(PyCapsule_GetPointer(threadHandlePy,
                                                    "ThreadHandle"));

  ompd_word_t icv_value;

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc = ompd_get_icv_from_scope(
      thread_handle, ompd_scope_thread,
      22 /* ompd_icv_thread_num_var  check enum ompd_icv in omp-icv.cpp */,
      &icv_value);
  if (rc != ompd_rc_ok) {
    printf("Failed. with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success.\n");

  printf("Test: with nthreads_var for ompd_rc_incomplete.\n");
  rc = ompd_get_icv_from_scope(thread_handle, ompd_scope_thread,
                               7 /*ompd_icv_nthreads_var*/, &icv_value);
  if (rc != ompd_rc_incomplete) {
    printf("Failed. with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success.\n");

  return Py_None;
}

PyObject *test_ompd_get_icv_from_scope_with_parallel_handle(PyObject *self,
                                                            PyObject *args) {
  printf("Testing \"ompd_get_icv_from_scope with parallel_handle\"...\n");

  PyObject *parallelHandlePy = PyTuple_GetItem(args, 0);
  ompd_parallel_handle_t *parallel_handle =
      (ompd_parallel_handle_t *)(PyCapsule_GetPointer(parallelHandlePy,
                                                      "ParallelHandle"));

  ompd_word_t icv_value;

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc = ompd_get_icv_from_scope(
      parallel_handle, ompd_scope_parallel,
      15 /*ompd_icv_active_levels_var:check enum ompd_icv in omp-icv.cpp */,
      &icv_value);
  if (rc != ompd_rc_ok) {
    printf("Failed. with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success.\n");

  return Py_None;
}

PyObject *test_ompd_get_icv_from_scope_with_task_handle(PyObject *self,
                                                        PyObject *args) {
  printf("Testing \"ompd_get_icv_from_scope with task_handle\"...\n");

  PyObject *taskHandlePy = PyTuple_GetItem(args, 0);
  ompd_task_handle_t *task_handle =
      (ompd_task_handle_t *)(PyCapsule_GetPointer(taskHandlePy, "TaskHandle"));

  ompd_word_t icv_value;

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc = ompd_get_icv_from_scope(
      task_handle, ompd_scope_task,
      16 /*ompd_icv_thread_limit_var: check enum ompd_icv in omp-icv.cpp */,
      &icv_value);
  if (rc != ompd_rc_ok) {
    printf("Failed. with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success.\n");

  return Py_None;
}
/*
   Test API: ompd_get_icv_string_from_scope

  Program:
    1.    #include <stdio.h>
    2.    #include <omp.h>
    3.    int main () {
    4.        omp_set_num_threads(4);
    5.        #pragma omp parallel
    6.        {
    7.            printf("Parallel level 1, thread num = %d",
                     omp_get_thread_num());
    8.        }
    9.        return 0;
    10.   }

  GDB Commands:
    ompd init
    b 7
    c
   ompdtestapi ompd_get_icv_string_from_scope
*/
PyObject *test_ompd_get_icv_string_from_scope(PyObject *self, PyObject *args) {
  printf("Testing \"ompd_get_icv_string_from_scope\"...\n");

  PyObject *addrSpaceTup = PyTuple_GetItem(args, 0);
  ompd_address_space_handle_t *addr_handle =
      (ompd_address_space_handle_t *)PyCapsule_GetPointer(addrSpaceTup,
                                                          "AddressSpace");

  const char *icv_string;

  printf("Test: With Correct Arguments.\n");
  ompd_rc_t rc = ompd_get_icv_string_from_scope(
      addr_handle, ompd_scope_address_space,
      12 /*ompd_icv_tool_libraries_var: check enum ompd_icv in omp-icv.cpp */,
      &icv_string);
  if (rc != ompd_rc_ok) {
    printf("Failed. with return code = %d\n", rc);
    return Py_None;
  } else
    printf("Success.\n");

  // ompd_rc_bad_input if an unknown value is provided in icv_id.
  printf("Test: bad_input for unknown icv_id.\n");
  rc = ompd_get_icv_string_from_scope(addr_handle, ompd_scope_address_space,
                                      99 /*wrong value*/, &icv_string);
  if (rc != ompd_rc_bad_input)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  // Random checks with  null and invalid args.
  /*
     ompd_rc_stale_handle: is returned when the specified handle is no
     longer valid;
     ompd_rc_bad_input: is returned when the input parameters
     (other than handle) are invalid;
     ompd_rc_error:    is returned when a fatal error occurred;
  */

  printf("Test: Expecting ompd_rc_bad_input for NULL icv_string.\n");
  rc = ompd_get_icv_string_from_scope(addr_handle, ompd_scope_address_space,
                                      12 /*ompd_icv_tool_libraries_var*/, NULL);
  if (rc != ompd_rc_bad_input)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  printf("Test: Expecting ompd_rc_error for NULL handle.\n");
  rc = ompd_get_icv_string_from_scope(NULL, ompd_scope_address_space,
                                      12 /*ompd_icv_tool_libraries_var*/,
                                      &icv_string);
  if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
    printf("Failed. with return code = %d\n", rc);
  else
    printf("Success.\n");

  return Py_None;
}

PyObject *test_ompd_get_tool_data(PyObject *self, PyObject *args) {
  printf("Disabled: Testing Not enabled for \"ompd_get_tool_data\".\n");

  return Py_None;
}
PyObject *test_ompd_enumerate_states(PyObject *self, PyObject *args) {
  printf("Disabled: Testing Not enabled for \"ompd_enumerate_states\".\n");

  return Py_None;
}
