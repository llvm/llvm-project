/*
 * ompd_test.c
 *
 *  Created on: Dec 28, 2014
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */

/*******************************************************************************
 * This implements an OMPD DLL for testing purposes.
 * It can be used as a template to implement (runtime-specific) OMPD DLLs.
 */

#include "ompd_test.h"
#include "assert.h"
#include <ompd.h>

static ompd_callbacks_t *callbacks = NULL;

ompd_rc_t ompd_initialize(ompd_callbacks_t *table) {
  ompd_rc_t ret = table ? ompd_rc_ok : ompd_rc_bad_input;
  callbacks = table;
  return ret;
}

/*******************************************************************************
 * Testing interface.
 * NOTE: *** These calls are not part of OMPD ***
 * These calls perform tests of each callback routine that is defined in the
 * debugger. The test routines start with "test_CB_".
 */

void test_print_header() { printf("\n*** OMPD Test ***\n"); }

void test_CB_dmemory_alloc() {
  assert(callbacks && "Invalid callbacks table");
  test_print_header();

  ompd_rc_t ret;
  ompd_size_t bytes = 1024;
  void *ptr = NULL;
  printf("Allocate %lu bytes of memory...", bytes);
  ret = callbacks->dmemory_alloc((ompd_context_t *)1, bytes, &ptr);
  if (ret == ompd_rc_ok && ptr != NULL)
    printf("Bytes allocated!\n");
  else
    printf("Failed!\n");

  printf("Free memory...");
  ret = callbacks->dmemory_free((ompd_context_t *)1, ptr);
  if (ret == ompd_rc_ok)
    printf("Memory freed.\n");
  else
    printf("Failed!\n");
}

void test_CB_tsizeof_prim() {
  assert(callbacks && "Invalid callbacks table");
  test_print_header();

  ompd_rc_t ret;
  ompd_device_type_sizes_t sizes;
  ret = callbacks->tsizeof_prim((ompd_context_t *)1, &sizes);
  if (ret == ompd_rc_ok) {
    printf("%-20s %du\n", "Size of char:", sizes.sizeof_char);
    printf("%-20s %du\n", "Size of short:", sizes.sizeof_short);
    printf("%-20s %du\n", "Size of int:", sizes.sizeof_int);
    printf("%-20s %du\n", "Size of long:", sizes.sizeof_long);
    printf("%-20s %du\n", "Size of long long:", sizes.sizeof_long_long);
    printf("%-20s %du\n", "Size of pointer:", sizes.sizeof_pointer);
  } else
    printf("Failed getting primitive sizes\n");
}
