// RUN: %test_c_compiler %s %ompd-dl-src -I%ompd-dl-inc -ldl -o %t
// RUN: %t %ompd-lib | FileCheck %s
// REQUIRES: linux

#include "ompdDLService.h"

#include <stdio.h>

static const char *err_or_none(void) {
  const char *err = ompd_get_dl_error();
  return err ? err : "none";
}

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;

  if (ompd_load_library("") == 0)
    return 2;
  printf("empty: fail\n");
  printf("empty-err: %s\n", err_or_none());

  if (ompd_load_library("/no/such/libompd.so") == 0)
    return 3;
  printf("missing: fail\n");
  printf("missing-err: %s\n", err_or_none());

  if (ompd_load_library(argv[1]) != 0)
    return 4;
  printf("load: ok\n");
  printf("load-err: %s\n", err_or_none());

  if (!ompd_get_symbol("ompd_initialize"))
    return 5;
  printf("init-sym: ok\n");
  printf("init-sym-err: %s\n", err_or_none());

  if (ompd_get_symbol("no_such_ompd_symbol"))
    return 6;
  printf("missing-sym: fail\n");
  printf("missing-sym-err: %s\n", err_or_none());

  return 0;
}

// CHECK: empty: fail
// CHECK-NEXT: empty-err: OMPD library path is empty
// CHECK-NEXT: missing: fail
// CHECK-NEXT: missing-err: {{.+}}
// CHECK-NEXT: load: ok
// CHECK-NEXT: load-err: none
// CHECK-NEXT: init-sym: ok
// CHECK-NEXT: init-sym-err: none
// CHECK-NEXT: missing-sym: fail
// CHECK-NEXT: missing-sym-err: {{.+}}
