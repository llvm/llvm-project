#include "dylib.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char const *argv[]) {
  const char *a_name = "load_a";
  void *a_dylib_handle = NULL;

  a_dylib_handle = dylib_open(a_name); // Set a breakpoint here.
  if (a_dylib_handle == NULL) { // Set another here - we should not hit this one
    fprintf(stderr, "%s\n", dylib_last_error());
    exit(1);
  }

  const char *b_name = "load_b";
  void *b_dylib_handle = NULL;

  b_dylib_handle = dylib_open(b_name);
  if (b_dylib_handle == NULL) { // Set a third here - we should not hit this one
    fprintf(stderr, "%s\n", dylib_last_error());
    exit(1);
  }

  return 0;
}
