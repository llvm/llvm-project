#include "dylib.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char const *argv[]) {
  const char *a_name = "loadunload_a";
  const char *c_name = "loadunload_c";
  void *a_dylib_handle = NULL;
  void *c_dylib_handle = NULL; // Set break point at this line for test_lldb_process_load_and_unload_commands().
  int (*a_function)(void);

  a_dylib_handle = dylib_open(a_name);
  if (a_dylib_handle == NULL) {
    fprintf(stderr, "%s\n", dylib_last_error());
    exit(1);
  }

  a_function = (int (*)())dylib_get_symbol(a_dylib_handle, "a_function");
  if (a_function == NULL) {
    fprintf(stderr, "%s\n", dylib_last_error());
    exit(2);
  }
  printf("First time around, got: %d\n", a_function());
  dylib_close(a_dylib_handle);

  c_dylib_handle = dylib_open(c_name);
  if (c_dylib_handle == NULL) {
    fprintf(stderr, "%s\n", dylib_last_error());
    exit(3);
  }
  a_function = (int (*)())dylib_get_symbol(c_dylib_handle, "c_function");
  if (a_function == NULL) {
    fprintf(stderr, "%s\n", dylib_last_error());
    exit(4);
  }

  a_dylib_handle = dylib_open(a_name);
  if (a_dylib_handle == NULL) {
    fprintf(stderr, "%s\n", dylib_last_error());
    exit(5);
  }

  a_function = (int (*)())dylib_get_symbol(a_dylib_handle, "a_function");
  if (a_function == NULL) {
    fprintf(stderr, "%s\n", dylib_last_error());
    exit(6);
  }
  printf("Second time around, got: %d\n", a_function());
  dylib_close(a_dylib_handle);

  int LLDB_DYLIB_IMPORT d_function(void);
  printf("d_function returns: %d\n", d_function());

  return 0;
}
