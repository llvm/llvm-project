#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <unistd.h>

int get_return_value();
int get_return_value2();

int main(int argc, char **argv) {

  // Remove libno-nlists.dylib that we are linked against.
  char executable_path[PATH_MAX];
  realpath(argv[0], executable_path);
  executable_path[PATH_MAX - 1] = '\0';

  char *dir = dirname(executable_path);
  char dylib_path[PATH_MAX];
  snprintf(dylib_path, PATH_MAX, "%s/%s", dir, "libno-nlists.dylib");
  dylib_path[PATH_MAX - 1] = '\0';
  struct stat sb;
  if (stat(dylib_path, &sb) == -1) {
    printf("Could not find dylib %s to remove it\n", dylib_path);
    exit(1);
  }
  if (unlink(dylib_path) == -1) {
    printf("Could not remove dylib %s\n", dylib_path);
    exit(2);
  }
  snprintf(dylib_path, PATH_MAX, "%s/%s", dir, "libhas-nlists.dylib");
  dylib_path[PATH_MAX - 1] = '\0';
  if (stat(dylib_path, &sb) == -1) {
    printf("Could not find dylib %s to remove it\n", dylib_path);
    exit(1);
  }
  if (unlink(dylib_path) == -1) {
    printf("Could not remove dylib %s\n", dylib_path);
    exit(2);
  }

  // This sleep will exit as soon as lldb attaches
  // and interrupts it.
  sleep(200);

  int retval = get_return_value();
  return retval + get_return_value2();
}
