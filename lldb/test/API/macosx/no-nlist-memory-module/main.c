#include <fcntl.h>
#include <libgen.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/errno.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <unistd.h>

int get_return_value();
int get_return_value2();

// Create \a file_name with the c-string of our
// pid in it.  Initially open & write the contents
// to a temporary file, then move it to the actual
// filename once writing is completed.
bool writePid(const char *file_name, const pid_t pid) {
  char *tmp_file_name = (char *)malloc(strlen(file_name) + 16);
  strcpy(tmp_file_name, file_name);
  strcat(tmp_file_name, "_tmp");
  int fd = open(tmp_file_name, O_CREAT | O_WRONLY | O_TRUNC, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    fprintf(stderr, "open(%s) failed: %s\n", tmp_file_name, strerror(errno));
    free(tmp_file_name);
    return false;
  }
  char buffer[64];
  snprintf(buffer, sizeof(buffer), "%ld", (long)pid);
  bool res = true;
  if (write(fd, buffer, strlen(buffer)) == -1) {
    fprintf(stderr, "write(%s) failed: %s\n", buffer, strerror(errno));
    res = false;
  }
  close(fd);

  if (rename(tmp_file_name, file_name) == -1) {
    fprintf(stderr, "rename(%s, %s) failed: %s\n", tmp_file_name, file_name,
            strerror(errno));
    res = false;
  }
  free(tmp_file_name);

  return res;
}

int main(int argc, char **argv) {
  if (writePid(argv[1], getpid())) {
    // we've signaled lldb we are ready to be attached to,
    // this sleep() call will be interrupted when lldb
    // attaches.
    sleep(200);
  } else {
    printf("Error writing pid to '%s', exiting.\n", argv[1]);
    exit(3);
  }

  int retval = get_return_value();
  return retval + get_return_value2();
}
