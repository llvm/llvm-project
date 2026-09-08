#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// Write our pid into file_name atomically (write to a temp file, then rename)
// so the test never observes a partially written pid.
static int write_pid(const char *file_name) {
  char tmp_name[1024];
  snprintf(tmp_name, sizeof(tmp_name), "%s_tmp", file_name);

  int fd = open(tmp_name, O_CREAT | O_WRONLY | O_TRUNC, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    fprintf(stderr, "open(%s) failed: %s\n", tmp_name, strerror(errno));
    return 1;
  }

  char buffer[64];
  int len = snprintf(buffer, sizeof(buffer), "%ld", (long)getpid());
  int result = 0;
  if (write(fd, buffer, len) == -1) {
    fprintf(stderr, "write failed: %s\n", strerror(errno));
    result = 1;
  }
  close(fd);

  if (rename(tmp_name, file_name) == -1) {
    fprintf(stderr, "rename failed: %s\n", strerror(errno));
    result = 1;
  }
  return result;
}

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "invalid number of command line arguments\n");
    return 1;
  }

  // Tell the kernel to refuse all debugger attachments to this process. Any
  // subsequent ptrace(PT_ATTACHEXC) against us makes the kernel deliver SIGSEGV
  // to the attaching process (debugserver).
  if (ptrace(PT_DENY_ATTACH, 0, 0, 0) == -1) {
    fprintf(stderr, "ptrace(PT_DENY_ATTACH) failed: %s\n", strerror(errno));
    return 1;
  }

  if (write_pid(argv[1]) != 0)
    return 1;

  // Wait for the debugger to try (and fail) to attach.
  while (1)
    sleep(60);

  return 0;
}
