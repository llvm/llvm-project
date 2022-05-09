#include <unistd.h>

int main(int argc, const char *argv[], const char *envp[]) {
  if (argc == 2) {
    // If we have two arguments the first is the path to this executable,
    // the second is the path to the linux dynamic loader that we should
    // exec with. We want to re-run this problem under the dynamic loader
    // and make sure we can hit the breakpoint in the "else".
    const char *interpreter = argv[1];
    const char *this_program = argv[0];
    const char *exec_argv[3] = {interpreter, this_program, nullptr};
    execve(interpreter, (char *const *)exec_argv, (char *const *)envp);
  }
  // Break here
  return 0;
}
