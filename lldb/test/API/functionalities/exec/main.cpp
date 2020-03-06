#define _POSIX_C_SOURCE 200809L

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <libgen.h>
#include <string>
#include <unistd.h>

int main(int argc, char const **argv) {
  char *buf = strdup(argv[0]); // Set breakpoint 1 here
  std::string directory_name(::dirname(buf));

  std::string other_program = directory_name + "/secondprog";
  execve(other_program.c_str(), const_cast<char *const *>(argv), nullptr);
  perror("execve");
  abort();
}
