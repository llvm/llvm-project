#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char const **argv) {
  // Set breakpoint 1 here
  execl("secondprog", "secondprog", NULL);
  perror("execve");
  abort();
}
