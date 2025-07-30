#include "attach.h"
#include <cstdio>
#include <unistd.h>

int main(int argc, char *argv[]) {
  lldb_enable_attach();
  execvp(argv[1], argv+1);
  perror("execlp");
  return 1;
}
