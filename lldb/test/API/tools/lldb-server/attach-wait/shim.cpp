#include <unistd.h>
#include <cstdio>

int main(int argc, char *argv[]) {
  lldb_enable_attach();
  execlp(argv[1], argv[1], nullptr);
  perror("execlp");
  return 1;
}
