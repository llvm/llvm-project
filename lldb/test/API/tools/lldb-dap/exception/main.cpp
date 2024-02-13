#include <signal.h>

int main() {
  raise(SIGABRT);
  return 0;
}
