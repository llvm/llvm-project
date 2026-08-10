#include <signal.h>

int main(int argc, char const *argv[]) {
  raise(SIGABRT);
  return 0;
}
