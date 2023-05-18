#include <unistd.h>

int
SomethingToCall() {
  return 100;
}

int
main()
{
  while (1) {
    sleep(1);
  }
  return SomethingToCall();
}
