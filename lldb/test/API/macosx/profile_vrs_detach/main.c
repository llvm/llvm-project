#include <stdio.h>
#include <unistd.h>

int
main()
{
  while (1) {
    sleep(1); // Set a breakpoint here
    printf("I slept\n");
  }
  return 0;
}
