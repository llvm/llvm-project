#include <stdio.h>
#include <chrono>
#include <thread>

void call_me() {
  printf("I was called");
  std::this_thread::sleep_for(std::chrono::seconds(3));
}

int
main()
{
  call_me(); // Set a breakpoint here
  return 0;
}

