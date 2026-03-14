#include <chrono>
#include <stdio.h>
#include <thread>


int main(int argc, char const *argv[])
{
    static bool done = false;
    while (!done)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }
    printf("Set a breakpoint here.\n");
    return 0;
}

