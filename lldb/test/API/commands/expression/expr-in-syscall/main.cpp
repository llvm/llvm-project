#include <chrono>
#include <thread>

volatile int release_flag = 0;

int main(int argc, char const *argv[])
{
    while (! release_flag) // Wait for debugger to attach
        std::this_thread::sleep_for(std::chrono::seconds(3));

    return 0;
}
