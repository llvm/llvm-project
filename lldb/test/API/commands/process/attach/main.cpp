#include <stdio.h>

#include <chrono>
#include <thread>

int main(int argc, char const *argv[]) {
    int temp;
    lldb_enable_attach();

    // Waiting to be attached by the debugger.
    temp = 0;

    while (temp < 30) // Waiting to be attached...
    {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        temp++;
    }

    printf("Exiting now\n");
}
