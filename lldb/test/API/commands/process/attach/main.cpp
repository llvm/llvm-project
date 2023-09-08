#include <stdio.h>

#include <chrono>
#include <thread>

volatile int g_val = 12345;

int main(int argc, char const *argv[]) {
    int temp;
    lldb_enable_attach();

    // Waiting to be attached by the debugger.
    temp = 0;

    if (argc > 1 && argv[1][0] == 'q')
      return 0;
   
    while (temp < 30) {
      std::this_thread::sleep_for(std::chrono::seconds(2)); // Waiting to be attached...
      temp++;
    } 

    printf("Exiting now\n");
}
