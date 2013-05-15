#include <stdio.h>
#include <unistd.h>

int main(int argc, char const *argv[]) {
    // Waiting to be attached by the debugger.
    int temp = 0;
    while (temp < 30) // Waiting to be attached...
    {
        sleep(1);
        temp++;
    }

    printf("Exiting now\n");
}
