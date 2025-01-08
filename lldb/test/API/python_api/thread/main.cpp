#include <stdio.h>

// This simple program is to test the lldb Python API related to thread.

char my_char = 'u';
int my_int = 0;

void
call_me(bool should_spin) {
    int counter = 0;
    if (should_spin) {
        while (1)
            counter++;  // Set a breakpoint in call_me
     }
}

int main (int argc, char const *argv[])
{
    call_me(false);
    for (int i = 0; i < 3; ++i) {
        printf("my_char='%c'\n", my_char);
        ++my_char;
    }

    printf("after the loop: my_char='%c'\n", my_char); // 'my_char' should print out as 'x'.

    return 0; // Set break point at this line and check variable 'my_char'.
}
