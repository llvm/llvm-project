#include <stdio.h>

// This simple program is to test the lldb Python API related to thread.

char my_char = 'u';
int my_int = 0;

int main (int argc, char const *argv[])
{
    for (int i = 0; i < 3; ++i) {
        printf("my_char='%c'\n", my_char);
        ++my_char;
    }

    printf("after the loop: my_char='%c'\n", my_char); // 'my_char' should print out as 'x'.

    return 0; // Set break point at this line and check variable 'my_char'.
}
