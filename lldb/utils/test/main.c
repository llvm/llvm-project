#include <stdio.h>
#include <stdlib.h>

int main(int argc, const char* argv[])
{
    int *null_ptr = 0;
    printf("Hello, fault!\n");
    u_int32_t val = (arc4random() & 0x0f);
    printf("val=%u\n", val);
    if (val == 0x07) // Lucky 7 :-)
        printf("Now segfault %d\n", *null_ptr);
    else
        printf("Better luck next time!\n");
}
