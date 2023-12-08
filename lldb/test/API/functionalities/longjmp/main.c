#include <setjmp.h>
#include <stdio.h>
#include <time.h>

jmp_buf j;

void do_jump(void)
{
    // We can't let the compiler know this will always happen or it might make
    // optimizations that break our test.
    if (!clock())
        longjmp(j, 1); // non-local goto
}

int main (void)
{
    if (setjmp(j) == 0)
        do_jump();
    else
        return 0; // destination of longjmp

    return 1;
}
