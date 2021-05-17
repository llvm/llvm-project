#import <stdio.h>

void foo(int a, int b)
{
    printf("%d %d\n", a, b);
}

void bar(int *ptr) { printf("%d\n", *ptr); }

int main (int argc, const char * argv[])
{
    foo(42, 56);
    int i = 78;
    bar(&i);
    return 0;
}
