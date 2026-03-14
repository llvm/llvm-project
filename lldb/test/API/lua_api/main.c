#include <stdio.h>

void BFunction()
{
}

void AFunction()
{
    printf("I am a function.\n");
}

int main(int argc, const char *argv[])
{
    int inited = 0xDEADBEEF;
    int sum = 0;
    if(argc > 1)
    {
        for(int i = 0; i < argc; i++)
        {
            puts(argv[i]);
        }
        if(argc > 2)
        {
            return argc;
        }
    }
    AFunction();
    for(int i = 1; i <= 100; i++)
    {
        BFunction();
        sum += i;
    }
    printf("sum = %d\n", sum);
    return 0;
}
