
int func() { return 1; }

int
main(int argc, char const *argv[])
{
    int a = 0;      // breakpoint_1
    int b = func(); // breakpoint_2
    a = b + func(); // breakpoint_3
    return 0;       // breakpoint_4
}

