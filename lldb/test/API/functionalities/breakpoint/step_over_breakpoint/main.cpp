
int func() { return 1; }

int
main() { int argc = 0; char **argv = (char **)0;

    int a = 0;      // breakpoint_1
    int b = func(); // breakpoint_2
    a = b + func(); // breakpoint_3
    return 0;       // breakpoint_4
}

