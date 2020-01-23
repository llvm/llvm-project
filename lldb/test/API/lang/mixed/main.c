int foo(void);
static int static_value = 0;

int
bar()
{
    static_value++;
    return static_value;
}

int main() { int argc = 0; char **argv = (char **)0;

    bar(); // breakpoint_in_main
    return foo();
}
