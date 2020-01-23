extern "C" int foo(void);
static int static_value = 0;
static int id = 1234;

int
bar()
{
    static_value++;
    id++;
    return static_value + id;
}

int main() { int argc = 0; char **argv = (char **)0;

    bar(); // breakpoint_in_main
    return foo();
}
