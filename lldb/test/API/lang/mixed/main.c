int foo(void);
static int static_value = 0;

int
bar()
{
    static_value++;
    return static_value;
}

int main (int argc, char const *argv[])
{
    bar(); // breakpoint_in_main
    return foo();
}
