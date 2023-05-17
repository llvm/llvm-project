class Foo::Bar { int i = 123; };

int main(int argc, const char * argv[])
{
    IntContainer test(42);
    Foo::Bar bar;
    return 0; // break here
}
