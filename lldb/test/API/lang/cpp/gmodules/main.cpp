class Foo::Bar { int i = 123; };

int main() { int argc = 0; char **argv = (char **)0;

    IntContainer test(42);
    Foo::Bar bar;
    return 0; // break here
}
