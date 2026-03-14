struct Base { int x; };
struct Derived : public Base { int y; };

struct NonDerived { int z; };

int main()
{
    Base base = {1111};

    Derived derived;
    derived.x = 2222;
    derived.y = 3333;

    NonDerived nd = {4444};
    return 0;     // Set break point at this line.
}
