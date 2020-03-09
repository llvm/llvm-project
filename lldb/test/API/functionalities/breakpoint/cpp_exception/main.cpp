#include <exception>

void
throws_int ()
{
    throw 5;
}

int
main ()
{
    throws_int();
}
