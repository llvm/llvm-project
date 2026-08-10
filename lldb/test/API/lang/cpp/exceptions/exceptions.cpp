#include <exception>
#include <stdio.h>

int throws_exception_on_even (int value);
int intervening_function (int value);
int catches_exception (int value);

int
catches_exception (int value)
{
    try
    {
        return intervening_function(value); // This is the line you should stop at for catch
    }
    catch (int value)
    {
        return value;  
    }
}

int 
intervening_function (int value)
{
    return throws_exception_on_even (2 * value);
}

int
throws_exception_on_even (int value)
{
    printf ("Mod two works: %d.\n", value%2);
    if (value % 2 == 0)
        throw 30;
    else
        return value;
}

int 
main ()
{
    catches_exception (10); // Stop here
    return 5;
}
