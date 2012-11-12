#include <stdio.h>
#include "foo.h"

int 
foo (struct bar *bar_ptr)
{
    return printf ("bar_ptr = %p\n", bar_ptr);
}
