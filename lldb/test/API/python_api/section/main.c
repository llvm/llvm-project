#include <stdio.h>
#include <string.h>

// This simple program is to test the lldb Python API SBSection. It includes
// somes global data, and so the build process produces a DATA section, which 
// the test code can use to query for the target byte size

char my_global_var_of_char_type = 'X';

int main (int argc, char const *argv[])
{
    // this code just "does something" with the global so that it is not
    // optimised away
    if (argc > 1 && strlen(argv[1]))
    {
        my_global_var_of_char_type += argv[1][0];
    }

    return my_global_var_of_char_type;
}
