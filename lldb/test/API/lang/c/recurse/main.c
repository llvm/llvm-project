#include <stdint.h>
#include <stdio.h>

uint32_t
recurse_crash (uint32_t depth)
{
    if (depth > 0)
        return recurse_crash (depth - 1);
    return 0;
}

int
main (int argc, char const *argv[])
{
    // If we have more than one argument, then it should a depth to recurse to.
    // If we have just the program name as an argument, use UINT32_MAX so we
    // eventually crash the program by overflowing the stack
    uint32_t depth = UINT32_MAX;
    if (argc > 1)
    {
        char *end = NULL;
        depth = strtoul (argv[1], &end, 0);
        if (end == NULL || *end != '\0')
            depth = UINT32_MAX;
    }
    recurse_crash (depth);
    return 0;
}