#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* This program writes its arguments (separated by '\0') to stdout. */
int
main(int argc, char const *argv[])
{
    int i;

    FILE *output = fopen (argv[1], "w");
    if (output == NULL)
        exit (1);

    for (i = 2; i < argc; ++i)
        fwrite(argv[i], strlen(argv[i])+1, 1, output);

    fclose (output);

    return 0;
}
