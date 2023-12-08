#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <stdio.h>

int
main (int argc, char const **argv)
{
    struct stat buf;
    int i, rv = 0; // Set breakpoint here.

    // Make sure stdin/stdout/stderr exist.
    for (i = 0; i <= 2; ++i) {
        if (fstat(i, &buf) != 0)
            return 1;
    }

    // Make sure no other file descriptors are open.
    for (i = 3; i <= 256; ++i) {
        if (fstat(i, &buf) == 0 || errno != EBADF) {
            fprintf(stderr, "File descriptor %d is open.\n", i);
            rv = 2;
        }
    }

    return rv;
}
