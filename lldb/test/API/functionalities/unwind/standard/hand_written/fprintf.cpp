#include <cstdio>

int
main(int argc, char const *argv[])
{
    fprintf(stderr, "%d %p %s\n", argc, argv, argv[0]);
}
