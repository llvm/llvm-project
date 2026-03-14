#include <stdio.h>
#include <assert.h>

const char *hello_world = "Hello, assertion!";

int main(int argc, const char* argv[])
{
    int *null_ptr = 0;
    printf("%s\n", hello_world);
    assert(null_ptr); // Assert here.
}
