#include <stdio.h>
#include <stdint.h>

int32_t global = 10; // Watchpoint variable declaration.

int main(int argc, char** argv) {
    int local = 0;
    printf("&global=%p\n", &global);
    printf("about to write to 'global'...\n"); // Set break point at this line.
                                               // When stopped, watch 'global' for read&write.
    global = 20;
    local += argc;
    ++local;
    printf("local: %d\n", local);
    printf("global=%d\n", global);
}
