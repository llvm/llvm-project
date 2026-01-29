#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int32_t global = 10; // Watchpoint variable declaration.

int main(int argc, char** argv) {
    int local = 0;
    printf("&global=%p\n", &global);
    printf("about to write to 'global'...\n"); // Set break point at this line.
                                               // When stopped, watch 'global'.
    global = 20;
    local += argc;
    ++local; // Set 2nd break point for disable_then_enable test case.
    printf("local: %d\n", local);

    const char *s = getenv("SW_WP_CASE");
    if (s == NULL)
      printf("global=%d\n", global);
    else
      global = 30;
}
