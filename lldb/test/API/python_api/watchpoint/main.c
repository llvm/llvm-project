#include <stdio.h>
#include <stdint.h>

int32_t global = 10; // Watchpoint variable declaration.

static void watch_local(void) {
  int32_t local_value = 10;
  printf("local_value: %d\n", local_value); // local_value_breakpoint
}

static void reuse_stack_after_return(void) {
  int32_t scratch[20] = {0};
  for (int i = 0; i < 20; ++i)
    scratch[i] = 20;
  printf("scratch[0]: %d\n", scratch[0]);
}

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

    watch_local();
    reuse_stack_after_return();
}
