__attribute__((weak, noinline)) int action(void) { return 22; }
int library_call(void) { return action(); }
