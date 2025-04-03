int func(void) {
  __builtin_printf("Break here");
  return 5;
}

int main(int argc, const char *argv[]) { return func(); }
