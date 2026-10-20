// This is a dump little pair of test files

int func(int argc, const char *argv[]) {
  return (argc + 1) * (argv[argc][0] + 2);
}

int main(int argc, const char *argv[]) { return func(0, argv); }
