// The real entry point. The default source file for line-only breakpoints
// should resolve to this file, not to other.cpp's ns::main.
namespace ns {
int main();
}

int main(int argc, char **argv) {
  int local = argc; // BREAK: entry point
  return ns::main() + local;
}
