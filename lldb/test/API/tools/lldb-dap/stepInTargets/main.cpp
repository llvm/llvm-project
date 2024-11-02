
int foo(int val, int extra) { return val + extra; }

int bar() { return 22; }

int bar2() { return 54; }

int main(int argc, char const *argv[]) {
  foo(bar(), bar2()); // set breakpoint here
  return 0;
}
