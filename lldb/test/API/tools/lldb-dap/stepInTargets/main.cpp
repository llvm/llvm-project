
int foo(int val, int extra) { return val + extra; }

int funcA() { return 22; }

int funcB() { return 54; }

int main(int argc, char const *argv[]) {
  foo(funcA(), funcB()); // set breakpoint here
  return 0;
}
