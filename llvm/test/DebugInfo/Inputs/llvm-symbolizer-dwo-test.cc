int f(int a, int b) {
  return a + b;
}

int g(int a) {
  return a + 1;
}


int main() {
  return f(2, g(2));
}

// Built with Clang 3.5.0:
// $ mkdir -p /tmp/dbginfo
// $ cp llvm-symbolizer-dwo-test.cc /tmp/dbginfo
// $ cd /tmp/dbginfo
// $ clang -gsplit-dwarf llvm-symbolizer-dwo-test.cc
