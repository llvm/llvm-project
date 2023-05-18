class DummyClass {
  int a_;
 public:
  DummyClass(int a) : a_(a) {}
  int add(int b) {
    return a_ + b;
  }
};

int f(int a, int b) {
  DummyClass c(a);
  return c.add(b);
}

int main() {
  return f(2, 3);
}

// Built with Clang 3.2:
// $ mkdir -p /tmp/dbginfo
// $ cp dwarfdump-test.cc /tmp/dbginfo
// $ cd /tmp/dbginfo
// $ clang++ -g dwarfdump-test.cc -o <output>

// The result is also used as an input to .dwz tool:
// $ cp <output> output1.dwz
// $ cp <output> output2.dwz
// $ dwz -m output.dwz -r output1.dwz output2.dwz
// $ rm output2.dwz

// The mach-o version was generated using clang-3.6.2.
