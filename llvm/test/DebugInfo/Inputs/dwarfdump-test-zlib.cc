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

// Built with Clang 3.9 and GNU gold (GNU Binutils for Ubuntu 2.26) 1.11:
// Note: llvm-symbolizer-zlib.test relies on the path and filename used !
// $ mkdir -p /tmp/dbginfo
// $ cp dwarfdump-test-zlib.cc /tmp/dbginfo
// $ cd /tmp/dbginfo
// $ clang++ -g dwarfdump-test-zlib.cc -Wl,--compress-debug-sections=zlib -o dwarfdump-test-zlib.elf-x86-64
// $ clang++ -g dwarfdump-test-zlib.cc -Wa,--compress-debug-sections=zlib -c -o dwarfdump-test-zlib.o.elf-x86-64
// $ clang++ -g dwarfdump-test-zlib.cc -Wl,--compress-debug-sections=zlib-gnu -o dwarfdump-test-zlibgnu.elf-x86-64
// llvm-readobj --sections can be used to see that outputs really contain the compressed sections, also output in both
//   cases is slightly smaller, that is because of compression.
