static inline int inlined_f() {
  volatile int x = 2;
  return x;
}

int main() {
  return inlined_f();
}

// Build instructions:
// $ mkdir /tmp/dbginfo
// $ cp fission-ranges.cc /tmp/dbginfo/
// $ cd /tmp/dbginfo
// $ gcc -gsplit-dwarf -O2 -fPIC fission-ranges.cc -c -o obj2.o
// $ clang -gsplit-dwarf -O2 -fsanitize=address -fPIC -Dmain=foo fission-ranges.cc -c -o obj1.o
// $ gcc obj1.o obj2.o -shared -o <output>
// $ objcopy --remove-section=.debug_aranges <output>
