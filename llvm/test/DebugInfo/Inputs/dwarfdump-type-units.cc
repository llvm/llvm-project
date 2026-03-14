struct foo {};
struct bar {};
void sink(void*);
int main() {
  foo f;
  sink(&f);
  bar b;
  sink(&b);
}

// Built with GCC 4.8.1
// $ mkdir -p /tmp/dbginfo
// $ cp dwarfdump-type-units.cc /tmp/dbginfo
// $ cd /tmp/dbginfo
// $ g++-4.8.1 -g -fdebug-types-section -c dwarfdump-type-units.cc -o dwarfdump-type-units.elf-x86-64
