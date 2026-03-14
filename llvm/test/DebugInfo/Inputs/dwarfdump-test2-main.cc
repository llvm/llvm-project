extern "C" int a();

int main() {
  return a();
}

// Built with gcc 4.6.3
// $ mkdir -p /tmp/dbginfo
// $ cp dwarfdump-test2-helper.cc dwarfdump-test2-main.cc /tmp/dbginfo/
// $ cd /tmp/dbginfo
// $ g++ -g dwarfdump-test2-helper.cc dwarfdump-test2-main.cc -o <output>
