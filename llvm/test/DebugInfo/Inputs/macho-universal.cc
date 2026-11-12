// Built with Apple LLVM version 4.2 (clang-425.0.24) (based on LLVM 3.2svn)
// clang++ -arch x86_64 -arch i386 macho-universal.cc

int inc(int x) {
  return x + 1;
}

int main(int argc, char *argv[]) {
  return inc(argc);
}
