// Regenerate with Clang 22 or later, with default system-header coverage:
// clang++ -fprofile-instr-generate -fcoverage-mapping -mllvm -enable-name-compression=false \
//   -fcoverage-compilation-dir=/llvm-cov trailing-zero-length-region.cpp -o test
// llvm-cov convert-for-testing test -o trailing-zero-length-region.covmapping
// LLVM_PROFILE_FILE=zero.profraw ./test
// LLVM_PROFILE_FILE=nonzero.profraw ./test throw
// llvm-profdata merge -text zero.profraw -o trailing-zero-length-region-zero.proftext
// llvm-profdata merge -text nonzero.profraw -o trailing-zero-length-region-nonzero.proftext
#ifdef SYSTEM_HEADER
#pragma clang system_header
#define DIAG_IMPL() do { if (should_diag()) { diag(); } } while (0)
#define DIAG_ERROR() DIAG_IMPL()
#else
#define SYSTEM_HEADER
#include __FILE__
bool should_diag() { return true; }
void diag() {}
void may_throw(bool value) { if (value) throw 1; }
void watcher(bool);
int main(int argc, char **) {
  watcher(argc > 1);
  return 0;
}
void watcher(bool value) {
  try {
    may_throw(value);
  } catch (int) {
    DIAG_ERROR();
  }
}
#endif
