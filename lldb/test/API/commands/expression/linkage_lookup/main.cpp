extern "C" int linkage_library_anchor();
extern "C" int missing_weak() __attribute__((weak_import));

__attribute__((noinline)) static int dwarf_callable() { return 31; }

int main() {
  int (*volatile missing_address)() = missing_weak;
  volatile int observed = dwarf_callable() + linkage_library_anchor();
  return observed + (missing_address != nullptr); // break here
}
