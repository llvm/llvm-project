#if !defined(VERBOSE_TRAP_TEST_CATEGORY) || !defined(VERBOSE_TRAP_TEST_MESSAGE)
#error Please define required macros
#endif

struct Dummy {
  void func() { __builtin_verbose_trap(VERBOSE_TRAP_TEST_CATEGORY, VERBOSE_TRAP_TEST_MESSAGE); }
};

int main() {
  Dummy{}.func();
  return 0;
}
