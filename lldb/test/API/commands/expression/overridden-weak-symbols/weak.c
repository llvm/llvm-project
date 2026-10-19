// Paired with main.c to test weak overrides in both CU orders:
// `action` is strong (or weak under BOTH_WEAK) and `other_action` is weak.
typedef int StrongResult;
#ifdef BOTH_WEAK
__attribute__((weak))
#endif
__attribute__((noinline)) StrongResult action(void) {
  return 11;
}
__attribute__((weak, noinline)) int other_action(void) { return 23; }
__attribute__((weak, noinline)) int weak_only(void) { return 7; }
__attribute__((weak)) int value = 22;

static int local_only(void) { return 77; }
static int scoped_action(void) { return 77; }

__attribute__((noinline)) void pointer_context(void) {
  int (*action)(void) = local_only;
  int value = 99;
  volatile int result = action() + value; // lookup in pointer scope
  (void)result;
}

#ifdef __cplusplus
namespace ns {
  __attribute__((noinline)) int selected(void) { return 33; }
} // namespace ns
#endif

void weak_context(void) {
  int local_value = local_only();
  local_value += scoped_action();
  local_value += 1; // break here
  pointer_context();
}
