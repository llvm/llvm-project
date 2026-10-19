// Paired with weak.c: `action` is weak and `other_action` is strong.
typedef int WeakResult;
__attribute__((weak, noinline)) WeakResult action(void) { return 22; }
#ifndef __cplusplus
extern __typeof(action) retained_action __attribute__((alias("action")));
#endif
__attribute__((noinline)) int other_action(void) { return 13; }
__attribute__((noinline)) int scoped_action(int n) { return 66 + n; }

int value = 11;
int (*program_action)(void) = action;
int *program_value = &value;
int observed_action, observed_other;

void weak_context(void);
int weak_only(void);
__attribute__((noinline)) void main_context(void) {}

#ifdef __cplusplus
int &value_ref = value;

namespace ns {
  __attribute__((weak, noinline)) int selected(void) { return 32; }
  __attribute__((noinline)) int selected(int n) { return n + 1; }
} // namespace ns
#endif

int main(void) {
  observed_action = action();
  observed_other = other_action();
  weak_context();
  main_context();
  return 0;
}
