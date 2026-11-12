typedef int Result;
__attribute__((weak, noinline)) Result action(void) { return 22; }
int (*program_action)(void) = action;
void context(void) {}
int main(void) {
  context();
  return action() != 11;
}
