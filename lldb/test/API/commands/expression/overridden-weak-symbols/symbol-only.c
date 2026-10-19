int action(void) { return 11; }

__attribute__((noinline)) void context(void) {}

int main(void) {
  context();
  return action() != 11;
}
