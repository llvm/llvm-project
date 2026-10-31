int action(void) { return 22; }
int (*local_action)(void) = action;

void local_context(void) {}

int local_call(void) {
  int result = action();
  local_context();
  return result;
}
