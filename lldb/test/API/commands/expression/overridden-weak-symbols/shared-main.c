extern int action(void);
extern int library_call(void);
int (*program_action)(void) = action;
int observed_action;

void main_context(void) {}

int main(void) {
  observed_action = library_call();
  main_context();
  return observed_action != 11;
}
