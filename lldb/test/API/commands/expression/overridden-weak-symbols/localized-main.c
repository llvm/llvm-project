int action(void) { return 11; }
int (*global_action)(void) = action;
extern int local_call(void);

int main(void) { return local_call() != 22 || action() != 11; }
