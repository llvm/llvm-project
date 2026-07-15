int false_condition() { return 0; }

int *g_watched_var_ptr;

static void start_recording() {}

static void trigger_watchpoint() { *g_watched_var_ptr = 2; }

static void trigger_breakpoint() {}

static void stop_recording() {}

int main() {
  // The watched memory location is on the stack because
  // that's what our reverse execution engine records and
  // replays.
  int watched_var = 1;
  g_watched_var_ptr = &watched_var;

  start_recording();
  trigger_watchpoint();
  trigger_breakpoint();
  stop_recording();
  return 0;
}
