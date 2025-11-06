long g_watch_me_read = 1;
long g_watch_me_write = 2;
long g_temp = 3;

void watch_read() {
    g_temp = g_watch_me_read;
}

void watch_write() { g_watch_me_write = g_temp++; }

void read_watchpoint_testing() {
  watch_read(); // break here for read watchpoints
  g_temp = g_watch_me_read;
}

void watch_breakpoint_testing() {
  watch_write(); // break here for modify watchpoints
  g_watch_me_write = g_temp;
}

int main() {
  read_watchpoint_testing();
  watch_breakpoint_testing();
  return 0;
}
