long g_watch_me_read = 1;
long g_watch_me_write = 2;
long g_temp = 3;

void watch_read() {
    g_temp = g_watch_me_read;
}

void watch_write() {
    g_watch_me_write = g_temp;
}

int main() {
    watch_read(); // Set a breakpoint here
    g_temp = g_watch_me_read; // Set breakpoint after call
    watch_write();
    g_watch_me_write = g_temp + 1;
    return 0;
}
