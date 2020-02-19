char g_watch_me_read;
char g_watch_me_write;
char g_temp;

void watch_read() {
    g_temp = g_watch_me_read;
}

void watch_write() {
    g_watch_me_write = g_temp;
}

int main() {
    watch_read();
    g_temp = g_watch_me_read;
    watch_write();
    g_watch_me_write = g_temp;
    return 0;
}
