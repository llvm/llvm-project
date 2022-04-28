int bar(int x, int y) { return x + y + 5; }

int foo(int x, int y) { return bar(x, y) + 12; }

int main(int argc, char **argv) { return foo(33, 78); }
