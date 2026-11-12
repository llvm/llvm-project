static int foo(int x, int y) {
  return x + y; // BREAK HERE
}

static int bar(int x) {
  return foo(x + 1, x * 2);
}

int main (int argc, char const *argv[])
{
    return bar(argc + 2);
}
