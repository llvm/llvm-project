extern int callme(int input);

struct Foo {
  int a;
  int b;
  int c;
};

int
main(int argc, char **argv)
{
  // Set a breakpoint in main
  struct Foo mine = {callme(argc), 10, 20};
  return mine.a;
}

