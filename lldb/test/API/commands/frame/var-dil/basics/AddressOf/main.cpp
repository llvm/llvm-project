int globalVar = 0xDEADBEEF;

int main(int argc, char **argv) {
  int x = 42;
  int &r = x;
  int *p = &x;
  int *&pr = p;

  typedef int *&mypr;
  mypr my_pr = p;

  const char *s_str = "hello";

  char c = 1;
  return 0; // Set a breakpoint here
}
