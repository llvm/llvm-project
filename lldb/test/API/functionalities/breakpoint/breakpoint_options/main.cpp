extern "C" int foo(void);
int main() { int argc = 0; char **argv = (char **)0;  // Set break point at this line.
  return foo();
}
