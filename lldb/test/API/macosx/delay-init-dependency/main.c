int foo();
int main(int argc, char **argv) {
  int retval = 0;
  // Only call foo() if one argument is passed
  if (argc == 2)
    retval = foo();

  return retval; // break here
}
