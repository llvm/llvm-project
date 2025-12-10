int main(int argc, char **argv) {
  int val = 1;
  int *p = &val;

  typedef int *myp;
  myp my_p = &val;

  typedef int *&mypr;
  mypr my_pr = p;

  return 0; // Set a breakpoint here
}
