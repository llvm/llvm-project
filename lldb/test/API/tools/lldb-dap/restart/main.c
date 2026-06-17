int main(int argc, char const *argv[], char const *envp[]) {
  int i = 0;
  (void)i; // breakpoint A
  i = 1234;
  return 0; // breakpoint B
}
