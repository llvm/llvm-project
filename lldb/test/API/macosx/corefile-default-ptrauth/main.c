int main();
int (*fmain)() = main;
int main () {
  return fmain();
}

