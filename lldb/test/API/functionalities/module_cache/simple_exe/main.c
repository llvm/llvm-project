#ifdef EXTRA_FUNCTION
int foo(int i) {
  return i*3;
}
#endif

int main (int argc, char const *argv[]) {
#ifdef EXTRA_FUNCTION
  return foo(argc);
#else
  return 0;
#endif
}
