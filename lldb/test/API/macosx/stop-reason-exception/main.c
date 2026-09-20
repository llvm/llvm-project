int main() {
  char *bad_ptr = (char *)0x400; // Set a breakpoint here
  bad_ptr[0] = 'a';
  return 0;
}
