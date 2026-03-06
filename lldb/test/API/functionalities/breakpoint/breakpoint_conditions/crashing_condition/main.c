unsigned int do_crash() {
  char *bad_ptr = (char *)(0x20);
  bad_ptr[0] = 'a';
  return bad_ptr[0] == 'a';
}

int main() {
  char *filler = "Set a start breakpoint here";
  do_crash(); // Set the test breakpoint here
  return 0;
}
