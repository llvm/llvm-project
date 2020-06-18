int main(int argc, char **argv) {
  // Perform a null pointer access.
  int *const null_int_ptr = nullptr;
  *null_int_ptr = 0xDEAD;

  return 0;
}
