void stop() {}

int main(int argc, char **argv) {
  int offset = 5;
  int array[10];
  array[0] = 0;
  array[offset] = offset;
  int (&array_ref)[10] = array;
  int *p_int0 = &array[0];

  const char *p_char = "hello!";
  const char *p_char5 = p_char + 5;
  typedef const char *my_char_ptr;
  my_char_ptr my_p_char = p_char;

  int **pp_int0 = &p_int0;
  const int *cp_int0 = &array[0];
  const int *cp_int5 = &array[offset];

  typedef int *td_int_ptr_t;
  td_int_ptr_t td_int_ptr0 = &array[0];

  void *p_void = (void *)p_char;
  void **pp_void0 = &p_void;
  void **pp_void1 = pp_void0 + 1;

  int *int_null = nullptr;

  stop(); // Set a breakpoint here
  return 0;
}
