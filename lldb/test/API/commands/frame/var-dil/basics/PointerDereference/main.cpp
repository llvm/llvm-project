#include <cstddef>

int main(int argc, char **argv) {
  int *p_null = nullptr;
  const char *p_char1 = "hello";

  typedef const char *my_char_ptr;
  my_char_ptr my_p_char1 = p_char1;

  int offset = 5;
  int array[10];
  array[0] = 0;
  array[offset] = offset;

  int (&array_ref)[10] = array;

  int *p_int0 = &array[0];
  int **pp_int0 = &p_int0;
  const int *cp_int0 = &array[0];
  const int *cp_int5 = &array[offset];

  typedef int *td_int_ptr_t;
  td_int_ptr_t td_int_ptr0 = &array[0];

  void *p_void = (void *)p_char1;
  void **pp_void0 = &p_void;
  void **pp_void1 = pp_void0 + 1;

  void **pp_void0_2 = &pp_void0[2];
  int *pp_int0_2stars = &**pp_int0;
  std::nullptr_t std_nullptr_t = nullptr;

  return 0; // Set a breakpoint here
}
