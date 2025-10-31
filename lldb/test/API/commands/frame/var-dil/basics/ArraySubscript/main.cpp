#include <vector>

class myArray {
public:
  int m_array[4] = {7, 8, 9, 10};
  int m_arr_size = 4;
};

int main(int argc, char **argv) {
  int int_arr[] = {1, 2, 3};
  int *int_ptr = int_arr;
  int *int_ptr_1 = &int_arr[1];
  int(&int_arr_ref)[3] = int_arr;
  void *p_void = (void *)int_arr;

  int idx_1 = 1;
  const int &idx_1_ref = idx_1;

  typedef int td_int_t;
  typedef td_int_t td_td_int_t;
  typedef int *td_int_ptr_t;
  typedef int &td_int_ref_t;

  td_int_t td_int_idx_1 = 1;
  td_td_int_t td_td_int_idx_2 = 2;

  td_int_t td_int_arr[3] = {1, 2, 3};
  td_int_ptr_t td_int_ptr = td_int_arr;

  td_int_ref_t td_int_idx_1_ref = td_int_idx_1;
  td_int_t(&td_int_arr_ref)[3] = td_int_arr;

  enum Enum { kZero, kOne } enum_one = kOne;
  Enum &enum_ref = enum_one;

  std::vector<int> vector = {1, 2, 3};

  myArray ma;
  myArray *ma_ptr = &ma;

  return 0; // Set a breakpoint here
}
