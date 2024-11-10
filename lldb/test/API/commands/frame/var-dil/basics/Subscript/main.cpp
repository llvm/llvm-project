#include <cstdint>
#include <limits>

class C {
 public:
  int field_ = 1337;
};


int
main(int argc, char **argv)
{

  const char* char_ptr = "lorem";
  const char char_arr[] = "ipsum";

  int int_arr[] = {1, 2, 3};

  C c_arr[2];
  c_arr[0].field_ = 0;
  c_arr[1].field_ = 1;

  C(&c_arr_ref)[2] = c_arr;

  int idx_1 = 1;
  const int& idx_1_ref = idx_1;

  typedef int td_int_t;
  typedef td_int_t td_td_int_t;
  typedef int* td_int_ptr_t;
  typedef int& td_int_ref_t;

  td_int_t td_int_idx_1 = 1;
  td_td_int_t td_td_int_idx_2 = 2;

  td_int_t td_int_arr[3] = {1, 2, 3};
  td_int_ptr_t td_int_ptr = td_int_arr;

  td_int_ref_t td_int_idx_1_ref = td_int_idx_1;
  td_int_t(&td_int_arr_ref)[3] = td_int_arr;

  unsigned char uchar_idx = std::numeric_limits<unsigned char>::max();
  uint8_t uint8_arr[256];
  uint8_arr[255] = 0xAB;
  uint8_t* uint8_ptr = uint8_arr;

  enum Enum { kZero, kOne } enum_one = kOne;
  Enum& enum_ref = enum_one;

  return 0; // Set a breakpoint here
}
