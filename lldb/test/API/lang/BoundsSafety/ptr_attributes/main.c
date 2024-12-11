#include <ptrcheck.h>

int test_attrs(int *__counted_by(num_elements - 1) ptr_counted_by,
               int *__sized_by(num_elements * 4) ptr_sized_by,
               int *__counted_by_or_null(num_elements - 5) ptr_counted_by_or_null,
               int *__sized_by_or_null(num_elements * 2) ptr_sized_by_or_null, int *end,
               int *__ended_by(end) ptr_ended_by, int num_elements) {
  return 0; // break here
  // We expect to see this kind of output for a frame var:
  //   (__bounds_safety::counted_by::num_elements - 1)  ptr_counted_by = (int * ptr: 0x00016fdff0e0 counted_by: num_elements - 1)
  //   (__bounds_safety::sized_by::num_elements * 4)    ptr_sized_by = (int * ptr: 0x00016fdff0e0 sized_by: num_elements * 4)
  //   (__bounds_safety::counted_by_or_null::num_elements - 5)  ptr_counted_by_or_null = (int * ptr: 0x00016fdff0e0 counted_by_or_null: num_elements - 5)
  //   (__bounds_safety::sized_by_or_null::num_elements * 2)    ptr_sized_by_or_null = (int * ptr: 0x00016fdff0e0 sized_by_or_null: num_elements * 2)
  //   (__bounds_safety::dynamic_range::ptr_ended_by::) end = (int * ptr: 0x00016fdff0f4 start_expr: ptr_ended_by)
  //   (__bounds_safety::dynamic_range::::end)          ptr_ended_by = (int * ptr: 0x00016fdff0e0 end_expr: end)
}

int main() {
  int array[6] = {1, 2, 3, 4, 5, 6};
  test_attrs(array /*counted_by*/, array /*sized_by*/,
             array /*counted_by_or_null*/, array /*sized_by_or_null*/,
             array + 5 /*end*/, array /*ended by*/, 6);
  return 0; // break here
}
