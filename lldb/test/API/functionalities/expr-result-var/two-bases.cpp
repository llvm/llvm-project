#include <stdint.h>
#include <stdio.h>

struct Base {
  virtual ~Base() = default;
  int base_int = 100;
  Base *return_me() { return this; }
};

struct Base_1 : public VIRTUAL Base {
  virtual ~Base_1() = default;
  int base_1_arr[10] = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
  Base *return_base_1() { return return_me(); }
};

struct Base_2 : public VIRTUAL Base {
  virtual ~Base_2() = default;
  int base_2_arr[10] = {200, 201, 202, 203, 204, 205, 206, 207, 208, 209};
  Base *return_base_2() { return return_me(); }
};

struct Derived : public Base_1, Base_2 {
  virtual ~Derived() = default;
  int derived_int = 1000;
  int method_of_derived() { return 500; }
};

Base *MakeADerivedReportABase() { return (Base *)((Base_1 *)new Derived()); }

int main() {
  Derived my_derived;
  int call_it = my_derived.method_of_derived();

  Base_1 *base_1_ptr = (Base_1 *)&my_derived;

  Base_2 *base_2_ptr = (Base_2 *)&my_derived;

  Base *base_through_1 = my_derived.return_base_1();
  Base *base_through_2 = my_derived.return_base_2();

  // Call this to make sure the compiler makes it.
  Base *fake_base = MakeADerivedReportABase();

  uint64_t base_through_1_addr = (uint64_t)base_through_1;
  uint64_t base_through_2_addr = (uint64_t)base_through_2;
  int64_t base_offset = base_through_2_addr - base_through_1_addr;
  printf("Base offset (should be 0): 0x%llx.\n", base_offset);
  uint64_t base_1_addr = (uint64_t)base_1_ptr;
  uint64_t base_2_addr = (uint64_t)base_2_ptr;
  int64_t offset = base_2_addr - base_1_addr;

  // Set a breakpoint here
  return my_derived.derived_int + base_1_ptr->base_1_arr[0] +
         base_2_ptr->base_2_arr[0] + my_derived.return_base_1()->base_int;
}
