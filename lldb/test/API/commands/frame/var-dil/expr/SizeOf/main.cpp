#include <cstdint>

void stop() {}

struct SizeOfFoo {
  int x, y;
  double d;
  static int z;
  virtual void foo() {}
} foo;

int main(int argc, char **argv) {
  int i = 1;
  short sh = 1;
  double d = 1.0;
  int *ptr = &i;
  int &iref = i;
  int arr[] = {1, 2, 3};
  int arr2d[2][3] = {{1, 2}, {3, 4, 5}};

  SizeOfFoo *foo_ptr = &foo;

  enum UnscopedEnum16 : int16_t { kZero16, kOne16 };
  UnscopedEnum16 enum_one = kOne16;

  auto int_size = sizeof(int);
  auto short_size = sizeof(short);
  auto double_size = sizeof(double);
  auto ptr_size = sizeof(int *);
  auto intref_size = sizeof(int &);
  auto arr_size = sizeof(arr);
  auto foo_size = sizeof(SizeOfFoo);
  auto enum_size = sizeof(UnscopedEnum16);

  stop(); // Set a breakpoint here
  return 0;
}
