void stop() {}

int main(int argc, char **argv) {
  int i = 1;
  int &iref = i;
  int *iptr = &i;
  short sh = 2;
  int arr2[2] = {1, 2};
  int arr3[3] = {0, 1, 2};
  double dbl_arr[2] = {1.0, 2.0};
  void *nptr = nullptr;

  struct S {
  } s;

  enum EnumA { kOneA = 1, kTwoA } a_enum = kTwoA;
  enum EnumB { kOneB = 1 } b_enum = kOneB;

  stop(); // Set a breakpoint here
}
