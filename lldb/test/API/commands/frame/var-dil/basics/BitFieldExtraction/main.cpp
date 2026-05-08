void stop() {}

int main(int argc, char **argv) {
  int value = 0b01110011;
  int &value_ref = value;
  int *value_ptr = &value;

  int idx_0 = 0;
  int idx_1 = 1;
  const int &idx_1_ref = idx_1;
  enum Enum { kZero, kOne } enum_one = kOne;

  int int_arr[] = {7, 3, 1};

  stop(); // Set a breakpoint here
  return 0;
}
