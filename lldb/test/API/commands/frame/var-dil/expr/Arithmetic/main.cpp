#include <cstdint>

int main(int argc, char **argv) {
  short s = 10;
  unsigned short us = 1;

  int x = 2;
  int &r = x;
  int *p = &x;
  int array[] = {1};
  enum Enum { kZero, kOne } enum_one = kOne;

  struct BitFieldStruct {
    uint16_t a : 10;
  };
  BitFieldStruct bf = {7};

  return 0; // Set a breakpoint here
}
