#include <cstdint>

int main(int argc, char **argv) {
  short s = 10;
  unsigned short us = 1;

  int x = 2;
  int &ref = x;
  enum Enum { kZero, kOne } enum_one = kOne;
  wchar_t wchar = 1;
  char16_t char16 = 2;
  char32_t char32 = 3;

  struct BitFieldStruct {
    char a : 4;
    int b : 32;
    unsigned int c : 32;
    uint64_t d : 48;
  };
  BitFieldStruct bitfield = {1, 2, 3, 4};

  return 0; // Set a breakpoint here
}
