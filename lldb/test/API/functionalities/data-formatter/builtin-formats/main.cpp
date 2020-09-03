#include <cstdint>

const char cstring[15] = " \033\a\b\f\n\r\t\vaA09\0";

int main() {
  int use = *cstring;
  return use; // break here
}
