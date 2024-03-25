#include <cstdint>

const char cstring[15] = " \033\a\b\f\n\r\t\vaA09\0";
const char *empty_cstring = "";

int main() {
  int use = *cstring;
  void *void_empty_cstring = (void *)empty_cstring;
  return use; // break here
}
