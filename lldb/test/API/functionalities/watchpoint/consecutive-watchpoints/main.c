#include <stdint.h>
struct fields {
  uint32_t field1;
  uint32_t field2; // offset +4
  uint16_t field3; // offset +8
  uint16_t field4; // offset +10
  uint16_t field5; // offset +12
  uint16_t field6; // offset +14
};

int main() {
  struct fields var = {0, 0, 0, 0, 0, 0};

  var.field1 = 5; // break here
  var.field2 = 6;
  var.field3 = 7;
  var.field4 = 8;
  var.field5 = 9;
  var.field6 = 10;

  return var.field1 + var.field2 + var.field3;
}
