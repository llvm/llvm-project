#include "type_definitions.h"

extern struct compile_unit1_type compile_unit1_var;
extern struct compile_unit2_type compile_unit2_var;

int main() {
  compile_unit1_var.x = 5;
  compile_unit2_var.y = 10;

  return 0;
}
