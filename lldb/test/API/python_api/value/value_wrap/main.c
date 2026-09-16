#include <stdint.h>

struct Engine {
  int kind;
};

struct Car {
  int doors;
  int wheels;
  struct Engine *engine;
};

int main() {
  uint32_t u32_zero = 0;
  uint32_t u32_one = 1;
  uint32_t u32_two = 2;
  uint32_t u32_four = 4;

  int32_t i32_zero = 0;
  int32_t i32_one = 1;
  int32_t i32_two = 2;

  int32_t i32_minus_one = -1;
  int32_t i32_minus_two = -2;

  const char *cstr = "This is a c string";
  uint16_t arr[] = {1, 2, 3, 4, 5, 6};
  uint16_t *arr_start = &arr[0];
  uint16_t *arr_second = &arr[1];

  struct Engine engine = {.kind = 1};
  struct Car my_car = {
      .doors = 3,
      .wheels = 4,
      .engine = &engine,
  };

  return my_car.engine->kind; // break here
}
