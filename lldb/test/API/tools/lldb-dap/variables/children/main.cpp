struct Indexed {};
struct NotIndexed {};

#define BUFFER_SIZE 16
struct NonPrimitive {
  char buffer[BUFFER_SIZE];
  int x;
  long y;
};

NonPrimitive test_return_variable_with_children() {
  return NonPrimitive{"hello world!", 10, 20};
}

int main() {
  Indexed indexed;
  NotIndexed not_indexed;
  NonPrimitive non_primitive_result = test_return_variable_with_children();
  return 0; // break here
}
