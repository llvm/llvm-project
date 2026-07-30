struct CG {
  int x;
  int y;
};

int g(int (^callback)(struct CG)) {
  struct CG cg = {.x = 1, .y = 2};

  int z = callback(cg); // Set breakpoint 2 here.

  return z;
}

int h(struct CG cg) { return 42; }

int main() {
  int c = 1;

  int (^add)(int, int) = ^int(int a, int b) {
    return a + b + c; // Set breakpoint 0 here.
  };

  int (^neg)(int) = ^int(int a) {
    return -a;
  };

  int sum = add(3, 4);
  int negated = neg(-5); // Set breakpoint 1 here.

  int (^add_struct)(struct CG) = ^int(struct CG cg) {
    return cg.x + cg.y;
  };

  g(add_struct);

  // A block that captures variables of several different types so we can check
  // that lldb reads captured variables of each type correctly.
  char captured_char = 'a';
  int captured_int = 42;
  double captured_double = 3.5;
  int *captured_ptr = &captured_int;
  struct CG captured_struct = {.x = 10, .y = 20};

  int (^types_block)() = ^int() {
    return captured_char + captured_int + (int)captured_double + *captured_ptr +
           captured_struct.x; // Set breakpoint 3 here.
  };

  int result = types_block();

  return sum + negated + result;
}
