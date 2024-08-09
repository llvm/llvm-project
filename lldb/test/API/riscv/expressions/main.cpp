struct S {
  int a;
  int b;
};

struct U {
  int a;
  double d;
};

int g;

int func_with_double_arg(int a, double b) { return 1; }

int func_with_ptr_arg(char *msg) { return 2; }

int func_with_struct_arg(struct S s) { return 3; }

int func_with_unsupported_struct_arg(struct U u) { return 4; }

double func_with_double_return() { return 42.0; }

int *func_with_ptr_return() { return &g; }

struct S func_with_struct_return() {
  struct S s = {3, 4};
  return s;
}

struct U func_with_unsupported_struct_return() {
  struct U u = {3, 42.0};
  return u;
}

int foo() { return 3; }

int foo(int a) { return a; }

int foo(int a, int b) { return a + b; }

int main() {
  struct S s = {1, 2};
  struct U u = {1, 1.0};
  double d = func_with_double_arg(1, 1.0) + func_with_ptr_arg("msg") +
             func_with_struct_arg(s) + func_with_double_return() +
             func_with_unsupported_struct_arg(u) + foo() + foo(1) + foo(1, 2);
  int *ptr = func_with_ptr_return();
  s = func_with_struct_return();
  u = func_with_unsupported_struct_return();
  return 0; // break here
}
