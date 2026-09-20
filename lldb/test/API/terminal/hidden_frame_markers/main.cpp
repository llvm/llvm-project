void foo();
void bar();

namespace std {
namespace __1 {
void __test_hidden_frame() { foo(); }
void __test_nested_hidden_frame() { bar(); }

void outer_function() { __test_hidden_frame(); }
void other_outer_function() { __test_nested_hidden_frame(); }
} // namespace __1
} // namespace std

void foo() {
  std::__1::other_outer_function(); // break here first
}

void bar() {
  int a = 0; // break here after
}

int main() {
  std::__1::outer_function();
  return 0;
}
