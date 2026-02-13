static void foo() {
  int a = 0; // break here
}

namespace std {
namespace __1 {
void __test_hidden_frame() { foo(); }

void outer_function() { __test_hidden_frame(); }
} // namespace __1
} // namespace std

int main() {
  std::__1::outer_function();
  return 0;
}
