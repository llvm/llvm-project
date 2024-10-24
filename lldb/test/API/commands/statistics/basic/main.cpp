// Test that the lldb command `statistics` works.
#include <string>
#include <vector>

template <typename T> class Box {
  T m_value;

public:
  Box(T value) : m_value(value) {}
};

void foo() {
  std::string str = "hello world";
  str += "\n"; // stop here
}

void bar(int x) {
  auto box = Box<int>(x);
  // stop here
}

void vec() {
  std::vector<int> int_vec = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<double> double_vec = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  // stop vector
  int x = int_vec.size();
}

int main(void) {
  int patatino = 27;
  foo();
  bar(patatino);
  vec();
  return 0; // break here
}
