#include <iostream>

namespace ns {
template <typename T> class MyClass {
public:
  void templateFunc() {
    std::cout << "In templateFunc"
              << std::endl; // Find the line number for breakpoint 1 here.
  }
};
} // namespace ns

int main() {
  ns::MyClass<int> obj;
  obj.templateFunc();
  return 0;
}
