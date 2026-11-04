#include <iostream>

namespace ns {
template <typename T> class MyClass {
public:
  void templateFunc() {
    std::cout << "In templateFunc" << std::endl; // Set a breakpoint here
  }
};
} // namespace ns

int main() {
  ns::MyClass<int> obj;
  obj.templateFunc();
  return 0;
}
