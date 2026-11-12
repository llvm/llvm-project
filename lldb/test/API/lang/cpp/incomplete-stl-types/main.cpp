#include <set>

void f(std::set<int> &v);

int main() {
  std::set<int> v;
  f(v);
}
