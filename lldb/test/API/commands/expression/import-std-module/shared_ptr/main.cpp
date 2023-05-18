#include <memory>

int main(int argc, char **argv) {
  std::shared_ptr<int> s(new int);
  *s = 3;
  return *s; // Set break point at this line.
}
