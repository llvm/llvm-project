#include <memory>

int main() { int argc = 0; char **argv = (char **)0; 
  std::shared_ptr<int> s(new int);
  *s = 3;
  return *s; // Set break point at this line.
}
