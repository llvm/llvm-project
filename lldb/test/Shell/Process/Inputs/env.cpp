#include <cstdlib>
#include <iostream>

int main() {
  if (const char *env_p = std::getenv("FOO"))
    std::cout << "FOO=" << env_p << '\n';
}
