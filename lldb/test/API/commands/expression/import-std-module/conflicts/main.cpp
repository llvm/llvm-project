#include <cstdlib>
#include <utility>
#include <cmath>

int main() { int argc = 0; char **argv = (char **)0; 
  std::size_t f = argc;
  f = std::abs(argc);
  f = std::div(argc * 2, argc).quot;
  std::swap(f, f);
  return f; // Set break point at this line.
}
