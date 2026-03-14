#include <cstdlib>

int counter = 0;

void inc_counter() { ++counter; }

void do_abort() { std::abort(); }

int main() {
  return 0; // break here
}
