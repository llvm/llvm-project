#include <bitset>
#include <stdio.h>

template <std::size_t N> void fill(std::bitset<N> &b) {
  b.set();
  b[0] = b[1] = false;
  for (std::size_t i = 2; i < N; ++i) {
    for (std::size_t j = 2 * i; j < N; j += i)
      b[j] = false;
  }
}

template <std::size_t N>
void by_ref_and_ptr(std::bitset<N> &ref, std::bitset<N> *ptr) {
  // Check ref and ptr
  return;
}

int main() {
  std::bitset<0> empty;
  std::bitset<13> small;
  fill(small);
  std::bitset<70> large;
  fill(large);
  by_ref_and_ptr(small, &small); // break here
  by_ref_and_ptr(large, &large);
  return 0;
}
