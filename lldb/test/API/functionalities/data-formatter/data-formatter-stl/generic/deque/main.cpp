#include <cstdio>
#include <deque>

struct Foo_small {
  int a;
  int b;
  int c;

  Foo_small(int a, int b, int c) : a(a), b(b), c(c) {}
};

struct Foo_large {
  int a;
  int b;
  int c;
  char d[1000] = {0};

  Foo_large(int a, int b, int c) : a(a), b(b), c(c) {}
};

template <typename T> T fill(T deque) {
  for (int i = 0; i < 100; i++) {
    deque.push_back({i, i + 1, i + 2});
    deque.push_front({-i, -(i + 1), -(i + 2)});
  }
  return deque;
}

int main() {
  std::deque<int> empty;
  std::deque<int> deque_1 = {1};
  std::deque<int> deque_3 = {3, 1, 2};

  std::deque<Foo_small> deque_200_small;
  deque_200_small = fill<std::deque<Foo_small>>(deque_200_small);

  std::deque<Foo_large> deque_200_large;
  deque_200_large = fill<std::deque<Foo_large>>(deque_200_large);

  return empty.size() + deque_1.front() + deque_3.front(); // break here
}
