#include <algorithm>
#include <functional>
#include <map>
#include <ranges>
#include <vector>

bool sort_less(int a, int b) {
  __builtin_printf("break here");
  return a < b;
}

bool ranges_sort_less(int a, int b) {
  __builtin_printf("break here");
  return a < b;
}

int view_transform(int a) {
  __builtin_printf("break here");
  return a * a;
}

void test_algorithms() {
  std::vector<int> vec{8, 1, 3, 2};

  // The internal frames for `std::sort` should be hidden
  std::sort(vec.begin(), vec.end(), sort_less);

  // The internal frames for `ranges::sort` should be hidden
  std::ranges::sort(vec.begin(), vec.end(), ranges_sort_less);

  // Same for views
  for (auto x : vec | std::ranges::views::transform(view_transform)) {
    // no-op
  }
}

void consume_number(int i) { __builtin_printf("break here"); }

int invoke_add(int i, int j) {
  __builtin_printf("break here");
  return i + j;
}

struct Callable {
  Callable(int num) : num_(num) {}
  void operator()(int i) const { __builtin_printf("break here"); }
  void member_function(int i) const { __builtin_printf("break here"); }
  int num_;
};

void test_invoke() {
  // Invoke a void-returning function
  std::invoke(consume_number, -9);

  // Invoke a non-void-returning function
  std::invoke(invoke_add, 1, 10);

  // Invoke a member function
  const Callable foo(314159);
  std::invoke(&Callable::member_function, foo, 1);

  // Invoke a function object
  std::invoke(Callable(12), 18);
}

struct MyKey {
  int x;
  bool operator==(const MyKey &) const = default;
  bool operator<(const MyKey &other) const {
    __builtin_printf("break here");
    return x < other.x;
  }
};

void test_containers() {
  std::map<MyKey, int> map;
  map.emplace(MyKey{1}, 2);
  map.emplace(MyKey{2}, 3);
}

int main() {
  test_algorithms();
  test_invoke();
  test_containers();
  return 0;
}
