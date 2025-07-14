#include <cstdio>
#include <ranges>
#include <string>
#include <vector>

using string_vec = std::vector<std::string>;

string_vec svec{"First", "Second", "Third", "Fourth"};

struct Foo {
  string_vec vec = svec;
};

int main() {
  {
    auto single = std::ranges::ref_view(svec[0]);
    auto all = std::views::all(svec);
    auto subset = all | std::views::take(2);
    std::puts("Break here");
  }

  {
    Foo f[2];
    auto view = std::ranges::ref_view(f);
    std::puts("Break here");
  }
}
