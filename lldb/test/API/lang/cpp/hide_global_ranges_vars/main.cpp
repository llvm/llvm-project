#include <ranges>

int main(int argc, char const *argv[]) {
  int arr[3] = {1, 2, 3};
  auto arr_view = std::ranges::views::all(arr);
  auto arr_max = std::ranges::max_element(arr);

  return 0;
}
