#include <compare>

int main() {
  auto po_less = std::partial_ordering::less;
  auto po_equivalent = std::partial_ordering::equivalent;
  auto po_greater = std::partial_ordering::greater;
  auto po_unordered = std::partial_ordering::unordered;
  auto wo_less = std::weak_ordering::less;
  auto wo_equivalent = std::weak_ordering::equivalent;
  auto wo_greater = std::weak_ordering::greater;
  auto so_less = std::strong_ordering::less;
  auto so_equal = std::strong_ordering::equal;
  auto so_equivalent = std::strong_ordering::equivalent;
  auto so_greater = std::strong_ordering::greater;
  return 0; // break here
}
