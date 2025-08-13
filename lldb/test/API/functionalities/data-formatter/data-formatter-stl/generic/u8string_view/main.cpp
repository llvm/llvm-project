#include <cstdio>
#include <string_view>

int main() {
  std::u8string_view u8_string_small(u8"🍄");
  std::u8string_view u8_string(u8"❤️👍📄📁😃🧑‍🌾");
  std::u8string_view u8_empty(u8"");
  std::u8string_view u8_text(u8"ABC");
  u8_text = u8"ABCd";

  std::puts("// Set break point at this line.");
}
