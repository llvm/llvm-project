#include <string>

#ifndef _MSVC_STL_VERSION
// this is more of a sanity check that the categories work as expected
#error Not using MSVC STL
#endif

int main() {
  std::u8string u8_string_small(u8"🍄");
  std::u8string u8_string(u8"❤️👍📄📁😃🧑‍🌾");
  std::u8string u8_empty(u8"");
  std::u8string u8_text(u8"ABC");
  u8_text.assign(u8"ABCd"); // Set break point at this line.
}
