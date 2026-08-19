// Layout approximation of MSVC STL std::source_location.

#include <stdint.h>

namespace std {
struct source_location {
  const char *_File;
  const char *_Function;
  uint32_t _Line;
  uint32_t _Column;
};
} // namespace std

int main() {
  std::source_location loc{"main.cpp", "int __cdecl main(void)", 6, 1};
  std::source_location loc_empty{"", "", 0, 0};
  return 0; // break here
}
