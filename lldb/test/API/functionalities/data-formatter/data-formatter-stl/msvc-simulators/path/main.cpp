// Layout approximation of MSVC STL std::filesystem::path.

#include <wchar.h>

namespace std {
namespace filesystem {
struct path {
  const wchar_t *_Text;
};
} // namespace filesystem
} // namespace std

int main() {
  const wchar_t path_text[] = L"C:\\tmp\\file.txt";
  std::filesystem::path p{path_text};
  std::filesystem::path empty{L""};
  return 0; // break here
}
