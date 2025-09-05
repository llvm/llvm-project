int externGlobalVar = 2;

namespace ext {
int externGlobalVar = 4;
} // namespace ext

struct ExtStruct {
private:
  static constexpr inline int static_inline = 16;
} es;
