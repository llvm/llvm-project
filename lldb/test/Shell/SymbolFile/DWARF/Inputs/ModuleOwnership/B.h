typedef struct {
  int anon_field_b;
} StructB;

namespace Namespace {
template <typename T> struct AlsoInNamespace { T field; };
extern template struct AlsoInNamespace<int>;
} // namespace Namespace
