typedef struct {
  int b;
} StructB;

namespace Namespace {
template<typename T> struct AlsoInNamespace { T field; };
 extern template struct AlsoInNamespace<int>;
}
