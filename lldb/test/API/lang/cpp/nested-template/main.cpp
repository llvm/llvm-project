struct Outer {
  Outer() {}

  template <class T>
  struct Inner {};
};

namespace NS {
namespace {
template <typename T> struct Struct {};
} // namespace
} // namespace NS

int main() {
  Outer::Inner<int> oi;
  NS::Struct<int> ns_struct;
}
