
namespace foo {
struct CxxClass {
  long long foo_field = 10;
};

struct InheritedCxxClass: CxxClass {
  long long foo_subfield = 20;
};
} // namespace foo
namespace bar {
struct CxxClass {
  long long bar_field = 30;
};

struct InheritedCxxClass: CxxClass {
  long long bar_subfield = 40;
};

namespace baz {
struct CxxClass {
  long long baz_field = 50;
};

struct InheritedCxxClass: CxxClass {
  long long baz_subfield = 60;
};
} // namespace baz
} // namespace bar
