#include "lib1.h"

[[gnu::weak]] void lib1_internal() {
  Foo{}.func<decltype([]{})>();
}
