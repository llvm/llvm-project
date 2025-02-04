// Test that we can correctly disambiguate a nested type (where the outer type
// is in a type unit) from a non-nested type with the same basename. Failure to
// do so can cause us to think a type is a member of itself, which caused
// infinite recursion (crash) in the past.

// REQUIRES: lld

// RUN: %clang --target=x86_64-pc-linux -c %s -o %t-a.o -g -fdebug-types-section -flimit-debug-info -DFILE_A
// RUN: %clang --target=x86_64-pc-linux -c %s -o %t-b.o -g -fdebug-types-section -flimit-debug-info -DFILE_B
// RUN: ld.lld -z undefs %t-a.o %t-b.o -o %t
// RUN: %lldb %t -o "target variable x" -o exit | FileCheck %s

// CHECK: (lldb) target variable
// CHECK-NEXT: (const X) x = {
// CHECK-NEXT:   NS::Outer::Struct = {
// CHECK-NEXT:     x = 42
// CHECK-NEXT:     o = (x = 47)
// CHECK-NEXT:     y = 24
// CHECK-NEXT:   }
// CHECK-NEXT: }

namespace NS {
struct Struct {
  int x = 47;
  virtual void anchor();
};
} // namespace NS

#ifdef FILE_A
namespace NS {
struct Outer {
  struct Struct {
    int x = 42;
    NS::Struct o;
    int y = 24;
  };
};
} // namespace NS

struct X : NS::Outer::Struct {};
extern constexpr X x = {};
#endif
#ifdef FILE_B
void NS::Struct::anchor() {}
#endif
