// Test that we can use .debug_names to lookup a type that is only referenced
// from within a type unit. In the code below the type named "stype" is only
// referenced within the type unit itself and when we enable .debug_names, we
// expect the have an entry for this and to be able to find this type when
// we do a lookup.

// REQUIRES: lld

// RUN: %clang %s -target x86_64-pc-linux -gdwarf-5 -fdebug-types-section \
// RUN:   -gpubnames -fno-limit-debug-info -c -o %t.o
// RUN: ld.lld %t.o -o %t
// RUN: %lldb %t -o "type lookup stype" -b | FileCheck %s --check-prefix=BASE
// RUN: %lldb %t -o "type lookup bar::stype" -b | FileCheck %s --check-prefix=PART
// RUN: %lldb %t -o "type lookup foo::bar::stype" -b | FileCheck %s --check-prefix=FULL

// BASE: (lldb) type lookup stype
// BASE-NEXT: int

// PART: (lldb) type lookup bar::stype
// PART-NEXT: int

// FULL: (lldb) type lookup foo::bar::stype
// FULL-NEXT: int

namespace foo {
class bar {
public:
  typedef unsigned utype;
  // This type is only referenced from within the type unit and we need to
  // make sure we can find it with the new type unit support in .debug_names.
  typedef int stype;

private:
  utype m_unsigned;

public:
  bar(utype u) : m_unsigned(u) {}

  utype get() const { return m_unsigned; }
  void set(utype u) { m_unsigned = u; }
  stype gets() const { return (stype)m_unsigned; }
};
} // namespace foo

int main() {
  foo::bar b(12);
  return 0;
}
