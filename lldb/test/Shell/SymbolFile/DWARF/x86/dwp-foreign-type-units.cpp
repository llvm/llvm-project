// REQUIRES: lld

// This test will make a type that will be compiled differently into two
// different .dwo files in a type unit with the same type hash, but with
// differing contents. Clang's type unit signature is based only on the mangled
// name of the type, regardless of the contents of the type, so that will help
// us test foreign type units in the .debug_names section of the main
// executable. When a DWP file is made, only one type unit will be kept and the
// type unit that is kept has the .dwo file name that it came from. When LLDB
// loads the foreign type units, it needs to verify that any entries from
// foreign type units come from the right .dwo file. We test this since the
// contents of type units are not always the same even though they have the
// same type hash. We don't want invalid accelerator table entries to come from
// one .dwo file and be used on a type unit from another since this could cause
// invalid lookups to happen. LLDB knows how to track down which .dwo file a
// type unit comes from by looking at the DW_AT_dwo_name attribute in the
// DW_TAG_type_unit.

// RUN: %clang -target x86_64-pc-linux -gdwarf-5 -gsplit-dwarf \
// RUN:   -fdebug-types-section -gpubnames -c %s -o %t.main.o
// RUN: %clang -target x86_64-pc-linux -gdwarf-5 -gsplit-dwarf -DVARIANT \
// RUN:   -fdebug-types-section -gpubnames -c %s -o %t.foo.o
// RUN: ld.lld %t.main.o %t.foo.o -o %t

// Check when have no .dwp file that we can find the types in both .dwo files.
// RUN: rm -f %t.dwp
// RUN: %lldb \
// RUN:   -o "type lookup IntegerType" \
// RUN:   -o "type lookup FloatType" \
// RUN:   -o "type lookup CustomType" \
// RUN:   -b %t | FileCheck %s --check-prefix=NODWP
// NODWP: (lldb) type lookup IntegerType
// NODWP-NEXT: int
// NODWP-NEXT: unsigned int
// NODWP: (lldb) type lookup FloatType
// NODWP-NEXT: double
// NODWP-NEXT: float
// NODWP: (lldb) type lookup CustomType
// NODWP-NEXT: struct CustomType {
// NODWP-NEXT:     typedef int IntegerType;
// NODWP-NEXT:     typedef double FloatType;
// NODWP-NEXT:     CustomType::IntegerType x;
// NODWP-NEXT:     CustomType::FloatType y;
// NODWP-NEXT: }
// NODWP-NEXT: struct CustomType {
// NODWP-NEXT:     typedef unsigned int IntegerType;
// NODWP-NEXT:     typedef float FloatType;
// NODWP-NEXT:     CustomType::IntegerType x;
// NODWP-NEXT:     CustomType::FloatType y;
// NODWP-NEXT: }

// Check when we make the .dwp file with %t.main.dwo first so it will
// pick the type unit from %t.main.dwo. Verify we find only the types from
// %t.main.dwo's type unit.
// RUN: llvm-dwp %t.main.dwo %t.foo.dwo -o %t.dwp
// RUN: %lldb \
// RUN:   -o "type lookup IntegerType" \
// RUN:   -o "type lookup FloatType" \
// RUN:   -o "type lookup CustomType" \
// RUN:   -b %t | FileCheck %s --check-prefix=DWPMAIN
// DWPMAIN: (lldb) type lookup IntegerType
// DWPMAIN-NEXT: int
// DWPMAIN: (lldb) type lookup FloatType
// DWPMAIN-NEXT: double
// DWPMAIN: (lldb) type lookup CustomType
// DWPMAIN-NEXT: struct CustomType {
// DWPMAIN-NEXT:     typedef int IntegerType;
// DWPMAIN-NEXT:     typedef double FloatType;
// DWPMAIN-NEXT:     CustomType::IntegerType x;
// DWPMAIN-NEXT:     CustomType::FloatType y;
// DWPMAIN-NEXT: }

// Next we check when we make the .dwp file with %t.foo.dwo first so it will
// pick the type unit from %t.main.dwo. Verify we find only the types from
// %t.main.dwo's type unit.
// RUN: llvm-dwp %t.foo.dwo %t.main.dwo -o %t.dwp
// RUN: %lldb \
// RUN:   -o "type lookup IntegerType" \
// RUN:   -o "type lookup FloatType" \
// RUN:   -o "type lookup CustomType" \
// RUN:   -b %t | FileCheck %s --check-prefix=DWPFOO

// DWPFOO: (lldb) type lookup IntegerType
// DWPFOO-NEXT: unsigned int
// DWPFOO: (lldb) type lookup FloatType
// DWPFOO-NEXT: float
// DWPFOO: (lldb) type lookup CustomType
// DWPFOO-NEXT: struct CustomType {
// DWPFOO-NEXT:     typedef unsigned int IntegerType;
// DWPFOO-NEXT:     typedef float FloatType;
// DWPFOO-NEXT:     CustomType::IntegerType x;
// DWPFOO-NEXT:     CustomType::FloatType y;
// DWPFOO-NEXT: }

struct CustomType {
  // We switch the order of "FloatType" and "IntegerType" so that if we do
  // end up reading the wrong accelerator table entry, that we would end up
  // getting an invalid offset and not find anything, or the offset would have
  // matched and we would find the wrong thing.
#ifdef VARIANT
  typedef float FloatType;
  typedef unsigned IntegerType;
#else
  typedef int IntegerType;
  typedef double FloatType;
#endif
  IntegerType x;
  FloatType y;
};

#ifdef VARIANT
int foo() {
#else
int main() {
#endif
  CustomType c = {1, 2.0};
  return 0;
}
