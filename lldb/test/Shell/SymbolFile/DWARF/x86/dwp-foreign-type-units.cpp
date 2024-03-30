// REQUIRES: lld

// This test will make a type that will be compiled differently into two
// different .dwo files in a type unit with the same type hash, but with
// differing contents. I have discovered that the hash for the type unit is
// simply based off of the typename and doesn't seem to differ when the contents
// differ, so that will help us test foreign type units in the .debug_names
// section of the main executable. When a DWP file is made, only one type unit
// will be kept and the type unit that is kept has the .dwo file name that it
// came from. When LLDB loads the foreign type units, it needs to verify that
// any entries from foreign type units come from the right .dwo file. We test
// this since the contents of type units are not always the same even though
// they have the same type hash. We don't want invalid accelerator table entries
// to come from one .dwo file and be used on a type unit from another since this
// could cause invalid lookups to happen. LLDB knows how to track down which
// .dwo file a type unit comes from by looking at the DW_AT_dwo_name attribute
// in the DW_TAG_type_unit.

// Now test with DWARF5
// RUN: %clang -target x86_64-pc-linux -gdwarf-5 -gsplit-dwarf \
// RUN:   -fdebug-types-section -gpubnames -c %s -o %t.main.o
// RUN: %clang -target x86_64-pc-linux -gdwarf-5 -gsplit-dwarf -DVARIANT \
// RUN:   -fdebug-types-section -gpubnames -c %s -o %t.foo.o
// RUN: ld.lld %t.main.o %t.foo.o -o %t

// First we check when we make the .dwp file with %t.main.dwo first so it will
// pick the type unit from %t.main.dwo. Verify we find only the types from
// %t.main.dwo's type unit.
// RUN: llvm-dwp %t.main.dwo %t.foo.dwo -o %t.dwp
// RUN: %lldb \
// RUN:   -o "type lookup IntegerType" \
// RUN:   -o "type lookup FloatType" \
// RUN:   -o "type lookup IntegerType" \
// RUN:   -b %t | FileCheck %s
// CHECK: (lldb) type lookup IntegerType
// CHECK-NEXT: int
// CHECK-NEXT: (lldb) type lookup FloatType
// CHECK-NEXT: double
// CHECK-NEXT: (lldb) type lookup IntegerType
// CHECK-NEXT: int

// Next we check when we make the .dwp file with %t.foo.dwo first so it will
// pick the type unit from %t.main.dwo. Verify we find only the types from
// %t.main.dwo's type unit.
// RUN: llvm-dwp %t.foo.dwo %t.main.dwo -o %t.dwp
// RUN: %lldb \
// RUN:   -o "type lookup IntegerType" \
// RUN:   -o "type lookup FloatType" \
// RUN:   -o "type lookup IntegerType" \
// RUN:   -b %t | FileCheck %s --check-prefix=VARIANT

// VARIANT: (lldb) type lookup IntegerType
// VARIANT-NEXT: unsigned int
// VARIANT-NEXT: (lldb) type lookup FloatType
// VARIANT-NEXT: float
// VARIANT-NEXT: (lldb) type lookup IntegerType
// VARIANT-NEXT: unsigned int


// We need to do this so we end with a type unit in each .dwo file and that has
// the same signature but different contents. When we make the .dwp file, then
// one of the type units will end up in the .dwp file and we will have
// .debug_names accelerator tables for both type units and we need to ignore
// the type units .debug_names entries that don't match the .dwo file whose
// copy of the type unit ends up in the final .dwp file. To do this, LLDB will
// look at the type unit and take the DWO name attribute and make sure it
// matches, and if it doesn't, it will ignore the accelerator table entry.
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
