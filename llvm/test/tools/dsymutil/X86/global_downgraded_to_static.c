// REQUIRES : system-darwin
// RUN: dsymutil -oso-prepend-path %p/.. -dump-debug-map \
// RUN: %p/../Inputs/global_downgraded_to_static.x86_64 2>&1 | FileCheck %s
//
// RUN: dsymutil --linker parallel -oso-prepend-path %p/.. -dump-debug-map \
// RUN: %p/../Inputs/global_downgraded_to_static.x86_64 2>&1 | FileCheck %s
//
//  To build:
//    clang -g -c -DFILE1 global_downgraded_to_static.c -o 1.o
//    clang -g -c -DFILE2 global_downgraded_to_static.c -o 2.o
//    ld -r -exported_symbol _foo 1.o -o 1.r.o
//    clang 1.r.o 2.o -o global_downgraded_to_static.x86_64

#if defined(FILE1)
int global_to_become_static = 42;
// CHECK: sym: _global_to_become_static,
// CHECK-SAME: binAddr: 0x100001000
int foo() {
  return global_to_become_static;
}
#elif defined(FILE2)
int foo(void);
int main() {
  return foo();
}
#else
#error Define FILE1 or FILE2
#endif
